#!/usr/bin/python3
#
# Use triplet loss training to create a model for uniquely identifying
# logs based on end-face images.
#
# Eero Holmstrom (2021)
#

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Lambda, Input, Layer, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras import optimizers
import math
from os import remove, makedirs
import matplotlib.pyplot as plt
import matplotlib
from shutil import rmtree
import random
import sys
import glob
import time
import numpy as np
import warnings


#
# Define some auxiliary functions
#


#
# This function computes and returns the mean triplet loss for a set
# of triplet distances given in the tuple triplet_distances, which is
# of the format
#
# ( [< |xa - xp|**2 for triplet 1>,  < |xa - xp|**2 for triplet 2>, ...], [< |xa - xn|**2 for triplet 1>, |xa - xn|**2 for triplet 2>, ...] )
# 
# where "[]" signifies a numpy array in the form of a
# TensorFlow tensor.
#

def get_triplet_loss(triplet_distances, triplet_loss_margin):

    mean_triplet_loss = 0.0

    number_of_triplets = triplet_distances[0].shape[0]
    
    for i_triplet in range(0, number_of_triplets):
        
        this_positive_example_distance_squared = triplet_distances[0][i_triplet]
        this_negative_example_distance_squared = triplet_distances[1][i_triplet]

        mean_triplet_loss = mean_triplet_loss + tf.maximum(0, triplet_loss_margin - this_negative_example_distance_squared + this_positive_example_distance_squared)
        
    mean_triplet_loss = mean_triplet_loss / number_of_triplets

    return mean_triplet_loss



#
# Compute identification accuracy for a given dataset (e.g., training
# or validation set). Takes as input a data dictionary of the
# following format:
#
# (log number (1 to 500), imaging day (01 or 02), image number (0 to 4)) : (numpy array of image, embedding of the image)
#
# Outputs identification accuracy as the number of correct
# identifications divided by the number of all queried
# identifications.
#

def get_identification_accuracy(data_dictionary):

    #
    # First, register the headshots of the first shooting day as the
    # database of images to match other images against.
    #

    database_log_numbers = []
    database_log_embeddings = []

    for key in data_dictionary:

        if key[1] == 1 and key[2] == 0:

            this_log_number = key[0]
            this_embedding = data_dictionary[key][1].numpy()

            database_log_numbers.append(this_log_number)
            database_log_embeddings.append(this_embedding)


    #
    # Then, get the images to be identified, i.e., non-headshots of
    # the second shooting day.
    #

    log_numbers_of_images_to_identify = []
    embeddings_of_images_to_identify = []

    for key in data_dictionary:

        if key[1] == 2 and key[2] != 0:

            this_log_number = key[0]
            this_embedding = data_dictionary[key][1].numpy()

            log_numbers_of_images_to_identify.append(this_log_number)
            embeddings_of_images_to_identify.append(this_embedding)
    
    
    #
    # Do the identification. For each image to be identified, find the
    # closest embedding in the database. If the log numbers match, the
    # identification was successful. Otherwise it was not.
    #

    database_log_embeddings_as_nparray = np.array(database_log_embeddings)
    embeddings_of_images_to_identify_as_nparray = np.array(embeddings_of_images_to_identify)
    
    number_of_correct_identifications = 0
    total_number_of_identifications = 0

    for i_log in range(0, len(log_numbers_of_images_to_identify)):

        embedding_of_image_to_identify_as_nparray = embeddings_of_images_to_identify_as_nparray[i_log]
        true_log_number = log_numbers_of_images_to_identify[i_log]

        distances_to_database_embeddings = np.linalg.norm(embedding_of_image_to_identify_as_nparray[0, :] - database_log_embeddings_as_nparray[:, 0, :], axis = 1)

        i_closest_match = np.argmin(distances_to_database_embeddings)

        log_number_as_claimed_by_model = database_log_numbers[i_closest_match]

        if log_number_as_claimed_by_model == true_log_number:

            number_of_correct_identifications = number_of_correct_identifications + 1

        total_number_of_identifications = total_number_of_identifications + 1

    return number_of_correct_identifications / total_number_of_identifications



#
# Define some auxiliary classes
#


#
# This layer takes as input three feature vectors xa, xp, xn (i.e.,
# three embeddings), and computes and returns the distances
# |xa - xp|**2 and |xa - xn|**2 as a tuple.
#

class DistanceLayer(Layer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):

        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)

        return (ap_distance, an_distance)


    
#
# The script begins here
#


#
# Usage
#

if len(sys.argv) < 13:

    print("Usage: %s [number of epochs] [learning rate] [save model every n epochs] [training data directory] [validation data directory] [triplet loss margin] [fully connected layer one size] [fully connected layer two size] [embedding size] [training batch size] [requested CNN] [0 = freeze entire CNN; 1 = train BN layers; 2 = train entire CNN]" % sys.argv[0])
    exit(1)


#
# Assign input parameters
#

epochs = int(sys.argv[1])
learning_rate = float(sys.argv[2])
save_model_every_n_epochs = int(sys.argv[3])
training_data_dir = str(sys.argv[4])
validation_data_dir = str(sys.argv[5])
triplet_loss_margin = float(sys.argv[6])
fully_connected_layer_one_size = int(sys.argv[7])
fully_connected_layer_two_size = int(sys.argv[8])
embedding_size = int(sys.argv[9])
training_batch_size = int(sys.argv[10])
requested_cnn = str(sys.argv[11])
free_mode = int(sys.argv[12])

print("")
print("Using the following parameters:")
print("")
print("epochs = %d" % epochs)
print("learning rate = %f" % learning_rate)
print("Saving model every %d epochs" % save_model_every_n_epochs)
print("triplet loss margin = %f" % triplet_loss_margin)
print("fully connected layer one size = %d" % fully_connected_layer_one_size)
print("fully connected layer two size = %d" % fully_connected_layer_two_size)
print("embedding size = %d" % embedding_size)
print("training batch size = %d" % training_batch_size)
print("requested CNN = %s" % requested_cnn)
print("free mode = %d" % free_mode)
print("")
print("Training data will be read from the directory %s and validation data from %s." % (training_data_dir, validation_data_dir))
print("")


#
# Import the necessary stuff for the requested CNN
#

if requested_cnn == "ResNet50V2":


    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as cnn_preprocessing_function
    print("Loaded the ResNet50V2 preprocessing function")


elif requested_cnn == "ResNet152V2":


    from tensorflow.keras.applications.resnet_v2 import ResNet152V2
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as cnn_preprocessing_function
    print("Loaded the ResNet152V2 preprocessing function")


elif requested_cnn == "Xception":


    from tensorflow.keras.applications.xception import Xception
    from tensorflow.keras.applications.xception import preprocess_input as cnn_preprocessing_function
    print("Loaded the Xception preprocessing function")


elif requested_cnn == "InceptionV3":


    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input as cnn_preprocessing_function
    print("Loaded the InceptionV3 preprocessing function")
    

elif requested_cnn == "InceptionResNetV2":


    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as cnn_preprocessing_function
    print("Loaded the InceptionResNetV2 preprocessing function")
    

elif requested_cnn == "VGG19":


    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input as cnn_preprocessing_function
    print("Loaded the VGG19 preprocessing function")


elif requested_cnn == "MobileNetV2":


    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as cnn_preprocessing_function
    print("Loaded the MobileNetV2 preprocessing function")


elif requested_cnn == "DenseNet201":


    from tensorflow.keras.applications.densenet import DenseNet201
    from tensorflow.keras.applications.densenet import preprocess_input as cnn_preprocessing_function
    print("Loaded the DenseNet201 preprocessing function")


elif requested_cnn == "EfficientNetB4":


    from tensorflow.keras.applications.efficientnet import EfficientNetB4
    from tensorflow.keras.applications.efficientnet import preprocess_input as cnn_preprocessing_function
    print("Loaded the EfficientNetB4 preprocessing function")


else:

    print("ERROR! Unknown CNN model %s requested. Exiting." % requested_cnn)
    exit(1)



#
# Define some other parameters
#


#
# Increase the dynamic triplet loss margin by this amount in case the
# triplet extension loop fails to find suitable negative candidates
#

delta_triplet_loss_margin = 0.1

print("")
print("Using delta triplet loss margin = ", delta_triplet_loss_margin)


#
# Size of images input to the CNN
#

input_height = 512
input_width = 512
input_channels = 3


print("")
print("Using input height = ", input_height)
print("Using input width = ", input_width)
print("Using input channels = ", input_channels)


#
# Parameters for creating plots after training
#

width_for_figure_size = 15.0
height_for_figure_size = 12.0
line_width_in_plots = 4.0


#
# Optimization parameters for Adam
#

beta_1 = 0.9
beta_2 = 0.999
clipnorm = 1.0


print("")
print("Setting beta 1 to ", beta_1)
print("Setting beta 2 to ", beta_2)
print("Setting clipnorm to ", clipnorm)


#
# Where to save images
#

save_to_dir_training = None # './augmented_training_set_images_output'
save_to_dir_validation = None # './validation_set_images_output'


#
# Parameters for data augmentation
#

zoom_range = (-0.2, 0.2)


#
# Allow rotations of about +- 18 degrees for each log
#

rotation_range_as_fraction_of_2pi = 0.05

contrast_factor = 0.25


print("")
print("Using zoom range of ", zoom_range)
print("Using rotation range as fraction of 2pi of ", rotation_range_as_fraction_of_2pi)
print("Using contrast factor of ", contrast_factor)


#
# Clean up previous results
#

rmtree(save_to_dir_training, ignore_errors = True)
rmtree(save_to_dir_validation, ignore_errors = True)

try:

    remove('loss_vs_epoch.png')
    remove('accuracy_vs_epoch.png')

except OSError:

    pass


if save_to_dir_training:

    makedirs(save_to_dir_training)

if save_to_dir_validation:
    
    makedirs(save_to_dir_validation)


#
# Read in all training images into a dictionary of the following
# format:
#
# (log number (1 to 500), imaging day (1 or 2), image number (0 to 4)) : (latest augmented version of the original image preprocessed for the CNN, up-to-date embedding of the latest version of the original image)
#
# For most CNNs, preprocessing here means scaling pixel values to the
# range -1...1. The embedding element holds the embedding f = f(image)
# using the latest version of the embedding model f being trained.
#
# Scale each image to the target size.
#

training_data_dictionary = {}


#
# Save the original images into a dictionary. The idea is to use this
# dictionary to repeatedly create modified images for training through
# image augmentation.
#

training_data_original_images_dictionary = {}


#
# First, form the list of image paths to read in. From these, pick out
# the log number, imaging day, and image number.
#

training_data_image_names = glob.glob(training_data_dir + '/*/*/*.jpg')

print("")
print("Found a total of %d training images." % len(training_data_image_names))


#
# Then, read in the images, one by one. Preprocess them for the CNN.
#

print("Now reading in the images and scaling them to size %d by %d pixels..." % (input_height, input_width))

for filename in training_data_image_names:

    #
    # Load this image and scale it to the target size.
    #

    this_image = image.load_img(filename, target_size = (input_height, input_width))
    this_image_as_numpy_array = image.img_to_array(this_image)
    
    #
    # Form the index tuple for this image
    #

    this_image_log_number = int(filename.split(sep = "/")[-3])
    this_image_imaging_day = int(filename.split(sep = "/")[-2])
    this_image_image_number = int(filename.split(sep = "/")[-1][:-4])

    #
    # Save the original, raw image into a dictionary
    #

    training_data_original_images_dictionary[(this_image_log_number, this_image_imaging_day, this_image_image_number)] = this_image_as_numpy_array

    #
    # Preprocess the image for the CNN
    #

    this_preprocessed_image = cnn_preprocessing_function(np.copy(this_image_as_numpy_array))

    #
    # Update the training data dictionary. The embedding will be computed later.
    #

    training_data_dictionary[(this_image_log_number, this_image_imaging_day, this_image_image_number)] = [this_preprocessed_image, None]

    
print("Done. Read in a total of %d images." % len(training_data_dictionary))


#
# Read in all validation images into a dictionary of the following
# format:
#
# (log number (1 to 500), imaging day (01 or 02), image number (0 to 4)) : (original image preprocessed for the CNN, up-to-date embedding of the image)
#
# For most CNNs, preprocessing here means scaling pixel values to the
# range -1...1. The embedding element holds the embedding f = f(image)
# using the latest version of the embedding model f being trained.
#
# Scale each image to the target size.
#


validation_data_dictionary = {}


#
# First, form the list of image paths to read in. From these, pick out
# the log number, imaging day, and image number.
#

validation_data_image_names = glob.glob(validation_data_dir + '/*/*/*.jpg')

print("")
print("Found a total of %d validation images." % len(validation_data_image_names))


#
# Then, read in the images, one by one. Preprocess them for the CNN.
#

print("Now reading in the images and scaling them to size %d by %d pixels..." % (input_height, input_width))

for filename in validation_data_image_names:

    #
    # Load this image and scale it to the target size.
    #

    this_image = image.load_img(filename, target_size = (input_height, input_width))
    this_image_as_numpy_array = image.img_to_array(this_image)

    #
    # Form the index tuple for this image
    #

    this_image_log_number = int(filename.split(sep = "/")[-3])
    this_image_imaging_day = int(filename.split(sep = "/")[-2])
    this_image_image_number = int(filename.split(sep = "/")[-1][:-4])

    #
    # Preprocess the image for the CNN
    #

    this_preprocessed_image = cnn_preprocessing_function(np.copy(this_image_as_numpy_array))

    #
    # Update the validation data dictionary. The embedding will be computed later.
    #

    validation_data_dictionary[(this_image_log_number, this_image_imaging_day, this_image_image_number)] = [this_preprocessed_image, None]
    

print("Done. Read in a total of %d images." % len(validation_data_dictionary))
print("")


#
# Set up our core model, i.e., a function which takes as input an
# image of a log and produces an embedding in the form of an
# L2-normalized, one-dimensional feature vector.
#


if requested_cnn == "ResNet50V2":

    base_cnn = ResNet50V2(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded ResNet50V2 as the base CNN")


elif requested_cnn == "ResNet152V2":

    base_cnn = ResNet152V2(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded ResNet152V2 as the base CNN")


elif requested_cnn == "Xception":

    base_cnn = Xception(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded Xception as the base CNN")
    

elif requested_cnn == "InceptionV3":

    base_cnn = InceptionV3(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded InceptionV3 as the base CNN")


elif requested_cnn == "InceptionResNetV2":

    base_cnn = InceptionResNetV2(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded InceptionResNetV2 as the base CNN")


elif requested_cnn == "VGG19":

    base_cnn = VGG19(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded VGG19 as the base CNN")
    

elif requested_cnn == "MobileNetV2":

    base_cnn = MobileNetV2(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded MobileNetV2 as the base CNN")
    

elif requested_cnn == "DenseNet201":

    base_cnn = DenseNet201(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded DenseNet201 as the base CNN")
    

elif requested_cnn == "EfficientNetB4":

    base_cnn = EfficientNetB4(input_shape = (input_height, input_width, input_channels), include_top = False, weights = 'imagenet', pooling = None)
    print("Loaded EfficientNetB4 as the base CNN")
    

else:

    print("ERROR! Unknown CNN model %s requested. Exiting." % requested_cnn)
    exit(1)


y = base_cnn.output
y = GlobalAveragePooling2D()(y)
y = Dense(fully_connected_layer_one_size, activation = "relu")(y)
y = BatchNormalization()(y)
y = Dense(fully_connected_layer_two_size, activation = "relu")(y)
y = BatchNormalization()(y)
y = Dense(embedding_size)(y)
y = Lambda(lambda f: tf.keras.backend.l2_normalize(f, axis = 1))(y)
image_feature_embedding = y

embedding_model = Model(inputs = base_cnn.input, outputs = image_feature_embedding, name = "EmbeddingModel")


#
# Freeze the entire CNN
#

if free_mode == 0:

    print("")
    print("Freezing the entire CNN.")

    for layer in base_cnn.layers:

        layer.trainable = False


#
# Train just the batch normalization layers
#

elif free_mode == 1:

    print("")
    print("Training only the BN layers of the CNN.")

    for layer in base_cnn.layers:

        if isinstance(layer, BatchNormalization):
    
            layer.trainable = True

        else:

            layer.trainable = False


#
# Train the entire model
#

elif free_mode == 2:

    print("")
    print("Training the entire CNN.")

    for layer in base_cnn.layers:
    
        layer.trainable = True


else:

    print("ERROR! Found unknown free mode of %d. Exiting." % free_mode)
    exit(1)



#
# Then, set up the three-fold Siamese network, which takes as input
# three images: anchor image, positive example, negative example. The
# model outputs the squared distances between the anchor and the
# positive example as well as the anchor and the negative example.
#


#
# First, create the Keras tensors for building the Siamese model.
#

anchor_input = Input(name = "anchor", shape = (input_height, input_width, input_channels))
positive_input = Input(name = "positive", shape = (input_height, input_width, input_channels))
negative_input = Input(name = "negative", shape = (input_height, input_width, input_channels))


#
# Then, create the Siamese model. Use our custom-built distance layer
# for this.
#

distances = DistanceLayer()(embedding_model(anchor_input), embedding_model(positive_input), embedding_model(negative_input))

siamese_network = Model(inputs = [anchor_input, positive_input, negative_input], outputs = distances)


#
# Next, create a model for doing data augmentation
#

data_augmentator = tf.keras.Sequential()

#
# Take the mirror image of the image randomly in the horizontal or
# vertical direction.
#

#data_augmentator.add(RandomFlip("horizontal", input_shape = (input_height, input_width, input_channels)))
#data_augmentator.add(RandomFlip("vertical"))


#
# Apply a random in-plane rotation to the image.
#

data_augmentator.add(RandomRotation(rotation_range_as_fraction_of_2pi, input_shape = (input_height, input_width, input_channels), fill_mode = "constant"))


#
# This layer creates zooming and shearing of the image.
#

data_augmentator.add(RandomZoom(height_factor = zoom_range, width_factor = zoom_range, fill_mode = "constant"))


#
# Make random changes to the image contrast.
#

data_augmentator.add(RandomContrast(contrast_factor))


#
# Print out the structure of the original CNN model, the base CNN, our
# final embedding model, our three-fold Siamese model, and our data
# augmentation model.
#

if requested_cnn == "ResNet50V2":

    full_resnet50v2 = ResNet50V2(input_shape = (224, 224, 3), include_top = True, weights = 'imagenet', pooling = None)

    print("")
    print("*** Full ResNet50V2 summary:")
    print("")
    full_resnet50v2.summary()


elif requested_cnn == "ResNet152V2":

    full_resnet152v2 = ResNet152V2(input_shape = (224, 224, 3), include_top = True, weights = 'imagenet', pooling = None)

    print("")
    print("*** Full ResNet152V2 summary:")
    print("")
    full_resnet152v2.summary()


elif requested_cnn == "Xception":

    full_xception = Xception(input_shape = (299, 299, 3), include_top = True, weights = 'imagenet', pooling = None)

    print("")
    print("*** Full Xception summary:")
    print("")
    full_xception.summary()


elif requested_cnn == "InceptionV3":

    full_inceptionv3 = InceptionV3(input_shape = (299, 299, 3), include_top = True, weights = 'imagenet', pooling = None)

    print("")
    print("*** Full InceptionV3 summary:")
    print("")
    full_inceptionv3.summary()


elif requested_cnn == "InceptionResNetV2":

    full_inceptionresnetv2 = InceptionResNetV2(input_shape = (299, 299, 3), include_top = True, weights = 'imagenet', pooling = None)

    print("")
    print("*** Full InceptionResNetV2 summary:")
    print("")
    full_inceptionresnetv2.summary()


elif requested_cnn == "VGG19":

    full_vgg19 = VGG19(input_shape = (224, 224, 3), include_top = True, weights = 'imagenet', pooling = None)
    
    print("")
    print("*** Full VGG-19 summary:")
    print("")
    full_vgg19.summary()


elif requested_cnn == "MobileNetV2":

    full_mobilenetv2 = MobileNetV2(input_shape = (224, 224, 3), include_top = True, weights = 'imagenet', pooling = None)

    print("")
    print("*** Full MobileNetV2 summary:")
    print("")
    full_mobilenetv2.summary()


elif requested_cnn == "DenseNet201":

    full_densenet201 = DenseNet201(input_shape = (224, 224, 3), include_top = True, weights = 'imagenet', pooling = None)

    print("")
    print("*** Full DenseNet201 summary:")
    print("")
    full_densenet201.summary()


elif requested_cnn == "EfficientNetB4":

    full_efficientnetb4 = EfficientNetB4(input_shape = (380, 380, 3), include_top = True, weights = 'imagenet', pooling = None)

    print("")
    print("*** Full EfficientNetB4 summary:")
    print("")
    full_efficientnetb4.summary()


else:

    print("ERROR! Unknown CNN model %s requested. Exiting." % requested_cnn)
    exit(1)
    
    
print("")
print("*** Our base CNN summary:")
print("")
base_cnn.summary()

print("")
print("*** Our embedding model summary:")
print("")
embedding_model.summary()

print("")
print("*** Our three-fold Siamese model summary:")
print("")
siamese_network.summary()

print("")
print("*** Our data augmentation model summary:")
print("")
data_augmentator.summary()

print("")


#
# Define our optimization method, i.e., the method of stochastic
# gradient descent to use.
#

sgd = optimizers.Adam(learning_rate = learning_rate, beta_1 = beta_1, beta_2 = beta_2, amsgrad = False, clipnorm = clipnorm)


#
# Do the training. Here's the algorithm:
#
# - Loop over training epochs.
#
# - At the beginning of each epoch, form the triplets which will make
#   up the training data for this epoch. First, pair each anchor image
#   (headshot image of log X from the first imaging day) with a
#   positive example (one of the non-headshot images of log X from the
#   second imaging day), which gives a total of four (a, p) pairs for
#   each log. Then, extend each of these (a, p) pairs to a triplet (a,
#   p, n) by randomly sampling over negative non-headshot instances
#   that fulfill the requirement
#
#   (1) |f(a) - f(n)|**2 < |f(a) - f(p)|**2 + dynamic_triplet_loss_margin
#
#   This gives you the full list of triplets to use during this epoch.
#
# - Then, loop over the list of triplets in batches.
#
# - For each batch, compute the gradient of the mean triplet loss with
#   respect to the network weights, and update the network weights
#   using the gradient.
#
# - At the end of each epoch, compute identification accuracy
#   separately on the training set and on the validation set.
#


#
# Keep track of the total time used for the training
#

time_begin_training = time.time()


#
# Keep track of loss and accuracy as a function of epoch
#

loss_vs_epoch = []
training_accuracy_vs_epoch = []
validation_accuracy_vs_epoch = []


#
# Begin loop over training epochs
#


for i_epoch in range(1, epochs+1):


    print("")
    print("Start of epoch %d" % i_epoch)


    training_losses_for_this_epoch = []


    #
    # At the beginning of each epoch, take the original set of
    # training data images and use these to create a modified set of
    # training images through image augmentation.
    #

    print("Now performing image augmentation for the training set.")
    
    for key in training_data_original_images_dictionary:

        this_image = training_data_original_images_dictionary[key]

        #
        # Get an augmented version of this image. Pass training = True
        # to activate the augmentation layers.
        #

        augmented_version_of_this_image = data_augmentator(np.expand_dims(this_image, axis = 0), training = True)

        #
        # Cut out the extra axis introduced by augmentation
        #

        augmented_version_of_this_image = augmented_version_of_this_image[0]

        #
        # Preprocess this image for the CNN
        #
        
        augmented_version_of_this_image = cnn_preprocessing_function(augmented_version_of_this_image)
        
        #
        # Then insert this image into the training set
        #
        
        training_data_dictionary[key][0] = augmented_version_of_this_image


    #
    # Save the augmented training data images to disk, if desired
    #

    if save_to_dir_training:

        print("Saving augmented training data images to disk.")

        for key in training_data_dictionary:

            this_image = training_data_dictionary[key][0]
            this_image_name_string = str(key[0]) + "_" + str(key[1]) + "_" + str(key[2]) + "_epoch_" + str(i_epoch) + ".jpg"
            image.save_img(save_to_dir_training + "/" + this_image_name_string, this_image)


    #
    # Save also validation data images to disk, if desired
    #

    if save_to_dir_validation:

        print("Saving validation data images to disk.")

        for key in validation_data_dictionary:

            this_image = validation_data_dictionary[key][0]
            this_image_name_string = str(key[0]) + "_" + str(key[1]) + "_" + str(key[2]) + "_epoch_" + str(i_epoch) + ".jpg"
            image.save_img(save_to_dir_validation + "/" + this_image_name_string, this_image)


    #
    # Compute the embedding for each image in the training set using
    # the latest version of the embedding model.
    #
    
    print("Now computing the embedding for each image in the training set.")

    for key in training_data_dictionary:

        this_image = training_data_dictionary[key][0]

        this_embedding = embedding_model(np.expand_dims(this_image, axis = 0))

        training_data_dictionary[key][1] = this_embedding


    #
    # At the beginning of each epoch, reset the dynamic triplet loss
    # margin to the actual triplet loss margin, which is used for
    # computing triplet loss. This dynamic triplet loss will be used
    # to select suitable triplets for this epoch.
    #
    
    dynamic_triplet_loss_margin = triplet_loss_margin
    
    
    #
    # Form the triplets which make up the training data for this
    # epoch.
    #

    #
    # First, form all possible positive pairs, using the headshot from
    # the first shooting day as the anchor, and taking the
    # non-headshots from the second shooting day as positive examples.
    #

    print("Now forming the triplets for this epoch.")

    all_positive_pairs = []


    #
    # Get the unique set of log numbers in the training data
    #

    list_of_all_log_numbers = []

    for key in training_data_dictionary:

        list_of_all_log_numbers.append(key[0])

    set_of_all_log_numbers = set(list_of_all_log_numbers)


    #
    # Then, create all positive pairs for each unique log number
    #

    for this_anchor_image_log_number in set_of_all_log_numbers:

        this_anchor_image = (this_anchor_image_log_number, 1, 0)

        #
        # Find all positive pairs for this anchor image, and append
        # each one to the full list of positive pairs
        #

        for i_image in range(1, 5):

            this_positive_pair = (this_anchor_image, (this_anchor_image[0], 2, i_image))

            all_positive_pairs.append(this_positive_pair)
        
    #
    # Next, for each positive pair, find the negative examples of
    # non-headshot images from the second shooting day which fulfill
    # the requirement (1) above. If, for a given positive pair, you
    # cannot find any negative examples to create a triplet fulfilling
    # this requirement, increase the dynamic triplet loss margin and
    # restart the loop over positive pairs.
    #
    
    negative_candidates_found_for_all_positive_pairs = False
    
    while not negative_candidates_found_for_all_positive_pairs:

        triplets_for_training = []

        negative_candidates_found_for_all_positive_pairs = True
        
        for this_positive_pair in all_positive_pairs:
            
            this_anchor = this_positive_pair[0]
            this_positive_example = this_positive_pair[1]

            this_anchor_to_positive_distance = np.linalg.norm(training_data_dictionary[this_anchor][1] - training_data_dictionary[this_positive_example][1])

            #
            # Find candidate negative examples, i.e., find all negative
            # examples which extend the positive pair to a triplet so that
            # requirement (1) is fulfilled above. Then, choose one of
            # these, uniformly randomly, to complete the triplet.
            #

            candidate_negative_examples = []

            for key in training_data_dictionary:

                #
                # Only consider images that are of a different log AND
                # were taken on the second shooting day AND are
                # non-headshot images
                #

                if key[0] == this_anchor[0] or key[1] == 1 or key[2] == 0:

                    continue
    

                #
                # Furthermore, only consider images which fulfil
                # requirement (1) above
                #

                this_anchor_to_negative_distance = np.linalg.norm(training_data_dictionary[this_anchor][1] - training_data_dictionary[key][1])

                if this_anchor_to_negative_distance**2 < this_anchor_to_positive_distance**2 + dynamic_triplet_loss_margin:

                    candidate_negative_examples.append(key)

            
            #
            # Choose one of the candidate negative images randomly to
            # complete the triplet. If no negative candidates were
            # found, increase the triplet loss margin and restart the
            # triplet extension loop.
            #

            if len(candidate_negative_examples) == 0:

                print("No negative candidates fulfilling the requirements found for this positive pair!")

                print("Increasing dynamic triplet loss margin and restarting the triplet extension loop.")
                
                print("Old value of dynamic triplet loss margin: %f" % dynamic_triplet_loss_margin)

                dynamic_triplet_loss_margin = dynamic_triplet_loss_margin + delta_triplet_loss_margin

                print("New value of dynamic triplet loss margin: %f" % dynamic_triplet_loss_margin)

                negative_candidates_found_for_all_positive_pairs = False

                break

            else:

                this_negative_example = candidate_negative_examples[np.random.choice(np.arange(0, len(candidate_negative_examples)))]

                this_triplet = (this_anchor, this_positive_example, this_negative_example)

                triplets_for_training.append(this_triplet)


    print("Created a total of %d triplets to use for this training epoch." % len(triplets_for_training))

    
    #
    # Shuffle the ordering of the triplets for this training epoch.
    #

    np.random.shuffle(triplets_for_training)

    
    #
    # Create a numpy array holding the triplet image numpy arrays in
    # the following format:
    #
    # <anchor image> <positive example> <negative example>
    #
    
    training_data = np.empty((len(triplets_for_training), 3, input_width, input_height, 3))
    
    i_triplet = 0

    for this_triplet in triplets_for_training:

        training_data[i_triplet, 0] = training_data_dictionary[triplets_for_training[i_triplet][0]][0]
        training_data[i_triplet, 1] = training_data_dictionary[triplets_for_training[i_triplet][1]][0]
        training_data[i_triplet, 2] = training_data_dictionary[triplets_for_training[i_triplet][2]][0]

        i_triplet = i_triplet + 1

    
    #
    # Iterate over the batches of the dataset and do gradient descent.
    #

    for i_batch in range(0, math.ceil(training_data.shape[0] / training_batch_size)):

        this_training_batch = training_data[i_batch*training_batch_size:(i_batch+1)*training_batch_size, :, :, :, :]

        print("Now training on batch number %d, which comprises %d triplets." % (i_batch + 1, this_training_batch.shape[0]))

        #
        # Open a GradientTape to record the operations run during the
        # forward pass, which enables auto-differentiation.
        #

        with tf.GradientTape() as tape:

            #
            # Run the forward pass of the model. The operations that
            # the model applies to its inputs are going to be recorded
            # on the GradientTape.
            #
            
            triplet_distances_for_this_batch = siamese_network([this_training_batch[:, 0], this_training_batch[:, 1], this_training_batch[:, 2]], training = True)

            #
            # The tuple triplet_distances_for_this_batch is of the format
            #
            # ( [< |xa - xp|**2 for triplet 1>,  < |xa - xp|**2 for triplet 2>, ...], [< |xa - xn|**2 for triplet 1>, |xa - xn|**2 for triplet 2>, ...] )
            # 
            # where "[]" signifies a numpy array in the form of a
            # TensorFlow tensor.
            #

            #
            # Compute the loss for this minibatch.
            #

            loss_value = get_triplet_loss(triplet_distances_for_this_batch, triplet_loss_margin)

            print("Triplet loss for this batch was %.5e" % loss_value)


        #
        # Keep track of the training loss of the batches during this epoch
        #
        
        training_losses_for_this_epoch.append(loss_value.numpy())
        
        #
        # Use gradient tape to compute the gradient of the triplet
        # loss with respect to the trainable model parameters
        #
        
        print("Now computing gradients of loss with respect to model weights.")
        
        gradients = tape.gradient(loss_value, siamese_network.trainable_weights)
        
        #
        # Run one step of gradient descent by updating the values of
        # the network weights to minimize the loss
        #

        print("Now updating model weights.")

        sgd.apply_gradients(zip(gradients, siamese_network.trainable_weights))


    #
    # Re-compute the embeddings of all log images using the latest
    # version of the embedding model.
    #

    print("Computing embedding for each image in the training set and the validation set.")
    
    for key in training_data_dictionary:

        this_image = training_data_dictionary[key][0]

        this_embedding = embedding_model(np.expand_dims(this_image, axis = 0))

        training_data_dictionary[key][1] = this_embedding


    for key in validation_data_dictionary:

        this_image = validation_data_dictionary[key][0]

        this_embedding = embedding_model(np.expand_dims(this_image, axis = 0))

        validation_data_dictionary[key][1] = this_embedding

    
    #
    # Then, compute accuracy over the training data and over the validation data
    #

    print("Now computing identification accuracy over training data and validation data.")

    training_data_accuracy_for_this_epoch = get_identification_accuracy(training_data_dictionary)
    validation_data_accuracy_for_this_epoch = get_identification_accuracy(validation_data_dictionary)

    #
    # Keep track of loss and accuracy as a function of epoch
    #

    loss_vs_epoch.append(np.mean(np.array(training_losses_for_this_epoch)))
    training_accuracy_vs_epoch.append(training_data_accuracy_for_this_epoch)
    validation_accuracy_vs_epoch.append(validation_data_accuracy_for_this_epoch)

    #
    # This epoch is now complete. Print out some stats.
    #

    print("Epoch number %d complete." % i_epoch)
    print("Mean loss over batches of this epoch was %.5e" % np.mean(np.array(training_losses_for_this_epoch)))
    print("Identification accuracy over the full training set is %3.5f" % training_data_accuracy_for_this_epoch)
    print("Identification accuracy over the full validation set is %3.5f" % validation_data_accuracy_for_this_epoch)


    #
    # Save the embedding model to disk at the desired frequency
    #

    if i_epoch % save_model_every_n_epochs == 0:

        embedding_model_save_name = "log_id_embedding_model_at_epoch_" + str(i_epoch) + ".h5"
        embedding_model.save(embedding_model_save_name, save_format = "h5")

        print("Embedding model saved to disk.")



#
# Model training is complete
#


time_end_training = time.time()
time_used_for_training = time_end_training - time_begin_training


print("")
print("Training complete. Total wallclock time was %6.3f s or %6.3f h." % (time_used_for_training, time_used_for_training / 3600.0))
print("")


#
# Plot loss and accuracy vs. epoch
#

print("Now plotting loss and accuracy vs. epoch into png files.")


#
# This is required to use matplotlib with the CSC TensorFlow 2.7
# module
#

matplotlib.use("agg")


#
# Disable all warnings from now on, as you're only plotting and they tend to fill up the output console
#

warnings.simplefilter("ignore")


#
# Set a decent font size
#

plt.rc('font', size = 15)


#
# Plot loss
#

plt.figure(figsize = (width_for_figure_size, height_for_figure_size))
plt.plot(np.arange(1, epochs + 1), np.array(loss_vs_epoch), 'r-', linewidth = line_width_in_plots)
plt.title('Model training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('loss_vs_epoch.png')


#
# Plot accuracy
#

plt.figure(figsize = (width_for_figure_size, height_for_figure_size))
h1, = plt.plot(np.arange(1, epochs + 1), np.array(training_accuracy_vs_epoch), 'r-', linewidth = line_width_in_plots)
h2, = plt.plot(np.arange(1, epochs + 1), np.array(validation_accuracy_vs_epoch), 'b--', linewidth = line_width_in_plots)
plt.title('Model accuracy')
plt.ylabel('Identification accuracy')
plt.xlabel('Epoch')
plt.legend([h1, h2], ['Training', 'Validation'], loc = 'upper left')
plt.savefig('accuracy_vs_epoch.png')


#
# All done.
#

print("")
print("All done. Exiting.")
print("")

exit(0)
