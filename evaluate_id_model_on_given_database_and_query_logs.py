#!/usr/bin/python3
#
# Read in a log identification model, and evaluate its performance on
# a given dataset. In this version of the code, you explicitly specify
# which logs to use as the database and which logs to try and
# identify. This is mainly for investigating how the ID accuracy
# behaves as a function of database size.
#
# Eero Holmstrom (2022)
#

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from os import makedirs
import matplotlib.pyplot as plt
import matplotlib
from shutil import rmtree
import subprocess
import sys
import glob
import time
import numpy as np
import warnings


#
# This is required to use matplotlib with the CSC TensorFlow 2.7
# module
#

matplotlib.use("agg")



#
# Define some auxiliary functions.
#

#
# Compute identification accuracy for a given dataset. Takes as input
# a data dictionary of the following format:
#
# (log number (1 to 500), imaging day (01 or 02), image number (0 to 4)) : (raw numpy array of image, preprocessed numpy array of image, embedding of the image)
#
# Outputs identification accuracy as the number of correct
# identifications divided by the number of all queried
# identifications.
#

def get_identification_accuracy(data_dictionary, user_given_numbers_of_database_logs, user_given_numbers_of_logs_to_identify):

    
    #
    # Create a list of the identification results (1 = success, 0 =
    # fail) for outputting these into a file for computing statistics
    # later on.
    #

    list_of_id_results = []
    

    #
    # First, register those headshots of the first shooting day that
    # the user desires as the database of images to match other images
    # against.
    #

    database_log_numbers = []
    database_log_embeddings = []
    database_log_images = []

    for key in data_dictionary:

        if key[0] in user_given_numbers_of_database_logs and key[1] == 1 and key[2] == 0:

            this_log_number = key[0]
            this_embedding = data_dictionary[key][2].numpy()
            this_log_image = data_dictionary[key][0]

            database_log_numbers.append(this_log_number)
            database_log_embeddings.append(this_embedding)
            database_log_images.append(this_log_image)


    #
    # Print out the log numbers of database logs
    #

    print("")
    print("Here is the list of log numbers to use as the database logs, a total of %d:" % len(set(database_log_numbers)))
    print("")

    print(set(database_log_numbers))


    #
    # Then, get the images to be identified, i.e., the non-headshots
    # of the second shooting day that the user wants to try and
    # identify.
    #

    log_numbers_of_images_to_identify = []
    angle_indeces_of_images_to_identify = []
    embeddings_of_images_to_identify = []
    images_of_logs_to_identify = []
    
    for key in data_dictionary:

        if key[0] in user_given_numbers_of_logs_to_identify and key[1] == 2 and key[2] != 0:

            this_log_number = key[0]
            this_angle_index = key[2]
            this_embedding = data_dictionary[key][2].numpy()
            this_log_image = data_dictionary[key][0]

            log_numbers_of_images_to_identify.append(this_log_number)
            angle_indeces_of_images_to_identify.append(this_angle_index)
            embeddings_of_images_to_identify.append(this_embedding)
            images_of_logs_to_identify.append(this_log_image)
    

    #
    # Print out the log numbers of logs to try and identify
    #

    print("")
    print("Here is the list of log numbers to try and identify, a total of %d:" % len(set(log_numbers_of_images_to_identify)))
    print("")

    print(set(log_numbers_of_images_to_identify))
    print("")

    
    #
    # Do the identification. For each image to be identified (i.e.,
    # probe image), find the closest embedding in the database. If the
    # log numbers match, the identification was successful. Otherwise
    # it was not.
    #

    database_log_embeddings_as_nparray = np.array(database_log_embeddings)
    embeddings_of_images_to_identify_as_nparray = np.array(embeddings_of_images_to_identify)
    
    number_of_correct_identifications = 0
    total_number_of_identifications = 0

    id_query_number = 0

    for i_log in range(0, len(log_numbers_of_images_to_identify)):

        id_query_number = id_query_number + 1

        image_to_identify = images_of_logs_to_identify[i_log]
        embedding_of_image_to_identify_as_nparray = embeddings_of_images_to_identify_as_nparray[i_log]
        true_log_number = log_numbers_of_images_to_identify[i_log]
        angle_index = angle_indeces_of_images_to_identify[i_log]

        distances_to_database_embeddings = np.linalg.norm(embedding_of_image_to_identify_as_nparray[0, :] - database_log_embeddings_as_nparray[:, 0, :], axis = 1)

        i_closest_match = np.argmin(distances_to_database_embeddings)

        log_number_as_claimed_by_model = database_log_numbers[i_closest_match]

        best_match_image = database_log_images[i_closest_match]


        if log_number_as_claimed_by_model == true_log_number:

            number_of_correct_identifications = number_of_correct_identifications + 1

            #
            # Append this "success" result to the list of identification results
            #

            list_of_id_results.append(1)

            successful_identification = True

        else:

            #
            # Append this "fail" result to the list of identification results
            #

            list_of_id_results.append(0)

            successful_identification = False


        #
        # Plot this probe image along with the best match and the
        # result of the identification
        #

        plot_probe_image_and_best_match(image_to_identify, true_log_number, 2, angle_index, best_match_image, log_number_as_claimed_by_model, 1, 0, successful_identification, id_query_number)


        total_number_of_identifications = total_number_of_identifications + 1

        
    #
    # Output the list of identification results to a file
    #
        
    np.savetxt('list_of_id_results.txt', np.array(list_of_id_results))


    return number_of_correct_identifications / total_number_of_identifications, total_number_of_identifications




#
# Plot a given probe image and its closest match in the identification
# process side by side.
#

def plot_probe_image_and_best_match(probe_image, probe_image_log_number, probe_image_day, probe_image_angle_index, best_match_image, best_match_log_number, best_match_image_day, best_match_image_angle_index, successful_identification, id_query_number):


    #
    # Plot the two images and the identification result
    #

    plt.figure(figsize = (width_for_figure_size, height_for_figure_size))

    ax_left = plt.subplot(1, 2, 1)

    ax_right = plt.subplot(1, 2, 2)


    if successful_identification:

        plt.suptitle('Identification result: success', fontsize = figure_title_font_size, y = figure_title_position)

    else:

        plt.suptitle('Identification result: failure', fontsize = figure_title_font_size, y = figure_title_position)


    #
    # The images are numpy arrays, and hence the pixel values are
    # floats. Scale the pixel values to the interval 0...1, which is
    # the data range that imshow expects when the pixel values are
    # floats.
    #

    ax_left.imshow(probe_image / 255.0)
    ax_left.set_title('Probe image (log number = ' + str(probe_image_log_number) + ', imaging day = ' + str(probe_image_day) + ', angle index = ' + str(probe_image_angle_index) + ')', fontsize = image_title_font_size, pad = image_titlepad)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    ax_right.imshow(best_match_image / 255.0)
    ax_right.set_title('Best match image (log number = ' + str(best_match_log_number) + ', imaging day = ' + str(best_match_image_day) + ', angle index = ' + str(best_match_image_angle_index) + ')', fontsize = image_title_font_size, pad = image_titlepad)
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    
    #
    # Save the image to a PNG file
    #

    plt.savefig('result_of_id_query_' + str(id_query_number) + '.png')

    plt.close()

    
    return


#
# The script begins here
#


#
# Define the list of logs to use as the database and the list of logs
# to try and identify
#

database_logs = []
query_logs = []


#
# Usage
#

if len(sys.argv) < 4:

    print("Usage: %s [model.h5] [validation data directory] [CNN name (for choosing preprocessing function for images)]" % sys.argv[0])
    exit(1)


#
# Assign input parameters
#

modelfile = str(sys.argv[1])
validation_data_dir = str(sys.argv[2])
cnn = str(sys.argv[3])

print("")
print("Using the following parameters:")
print("")
print("model file = %s" % modelfile)
print("validation data directory = %s" % validation_data_dir)
print("CNN = %s" % cnn)
print("")


#
# Import the necessary stuff for the requested CNN
#

if cnn == "ResNet50V2":


    from tensorflow.keras.applications.resnet_v2 import preprocess_input as cnn_preprocessing_function
    print("Loaded the ResNet50V2 preprocessing function")


elif cnn == "ResNet152V2":


    from tensorflow.keras.applications.resnet_v2 import preprocess_input as cnn_preprocessing_function
    print("Loaded the ResNet152V2 preprocessing function")


elif cnn == "Xception":


    from tensorflow.keras.applications.xception import preprocess_input as cnn_preprocessing_function
    print("Loaded the Xception preprocessing function")


elif cnn == "InceptionV3":


    from tensorflow.keras.applications.inception_v3 import preprocess_input as cnn_preprocessing_function
    print("Loaded the InceptionV3 preprocessing function")


elif cnn == "InceptionResNetV2":


    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as cnn_preprocessing_function
    print("Loaded the InceptionResNetV2 preprocessing function")


elif cnn == "VGG19":


    from tensorflow.keras.applications.vgg19 import preprocess_input as cnn_preprocessing_function
    print("Loaded the VGG19 preprocessing function")


elif cnn == "MobileNetV2":


    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as cnn_preprocessing_function
    print("Loaded the MobileNetV2 preprocessing function")


elif cnn == "DenseNet201":


    from tensorflow.keras.applications.densenet import preprocess_input as cnn_preprocessing_function
    print("Loaded the DenseNet201 preprocessing function")


elif cnn == "EfficientNetB4":


    from tensorflow.keras.applications.efficientnet import preprocess_input as cnn_preprocessing_function
    print("Loaded the EfficientNetB4 preprocessing function")


else:

    print("ERROR! Unknown CNN model %s requested. Exiting." % cnn)
    exit(1)



#
# Define some other parameters
#


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
# Parameters for creating plots
#

width_for_figure_size = 15.0
height_for_figure_size = 12.0
line_width_in_plots = 4.0
figure_title_font_size = 30
image_title_font_size = 11
figure_title_position = 0.85
image_titlepad = 10


#
# Where to save images
#

image_output_dir = './image_matches'



#
# Clean up previous results
#

rmtree(image_output_dir, ignore_errors = True)


if image_output_dir:

    makedirs(image_output_dir)


    
#
# Read in all validation images into a dictionary of the following
# format:
#
# (log number (1 to 500), imaging day (01 or 02), image number (0 to 4)) : (original image as nparray, original image preprocessed for the CNN, up-to-date embedding of the image)
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
    # Load this image and scale it to the target size
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

    validation_data_dictionary[(this_image_log_number, this_image_imaging_day, this_image_image_number)] = [this_image_as_numpy_array, this_preprocessed_image, None]
    

print("Done. Read in a total of %d images." % len(validation_data_dictionary))


#
# Set up the embedding function
#

embedding_model = load_model(modelfile)


#
# Print out the structure of the embedding model
#

print("")
print("*** Our embedding model summary:")
print("")

embedding_model.summary()


#
# Get the embedding size for the model being used
#

the_embedding_layer = embedding_model.get_layer(index = -1)
embedding_size = (the_embedding_layer.output_shape)[1]

print("")
print("")
print("Found an embedding size of %d" % embedding_size)


#
# Compute the embedding for each image
#

print("")
print("Now computing embedding for each image in the validation set.")
    

#
# Keep track of the total time required to compute the embeddings
#

time_begin_computing_embeddings = time.time()


#
# Create a numpy matrix for outputting the embeddings into a file for
# further analysis
#

embedding_matrix_to_output = np.zeros([len(validation_data_dictionary), embedding_size + 3])



number_of_embeddings_computed = 0


for key in validation_data_dictionary:

    this_image = validation_data_dictionary[key][1]

    this_embedding = embedding_model(np.expand_dims(this_image, axis = 0))
        
    validation_data_dictionary[key][2] = this_embedding

    number_of_embeddings_computed = number_of_embeddings_computed + 1

    #
    # Prepare to output the embeddings into an ASCII file of the
    # following format for each line:
    #
    # <log number> <imaging day> <image number> <e_1> <e_2> ... <e_N>
    # 
    # where e_i is the ith element of the embedding vector.
    #

    this_log_number = key[0]
    this_imaging_day = key[1]
    this_image_number = key[2]
    
    embedding_matrix_to_output[number_of_embeddings_computed - 1, 0] = this_log_number 
    embedding_matrix_to_output[number_of_embeddings_computed - 1, 1] = this_imaging_day
    embedding_matrix_to_output[number_of_embeddings_computed - 1, 2] = this_image_number
    embedding_matrix_to_output[number_of_embeddings_computed - 1, 3:] = this_embedding

    
time_end_computing_embeddings = time.time()

time_used_for_computing_embeddings = time_end_computing_embeddings - time_begin_computing_embeddings


#
# Output the embeddings to a file
#

np.savetxt('embeddings.txt', embedding_matrix_to_output)


print("Done. Total time used for computing embeddings was %6.10f s, i.e., %6.10f s per image." % (time_used_for_computing_embeddings, time_used_for_computing_embeddings / len(validation_data_dictionary)))
print("Computed a total of %d embeddings." % number_of_embeddings_computed)


#
# Then, compute accuracy over the training data and over the validation data
#

print("")
print("Now computing identification accuracy over the validation data.")


#
# Keep track of the total time used for log identification
#

time_begin_identification = time.time()

validation_data_accuracy, total_number_of_identifications = get_identification_accuracy(validation_data_dictionary, database_logs, query_logs)

time_end_identification = time.time()

time_used_for_identification = time_end_identification - time_begin_identification


#
# Print out identification accuracy and the time required to do one identification attempt
#

print("Done. Identification accuracy over the validation set is %3.5f" % validation_data_accuracy)
print("The total time used for identification was %6.10f s, i.e., %6.10f s per identification query." % (time_used_for_identification, time_used_for_identification / total_number_of_identifications))
print("The total number of identification queries was %d." % total_number_of_identifications)



#
# Move all created images to the designated output directory
#

subprocess.call("mv result_of_id_query_*.png %s" % image_output_dir, shell = True)


#
# All done.
#

print("")
print("All done. Exiting.")
print("")

exit(0)
