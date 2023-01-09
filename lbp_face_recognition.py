#!/usr/bin/python3
#
# Perform local binary pattern face recognition on a set of logs using
# the same basic approach as reported in Ahonen et al. (2006).
#
# Eero Holmström, 2021-2022
#


import sys
import glob
from sklearn.metrics.pairwise import additive_chi2_kernel as chi2_distance
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
import numpy as np



#
# Define auxiliary functions
#


#
# Compute and return a concatenated LBP histogram to serve as the
# feature vector for a given numpy-array image.
#

def get_lbp_feature_vector(lbp_grid_size, lbp_p, lbp_r, lbp_number_of_bins, this_image_as_numpy_array, lbp_method):


    
    if verbose_output:

        print("")
        print("*** Started work on a new image! ***")
        print("")
        print("Now computing the LBP feature vector for this image.")

        

    #
    # Compute the LBP for the entire image
    #
    
    this_image_lbp = local_binary_pattern(this_image_as_numpy_array, lbp_p, lbp_r, method = lbp_method)


    
    if verbose_output:
    
        print("min, mean, max, std of pixel values in the original image:", np.min(this_image_as_numpy_array), np.mean(this_image_as_numpy_array), np.max(this_image_as_numpy_array), np.std(this_image_as_numpy_array))
        print("min, mean, max, std of pixel values in the LBP image:", np.min(this_image_lbp), np.mean(this_image_lbp), np.max(this_image_lbp), np.std(this_image_lbp))
        print("")
    
    
    
    #
    # Then, break the LBP image down into the desired grid and compute
    # the normalized pixel intensity histogram for each grid cell
    #

    this_image_height = this_image_lbp.shape[0]
    this_image_width = this_image_lbp.shape[1]


    
    if verbose_output:

        print("LBP matrix has height of", this_image_height, "and width of", this_image_width)
        

        
    this_concatenated_histogram = np.empty(0)

    

    if verbose_output:

        print("Now computing normalized histogram for each cell of the image.")
    

        
    for i in range(0, lbp_grid_size):

        for j in range(0, lbp_grid_size):

            #
            # Determine indeces for this grid cell
            #
            
            i_start = round(i*(this_image_height / lbp_grid_size))
            i_end = round((i+1)*(this_image_height / lbp_grid_size))
            
            j_start = round(j*(this_image_width / lbp_grid_size))
            j_end = round((j+1)*(this_image_width / lbp_grid_size))


            
            if verbose_output:

                print("")
                print("--- Started work on a new cell! ---")
                print("")
                print("The indeces for this cell are i_start = %d, i_end = %d, j_start = %d, j_end = %d" % (i_start, i_end, j_start, j_end))


                
            this_section_of_lbp_image = this_image_lbp[i_start:i_end, j_start:j_end]

            
            #
            # Compute the histogram for this cell. This is where the
            # different LBP methods need to be treated differently:
            #
            # - For the "default" method, the possible number of
            #   unique values is equal to 2**lbp_p, e.g., for the
            #   canonical 3x3 case, this means 2**8 = 256 different
            #   values (from 0 to 255).
            #
            # - For the "uniform" method, the possible number of
            #   unique values is equal to P + 2. See, e.g., Ojala et
            #   al. (2002).
            #
            # For the "uniform" method, we always use binning into P +
            # 2 different values. For the "default" method, we use the
            # user-given number of bins, as this is an important
            # hyperparameter for the "default" method (Ojala et
            # al. 1996).
            #

            
            if lbp_method == 'uniform':
                
                this_section_histogram, bin_edges = np.histogram(this_section_of_lbp_image, bins = np.arange(0, lbp_p + 3))


            elif lbp_method == 'default':

                this_section_histogram, bin_edges = np.histogram(this_section_of_lbp_image, bins = lbp_number_of_bins, range = (0.0, 2**lbp_p))


                
            if verbose_output:
                
                print("")
                print("Histogram computed. Here are the bin edges:", bin_edges)
                print("")
                print("The total number of bins was", len(bin_edges) - 1)
                print("")
                print("Here are the histogram values:", this_section_histogram)

                
                
            #
            # Normalize the histogram so that the sum of all histogram
            # values for this cell is equal to one
            #
        
            this_section_histogram = this_section_histogram / np.sum(this_section_histogram)


            
            if verbose_output:

                print("")
                print("Histogram normalized. The sum of histogram values is now ", np.sum(this_section_histogram))
                print("")
                print("Here is the histogram:", this_section_histogram)

                
            
            #
            # Add this histogram to the total, concatenated histogram,
            # which will become the feature vector for the full image
            #
            
            this_concatenated_histogram = np.append(this_concatenated_histogram, this_section_histogram)

            

            if verbose_output:

                print("")
                print("Appended the histogram to the total feature vector being formed.")
            



    if verbose_output:


        print("")
        print("----------------------------------------------------------")
        print("")
        print("The full feature vector for this image has now been formed.")
        print("")

        print("Data for concatenated histogram, i.e., full feature vector:")
        print("")
        print("min, mean, max, std, shape:", np.min(this_concatenated_histogram), np.mean(this_concatenated_histogram), np.max(this_concatenated_histogram), np.std(this_concatenated_histogram), this_concatenated_histogram.shape)
        print("")
        print("Here's the full feature vector:")
        print("")
        print(this_concatenated_histogram)

        
        
    #
    # Create plots, if desired. Not to be run on CSC.
    #

    if create_plots:


        print("")
        print("Now plotting various useful images.")

            
        #
        # 1. Plot the original image
        #

        plt.figure(figsize = (15, 12))
        plt.imshow(this_image_as_numpy_array)
        plt.title("Original image")
        plt.colorbar()
        plt.savefig("original_image.jpg")

        print("Original image shape:", this_image_as_numpy_array.shape)
            

        #
        # 2. Plot the LBP image
        #

        plt.figure(figsize = (15, 12))
        plt.imshow(this_image_lbp)
        plt.title("LBP image")
        plt.colorbar()
        plt.savefig("lbp_image.jpg")

        print("LBP image shape:", this_image_lbp.shape)

        
        #
        # Get the axis limits for use in the image with the grid lines
        #

        axis_limits = [plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]]
        
        
        #
        # 3. Plot the LBP image with the grid cell limits
        #
        
        plt.figure(figsize = (15, 12))
        plt.imshow(this_image_lbp)
        plt.colorbar()

        for i in range(0, lbp_grid_size + 1):
        
            this_grid_line_v_coordinate = round(i*(this_image_height / lbp_grid_size))
                
            plt.plot([0, this_image_width], [this_grid_line_v_coordinate, this_grid_line_v_coordinate], 'r-', linewidth = 5)
            

        for j in range(0, lbp_grid_size + 1):

            this_grid_line_h_coordinate = round(j*(this_image_width / lbp_grid_size))
            
            plt.plot([this_grid_line_h_coordinate, this_grid_line_h_coordinate], [0, this_image_height], 'r-', linewidth = 5)

        
        #
        # Set axis limits so that the grid borders are visible, too
        #

        axis_limits = axis_limits + np.array([-1.0*axis_limits_buffer, axis_limits_buffer, axis_limits_buffer, -1.0*axis_limits_buffer])

        plt.axis(axis_limits)
        plt.title("LBP image with grid")
        plt.savefig("lbp_image_with_grid.jpg")

    
        #
        # 4. Plot the concatenated LBP histogram
        #

        plt.figure(figsize = (15, 12))
        plt.bar(np.arange(1, this_concatenated_histogram.shape[0]+1), this_concatenated_histogram)
        plt.title("Concatenated histogram")
        plt.savefig("concatenated_histogram.jpg")
            
    
        #
        # The plot creation is meant to be run on your local Linux
        # machine, not CSC.
        #
            
        plt.show()
            
    
    return this_concatenated_histogram



#
# Compute identification accuracy for a given dataset. Takes as input
# a data dictionary of the following format:
#
# (log number (1 to 500), imaging day (01 or 02), image number (0 to 4)) : (numpy array of image, feature vector of the image)
#
# Outputs identification accuracy as the number of correct
# identifications divided by the number of all queried
# identifications.
#

def get_identification_accuracy(data_dictionary):

    
    #
    # Create a list of the identification results (1 = success, 0 =
    # fail) for outputting into a file for computing statistics later
    # on.
    #

    list_of_id_results = []


    #
    # First, register the headshots of the first shooting day as the
    # database of images to match other images against.
    #

    database_log_numbers = []
    database_log_embeddings = []

    
    for key in data_dictionary:

        if key[1] == 1 and key[2] == 0:

            this_log_number = key[0]
            this_embedding = data_dictionary[key][1]

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
            this_embedding = data_dictionary[key][1]

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

        
        embedding_of_image_to_identify_as_nparray = embeddings_of_images_to_identify_as_nparray[i_log, :]
        true_log_number = log_numbers_of_images_to_identify[i_log]

        
        embedding_of_image_to_identify_as_nparray = embedding_of_image_to_identify_as_nparray.reshape(1, -1)
        distances_to_database_embeddings = -1.0*chi2_distance(database_log_embeddings_as_nparray, embedding_of_image_to_identify_as_nparray)

        
        i_closest_match = np.argmin(distances_to_database_embeddings)


        log_number_as_claimed_by_model = database_log_numbers[i_closest_match]

        
        if log_number_as_claimed_by_model == true_log_number:

            
            number_of_correct_identifications = number_of_correct_identifications + 1

            #
            # Append this "success" result to the list of identification results
            #

            list_of_id_results.append(1)

            
        else:

            
            #
            # Append this "fail" result to the list of identification results
            #

            list_of_id_results.append(0)


        total_number_of_identifications = total_number_of_identifications + 1


    #
    # Output the list of identification results to a file
    #

    np.savetxt('list_of_id_results.txt', np.array(list_of_id_results))

        
    return number_of_correct_identifications / total_number_of_identifications, total_number_of_identifications



#
# The script begins here
#


#
# Usage
#

if len(sys.argv) < 7:

    print("Usage: %s [LBP grid size in both dimensions] [P] [R] [number of bins] [lbp method] [image directory]" % sys.argv[0])
    exit(1)


#
# Assign input parameters
#

lbp_grid_size = int(sys.argv[1])
lbp_parameter_p = int(sys.argv[2])
lbp_parameter_r = int(sys.argv[3])
lbp_number_of_bins = int(sys.argv[4])
lbp_method = str(sys.argv[5])
image_directory = str(sys.argv[6])


#
# Check that a valid LBP method was given
#

if (lbp_method != "uniform") and (lbp_method != "default"):

    print("ERROR! Found unknown method \"%s\" for LBP. Exiting." % lbp_method)
    exit(1)


#
# Set some other parameters
#

image_target_height = 512
image_target_width = 512
axis_limits_buffer = 20
verbose_output = True
create_plots = False


print("")
print("Using the following parameter values:")
print("")
print("lbp grid size = %d" % lbp_grid_size)
print("lbp parameter P = %d" % lbp_parameter_p)
print("lbp parameter R = %d" % lbp_parameter_r)
print("lbp number of bins = %d" % lbp_number_of_bins)
print("lbp method = %s" % lbp_method)
print("image directory = %s" % image_directory)
print("image target width = %d" % image_target_width)
print("image target height = %d" % image_target_height)



#
# This is required to use matplotlib with the CSC TensorFlow 2.7
# module
#

#
#matplotlib.use("agg")
#



if create_plots:

    #
    # Set the default color map for plots to gray
    #

    plt.gray()
    plt.close()



#
# Read in all images into a dictionary of the following format:
#
# (log number (1 to 500), imaging day (01 or 02), image number (0 to 4)) : (original RGB image, LBP feature vector)
#
# Scale each image to the target size prior to computing the LBP
# histogram.
#

image_data_dictionary = {}


#
# First, form the list of image paths to read in. From these, pick out
# the log number, imaging day, and image number.
#

image_names = glob.glob(image_directory + '/*/*/*.jpg')

print("")
print("Found a total of %d images." % len(image_names))


#
# Then, read in the images, one by one. Compute the LBP feature vector
# for each.
#

print("")
print("Now reading in the images, scaling them to size %d by %d pixels, and computing the LBP feature vector for each..." % (image_target_height, image_target_width))

for filename in image_names:

    
    #
    # Load this image, convert it to grayscale, and scale it to the target size
    #

    this_image_as_numpy_array = imread(filename, as_gray = True)
    this_image_as_numpy_array = resize(this_image_as_numpy_array, (image_target_height, image_target_width), anti_aliasing = True)


    
    if verbose_output:

        print("")
        print("Read in the image ", filename, " into a numpy array of shape", this_image_as_numpy_array.shape, " and pixel values ranging from ", np.min(this_image_as_numpy_array), " to ", np.max(this_image_as_numpy_array))

        
    
    #
    # Form the index tuple for this image
    #

    this_image_log_number = int(filename.split(sep = "/")[-3])
    this_image_imaging_day = int(filename.split(sep = "/")[-2])
    this_image_image_number = int(filename.split(sep = "/")[-1][:-4])

    
    #
    # Compute the LBP feature vector for this image
    #

    this_lbp_feature_vector = get_lbp_feature_vector(lbp_grid_size, lbp_parameter_p, lbp_parameter_r, lbp_number_of_bins, this_image_as_numpy_array, lbp_method)

    
    #
    # Update the image data dictionary.
    #

    image_data_dictionary[(this_image_log_number, this_image_imaging_day, this_image_image_number)] = [this_image_as_numpy_array, this_lbp_feature_vector]
    


print("Done. Processed a total of %d images." % len(image_data_dictionary))



#
# Compute the identification accuracy using Chi-squared distance
#


print("")
print("Now computing identification accuracy over the given image set...")


identification_accuracy, total_number_of_identifications = get_identification_accuracy(image_data_dictionary)


print("Done. The result is %.5f" % identification_accuracy)
print("The total number of identification queries was %d." % total_number_of_identifications)
print("")



#
# All done.
#


print("Exiting.")
print("")


exit(0)
