# Computer Vision Project
# Exporing Color Space Conversion and Transformations
# Author: Jimmy Nguyen
# Email: Jimmy@Jimmyworks.net

# Image Processing

# This file contains all the transformations for this project.

import math
import copy

# Linear Scaling
# Performs linear scaling on pixel value, x, passed in.
# Arguments:
#   x   original pixel value
#   a   original image lower bound
#   b   original image upper bound
#   A   new image lower bound
#   B   new image upper bound
# Returns:
#   x   new pixel value
def linearScaling(x, a, b, A, B):
    x = ((x-a)*(B-A))/(b-a)+A
    return x

# Histogram Equalization
# Performs histogram equalization based on histogram
# and image size passed in.
# Arguments:
#   histogram    A 2D list where first column are pixel
#                values and second are pixel count
#   image_size   Size of image (pixel count)
# Returns:
#   g            A 2D list representing the translation
#                table where first column is the original
#                pixel values and second column is the new
#                pixel values
def histogramEqualization(histogram, image_size):
    # Sizing Factor = number of pixels in new range / image size
    sizingFactor = len(histogram)/float(image_size)
    # Create array to calculate accumalated pixels from [0, i]
    f = []
    # Create deep copy of the histogram to overwrite the second
    # column with new pixel values
    g = copy.deepcopy(histogram)

    # Create the accumulated pixels array using the histogram
    f.append(histogram[0][1])
    for i in range(1, len(histogram)):
        f.append(f[i-1]+histogram[i][1])

    # Calculate the new pixel value for the full range of histogram
    g[0][1] = math.floor(f[0]/2.*sizingFactor)
    for i in range(1, len(histogram)):
        g[i][1] = math.floor((f[i-1]+f[i])/2.*sizingFactor)

    return g

