# Computer Vision Project
# Exporing Color Space Conversion and Transformations
# Author: Jimmy Nguyen
# Email: Jimmy@Jimmyworks.net

# Program 3: Histogram Equalization Transformations

# Using an input image and specified boxed region in that image,
# perform histogram equalization on the luminance values (L-values)
# in the boxed region.  This is all done through a custom functions
# in ColorConversions.py and ImageProcessing.py to demonstrate
# understanding of histogram equalization.   The remainder of the
# image should remain untouched.

import sys
import numpy as np
import cv2
import ColorConversions as colors
import ImageProcessing as image
import math

# L-value scaling range
SCALE_MIN = 0
SCALE_MAX = 100

# Check if valid number of user input arguments
if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

# Variable assignments
x1 = float(sys.argv[1])
y1 = float(sys.argv[2])
x2 = float(sys.argv[3])
y2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

# Check for valid range for top-left pixel and bottom-right pixel
if(x1<0 or y1<0 or x2<=x1 or y2<=y1 or x2>1 or y2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

# Read the input image and check if successful
inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

# Show the original image
cv2.imshow("input image: " + name_input, inputImage)

# Calculate the pixel index based on image size and user-defined width x height
rows, cols, bands = inputImage.shape
X1 = int(round(x1 * (cols - 1)))
Y1 = int(round(y1 * (rows - 1)))
X2 = int(round(x2 * (cols - 1)))
Y2 = int(round(y2 * (rows - 1)))
print("W1:", X1, "H1", Y1, "W2:", X2, "H2", Y2)

# Create histogram and count pixel size of image
image_size = 0
histogram = np.zeros((SCALE_MAX - SCALE_MIN + 1, 2))
# Write pixel values in first column of histogram table
for i in range(0, len(histogram)):
    histogram[i][0] = SCALE_MIN + i
# Lookup array for finding indices based on pixel value
pixels = [i[0] for i in histogram]

# Create a temp matrix to store Luv values
temp_box = np.zeros((Y2 + 1, X2 + 1, 3))

# Iterate through the boxed region to generate the histogram
# for the discretized L-values
for Y in range(Y1, Y2 + 1):
    for X in range(X1, X2 + 1):
        # Get BGR -> XYZ -> Luv value for this pixel
        b, g, r = inputImage[Y, X]
        x, y, z = colors.BGR_XYZ(b, g, r)
        L, u, v = colors.XYZ_Luv(x, y, z)

        # Find the index of the discretizated L-value (using floor)
        # to find the index in the histogram table
        index = pixels.index(math.floor(L))

        # Increment the number of pixels for that pixel value
        # and number of pixels in the boxed region
        histogram[index][1] = histogram[index][1] + 1
        image_size = image_size + 1

        # Store the Luv values for this pixel
        temp_box[Y, X] = [L, u, v]

# Get a translation table for old to new pixel values using the
# histogram equalization function in ImageProcessing.py
translation = image.histogramEqualization(histogram, image_size)

# Iterate through boxed region to translate pixel values based
# on translation table and write back new BGR values
for Y in range(Y1, Y2 + 1):
    for X in range(X1, X2 + 1):
        # Get saved Luv value
        L, u, v = temp_box[Y, X]

        # Find discretized L value index and look up the new
        # L value in the translation table
        index = pixels.index(math.floor(L))
        L = translation[index][1]

        # Convert Luv -> XYZ -> BGR
        x, y, z = colors.Luv_XYZ(L, u, v)
        b, g, r = colors.XYZ_BGR(x, y, z)

        # Write new BGR to image
        inputImage[Y, X] = b, g, r

# Show new image with modified boxed region and write to file
cv2.imshow("output:", inputImage)
cv2.imwrite(name_output, inputImage);


# Wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
