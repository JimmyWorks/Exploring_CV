# Computer Vision Project
# Exporing Color Space Conversion and Transformations
# Author: Jimmy Nguyen
# Email: Jimmy@Jimmyworks.net

# Program 1: Color Space Conversions

# Write a program that displays continuous changes in color for Luv representation.
# The input to the program is a width and a height. The output is an image of
# dimension width x height that is displayed on the screen. For the image created,
# the pixel at row i and column j should have the color value:
# L= 90,
# u= 354*j/width−134,
# v= 262*i/height−140

# Functions for color space conversions can be found in ColorConversions.py

from __future__ import print_function
import cv2
import numpy as np
import sys
import ColorConversions as colors

# Check if correct number of user input fields
if (len(sys.argv) != 4):
    print(sys.argv[0], ": takes 3 arguments. Not ", len(sys.argv) - 1)
    print("Expecting arguments: width height and output filename.")
    print("Example:", sys.argv[0], " 200 300 output.png")
    sys.exit()

# Variable assignments
cols = int(sys.argv[1]) # pixel columns
rows = int(sys.argv[2]) # pixel rows
output = sys.argv[3]    # output filename

# Initialize the image with all 0s
image = np.zeros([rows, cols, 3], dtype='uint8')

# Iterate over the image rows and columns
for i in range(0, rows):
    for j in range(0, cols):
        # Calculate this pixel's Luv value
        # 0≤L≤100 , −134≤u≤220, −140≤v≤122
        L = 90
        u = (354 * j / cols) - 134
        v = (262 * i / rows) - 140

        # Convert color from Luv -> XYZ -> BGR
        x, y, z = colors.Luv_XYZ(L, u, v)
        b, g, r = colors.XYZ_BGR(x, y, z)

        # Write the pixel into the image
        image[i, j] = np.array([b, g, r], dtype='uint8')

# Show output and write to file
cv2.imshow("Luv Color Spectrum:", image)
cv2.imwrite(output, image)

# Wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
