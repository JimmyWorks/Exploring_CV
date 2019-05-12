# Computer Vision Project
# Exporing Color Space Conversion and Transformations
# Author: Jimmy Nguyen
# Email: Jimmy@Jimmyworks.net

# Program 2: Linear Scaling Transformations

# Using an input image and specified boxed region in that image,
# perform linear scaling on the luminance values (L-values) in
# that boxed region.  This is all done through a custom functions
# in ColorConversions.py and ImageProcessing.py to demonstrate
# understanding of linear scaling.   The remainder of the image
# should remain untouched.

import sys
import numpy as np
import cv2
import ColorConversions as colors
import ImageProcessing as image

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
if(inputImage is None):
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

# Find the min and max range for L value in the window
L_min = 100
L_max = 0
# Use a temp matrix to store the Luv for the boxed region
temp_box = np.zeros((Y2 + 1, X2 + 1, 3))

# Iterate through the boxed region for Luv values
# and collect the L min and max
for Y in range(Y1, Y2 + 1):
    for X in range(X1, X2 + 1):
        # Get the BGR value and convert to Luv
        b, g, r = inputImage[Y, X]
        x, y, z = colors.BGR_XYZ(b, g, r)
        L, u, v = colors.XYZ_Luv(x, y, z)

        # Check if new L value is new L min
        if L < L_min:
            L_min = L
        # Check if new L value is new L max
        if L > L_max:
            L_max = L

        # Keep the Luv value in temp matrix
        temp_box[Y, X] = [L, u, v]

print("L range:", L_min, L_max)

# Using the L min and max values, iterate through the
# boxed region again and linearly scale the L-value
# writing it back to the image
for Y in range(Y1, Y2 + 1):
    for X in range(X1, X2 + 1):
        # Get the Luv value saved
        L, u, v = temp_box[Y, X]
        # Perform linear scaling on L value
        L = image.linearScaling(L, L_min, L_max, SCALE_MIN, SCALE_MAX)

        # Convert it Luv -> XYZ -> BGR
        x, y, z = colors.Luv_XYZ(L, u, v)
        b, g, r = colors.XYZ_BGR(x, y, z)

        # Write BGR back to image
        inputImage[Y, X] = b, g, r

# Show the final result and write to output file
cv2.imshow("output:", inputImage)
cv2.imwrite(name_output, inputImage);


# Wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
