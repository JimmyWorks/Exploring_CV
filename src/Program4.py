# Computer Vision Project
# Exporing Color Space Conversion and Transformations
# Author: Jimmy Nguyen
# Email: Jimmy@Jimmyworks.net

# Program 4: Histogram Equalization Transformations with OpenCV Libraries

# Using an input image and specified boxed region in that image,
# perform histogram equalization on the luminance values (L-values)
# in the boxed region.  Similar to Program 3 but this time OpenCV libraries
# are used.

import sys
import cv2

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

# Simply use OpenCV to convert to Luv and perform histogram equalization in boxed region
# Then convert back to BGR
inputImage_LUV = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LUV)
inputImage_LUV[Y1:Y2+1,X1:X2+1,0] = cv2.equalizeHist(inputImage_LUV[Y1:Y2+1,X1:X2+1,0])
inputImage = cv2.cvtColor(inputImage_LUV, cv2.COLOR_LUV2BGR)

# Show new image with modified boxed region and write to file
cv2.imshow("output:", inputImage)
cv2.imwrite(name_output, inputImage);


# Wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
