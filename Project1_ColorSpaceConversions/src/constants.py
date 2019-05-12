# Computer Vision Project
# Exporing Color Space Conversion and Transformations
# Author: Jimmy Nguyen
# Email: Jimmy@Jimmyworks.net

# Constants used for color space conversions

import numpy as np

# XYZ/Luv Constants
# Using D65 with Yw = 1:
Xw = 0.95
Yw = 1.0
Zw = 1.09
uw = 4*Xw/(Xw+15*Yw+3*Zw)
vw = 9*Yw/(Xw+15*Yw+3*Zw)
# Threshold values
t_threshold = 0.008856
L_threshold = 7.9996

# Matrix for XYZ/BGR Conversions
RGB_XYZ_ConvMatrix = np.array([[0.412453, 0.35758, 0.180423],
                               [0.212671, 0.71516, 0.072169],
                               [0.019334, 0.119193, 0.950227]])