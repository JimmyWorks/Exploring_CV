# Computer Vision Project
# Exporing Color Space Conversion and Transformations
# Author: Jimmy Nguyen
# Email: Jimmy@Jimmyworks.net

# Color Space Conversion Functions
# All color space conversions check for valid color
# ranges and clips values, if necessary.  Optional
# debug flags can be passed in for debug statements.

from __future__ import print_function
import constants as const
import numpy as np
import cv2

# Luv to XYZ
# Arguments:
#   L, u, v     original L, u, v values
#   debug       optional debug flag for debug statements
# Returns:
#   X, Y, Z     converted X, Y, Z values
def Luv_XYZ(L, u, v, debug = False):
    # Clip valid L range
    L = clip(L, 0, 100)

    if debug:
        print("L:", L)

    if L == 0:
        u_prime = 0
        v_prime = 0
    else:
        u_prime = (u + 13.*const.uw*L)/(13.*L)
        v_prime = (v+13.*const.vw*L)/(13.*L)

    if debug:
        print("u_w:", const.uw, "v_w:", const.vw)
        print("u':", u_prime, "v':", v_prime)

    if L > const.L_threshold:
        Y = const.Yw*((L+16.)/116.)**3
    else:
        Y = const.Yw*L/903.3

    if debug:
        print("Y:", Y)

    if v_prime == 0:
        X = 0
        Z = 0
    else:
        X = Y*2.25*u_prime/v_prime
        Z = Y*(3-0.75*u_prime-5.*v_prime)/v_prime

    if debug:
        print("X:", X, "Z:", Z)

    return X, Y, Z

# BGR to XYZ
# Arguments:
#   B, G, R     original B, G, R values
#   debug       optional debug flag for debug statements
# Returns:
#   X, Y, Z     converted X, Y, Z values
def BGR_XYZ(B, G, R, debug = False):
    if debug:
        print ("Converting to XYZ, BGR:", B, G, R)

    B = clip(int(B), 0, 255)
    G = clip(int(G), 0, 255)
    R = clip(int(R), 0, 255)

    if debug:
        print ("Clipped, BGR:", B, G, R)

    R_nonlinear = R/255.
    G_nonlinear = G/255.
    B_nonlinear = B/255.

    if debug:
        print ("nonlinear, BGR:", B_nonlinear, G_nonlinear, R_nonlinear)

    R_linear = nonlinear2linearGammaCorr(R_nonlinear)
    G_linear = nonlinear2linearGammaCorr(G_nonlinear)
    B_linear = nonlinear2linearGammaCorr(B_nonlinear)

    if debug:
        print ("linear, BGR:", B_linear, G_linear, R_linear)

    rgb = np.array([R_linear, G_linear, B_linear])
    X, Y, Z = np.linalg.solve(np.linalg.inv(const.RGB_XYZ_ConvMatrix), rgb)

    if debug:
        print ("conversion matrix to X, Y, Z:", X, Y, Z)
        cv2.waitKey(0)

    return X, Y, Z

# XYZ to Luv
# Arguments:
#   X, Y, Z     original X, Y, Z values
#   debug       optional debug flag for debug statements
# Returns:
#   L, u, v     converted L, u, v values
def XYZ_Luv(X, Y, Z, debug = False):

    if debug:
        print("X, Y, Z =", X, Y, Z)

    t = Y/const.Yw

    if debug:
        print("t", t)

    if t > const.t_threshold:
        if debug:
            print("L (pre-clip):", t)
        L = 116*t**(1/3.)-16

        if debug:
            print("L function:", L)
    else:
        L = 903.3*t

    if debug:
        print("L (pre-clip):", L)

    L = clip(L, 0, 100)

    if debug:
        print("L (post-clip):", L)

    d = X + 15.*Y + 3.*Z

    if d == 0:
        u_prime = 0
        v_prime = 0
    else:
        u_prime = 4.*X/d
        v_prime = 9.*Y/d

    u = 13.*L*(u_prime - const.uw)
    v = 13.*L*(v_prime-const.vw)

    if debug:
        cv2.waitKey(0)

    return L, u, v

# XYZ to BGR
# Arguments:
#   X, Y, Z     original X, Y, Z values
#   debug       optional debug flag for debug statements
# Returns:
#   B, G, R     converted B, G, R values
def XYZ_BGR(X, Y, Z, debug = False):
    xyz = np.array([X, Y, Z])
    C = np.linalg.solve(const.RGB_XYZ_ConvMatrix, xyz)

    if debug:
        r, g, b = C
        print("R:", r, "G:", g, "B:", b)

    R_linear = clip(C.item(0), 0, 1)
    G_linear = clip(C.item(1), 0, 1)
    B_linear = clip(C.item(2), 0, 1)

    if debug:
        print("R linear:", R_linear, "G_linear:", G_linear,
              "B_linear:", B_linear)

    R_nonLinear = linear2nonlinearGammaCorr(R_linear)
    G_nonLinear = linear2nonlinearGammaCorr(G_linear)
    B_nonLinear = linear2nonlinearGammaCorr(B_linear)

    if debug:
        print("R nonlinear:", R_nonLinear, "G_nonLinear:",
              G_nonLinear, "B_nonLinear:", B_nonLinear)

    R = R_nonLinear * 255.
    G = G_nonLinear * 255.
    B = B_nonLinear * 255.

    return B, G, R

# Linear to Non-linear Gamma Correction
# Corrects BGR values to convert linear
# to non-linear BGR
# Arguments:
#   D     B, G, or R value
# Returns:
#   I     Corrected B, G, or R value
def linear2nonlinearGammaCorr(D):
    if D < 0.00304:
        I = 12.92*D
    else:
        I = 1.055*D**(1/2.4)-0.055
    return I

# Non-linear to Linear Gamma Correction
# Corrects BGR values to convert non-linear
# to linear BGR
# Arguments:
#   v       B, G, or R value
# Returns:
#   v_corr  Corrected B, G, or R value
def nonlinear2linearGammaCorr(v):
    if v < 0.03928:
        v_corr = v/12.92
    else:
        v_corr = ((v+0.055)/1.055)**2.4

    return v_corr

# Clip
# Ensures value passed in is in valid range.
# If outside of the bounds, function returns
# the closer bound value.
# Arguments:
#   val     value to check
#   lower   lower bound
#   upper   upper bound
# Returns:
#   val     clipped value
def clip(val, lower, upper):
    if val < lower:
        val = lower
    elif val > upper:
        val = upper
    return val
