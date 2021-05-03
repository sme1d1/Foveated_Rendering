# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:32:25 2021

@author: Dietrich
"""

import os.path
import numpy as np
import cv2
from skimage import data, io
from skimage.transform import warp_polar, rescale
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from copy import deepcopy
import random
import time

#%% Image Import and Warping

# Import Image
filepath = os.path.dirname(__file__)
filename = os.path.join(filepath, 'mountain.png')
img = io.imread(filename)

# Save Height and Width of Image to variables
H = len(img[:,0])
W = len(img[0,:])

# Set (x0, y0) to be pixel coordinates of the center of the image
x0 = W/2
y0 = H/2
center = (x0,y0)

# Calculate radius as the distance from the center of the image to the corner.
r = np.sqrt(x0**2 + y0**2)

# Initialize Variables
size = (W,H)

# Perform Log Polar Transform
warp = cv2.warpPolar(img, size, center, r, 256)

# Transform Log Polar back to Cartesian
warp_recovered = cv2.warpPolar(warp, size, center, r, flags=256+16)

#%% Scaling
scale = [1,1]
rescaled = rescale(img, scale, multichannel=True)
rescaled_warped = warp_polar(rescaled, scaling='log', multichannel=True)


#%% Create Exponential Distribution

# Create linear distribution from 0 to 1 as x values
# 0 is pixel at center, 1 is pixel at furthest point from center
linear_dist = np.arange(0, 1, 1/W)
# Create inverse exponential
# Set scalar variable
a = 1.5
# Calculate y values using inverse exponential equation
log_dist = np.zeros(len(linear_dist))
for i in range(len(linear_dist)):
    log_dist[i] = np.exp(-a*linear_dist[i])

#%% Generate Sparse Density Pixel Distribution from warped image

# Deepcopy warped image to preserve original
warp_sparse = deepcopy(warp)    

columns = W
rows = H

# Loop through columns of warped image
for i in range(columns-1):    

    # Get desired number of pixels to replace based on exponential distribution
    num_pixels = round(rows - (rows*log_dist[i]))
    
    # Generate array of random pixel indices with length num_pixels
    random_pixels = np.round(random.sample(range(rows), k=num_pixels)).astype('int')
    
    # Convert desired pixels to white
    for j in range(len(random_pixels)-1):
        if (len(warp_sparse[random_pixels[j], i]) == 4):
            warp_sparse[random_pixels[j], i] = [255,255,255,0]
        elif (len(warp_sparse[random_pixels[j], i]) == 3):
            warp_sparse[random_pixels[j], i] = [255,255,255]

# Recover image from sparse warp
sparse_recovered = cv2.warpPolar(warp_sparse, size, center, r, flags=256+16)

#%% Plots

###### Plot Images #########
fig, axes = plt.subplots(1, 3)
ax = axes.ravel()
ax[0].imshow(warp)
ax[1].imshow(warp_sparse)
sparse_recovered = cv2.warpPolar(warp_sparse, size, center, r, flags=256+16)
ax[2].imshow(sparse_recovered)

###### Plot Exponential Distribution #########
plt.figure(2)
plt.plot(linear_dist, log_dist)
plt.title("Inverse Exponential Distribution")
plt.xlabel("Distance from center")
plt.ylabel("Weight")


