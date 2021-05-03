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

#%%

# Import Image
filepath = os.path.dirname(__file__)
filename = os.path.join(filepath, 'fox.png')
img = io.imread(filename)

# Save Height and Width of Image to variables
H = len(img[:,0])
W = len(img[0,:])

# Set (x0, y0) to be pixel coordinates of the center of the image
x0 = W/2
y0 = H/2

# Calculate radius as the distance from the center of the image to the corner.
r = np.sqrt(x0**2 + y0**2)

##### LPT built-in

# Initialize Variables
size = (W,H)
increased_size = (int(W), int(H))
center = (x0,y0)

# Perform Log Polar Transform
warp = cv2.warpPolar(img, size, center, r, 256)

# Transform Log Polar back to Cartesian
warp_recovered = cv2.warpPolar(warp, size, center, r, flags=256+16)

# Plots
fig, axes = plt.subplots(1, 3)
ax = axes.ravel()
ax[0].imshow(img)
ax[1].imshow(warp)
ax[2].imshow(warp_recovered)



#ax[2].imshow(warp_recovered)
# scale = [0.5,0.5]
# rescaled = rescale(img, scale, multichannel=True)
# rescaled_warped = warp_polar(rescaled, scaling='log', multichannel=True)
# ax[2].imshow(rescaled)
# ax[3].imshow(rescaled_warped)

#%% Normalize log dist

# Set scalar variable
a = 2

# Create linear distribution from 0 to 1 as x values
# 0 is pixel at center, 1 is pixel at furthest point from center
linear_dist = np.arange(0, 1, 1/W)

# Calculate y values using inverse exponential equation
log_dist = np.zeros(len(linear_dist))
for i in range(len(linear_dist)):
    log_dist[i] = np.exp(-a*linear_dist[i])

plt.plot(linear_dist, log_dist)
plt.title("Inverse Exponential Distribution")
plt.xlabel("Distance from center")
plt.ylabel("Weight")

#%% 

warp_sparse = deepcopy(warp)

for i in range(H):
    for j in range(W):
        num = np.random.random()
        if (np.random.random() > log_dist[j]):
            warp_sparse[i, j] = np.nan

sparse_recovered = cv2.warpPolar(warp_sparse, size, center, r, flags=256+16)
plt.imshow(sparse_recovered)
#%%
            
fig, axes = plt.subplots(1, 3)
ax = axes.ravel()
ax[0].imshow(warp)
ax[1].imshow(warp_sparse)
sparse_recovered = cv2.warpPolar(warp_sparse, size, center, r, flags=256+16)
ax[2].imshow(sparse_recovered)
