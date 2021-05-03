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


# Import Image
filepath = os.path.dirname(__file__)
filename = os.path.join(filepath, 'test.png')
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
increased_size = (int(W*0.8), int(H*0.8))
center = (x0,y0)

# Perform Log Polar Transform
warp = cv2.warpPolar(img, size, center, r, 256)

# Transform Log Polar back to Cartesian
warp_recovered = cv2.warpPolar(warp, size, center, r, flags=256+16)

# # Plots
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

#%%
# Normalize log dist

# Set scalar variable
a = 10

# Create linear distribution from 0 to 1 as x values
# 0 is pixel at center, 1 is pixel at furthest point from center
linear_dist = np.arange(0, 1, 1/W)

# Calculate y values using inverse exponential equation
log_dist = np.zeros(len(linear_dist))
for i in range(len(linear_dist)):
    log_dist[i] = np.exp(-a*linear_dist[i])

# plt.figure(1)
# plt.plot(linear_dist, log_dist)
# plt.title("Inverse Exponential Distribution")
# plt.xlabel("Distance from center")
# plt.ylabel("Weight")



warp_sparse = deepcopy(warp)

# for i in range(2):
#     if i != 0:
    
columns = W
rows = H

for i in range(columns-1):
    
    if i != 0:
        # get number of pixels
        num_pixels = round(rows - (rows*log_dist[i]))
        print(num_pixels)
        #print(num_pixels)
        # get pixel indices
        #random_num = np.random.random(num_pixels)*(rows-1)
        random_num = random.choices(range(480), k=num_pixels)
        #print(random_num)
        rounded_num = np.round(random_num).astype('int')
        #print(rounded_num)
        
        #warp_sparse[i, rounded_num[0]] = np.nan
        # assign nan to pixel indices
        for j in range(len(rounded_num)-1):
            #print(i,rounded_num[j])
            warp_sparse[rounded_num[j], i] = [255,255,255,0]
            

sparse_recovered = cv2.warpPolar(warp_sparse, size, center, r, flags=256+16)
plt.figure(2)
plt.imshow(warp_sparse)
#%%
            
fig, axes = plt.subplots(1, 3)
ax = axes.ravel()
ax[0].imshow(warp)
ax[1].imshow(warp_sparse)
sparse_recovered = cv2.warpPolar(warp_sparse, size, center, r, flags=256+16)
ax[2].imshow(sparse_recovered)
