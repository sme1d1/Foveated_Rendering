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
sparse_recovered = cv2.warpPolar(warp2, size, center, r, flags=256+16)
ax[2].imshow(sparse_recovered)
#%% Standard LPT from paper
x0 = W/2
y0 = H/2
l1 = norm([x0,y0], 2)
l2 = norm([W - x0, H - y0], 2)
l3 = norm([x0, H - y0], 2)
l4 = norm([W - x0, y0], 2)

#L = np.log(norm([W/2, H/2], 2))
L = np.log(max(max(l1,l2), max(l3,l4)))
w = W
h = H
A = L/w
B = (2*np.pi)/h
u = []
v = []
x_pp = np.zeros(int(h*w))
y_pp = np.zeros(int(h*w))
rgb = np.empty([int(H*W),4])
rgb_pp = np.empty([int(H*W),4])

i = 0
for x in range(W):
    for y in range(H):
        
        
        # Calculate distance of current pixel from center
        x_p = x - x0;
        y_p = y - y0;
        
        # Screen space pixel (x,y) in Cartesian Coordinates
        u_temp = (np.log(norm([x_p, y_p]))/L) * w
        u = np.append(u,u_temp)
        
        if (y_p >= 0):
            b = 0
        elif (y_p < 0):
            b = 1
            
        if (x_p == 0):
            v_temp = 0
        else:
            v_temp = ((np.arctan(y_p/x_p) / (2*np.pi)) * h) + (b*h)
            #v_temp = np.arctan2(y_p, x_p) + (b*h)
        v = np.append(v,v_temp)
        rgb_temp = img[y,x]
        rgb[i] = rgb_temp 
        i = i+1


#%%        
for i in range(int(w*h)):
    x_pp[i] = np.exp(A*u[i])*np.cos(B*v[i]) + x0
    y_pp[i] = np.exp(A*u[i])*np.sin(B*v[i]) + y0
    rgb_pp[i] = rgb[i]

x_pp = np.round(x_pp + abs(min(x_pp)))
y_pp = np.round(y_pp + abs(min(y_pp)))
rgb_uint8 = rgb_pp.astype(np.uint8)
x_uint8 = x_pp.astype(np.uint8)
y_uint8 = y_pp.astype(np.uint8)

img_pp = np.empty([max(y_uint8)+1, max(x_uint8)+1, 4])
for i in range(len(x_uint8)):
    #print(rgb_uint8[i])
    img_pp[y_uint8[i], x_uint8[i]] = rgb_uint8[i]

img_pp = img_pp.astype(np.uint8)

plt.imshow(img_pp)

#%% Kernel LPT

# Calculate distance from fovea point to each corner
l1 = norm([x0,y0], 2)
l2 = norm([W - x0, H - y0], 2)
l3 = norm([x0, H - y0], 2)
l4 = norm([W - x0, y0], 2)

# Set alpha value for kernal equation
a = 2

# Set resolution scale and calculate compressed image height and width
res_scale = 0.5
h = H*res_scale
w = W*res_scale

# Instantiate empty lpt_img variable
#lpt_img = np.zeros([w,h])

# Loop through each pixel

polar_coords = []
u = []
v = []
for x in range(W):
    for y in range(H):
        # print(x,y)
        # Calculate distance from fovea point
        x_p = x - x0
        y_p = y - y0
        
        # Calculate L
        L = np.log(max(max(l1,l2), max(l3,l4)))
        
        
        # Calculate u
        u_new = (1/((np.log(norm([x_p,y_p], 2))/L)**a))*w
        u = np.append(u, u_new)
        # Calculate v
        if (y_p >= 0):
            b = 0
        elif (y_p < 0):
            b = 1
        v_new = (np.arctan2(y_p, x_p) + (b * 2 * np.pi)) * (h/(2*np.pi))
        v = np.append(v, v_new)

x_pp = np.zeros(int(h*w))
y_pp = np.zeros(int(h*w))

A = L/w
B = (2*np.pi)/h  
for i in range(int(h*w)):
    x_pp[i] = np.exp(A*(u[i]**a)) * np.cos(B*v[i]) + x0
    y_pp[i] = np.exp(A*(u[i]**a)) * np.sin(B*v[i]) + y0
     

