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

from keras.datasets import mnist, fashion_mnist
def sparse(img, dist_scale):
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
    
    #Scaling
    scale = [1,1]
    rescaled = rescale(img, scale, multichannel=True)
    rescaled_warped = warp_polar(rescaled, scaling='log', multichannel=True)
    
    
    # Create Exponential Distribution
    
    # Create linear distribution from 0 to 1 as x values
    # 0 is pixel at center, 1 is pixel at furthest point from center
    linear_dist = np.arange(0, 1, 1/W)
    # Create inverse exponential
    # Set scalar variable
    a = dist_scale
    # Calculate y values using inverse exponential equation
    log_dist = np.zeros(len(linear_dist))
    for i in range(len(linear_dist)):
        log_dist[i] = np.exp(-a*linear_dist[i])
    
    # Generate Sparse Density Pixel Distribution from warped image
    
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
    
    
    ###### Plot Images #########
    # fig, axes = plt.subplots(1, 3)
    # ax = axes.ravel()
    # ax[0].imshow(warp)
    # ax[1].imshow(warp_sparse)
    # sparse_recovered = cv2.warpPolar(warp_sparse, size, center, r, flags=256+16)
    # ax[2].imshow(sparse_recovered)
    
    ###### Plot Exponential Distribution #########
    # plt.figure(2)
    # plt.plot(linear_dist, log_dist)
    # plt.title("Inverse Exponential Distribution")
    # plt.xlabel("Distance from center")
    # plt.ylabel("Weight")
    
    return sparse_recovered

# Import an image and generate sparse version
mainpath = os.path.dirname(__file__)
filepath = os.path.join(mainpath, 'images\cat')
filename = os.path.join(filepath, 'cat_0000.jpg')
img1 = io.imread(filename)
img1_sparse = sparse(img1,1.5)


#%% Save all sparse images to a folder
normal_imgs = []
sparse_imgs = []
print(filepath)
time1 = time.time()
a = 1
i=0
savepath = os.path.join(mainpath, 'images\cat_sparse')

for img in os.listdir(filepath)[0:3]:
    #print(img)
    pic = cv2.imread(os.path.join(filepath, img))
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    pic_sparse = sparse(pic, a)
    # sparse_imgs = np.append(sparse_imgs, pic_sparse)
    # normal_imgs = np.append(normal_imgs, pic)
    sparse_imgs.append(pic_sparse)
    normal_imgs.append(pic)
    #plt.imshow(pic_sparse)
    #cv2.imwrite(os.path.join(savepath,'sparse' + str(i) + '.jpg'), pic_sparse)
    i+=1
    
print(time.time() - time1)
#%% Load Sparse images from folder
sparse_imgs = [] 
sparse_path = os.path.join(mainpath, 'images\cat_sparse')
for i in range(len(os.listdir(sparse_path)[0:10])):
    print(os.listdir(sparse_path)[i])
    # pic = cv2.imread(os.path.join(sparse_path, img))
    # pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    # sparse_imgs.append(pic)
#%% Load Normal images from folder
normal_imgs = []
normal_path = os.path.join(mainpath, 'images\cat')
for img in os.listdir(normal_path)[0:10]:
    print(img)
    # pic = cv2.imread(os.path.join(normal_path, img))
    # pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    # normal_imgs.append(pic)

#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sparse_imgs, normal_imgs, shuffle=False)

#%%
i=5
plt.figure(1)
plt.imshow(sparse_imgs[i])
plt.figure(2)
plt.imshow(normal_imgs[i])

#%%
i=5
plt.figure(1)
plt.imshow(x_train[i])
plt.figure(2)
plt.imshow(y_train[i])

#%%
# Auto Encoder

from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

# Load dataset

print(len(x_test))
z = len(x_test)

'''
x_train, x_test = train_test_split(dataset, test_size=0.25)
zed = x_test[1,0]
cv2.imshow("testimage", zed)  # show image
cv2.waitKey(0)  # wait for keypress
cv2.destroyAllWindows()  # destroy windows

zed = sparseimg(zed)
cv2.imshow("testimage", zed)  # show image
cv2.waitKey(0)  # wait for keypress
cv2.destroyAllWindows()  # destroy windows
'''



# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# y_train = y_train.astype('float32') / 255.
# y_test = y_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = y_train.reshape((len(y_train), np.prod(y_train.shape[1:])))
y_test = y_test.reshape((len(y_test), np.prod(y_test.shape[1:])))

#%%
image_size_squared = 128**2

# this is the size of our encoded representations
encoding_dim = 2048  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# this is our input placeholder
input_img = Input(shape=(image_size_squared,))

# "encoded" is the encoded representation of the input

hidden_layer = Dense(1024, activation='relu')(input_img)  # add hidden layer
hidden_layer = Dense(512, activation='relu')(hidden_layer)  # add hidden layer
encoded = Dense(encoding_dim, activation='relu')(hidden_layer)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(image_size_squared, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Creating a model for encoder (predictions to display test images)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))

# Create a decoder model layer as the last layer from the autoencoder
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(x_train, y_train,
                          epochs=100,
                          batch_size=1024,
                          shuffle=True,
                          verbose=2,
                          validation_data=(x_test, y_test))

# Create predictions

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Plot our images and reconstructions
rowcol=128
n = 5  # How many images to display
plot1 = plt.figure(figsize=(16, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(rowcol, rowcol))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(rowcol, rowcol))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# print((history.history.keys()))

# "Loss Plot"
plot3 = plt.figure(4)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()