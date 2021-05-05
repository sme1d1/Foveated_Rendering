# TF AutoEncoder - Team Fovea - Deep Learning - sme1d1


import numpy as np
import os
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
##from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Reshape, UpSampling2D
##from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import math




from keras.layers import Dropout

from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# image stuff
import sys
np.set_printoptions(threshold=sys.maxsize)

from skimage.transform import warp_polar
from copy import deepcopy
import random

path = "./images/cat"
savepath = "./imgdb/"
# Set training image
rowcol = 128  # row and column variable

# # # Create our dataset
# training_data = []
# # for generating our dataset
# for img in os.listdir(path):
#     # print(img.title())
#     pic = cv2.imread(os.path.join(path, img))  # read the images in the path
#     #pic = pic[:, :, ::-1]
#     pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)  # convert images to grayscale
#     pic = cv2.resize(pic, (rowcol, rowcol))  # resize to square)
#     training_data.append([pic])  # append dataset
# np.save(os.path.join(savepath, 'catssmall'), np.array(training_data))  # save as .npy
# saved = np.load(os.path.join(savepath, 'catssmall.npy'))
# #dataset gen complete


filename = './imgdb/catssmall.npy'
dataset = np.load(filename) # load dataset

# -------- Our sparse image creator ---------

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
    # scale = [1,1]
    # rescaled = rescale(img, scale, multichannel=True)
    # rescaled_warped = warp_polar(rescaled, scaling='log', multichannel=True)
    
    
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

# END -------- Our sparse image creator ---------

#%%
# Auto Encoder

from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

# Load dataset

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(dataset, test_size=0.25)

sparse_train = []
normal_train = []
xran = len(x_train)
for i in range(xran):
    pic = sparse(x_train[i,0], 0.75)
    sparse_train.append(pic)
    
    #cv2.imshow("testimage", sparse[i])  # show image
    #cv2.waitKey(0)  # wait for keypress
    #cv2.destroyAllWindows()  # destroy windows
    #print("%i done" % i)

sparse_train = np.array(sparse_train)


sparse_test = []

xran = len(x_test)
for i in range(xran):
    pic = sparse(x_train[i,0], 0.75)
    sparse_test.append(pic)
    # zed = sparse(x_test[i,0], 1)
    # sparse_test.append(zed)

sparse_test = np.array(sparse_test)



#%%
#print(len(x_test))
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

#cv2.imshow("testimage", sparsenp[0])  # show image
#cv2.waitKey(0)  # wait for keypress
#cv2.destroyAllWindows()  # destroy windows

#img = x_test[xran,0]
#cv2.imshow("testimage", img)  # show image
#cv2.waitKey(0)  # wait for keypress
#cv2.destroyAllWindows()  # destroy windows

normal_train = x_train.astype('float32') / 255.
sparse_train = sparse_train.astype('float32') / 255.
normal_test = x_test.astype('float32') / 255.
sparse_test = sparse_test.astype('float32') / 255.


normal_train = normal_train.reshape((len(normal_train), np.prod(normal_train.shape[1:])))
sparse_train = sparse_train.reshape((len(sparse_train), np.prod(sparse_train.shape[1:])))
normal_test = normal_test.reshape((len(normal_test), np.prod(normal_test.shape[1:])))
sparse_test = sparse_test.reshape((len(sparse_test), np.prod(sparse_test.shape[1:])))

#%%  ########## Model 1 ############

image_size_squared = (128**2)*3

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

history = autoencoder.fit(sparse_train, normal_train,
                          epochs=200,
                          batch_size=1024,
                          shuffle=True,
                          verbose=2,
                          validation_data=(sparse_test, normal_test))

# Create predictions

encoded_imgs = encoder.predict(sparse_test)
decoded_imgs = decoder.predict(encoded_imgs)

#%%
# Plot our images and reconstructions
n = 5  # How many images to display
plot1 = plt.figure(figsize=(16, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(sparse_test[i].reshape(rowcol, rowcol, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(rowcol, rowcol, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# print((history.history.keys()))

#%%
# "Loss Plot"
plot3 = plt.figure(4)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


