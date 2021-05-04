# TF AutoEncoder - Team Fovea - Deep Learning - sme1d1


import numpy as np
import os
import cv2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import math
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
'''
# Create our dataset
training_data = []

# for generating our dataset
for img in os.listdir(path):
    # print(img.title())
    pic = cv2.imread(os.path.join(path, img))  # read the images in the path
    #pic = pic[:, :, ::-1]
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)  # convert images to grayscale
    pic = cv2.resize(pic, (rowcol, rowcol))  # resize to square)
    training_data.append([pic])  # append dataset

np.save(os.path.join(savepath, 'catssmall'), np.array(training_data))  # save as .npy
saved = np.load(os.path.join(savepath, 'catssmall.npy'))

#dataset gen complete
'''

filename = './imgdb/catssmall.npy'
dataset = np.load(filename) # load dataset

# -------- Our sparse image creator ---------

def sparseimg(image):
    # Displaying the image

    # cv2.imshow("foxhead", img)  # show image
    # cv2.waitKey(0)  # wait for keypress
    # cv2.destroyAllWindows()  # destroy windows

    radius = 362
    # radius = 1024
    angle = 45
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)  # resize image

    # Display Log Polar conversion
    image_polar = warp_polar(image, radius=radius,  scaling='log') #multichannel=True for color

    # y=a(1-b)^x
    # y = final amount
    # a = original amount
    # b = decay factor
    # x = time passed

    # Decay function
    red_size = .005  # .01 max size
    dim = image_polar.shape[0] * image_polar.shape[1]
    t = np.arange(0.0, radius, 1)
    s = (dim * (1 - red_size) ** t) / dim

    warp_sparse = deepcopy(image_polar)
    p = (warp_sparse.shape[1] * s)
    for val in range(image_polar.shape[1]):
        p[val] = image_polar.shape[1] - round(p[val], 0)

    columns = warp_sparse.shape[1]
    rows = warp_sparse.shape[0]

    # Loop through columns of warped image
    for i in range(columns - 1):

        # Get desired number of pixels to replace based on exponential distribution
        num_pixels = int(p[i])

        # Generate array of random pixel indices with length num_pixels
        random_pixels = np.round(random.sample(range(rows), k=num_pixels)).astype('int')
        # print(len(random_pixels))

        # Convert desired pixels to white
        for j in range(len(random_pixels) - 1):
            warp_sparse[random_pixels[j], i] = 255
            '''
            if (len(warp_sparse[random_pixels[j], i]) == 4):
                warp_sparse[random_pixels[j], i] = [255, 255, 255, 255]
            elif (len(warp_sparse[random_pixels[j], i]) == 3):
                warp_sparse[random_pixels[j], i] = [255, 255, 255]
            '''
    # Transform Log Polar back to Cartesian
    warp_recovered = cv2.warpPolar(warp_sparse, (128, 128), (64, 64), radius, flags=256 + 16)
    warp_recovered = np.float32(warp_recovered)
    #warp_recovered = cv2.cvtColor(warp_recovered, cv2.COLOR_BGR2GRAY)
    return warp_recovered


# END -------- Our sparse image creator ---------


# Auto Encoder

from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

# Load dataset

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(dataset, test_size=0.25)

print(len(x_test))
z = len(x_test)
xran = len(x_test) - 1


for i in range(xran):
    zed = sparseimg(x_test[i,0])
    x_test[i,0] = zed
    print("%i done" % i)

img = x_test[xran,0]
#cv2.imshow("testimage", img)  # show image
#cv2.waitKey(0)  # wait for keypress
#cv2.destroyAllWindows()  # destroy windows

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

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

history = autoencoder.fit(x_train, x_train,
                          epochs=100,
                          batch_size=1024,
                          shuffle=True,
                          verbose=2,
                          validation_data=(x_test, x_test))

# Create predictions

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Plot our images and reconstructions

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
