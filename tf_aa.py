# Team Fovea - Autoencoder of Foveated Reduced images
# Authored by Scott McElfresh and Dietrich Kruse - 2021

import numpy as np
import cv2
import os
from skimage.transform import warp_polar
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from keras.models import Model
from sklearn.model_selection import train_test_split

# Generate Dataset

path = './images/cat/'  # location of our image
rowcol = 64  # pixel dimensions for row and columns of our resized images
training_data = []  # temp list
savepath = './imgdb' # save path for .npy file
'''
# for generating our dataset
for img in os.listdir(path):
    # print(img.title())
    pic = cv2.imread(os.path.join(path, img))  # read the images in the path
    #pic = pic[:, :, ::-1]
    pic = cv2.resize(pic, (rowcol, rowcol))  # resize to square)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)  # convert images to RGB
    training_data.append(pic)  # append dataset

np.save(os.path.join(savepath, 'catcolor.npy'), np.array(training_data))  # save as .npy



def sparse(img, dist_scale):
    # Save Height and Width of Image to variables
    H = len(img[:, 0])
    W = len(img[0, :])

    # Set (x0, y0) to be pixel coordinates of the center of the image
    x0 = W / 2
    y0 = H / 2
    center = (x0, y0)

    # Calculate radius as the distance from the center of the image to the corner.
    r = np.sqrt(x0 ** 2 + y0 ** 2)

    # Initialize Variables
    size = (W, H)

    # Perform Log Polar Transform
    warp = cv2.warpPolar(img, size, center, r, 256)

    # Transform Log Polar back to Cartesian
    warp_recovered = cv2.warpPolar(warp, size, center, r, flags=256 + 16)


    # Create Exponential Distribution
    # Create linear distribution from 0 to 1 as x values
    # 0 is pixel at center, 1 is pixel at furthest point from center
    
    linear_dist = np.arange(0, 1, 1 / W)
    
    # Create inverse exponential
    
    # Set scalar variable
    a = dist_scale
    # Calculate y values using inverse exponential equation
    log_dist = np.zeros(len(linear_dist))
    for i in range(len(linear_dist)):
        log_dist[i] = np.exp(-a * linear_dist[i])

    # Generate Sparse Density Pixel Distribution from warped image

    # Deepcopy warped image to preserve original
    warp_sparse = deepcopy(warp)

    columns = W
    rows = H

   # Loop through num_pixels
    pix = []
    for x in range(rows):
        for y in range(columns):
            px = img[y, x]
            tempix = []
        for x in px:
            tempix.append(x)
        pix.append(tempix)

    # create our color set
    colorset = []
    for x in pix:
        if x not in colorset:
            colorset.append(x)

    pixrange = len(pix)


    #print(random.randrange(0,pixrange,1))
    # Loop through columns of warped image
    for i in range(columns - 1):

        # Get desired number of pixels to replace based on exponential distribution
        num_pixels = round(rows - (rows * log_dist[i]))

        # Generate array of random pixel indices with length num_pixels
        random_pixels = np.round(random.sample(range(rows), k=num_pixels)).astype('int')


        # Convert desired pixels to colors from original image
        for j in range(len(random_pixels) - 1):
            pixInd = random.randrange(0, pixrange, 1)
            red = pix[pixInd][0]
            green = pix[pixInd][1]
            blue = pix[pixInd][2]
            rgb = 65536 * red + 256 * green + blue
            warp_sparse[random_pixels[j], i] = rgb

    # Recover image from sparse warp
    sparse_recovered = cv2.warpPolar(warp_sparse, size, center, r, flags=256 + 16)

    return sparse_recovered


# Load Dataset

filename = './imgdb/catcolor.npy'
X = np.load(filename)  # load dataset

print(X.shape)

plt.imshow(X[1])
plt.show()
a = .6
#generate sparse images
savepath = './imgdb'
sparsecats = []
Xlen = len(X) - 1
for i in range(Xlen):
    pic = sparse(X[i], a)
    plt.imsave('./images/savesparse_fill/sparse_%i.png'%i, pic)

savepath = './imgdb'
path_sparse = './images/savesparse_fill'
sparse_data = []
# for generating our dataset
for img in os.listdir(path_sparse):
    # print(img.title())
    pic = cv2.imread(os.path.join(path_sparse, img))  # read the images in the path
    pic = pic[:, :, ::-1]
    sparse_data.append(pic)  # append dataset

np.save(os.path.join(savepath, 'sparse.npy'), np.array(sparse_data))  # save as .npy
'''
filename = './imgdb/sparse.npy'
Z = np.load(filename)
Z = Z.astype('float32') / 255.0
# print(Z.shape)
# plt.imshow(Z[1])
# plt.show()

filename = './imgdb/catcolor.npy'
X = np.load(filename)
X = X.astype('float32') / 255.0
# plt.imshow(X[1])
# plt.show()

image_shape2 = X.shape[1:]
# print(image_shape2)
shapeSize2 = np.prod(X.shape[1:])

randstate = 120  # change for different input images

x_train1, x_test = train_test_split(Z, test_size=0.25, random_state=randstate)
x_train, x_test1 = train_test_split(X, test_size=0.25, random_state=randstate)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
'''
image_size_squared = (64 ** 2) * 3

# this is the size of our encoded representations
encoding_dim = 3072 # -> compression of factor 4

# this is our input image placeholder
input_img = Input(shape=(image_size_squared,))

# Define our encoder 

hidden_layer1 = Dense(1024)(input_img)  # add hidden layer
hidden_layer1 = BatchNormalization()(hidden_layer1)
hidden_layer1 = LeakyReLU()(hidden_layer1)

hidden_layer = Dense(512)(hidden_layer1)  # add hidden layer
hidden_layer = BatchNormalization()(hidden_layer)
hidden_layer = LeakyReLU()(hidden_layer)

encoded = Dense(encoding_dim)(hidden_layer)
encoded = LeakyReLU()(encoded)

decoded = Dense(image_size_squared, activation='sigmoid')(encoded)

# Create autoencoder model

autoencoder = Model(input_img, decoded)

# Creating a model for encoder (predictions to display test images)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))

# Input last layer from encoder as input to decoder
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(x_train, x_train,
                          epochs=100,
                          batch_size=2048,
                          shuffle=True,
                          verbose=2,
                          validation_data=(x_test, x_test))

# saving whole model
autoencoder.save('autoencoder_model.h5')
encoder.save('encoder_model.h5')
decoder.save('decoder_model.h5')
'''
# loading whole model
from keras.models import load_model

encoder = load_model('encoder_model.h5', compile=False)
decoder = load_model('decoder_model.h5', compile=False)
autoencoder = load_model('autoencoder_model.h5', compile=False)

# Create predictions

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Plot our images and reconstructions
n = 20  # How many images to display
plotrow = 4
plot1 = plt.figure(figsize=(20, 4))

for i in range(n):
    # Display original
    ax = plt.subplot(plotrow, n-10, i + 1)
    plt.imshow(x_test[i].reshape(rowcol, rowcol, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(plotrow, n-10, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(rowcol, rowcol, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(plotrow, n-10, i + 1)
    plt.imshow(x_test[i].reshape(rowcol, rowcol, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(plotrow, n-10, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(rowcol, rowcol, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

'''
# "Loss Plot"
plot3 = plt.figure(4)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

'''