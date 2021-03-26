# Fox Gan 1 - Team Fovea - Deep Learning - sme1d1
# Derived from MNIST digit generation GAN

import numpy as np
import os
import cv2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

path = "./images/"
savepath = "./imgdb/"
# Set training image
rowcol = 256  # row and column variable
chan = 1  # channel variable
# Create our dataset
training_data = []
for img in os.listdir(path):
    # print(img.title())
    pic = cv2.imread(os.path.join(path, img))  # read the images in the path
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)  # convert images to grayscale
    pic = cv2.resize(pic, (rowcol, rowcol))  # resize to square)
    training_data.append([pic])  # append dataset

np.save(os.path.join(savepath, 'fox'), np.array(training_data))  # save as .npy
saved = np.load(os.path.join(savepath, 'fox.npy'))

filename = './imgdb/fox.npy'
dataset = np.load(filename)
print(dataset.shape)
for x in dataset:
    plt.axis("off")
    plt.imshow(x.astype("int32")[0], cmap='gray')
    break
plt.show()  # show our first dataset image


class GAN():  # define our GAN class
    def __init__(self):
        self.img_rows = rowcol  # rows and columns based on desired image output (must be < or = to image input)
        self.img_cols = rowcol
        self.channels = chan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # Build
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.img_rows, self.img_cols,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # noise as input => generate image => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (self.img_rows, self.img_cols,)

        model = Sequential()

        model.add(Reshape((rowcol * rowcol,), input_shape=noise_shape))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        # input layer
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(32,kernel_size=(3, 3),
                         input_shape=img_shape),)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, save_interval=1):

        # Load the dataset
        filename = './imgdb/fox.npy'
        dataset = np.load(filename)
        # Rescale -1 to 1
        dataset = (dataset[0].astype(np.float32) - 127.5) / 127.5
        dataset = np.expand_dims(dataset, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # Select a random half batch of images (currently just single image - will work with batches going forward)
            idx = np.random.randint(0, dataset.shape[0], half_batch)
            imgs = dataset[idx]

            noise = np.random.normal(0, 1, (half_batch, rowcol, rowcol,))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, rowcol, rowcol,))

            # The generator wants the discriminator to label the generated samples as valid
            # and discriminator must make mistakes for generator to learn
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot our progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # Save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 2, 2 # create a 2 x 2 image grid
        noise = np.random.normal(0, 1, (r * c, rowcol, rowcol,))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 to 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("gan/images/fox_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=250, batch_size=20, save_interval=24)
