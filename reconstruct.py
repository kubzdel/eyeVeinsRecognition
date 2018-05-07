  # -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
from numpy import array
import os
from skimage import data as img
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

MASK_SIZE=25
network = input_data(shape=[MASK_SIZE, MASK_SIZE, 1],
                     data_preprocessing=img_prep)
print(network)

# Step 1: Convolution
network = conv_2d(network, 32, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 1024, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='eye-veins.tfl.ckpt')
model.load("eye-veins-classifier.tfl")
directory = os.getcwd()+'/'
# Load the image file
img = array(img.load(directory+"test2.jpg",True))
reconstructed=np.zeros((img.shape[0],img.shape[1]))

for x in range(0,img.shape[0]-MASK_SIZE-1):
    print(x)
    for y in range(0,img.shape[1]-MASK_SIZE-1):
        centerX = x + int(MASK_SIZE / 2)
        centerY = y + int(MASK_SIZE / 2)
        sample = img[x:(x+MASK_SIZE), y:(y+MASK_SIZE)]
        sample = array(sample).reshape(MASK_SIZE, MASK_SIZE, 1)
        prediction = model.predict([sample])
        result = prediction.T[0]

        if result>0.4:
             reconstructed[centerX][centerY]=1
        else:
             reconstructed[centerX][centerY]=0


# Predict
reconstructed = array(reconstructed)
np.save("plik2",reconstructed)
#reconstructed = np.load("plik.npy")
#PostProcessing
#reconstructed = morph.dilation(morph.erosion(reconstructed))
# reconstructed = np.load("plik2.npy")
plt.imshow(reconstructed)
plt.show()

# # Check the result.
# is_bird = np.argmax(prediction[0]) == 1
#
# if is_bird:
#     print("That's a bird!")
# else:
#     print("That's not a bird!")