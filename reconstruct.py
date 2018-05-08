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

MASK_SIZE = 25
network = input_data(shape=[MASK_SIZE, MASK_SIZE, 1],
                     data_preprocessing=img_prep)
print(network)

network = conv_2d(network, 32, 3, activation='relu')

network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')

network = conv_2d(network, 64, 3, activation='relu')

network = max_pool_2d(network, 2)

network = fully_connected(network, 1024, activation='relu')

network = dropout(network, 0.5)

network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='eye-veins.tfl.ckpt')
model.load("eye-veins-classifier.tfl")
directory = os.getcwd() + '/'
# Load the image file
img = array(img.load(directory + "test2.jpg", True))
reconstructed = np.zeros((img.shape[0], img.shape[1]))

for x in range(0, img.shape[0] - MASK_SIZE - 1):
    print(x)
    for y in range(0, img.shape[1] - MASK_SIZE - 1):
        centerX = x + int(MASK_SIZE / 2)
        centerY = y + int(MASK_SIZE / 2)
        sample = img[x:(x + MASK_SIZE), y:(y + MASK_SIZE)]
        sample = array(sample).reshape(MASK_SIZE, MASK_SIZE, 1)
        prediction = model.predict([sample])
        result = prediction.T[0]

        if result > 0.4:
            reconstructed[centerX][centerY] = 1
        else:
            reconstructed[centerX][centerY] = 0

# Predict
reconstructed = array(reconstructed)
np.save("plik2", reconstructed)
# reconstructed = np.load("plik.npy")
# PostProcessing
# reconstructed = morph.dilation(morph.erosion(reconstructed))
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
