from __future__ import division, print_function, absolute_import

import datetime

import tensorflow as tf
import tflearn
from scipy._lib.six import xrange
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import random as rand
from skimage import data as img
from PIL import Image
from numpy import array
import os
import numpy as np
import matplotlib.pyplot as plt


def chunkify(lst, n):
    return [lst[i::n] for i in xrange(n)]


images = []
masks = []
k = 10
X = []
Y = []
Xtest = []
Ytest = []
MASK_SIZE = 25  # musi być nieparzysta
PHOTO_SAMPLES = 8000
# ładuj zdjęcia
directory = os.getcwd() + "\Small" + '/'
for file in os.listdir("Small"):
    if file.endswith(".jpg"):
        jpgfile = img.load(directory + file, True)
        images.append(array(jpgfile))
    if file.endswith(".tif"):
        tiffile = Image.open("Small/" + file)
        masks.append(array(tiffile))
    if file.endswith(".png"):
        pngfile = img.load(directory + file, True)
        masks.append(array(pngfile))
# plt.imshow(array(masks[1]))
# plt.show()
for mask in masks:
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] > 10:
                mask[i][j] = 255
            else:
                mask[i][j] = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

for image, mask in zip(images, masks):
    for i in range(0, PHOTO_SAMPLES):
        flag = -1;
        veinFlag = rand.randint(0, 2)
        while (flag != veinFlag):
            startX = rand.randint(5, image.shape[0] - (MASK_SIZE + 5))
            endX = startX + MASK_SIZE
            startY = rand.randint(5, image.shape[1] - (MASK_SIZE + 5))
            endY = startY + MASK_SIZE
            centerX = startX + int(MASK_SIZE / 2)
            centerY = startY + int(MASK_SIZE / 2)
            sample = image[startX:endX, startY:endY]
            sample = array(sample).reshape(MASK_SIZE, MASK_SIZE, 1)
            temp = mask[centerX, centerY]
            result = []
            if (temp == 255):
                result = [1, 0]
                flag = rand.randint(1, 2)
            else:
                result = [0, 1]
                flag = 0
        # result = array(result).reshape(2)
        X.append(sample)
        Y.append(result)

X, Y = shuffle(X, Y)

splittedSamples = chunkify(X, k)
splittedMasks = chunkify(Y, k)

timeStamp = datetime.datetime.now().time()

averageAcc = 0;
for i in range(0, k):

    trainSamples = []
    testSamples = []

    trainMasks = []
    testMasks = []
    splittedSamplesTemp = list(splittedSamples)
    splittedMasksTemp = list(splittedMasks)

    testSamples.extend(splittedSamplesTemp.pop(i))
    testMasks.extend(splittedMasksTemp.pop(i))

    trainSamples = [item for sublist in splittedSamplesTemp for item in sublist]
    trainMasks = [item for sublist in splittedMasksTemp for item in sublist]

    X = list(trainSamples)
    Y = list(trainMasks)

    Xtest = list(testSamples)
    Ytest = list(testMasks)

    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    network = input_data(shape=[MASK_SIZE, MASK_SIZE, 1],
                         data_preprocessing=img_prep, name=('InputData' + str(i)))
    print(network)

    network = conv_2d(network, 32, 3, activation='relu')

    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 3, activation='relu')

    network = conv_2d(network, 64, 3, activation='relu')

    network = max_pool_2d(network, 2)

    network = fully_connected(network, 256, activation='relu')

    network = dropout(network, 0.4)

    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=8, shuffle=True, validation_set=(Xtest, Ytest),
              show_metric=True, batch_size=64, snapshot_epoch=True)

    accuracy = 0

    for toPredict, actual in zip(Xtest, Ytest):
        prediction = model.predict([toPredict])
        predicted_class = np.argmax(prediction)
        actual_class = np.argmax(actual)
        if (predicted_class == actual_class):
            accuracy += 1
    accuracy = accuracy / len(Ytest)
    averageAcc += accuracy
    print("Acucracy: " + str(accuracy))

    model.save("docelowy4/" + str(accuracy) + " eye-veins-classifier.tfl")

    print("Network trained and saved as eye-veinsclassifier.tfl!")
    tf.reset_default_graph()

averageAcc = averageAcc / k
print(averageAcc)
text_file = open("docelowy4/averageACC.txt", "w")
text_file.write("Średnia dokładność: " + str(averageAcc))
text_file.close()
