from __future__ import print_function

import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
import numpy as np

pathToTrainImages = './FlowchartDataMRCNN/TrainingImages/'
# Create a list of all image names in the directory
img_names = glob.glob(pathToTrainImages + '*.jpg')

train_images = []

for i in img_names:  # img is list of full path names of all images
    train_images.append(i)  # Add it to the list of train images

print(train_images)












#
#
# n = 100000
# x_train = np.zeros((n, 43))
# for i in range(n):
#     x_train[i, :] = np.random.choice(13, 43, replace=True)
#
# # x_train = x_train.reshape(n, 4, 1)
# y_train = np.sort(x_train, axis=1)
#
# n = 1000
# x_test = np.zeros((n, 43))
# for i in range(n):
#     x_test[i, :] = np.random.choice(13, 43, replace=True)
#
# # x_test = x_test.reshape(n, 4, 1)
# y_test = np.sort(x_test, axis=1)
#
# print(x_train[0])
# print(y_train[0])
# print(x_train.shape)
# print(y_train.shape)
#
# input_shape = (43,)
#
# model = Sequential()
# model.add(Dense(128, input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.15))
# model.add(Dense(43))
#
# model.compile(loss=keras.losses.mean_squared_error,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# epochs = 10
# batch_size = 128
# # Fit the model weights.
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
#
# test_list = [[10, 2, 5, 12, 4, 3, 9, 3, 6, 8, 11, 12, 12, 1, 0, 1, 8, 2,
#               0, 6, 1, 9, 8, 9, 11, 0, 2, 6, 11, 5, 0, 1, 7, 10, 11, 8,
#               0, 3, 11, 11, 8, 8, 7]]
# print(np.shape(test_list))
# print(np.asarray(test_list))
# print(np.shape(test_list))
# pred = model.predict(np.asarray(test_list))
# print(test_list)
# print(pred)
# print([np.asarray(test_list).reshape(43, )[np.abs(np.asarray(test_list).reshape(43, ) - i).argmin()] for i in
#        list(pred[0])])
