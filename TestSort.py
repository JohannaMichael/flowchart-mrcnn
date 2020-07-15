from __future__ import print_function
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, concatenate
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K, Model
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import json
import os
import skimage.io
import tensorflow as tf


def load_doc(filename_data_input):
    # open the file as read only
    file = open(filename_data_input, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_descriptions(data):
    sorted_flowchart_array = []
    # process lines
    for line in data.split('\n'):
        # split lines by white space
        tokens = line.split()
        int_tokens = []
        for symbols in tokens:
            int_tokens.append(int(symbols))
        sorted_flowchart_array.append(int_tokens)
    return sorted_flowchart_array


filename = './FlowchartDataMRCNN/TrainingImages/flowchart_symbol_order.txt'
doc = load_doc(filename)
sorted_array = load_descriptions(doc)

flowcharts_symbols_array = []

ROOT_DIR = os.path.abspath('./')
dataset_dir = os.path.join(ROOT_DIR, 'FlowchartDataMRCNN/TrainingImages')

annotations = json.load(open(os.path.join(dataset_dir, 'via_region_data.json')))
annotations = list(annotations.values())  # don't need the dict keys

# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
annotations = [a for a in annotations if a['regions']]
# Add images
for a in annotations:
    single_flowchart_symbol_array = []
    flowchart_symbols = [r['region_attributes'] for r in a['regions']]

    image_path = os.path.join(dataset_dir, a['filename'])

    for i, p in enumerate(flowchart_symbols):

        if p['flowchart_symbols'] == 'terminal_start':
            single_flowchart_symbol_array.insert(i, 1)
        elif p['flowchart_symbols'] == 'flowline':
            single_flowchart_symbol_array.insert(i, 2)
        elif p['flowchart_symbols'] == 'input':
            single_flowchart_symbol_array.insert(i, 3)
        elif p['flowchart_symbols'] == 'decision':
            single_flowchart_symbol_array.insert(i, 4)
        elif p['flowchart_symbols'] == 'process':
            single_flowchart_symbol_array.insert(i, 5)
        elif p['flowchart_symbols'] == 'terminal_end':
            single_flowchart_symbol_array.insert(i, 6)
        elif p['flowchart_symbols'] == 'process_end':
            single_flowchart_symbol_array.insert(i, 7)
        elif p['flowchart_symbols'] == 'process_start':
            single_flowchart_symbol_array.insert(i, 8)
        elif p['flowchart_symbols'] == 'connector':
            single_flowchart_symbol_array.insert(i, 9)
        elif p['flowchart_symbols'] == 'document':
            single_flowchart_symbol_array.insert(i, 10)
        elif p['flowchart_symbols'] == 'terminal':
            single_flowchart_symbol_array.insert(i, 11)
    flowcharts_symbols_array.append(single_flowchart_symbol_array)

x_train_symbols = tf.keras.preprocessing.sequence.pad_sequences(
    flowcharts_symbols_array, padding="post"
)

y_train_symbols_sorted = tf.keras.preprocessing.sequence.pad_sequences(
    sorted_array, padding="post"
)

# print('Flowcharts Array: ')
# print(padded_flowchart_symbols)
# print(type(padded_flowchart_symbols))
# print('Sorted Flowcharts Array: ')
# print(padded_sorted_flowchart_symbols)
# print(type(padded_sorted_flowchart_symbols))

pathToTrainImages = './FlowchartDataMRCNN/TrainingImages/'
# Create a list of all image names in the directory
img_names = glob.glob(pathToTrainImages + '*.jpg')

train_images = []
x_train_pics = []

for i in img_names:
    train_images.append(i)

for image_path in train_images:
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, color_mode='grayscale', target_size=(250, 250))
    # print(np.shape(img))
    # Convert PIL image to numpy array
    x = image.img_to_array(img)
    x_train_pics.append(x)

print(np.shape(x_train_pics))
max_len = max([len(i) for i in x_train_symbols])
print(max_len)
print(np.shape(x_train_pics)[1])


def create_cnn(width, height, depth):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    input_shape = (height, width, depth)
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(128))
    cnn_model.add(Activation('relu'))
    return cnn_model


def create_mlp(dim):
    mlp_model = Sequential()
    mlp_model.add(Dense(128, input_shape=(dim,), activation="relu"))
    mlp_model.add(Activation('relu'))
    mlp_model.add(Dropout(0.3))
    mlp_model.add(Dense(128))
    mlp_model.add(Activation('relu'))
    return mlp_model


mlp = create_mlp(max_len)
cnn = create_cnn(np.shape(x_train_pics)[1], np.shape(x_train_pics)[2], np.shape(x_train_pics)[3])
combinedInput = concatenate([mlp.output, cnn.output])
x = Dense(128, activation="relu")(combinedInput)
x = Dense(max_len)(x)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# train the model
model.fit(
    x=[np.array(x_train_symbols), np.array(x_train_pics)], y=np.array(y_train_symbols_sorted),
    epochs=30, batch_size=4, verbose=1)

# preds = model.predict([testAttrX, testImagesX])

#
# model = Sequential()
# model.add(Dense(128, input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Activation('relu'))
# # model.add(Dropout(0.15))
# # model.add(Dense(43))
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
