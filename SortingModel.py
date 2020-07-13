import json
import os

import keras
import skimage.io
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

flowcharts_symbols_array = []

ROOT_DIR = os.path.abspath("./")
dataset_dir = os.path.join(ROOT_DIR, "FlowchartDataMRCNN/TrainingImages")

annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
annotations = list(annotations.values())  # don't need the dict keys

# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
annotations = [a for a in annotations if a['regions']]
# Add images
for a in annotations:
    single_flowchart_symbol_array = []
    flowchart_symbols = [r['region_attributes'] for r in a['regions']]

    # load_mask() needs the image size to convert polygons to masks.
    # Unfortunately, VIA doesn't include it in JSON, so we must read
    # the image. This is only managable since the dataset is tiny.
    image_path = os.path.join(dataset_dir, a['filename'])
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]
    for i, p in enumerate(flowchart_symbols):
        # "name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
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
# -----------------Model starts------------------

n = 100000
x_train = np.zeros((n, 13))
for i in range(n):
    x_train[i, :] = np.random.permutation(13)

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    flowcharts_symbols_array, padding="post"
)
x_train_sorted = np.sort(x_train, axis=1)
padded_inputs_sorted = np.sort(padded_inputs, axis=1)
print(x_train_sorted)
print(x_train.shape)
y_train = np.zeros((10, 13,))
print(y_train)

# i = Input(shape=(43,))
# a = Dense(1024, activation='relu')(i)
# b = Dense(512, activation='relu')(a)
# ba = Dropout(0.3)(b)
# c = Dense(256, activation='relu')(ba)
# d = Dense(128, activation='relu')(c)
# o = Dense(43)(d)
#
# model = Model(inputs=i, outputs=o)
#
# model.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.mean_squared_error, metrics=['accuracy'])
#
# model.fit(padded_inputs, padded_inputs_sorted, epochs=15, batch_size=8)
