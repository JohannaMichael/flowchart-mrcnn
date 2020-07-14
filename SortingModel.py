import json
import os
import skimage.io
import numpy as np
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


filename = "./FlowchartDataMRCNN/TrainingImages/flowchart_symbol_order.txt"
doc = load_doc(filename)
sorted_array = load_descriptions(doc)

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

    image_path = os.path.join(dataset_dir, a['filename'])
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]
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

padded_flowchart_symbols = tf.keras.preprocessing.sequence.pad_sequences(
    flowcharts_symbols_array, padding="post"
)

padded_sorted_flowchart_symbols = tf.keras.preprocessing.sequence.pad_sequences(
    sorted_array, padding="post"
)

print('Flowcharts Array: ')
print(padded_flowchart_symbols)
print(type(padded_flowchart_symbols))
print('Sorted Flowcharts Array: ')
print(padded_sorted_flowchart_symbols)
print(type(padded_sorted_flowchart_symbols))
