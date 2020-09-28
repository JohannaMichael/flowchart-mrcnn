# Flowchart Instance Segmentation
## with Mask R-CNN

This project aims to detect flowcharts symbols using a Mask Region Based Convolutional Neural Network.

A Mask R-CNN can not only detect objects
in an image, but also identify all specific pixels that belong to the objects. Below 
is an example in which all flowchart symbols where detected. The masks are shown in different colors
for identification. They also have bounding boxes with the identified labels of the symbol.

![Flowchart Symbol Recognition](/assets/flowchart_instance_segmentation_results_1.png)

The code that is used for the Neural Network is based on the Matterport Mask-RCNN
implementation, which you can find [here](https://github.com/matterport/Mask_RCNN).

This project still holds lots of room for improvement. Some of the main ones are:
1. Involving Text Recognition
2. More flowchart data + annotations. Until now I only fed 37 annotated flowcharts into the model.
More data might also raise the chance of detecting better/more flowlines (arrows), for example.

### Usage
1. The model that is used will be uploaded under releases in the future. It is called `mask_rcnn_flowchart.h5`.
You can set the path to the model in the file `inspect_flowchart_model.ipynb`.
2. Due to copyright issues I do not want to upload flowchart images that I used to train and test the model. 
But for testing the model, you can use any flowchart found online. Or for building ones own flowchart training dataset,
look into [this tutorial](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/).
I used the [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/) by the University of Oxford for flowchart symbol annotation.

### Installation

1. Clone this repository

2. Install dependencies (best to use Anaconda package manager for this)

`pip install -r requirements.txt`

Note: Use Tensorflow 1.14.0 or 1.15.0 (Version 2 of Tensorflow is not yet supported by the Matterport Mask R-CNN Code).
Best to use the GPU version of Tensorflow.

### For Training Model Only
3. Run setup from the repository root directory

`python setup.py install`

Look into `samples/flowchart/flowchart.py` to find the specific code and configuration for the project. The code
basically inherits from `mrcnn/`.

The code in `flowchart.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.


### Run Jupyter notebooks
Run jupyter notebook in the terminal.
Open the `inspect_flowchart_data.ipynb` or `inspect_flowchart_model.ipynb` Jupyter Notebooks. 
You can use these notebooks to explore the dataset and model.
