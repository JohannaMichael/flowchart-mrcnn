# Flowchart2Code
## with Mask R-CNN

This project aims to detect and 'translate' flowcharts into simple pseudo code.
The approach to this task is as follows:

1. Detect, identify and create masks of all possible symbols and objects in a flowchart.
2. According to the given flowchart, extract the masks and sort them in the correct order.

Step 1. is achieved by a deep neural network (Mask R-CNN), which can not only detect objects
in an image, but also identify all specific pixels that belong to the objects. Below 
is an example of a flowchart where almost all flowchart symbols where detected. The masks are shown in different colors
for identification. The masks also have bounding boxes with the identified labels of the symbol.

![Flowchart Symbol Recognition](/assets/flowchart_symbols_recognition.PNG)

The code that is used for the Neural Network is based on the Matterport Mask-RCNN
implementation, which you can find [here](https://github.com/matterport/Mask_RCNN).

Step 2. is still partly in progress. Until now, the right order of the flowchart symbols is found by
starting with the start-symbol (ideally already found by the neural network) and then finding
other flowchart symbols that overlap or are nearest to the start symbol. 
This step is repeated for every symbol until the end of the flowchart is reached.
After the flowchart order is found, one can then translate these symbols into pseudo code.

Ideally, the correct order for the flowchart above would be: 
1.terminal_start 2.flowline 3.input 4.flowline 5.input 6.flowline
7.decision 8.flowline 9.flowline 10.input 11. flowline

This project still holds lots of room for improvement. Some of the main ones are:
1. Involving OCR when detecting the flowchart symbols. Text is often also used to determine the flow of the flowchart
(especially after a decision).
2. More flowchart data + annotations. Until now I only fed 20 annotated flowcharts into the model.
More data might also raise the chance of detecting better/more flowlines (arrows), for example.
3. Another neural network, which can sort the flowchart symbols into their correct order/flow.

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

Note: Use Tensorflow 1.14.0 (Version 2 of Tensorflow is not yet supported by the Matterport Mask R-CNN Code). Also
best to use GPU version of Tensorflow.

3. Run setup from the repository root directory

`python setup.py install`

### Flowchart Code

Look into `samples/flowchart/flowchart.py` to find the specific code and configuration for the project. The code
basically inherits from `mrcnn/`.

### Run Jupyter notebooks
Open the `inspect_flowchart_data.ipynb` or `inspect_flowchart_model.ipynb` Jupyter Notebooks. 
You can use these notebooks to explore the dataset and model.

The code in `flowchart.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.