

# Home assignment - Tomato allergies

You will find here the implementation of tasks requested on this [page](https://github.com/Foodvisor/home-assignment).


## Overview of repository

This folder is organised as:
* ```main.py```: help to launch automatically training or detection tasks
* ```functions.py```: different functions helping for data setup
* ```tomato_training```: class offering all the necessary elements CNN training as: train and test data split, possibility to use two CNN (a custom one and transfer learning based on imagenet), model evaluation (loss and accuracy functions, confusion matrix)
* ```detection.py```: class offering the possibility to apply detection with a previous model on an image. There is also the possibility to apply activation extraction module based on the custom model.
* ```constants.py```: columns names and Regex


## Installation

Libraries needed are list in the `requirements.txt` file. Please, be sure to have Python 3.x installed.


## Terminal commands

For all commands, please make sure to be located in the project folder. You can access to all details by using help command in your terminal: ```python3 main.py --help```

* To lauch custom training execution:
```python3 main.py --command custom --label_path path/to/label.csv --annotation_path path/to/images_annotation.json --image_folder path/to/images/folder --image_resize 300 --split_rate 0.2 --epochs_number 20```
* To lauch Keras Xception training execution (transfert learning based on imagenet):
```python3 main.py --command xception --label_path path/to/label.csv --annotation_path path/to/images_annotation.json --image_folder path/to/images/folder --image_resize 300 --split_rate 0.2 --epochs_number 20```
* To lauch activation map on image (only for the custom model):
```python3 main.py --command visualize_cam --model_path path/to/model.h5 --image_path path/to/image.jpeg```
* To lauch detection on image:
```python3 main.py --command predict --model_path path/to/model.h5--image_path path/to/image.jpeg```


## Elements saved:

During the training, theses elements are saved in the current folder:
* model.h5
* weights.h5
* training.png (loss and accuracy functions)
* confusion_matrix.png (confusion matrix)

For the activation map execution:
* class_activation_maps.png (activation map result)

For the prediction execution:
* predict.png (image with probability of tomato presence)

