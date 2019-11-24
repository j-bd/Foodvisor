

# Home assignment - Tomato allergies

You will find in this repository the implementation of tasks requested on this [page](https://github.com/Foodvisor/home-assignment).

Image Data can be downloaded by clicking [here](https://drive.google.com/file/d/1N7NSt8vZT20wslX00Fpeyx989QjK5GFZ/view?usp=sharing).

File Data can be downloaded by clicking [here](https://github.com/Foodvisor/home-assignment/releases/tag/v0.1.0).


## Overview of repository

This folder is organised as:
* ```main.py```: help to launch automatically training or detection tasks
* ```functions.py```: different functions helping for data setup
* ```tomato_training```: class offering all the necessary CNN elements training as: train and test data split, possibility to use __two CNN__ (a __custom one__ and __transfer learning based on Keras Xception imagenet__), model evaluation (loss and accuracy functions, confusion matrix)
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


## Few results:

Training results.

Custom CNN:
![Image of custom functions](https://raw.githubusercontent.com/j-bd/foodvisor/master/tomato_allergies/readme/custom_cnn-im_s300-ep25-training.png)
![Image of custom matrix](https://raw.githubusercontent.com/j-bd/foodvisor/master/tomato_allergies/readme/custom_cnn-im_s300-ep25-confusion_matrix.png)

Transfer Learning (Keras Xception based on imagenet):
![Image of xception functions](https://raw.githubusercontent.com/j-bd/foodvisor/master/tomato_allergies/readme/xception_le_tr-im_s300-ep60-training.png)
![Image of xception matrix](https://raw.githubusercontent.com/j-bd/foodvisor/master/tomato_allergies/readme/xception_le_tr-im_s300-ep60-confusion_matrix.png)


Predictions results:

Custom CNN:
![Image of custom pred](https://raw.githubusercontent.com/j-bd/foodvisor/master/tomato_allergies/readme/predict-custom.png)

Transfer Learning (Keras Xception based on imagenet):
![Image of xception pred](https://raw.githubusercontent.com/j-bd/foodvisor/master/tomato_allergies/readme/Transfer_le-predict.png)

Activation map based on the custom CNN:
![Image of cam](https://raw.githubusercontent.com/j-bd/foodvisor/master/tomato_allergies/readme/class_activation_maps.png)


## Discussion:

I took the decision to offer the possibility to train two differents CNN. Indeed, we can have our own private CNN that we want to customize during the R&D process. Also, for a fast first solution in case of commercial POC for instance, I offered the possibility to use transfer learning.

The next steps could be :
* Split tomato type (jus, sauce and full vegetable) to see if it improves detection
* Implement a GAN CNN in order to improve input data and results
* Displaying box detection
* Displaying mask
* Spread activation map execution to others models


## Citations

For activation map :
https://github.com/raghakot/keras-vis/blob/master/examples/resnet/attention.ipynb

```@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}```


