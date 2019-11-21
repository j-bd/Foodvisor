#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:21:03 2019

@author: j-bd
"""
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import constants


def add_opposite_label_data(df, dic_img_annot, exclusive_list, data_nb, img_folder, label):
    '''Add imgages and label from file annotation to a panda dataframe excluding
    specific data'''
    count = 0
    for key in dic_img_annot.keys():
        if (key not in exclusive_list and count < data_nb):
            df = df.append(
                {"path" : os.path.join(img_folder, key), "img_names" : key,
                "tomatoes" : label}, ignore_index=True
            )
            count += 1
    return df

def collect_data(df,dic_img_annot, list_label, img_folder, label):
    '''Add imgages and label from file annotation to a panda dataframe if
    corresponding to a pre-determinated label contains in a list'''
    for key in dic_img_annot.keys():
        elements = len(dic_img_annot[key]) - 1
        for element in range(elements):
            if dic_img_annot[key][element]['id'] in list_label:
                df = df.append(
                    {"path" : os.path.join(img_folder, key), "img_names" : key,
                    "tomatoes" : label}, ignore_index=True
                )
    return df.drop_duplicates(subset="img_names")

def label_selection(df, column_name, column_label_id, target):
    '''Select label_id corresponding to target in to a pandas dataframe column'''
    label_selec_df = df[df[column_name].str.contains(target)]
    #!!! Drop class containing "without tomato", to be improved by re exp
    label_selec_df = label_selec_df.drop(index = 639)
    list_label = label_selec_df[column_label_id].tolist()
    return list_label




## In main.py

# Open files
label_df = pd.read_csv(constants.PATH_LABEL)
with open(constants.PATH_IMGS_ANNOT) as annot:
    annot_db = json.load(annot)

list_tomato_label = label_selection(
    label_df, constants.LABEL_NAME_FR, constants.LABEL_ID, constants.TARGET
)

input_df = pd.DataFrame(columns=["path", "img_names", "tomatoes"])
input_df = collect_data(
    input_df, annot_db, list_tomato_label, constants.PATH_IMGS_FOLDER, 1
)

input_df = add_opposite_label_data(
    input_df, annot_db, input_df["img_names"].tolist(), len(input_df),
    constants.PATH_IMGS_FOLDER, 0
)

# In tomato_detection.py in class
# A verifier si necessaire
#import skimage.io
#for index, line in input_df.iterrows():
#    image = skimage.io.imread(line[0])
#    line[2] = image



from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    input_df.iloc[:, :3], input_df.iloc[:,-1], random_state = 42, test_size = 0.2,
    stratify=input_df.iloc[:,-1]
)

xtrain = xtrain.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)
















## In tomato_detection.py in class

#import skimage.io
#for index, line in input_df.iterrows():
#    image = skimage.io.imread(line[0])
#    line[2] = image


# Création d'un train set avec uniquement les images d'entrainement
tf_train_set = tf.data.Dataset.from_tensor_slices(xtrain["path"].tolist())

# Preprocess the images and data augmentation
def load_and_preprocess_images(img):
  img = tf.io.read_file(img)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, [192, 192])
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_contrast(img, 0.50, 0.90)
  img = img / 255.0

  return img

# Apply function to the dataset
tf_train_set = tf_train_set.map(load_and_preprocess_images)
# Get an example tensor
for example_tensor in tf_train_set.take(1):
    print(example_tensor[2])


#associer un label à chaque tenseur

#labels
all_image_labels = ytrain.tolist()
#insérez ces labels dans un tf.data.Dataset
tf_labels = tf.data.Dataset.from_tensor_slices(all_image_labels)
for example in tf_labels.take(1):
      print(example)

# fusionner les deux tf.data.Dataset
# Create a full dataset
full_ds = tf.data.Dataset.zip((tf_train_set, tf_labels))

for example in full_ds.take(1):
  print(example)

for example in full_ds.take(20):
    plt.figure()
    plt.title(example[1].numpy())
    plt.imshow(example[0].numpy())

plt.show()

for e in full_ds.take(1):
  print(e)
#Maintenant, nous avons besoin d'effectuer un shuffle de notre dataset et de créer des batch d'images. Effectuez ceci en utilisant :
#    tf.data.Dataset.shuffle
#    tf.data.Dataset.batch
# Shuffle the dataset & create batchs
full_ds = full_ds.shuffle(len(input_df["path"].tolist())).batch(16)

# Test dizaine de batch et visualiser la première image de chaque
# Visualize some data
for example_x, example_y in full_ds.take(1):
    plt.figure()
    plt.title(example_y[0])
    plt.imshow(example_x[0].numpy())

plt.show()
