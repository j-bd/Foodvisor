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
from sklearn.model_selection import train_test_split

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

def settle_data(label, images_features):
    '''From a file label + images details file, return a full dataframe and
    xtrain, xtest, ytrain, ytest dataframe division'''
    label_df = pd.read_csv(label)
    with open(images_features) as annot:
        features_db = json.load(annot)

    list_tomato_label = label_selection(
        label_df, constants.LABEL_NAME_FR, constants.LABEL_ID, constants.TARGET
    )

    empty_df = pd.DataFrame(columns=["path", "img_names", "tomatoes"])
    tomatoes_df = collect_data(
        empty_df, features_db, list_tomato_label, constants.PATH_IMGS_FOLDER, 1
    )

    input_df = add_opposite_label_data(
        tomatoes_df, features_db, tomatoes_df["img_names"].tolist(), len(tomatoes_df),
        constants.PATH_IMGS_FOLDER, 0
    )

    return input_df

#
### In main.py
#
## Open files
#
#
## In tomato_detection.py in class
## A verifier si necessaire
##import skimage.io
##for index, line in input_df.iterrows():
##    image = skimage.io.imread(line[0])
##    line[2] = image
#
#label_df = pd.read_csv(constants.PATH_LABEL)
#with open(constants.PATH_IMGS_ANNOT) as annot:
#    features_db = json.load(annot)
#
#list_tomato_label = label_selection(
#    label_df, constants.LABEL_NAME_FR, constants.LABEL_ID, constants.TARGET
#)
#
#empty_df = pd.DataFrame(columns=["path", "img_names", "tomatoes"])
#tomatoes_df = collect_data(
#    empty_df, features_db, list_tomato_label, constants.PATH_IMGS_FOLDER, 1
#)
#
#input_df = add_opposite_label_data(
#    tomatoes_df, features_db, tomatoes_df["img_names"].tolist(), len(tomatoes_df),
#    constants.PATH_IMGS_FOLDER, 0
#)
#
#xtrain, xtest, ytrain, ytest = train_test_split(
#    input_df.iloc[:, :3], input_df.iloc[:,-1], random_state = 42,
#    test_size = 0.2, stratify=input_df.iloc[:,-1]
#)
#
##from sklearn.model_selection import train_test_split
#xtrain, xtest, ytrain, ytest = train_test_split(
#    input_df.iloc[:, :3], input_df.iloc[:,-1], random_state = 42, test_size = 0.2,
#    stratify=input_df.iloc[:,-1]
#)
##
##xtrain = xtrain.reset_index(drop=True)
##ytrain = ytrain.reset_index(drop=True)
#
#
#
#
### In tomato_detection.py in class
#
## Preprocess the images and data augmentation
#def load_and_preprocess_images(img):
#    '''Preprocess the images plus data augmentation'''
#    img = tf.io.read_file(img)
#    img = tf.image.decode_jpeg(img, channels=3)
#    img = tf.image.resize(img, [300, 300])
#    img = tf.image.random_flip_left_right(img)
#    img = tf.image.random_contrast(img, 0.50, 0.90)
#    img = img / 255.0
#
#    return img
#
#
## Création d'un train set avec uniquement les images d'entrainement
#tf_train_set = tf.data.Dataset.from_tensor_slices(xtrain["path"].tolist())
#
## Apply function to the dataset
#tf_train_set = tf_train_set.map(load_and_preprocess_images)
## Get an example tensor
#for example_tensor in tf_train_set.take(1):
#    print(example_tensor)
#
#
##associer un label à chaque tenseur
#
##labels
#all_image_labels = ytrain.tolist()
##insérez ces labels dans un tf.data.Dataset
#tf_labels = tf.data.Dataset.from_tensor_slices(all_image_labels)
#for example in tf_labels.take(1):
#    print(example)
#
## fusionner les deux tf.data.Dataset
## Create a full dataset
#full_ds = tf.data.Dataset.zip((tf_train_set, tf_labels))
#
#
#
#for e in full_ds.take(1):
#    print(e)
#
#for example in full_ds.take(5):
#    plt.figure()
#    plt.title(example[1].numpy())
#    plt.imshow(example[0].numpy())
#
#plt.show()
#
#
##Maintenant, nous avons besoin d'effectuer un shuffle de notre dataset et de créer des batch d'images. Effectuez ceci en utilisant :
##    tf.data.Dataset.shuffle
##    tf.data.Dataset.batch
## Shuffle the dataset & create batchs
##full_ds = full_ds.shuffle(len(xtrain["path"].tolist())).batch(16)
##full_ds = full_ds.shuffle(2000).batch(16)
#full_ds = full_ds.shuffle(2000)
#
#for example in full_ds.take(5):
#    plt.figure()
#    plt.title(example[1].numpy())
#    plt.imshow(example[0].numpy())
#
#plt.show()
#
#batch_ds = full_ds.batch(16, drop_remainder=True)
#
#
#
## Test dizaine de batch et visualiser la première image de chaque
## Visualize some data
#for example_x, example_y in batch_ds.take(5):
#    plt.figure()
##    plt.title(example_y[0])
#    plt.imshow(example_x[0].numpy())
#
#plt.show()
#
#
#
#
#
#
#
## Création d'un model
#model = tf.keras.Sequential([
#    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[300, 300, 3]),
#    tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
#    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
#    tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
#    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
#    tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(units=64, activation='relu'),
#    tf.keras.layers.Dropout(0.05),
#    tf.keras.layers.Dense(units=32, activation ="relu"),
#    tf.keras.layers.Dense(units=16, activation ="relu"),
#    tf.keras.layers.Dense(units=1, activation='sigmoid')
#])
#
## Création d'un schedule learning rate
#initial_learning_rate = 0.0001
#
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate,
#    decay_steps=500,
#    decay_rate=0.96,
#    staircase=True)
#
#
## Création d'un compileur
#model.compile(optimizer = tf.keras.optimizers.Adam(lr_schedule),
#              loss= tf.keras.losses.binary_crossentropy,
#              metrics = [tf.keras.metrics.binary_accuracy])
#
#
#model.fit(batch_ds, epochs=30)
#
#
#
#
#
#
## Préparation des données validation
#
## Preprocess the images and data augmentation for test
#def load_valid_images(img):
#    '''Preprocess the images and data augmentation for test'''
#    img = tf.io.read_file(img)
#    img = tf.image.decode_jpeg(img, channels=3)
#    img = tf.image.resize(img, [300, 300])
#    img = img / 255.0
#
#    return img
#
#
## Create a valid tf.data.Dataset
#tf_valid_set = tf.data.Dataset.from_tensor_slices(xtest["path"].tolist())
#
## Preprocess images
#tf_valid_set = tf_valid_set.map(load_valid_images)
##labels
#test_image_labels = ytest.tolist()
##insérez ces labels dans un tf.data.Dataset
#tf_test_labels = tf.data.Dataset.from_tensor_slices(test_image_labels)
#
## Create a full dataset
#full_valid_ds = tf.data.Dataset.zip((tf_valid_set, tf_test_labels))
#full_valid_ds = full_valid_ds.shuffle(len(xtest)).batch(16)
#
#
##Evaluation du modele
#model.evaluate(full_valid_ds)
#
##Visualisation
#for example in full_valid_ds.take(1):
#  y_pred = model.predict(example)
#
#  for i in range(len(y_pred)):
#    plt.figure()
#    plt.title(y_pred[i])
#    plt.imshow(example[0][i])
#
#plt.show()
#
#
#
##savoir quelles sont les prédictions sur lesquelles le modèle s'est le plus trompé.
## Pour cela, nous allons définir une fonction most_confused qui bouclera sur tous
## les batchs de notre dataset de validation et sortira les MAE la plus élevée entre
## la prédiction et la valeur réelle.
##
##Pour rappel, la MAE (Mean Absolute Error) : np.abs(y_pred - y_true)
#
#def most_confused(full_valid_ds, threshold):
#  for example, labels in full_valid_ds.take(-1):
#    y_pred = model.predict(example)
#    mae = np.abs(y_pred.squeeze() - labels.numpy().squeeze())
#
#    for i in np.where(mae>threshold)[0]:
#      plt.figure()
#      plt.title("prediction: {}\n MAE : {}".format(y_pred[i], mae[i]))
#      plt.imshow(example[i])
#
#  plt.show()
#
#most_confused(full_valid_ds, 0.8)
#
#
#
##matrice de confusion pour voir où le modèle s'est trompé. Cela va se passer en
## trois étapes :
##
##    Créez deux listes vides : y_true & y_pred
##    Bouclez sur tout votre dataset de validation et ajoutez les prédictions de
## chaque batch dans y_pred et les valeurs réelles dans y_true
##    Concaténez via tf.concat() chacun items de y_pred et y_true pour avoir deux
## listes y_predet y_true de longueur 1000.
##    Insérez y_pred et y_true dans une matrice de confusion de sklearn.
#
#y_true = []
#y_pred = []
#
#for batch, true_labels in full_valid_ds.take(-1):
#  y_true += [true_labels.numpy()]
#  y_pred += [model.predict_classes(batch)]
#
#y_true = tf.concat([batch for batch in y_true], axis=0).numpy()
#y_pred = tf.concat([batch for batch in y_pred], axis=0).numpy()
#
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
#
#cm = confusion_matrix(y_true, y_pred)
#sns.heatmap(cm, annot=True, fmt='d')
