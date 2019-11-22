#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:07:31 2019

@author: j-bd
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class TomatoDetection:
    '''class to detect tomato in images'''

    def __init__(self, df):
        '''Create a new object with the following base structure'''
        self.df = df

    def data_split(self, random_state, test_size, stratify=False):
        '''Create set of df : xtrain, xtest, ytrain, ytest'''
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(
            self.df.iloc[:, :3], self.df.iloc[:,-1], random_state = 42,
            test_size = 0.2, stratify=self.df.iloc[:,-1]
        )

    def home_made_cnn(self, im_size, epochs):
        '''Use an home made CNN to detect tomatoes'''

        def load_and_preprocess_images(img):
            '''Preprocess the images and data augmentation for train'''
            img = tf.io.read_file(img)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [im_size, im_size])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_contrast(img, 0.50, 0.90)
            img = img / 255.0
            return img

        def load_valid_images(img):
            '''Preprocess the images and data augmentation for test'''
            img = tf.io.read_file(img)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [im_size, im_size])
            img = img / 255.0
            return img

        def model():
            # Création d'un model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[im_size, im_size, 3]),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dropout(0.05),
                tf.keras.layers.Dense(units=32, activation ="relu"),
                tf.keras.layers.Dense(units=16, activation ="relu"),
                tf.keras.layers.Dense(units=1, activation='sigmoid')
            ])
            # Création d'un schedule learning rate
            initial_learning_rate = 0.0001
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=500,
                decay_rate=0.96,
                staircase=True)
            # Création d'un compileur
            model.compile(optimizer = tf.keras.optimizers.Adam(lr_schedule),
                          loss= tf.keras.losses.binary_crossentropy,
                          metrics = [tf.keras.metrics.binary_accuracy])
            model.fit(batch_ds, epochs=epochs)
            return model

        def most_confused(full_valid_ds, threshold):
            '''Display the most wrong prediction'''
            for example, labels in full_valid_ds.take(-1):
                y_pred = model.predict(example)
                mae = np.abs(y_pred.squeeze() - labels.numpy().squeeze())
            for i in np.where(mae>threshold)[0]:
                plt.figure()
                plt.title("prediction: {}\n MAE : {}".format(y_pred[i], mae[i]))
                plt.imshow(example[i])
            plt.show()

        # Création d'un train set avec uniquement les images d'entrainement
        tf_train_set = tf.data.Dataset.from_tensor_slices(self.xtrain["path"].tolist())
        # Apply function to the dataset
        tf_train_set = tf_train_set.map(load_and_preprocess_images)
        #associer un label à chaque tenseur
        all_image_labels = self.ytrain.tolist()
        #insérez ces labels dans un tf.data.Dataset
        tf_labels = tf.data.Dataset.from_tensor_slices(all_image_labels)
        # Create a full dataset
        full_ds = tf.data.Dataset.zip((tf_train_set, tf_labels))
        # Split in 16 batch
        batch_ds = full_ds.batch(16)

        model = model()

        # Create a valid tf.data.Dataset
        tf_valid_set = tf.data.Dataset.from_tensor_slices(self.xtest["path"].tolist())
        # Preprocess images
        tf_valid_set = tf_valid_set.map(load_valid_images)
        #labels
        test_image_labels = self.ytest.tolist()
        #insérez ces labels dans un tf.data.Dataset
        tf_test_labels = tf.data.Dataset.from_tensor_slices(test_image_labels)
        # Create a full dataset
        full_valid_ds = tf.data.Dataset.zip((tf_valid_set, tf_test_labels))
        full_valid_ds = full_valid_ds.batch(16)

        #Evaluation du modele
        model.evaluate(full_valid_ds)

        # Display the worst prediction
        most_confused(full_valid_ds, 0.8)

        # Confusion Matrice
        y_true = []
        y_pred = []
        for batch, true_labels in full_valid_ds.take(-1):
          y_true += [true_labels.numpy()]
          y_pred += [model.predict_classes(batch)]
        y_true = tf.concat([batch for batch in y_true], axis=0).numpy()
        y_pred = tf.concat([batch for batch in y_pred], axis=0).numpy()
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
