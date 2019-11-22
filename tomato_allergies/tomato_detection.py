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

    def __init__(self, df, im_resize):
        '''Create a new object with the following base structure'''
        self.df = df
        self.im_resize = im_resize

    def data_split(self, random_state, test_size, stratify=False):
        '''Create set of df : xtrain, xtest, ytrain, ytest'''
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(
            self.df.iloc[:, :3], self.df.iloc[:,-1], random_state = 42,
            test_size = 0.2, stratify=self.df.iloc[:,-1]
        )

    def data_preparation(self):
        '''Prepare data to return a full (images + label) TensorFlow Dataset'''
        def load_and_preprocess_train_images(img):
            '''Preprocess the images and data augmentation for train'''
            img = tf.io.read_file(img)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [self.im_resize, self.im_resize])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_contrast(img, 0.50, 0.90)
            img = img / 255.0
            return img

        def load_and_preprocess_test_images(img):
            '''Preprocess the images and data augmentation for test'''
            img = tf.io.read_file(img)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [self.im_resize, self.im_resize])
            img = img / 255.0
            return img

        # Train images upload
        tf_train_set = tf.data.Dataset.from_tensor_slices(self.xtrain["path"].tolist())
        # Apply preprocessing to the dataset
        tf_train_set = tf_train_set.map(load_and_preprocess_train_images)
        # Labels train upload
        train_image_labels = self.ytrain.tolist()
        tf_train_labels = tf.data.Dataset.from_tensor_slices(train_image_labels)
        # Merge Images and corresponding labels in a full train dataset
        full_train_ds = tf.data.Dataset.zip((tf_train_set, tf_train_labels))
        # Split in 16 batch
        self.batch_train_ds = full_train_ds.batch(16)

        # Test images upload
        tf_test_set = tf.data.Dataset.from_tensor_slices(self.xtest["path"].tolist())
        # Apply preprocessing to the dataset
        tf_test_set = tf_test_set.map(load_and_preprocess_test_images)
        # Labels test upload
        test_image_labels = self.ytest.tolist()
        tf_test_labels = tf.data.Dataset.from_tensor_slices(test_image_labels)
        # Merge Images and corresponding labels in a full test dataset
        full_test_ds = tf.data.Dataset.zip((tf_test_set, tf_test_labels))
        # Split in 16 batch
        self.batch_test_ds = full_test_ds.batch(16)

    def display_loss_accuracy(self, list_hist, epochs, name):
        '''Display loss and accuracy informations in a graph'''
        loss, accuracy = list_hist.keys()
        plt.figure("Training history", figsize=(15.0, 5.0))
        plt.subplot(121)
        plt.plot(range(1, len(list_hist[loss]) + 1), list_hist[loss], label="loss function")
        plt.title("Loss function evolution")
        plt.legend()
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss value")
        plt.subplot(122)
        plt.plot(range(1, len(list_hist[accuracy]) + 1), list_hist[accuracy], label="accuracy")
        plt.title("Accuracy evolution")
        plt.legend()
        plt.xlabel("Number of iterations")
        plt.ylabel("Accuracy value")
        plt.show()
        plt.savefig(f"{name}-im_s{self.im_resize}-ep{epochs}-Training.png")

    def confusion_matrix(self, model, epochs, name):
        '''Display and save a confusion matrix heatmap'''
        y_true = []
        y_pred = []
        for batch, true_labels in self.batch_test_ds.take(-1):
          y_true += [true_labels.numpy()]
          y_pred += [model.predict_classes(batch)]
        y_true = tf.concat([batch for batch in y_true], axis=0).numpy()
        y_pred = tf.concat([batch for batch in y_pred], axis=0).numpy()
        cm = confusion_matrix(y_true, y_pred)
        plt.figure("Confusion matrix")
        sns.heatmap(cm, annot=True, fmt='d')
        plt.savefig(f"{name}-im_s{self.im_resize}-ep{epochs}-Confusion_matrix.png")

    def xception_cnn(self, epochs):
        '''Use a pre-trained model to realize detection'''
        def model(batch_ds, epochs):
            '''Xception model adaptation'''
            model = tf.keras.applications.xception.Xception(
                input_shape=(self.im_resize, self.im_resize, 3),
                include_top = False, weights = "imagenet"
            )
            model.trainable = False
            model = tf.keras.Sequential([
                model, tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2, activation="softmax")
            ])
            initial_learning_rate = 0.001

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps=1000, decay_rate=0.96,
                staircase=True
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics = [tf.keras.metrics.SparseCategoricalAccuracy()
            ])
            model.summary()
            hist = model.fit(batch_ds, epochs=epochs)
            return model, hist

        model, hist = model(self.batch_train_ds, epochs)
        self.display_loss_accuracy(hist.history, epochs, "xception_le_tr_model")

        model.evaluate(self.batch_test_ds)

        self.confusion_matrix(model, epochs, "xception_le_tr_model")

        model.save_weights(f"xception_le_tr_model-im_s{self.im_resize}-ep{epochs}.h5")


    def home_made_cnn(self, epochs):
        '''Use an home made CNN to detect tomatoes'''
        def model(batch_ds, epochs):
            '''Home made model creation'''
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[self.im_resize, self.im_resize, 3]),
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
            # Schedule learning rate creation
            initial_learning_rate = 0.0001
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=500,
                decay_rate=0.96,
                staircase=True)

            model.compile(optimizer = tf.keras.optimizers.Adam(lr_schedule),
                          loss= tf.keras.losses.binary_crossentropy,
                          metrics = [tf.keras.metrics.binary_accuracy])
            model.summary()
            hist = model.fit(batch_ds, epochs=epochs)
            return model, hist

#        def most_confused(model, threshold):
#            '''Display the most wrong prediction'''
#            for example, labels in self.batch_test_ds.take(-1):
#                y_pred = model.predict(example)
#                mae = np.abs(y_pred.squeeze() - labels.numpy().squeeze())
#                for i in np.where(mae>threshold)[0]:
#                    plt.figure()
#                    plt.title("prediction: {}\n MAE : {}".format(y_pred[i], mae[i]))
#                    plt.imshow(example[i])
#            plt.show()

        model, hist = model(self.batch_train_ds, epochs)
        self.display_loss_accuracy(hist.history, epochs, "hm_cnn_model")

        model.evaluate(self.batch_test_ds)

#        # Display the worst prediction
#        most_confused(model, 0.8)

        self.confusion_matrix(model, epochs, "hm_cnn_model")

        model.save_weights(f"hm_cnn_model-im_s{self.im_resize}-ep{epochs}.h5")
