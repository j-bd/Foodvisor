#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:21:03 2019

@author: j-bd
"""
import os
import json

import pandas as pd

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