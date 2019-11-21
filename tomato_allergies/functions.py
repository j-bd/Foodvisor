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
                {"path" : os.path.join(img_folder, key), "imgs" : key,
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
                    {"path" : os.path.join(img_folder, key), "imgs" : key,
                    "tomatoes" : label}, ignore_index=True
                )
    return df.drop_duplicates(subset="imgs")

def label_selection(df, column_name, column_label_id, target):
    '''Select label_id corresponding to target in to a pandas dataframe column'''
    label_selec_df = df[df[column_name].str.contains(target)]
    # Drop class containing "without tomato", to be improved by re exp
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

input_df = pd.DataFrame(columns=["path", "imgs", "tomatoes"])
input_df = collect_data(
    input_df, annot_db, list_tomato_label, constants.PATH_IMGS_FOLDER, 1
)

input_df = add_opposite_label_data(
    input_df, annot_db, input_df["imgs"].tolist(), len(input_df),
    constants.PATH_IMGS_FOLDER, 0
)

