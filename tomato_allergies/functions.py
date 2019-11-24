#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:21:03 2019

@author: j-bd
"""
import os
import json
import argparse

import pandas as pd

import constants


def arguments_parser():
    '''Get the informations from the operator'''
    parser = argparse.ArgumentParser(
        prog='Tomatoes detection',
        usage='%(prog)s [Foodvisor challenge]',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python main.py --command custom --label_path path/to/label.csv,
        --annotation_path path/to/images_annotation.json --image_folder
        path/to/images/folder --image_resize 300 --split_rate 0.2
        --epochs_number 20

        The following arguments are mandatory: --command (model used for training),
        --label_path (path to label file with .csv extension), --annotation_path
        (path to annotation file with .json extension), --image_folder (path to
        images folder used for training)
        The following arguments are optionnals: --image_resize (default value 300),
        --split_rate (split rate wich is the percent of trainning and validation
        images. It must be between 0.05 and 0.3. The default value is 0.2),
        --epochs_number (default value 20)


        To lauch Keras Xception training execution (transfert learning based on imagenet):
        -------------------------------------
        python main.py --command xception --label_path path/to/label.csv,
        --annotation_path path/to/images_annotation.json --image_folder
        path/to/images/folder --image_resize 300 --split_rate 0.2
        --epochs_number 20

        The following arguments are mandatory: --command (model used for training),
        --label_path (path to label file with .csv extension), --annotation_path
        (path to annotation file with .json extension), --image_folder (path to
        images folder used for training)
        The following arguments are optionnals: --image_resize (default value 300),
        --split_rate (split rate wich is the percent of trainning and validation
        images. It must be between 0.05 and 0.3. The default value is 0.2),
        --epochs_number (default value 20)


        To lauch activation map on image:
        -------------------------------------
        python main.py --command visualize_cam --model_path path/to/model.h5
        --image_path path/to/image.jpeg

        The following arguments are mandatory: --command (activation map process),
        --model_path (path to model file with .h5 extension), --image_path (path to
        image file with .jpeg extension)


        To lauch detection on image:
        -------------------------------------
        python main.py --command predict --model_path path/to/model.h5
        --image_path path/to/image.jpeg

        The following arguments are mandatory: --command (prediction on the image),
        --model_path (path to model file with .h5 extension), --image_path (path to
        image file with .jpeg extension)
        '''
    )
    parser.add_argument(
        "-cmd", "--command", required=True,
        help="choice between 'custom', 'xception', 'visualize_cam' and 'predict'"
    )
    parser.add_argument(
        "-im_r", "--image_resize", type=int, default=300,
        help="resize image"
    )
    parser.add_argument(
        "-sr", "--split_rate", type=float, default=0.2,
        help="split rate between train and validation dataset during training"
    )
    parser.add_argument(
        "-ep", "--epochs_number", type=int, default=20,
        help="Number of epochs during the training"
    )
    parser.add_argument(
        "-lf", "--label_path",
        help="Path to the label file used to list all classes"
    )
    parser.add_argument(
        "-af", "--annotation_path",
        help="Path to the annotation file used to describe object in each image"
    )
    parser.add_argument(
        "-if", "--image_folder",
        help="Path to the images folder used during training"
    )
    parser.add_argument(
        "-m", "--model_path",
        help="Path to the model file used to detect object"
    )
    parser.add_argument(
        "-i", "--image_path",
        help="Path to the image file used by the model"
    )
    args = parser.parse_args()
    return args

def check_inputs(args):
    '''Check if inputs are right'''
    if args.command not in ["custom", "xception", "visualize_cam", "predict"]:
        raise ValueError(
            "Your choice for '-c', '--command' must be either custom' or "
            "'xception' or 'visualize_cam' or 'predict'."
        )
    if args.command in ["custom", "xception"]:
        if not 0.05 <= args.split_rate <= 0.3:
            raise ValueError(
                f"Split rate must be between 0.05 and 0.3, currently {args.split_rate}"
            )
        if not os.path.isfile(args.label_path):
            raise FileNotFoundError(
                "Your choice for '-lf', '--label_path' is not a valide file."
            )
        if not os.path.isfile(args.annotation_path):
            raise FileNotFoundError(
                "Your choice for '-af', '--annotation_path' is not a valide file."
            )
        if not os.path.isdir(args.image_folder):
            raise FileNotFoundError(
                "Your choice for '-if', '--image_folder' is not a valide folder."
            )
    if args.command in ["visualize_cam", "predict"]:
        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(
                "Your choice for '-m', '--model_path' is not a valide file."
            )
        if not os.path.isfile(args.image_path):
            raise FileNotFoundError(
                "Your choice for '-i', '--image_path' is not a valide file."
            )

def add_opposite_label_data(
        df, dic_img_annot, exclusive_list, img_folder, label
    ):
    '''Add imgages and label from file annotation to a panda dataframe excluding
    specific data'''
    count = 0
    data_pos_nb = len(df)
    for key in dic_img_annot.keys():
        if (key not in exclusive_list and count < data_pos_nb):
            df = df.append(
                {"path" : os.path.join(img_folder, key), "img_names" : key,
                 "tomatoes" : label}, ignore_index=True
            )
            count += 1
    return df

def collect_data(df, dic_img_annot, list_label, img_folder, label):
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
    list_label = label_selec_df[column_label_id].tolist()
    return list_label

def settle_data(args):
    '''From a label file + images details file, return a full dataframe and
    xtrain, xtest, ytrain, ytest dataframe division'''
    label_df = pd.read_csv(args.label_path)
    with open(args.annotation_path) as annot:
        features_db = json.load(annot)

    list_tomato_label = label_selection(
        label_df, constants.LABEL_NAME_FR, constants.LABEL_ID, constants.TARGET
    )

    empty_df = pd.DataFrame(columns=["path", "img_names", "tomatoes"])
    tomatoes_df = collect_data(
        empty_df, features_db, list_tomato_label, args.image_folder, 1
    )

    input_df = add_opposite_label_data(
        tomatoes_df, features_db, tomatoes_df["img_names"].tolist(),
        args.image_folder, 0
    )
    return input_df
