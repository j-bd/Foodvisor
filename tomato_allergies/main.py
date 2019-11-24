#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:09:15 2019

@author: j-bd
"""
#import constants
import functions
import tomato_training
import detection


def main():
    '''Launch the mains steps'''
    args = functions.arguments_parser()
    functions.check_inputs(args)

    # Training selection
    if args.command in ["custom", "xception"]:
        # Setup data in a df
        input_df = functions.settle_data(args)
        # Setup of the futur modele
        model = tomato_training.TomatoTraining(input_df, args.image_resize)
        model.data_split(args.split_rate)
        model.data_preparation()

        # Launch training
        if args.command == "custom":
            model.custom_cnn(args.epochs_number)
        else:
            model.xception_cnn(args.epochs_number)

    # Visualization or detection selection
    else:
        model = detection.Detection(args.model_path)
        if args.command == "visualize_cam":
            model.visualize_cam(args.image_path)
        else:
            model.predict(args.image_path)

if __name__ == "__main__":
    main()
