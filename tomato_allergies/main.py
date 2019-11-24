#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:09:15 2019

@author: j-bd
"""
#import constants
import functions
import tomato_detection
import activation_map
#
#
## Setup data in a df
#input_df = functions.settle_data(
#    constants.PATH_LABEL, constants.PATH_IMGS_ANNOT
#)
#
## Setup of the futur modele
#model = tomato_detection.TomatoDetection(input_df, 300)
#model.data_split(42, 0.2, model.df.iloc[:,-1])
#model.data_preparation()
#
## Choice of modele type
#model.xception_cnn(60)
#
#model.custom_cnn(25)
#
#
#model_path = "/home/latitude/Documents/foodvisor/tomato_allergies/readme/custom_cnn-model-im_s300-ep14.h5"
#visualisation = activation_map.Cam(model_path)
#img_path = "/home/latitude/Documents/foodvisor/tomato_allergies/data/assignment_imgs/0f92168eaab8fd44a02b74ad0f0972a8.jpeg"
##img_path = "/home/latitude/Documents/foodvisor/tomato_allergies/data/assignment_imgs/4dda082e4a1d820f7cc32f5cd9dc79be.jpeg"
#img, cam = visualisation.visualize_cam(img_path)
#
##0f92168eaab8fd44a02b74ad0f0972a8.jpeg

def main():
    '''Launch the mains steps'''
    args = functions.arguments_parser()
    functions.check_inputs(args)

    # Training selection
    if args.command in ["custom", "xception"]:
        # Setup data in a df
        input_df = functions.settle_data(args)
        # Setup of the futur modele
        model = tomato_detection.TomatoDetection(input_df, args.image_resize)
        model.data_split(42, args.split_rate, model.df.iloc[:,-1])
        model.data_preparation()

        # Launch training
        if args.command == "custom":
            model.custom_cnn(args.epochs_number)
        else:
            model.xception_cnn(args.epochs_number)

    # Visualization or detection selection
    else:
        if args.command == "visualize_cam":
            visualisation = activation_map.Cam(args.model_path)
            visualisation.visualize_cam(args.image_path)

if __name__ == "__main__":
    main()
