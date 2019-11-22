#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:09:15 2019

@author: j-bd
"""
import constants
import functions
import tomato_detection


def main():
    '''Launch the mains steps'''
    # Setup data in a df
    input_df = functions.settle_data(
        constants.PATH_LABEL, constants.PATH_IMGS_ANNOT
    )

    # Use an home made CNN to detect tomatoes
    hm_cnn = tomato_detection.TomatoDetection(input_df)
    hm_cnn.data_split(42, 0.2, hm_cnn.df.iloc[:,-1])
    hm_cnn.home_made_cnn(100, 2)



if __name__ == "__main__":
    main()
