#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:22:48 2019

@author: j-bd
"""

class Database:
    '''class making able to create items and linked with sub-items'''

    def __init__(self, master_node):
        '''create a new object with the following base structure'''
        self.graph = [(master_node, None),]
        self.images = dict()

    def add_nodes(self, list_tup):
        '''create child node if parent node exist'''
        for child_node, parent_node in list_tup:
            if parent_node in [node[0] for node in self.graph]:
                self.graph.append((child_node, parent_node))
            else:
                print(
                    f"error, {parent_node} does not exist in current graph\n"\
                    f"Please, add {parent_node} as a child node and its parent.")

    def add_extract(self, dic):
        '''linked image with node(s)'''

    def get_extract_status(self, img):
        '''give the status of an image'''
