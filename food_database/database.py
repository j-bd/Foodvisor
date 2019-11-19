#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:22:48 2019

@author: j-bd
"""
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

class Database:
    '''class making able to create items and linked with sub-items'''

    def __init__(self, root_node):
        '''create a new object with the following base structure'''
        self.graph = [(root_node, None),]
        self.im_extract = dict()

    def add_nodes(self, list_tup):
        '''create child node if parent node exist'''
        for child_id, parent_id in list_tup:
            if parent_id in [node[0] for node in self.graph]:
                self.graph.append((child_id, parent_id))
            else:
                logging.error(
                    f" '{parent_id}' does not exist in current graph\n"\
                    f"Please, add '{parent_id}' as a child id and its parent id before."
                    )

    def add_extract(self, dic):
        '''linked image with node(s)'''
        self.im_extract = {
            im_names : list_ids for im_names, list_ids in dic.items()
            }

    def get_extract_status(self, img):
        '''give the status of an image'''
