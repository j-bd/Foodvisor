#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:22:48 2019

@author: j-bd
"""
import logging
import copy

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

class Database:
    '''class making able to create items and linked with sub-items'''

    def __init__(self, root_node):
        '''create a new object with the following base structure'''
        self.name = root_node
        self.graph = {self.name : None}
        self.im_extract = list()

    def add_nodes(self, list_tup):
        '''edit graph with parents and childre nodes'''
        for child_id, parent_id in list_tup:
            if parent_id == self.name:
                self.graph[child_id] = list()
            elif parent_id in self.graph.keys():
                list_child = self.graph[parent_id]
                list_child.append(child_id)
                self.graph[parent_id] = list_child
            else:
                logging.error(
                    f" '{parent_id}' does not exist in current graph :\n"\
                    f"{self.graph}"\
                    f"Please, add '{parent_id}' as a child id and its parent id before."
                )

    def add_extract(self, dic):
        '''link image information with current graph'''
#        self.im_extract = {
#            im_names : list_ids for im_names, list_ids in dic.items()
#        }
        self.im_extract.append((dic, copy.deepcopy(self.graph)))

    def get_extract_status(self, img):
        '''give the status of an image'''
        #Comparison of current self.graph and graph existing at 'add_extract' creation

