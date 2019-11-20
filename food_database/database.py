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
        self.graph = {self.name : [None]}
        self.im_extract = list()

    def add_nodes(self, list_tup):
        '''edit graph with parents and children nodes'''
        for child_id, parent_id in list_tup:
            print(f"child_id {child_id}, parent_id {parent_id}")
            if parent_id in self.graph.keys():
                self.graph[child_id] = list()
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
        self.im_extract.append((dic, copy.deepcopy(self.graph)))

    def get_extract_status(self):
        '''give the status of each image in add_extract dictionary'''

        def invalid_label_test(list_labels, dic):
            '''check if the value exist in the dictionary'''
            for label in list_labels:
                if label not in dic.keys():
                    list_values = list(dic.values())
                    list_values = list_values[1:]
                    flatten_values = [
                        item for values in list_values for item in values
                    ]
                    if label not in flatten_values:
                        return True
            return False

        def coverage_test(list_labels, initial_graph, new_graph):
            '''Check if coverage staging has been created meanwhile'''
            for label in list_labels:
                for key, list_items in initial_graph.items():
                    if key == 'core':
                        pass
                    elif label in list_items:
                        if len(initial_graph[key]) != len(new_graph[key]):
                            return True
            return False

        def granularity_test(list_labels, initial_graph, new_graph):
            '''Check if granularity staging has been created meanwhile'''
            for label in list_labels:
                if label in initial_graph.keys():
                    if len(initial_graph[label]) != len(new_graph[label]):
                        return True
            return False

        status = dict()
        associate_graph = self.im_extract[0][1]
        for img_name, list_label in self.im_extract[0][0].items():
            if invalid_label_test(list_label, associate_graph):
                status[img_name] = "invalid"
            elif associate_graph == self.graph:
                status[img_name] = "valid"
            elif coverage_test(list_label, associate_graph, self.graph):
                status[img_name] = "coverage_staged"
            elif granularity_test(list_label, associate_graph, self.graph):
                status[img_name] = "granularity_staged"
            else:
                status[img_name] = "valid"
        return status
