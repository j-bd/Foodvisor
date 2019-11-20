#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:17:05 2019

@author: j-bd
"""
import database
import constants

def main():
    '''Execute test'''
    status = {}
    if len(constants.BUILD_1) > 0:
        # Build graph
        db = database.Database(constants.BUILD_1[0][0])
        if len(constants.BUILD_1) > 1:
            db.add_nodes(constants.BUILD_1[1:])
        # Add extract
        db.add_extract(constants.EXTRACT_1)
        # Graph edits
        db.add_nodes(constants.EDITS_1)
        # Update status
        status = db.get_extract_status()
    print(status)

if __name__ == "__main__":
    main()
