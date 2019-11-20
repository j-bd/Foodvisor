#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:17:05 2019

@author: j-bd
"""
import logging

import database
import constants

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def check_values(solution, status):
    '''Check answers'''
    if len(solution) != len(status):
        logging.warning(
            f"Solution file have {len(solution)} values,"
            f"Status file have {len(status)} values.\n"
            "Please check pipeline"
        )
    for key in solution.keys():
        if solution[key] != status[key]:
            logging.warning(
                f" {key} solution is '{solution[key]}' and Status is '{status[key]}'"
            )
        else:
            logging.info(
                f"{key} solution is '{solution[key]}' and Status is '{status[key]}'"
            )

def proceed_food_data(build, extract, edits):
    '''Execute test'''
    if len(build) > 0:
        # Build graph
        db = database.Database(build[0][0])
        if len(build) > 1:
            db.add_nodes(build[1:])
        # Add extract
        db.add_extract(extract)
        # Graph edits
        db.add_nodes(edits)
        # Update status
        status = db.get_extract_status()
    return status

def main():
    '''Setup test'''
    # Test1
    logging.info("Test 1")
    status = proceed_food_data(
        constants.BUILD_1, constants.EXTRACT_1, constants.EDITS_1
    )
    check_values(constants.EX_ANSWER_1, status)

    # Test2
    logging.info("Test 2")
    status = proceed_food_data(
        constants.BUILD, constants.EXTRACT, constants.EDITS
    )
    check_values(constants.EX_ANSWER, status)

if __name__ == "__main__":
    main()
