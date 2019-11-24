#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:17:05 2019

@author: j-bd
"""
import logging
import json
import ast

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

def data_loading():
    '''Load data from given files'''
    with open(constants.BUILD) as f_build:
        build = f_build.read()
        build = build.replace('null', 'None')
        build = ast.literal_eval(build)
    with open(constants.EDITS) as f_edits:
        edits = f_edits.read()
        edits = ast.literal_eval(edits)
    with open(constants.EXTRACT) as f_extract:
        extract = json.load(f_extract)
    with open(constants.EX_ANSWER) as f_answer:
        answer = json.load(f_answer)
    return build, edits, extract, answer

def proceed_food_data(build, extract, edits):
    '''Execute test'''
    if build:
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
    logging.info("Test 1")
    status = proceed_food_data(
        constants.BUILD_1, constants.EXTRACT_1, constants.EDITS_1
    )
    check_values(constants.EX_ANSWER_1, status)

    logging.info("Test 2")
    build, edits, extract, answer = data_loading()
    status = proceed_food_data(build, extract, edits)
    check_values(answer, status)

if __name__ == "__main__":
    main()
