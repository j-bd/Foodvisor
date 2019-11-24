#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:33:17 2019

@author: j-bd
"""

# Initial graph building adress
BUILD = "/home/latitude/Documents/foodvisor/food_database/data/graph_build_c.json"
# Images extractions adress
EXTRACT = "/home/latitude/Documents/foodvisor/food_database/data/img_extract.json"
# Graph edits adress
EDITS = "/home/latitude/Documents/foodvisor/food_database/data/graph_edits.json"
# Exercice solution adress
EX_ANSWER = "/home/latitude/Documents/foodvisor/food_database/data/expected_status.json"

# Initial graph
BUILD_1 = [("core", None), ("A", "core"), ("B", "core"), ("C", "core"), ("C1", "C")]
# Extract
EXTRACT_1 = {"img001": ["A", "B"], "img002": ["A", "C1"], "img003": ["B", "E"]}
# Graph edits
EDITS_1 = [("A1", "A"), ("A2", "A"), ("C2", "C")]
# Exercice solution
EX_ANSWER_1 = {"img001": "granularity_staged", "img002": "coverage_staged", "img003": "invalid"}
