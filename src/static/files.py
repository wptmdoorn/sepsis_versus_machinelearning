# -*- coding: utf-8 -*-
"""
src/static/files.py
Created on: 27/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file contains all static variables regarding file-handling and output directories.
"""

import os

# DATA_DIR variable
# Contains all directories for data related things
DATA_DIR = {
    'PROCESSED': os.path.join(os.getcwd(), '..', 'data', 'processed'),
    'RAW': os.path.join(os.getcwd(), '..', 'data', 'raw'),
    'LAB': os.path.join(os.getcwd(), '..', 'data', 'lab')
}

# OUTPUT_DIR variable
# Contains all directories for output related things
OUTPUT_DIR = {
    'CROSS_VALIDATION': os.path.join(os.getcwd(), 'figures', 'crossvalidation'),
    'ALGO_COMPARISON': os.path.join(os.getcwd(), 'figures', 'algorithmcomparison'),
    'MODEL_EXPLANATION': os.path.join(os.getcwd(), 'figures', 'modelexplanation')
}

# ALGO_DIR
# Contains all directories of algorithms
ALGO_DIR = {
    'XGBOOST': os.path.join(os.getcwd(), 'algorithms', 'xgboost2')
}
