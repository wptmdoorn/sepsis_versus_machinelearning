"""src/settings.py
Last edited: 17/07/2020
Author: William van Doorn

This script contains the settings for all of 
the src we apply in this project. Most of them
are statically defined but part of them can be adjusted
by specifying them as argument in the desired script.
See the --help function for each individual script to 
adjust settings. Parameter adjusted settings will always
override those defined here. 

Variables
---------
    CLEANING : dict
        containing all setting-specific information for the cleaning/processing of the data.
    OPTIMIZE : dict
        containing all setting-specific parameters for the specific algorithms
"""

CLEANING = {
    # File name
    'FILENAME': '20190214_full_database_perc90try.csv',
    # Preprocessing directory
    'PREPROCESS': 'mumc',
    # Percentage of test codes to use
    'PERCENT': 0.80,
    # Maximum amount of grouping minutes
    'MAX_MIN': 120,
    # How much hours between presentations to exclude
    'PRESENTATION_RATE': 24,
    # Minimal amount of laboratory values
    'LAB_THRESHOLD': 3,
    # Dump testcodes, columns and/or means
    'DUMP_TESTCODES': False,
    'DUMP_COLUMNS': False,
    'DUMP_MEANS': False,
    # Mortality outcome
    'MORTALITY_DAYS': 31,
    # Impute method
    'IMPUTE_METHOD': 'complex_zero',
}

OPTIMIZE = {
    'XGBOOST': {
        'ITERATIONS': 50,
        'METRIC': 'roc_auc',
        'K_FOLD_N_SPLITS': 5,
    }
    
}

HYPERPARAMS = {
    'PAPER': {
        'max_depth': 13,
        'learning_rate': 0.1,
        'base_score': 0.5,
        'missing': 0,
    }
}
