"""src package

This is the main package following all other packages. 
This directory solely consists of other directories,
except for the settings.py in which some of the settings
are defined.

Scripts
-------
    settings.py
        Contains general settings about all of the cleaning, processing and
        modelling.
        
Directories
-----------
    cleaning
        Package containing the cleaning src to go from a preprocessed database
        to a cleaned up database to be used for model training.
    xgboost2
        Package containing all the model src and functions to be employed using
        the XGBoost algorithm
    static
        Package containing all static data, variables and other relevant fields
        to be used in our src.
    utils
        Package containing all util files to be used with all our src.
    explorative
        Package containing all files to perform exploratory analysis.
"""
