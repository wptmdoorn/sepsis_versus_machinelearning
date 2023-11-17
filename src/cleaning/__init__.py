"""src/cleaning package

This package contains all src to process,
and clean data models. Refer to individual files
for detailed descriptions. 

Classes
-------
    process_database
        this script processes the database into a full processed database,
        please use export_data_models.py hereafter to derive functional
        data models
    export_data_models
        this script derives different data models from the full database and
        exports this to create actual functional data models for algorithm usage.

Files
-----
    convert_sav_to_csv.R
        this script, written in R, processes a SPSS (.sav) file and processes it
        to a CSV file
"""