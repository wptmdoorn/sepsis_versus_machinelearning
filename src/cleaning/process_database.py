"""src/cleaning/process_database.py
Created on: unknown
Last edited: 17/07/2020
Author: William van Doorn

This script contains the methods to process the
raw databases into a full database which can be used
for further derivatization of data models from the total database.
Please see the --help function of the script file to obtain
detailed description of processing options.

Methods
-------
    main()
        the main function which processes the database, please never
        call directly and always run the script as a file (see --help)
    setup_arguments()
        method to setup the arguments when running this file
"""

# general, data and datetime imports
import os
import numpy as np
import pandas as pd
import datetime as dt

# static and settings
from src.settings import CLEANING
from src.static.files import DATA_DIR

# utils
from src.utils.cleaning import assign_id, process_and_transform, correct_times
from src.utils.testcode import cleanup_testcode, encode_categories
from src.utils.impute import impute
from src.utils.files import get_latest_file

# logging
import logging
import logging.config
logging.config.fileConfig("logging/logging.conf.txt")


def main():
    """Main function which performs all the
    processing. See documentation and script comments
    for detailed explanation of the process.

    Parameters
    ----------
    N/A
    
    Returns
    -------
    N/A
    """
    
    # Our design is flawed somewhere, and therefore we get a warning that we work
    # on a copy of the pd.DataFrame. While this error is actually worth to notice,
    # we disable it for now as it does not affect current script. Nevertheless, we
    # should be aware that we work on a copy of the pd.DataFrame.
    pd.options.mode.chained_assignment = None 
    
    # parse arguments and define the output file
    logging.info('Starting processing database')
    logging.info('Processing command-line arguments')
    args = setup_arguments().parse_args()
    out_file = args.outfile or CLEANING['FILENAME']
    
    # reading 2015 and 2016 datasets
    logging.info("Reading 2015 and 2016 datasets")
    _df2015 = pd.read_csv(os.path.join(DATA_DIR['LAB'], '2015-2016 PEHU EEHH .TXT'), 
                          sep=';',
                          header=0,
                          error_bad_lines=False,
                          encoding='latin-1')
    _df2016 = pd.read_csv(os.path.join(DATA_DIR['LAB'], '2016-2017 PEHU EEHH .TXT'), 
                          sep=';',
                          header=0,
                          error_bad_lines=False,
                          encoding='latin-1')
    logging.info("Shapes of dataframes")
    logging.info("2015: {}, 2016: {}".format(_df2015.shape, _df2016.shape))
    
    # concatting into one dataset
    logging.info("Concatting into one dataset")
    df = pd.concat([_df2015, _df2016])
    logging.info("Dataset shape: {}".format(df.shape))
    logging.info("Dataset columns: {}".format(df.columns))
    
    # check for pilot experiment
    if args.pilot:
        logging.info('Running pilot experiment with 10K data points')
        df = df[1:10000]
        
    # Manually parse date times... it has been very bad
    logging.info('Processing dates')
    df['Datum_tijd'] = pd.to_datetime(df['Datum_tijd'],
                                      format='%d-%m-%Y %H:%M',
                                      errors='coerce')
    
    df['Geboortedatum'] = pd.to_datetime(df['Geboortedatum'],
                                         format='%d-%m-%Y',
                                         errors='coerce')
    
    df['Overleden'] = pd.to_datetime(df['Overleden'],
                                     format='%d-%m-%Y',
                                     errors='coerce')
    
    # Drop '^Unnamed' columns
    logging.info("Removing unnecessary columns")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Drop irrelevant columns
    for c in ['Naam', 'Geboortedatum', 'Partnernaam', 'Geslacht', 'Overleden', 
              'Artscode', 'Afdeling', 'Opmerkingrapport', 'Artsnaam', 'Labnummer']: 
        df.drop(c, 1, inplace=True)
        
    logging.info("Dataset shape: {}".format(df.shape))
    logging.info("Dataset columns: {}".format(df.columns))
    
    # Removing missings
    logging.info("Removing missings")
    df.dropna(subset=['Patientnummer'], inplace=True)
    logging.info("Dataset shape: {}".format(df.shape))
    
    # cleanup testcodes
    logging.info('Cleaning up testcodes - percentage top testcodes: {}'.format(args.perc or CLEANING['PERCENT']))
    df['Testcode'] = df['Testcode'].astype('category')
    df = cleanup_testcode(df, percentage=args.perc or CLEANING['PERCENT'])
    df.reset_index(inplace=True)
    
    # Correct all the times for presenting patients 
    logging.info("Correcting times")
    logging.info("Current unique times: {}".format(df['Datum_tijd'].nunique()))
    df['Datum_tijd'] = pd.to_datetime(df['Datum_tijd'],
                                      format='%d-%m-%Y %H:%M',
                                      errors='coerce')
    df = df.groupby(df['Patientnummer']).apply(correct_times, max_min=120)
    df = df[df['Datum_tijd'] != np.datetime64('2000-12-08T19:00:00.000000000')]
    logging.info("Processed unique times: {}".format(df['Datum_tijd'].nunique()))
    
    # assign ID
    logging.info("Assigning ID")
    df = assign_id(df)
    
    # Process and transform data ready for Neural Network
    logging.info("Processing and transforming network")
    df = process_and_transform(df, args.debug)
    
    # Now we're left with the string results (keep them for now)
    for d in df.columns:
        _colmean = pd.to_numeric(df[d], errors='coerce').mean()
        df[d] = df[d].replace('STRING', _colmean)
        
    # impute with zeros and binary 
    logging.info('Imputing data with impute: {}'.format(args.impute or CLEANING['IMPUTE_METHOD']))
    df = impute(df, impute_meth=args.impute or CLEANING['IMPUTE_METHOD'])
    
    # encode categorical variables
    logging.info('Encoding categories')
    df = encode_categories(df)
    
    # fill zeroes
    if not args.nofill:
        logging.info('Filling NAs with zeros')
        df = df.fillna(0)
    else:
        logging.info('Not filling NAs')
    
    # add Num_vars variable
    df['Num_vars'] = df.reset_index().drop(columns=['UniekLabnummer', 'Patientnummer',
                                                    'Datum_SEH']).apply(func=lambda r: sum(r.values > 0), axis=1)
    
    # Now we want to map the all sepsis DB patients to it
    logging.info("Selecting relevant patients from sepsis database")
    logging.info("Dataset shape: {}".format(df.shape))
    _infile = args.infile or get_latest_file(DATA_DIR['RAW']).split('/')[-1]
    sepsis_db = pd.read_csv(os.path.join(DATA_DIR['RAW'], _infile),
                            sep=',',
                            encoding='utf-8')
    sepsis_db['Uur_Binnenkomst_SEH'] = sepsis_db['Uur_Binnenkomst_SEH'].fillna('07:00:00')
    sepsis_db['Datum_SEH'] = pd.to_datetime(sepsis_db['Datum_SEH'] + ' ' + sepsis_db['Uur_Binnenkomst_SEH'],
                                            format="%Y-%m-%d %H:%M")

    # print some data
    logging.info("Sepsis db, dates: {}".format(sepsis_db['Datum_SEH'].head(5)))
    logging.info("Main db, dates: {}".format(df['Datum_SEH'].head(5)))
    logging.info("Sepsis db, nrs: {}".format(sepsis_db['nummer'].head(5)))
    logging.info("Main db, nrs: {}".format(df['Patientnummer'].head(5)))

    # Create new DataFrame
    new_df = pd.DataFrame(columns=list(sepsis_db.columns) + [x for x in df.columns if x not in sepsis_db.columns])

    # this loop is through the sepsis DB, finding the lab values of the specific patient
    # and then combining this into one specific dataframe
    for index, row in sepsis_db.iterrows():
        # maximum 4hours after SEH presentation the lab values should be ordered
        max_time = row['Datum_SEH'] + dt.timedelta(minutes=900)

        # select the lab values which are requested with same patient ID ('nummer') and which are within the time frame
        subset_lab = df[df['Patientnummer'] == row['nummer']]
        subset_lab = subset_lab[(subset_lab['Datum_SEH'] < max_time) & (row['Datum_SEH'] < subset_lab['Datum_SEH'])]

        # drop irrelevant columns
        subset_lab = subset_lab.drop(columns=[x for x in subset_lab.columns if x in sepsis_db.columns])

        # add in back later, just for test purposes now (12.2 07:48)
        if args.debug:
            logging.info("Patient nr: {}".format(row['nummer']))
            logging.info("Patient time: {}, max time: {}".format(row['Datum_SEH'], max_time))
            logging.info("Found lab times shape: {}".format(subset_lab.shape))

        # if shape[0] (rows) is actually 1 (meaning only 1 instance of lab values was found)
        # we can combine this with the sepsis DB and add it to a new dataframe
        if subset_lab.shape[0] == 1:
            # create new dataframe
            app = pd.DataFrame(pd.concat([row, subset_lab.iloc[0]])).T
            if args.debug:
                logging.info("Shape[0] is 1, so appending: {}".format(app.shape))
            # add it to the big dataframe
            new_df = new_df.append(app)

        else:
            logging.info("Did not find this patient;")
            logging.info("Patient nr: {}".format(row['nummer']))
            logging.info("Patient time: {}, max time: {}".format(row['Datum_SEH'], max_time))
            logging.info("Found lab times shape: {}".format(subset_lab.shape))

    # print final shape and export to XLSX file
    logging.info("Dataset shape after selection: {}".format(new_df.shape))
    writer = pd.ExcelWriter(os.path.join(DATA_DIR['PROCESSED'],
                                         out_file))
    new_df.to_excel(writer, 'Final')
    writer.save()
    logging.info('Saved final database: {}'.format(out_file))


def setup_arguments():
    """Function to setup the arguments which can 
    be specified by running this file. See the source code
    documentation for detailed overview or use --help command-line
    argument on this file.

    Parameters
    ----------
    N/A
    
    Returns
    -------
    ArgumentParser
        instanced ArgumentParser object with all arguments setup
    
    """
    
    import argparse
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    # percentage
    ap.add_argument("-perc",
                    metavar="T",
                    required=False,
                    type=float,
                    help="Percentage (0-1) of top testcodes to filter for.")
    
    # out-filename
    ap.add_argument('-outfile',
                    metavar="O",
                    required=False,
                    type=str,
                    help='Specify file name.')
    
    # in-filename
    ap.add_argument('-infile',
                    metavar="O",
                    required=False,
                    type=str,
                    help='Specify which raw-file to use. Default is to use latest in directory')
    
    # impute method
    ap.add_argument('-impute',
                    metavar="I",
                    required=False,
                    type=str,
                    help='Specify which imputing method to use. Default is complex_zero.')
    
    # args minute
    ap.add_argument('-mins',
                    metavar="M",
                    required=False,
                    type=int,
                    help='Specify in which timeframe to take the data. Default is 120 minutes.')
    
    # args minute
    ap.add_argument('-pilot',
                    required=False,
                    action='store_true',
                    help='Run a small pilot experiment of 100.000.')
    
    # args nofill
    ap.add_argument('-nofill',
                    required=False,
                    action='store_true',
                    help='If we do not want to fill the dataframe with zeroes. Default: false.')
    
    # args debug
    ap.add_argument('-debug',
                    required=False,
                    action='store_true',
                    help='If we want to debug parts of our src. Default: false.')

    return ap


if __name__ == '__main__':
    main()
