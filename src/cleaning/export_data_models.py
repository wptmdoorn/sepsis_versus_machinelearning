"""src/cleaning/export_data_models.py
Last edited: 17/07/2020
Author: William van Doorn

This script contains the functionality to export so-called
'data models' from our total database. We define data models as 
a set of features extracted from the total database (e.g. exclusively
lab features or lab + clinical features). Individual methods should never
be called directly. Please see the --help syntax of this script on how
to export data models from the database.

Methods
-------
    generate_out_variable(df, out)
        method to modify the `df` variable to contain a new column containing the
        output (prediction) variable depending on the `out` variable
    process_1(df, out)
        method to modify the total database (`df`) to only contain features
        according to data model 1 (base + laboratory)
    process_2(df, out)
        method to modify the total database (`df`) to only contain features
        according to data model 2 (base + laboratory + history/drugs)
    process_3(df, out)
        method to modify the total database (`df`) to only contain features
        according to data model 3 (base + laboratory + history/drugs + clinical)
    main()
        the main function which processes the database, please never
        call directly and always run the script as a file (see --help)
    setup_arguments()
        method to setup the arguments when running this file
"""

# general imports
import os
import pandas as pd
import datetime as dt

# utils
from utils.cleaning import mortality
from utils.files import get_latest_file, get_latest_version, create_directory

# settings
from static.files import DATA_DIR

# static
from static.models import FEATURES, VALIDATION_IDS, VALIDATION_DB

# logging
import logging
import logging.config

logging.config.fileConfig("logging/logging.conf.txt")


def generate_out_variable(df, out):
    """Function to assign a out, predictor variable
    to the original database `df`. 

    Parameters
    ----------
        df : pd.DataFrame
            the 'total database' produced by process_database.py
        out : int
            value representing which value we want to add to `df`,
            1=shock, 2=in-house mortality, 3=one-month mortality
    
    Returns
    -------
    pd.DataFrame
        modified pd.DataFrame with a out-variable assigned according
        to the `out` value.
    
    """

    # Define the out vars
    _outvars = ['septischeshock',
                'Sterfte_in_ziekenhuis',
                'Datum_overlijden']

    # if out var is 1 or 2, 
    # we can directly process these
    if out == 1 or out == 2:
        # select y_var and drop all other variables
        df['y_var'] = df[_outvars[out - 1]]
        df = df.drop(columns=_outvars)

    elif out == 3:
        # if out is 3, meaning mortality
        # we need to process this using mortality (src/utils/cleaning) function
        # then also drop all other columns
        df['Datum_overlijden'] = pd.to_datetime(df['Datum_overlijden'], format='%Y-%m-%d', errors='coerce')
        df['y_var'] = df.apply(mortality, axis=1)
        df = df.drop(columns=_outvars)

    return df


def process_1(df, out):
    """Function to process `df` according to the
    features we defined in data model 1. Refer to source
    code documentation or handbook for detailed explanation
    on the different data models. 

    Parameters
    ----------
        df : pd.DataFrame
            the 'total database' produced by process_database.py
        out : int
            value representing which value we want to predict; 
            1=shock, 2=in-house mortality, 3=one-month mortality
    
    Returns
    -------
    pd.DataFrame
        modified pd.DataFrame object according to the features of
        data model 1
    
    """

    logging.info("Processing model 1 - lab values only")

    final_columns = ['y_var']
    # base values
    final_columns = final_columns + FEATURES[0]
    # all lab values
    final_columns = final_columns + list(df.columns[411:])

    # select values from dataframe
    df = generate_out_variable(df, out)
    df = df[final_columns]

    # convert geboortedatum to age
    df['Geboortedatum'] = (pd.to_datetime('today') -
                           pd.to_datetime(df['Geboortedatum'])) // dt.timedelta(days=365.2425)
    df['Uur_Binnenkomst_SEH'] = pd.to_datetime(df['Uur_Binnenkomst_SEH'])
    df['Uur_Binnenkomst_SEH'] = (df['Uur_Binnenkomst_SEH'].dt.hour + df['Uur_Binnenkomst_SEH'].dt.minute) / 60.0

    # TODO
    # we should think about not filling it with zero here,
    # maybe we do not need to fill it even? Otherwise we should
    # fill it with the same value we specified when processing
    # the database (process_database.py -nofill). XGBoost has the
    # missing value option, so we might not fill it!
    return df.fillna(0)


def process_2(df, out):
    """Function to process `df` according to the
    features we defined in data model 2. Refer to source
    code documentation or handbook for detailed explanation
    on the different data models. 

    Parameters
    ----------
        df : pd.DataFrame
            the 'total database' produced by process_database.py
        out : int
            value representing which value we want to predict; 
            1=shock, 2=in-house mortality, 3=one-month mortality
    
    Returns
    -------
    pd.DataFrame
        modified pd.DataFrame object according to the features of
        data model 2
    
    """

    logging.info("Processing model 2 - model 1 + history + drugs")
    logging.info("Obtaining model 1")

    # First, we want to obtain model 1 by chaining 
    # the df and out variables
    model_1 = process_1(df, out)

    logging.info("Obtaining additional information for model 2")
    # Select the features from FEATURES variable
    final_columns = FEATURES[1]

    # Obtain the columns from the original df
    df = df[final_columns]

    # Merge both models together
    logging.info("Merging model 1 ({}) and model 2 ({})".format(model_1.shape, df.shape))
    df = pd.concat([model_1.reset_index(drop=True),
                    df.reset_index(drop=True)], axis=1)
    logging.info("Final shape: {}".format(df.shape))

    # see comments process_1 function on this line (line 136-140)
    return df.fillna(0)


def process_3(df, out):
    """Function to process `df` according to the
    features we defined in data model 3. Refer to source
    code documentation or handbook for detailed explanation
    on the different data models. 

    Parameters
    ----------
        df : pd.DataFrame
            the 'total database' produced by process_database.py
        out : int
            value representing which value we want to predict; 
            1=shock, 2=in-house mortality, 3=one-month mortality
    
    Returns
    -------
    pd.DataFrame
        modified pd.DataFrame object according to the features of
        data model 3
    
    """

    logging.info("Processing model 2 - model 2 + drugs")
    logging.info("Obtaining model 2")

    # First, we want to obtain model 2 (and thus automatically 1) by chaining 
    # the df and out variables
    model_2 = process_2(df, out)

    logging.info("Obtaining additional information for model 3")

    # Select the features from FEATURES variable
    final_columns = FEATURES[2]

    # Obtain columns from original pd.DataFrame
    df = df[final_columns]

    # Simple processing of 'zuurstoftherapie' variable
    df['zuurstoftherapie'] = df['zuurstoftherapie'].str.split(" ").str.get(0).str.replace("L", "")
    df['zuurstoftherapie'].fillna(0)
    df['zuurstoftherapie'] = pd.to_numeric(df['zuurstoftherapie'], downcast='float', errors='coerce')

    # Merge models together
    logging.info("Merging model 2 ({}) and model 3 ({})".format(model_2.shape, df.shape))
    df = pd.concat([model_2.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    logging.info("Final shape: {}".format(df.shape))

    # see comments process_1 function on this line
    return df.fillna(0)


def main():
    """Main function which controls the total workflow to
    export data models from the main database.
    See documentation and script comments for detailed 
    explanation of the process.

    Parameters
    ----------
    N/A
    
    Returns
    -------
    N/A
    """

    # parse arguments
    args = setup_arguments().parse_args()

    # define the input file
    infile = args.infile or get_latest_file(DATA_DIR['PROCESSED'],
                                            extension='*.xlsx').split('/')[-1]

    # processing functions for each of the models
    funcs = [process_1, process_2, process_3]

    # reading input file
    logging.info("Reading file: {}".format(infile))
    df = pd.read_excel(os.path.join(DATA_DIR['PROCESSED'],
                                    infile))
    logging.info("Dataframe shape: {}".format(df.shape))

    # creating new output directory
    # based on last version + the date of today
    logging.info("Creating new output directory")
    last_version = get_latest_version(os.path.join(os.getcwd(),
                                                   '..',
                                                   'models'))

    # Obtain current date and create new directory
    current_date = dt.datetime.now().strftime('%d %B %y')
    new_dir = os.path.join(os.getcwd(), '..',
                           'models',
                           'v{0:03d} - {1}'.format(last_version + 1, current_date))
    create_directory(new_dir)

    # loop through the supplied models by the user
    for model in args.model:
        # generate the data model based on the function
        generated_model = funcs[model - 1](df.copy(), args.out)
        generated_model['Datum_SEH'] = pd.to_datetime(generated_model['Datum_SEH'].dt.date)

        # obtain the validation dataframe if necessary
        validation_pd = pd.DataFrame(VALIDATION_DB)
        validation_pd['Datum_SEH'] = pd.to_datetime(validation_pd['Datum_SEH'])

        # if we do not want to split
        if args.nosplit:
            # export the whole dataframe to a file
            # dropping the study number and Date columns
            logging.info("Exporting data frame directly to file")
            generated_model.drop(columns=['nummer', 'Datum_SEH'], inplace=True)
            generated_model.to_csv(os.path.join(new_dir,
                                                'model_{}_out_{}_fulldb.csv'.format(model,
                                                                                    args.out)),
                                   sep=',',
                                   encoding='utf-8',
                                   index=False)

        # else, if we do want to split (normal behaviour)
        else:
            # first; we merge the validation DB containing the study ID's
            # with the generated model so we only get the validation database
            logging.info("Generating training and validation databases")
            logging.info("Shapes before merging: {}".format(generated_model.shape))
            validation_df = pd.merge(validation_pd, generated_model,
                                     on=['nummer', 'Datum_SEH'], how='inner')

            # next we want to remove all validation datapoints out of the main database
            # we use a clever trick;
            # we first combine both databases, and then drop all instances where there are duplicates 
            # (these come from the validation database)
            main_df = pd.merge(generated_model, validation_pd, how='left', indicator=True) \
                .query("_merge == 'left_only'") \
                .drop('_merge', 1)

            # print shapes
            logging.info("Shapes after merging")
            logging.info("Main DB: {} - Validation DB: {}".format(main_df.shape, validation_df.shape))

            # remove the unnecessary columns
            main_df.drop(columns=['nummer', 'Datum_SEH'], inplace=True)
            validation_df.drop(columns=['nummer', 'Datum_SEH'], inplace=True)

            # export both dataframes
            main_df.to_csv(os.path.join(new_dir,
                                        'model_{}_out_{}_maindb.csv'.format(model, args.out)),
                           sep=',', encoding='utf-8', index=False)
            validation_df.to_csv(os.path.join(new_dir,
                                              'model_{}_out_{}_validationdb.csv'.format(model, args.out)),
                                 sep=',', encoding='utf-8', index=False)


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

    # in-filename
    ap.add_argument('-infile', metavar="O", required=False, type=str,
                    help='Specify which in-file to use. Default is to use latest in directory')

    # which models to export
    ap.add_argument("-model", metavar="N", required=True, choices=(1, 2, 3), nargs="+", type=int,
                    help="Specify which model to build.\n1=base (lab)\n2=model 1 + history + drugs\n3=model 2 + vital "
                         "function")

    # to split data or not
    ap.add_argument('-nosplit', required=False, action='store_true',
                    help='If we do not want to fill the dataframe with zeroes. Default: false.')

    # specify out model
    ap.add_argument("-out", metavar="O", required=True, choices=(1, 2, 3), type=int,
                    help="Specify which output variable to target.\n1=septic shock\n2=mortality ("
                         "in-house)\n3=mortality (1-month)")

    return ap


if __name__ == "__main__":
    main()
