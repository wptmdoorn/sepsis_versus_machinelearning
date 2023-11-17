# -*- coding: utf-8 -*-
"""
src/xgboost/clinical_validation.py
Created on: 25/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file is used to cross-validate the XGBoost model.
"""

# general imports
import os
import pandas as pd
import numpy as np

# utils import
from utils.files import get_latest_version
from utils.algorithms import get_hyper_params

# xgboost
from xgboost import XGBClassifier

# sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# logging
import logging
import logging.config
logging.config.fileConfig("logging/logging.conf.txt")


def create_model(data_dict: dict) -> XGBClassifier:
    """Creates a model from a data dict.

    Parameters
    ----------
    data_dict: dict
        the loaded data dict

    Returns
    -------
    XGBClassifier
        returns the newly developed XGBoost model
    """

    logging.info("Creating model")
    h_params = get_hyper_params('XGBOOST',
                                data_dict['model_version'],
                                data_dict['data_model'],
                                data_dict['output'])

    logging.info("Hyper parameters: {}".format(h_params))

    model = XGBClassifier(h_params)
    logging.info("Fitting model using main database")
    x_train, x_test, y_train, y_test = train_test_split(data_dict['x'],
                                                        data_dict['y'],
                                                        test_size=0.3,
                                                        random_state=42)

    eval_set = [(x_train, y_train), (x_test, y_test)]
    model.fit(x_train, y_train,
              eval_metric=["logloss", "error"],
              eval_set=eval_set,
              early_stopping_rounds=10,
              verbose=True)

    return model


def read_data(model: int = 1,
              out: int = 1,
              version: int = 1,
              which: str = 'main',
              scale: bool = False) -> dict:
    """Reads the data for a specific dataset model and a specific outcome.
    This will be used to compare to algorithms on.

    Parameters
    ----------
    model: int
        the data model to use (1 = laboratory, 3 = laboratory + clinical)
    out: int
        the outcome to use (1 = septic shock, 2 = in-hospital mort, 3 = 31-d mortality)
    version: int
        the version of the data file to use
    which: str
        specify which database to use; either 'main' or 'validation'
    scale: bool
        if True, will scale the X-values using `sklearn.preprocessing.StandardScaler`

    Returns
    -------
    dict
        returns a dict containing the x, y and features values
    """

    # Read data from file
    data_dict = {}
    model_dir = [x for x in os.listdir(os.path.join(os.getcwd(), '..',
                                                    'models')) if str(version).zfill(3) in x]
    if len(model_dir) > 1:
        logging.error("Found multiple versions with number: {}".format(version))
        logging.error("Please look into this!")
    else:
        model_dir = model_dir[0]
    
    model_file = "model_{}_out_{}_{}db.csv".format(model,
                                                   out,
                                                   which)
    
    logging.info("Reading data from file: {}".format(model_file))
    try:
        df = pd.read_csv(os.path.join(os.getcwd(), '..', 'models', model_dir, model_file),
                         sep=',',
                         header=0,
                         infer_datetime_format=True,
                         error_bad_lines=False,
                         engine='python',
                         encoding='utf-8')

    except Exception as e:
        logging.error("This version has no data files for output: {}".format(out))
        logging.error("Please try another version by using -version or use the right output.")
        import sys
        sys.exit()
    
    # Selecting outcome
    logging.info("Selecting Y-value")
    data_dict['y'] = df['y_var']
    
    # Selecting X-values
    logging.info("Selecting X-values")
    # drop all Y-values from data frame
    data_dict['x'] = df.drop(columns=['y_var'])
    data_dict['features'] = data_dict['x'].columns

    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_dict['x'] = scaler.fit_transform(data_dict['x'])
    else:
        data_dict['x'] = data_dict['x'].values

    return data_dict


def main() -> None:
    """Main function which performs clinical validation.

    Parameters
    ----------
    N/A

    Returns
    -------
    N/A
    """

    # parse arguments
    args = setup_arguments().parse_args()

    model_version = args.version or get_latest_version(os.path.join(os.getcwd(),
                                                                    'models'))
    
    # Loop through models
    for model_n in args.models:
        # Read the data
        data_dict = read_data(model=model_n,
                              out=args.out,
                              version=model_version,
                              which='main')

        data_dict['data_model'] = model_n
        data_dict['output'] = args.out
        data_dict['model_version'] = model_version
        model = create_model(data_dict)

        # Read the validation data
        data_dict_validation = read_data(model=model_n,
                                         out=args.out,
                                         version=model_version,
                                         which='validation')
        
        y_pred = np.where(model.predict_proba(data_dict_validation['x'],
                                              ntree_limit=model.best_ntree_limit)[:, 1] + 0.1 > 0.5, 1, 0)
        # Print AUC
        print(roc_auc_score(data_dict_validation['y'],
                            np.where(model.predict_proba(data_dict_validation['x'])[:, 1] > 0.25, 1, 0)))

        # Print second AUC
        print(roc_auc_score(data_dict_validation['y'], y_pred))


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
    ap.add_argument("-models", metavar="N", required=True, choices=(1, 2, 3, 4), nargs="+", type=int,
                    help="Specify which models to compare.\n1=base (lab)\n2=model 1 + history + drugs\n3=model 2 + "
                         "vital function")
    ap.add_argument("-out", metavar="O", required=True, choices=(1, 2, 3), type=int,
                    help="Specify which output variable to target.\n1=septic shock\n2=mortality ("
                         "in-house)\n3=mortality (1-month)")
    ap.add_argument("-verbose", metavar="V", required=False, type=int, default=0,
                    help="Output verbose yes (1) or no (0).")
    ap.add_argument("-version", metavar="V", required=False, type=int,
                    help="Specify which version of model to use for cross-validation.")
    ap.add_argument('-scale', dest='scale', action='store_true',
                    help="Scale data to {0,1}]")
    return ap

# main function
if __name__ == "__main__":
    main()
