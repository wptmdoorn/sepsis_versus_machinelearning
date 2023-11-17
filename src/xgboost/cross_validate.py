# -*- coding: utf-8 -*-
"""
xgboost/cross_validate.py
Created on: 25/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file is used to cross-validate the XGBoost model.
"""

# general imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

# utils import
from utils.files import get_latest_version
from utils.cross_validation import cross_validate_ROC, cross_validate_risk_ROC, compare_models
from utils.risk_scores import read_data_risk_score

# xgboost2
from xgboost import XGBClassifier

# logging
import logging
import logging.config
logging.config.fileConfig("logging/logging.conf.txt")


def read_data(model: int = 1,
              out: int = 1,
              version: int = 1,
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
    
    model_file = "model_{}_out_{}_maindb.csv".format(model,
                                                     out)
    
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
    
    logging.info("Selecting Y-value")
    data_dict['y'] = df['y_var']
    
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


def main():
    """Main function which performs the K-Fold cross validation.

    Parameters
    ----------
    N/A

    Returns
    -------
    N/A
    """

    # parse arguments
    args = setup_arguments().parse_args()
    modelslist = []
    scoreslist = []

    model_version = args.version or get_latest_version(os.path.join(os.getcwd(), '..',
                                                                    'models'))
    i = 0
    fig, ax = plt.subplots(ncols=3, figsize=(18, 6))
    
    for model in args.models:
        data_dict = read_data(model=model,
                              out=args.out,
                              version=model_version,
                              scale=args.scale)
        
        modelslist.append(['XGBoost', model, args.out, 
                           cross_validate_ROC(('XGBoost',
                                               XGBClassifier(),
                                               data_dict['features']),
                                              data_dict,
                                              kfolds=5,
                                              model_nr = model,
                                              output_nr = args.out,
                                              ax=ax[i])])
        
        i += 1
        
    if args.rscore:
        data_dict = read_data_risk_score(version=19, score='CURB')
        scoreslist.append(['CURB', cross_validate_risk_ROC('CURB', data_dict, show=False)])
        data_dict = read_data_risk_score(version=19, score='MEDS')
        scoreslist.append(['MEDS', cross_validate_risk_ROC('MEDS', data_dict, show=False)])

    compare_models(modelslist,
                   scoreslist,
                   risk_score=args.rscore, ax=ax[i])

    if args.save:
        last_version = get_latest_version(os.path.join(os.getcwd(),
                                                       '..',
                                                       'figures',
                                                       'crossvalidation'))
        current_date = dt.datetime.now().strftime('%d %B %y')
        file_name = 'v{0:03d} - {1}.svg'.format(last_version + 1, current_date)
        print(file_name)
        plt.savefig(os.path.join(os.getcwd(), '..',
                                 'figures',
                                 'crossvalidation',
                                 file_name), format='svg', dpi=1200)

    plt.show()


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
    ap.add_argument('-save', dest='save', action='store_true',
                    help='Save figures to .SVG files')
    ap.add_argument('-riskscores', dest='rscore', action='store_true',
                    help='Include risk scores')
    return ap


# main function
if __name__ == "__main__":
    main()
