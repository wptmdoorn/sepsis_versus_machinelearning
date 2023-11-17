# -*- coding: utf-8 -*-
"""
src/explorative/compare_algorithms.py
Created on: 26/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file is used to compare different algorithms with each other.
"""

# general imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import string
import datetime as dt
from typing import List

# utils import
from src.utils.files import get_latest_version
from src.utils.cross_validation import cross_validate_ROC, compare_algorithms

# random forest, LR, light GBM, XGBOOST
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# logging
import logging
import logging.config

logging.config.fileConfig("src/logging/logging.conf.txt")


def _neuralnet(input_dimension: int) -> Sequential:
    """Generates a simple neural network for exploratory algorithm analysis.

    Parameters
    ----------
    input_dimension: int
        the input dimension of the neural network model


    Returns
    -------
    Sequential
        returns a keras sequential neural network model
    """

    # Create a model
    model = Sequential()
    model.add(Dense(int(0.8 * input_dimension),
                    kernel_initializer='uniform',
                    activation='sigmoid',
                    input_dim=input_dimension))

    model.add(Dense(1,
                    kernel_initializer='normal',
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])

    return model


def _get_algorithms(nn_input_dim: int) -> List[tuple]:
    """Generates a list with models to test in our exploratory analysis.
    Currently contains:
        - Neural Network
        - XGBoost
        - Logistic Regression
        - Random Forest
        - LightGBM

    Parameters
    ----------
    nn_input_dim: int
        the input dimension of the neural network model


    Returns
    -------
    List[tuple]
        returns a list with tuple containing a pair of (name of model, model class)
    """

    # Return list of classifiers
    return [('NN', KerasClassifier(build_fn=_neuralnet,
                                   epochs=10,
                                   batch_size=100,
                                   verbose=0,
                                   input_dimension=nn_input_dim)),
            ('XGB', XGBClassifier()),
            ('LR', LogisticRegression()),
            ('RF', RandomForestClassifier()),
            ('LGB', LGBMClassifier())]


def read_data(model: int = 1,
              out: int = 1,
              version: int = 1,
              scale: bool = False):
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
    model_dir = [x for x in os.listdir(os.path.join(os.getcwd(),
                                                    'models')) if str(version).zfill(3) in x]

    # Checking the amount of files
    if len(model_dir) > 1:
        logging.error("Found multiple versions with number: {}".format(version))
        logging.error("Please look into this!")
    else:
        model_dir = model_dir[0]

    # Define the model file
    model_file = "model_{}_out_{}_maindb.csv".format(model,
                                                     out)

    # Read the data
    logging.info("Reading data from file: {}".format(model_file))
    try:
        df = pd.read_csv(os.path.join(os.getcwd(), 'models', model_dir, model_file),
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

    # Select the outcome value
    logging.info("Selecting Y-value")
    data_dict['y'] = df['y_var']

    # Select the X-values
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

    # Return the data dictionary
    return data_dict


def main():
    """Main function which performs the algorithm
    comparison.

    Parameters
    ----------
    N/A

    Returns
    -------
    N/A
    """

    # Obtain command line arguments
    args = setup_arguments().parse_args()

    # Obtain the model version from the arguments
    model_version = args.version or get_latest_version(os.path.join(os.getcwd(),
                                                                    'models'))

    # Setup the plotting system
    i = 0
    fig, ax = plt.subplots(ncols=len(args.models), figsize=(18, 6))

    # Loop through the data models
    for model in args.models:
        list_of_algorithms = []
        # Read data
        data_dict = read_data(model=model,
                              out=args.out,
                              version=model_version,
                              scale=args.scale)

        # Get all algorithms which will be evaluated
        algorithms = _get_algorithms(data_dict['x'].shape[1])

        for name, algorithm in algorithms:
            logging.info("Algorithm: {}".format(name))
            list_of_algorithms.append([name,
                                       name,
                                       args.out,
                                       cross_validate_ROC((name,
                                                           algorithm,
                                                           data_dict['features']),
                                                          data_dict,
                                                          kfolds=5,
                                                          model_nr=model,
                                                          output_nr=args.out,
                                                          show=False)
                                       ])

        # Compare the algorithms on a high scale
        compare_algorithms(list_of_algorithms, ax=ax[i])
        i += 1

    # Plot the algorithms
    for n, ax in enumerate(ax.flat):
        ax.text(0.05, 0.92, string.ascii_uppercase[n + 2], transform=ax.transAxes,
                size=18, weight='bold', bbox=dict(facecolor='none', edgecolor='black', pad=3.0))

    # If we want to save
    if args.save:
        # Get last version and current date
        last_version = get_latest_version(os.path.join(os.getcwd(),
                                                       'figures',
                                                       'algorithmcomparison'))
        current_date = dt.datetime.now().strftime('%d %B %y')

        # Define file name and save figure
        file_name = 'v{0:03d} - {1}.svg'.format(last_version + 1, current_date)
        plt.savefig(os.path.join(os.getcwd(),
                                 'figures',
                                 'algorithmcomparison',
                                 file_name), format='svg', dpi=1200)

    # Show plot
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


# Main function
if __name__ == "__main__":
    main()
