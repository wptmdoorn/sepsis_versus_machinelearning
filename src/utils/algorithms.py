"""
src/utils/algorithms.py
Created on: 28/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file contains utils for algorithm development, building and development.
"""

# general imports
import os

# static, utils
from static.files import ALGO_DIR
from utils.files import get_latest_file, load_pickle


def _get_alg_dir(algo_type: str,
                 algo_version: int) -> str:
    """Generates a simple neural network for exploratory algorithm analysis.

    Parameters
    ----------
    algo_type: str
        the type of the algorithm (e.g. XGBoost)
    algo_version: int
        the version of the algorithm

    Returns
    -------
    str
        returns the directory of the algorithm
    """

    return [x for x in os.listdir(ALGO_DIR[algo_type]) if "{0:03d}".format(algo_version) in x][0]


def get_hyper_params(algo_type: str,
                     model_version: int,
                     data_model: int,
                     out: int,
                     algo_version: int = None) -> object:
    """Generates a simple neural network for exploratory algorithm analysis.

    Parameters
    ----------
    algo_type: str
        define the type of the algorithm
    algo_version: int or None
        define the version of the algorithm. if None, uses latest.
    model_version: int
        the model version
    data_model: int
        the data model (1 = laboratory, 3 = laboratory + clinical)
    out: int
        the outcome (1=septic shock, 2=in-hosp mort, 3= 31-d mort)

    Returns
    -------
    object
        returns a dictionary containing the hyper parameters
    """

    algo_dir = get_latest_file(os.path.join(ALGO_DIR[algo_type]),
                               extension="* m{}".format(model_version)) \
        if algo_version is None else _get_alg_dir(algo_type, algo_version)
    
    return load_pickle(os.path.join(algo_dir,
                                    'model_{}_out_{}.pkl'.format(data_model, out)))
