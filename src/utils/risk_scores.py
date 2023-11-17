# -*- coding: utf-8 -*-
"""
src/utils/risk_scores.py
Created on: 26/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file is used to generate risk-scores for the data frames.
"""

# general imports
import os
import pandas as pd

# logging
import logging
import logging.config
logging.config.fileConfig("logging/logging.conf.txt")


def _meds(row: pd.Series) -> int:
    """Returns the abbMEDS risk score.

    Parameters
    ----------
    row: pd.Series
        the individual row

    Returns
    -------
    int
        the abbMEDS score
    """

    points = 3
    
    if row['Geboortedatum'] >= 65:
        points += 3
    if row['Woonsituatie'] in [2, 3]:
        points += 2
    if row['Comorb_dementie_Psychiatrisch']:
        points += 2
    #if row['Definitieve_Focus'] == 2:
    #    points += 2
    if row['TRC'] < 150:
        points += 3
    if row['septischeshock'] == 1:
        points += 3
    if row['AF_berekend'] > 30:
        points += 3
    if row['metastase_of_chronische_ziekte_met_hoge_mortaliteit'] == 1:
        points += 6
        
    return points


def _rems(row: pd.Series) -> int:
    """Returns the mREMS risk score.

    Parameters
    ----------
    row: pd.Series
        the individual row

    Returns
    -------
    int
        the mREMS score
    """

    return 0
        

def _curb(row: pd.Series) -> int:
    """Returns the CURB-65 risk score.

    Parameters
    ----------
    row: pd.Series
        the individual row

    Returns
    -------
    int
        the CURB-65 score
    """

    points = 0

    if row['Comorb_dementie_Psychiatrisch'] > 0:
        points += 1
    if row['URESE'] > 7:
        points += 1
    if row['AF_berekend'] > 30:
        points += 1
    if row['Geboortedatum'] >= 65:
        points += 1
    if 60 > row['RR_diast'] > 0:
        points += 1
    elif 90 > row['RR_syst'] > 0:
        points += 1
    
    return points


def _calculate_risk_score(df: pd.DataFrame, score: str) -> list:
    """Calculates the clinical RISK scores.

    Parameters
    ----------
    df: pd.DataFrame
        the DataFrame to calculate RISK scores for
    score: str
        the score, should be either 'CURB', 'REMS' or 'MEDS'

    Returns
    -------
    list
        the list of scores for the pd.DataFrame
    """

    _scores = ['CURB', 'REMS', 'MEDS']
    _functions = [_curb, _rems, _meds]
    
    if score not in _scores:
        logging.error('Score not found, please check score!')
        
    # Calculate raw risk scores
    _rawscore = df.apply(_functions[_scores.index(score)], axis=1)
    _scores = ['CURB', 'REMS', 'MEDS']
    _risks = [{0: 0.015, 1: 0.015, 2: 0.092, 3: 0.22, 4: 0.22, 5: 0.22},
              {},
              {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01,
               5: 0.032, 6: 0.032, 7: 0.032,
               8: 0.080, 9: 0.080, 10: 0.080, 11: 0.080,
               12: 0.18, 13: 0.18, 14: 0.18, 15: 0.18,
               16: 0.42, 17: 0.42, 18: 0.42, 19: 0.42, 20: 0.42, 21: 0.42,
               22: 0.42, 23: 0.42, 24: 0.42, 25: 0.42, 26: 0.42,
               }]
    
    # Convert to mortality risks
    return _rawscore.replace(_risks[_scores.index(score)])


def read_data_risk_score(model: int = 3,
                         out: int = 3,
                         version: int = 1,
                         score='CURB') -> dict:
    """Reads data from a model and calculates specific RISK score.

    Parameters
    ----------
    model: int
        the data model to use
    out: int
        the specific outcome variable to use
    version: int
        the data version to use
    score: str
        the score, should be either 'CURB', 'REMS' or 'MEDS'

    Returns
    -------
    dict
        the data dictionary with X and Y-values and associated features
    """

    # Read data from file
    data_dict = {}
    model_dir = [x for x in os.listdir(os.path.join(os.getcwd(), 
                                                    'models')) if str(version).zfill(3) in x]
    if len(model_dir) > 1:
        logging.error("Found multiple versions with number: {}".format(version))
        logging.error("Please look into this!")
    else:
        model_dir = model_dir[0]
    
    model_file = "model_{}_out_{}_maindb.csv".format(model,
                                                     out)
    model_file2 = "model_{}_out_1_maindb.csv".format(model,
                                                     1)
    
    logging.info("Reading data from file: {}".format(model_file))
    try:
        df = pd.read_csv(os.path.join(os.getcwd(), 'models', model_dir, model_file),
                         sep=',',
                         header=0,
                         infer_datetime_format=True,
                         error_bad_lines=False,
                         engine='python',
                         encoding='utf-8')
        
        df2 = pd.read_csv(os.path.join(os.getcwd(), 'models', 'v018 - 25 March 19', model_file2),
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
        
    # Define outcome variable
    logging.info("Selecting Y-value")
    data_dict['y'] = df['y_var']
    df['septischeshock'] = df2['y_var']
    
    # Select X values
    logging.info("Selecting X-values")
    data_dict['x'] = _calculate_risk_score(df, score)

    return data_dict
