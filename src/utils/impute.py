"""
src/utils/impute.py
Created on: 23/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file contains all imputing utils needed in our processing process.

Functions:
    - impute
"""

# imports
import pandas as pd
import numpy as np

# encode imports
from fancyimpute import KNN, SimpleFill, IterativeImputer

# LOGGING
import logging, traceback
import logging.config
logging.config.fileConfig("logging/logging.conf.txt")


def impute(df: pd.DataFrame, impute_meth: str = 'KNN') -> pd.DataFrame:
    """Method to perform calibration in a K-Fold cross validation setting.

    Parameters
    ----------
    df: pd.DataFrame
        the dataframe to modify
    impute_meth: str
        the impute method, options are:
            - KNN
            - median
            - zero
            - zero_fill
            - MICE
            - complex
            - complex_zero
            - complex_zero2

    Returns
    -------
    pd.DataFrame
        returns the imputed pd.DataFrame

    """

    try:
        if impute_meth == 'KNN':
            _i = KNN(k=10).fit_transform(df)
            return pd.DataFrame(_i,
                                columns=df.columns,
                                index=df.index)

        elif impute_meth == 'median':
            _i = SimpleFill(fill_method='median').fit_transform(df)
            return pd.DataFrame(_i,
                                columns=df.columns,
                                index=df.index)
        elif impute_meth == 'zero':
            _i = SimpleFill(fill_method='zero').fit_transform(df)
            return pd.DataFrame(_i,
                                columns=df.columns,
                                index=df.index)

        elif impute_meth == 'zero_fill':
            return df.fillna(0)

        elif impute_meth == 'MICE':
            _i = IterativeImputer().fit_transform(df)
            return pd.DataFrame(_i,
                                columns=df.columns,
                                index=df.index)

        elif impute_meth == 'complex':
            for c in df.columns[2:]:
                if c == '1m-mortality':
                    break
                else:
                    df[c + 'binary'] = np.where(df[c].isnull(),
                                                0,
                                                1)
            _i = KNN(k=10).fit_transform(df)
            return pd.DataFrame(_i,
                                columns=df.columns,
                                index=df.index)

        elif impute_meth == 'complex_zero':
            for c in df.columns[2:]:
                if c in ('1m-mortality', 'Num_vars'):
                    break
                else:
                    df[c + 'binary'] = np.where(df[c].isnull(),
                                                0,
                                                1)

            return df

        elif impute_meth == 'complex_zero2':
            for c in df.columns[2:]:
                if c in ('1m-mortality', 'Num_vars'):
                    break
                else:
                    df[c + 'binary'] = np.where(df[c] != 0,
                                                1,
                                                0)
            return df
    
    except Exception as e:
        logging.error("Failed imputing with imputer {}".format(impute_meth))
        logging.error(traceback.format_exc()) 
