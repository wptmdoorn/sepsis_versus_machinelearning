"""
src/utils/cleaning.py
Created on: 23/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file contains all cleaning utils needed for cleaning.

Functions:
    - mortality
    - assign_id
    - process_and_transform
    - correct_times
"""

# general imports
import pandas as pd
import numpy as np
import datetime as dt

# settings
from settings import CLEANING
from utils.testcode import process_testcode

# LOGGING
import logging
import logging.config

logging.config.fileConfig("logging/logging.conf.txt")


def mortality(row: pd.Series, days: int = 31) -> int:
    """Generates a binary mortality variable.

    Parameters
    ----------
    row: pd.Series
        the row containing the data of the individual
    days: int
        the days to check for mortality in (e.g. 31-day)

    Returns
    -------
    int
        boolean value, 1 if True else 0
    """

    # If field is null, return 0
    if pd.isnull(row['Datum_overlijden']):
        return 0

    diff = (row['Datum_overlijden'] - row['Datum_SEH']).days
    if (diff >= -1) & (diff <= days):
        return 1
    else:
        return 0


def mortality_days(row: pd.Series) -> int:
    """Returns the amount of days the individual has died
    since the presentation at the ED.

    Parameters
    ----------
    row: pd.Series
        the row containing the data of the individual

    Returns
    -------
    int
        amount of days between ED presentation and death. If patient
        is still alive, it returns 1000.
    """

    if pd.isnull(row['Overleden']):
        return 1000
    
    diff = (row['Overleden'] - row['Datum_tijd']).days
    if (diff >= -1) & (diff < 31):
        return diff
    else:
        return 1000


def assign_id(df: pd.DataFrame) -> pd.DataFrame:
    """This method assigns an unique ID to a presentation of one specific patient
    at one time. Because a patient can enter the ED more than once, and also
    at one time more patients can enter; only the combination of this results
    in a truly unique identifier. Based on this combination we hereby assign
    an unique ID to each presentation event.

    Parameters
    ----------
    df: pd.DataFrame
        the DataFrame containing all the data

    Returns
    -------
    pd.DataFrame
        the modified pd.DataFrame
    """

    return df.assign(id=df.groupby(['Datum_tijd', 'Patientnummer']).ngroup())


def process_and_transform(df, debug: bool = False) -> pd.DataFrame:
    """This method is the general method for processing our dataframe to an useful
    matrix. This function essentially alters the data structure in a way that it
    becomes patient-centered instead of parameter-centered. It will be move all
    parameters to columns and make each row unique to the presentation of one
    patient with its respective lab-values <4 hours. Additionally, it mainly
    performs data processing.

    Parameters
    ----------
    df: pd.DataFrame
        the DataFrame containing all the data
    debug: bool = False
        if True, prints additional debug messages

    Returns
    -------
    pd.DataFrame
        the processed pd.DataFrame
    """

    # Transform resultaat parameters
    df['Resultaat'] = df.apply(process_testcode, args=debug, axis=1)
    
    # We want to remove duplicate test-codes (<4 hours) where we will keep
    # the first testcode if any duplicates are appearing
    df = df.drop_duplicates(subset=['id', 'Testcode'], keep='first')
    df['Testcode'] = df['Testcode'].astype('object')
    df.reset_index()
    
    # Now we want to make a pivot table based on id, testcode and resultaat
    _p = df.pivot('id', 'Testcode', 'Resultaat')
    _p = _p.reset_index()
    
    # Next we merge the original DF with the pivoted table to obtain a 
    # patient centered table
    df['id'] = df['id'].drop_duplicates()
    _f = pd.merge(df, _p, right_on='id', left_on='id')
    
    # Normalize the presentation time values
    # Calculate an age for each of the patient
    _f['Datum_SEH'] = pd.to_datetime(_f['Datum_tijd'])
    _f['Tijd'] = (_f['Datum_tijd'].dt.hour + _f['Datum_tijd'].dt.minute) / 60.0
    
    # Drop all the other unnecessary columns
    _f = _f.drop(columns=['Testcode',
                          'Resultaat',
                          'index',
                          'Datum_tijd'])
    
    # Return all the columns with too few values (date, ID, leeftijd, mortaliteit)
    _thres = CLEANING['LAB_THRESHOLD'] + 4
    logging.info('Parameter threshold: {}'.format(_thres))
    
    return _f.dropna(thresh=_thres)


def correct_times(df: pd.DataFrame, max_min: int = 240) -> pd.DataFrame:
    """This method is the general method for processing our dataframe to an useful
    matrix. This function essentially alters the data structure in a way that it
    becomes patient-centered instead of parameter-centered. It will be move all
    parameters to columns and make each row unique to the presentation of one
    patient with its respective lab-values <4 hours. Additionally, it mainly
    performs data processing.

    Parameters
    ----------
    df: pd.DataFrame
        the DataFrame containing all the data
    max_min: int
        the time interval to group times within

    Returns
    -------
    pd.DataFrame
        the time-corrected pd.DataFrame
    """

    # Generate unique dates sorted (earliest - latest)
    _l = list(df['Datum_tijd'].sort_values().unique())
    prev_item = _l.pop(0)
    
    # Loop through the whole list
    while len(_l) > 0:
        _item = _l.pop(0) 
        _diff = (_item - prev_item).astype('timedelta64[m]')

        if _diff < dt.timedelta(minutes=max_min + 1):
            df.loc[df['Datum_tijd'] == _item,
                   'Datum_tijd'] = prev_item
        elif _diff > dt.timedelta(minutes=(60 * 24)):
            prev_item = _item
        else:
            df.loc[df['Datum_tijd'] == _item, 'Datum_tijd'] = np.datetime64('2000-12-08T19:00:00.000000000')
    
    return df
