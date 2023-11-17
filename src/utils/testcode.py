"""
src/utils/testcode.py
Created on: 23/03/2019
Last edited: 17/07/2020
Author: William van Doorn

This file contains all functions needed around testcode processing.

Functions:
    - cleanup_testcode
    - process_testcode
    - encode_categories
"""

# general imports
import pandas as pd
import re

from collections import Counter

_TESTCODES_DROP: tuple = ('ARCHSE',
                          'ARCHFL',
                          'ARCHDA',
                          'ARCHUR',
                          'ARCHZE',
                          'ATLAS',
                          'TRE',
                          'STOL',
                          'CARDIOPL',
                          'CARDIOSE',
                          'ALLSTR',
                          'ARCHDA2',
                          'ARCHEP',
                          'ARCHZE2',
                          'ASFAC',
                          'ASTEKST',
                          'BSPTX',
                          'CYTO',
                          'EXTRA1',
                          'MATPOC',
                          'WOPM')

_CAT_VARS: list = ['ASTYP',
                   'ALLIND',
                   'ASNMA',
                   'BG',
                   'BRC',
                   'BGRH',
                   'RH',
                   'DIF',
                   'IRA',
                   'KETUP',
                   'GRTUP',
                   'SLEUUP',
                   'UGLUP',
                   'SERYUP',
                   'NITUP',
                   'PROUP',
                   'BACUP',
                   'MREB',
                   'Afdeling',
                   'Geslacht',
                   'ISE',
                   'LSE',
                   'HSE',
                   'LASP',
                   'RHFK']


def cleanup_testcode(df: pd.DataFrame, percentage: float = 0.9) -> pd.DataFrame:
    """Removes all redundant test codes.

    Parameters
    ----------
    df: pf.DataFrame
        the dataframe to modify
    percentage: float
        the amount of test codes to keep

    Returns
    -------
    pd.DataFrame
        returns the modified DataFrame

    """

    # Remove all the unused testcodes (e.g. archive ones)
    df = df[~df['Testcode'].isin(_TESTCODES_DROP)]

    # lets try percentages
    n_fraction = df['Testcode'].nunique() * percentage
    top_testcodes = df['Testcode'].value_counts().nlargest(round(n_fraction))
    df = df[df['Testcode'].isin(top_testcodes.index)]

    # Remove unused categories and return the dataframe
    df['Testcode'] = df.Testcode.cat.remove_unused_categories()

    return df


def process_testcode(row: pd.Series, debug: bool = False) -> object:
    """Processes all test codes for a specific row.

    Parameters
    ----------
    row: pd.Series
        the row to modify
    debug: bool
        if True, prints additional debug messages

    Returns
    -------
    pd.Series
        returns the modified row

    """

    if pd.isnull(row['Resultaat']):
        if debug:
            print(row['Testcode'], 'has NaN value - returning zero')
        return 'STRING'

    if row['Testcode'] in _CAT_VARS:
        if row['Testcode'] == 'MREB':
            if row['Resultaat'] in ['Oke', 'Niet afwijkend'] or row['Resultaat'] == 'Negatief':
                return 'Negatief'
            elif row['Resultaat'] == 'Positief':
                return 'Positief'
            else:
                return 'Bijzonder'
        else:
            return row['Resultaat']

    if row['Testcode'] == 'DAT':
        # process
        return 1

    if row['Testcode'] == 'SEDUP':
        return 1

    if row['Testcode'] in ['ATYL', 'FRAG', 'TOXK']:
        return Counter(row['Resultaat'])['+']

    if row['Testcode'] in ['DYSEUP', 'KP01', 'KP02', 'LASLI', 'POIK']:
        return 1

    if row['Testcode'] == 'DIF':
        if row['Resultaat'] == 'Oke':
            return 1
        else:
            return 0.5

    if '<' in row['Resultaat'] or '>' in row['Resultaat']:
        return float(re.findall(r"[-+]?\d*\.\d+|\d+", row['Resultaat'])[0])

    try:
        return pd.to_numeric(row['Resultaat'], errors='raise')
    except Exception as e:
        return 'STRING'


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Processes all test codes for a specific row.

    Parameters
    ----------
    df: pd.Series
        the dataframe to encode

    Returns
    -------
    pd.DataFrame
        returns the dataframe modified with one-hot encoding

    """

    for col in _CAT_VARS:
        if col in df.columns:
            one_hot = pd.get_dummies(df[col])
            one_hot = one_hot.add_prefix(col)
            df = df.join(one_hot)
            df = df.drop(col, 1)

    return df
