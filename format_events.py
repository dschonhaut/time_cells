"""
format_events.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com
    
Description: 
    Functions for formatting events files from the time cell study (Goldmine).

Last Edited: 
    11/6/19
"""

# General
import sys
import os
from time import time
from collections import OrderedDict as od
from glob import glob
import itertools

# Scientific
import numpy as np
import pandas as pd
import scipy.io as sio

# Stats
import scipy.stats as stats
import statsmodels.api as sm
import random

# Personal
sys.path.append('/Volumes/rhino/home1/dscho/code/general')
import data_io as dio
import array_operations as aop

def read_json(json_file):
    """Read the Goldmine json file.
    
    Stitches together broken lines and then
    checks that all lines are correctly formatted.
    
    Parameters
    ----------
    json_file : str
        Filepath to the json file
        
    Returns
    -------
    pandas.core.frame.DataFrame
        A DataFrame with len() == number of rows 
        in the json file
    """
    with open(json_file, 'r') as f_open:
        f_lines = [line.strip() for line in f_open.readlines()]

        # Stitch together broken lines
        f_lines_cleaned = []
        for iLine, line in enumerate(f_lines):
            if len(line) > 0:
                if (line[0]=='{'):
                    f_lines_cleaned.append(line)
                else:
                    f_lines_cleaned[-1] += line

        # Check that all lines are now correctly formatted
        assert np.all([((line[0]=='{') and (line[-1:]=='}')) for line in f_lines_cleaned])

        # Convert json list to a pandas DataFrame
        return pd.read_json('\n'.join([line for line in f_lines_cleaned]), lines=True)

def fill_column(df, key, key_, fill_back=False):
    """Create a column from the values in a df['value'][key_] 
    category for df['key']==key.
    
    if fill_back == True, then values are filled backwards from
    the indices where they occur. Otherwise values are filled
    forward from the indices where they occur.
    
    Returns
    -------
    newcol : list
        The new column values with len() == len(df)
    """
    df_ = df.loc[df['key']==key]
    if len(df_) == 0:
        return None
    
    inds = df_.index.tolist()
    vals = [row[key_] for row in df_['value']]
    
    for i in range(len(inds)+1):
        # Select the value that will be filled
        if i == 0:
            val = vals[i]
        elif fill_back: 
            if i == len(inds):
                val = vals[i - 1]
            else:
                val = vals[i]
        else:
            val = vals[i-1]
        
        # Fill the value over the appropriate
        # number of rows
        if i == 0:
            newcol = [val] * inds[i]
        elif i == len(inds):
            newcol += [val] * (len(df) - inds[i-1])
        else:
            newcol += [val] * (inds[i] - inds[i-1])
    
    return newcol

def get_trial_inds(df):
    """Figure out where each trial begins and ends based on gameState.

    Only complete trials are included.

    Returns
    -------
        trial_inds : itertools.OrderedDict
            (trial, [df_inds]) key/value pairs
    """
    inds = [idx for idx, row in df.query("(key=='gameState')").iterrows() 
            if row['value']['stateName'] in ['InitTrial', 'DoNextTrial']]
    df_ = df.loc[inds]
    trial_inds = od([])
    trial = 1
    iRow = 0
    while iRow < (len(df_)-1):
        if (df_.iloc[iRow]['gameState'] == 'InitTrial') and (df_.iloc[iRow+1]['gameState'] == 'DoNextTrial'):
            trial_inds[trial] = list(np.arange(df_.iloc[iRow].name, df_.iloc[iRow+1].name+1, dtype=int))
            trial += 1
            iRow += 2
        else:
            iRow += 1
    return trial_inds

def game_state_intervals(exp_df, game_state, cols=['time']):
    """Return trial-wise start and stop values for a game state.
    
    Values are determined by the column names in cols and are
    referenced against the index, with a trial period running
    from the first index of the trial to the first index of
    the next trial.
    
    Returns
    -------
    pandas.core.frame.DataFrame
    """
    def first_last(row):
        """Return first and last values in the col iterable."""
        vals = row.index.tolist()
        return [vals[0], vals[-1]+1] 
    
    # Format inputs correctly.
    if type(cols) == str:
        cols = [cols]
    
    # Ensure that all indices are consecutive (i.e. we are not accidentally
    # including another gameState in between values for the desired gameState)
    assert np.all([np.all(np.diff(x)==1) 
                   for x in exp_df.query("(gameState=='{}')".format(game_state))
                   .groupby('trial').indices.values()])

    # Group by trial and get the first and last indices for the gameState.
    output_df = (exp_df.query("(gameState=='{}')".format(game_state))
                       .groupby('trial')
                       .apply(lambda x: first_last(x))
                       .reset_index()
                       .rename(columns={0:'index'}))
    
    # Apply the indices to each column that we want to grab values for.
    for col in cols:
        output_df[col] = output_df['index'].apply(lambda x: [exp_df.loc[x[0], col], 
                                                             exp_df.loc[x[1], col]])
    
    return output_df