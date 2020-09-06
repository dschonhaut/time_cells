"""
events_preproc.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Functions for reading and processing Goldmine event logfiles.

Last Edited
----------- 
9/6/20
"""
import sys
import os
from collections import OrderedDict as od

import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd

sys.path.append('/home1/dscho/code/general')
import data_io as dio


def create_event_time_bins(subj_sess,
                           game_states=['Delay1', 'Encoding',
                                        'Delay2', 'Retrieval'],
                           proj_dir='/home1/dscho/projects/time_cells',
                           overwrite=False,
                           save_output=True,
                           verbose=False):
    """Break up event windows into evenly spaced time bins.

    Time bins are ~500 ms each (20 for the delay intervals
    and 60 for the navigation intervals).
    """
    # Look for existing output file.
    output_f = os.path.join(proj_dir, 'analysis', 'events', 'event_times',
                            '{}-event_times'.format(subj_sess))
    if os.path.exists(output_f) and not overwrite:
        print('Found event_times')
        event_times = dio.open_pickle(output_f)
        return event_times

    # Load events.
    events = dio.open_pickle(os.path.join(proj_dir, 'analysis', 'events',
                                          '{}-events_formatted.pkl'.format(subj_sess)))

    # Get the event time windows for each game state.
    dfs = []
    for iState, game_state in enumerate(game_states):
        dfs.append(game_state_intervals(
            events, game_state=game_state, cols=['time']))
        dfs[-1].insert(1, 'trial_phase', iState + 1)
        dfs[-1].insert(2, 'gameState', game_state)
        dfs[-1].insert(0, 'subj_sess', subj_sess)
    event_times = pd.concat(dfs, axis=0)

    # Divide each task period into 60 time bins of equal duration
    # (should be ~500 ms each)
    time_bins = []
    for idx, row in event_times.iterrows():
        if row['gameState'] in ['Encoding', 'Retrieval']:
            n_time_bins = 60
            time_bins.append(np.linspace(
                row['time'][0], row['time'][1], num=n_time_bins + 1))
        elif row['gameState'] in ['Delay1', 'Delay2']:
            n_time_bins = 20
            time_bins.append(np.linspace(
                row['time'][0], row['time'][1], num=n_time_bins + 1))
        else:
            # 500 ms time bin
            time_bins.append(np.arange(row['time'][0], row['time'][1], 500))

    event_times['time_bins'] = time_bins
    event_times = event_times.sort_values(
        ['trial', 'trial_phase']).reset_index(drop=True)

    # Save event times.
    if save_output:
        dio.save_pickle(event_times, output_f, verbose)

    return event_times


def fill_column(df,
                key,
                key_,
                fill_back=False):
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
    df_ = df.loc[df['key'] == key]
    if len(df_) == 0:
        return None

    inds = df_.index.tolist()
    vals = [row[key_] for row in df_['value']]

    for i in range(len(inds) + 1):
        # Select the value that will be filled
        if i == 0:
            val = vals[i]
        elif fill_back:
            if i == len(inds):
                val = vals[i - 1]
            else:
                val = vals[i]
        else:
            val = vals[i - 1]

        # Fill the value over the appropriate
        # number of rows
        if i == 0:
            newcol = [val] * inds[i]
        elif i == len(inds):
            newcol += [val] * (len(df) - inds[i - 1])
        else:
            newcol += [val] * (inds[i] - inds[i - 1])

    return newcol


def format_events(events,
                  experiment_scene='SMaze2',
                  overwrite=False,
                  save_output=True,
                  proj_dir='/home1/dscho/projects/time_cells',
                  verbose=True):
    """Format Goldmine events for a testing session.

    This function is run *after* event timestamps
    have been aligned to LFP timestamps.

    Returns
    -------
    events : pandas DataFrame
    """
    # Look for existing output file.
    subj_sess = events.iloc[0]['subj_sess']
    output_f = os.path.join(proj_dir, 'analysis', 'events',
                            '{}-events_formatted.pkl'.format(subj_sess))
    if os.path.exists(output_f) and not overwrite:
        print('Found formatted events')
        events = dio.open_pickle(output_f)
        return events

    # Add column for scene.
    events['scene'] = fill_column(
        events, 'loadScene', 'sceneName', fill_back=False)

    # Get the main experiment events (dropping the tutorial events).
    events = events.loc[events['scene']==experiment_scene].reset_index(drop=True).copy()

    # Add column for game states.
    events['gameState'] = fill_column(
        events, 'gameState', 'stateName', fill_back=False)

    # Add column for trial.
    events['trial'] = 0
    trial_inds = get_trial_inds(events)
    for trial, inds in trial_inds.items():
        events.loc[inds, 'trial'] = trial

    # Add whether each trial has a time penalty or not. (-1 means we could not resolve.)
    events['time_penalty'] = -1
    for trial, has_penalty in {x['trial']: x['value']['isTimedTrial']
                               for idx, x in events.query("(key=='timedTrial')").iterrows()}.items():
        events.loc[events['trial'] == trial,
                   'time_penalty'] = 1 if has_penalty else 0

    # Reorder columns.
    events = events[['time', 'key', 'value', 'scene',
                     'trial', 'time_penalty', 'gameState']]

    # Distinguish between pre-encoding delays (Delay1)
    # and pre-retrieval delays (Delay2),
    # and between post-encoding returns to base (ReturnToBase1)
    # and post-retrieval returns to base (ReturnToBase2)
    for trial in range(1, events['trial'].max() + 1):
        for game_state in ['Delay', 'ReturnToBase']:
            inds = events.loc[(events['trial'] == trial) & (
                events['gameState'] == game_state)].index.tolist()
            sep = np.where(np.diff(inds) > 1)[0]
            assert len(sep) == 1
            events.loc[(events.index.isin(inds[:sep[0] + 1])) &
                       (events['trial'] == trial) &
                       (events['gameState'] == game_state), 'gameState'] = game_state + '1'
            events.loc[(events.index.isin(inds[sep[0] + 1:])) &
                       (events['trial'] == trial) &
                       (events['gameState'] == game_state), 'gameState'] = game_state + '2'

    # Take note of which trial periods should be thrown out.
    events['bad_trials'] = ''

    # Flag incomplete trials.
    check_game_states = ['InitTrial', 'Delay1', 'Encoding', 'ReturnToBase1',
                         'Delay2', 'Retrieval', 'ReturnToBase2', 'DoNextTrial']
    events.loc[(events['trial'] == 0), 'bad_trials'] = 'incomplete'
    for trial in range(1, events['trial'].max() + 1):
        game_states = list(
            events.loc[(events['trial'] == trial), 'gameState'].unique())
        if not np.all([x in game_states for x in check_game_states]):
            events.loc[(events['trial'] == trial), 'bad_trials'] = 'incomplete'

    # Flag trial periods with manual pauses.
    timed_game_states = ['Delay1', 'Encoding', 'Delay2', 'Retrieval']
    pause_inds = [idx for idx, row in events.query("(key=='gamePaused')").iterrows()
                  if row['value']['pauseType'] == 'manualPause']
    for idx in pause_inds:
        game_state = events.iloc[idx]['gameState']
        if game_state in timed_game_states:
            events.loc[(events['trial'] == trial) & (
                events['gameState'] == game_state), 'bad_trials'] = 'paused'

    # Remove bad trial periods.
    if verbose:
        print('Removing trial periods:\n')

    events = events.loc[events['bad_trials'] == ''].reset_index(drop=True)

    if verbose:
        print('Main experiment has {} events recorded over {} min and {} sec'
              .format(len(events), np.int((events.iloc[-1]['time'] - events.iloc[0]['time']) / 60000),
                      np.int(((events.iloc[-1]['time'] - events.iloc[0]['time']) % 60000) / 1000)))

    # Save events.
    if save_output:
        dio.save_pickle(events, output_f, verbose)

    return events


def game_state_intervals(exp_df,
                         game_state,
                         cols=['time']):
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
        return [vals[0], vals[-1] + 1]

    # Format inputs correctly.
    if type(cols) == str:
        cols = [cols]

    # Ensure that all indices are consecutive (i.e. we are not accidentally
    # including another gameState in between values for the desired gameState)
    assert np.all([np.all(np.diff(x) == 1)
                   for x in exp_df.query("(gameState=='{}')".format(game_state))
                   .groupby('trial').indices.values()])

    # Group by trial and get the first and last indices for the gameState.
    output_df = (exp_df.query("(gameState=='{}')".format(game_state))
                       .groupby('trial')
                       .apply(lambda x: first_last(x))
                       .reset_index()
                       .rename(columns={0: 'index'}))

    # Apply the indices to each column that we want to grab values for.
    for col in cols:
        output_df[col] = output_df['index'].apply(lambda x: [exp_df.loc[x[0], col],
                                                             exp_df.loc[x[1], col]])

    return output_df


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
    while iRow < (len(df_) - 1):
        if (df_.iloc[iRow]['gameState'] == 'InitTrial') and (df_.iloc[iRow + 1]['gameState'] == 'DoNextTrial'):
            trial_inds[trial] = list(
                np.arange(df_.iloc[iRow].name, df_.iloc[iRow + 1].name + 1, dtype=int))
            trial += 1
            iRow += 2
        else:
            iRow += 1
    return trial_inds
