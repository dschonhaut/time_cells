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
12/4/20
"""
import sys
import os
from collections import OrderedDict as od
import h5py
import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import statsmodels.api as sm
sys.path.append('/home1/dscho/code/general')
import data_io as dio


def align_sync_pulses(event_synctimes,  # vector of event sync times
                      lfp_synctimes,  # vector of LFP sync times in ms
                      verbose=True):
    """Return the slope and intercept to align event to LFP times.
    
    Uses robust linear regression to estimate the intercept and slope 
    that best aligns event to LFP sync times.
       
    Parameters
    ----------
    event_synctimes : numpy.ndarray
        Vector of event sync times
    lfp_synctimes : numpy.ndarray
        Vector of LFP sync times
        
    Returns
    -------
    sync_params : collections.OrderedDict
        Intercept and slope to align
        event timestamps to LFP timestamps
    before_stats : collections.OrderedDict
        Pearson correlation and RMSE (in ms) between
        event and LFP sync times before alignment.
    after_stats : collections.OrderedDict
        Pearson correlation and RMSE (in ms) between
        event and LFP sync times after alignment.
    """
    def rmse(v1, v2):
        """Return the root mean squared error
        between equal-length vectors v1 and v2.
        """
        err = v1 - v2
        return np.sqrt(np.dot(err, err)/len(err))
        
    def error_percentiles(v1, v2):
        err = np.abs(v1 - v2)
        pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        return pd.Series([np.percentile(err, pct) for pct in pcts], index=pcts)
    
    # For each event sync time, find the closest LFP sync time.
    min_syncs = np.min((len(event_synctimes), len(lfp_synctimes)))
    sync_pairs = np.array([(event_synctimes[i], lfp_synctimes[i])
                           for i in range(min_syncs)])

    # Get a robust linear fit between the event/LFP sync pairs.
    X = sm.add_constant(sync_pairs[:, 0]) # the event sync times
    y = sync_pairs[:, 1] # the LFP channel sync times
    rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    intercept, slope = rlm_results.params
    
    # See how well the alignment went.
    sync_params = od([('intercept', intercept), ('slope', slope)])
    event_synctimes_aligned = intercept + (slope * event_synctimes)
    before_stats = od([('ipi_r', stats.pearsonr(np.diff(event_synctimes[:min_syncs]), np.diff(lfp_synctimes[:min_syncs]))[0]),
                       ('ipi_diff', np.mean(np.abs(np.diff(event_synctimes[:min_syncs]) - np.diff(lfp_synctimes[:min_syncs])))),
                       ('rmse', rmse(event_synctimes[:min_syncs], lfp_synctimes[:min_syncs])),
                       ('err_pcts', error_percentiles(event_synctimes[:min_syncs], lfp_synctimes[:min_syncs]))])
    after_stats = od([('ipi_r', stats.pearsonr(np.diff(event_synctimes_aligned[:min_syncs]), np.diff(lfp_synctimes[:min_syncs]))[0]),
                      ('ipi_diff', np.mean(np.abs(np.diff(event_synctimes_aligned[:min_syncs]) - np.diff(lfp_synctimes[:min_syncs])))),
                      ('rmse', rmse(event_synctimes_aligned[:min_syncs], lfp_synctimes[:min_syncs])),
                      ('err_pcts', error_percentiles(event_synctimes_aligned[:min_syncs], lfp_synctimes[:min_syncs]))])
    
    output = od([('sync_params', sync_params),
                 ('before_stats', before_stats),
                 ('after_stats', after_stats),
                 ('sync_pairs', sync_pairs)])
    return output


def create_event_time_bins(subj_sess,
                           events=None,
                           game_states=['Prepare1', 'Delay1', 'Encoding',
                                        'Prepare2', 'Delay2', 'Retrieval'],
                           proj_dir='/home1/dscho/projects/time_cells',
                           overwrite=False,
                           save_output=True,
                           verbose=False):
    """Break up event windows into evenly spaced time bins.

    Time bins are 500 ms each.
    """
    # Look for existing output file.
    output_f = os.path.join(proj_dir, 'analysis', 'events', 'event_times',
                            '{}-event_times.pkl'.format(subj_sess))
    if os.path.exists(output_f) and not overwrite:
        print('Found event_times')
        event_times = dio.open_pickle(output_f)
        return event_times

    # Load events.
    if events is None:
        events = dio.open_pickle(os.path.join(proj_dir, 'analysis', 'events',
                                              '{}-events_formatted.pkl'.format(subj_sess)))

    # Only select game states that were actually present in the experiment.
    game_states = [x for x in game_states 
                   if x in np.unique(events['gameState'])]

    # Get the event time windows for each game state.
    dfs = []
    for iState, game_state in enumerate(game_states):
        dfs.append(game_state_intervals(events, game_state=game_state, cols=['time']))
        dfs[-1].insert(1, 'trial_phase', iState + 1)
        dfs[-1].insert(2, 'gameState', game_state)
    event_times = pd.concat(dfs, axis=0)
    event_times.insert(0, 'subj_sess', subj_sess)
    event_times['duration'] = event_times['time'].apply(lambda x: x[-1]-x[0])

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
        elif row['gameState'] in ['Prepare1', 'Prepare2']:
            n_time_bins = 4
            time_bins.append(np.linspace(
                row['time'][0], row['time'][1], num=n_time_bins + 1))
        else:
            raise Exception("gameState '{}' not recognized".format(row['gameState']))
    
    event_times['time_bins'] = time_bins
    event_times['n_time_bins'] = event_times['time_bins'].apply(lambda x: len(x) - 1)
    event_times['time_bin_duration'] = event_times['time_bins'].apply(lambda x: np.median(np.diff(x)))
    event_times = event_times.sort_values(['trial', 'trial_phase']).reset_index(drop=True)

    # Save event times.
    if save_output:
        dio.save_pickle(event_times, output_f, verbose)

    return event_times


def edit_events(events):
    """Modify the events DataFrame after it was saved. Return events."""

    # Round event times to the nearest ms.
    events['time'] = events['time'].apply(lambda x: np.int(np.rint(x)))

    # Change the pre-delay message game state names to something shorter.
    events.loc[events['gameState']=='PreEncodingDelayMsg', 'gameState'] = 'Prepare1'
    events.loc[events['gameState']=='PreRetrievalDelayMsg', 'gameState'] = 'Prepare2'
    
    return events


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


def find_pulse_starts(sync_chan, 
                      pulse_thresh=200,  # voltage change
                      sampling_rate=32000,  # Hz
                      interpulse_thresh_ms=780,
                      intrapulse_thresh_ms=20,
                      pos_only=False,
                      verbose=True): 
    """Return sync_chan indices that mark that start of each sync pulse.
    
    Note: the default arguments were defined on data that were sampled
    at 2000 Hz and might need to be adjusted if the sampling rate
    differs much from this.
    
    Algorithm
    ---------
    1) Identifies sync pulse periods by finding sync channel indices
       for which the absolute value of the trace derivative exceeds
       pulse_thresh. 
    2) Identifies the start of each sync pulse by finding suprathreshold
       sync pulse indices for which the inter-pulse interval exceeds
       interpulse_thresh, and for which the subsequent suprathreshold 
       sync pulse occurs within a certain number of indices, defined by
       intrapulse_thresh. In other words, we are looking for dramatic
       changes in voltage that occur some time after the last dramatic
       voltage change, and that are sustained for some period of time.
    
    Parameters
    ----------
    sync_chan : numpy.ndarray
        Voltage trace from the channel that
        the sync box was plugged into
    pulse_thresh : int or float
        See algorithm description
    sampling_rate : int or float
        Sampling rate of the LFP sync channel in Hz
    interpulse_thresh_ms : int or float
        See algorithm description
    intrapulse_thresh_ms : int or float
        See algorithm description
    
    Returns
    -------
    pulse_startinds : numpy.ndarray
        Array of indices that mark the start of each sync pulse.
    """
    # Find sync pulses by looking for suprathreshold changes 
    # in the absolute value of the derivative of the sync channel
    sync_pulses = np.abs(np.pad(np.diff(sync_chan), (1, 0), 'constant')) > pulse_thresh
    pulse_inds = np.where(sync_pulses)[0]

    # Find the inter-pulse intervals
    ipis = np.insert(np.diff(pulse_inds), 0, pulse_inds[0])

    # Identify the start of each pulse by finding suprathreshold
    # inter-pulse intervals that are followed by a short IPI.
    interpulse_thresh = interpulse_thresh_ms * (sampling_rate / 1000)
    intrapulse_thresh = intrapulse_thresh_ms * (sampling_rate / 1000)
    pulse_startinds = pulse_inds[[i for i in range(len(ipis)-1) 
                                  if ((ipis[i]>interpulse_thresh) & (ipis[i+1]<intrapulse_thresh))]]
    
    if verbose:
        print('Detected {} EEG sync pulses'.format(len(pulse_startinds)))

    return pulse_startinds


def find_sync_shift(event_synctimes,  # vector of event sync times
                    lfp_synctimes,  # vector of LFP sync times in ms
                    verbose=True):
    """Find the best circ-shift index to apply to lfp_synctimes.
    
    Finds the max correlation between event_synctimes and lfp_synctimes
    inter-pulse intervals at all possible circular shifts.
    """
    # Find the best starting fit between event and LFP sync times
    # by comparing the inter-pulse intervals for each, testing
    # LFP sync shifts at different rolling indices.
    min_syncs = np.min((len(event_synctimes), len(lfp_synctimes)))
    lfp_synctimes_diff = np.diff(lfp_synctimes[:min_syncs])
    event_synctimes_diff = np.diff(event_synctimes[:min_syncs])
    offsets = np.arange(-(min_syncs-2), 1)
    rvals = np.array([stats.pearsonr(np.roll(lfp_synctimes_diff, offset), event_synctimes_diff)[0]
                      for offset in offsets])
    shift_by = offsets[rvals.argmax()]
    
    if verbose:
        print('Shift by {}'.format(shift_by))
        print('Max r = {:.3f}'.format(rvals.max()))
        print('Stdev r = {:.3f}'.format(rvals.std()))
    
    output = od([('offsets', offsets),
                 ('rvals', rvals),
                 ('shift_by', shift_by)])
    return output


def format_events(subj_sess=None,
                  events=None,
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
    if subj_sess is None:
        subj_sess = events.iloc[0]['subj_sess']
    output_f = os.path.join(proj_dir, 'analysis', 'events',
                            '{}-events_formatted.pkl'.format(subj_sess))
    if os.path.exists(output_f) and not overwrite:
        if verbose:
            print('Found formatted events')
        events = dio.open_pickle(output_f)
        events = edit_events(events)
        return events

    experiment_scene = events.loc[events['key']=='startMainExperiment', 'value'].iloc[0]['experimentScene']

    # Add column for scene.
    events['scene'] = fill_column(events, 'loadScene', 'sceneName', fill_back=False)

    # Get the main experiment events (dropping the tutorial events).
    events = events.loc[events['scene']==experiment_scene].reset_index(drop=True).copy()

    # Add column for game states.
    events['gameState'] = fill_column(events, 'gameState', 'stateName', fill_back=False)

    # Add column for trial.
    events['trial'] = 0
    trial_inds = get_trial_inds(events)
    for trial, inds in trial_inds.items():
        events.loc[inds, 'trial'] = trial

    # Add whether each trial has a time penalty or not. (-1 means we could not resolve.)
    events['time_penalty'] = -1
    for trial, has_penalty in {x['trial']: x['value']['isTimedTrial']
                               for idx, x in events.query("(key=='timedTrial')").iterrows()}.items():
        events.loc[events['trial'] == trial, 'time_penalty'] = 1 if has_penalty else 0

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
        trial = events.iloc[idx]['trial']
        game_state = events.iloc[idx]['gameState']
        if game_state in timed_game_states:
            # events.loc[(events['trial'] == trial) & (
            #     events['gameState'] == game_state), 'bad_trials'] = 'paused'
            events.loc[(events['trial'] == trial), 'bad_trials'] = 'paused'

    # Remove bad trial periods.
    if verbose:
        print('Removing incomplete trials and trials with a manual pause...', end='\n\n')
        print(events.query("bad_trials!=''")
                    .groupby(['trial', 'gameState'])['bad_trials']
                    .apply(lambda x: np.unique(x)), end='\n\n')

    events = events.loc[events['bad_trials'] == ''].reset_index(drop=True)

    # Other changes to events go here.
    events = edit_events(events)

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
    referenced against the index, with a trial phase running
    from the first index of the trial to the first index of
    the next trial phase.

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


def load_syncs(subj_sess,
               data_key='data',
               proj_dir='/home1/dscho/projects/time_cells',
               verbose=True):
    """Load the EEG channel that stores sync pulses from the testing laptop."""
    subj, sess = subj_sess.split('_')
    sync_f = os.path.join(proj_dir, 'data', subj, sess, 'sync', 'sync_channel_micro.mat')
    try:
        sync_chan = np.squeeze(sio.loadmat(sync_f)[data_key])
    except NotImplementedError:
        with h5py.File(sync_f, 'r') as f:
            sync_chan = np.squeeze(f[data_key])
    return sync_chan


def pair_sync_pulses(event_synctimes,
                     lfp_synctimes,
                     step=5,
                     max_shift=100,
                     max_slide=20,
                     ipi_thresh=2,
                     verbose=True):
    """Find matching sync pulse pairs.
    
    --------------------------------------------------------------------
    Works by comparing the inter-pulse interval times for a short chain 
    of sync pulses at a time, testing different offsets between event 
    and LFP pulses. When a good fit is found we add those pulses to the 
    output vectors; if no good fit is found we move on until we reach 
    the end of the input vectors.
    
    Retains as many identifiable sync pulse pairs as can be discerned.
    
    Parameters
    ----------
    event_synctimes : np.ndarray
        Vector of sync times from the behavioral events file.
    lfp_synctimes : np.ndarray
        Vector of sync times from detected pulses in the EEG sync 
        channel.
    step : int
        Determines how many sync pulses we compare at a time.
    max_shift : int
        How many index positions we shift by, in either direction, in
        testing for a good LFP-to-event IPI fit.
    max_slide : int
        How many index positions we slide either sync time vector
        forward by *before* testing for a fit at the different index
        shifts. This prevents the algorithm from getting stuck if we
        have e.g. an event sync pulse that failed to be written into
        the EEG sync channel.
    ipi_thresh : positive number
        The maximum allowable difference between event and LFP IPIs,
        in ms, in order for us to call a given fit "good" and add
        a chain of pulses to the output vectors.
        
    Returns
    -------
    event_synctimes_adj : np.ndarray
        The adjusted event sync times.
    lfp_synctimes_adj : np.ndarray
        The adjusted LFP sync times, which now match event_synctmes_adj
        at every index position, and are equal length.
    """
    def alternate_2col(stop_at):
        """Interleave (0, i) and (i, 0) pairs for i = 0..stop_at-1.

        First 3 entries are (0, 0), (0, 1), (1, 0).

        Returns an n x 2 matrix where n = (2 * stop_at) + 1.
        """
        mat = [(0, 0)]
        for x1, x2 in np.vstack((np.zeros(stop_at), np.arange(1, stop_at+1))).T:
            mat.append((x1, x2))
            mat.append((x2, x1))
        mat = np.array(mat).astype(np.int)
        return mat

    def eval_ipi_fit(lfp_synctimes_diff,
                     event_synctimes_diff,
                     shift_lfp, 
                     shift_event,
                     ipi_thresh=2):
        """Evaluate fit between event and LFP inter-pulse intervals.

        Finds the maximum IPI difference at a specified index shift, 
        and returns whether or not this difference is below the 
        allowable threshold.
        """
        global shift_lfps
        global shift_events
        ipi_fit = np.max(np.abs(lfp_synctimes_diff[shift_lfp:shift_lfp+step] - 
                                event_synctimes_diff[shift_event:shift_event+step]))
        if ipi_fit < ipi_thresh:
            shift_lfps.append(shift_lfp)
            shift_events.append(shift_event)
            found_fit = True
        else:
            found_fit = False

        return found_fit

    def find_ipi_fit(lfp_synctimes_diff,
                     event_synctimes_diff,
                     **kws):
        """Try to find a good event-to-LFP inter-pulse interval fit.

        Tests a range of index shifts between event and LFP IPI vectors
        and returns True when an acceptable fit is found, returning False
        if no fit is found after all shifts have been tested.
        """
        n_syncs = np.min((len(lfp_synctimes_diff), len(event_synctimes_diff)))
        shifts = alternate_2col(np.min((n_syncs-(step+1), max_shift)))
        for ii in range(len(shifts)):
            shift_lfp, shift_event = shifts[ii, :]
            found_fit = eval_ipi_fit(lfp_synctimes_diff,
                                     event_synctimes_diff,
                                     shift_lfp,
                                     shift_event,
                                     **kws)
            if found_fit:
                break

        return found_fit
    
    # Get the inter-pulse intervals.
    lfp_synctimes_diff = np.diff(lfp_synctimes)
    event_synctimes_diff = np.diff(event_synctimes)
    
    # Iterate over input sync time vectors until we reach the end of one.
    global shift_lfps
    global shift_events
    slides = alternate_2col(max_slide)
    lfp_synctimes_adj = []
    event_synctimes_adj = []
    shift_lfps = []
    shift_events = []
    loop = 0
    while np.min((len(lfp_synctimes_diff), len(event_synctimes_diff))) > (max_slide + step + 1):
        for ii in range(len(slides)):
            slide_lfp, slide_event = slides[ii, :]
            found_fit = find_ipi_fit(lfp_synctimes_diff[slide_lfp:],
                                     event_synctimes_diff[slide_event:],
                                     ipi_thresh=ipi_thresh)
            if found_fit:
                inc_lfp = slide_lfp + shift_lfps[-1]
                inc_event = slide_event + shift_events[-1]

                # Add sync times to the output vectors.
                lfp_synctimes_adj += list(lfp_synctimes[inc_lfp:inc_lfp+step])
                event_synctimes_adj += list(event_synctimes[inc_event:inc_event+step])

                # Remove sync times from the input vectors.
                lfp_synctimes = lfp_synctimes[inc_lfp+step:]
                lfp_synctimes_diff = lfp_synctimes_diff[inc_lfp+step:]
                event_synctimes = event_synctimes[inc_event+step:]
                event_synctimes_diff = event_synctimes_diff[inc_event+step:]

                break

        if not found_fit:
            if verbose:
                print('Loop {}: Inter-sync times failed to converge'.format(loop))

            # Jump ahead and try to keep going.
            lfp_synctimes = lfp_synctimes[max_slide+step:]
            lfp_synctimes_diff = lfp_synctimes_diff[max_slide+step:]
            event_synctimes = event_synctimes[max_slide+step:]
            event_synctimes_diff = event_synctimes_diff[max_slide+step:]

        loop += 1

    event_synctimes_adj = np.array(event_synctimes_adj)
    lfp_synctimes_adj = np.array(lfp_synctimes_adj)
    
    if verbose:
        print('Retained {} sync pulses'.format(len(lfp_synctimes_adj)))
        
    return event_synctimes_adj, lfp_synctimes_adj


def read_events_json(subj_sess,
                     proj_dir='/home1/dscho/projects/time_cells',
                     verbose=True):
    """Read events and setup for alignment.
    
    Event times are stored in ms.
    
    Returns
    -------
    events : pd.DataFrame
        Each row is one line of the json file that tracks
        everything logged by the testing laptop during gameplay.
    event_synctimes : np.ndarray
        Vector of sync times for every sync pulse that was sent
        to the EEG recording system.
    """
    subj, sess = subj_sess.split('_')
    events_f = os.path.join(proj_dir, 'data', subj, sess, 'events', 'events.jsonl')
    events = read_json(events_f)
    
    events = events[['time', 'type', 'data']].rename(columns={'type': 'key', 'data': 'value'})
    events['time'] = (events['time'] - events.at[0, 'time'])
    events.insert(0, 'subj_sess', subj_sess)
    event_synctimes = np.array(events.loc[events.key=='syncPulse', 'time'].tolist())
    
    if verbose:
        print('{} events recorded over {} min and {} sec'
              .format(len(events), 
                      np.int(events.iloc[-1]['time'] / 6e4), 
                      np.int((events.iloc[-1]['time'] % 6e4) / 1e3)))
        
    return events, event_synctimes


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


def rmse(v1, 
         v2):
    """Return the root mean squared error
    between equal-length vectors v1 and v2.
    """
    err = v1 - v2
    return np.sqrt(np.dot(err, err)/len(err))


def shift_spike_inds(spike_inds, 
                     step, 
                     floor=0, 
                     ceiling=np.inf):
    """Return the time-shifted spike_inds array.
    
    Parameters
    ----------
    spike_inds : np.ndarray
        Array of spike time indices.
    floor : int
        The lowest spike index before rolling backward from stop.
    ceiling : int
        One above highest spike index before rolling forward from start.
    step : int
        Number of indices to shift the spike train by.
    """
    ceiling -= floor
    spike_inds_shifted = (spike_inds-floor) + step
    
    if step < 0:
        roll_by = -len(spike_inds_shifted[spike_inds_shifted<0])
        spike_inds_shifted[spike_inds_shifted<0] = spike_inds_shifted[spike_inds_shifted<0] + ceiling
    else:
        roll_by = len(spike_inds_shifted[spike_inds_shifted>=ceiling])
        spike_inds_shifted[spike_inds_shifted>=ceiling] = spike_inds_shifted[spike_inds_shifted>=ceiling] - ceiling

    spike_inds_shifted = np.roll(spike_inds_shifted, roll_by) + floor

    return spike_inds_shifted


def trial_intervals(exp_df,
                    cols=['time']):
    """Return start and stop values for each trial.
    
    Values are determined by the column names in cols and are
    referenced against the index, with a trial running from the
    first index of the trial to the first index of the next trial.

    Returns
    -------
    pandas.core.frame.DataFrame
    """
    def first_last(row):
        """Return first and last values in the col iterable."""
        vals = row.index.tolist()
        return [vals[0], vals[-1] + 1]
    
    output_df = (exp_df.groupby('trial')
                       .apply(lambda x: first_last(x))
                       .reset_index()
                       .rename(columns={0: 'index'}))
    output_df.iloc[-1]['index'][1] -= 1
    
    # Apply the indices to each column that we want to grab values for.
    for col in cols:
        output_df[col] = output_df['index'].apply(lambda x: [exp_df.loc[x[0], col],
                                                             exp_df.loc[x[1], col]])
        if col == 'time':
            output_df['duration'] = output_df['time'].apply(lambda x: x[-1]-x[0])

    return output_df
