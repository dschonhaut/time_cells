"""
spike_preproc.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Functions for reading and formatting wave_clus files
and for generating circ-shifted null distributions
in Goldmine data.

Last Edited
----------- 
9/6/20
"""
import sys
import os
from glob import glob
from collections import OrderedDict as od

import mkl
mkl.set_num_threads(1)
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import h5py
import scipy.io as sio

sys.path.append('/home1/dscho/code/general')
import data_io as dio


def add_fr_to_spikes(spikes,
                     sigma=500, # ms
                     proj_dir='/home1/dscho/projects/time_cells',
                     verbose=True,
                     save_output=True):
    """Add firing rate vectors to the spikes DataFrame.
    
    Parameters
    ----------
    subj_sess : str
        E.g. "U518_ses0"
    sess_stop : int
        Stop time for the session in ms (start time is 0 ms).
    sigma : int
        Standard deviation of the Gaussian kernel in ms.
    verbose : bool
        Whether or not to print anything to the standard output.
    save_output : bool
        If true, the spikes file will be overwritten with the original
        spikes DataFrame plus a column of firing rates for each neuron,
        sampled in ms. 
    
    Returns
    -------
    spikes : pandas DataFrame
    """
    # Load spikes.
    subj_sess = spikes.iloc[0]['subj_sess']
    stop_time = spikes.iloc[0]['sess_stop_time']
    
    spikes['fr_train'] = spikes['spike_times'].apply(lambda spike_times: 
                                                     spike_times_to_fr(spike_times, 
                                                                       stop_time, 
                                                                       sigma))
    
    if save_output:
        output_f = os.path.join(proj_dir, 'analysis', 'spikes', 
                                '{}-spikes.pkl'.format(subj_sess))
        dio.save_pickle(spikes, output_f, verbose)
        
    return spikes
    

def add_null_to_spikes(subj_sess,
                       event_times,
                       n_perms=1000,
                       proj_dir='/home1/dscho/projects/time_cells',
                       verbose=True,
                       save_output=True):
    """Add spike times null distribution to the spikes DataFrame.
    
    Contains all neurons in a session.
    """
    # Look for existing output file.
    output_f = os.path.join(proj_dir, 'analysis', 'spikes', 
                            '{}-spikes.pkl'.format(subj_sess))
    spikes = dio.open_pickle(output_f)
    
    event_times = event_times.reset_index(drop=True)
    spikes['spike_times_null'] = spikes['spike_times'].apply(lambda x: create_null_spike_times(x, event_times, n_perms)).tolist()
    
    if save_output:
        dio.save_pickle(spikes, output_f, verbose)
        
    return spikes
    

def apply_shift_spike_times(event_window, spike_times):
    """Circ-shift spike times within an event window.
    
    Circ-shifts spike times by a random number of 
    timepoints up to the duration of the event window.
    
    Parameters
    ----------
    event_window : list or tuple
        Contains [start, stop] times.
    spike_times : numpy.ndarray
        Vector of spike times for the whole session.
    
    Returns:
    --------
    spike_times_shifted_ : numpy.ndarray
        Vector of shifted spike times within the
        trial phase.
    """
    start, stop = event_window
    spike_times_ = spike_times[(spike_times>=start) & (spike_times<stop)]
    shift_by = int(np.random.rand() * (stop - start))
    spike_times_shifted_ = shift_spike_inds(spike_times_, shift_by, floor=start, ceiling=stop)
    
    return spike_times_shifted_


def create_null_spike_times(spike_times, event_times, n_perms=1000):
    """Create a null distribution of spike times for a neuron.
    
    Lambda function to apply over a pandas Series of spike_times arrays.
    """
    spike_times_shifted = np.array([np.concatenate(event_times['time'].apply(lambda x: apply_shift_spike_times(x, spike_times)))
                                    for i in range(n_perms)]) # perms x spike_times
    
    return spike_times_shifted


def format_spikes(subj_sess,
                  n_spike_thresh=0,
                  fr_thresh=0, # in Hz
                  calc_fr=True,
                  sigma=500,
                  add_montage_info=True,
                  proj_dir='/home1/dscho/projects/time_cells',
                  spikes_dirname='wave_clus3_sortbyhand',
                  overwrite=False,
                  save_output=True,
                  split_files=True,
                  verbose=False):
    """Gather spike times from all wave_clus single-units.
    
    Depends on the Goldmine file structure.
    
    Timestamps are converted to ms for compatability with
    the micro LFP files.
    
    Returns
    -------
    spikes : pandas.core.frame.DataFrame
    """
    # Look for existing output file.
    output_f = os.path.join(proj_dir, 'analysis', 'spikes', 
                            '{}-spikes.pkl'.format(subj_sess))
    
    if os.path.exists(output_f) and not overwrite:
        print('Found spikes')
        spikes = dio.open_pickle(output_f)
        return spikes
    
    # Check for required input files.
    subj, sess = subj_sess.split('_')
    spikes_dir = os.path.join(proj_dir, 'data', subj, sess, 'spikes', spikes_dirname)
    micros_dir = os.path.join(proj_dir, 'data', subj, sess, 'micro_lfps')
    lfp_timestamps_f = os.path.join(micros_dir, 'lfpTimeStamps_micro.mat')
    
    assert os.path.exists(lfp_timestamps_f)
    
    if add_montage_info:
        elec_montage_f = os.path.join(micros_dir, 'anatleads.txt')
        assert os.path.exists(elec_montage_f)
    
    # Get all spike files for the session.
    spike_files_ = glob(os.path.join(spikes_dir, 'times*.mat'))
    
    assert len(spike_files_) > 0
    
    if verbose:
        print('Found {} wave_clus files.'.format(len(spike_files_)), end='\n\n')
    
    # Reorder spike files.
    spike_fname = 'times_CSC{}.mat'
    spike_files = [os.path.join(spikes_dir, spike_fname.format(chan))
                   for chan in range(1, len(spike_files_)+1)
                   if os.path.exists(os.path.join(spikes_dir, spike_fname.format(chan)))]
    
    if verbose: 
        if len(spike_files) < len(spike_files_):
            print('Dropped these files: {}'
                  .format([f for f in spike_files_ if f not in spike_files]))
        print('{} wave_clus files after reordering.'.format(len(spike_files)), end='\n\n')
    
    # Get LFP timestamps, convert sec to ms,
    # and subtract the first timestamp from all timestamps
    try:
        lfp_timestamps = np.squeeze(sio.loadmat(lfp_timestamps_f)['timeStamps'])
    except NotImplementedError:
        with h5py.File(lfp_timestamps_f, 'r') as f:
            lfp_timestamps = np.squeeze(f['timeStamps'])
    lfp_timestamps = lfp_timestamps * 1e3
    sess_stop_time = int(lfp_timestamps[-1])
    session_dur = (lfp_timestamps[-1] - lfp_timestamps[0]) * 1e-3 # in sec
    
    if verbose:
        print('session duration is {} min and {} sec'.format(np.int(session_dur / 60), 
                                                             np.int(session_dur % 60)),
              end='\n\n')
        
    # Get spike times from all wave_clus single-units.
    chans = [int(os.path.basename(f).split('CSC')[1].split('.')[0]) 
             for f in spike_files]
    spikes = []
    for iChan, spike_file in enumerate(spike_files):
        chan = chans[iChan]
        cluster_class = sio.loadmat(spike_file)['cluster_class']
        units = np.unique(cluster_class[:, 0])
        if len(units) > 1:
            for unit in range(1, len(units)):
                spike_times = cluster_class[:, 1][np.where(cluster_class[:, 0]==unit)[0]]
                n_spikes = len(spike_times)
                fr = n_spikes/session_dur
                if (n_spikes>n_spike_thresh) & (fr>fr_thresh):
                    spikes.append([subj_sess, subj, sess, chan, unit,
                                   sess_stop_time, spike_times, n_spikes, fr])
    cols = ['subj_sess', 'subj', 'sess', 'chan', 'unit',
            'sess_stop_time', 'spike_times', 'n_spikes', 'fr']
    spikes = pd.DataFrame(spikes, columns=cols)
    
    if verbose:
        print('Found {} neurons'.format(len(spikes)))
        print('Firing rates:', spikes['fr'].describe(), sep='\n', end='\n\n')
        
    # Add electrode region info.
    if add_montage_info:
        def get_hemroi(chan, mont):
            for hemroi, chans in mont.items():
                if chan in chans:
                    return hemroi
        
        with open(elec_montage_f, 'r') as f:
            mont = f.readlines()
        mont = [line.strip('\n').split(', ') for line in mont]
        mont = od([(x[0], np.arange(int(x[1].split('-')[0]), int(x[1].split('-')[1])+1))
                   for x in mont])
        
        spikes['hemroi'] = spikes['chan'].apply(lambda x: get_hemroi(x, mont))
    
    if verbose:
        print(spikes.groupby('hemroi').agg({'unit': len, 
                                            'chan': lambda x: len(np.unique(x)), 
                                            'fr': np.median}))
    
    # Convolve spikes with a Gaussian filter to calculate firing rate.
    if calc_fr:
        spikes = add_fr_to_spikes(spikes, 
                                  sigma=sigma,
                                  proj_dir=proj_dir,
                                  verbose=verbose,
                                  save_output=False)
        
    # Save spikes.
    if save_output:
        # Split files will save one file per neuron (as a pandas Series);
        # otherwise all neurons in a session are saved in a DataFrame together.
        if split_files:
            output_f = os.path.join(proj_dir, 'analysis', 'spikes', '{}-CSC{}-unit{}-spikes.pkl')
            for idx, row in spikes.iterrows():
                dio.save_pickle(row, output_f.format(row['subj_sess'], row['chan'], row['unit']), verbose)
        else:
            dio.save_pickle(spikes, output_f, verbose)
    
    return spikes


def get_fr_train_events(fr_train, event_times):
    """Return the fr_train including only event windows."""
    def clip_fr_train(event_window, fr_train):
        start, stop = event_window
        start = int(np.round(start, 0))
        stop = int(np.round(stop, 0))
        return fr_train[start:stop]
        
    return np.concatenate(event_times['time'].apply(lambda x: clip_fr_train(x, fr_train)))


def get_unit_df(analysis_dir='/home1/dscho/projects/time_cells/analysis'):
    """Return a DataFrame with file pointers for each neuron."""
    def format_info(spikes_f):
        x = os.path.basename(spikes_f).split('-')
        subj_sess = x[0]
        chan = x[1][3:]
        unit = x[2][4:]
        fr_by_time_delay_f = os.path.join(analysis_dir, 'spikes_by_time_bin',
                                          '{}-CSC{}-unit{}-{}-fr_by_time_bin.pkl'
                                          .format(subj_sess, chan, unit, 'Delay1_Delay2'))
        fr_by_time_nav_f = os.path.join(analysis_dir, 'spikes_by_time_bin',
                                        '{}-CSC{}-unit{}-{}-fr_by_time_bin.pkl'
                                        .format(subj_sess, chan, unit, 'Encoding_Retrieval'))
        output = od([('subj_sess', subj_sess),
                     ('chan', chan),
                     ('unit', unit),
                     ('spikes_f', spikes_f),
                     ('fr_by_time_delay_f', fr_by_time_delay_f),
                     ('fr_by_time_nav_f', fr_by_time_nav_f)])
        return [subj_sess, chan, unit, spikes_f, fr_by_time_delay_f, fr_by_time_nav_f]
    
    cols = ['subj_sess', 'chan', 'unit', 'spikes_f', 'fr_by_time_delay_f', 'fr_by_time_nav_f']
    spike_files = glob(os.path.join(analysis_dir, 'spikes', '*CSC*.pkl'))
    units = [format_info(x) for x in spike_files]
    unit_df = pd.DataFrame(units, columns=cols)
    return unit_df


def shift_spike_inds(spike_inds, step, floor=0, ceiling=np.inf):
    """Return the time-shifted spike_inds array.
    
    Parameters
    ----------
    spike_inds : numpy.ndarray
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


def spike_times_to_fr(spike_times,
                      stop_time,
                      sigma=500):
    """Convert spikes times into firing rates.
    
    Converts spike times into a spike train
    that is convolved with a Gaussian filter.
    
    Time is assumed to start at 0 ms. Each value 
    in the output vector corresponds to 1 ms. 
    
    Parameters
    ----------
    spike_times : np.ndarray
        A vector of spike times in ms.
    stop_time : int
        Stop time for the session in ms (start time is 0 ms).
    sigma : int
        Standard deviation of the Gaussian kernel in ms.
    
    Returns
    -------
    fr = np.ndarray
        A vector whose length equals the stop
        time + 1. Each value is the cell's 
        Gaussian smoothed firing rate in spikes/sec. 
    """
    # Round spike times to the nearest ms.
    spike_times = np.unique(np.array(np.round(spike_times, 0), dtype=int))
    
    # Convolve the spike train with a Gaussian.
    spike_train = np.zeros(int(stop_time)+1)
    spike_train[spike_times] = 1
    fr = gaussian_filter(spike_train, sigma) * 1e3 # converting ms to sec
    return fr
