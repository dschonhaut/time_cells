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
3/5/21
"""
import sys
import os.path as op
from glob import glob
from collections import OrderedDict as od
<<<<<<< HEAD
# import mkl
# mkl.set_num_threads(1)
=======
import mkl
mkl.set_num_threads(1)
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import scipy.io as sio
sys.path.append('/home1/dscho/code/general')
import data_io as dio
sys.path.append('/home1/dscho/code/projects')
from time_cells import spike_sorting


def load_spikes(subj_sess,
                neuron,
                proj_dir='/home1/dscho/projects/time_cells'):
    """Load spike info for the selected neuron.

    Parameters
    ----------
    subj_sess : str
        e.g. 'U518_ses0'
    neuron : str
        e.g. '17-1' is channel 17, unit 1.

    Returns
    -------
    spikes : Series
    """
    chan, unit = neuron.split('-')
    filename = op.join(proj_dir, 'analysis', 'spikes', 
                       '{}-CSC{}-unit{}-spikes.pkl'.format(subj_sess, chan, unit))
    spikes = dio.open_pickle(filename)
    return spikes


def unit_from_file(file):
    """Return unique unit name from the spike file name."""
    _, chan, unit, _ = op.basename(file).split('-')
    chan = chan.replace('CSC', '')
    unit = unit.replace('unit', '')
    return '{}-{}'.format(chan, unit)


def add_null_to_spikes(subj_sess,
                       event_times,
                       spikes=None,
                       n_perms=1000,
                       output_col='spike_times_null',
                       proj_dir='/home1/dscho/projects/time_cells',
                       save_output=True,
                       split_files=True,
                       verbose=True):
    """Add spike times null distribution to the spikes DataFrame.

    Operates on the output of format_spikes().
    
    Creates a null distribution of randomly, circ-shifted spike times
    within each trial phase, for all neurons in a session.
    """
    # Check for existing output file.
    output_f = op.join(proj_dir, 'analysis', 'spikes', 
                       '{}-spikes.pkl'.format(subj_sess))
    if spikes is None:
        spikes = dio.open_pickle(output_f)
    
    event_times = event_times.reset_index(drop=True)
    spikes[output_col] = spikes['spike_times'].apply(lambda x: create_null_spike_times(x, event_times, n_perms)).tolist()
    
    if save_output:
        # Split files will save one file per neuron (as a pandas Series);
        # otherwise all neurons in a session are saved in a DataFrame together.
        if split_files:
            output_f = op.join(proj_dir, 'analysis', 'spikes', '{}-CSC{}-unit{}-spikes.pkl')
            for idx, row in spikes.iterrows():
                dio.save_pickle(row, output_f.format(row['subj_sess'], row['chan'], row['unit']), verbose)
        else:
            dio.save_pickle(spikes, output_f, verbose)
        
    return spikes
    

def apply_shift_spike_times(event_window, 
                            spike_times):
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
    shift_by = np.int(np.random.rand() * (stop - start))
    spike_times_shifted_ = shift_spike_inds(spike_times_, shift_by, floor=start, ceiling=stop)
    
    return spike_times_shifted_


def apply_spike_times_to_fr(event_window, 
                            spike_times,
                            sigma=500):
    """Converts spike times to firing rate within an event window.
    
    Parameters
    ----------
    event_window : list or tuple
        Contains [start, stop] times.
    spike_times : numpy.ndarray
        Vector of spike times for the whole session.
    sigma : int or float
        Standard deviation of the Gaussian smoothing kernel.
    
    Returns:
    --------
    spike_times_shifted_ : numpy.ndarray
        Vector of shifted spike times within the
        trial phase.
    """
    start, stop = event_window
    dur = stop - start
    
    # Select spike times that fall within the event window, and subtract
    # the start of the event window from each spike time.
    shp = spike_times.shape
    if len(shp) == 1:
        spike_times_ = spike_times[(spike_times>=start) & (spike_times<stop)]
        
    else:
        spike_times_ = np.array([spike_times[x, (spike_times[x, :]>=start) & (spike_times[x, :]<stop)]
                                 for x in range(shp[0])])
    spike_times_ -= start

    # Convert spike times into a spike train and convolve with a Gaussian
    # filter to obtain a firing rate at each ms in the event window.
    fr = spike_times_to_fr(spike_times_, dur, sigma)
    
    return fr


def create_null_spike_times(spike_times, 
                            event_times, 
                            n_perms=1000):
    """Create a null distribution of spike times for a neuron.
    
    Lambda function to apply over a pandas Series of spike_times arrays.
    """
    spike_times_shifted = np.array([np.concatenate(event_times['time'].apply(lambda x: apply_shift_spike_times(x, spike_times)))
                                    for i in range(n_perms)]) # perms x spike_times
    
    return spike_times_shifted


def event_fr(spikes,
             iUnit=None,
             trial_times=None,
             event_times=None,
             include_trial_null=True,
             include_event_null=True,
             sigma=500,
             proj_dir='/home1/dscho/projects/time_cells',
             overwrite=False,
             save_output=True,
             verbose=True):
    """Calculate a neuron's mean FR within each trial or trial phase time bin."""
    def mean_fr(time_range, 
                spike_times,
                bins=None):
        """Return the mean FR within a given time range 
        or within a given number of bins that evenly divide the range.
        """
        # Mean FR across the trial.
        # Returns a scalar.
        if (bins is None) & (len(spike_times.shape) == 1):
            return np.mean(apply_spike_times_to_fr(time_range, 
                                                   spike_times, 
                                                   sigma))
        # Mean FR across the trial, for each permutation.
        # Returns a vector whose length is the number of permutations.
        elif (bins is None) & (len(spike_times.shape) == 2):
            return np.mean(apply_spike_times_to_fr(time_range, 
                                                   spike_times, 
                                                   sigma),
                           axis=-1)
        # Mean FR within each time bin.
        # Returns a vector whose length is the number of time bins.
        elif (bins is not None) & (len(spike_times.shape) == 1):
            return np.array([np.mean(arr) 
                             for arr in np.array_split(apply_spike_times_to_fr(time_range, 
                                                                               spike_times, 
                                                                               sigma), 
                                                       bins)])
        # Mean FR within each time bin, for each permutation.
        # Returns a perm x time_bin array.
        elif (bins is not None) & (len(spike_times.shape) == 2):
            return np.swapaxes([np.mean(arr, axis=-1)
                                for arr in np.array_split(apply_spike_times_to_fr(time_range, 
                                                                                  spike_times, 
                                                                                  sigma), 
                                                          bins, axis=-1)], 0, 1)
        else:
            raise Exception('Unexpected inputs.')

    # Select the neuron to process.
    if iUnit is not None:
        spikes = spikes.iloc[iUnit]

    # Load the output file if it exists.
    output_f = op.join(proj_dir, 'analysis', 'fr_by_time_bin',
                       '{}-CSC{}-unit{}-event_fr.pkl'
                       .format(spikes['subj_sess'], spikes['chan'], spikes['unit']))
    if op.exists(output_f) and not overwrite:
        return dio.open_pickle(output_f)

    # Get firing rates!
    spike_times = spikes['spike_times']
    if trial_times is not None:
        trial_times['fr'] = trial_times['time'].apply(lambda x: mean_fr(x, spike_times))
        if include_trial_null:
            spike_times_trial_null = spikes['spike_times_null_trial']
            trial_times['fr_null'] = trial_times['time'].apply(lambda x: mean_fr(x, spike_times_trial_null))

    if event_times is not None:
        event_times['fr'] = event_times.apply(lambda x: mean_fr(x['time'], spike_times, x['n_time_bins']), axis=1)
        if include_event_null:
            spike_times_event_null = spikes['spike_times_null_trial_phase']
            event_times['fr_null'] = event_times.apply(lambda x: mean_fr(x['time'], spike_times_event_null, x['n_time_bins']), axis=1)

    # Format outputs.
    output = od([])
    if trial_times is not None:
        output['trial_times'] = trial_times
    if event_times is not None:
        output['event_times'] = event_times

    # Save outputs.
    if save_output:
        dio.save_pickle(output, output_f, verbose)

    return output


def format_spikes(subj_sess,
                  n_spike_thresh=0,
                  fr_thresh=0, # in Hz
                  add_montage_info=True,
                  conv_factor='infer',
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
    
    Parameters
    ----------
    conv_factor : int or float
        The number that lfp_timestamps must be multiplied by to be in ms

    Returns
    -------
    spikes : pandas.core.frame.DataFrame
    """
    # Look for existing output file.
    output_f = op.join(proj_dir, 'analysis', 'spikes', '{}-spikes.pkl'.format(subj_sess))
    
    if op.exists(output_f) and not overwrite:
        print('Found spikes')
        spikes = dio.open_pickle(output_f)
        return spikes
    
    # Load the required inputs.
    subj, sess = subj_sess.split('_')    
    lfp_timestamps, sr = spike_sorting.load_lfp_timestamps(subj_sess=subj_sess, 
                                                           conv_factor=conv_factor, 
                                                           proj_dir=proj_dir,
                                                           verbose=verbose)
    sess_stop_time = np.int(lfp_timestamps[-1])
    session_dur = lfp_timestamps[-1] - lfp_timestamps[0]

    if add_montage_info:
        elec_montage_f = op.join(proj_dir, 'data', subj, sess, 'micro_lfps', 'anatleads.txt')
        assert op.exists(elec_montage_f)

    # Get all spike files for the session.
    spike_files = np.array(glob(op.join(proj_dir, 'data', subj, sess, 'spikes',
                                        spikes_dirname, 'times_CSC*.mat')))
    assert len(spike_files) > 0
    if verbose:
        print('Found {} wave_clus files.'.format(len(spike_files)), end='\n\n')
    
    # Reorder spike files.
    chans = np.array([np.int(op.basename(f).split('CSC')[1].split('.')[0])
                      for f in spike_files])
    xsort = np.argsort(chans)
    spike_files = spike_files[xsort]
    chans = chans[xsort]
    
    # Get spike times from all wave_clus single-units.
    spikes = []
    for iChan, spike_file in enumerate(spike_files):
        chan = chans[iChan]
        cluster_class = sio.loadmat(spike_file)['cluster_class']
        units = np.unique(cluster_class[:, 0])
        if len(units) > 1:
            for unit in range(1, len(units)):
                spike_times = cluster_class[:, 1][np.where(cluster_class[:, 0]==unit)[0]]
                n_spikes = len(spike_times)
                fr = n_spikes / (session_dur * 1e-3) # spikes/s
                if (n_spikes>n_spike_thresh) & (fr>fr_thresh):
                    spikes.append([subj_sess, subj, sess, chan, unit,
                                   sess_stop_time, spike_times, n_spikes, fr])
    cols = ['subj_sess', 'subj', 'sess', 'chan', 'unit',
            'sess_stop_time', 'spike_times', 'n_spikes', 'fr']
    spikes = pd.DataFrame(spikes, columns=cols)
    # Round spike times to the nearest ms.
    spikes['spike_times'] = spikes['spike_times'].apply(lambda x: np.rint(x).astype(np.int))

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
    
    # Save spikes.
    if save_output:
        # Split files will save one file per neuron (as a pandas Series);
        # otherwise all neurons in a session are saved in a DataFrame together.
        if split_files:
            output_f = op.join(proj_dir, 'analysis', 'spikes', '{}-CSC{}-unit{}-spikes.pkl')
            for idx, row in spikes.iterrows():
                dio.save_pickle(row, output_f.format(row['subj_sess'], row['chan'], row['unit']), verbose)
        else:
            dio.save_pickle(spikes, output_f, verbose)
    
    return spikes


def roi_lookup(subj_sess,
               chan,
               proj_dir='/home1/dscho/projects/time_cells'):
    """Given a channel, return the region."""
<<<<<<< HEAD
    mont = get_montage(subj_sess, proj_dir=proj_dir)
    for hemroi, chans in mont.items():
        if np.int(chan) in chans:
            return hemroi
    return ''


def get_montage(subj_sess,
                proj_dir='/home1/dscho/projects/time_cells'):
    """Return the montage for a given session."""
=======
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
    subj, sess = subj_sess.split('_')
    elec_montage_f = op.join(proj_dir, 'data', subj, sess, 'micro_lfps', 'anatleads.txt')
    with open(elec_montage_f, 'r') as f:
        mont = f.readlines()
    mont = [line.strip('\n').split(', ') for line in mont]
<<<<<<< HEAD
    mont = pd.Series(od([(x[0].replace('-', '_'),
                          np.arange(int(x[1].split('-')[0]), int(x[1].split('-')[1])+1))
                         for x in mont]))
    return mont
=======
    mont = od([(x[0], np.arange(int(x[1].split('-')[0]), int(x[1].split('-')[1])+1))
               for x in mont])
    for hemroi, chans in mont.items():
        if np.int(chan) in chans:
            return hemroi
    return ''
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185


def roi_mapping(n=3):
    """Return a dictionary of region acronyms to aggregate ROIs.

    Parameters
    ----------
    n : int
        How many regions to map to.
        For n=2, regions are MTL (hippocampus, amygdala, EC,
        parahippocampus), and cortex (all other).
        For n=3, regions are hippocampus, MTL (amygdala, EC, 
        parahippocampus), and cortex (all other).
        For n=4, regions are hippocampus, MTL (amygdala, EC, 
        parahippocampus), frontal, and cortex (all other).
        For n=5, regions are hippocampus, MTL (amygdala, EC,
        parahippocampus), frontal, temporal (inferior/lateral),
        and cortex (all other).
    """
    if n == 2:
        rois = {'A': 'MTL',
                'AC': 'Cortex',
                'AH': 'MTL',
                'AI': 'Cortex',
                'EC': 'MTL',
                'FOp': 'Cortex',
                'FOP': 'Cortex',
                'FSG': 'Cortex',
                'HGa': 'Cortex',
                'MFG': 'Cortex',
                'MH': 'MTL',
                'O': 'Cortex',
                'OF': 'Cortex',
                'PHG': 'MTL',
                'PI-SMG': 'Cortex',
                'pSMA': 'Cortex',
                'TO': 'Cortex',
                'TP': 'Cortex',
                'TPO': 'Cortex'}
    elif n == 3:
        rois = {'A': 'MTL',
                'AC': 'Cortex',
                'AH': 'Hippocampus',
                'AI': 'Cortex',
                'EC': 'MTL',
                'FOp': 'Cortex',
                'FOP': 'Cortex',
                'FSG': 'Cortex',
                'HGa': 'Cortex',
                'MFG': 'Cortex',
                'MH': 'Hippocampus',
                'O': 'Cortex',
                'OF': 'Cortex',
                'PHG': 'MTL',
                'PI-SMG': 'Cortex',
                'pSMA': 'Cortex',
                'TO': 'Cortex',
                'TP': 'Cortex',
                'TPO': 'Cortex'}
    elif n == 4:
        rois = {'A': 'MTL',
<<<<<<< HEAD
                'AC': 'PFC',
                'AH': 'HPC',
                'AI': 'TPC',
                'EC': 'MTL',
                'FOp': 'PFC',
                'FOP': 'PFC',
                'FSG': 'TPC',
                'HGa': 'TPC',
                'MFG': 'PFC',
                'MH': 'HPC',
                'O': 'TPC',
                'OF': 'PFC',
                'PHG': 'MTL',
                'PI-SMG': 'PFC',
                'pSMA': 'PFC',
                'TO': 'TPC',
                'TP': 'TPC',
                'TPO': 'TPC'}
    elif n == 5:
        rois = {'A': 'A',
                'AC': 'PFC',
                'AH': 'HPC',
                'AI': 'O',
                'AO': 'O',
                'AT': 'O',
                'EC': 'EC',
                'FOp': 'PFC',
                'FOP': 'PFC',
                'FSG': 'O',
                'HGa': 'O',
                'I': 'O',
                'IP': 'O',
                'IPO': 'O',
                'MC': 'PFC',
                'MFG': 'PFC',
                'MH': 'HPC',
                'MI': 'O',
                'mOCC': 'O',
                'MO': 'O',
                'MST': 'O',
                'O': 'O',
                'OF': 'PFC',
                'PC': 'O',
                'PG': 'O',
                'PH': 'HPC',
                'PI': 'O',
                'PHG': 'O',
                'PI-SMG': 'PFC',
                'PI_SMG': 'PFC',
                'PO': 'O',
                'POp': 'O',
                'POSTI': 'O',
                'PS': 'O',
                'PST': 'O',
                'pSMA': 'PFC',
                'SAO': 'O',
                'SMA': 'PFC',
                'ST': 'O',
                'STS': 'O',
                'TO': 'O',
                'TP': 'O',
                'TPO': 'O'}
    elif n == 6:
        rois = {'A': 'A',
                'AC': 'mPFC',
                'AH': 'HPC',
                'AI': 'O',
                'AO': 'O',
                'AT': 'O',
                'EC': 'EC',
                'FOp': 'O',
                'FOP': 'O',
                'FSG': 'PHG-FSG',
                'HGa': 'O',
                'I': 'O',
                'IP': 'O',
                'IPO': 'O',
                'MC': 'O',
                'MFG': 'O',
                'MH': 'HPC',
                'MI': 'O',
                'mOCC': 'O',
                'MO': 'O',
                'MST': 'O',
                'O': 'O',
                'OF': 'mPFC',
                'PC': 'O',
                'PG': 'PHG-FSG',
                'PH': 'HPC',
                'PI': 'O',
                'PHG': 'PHG-FSG',
                'PI-SMG': 'O',
                'PI_SMG': 'O',
                'PO': 'O',
                'POp': 'O',
                'POSTI': 'O',
                'PS': 'O',
                'PST': 'O',
                'pSMA': 'mPFC',
                'SAO': 'O',
                'SMA': 'O',
                'ST': 'O',
                'STS': 'O',
                'TO': 'O',
                'TP': 'O',
                'TPO': 'O'}
    elif n == 7:
        rois = {'A': 'A',
                'AC': 'PFC',
                'AH': 'HPC',
                'AI': 'O',
                'AT': 'LTC',
                'EC': 'EC',
                'FOp': 'PFC',
                'FOP': 'PFC',
                'FSG': 'ITC',
                'HGa': 'LTC',
                'I': 'O',
                'IP': 'O',
                'MFG': 'PFC',
                'MH': 'HPC',
                'MI': 'O',
                'O': 'O',
                'OF': 'PFC',
                'PC': 'O',
                'PHG': 'ITC',
                'PI-SMG': 'PFC',
                'PI_SMG': 'PFC',
                'POp': 'O',
                'POSTI': 'O',
                'pSMA': 'PFC',
                'STS': 'LTC',
                'TO': 'LTC',
                'TP': 'LTC',
                'TPO': 'LTC'}
    elif n == 8:
        rois = {'A'     : 'AMY',
                'AC'    : 'CC',
                'AH'    : 'HPC',
                'AI'    : 'Other',
                'AO'    : 'Other',
                'AT'    : 'LTC',
                'EC'    : 'EC',
                'FOp'   : 'Other',
                'FOP'   : 'Other',
                'FSG'   : 'PHG',
                'HGa'   : 'LTC',
                'I'     : 'Other',
                'IP'    : 'Other',
                'IPO'   : 'Other',
                'MC'    : 'CC',
                'MFG'   : 'Other',
                'MH'    : 'HPC',
                'MI'    : 'Other',
                'mOCC'  : 'Other',
                'MO'    : 'Other',
                'MST'   : 'LTC',
                'O'     : 'Other',
                'OF'    : 'OFC',
                'PC'    : 'CC',
                'PG'    : 'PHG',
                'PH'    : 'HPC',
                'PI'    : 'Other',
                'PHG'   : 'PHG',
                'PI-SMG': 'Other',
                'PI_SMG': 'Other',
                'PO'    : 'Other',
                'POp'   : 'Other',
                'POSTI' : 'Other',
                'PS'    : 'LTC',
                'PST'   : 'LTC',
                'pSMA'  : 'Other',
                'SAO'   : 'Other',
                'SMA'   : 'Other',
                'ST'    : 'LTC',
                'STS'   : 'LTC',
                'TO'    : 'Other',
                'TP'    : 'LTC',
                'TPO'   : 'LTC'}
=======
                'AC': 'Frontal',
                'AH': 'Hippocampus',
                'AI': 'Cortex',
                'EC': 'MTL',
                'FOp': 'Frontal',
                'FOP': 'Frontal',
                'FSG': 'Cortex',
                'HGa': 'Cortex',
                'MFG': 'Frontal',
                'MH': 'Hippocampus',
                'O': 'Cortex',
                'OF': 'Frontal',
                'PHG': 'MTL',
                'PI-SMG': 'Frontal',
                'pSMA': 'Frontal',
                'TO': 'Cortex',
                'TP': 'Cortex',
                'TPO': 'Cortex'}
    elif n == 5:
        rois = {'A': 'MTL',
                'AC': 'Frontal',
                'AH': 'Hippocampus',
                'AI': 'Cortex',
                'EC': 'MTL',
                'FOp': 'Frontal',
                'FOP': 'Frontal',
                'FSG': 'Temporal',
                'HGa': 'Temporal',
                'MFG': 'Frontal',
                'MH': 'Hippocampus',
                'O': 'Cortex',
                'OF': 'Frontal',
                'PHG': 'MTL',
                'PI-SMG': 'Frontal',
                'pSMA': 'Frontal',
                'TO': 'Temporal',
                'TP': 'Temporal',
                'TPO': 'Temporal'}
    elif n == 6:
        rois = {'A': 'A',
                'AC': 'Frontal',
                'AH': 'Hippocampus',
                'AI': 'Cortex',
                'EC': 'EC',
                'FOp': 'Frontal',
                'FOP': 'Frontal',
                'FSG': 'Temporal',
                'HGa': 'Temporal',
                'MFG': 'Frontal',
                'MH': 'Hippocampus',
                'O': 'Cortex',
                'OF': 'Frontal',
                'PHG': 'Temporal',
                'PI-SMG': 'Frontal',
                'pSMA': 'Frontal',
                'TO': 'Temporal',
                'TP': 'Temporal',
                'TPO': 'Temporal'}
    elif n == 7:
        rois = {'A': 'A',
                'AC': 'Prefrontal',
                'AH': 'Hippocampus',
                'AI': 'Cortex',
                'EC': 'EC',
                'FOp': 'Prefrontal',
                'FOP': 'Prefrontal',
                'FSG': 'Inf. temporal',
                'HGa': 'Lat. temporal',
                'MFG': 'Prefrontal',
                'MH': 'Hippocampus',
                'O': 'Cortex',
                'OF': 'Prefrontal',
                'PHG': 'Inf. temporal',
                'PI-SMG': 'Prefrontal',
                'pSMA': 'Prefrontal',
                'TO': 'Lat. temporal',
                'TP': 'Lat. temporal',
                'TPO': 'Lat. temporal'}
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
    else:
        raise ValueError('n={} map regions has not been implemented'.format(n))

    return rois


def shift_spike_inds(spike_inds, 
                     step, 
                     floor=0, 
                     ceiling=np.inf):
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
    
    Converts spike times into one or more spike trains
    that are convolved with a 1D Gaussian filter.
    
    Time is assumed to start at 0 ms. Each value 
    in the output vector corresponds to 1 ms. 
    
    If simga == 0, returns the spike train instead of firing rate.

    Parameters
    ----------
    spike_times : np.ndarray
        An vector or matrix of integer spike times in ms. 
        Last dimension of the array is the one that will be smoothed.
    stop_time : int
        Stop time for the session in ms (start time is 0 ms).
    sigma : int
        Standard deviation of the Gaussian kernel in ms.

    Returns
    -------
    fr = np.ndarray
        An array whose last dimension length equals the stop time.
        Values are firing rates in spikes/s. 
    """    
    # Convolve the spike train with a Gaussian.
    shp = spike_times.shape
    if len(shp) == 1:
        spike_train = spike_times_to_train(spike_times, stop_time)
    elif len(shp) == 2:
        spike_train = np.array([spike_times_to_train(spike_times[x, :], stop_time)
                                for x in range(shp[0])])
    if sigma > 0:
        fr = gaussian_filter1d(spike_train, sigma, mode='reflect', axis=-1) * 1e3  # convert ms to s
        return fr
    else:
        return spike_train


def spike_times_to_train(spike_times,
                         stop_time):
    """Convert spike times into a spike train vector of 0s and 1s."""
    spike_train = np.zeros(stop_time)
    spike_train[spike_times] = 1
    return spike_train
