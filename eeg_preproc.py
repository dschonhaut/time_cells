"""
eeg_preproc.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Functions for reading and processing time domain EEG.

Last Edited
----------- 
1/26/22
"""
import sys
import os
import os.path as op
from collections import OrderedDict as od
import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
import xarray
import mne
from scipy import stats
from scipy import signal
from scipy.io import loadmat
import h5py
sys.path.append('/home1/dscho/code/general')
import data_io as dio
from helper_funcs import Timer
sys.path.append('/home1/dscho/code/projects')
from time_cells import spike_preproc, events_proc, spectral_analysis


def process_eeg(subj_sess,
                chan,
                eeg=None,
                sr=None,
                convert_v_to_muv=False,
                downsample_to=None,
                l_freq=None,
                h_freq=None,
                notch_freqs=None,
                save_output=False,
                overwrite=False,
                data_dir='/data7/goldmine/data',
                output_dir=None,
                verbose=False):
    """Load, process, and save EEG data for a single channel.
    
    Will (1) load the input EEG; (2) convert volts to microvolts; 
    (3) downsample the signal, (4) high-pass/low-pass/band-pass filter,
    and (5) notch filter, in that order.
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U518_ses0'
    chan : int | str
        The number of the EEG channel to load.
    eeg : array | None
        Input EEG; if None it will be imported from the presumed
        input filepath (see data_dir).
    sr : float | None
        Sampling rate of the input EEG.
    convert_v_to_muv : bool
        If True, multiplies EEG values by 1e6 to convert volts to
        microvolts.
    downsample_to : float | None
        The rate, in Hz, to downsample the data to.
    l_freq, h_freq : floats
        The frequencies below and above which to filter out of the
        data, respectively.
            * l_freq < h_freq: band-pass filter
            * l_freq > h_freq: band-stop filter
            * l_freq is not None and h_freq is None: high-pass filter
            * l_freq is None and h_freq is not None: low-pass filter
    notch_freqs : float | array of float | None
        Frequencies to notch filter.
    data_dir : str
        Directory where the input data are stored (expected file
        structure is: [data_dir]/subj/sess/micro_lfps/CSC[chan].mat).
    output_dir : str
        Directory where the processed data are saved. If None, the
        output file will be saved in a new directory like:
        [data_dir]/subj/sess/micro_lfps/[output_tag]/CSC[chan].pkl,
        where output_tag contains information about the preprocessing
        parameters that were used.

    Returns
    -------
    eeg : array
        A vector over time.
    sr : float
        The sampling rate, as returned.
    """
    subj, sess = subj_sess.split('_')
    
    # Get the output filename.
    if output_dir is None:
        output_tag = ''
        if convert_v_to_muv:
            output_tag += '_V-to-muV'
        if downsample_to is not None:
            output_tag += '_sr{}'.format(downsample_to)
        if (l_freq is not None) and (h_freq is None):
            output_tag += '_highpass{}'.format(l_freq)
        elif (l_freq is None) and (h_freq is not None):
            output_tag += '_lowpass{}'.format(h_freq)
        elif (l_freq is not None) and (h_freq is not None):
            output_tag += '_bandpass{}-{}'.format(l_freq, h_freq)
        if notch_freqs is not None:
            try:
                output_tag += '_notch' + '-'.join([str(x) for x in notch_freqs])
            except TypeError:
                output_tag += '_notch' + str(notch_freqs)
        output_tag = output_tag[1:]
        output_dir = op.join(data_dir, subj, sess, 'micro_lfps', output_tag)
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
    output_f = op.join(output_dir, 'CSC{}.pkl'.format(chan))

    # Return the processed EEG data if it already exists.
    if op.exists(output_f) and not overwrite:
        eeg, sr = dio.open_pickle(output_f)
        return eeg, sr
    
    # Load the raw EEG channel file.
    if (eeg is None) or (sr is None):
        if verbose:
            print('Loading {} chan{} raw EEG'.format(subj_sess, chan))
        chan_f = op.join(data_dir, subj, sess, 'micro_lfps',
                         'CSC{}.mat'.format(chan))
        try:
            dat = loadmat(chan_f)
            eeg = np.squeeze(dat['data'])
            sr = (1 / dat['samplingInterval'][0][0]) * 1e3
        except NotImplementedError:
            with h5py.File(chan_f, 'r') as dat:
                eeg = np.squeeze(dat['data'])
                sr = (1 / dat['samplingInterval'][0][0]) * 1e3
        eeg = eeg.astype(np.float64)
    
    # Convert values to microvolts.
    if convert_v_to_muv:
        eeg *= 1e6

    # Downsample the signal.
    if downsample_to is not None:
        if verbose:
            print('Downsampling from {} to {} Hz'.format(sr, downsample_to))
        downsample_by = sr / downsample_to
        eeg = signal.resample(eeg, num=int(eeg.size / downsample_by))
        sr /= downsample_by

    # Lowpass/highpass/bandpass filter.
    if (l_freq is not None) or (h_freq is not None):
        if verbose:
            if (l_freq is not None) & (h_freq is None):
                print('High-pass filtering above {} Hz'.format(l_freq))
            elif (l_freq is None) & (h_freq is not None):
                print('Low-pass filtering below {} Hz'.format(h_freq))
            else:
                print('Band-pass filtering from {}-{} Hz'.format(l_freq, h_freq))
        eeg = mne.filter.filter_data(eeg, sr, l_freq=l_freq, h_freq=h_freq,
                                     copy=False, verbose=verbose)
        
    # Notch filter.
    if notch_freqs is not None:
        if verbose:
            print('Notch filtering at {} Hz'.format(notch_freqs))
        eeg = mne.filter.notch_filter(eeg, sr, notch_freqs,
                                      copy=False, verbose=verbose)
        
    # Save the data.
    if save_output:
        dio.save_pickle((eeg, sr), output_f, verbose)
        
    return eeg, sr


def load_time_eeg(subj_sess,
                  mont=None,
                  regions=None,
                  sr=1000,
                  l_freq=0.1,
                  h_freq=80,
                  notch_freqs=[60],
                  chan_exclusion_thresh=0,
                  verbose=False,
                  **kws):
    """Return a channel x time DataFrame grouped by region, for the whole session.
    
    Loads processed EEG traces according to the inputs provided. If regions
    is None, all regions are returned.
    
    Channels are excluded that deviate by more than <chan_exclusion_thresh>
    from the mean, Z-scored power across channels and frequencies, calculated
    separately within each region.
    
    kws are passed to process_eeg()
    
    Returns
    -------
    time_eeg : OrderedDict of DataFrames shaped channel x time
        Each dict entry corresponds to all channels from one
        microwire bundle.
    """
    if verbose:
        timer = Timer()
        print(subj_sess, '-' * len(subj_sess), sep='\n')
        
    # Load the electrode montage.
    if mont is None:
        mont = spike_preproc.get_montage(subj_sess)
    if regions is None:
        regions = np.sort(list(mont.keys()))
    else:
        regions = np.sort([x for x in mont.keys()
                           if x in np.atleast_1d(regions)])
        
    n_chans = 0
    n_chans_kept = 0
    time_eeg = od([])
    for roi in regions:
        chans = mont[roi]
        
        # Load processed channels.
        eeg = []
        for chan in chans:
            _eeg, _ = process_eeg(subj_sess,
                                  chan=chan,
                                  downsample_to=sr,
                                  l_freq=l_freq,
                                  h_freq=h_freq,
                                  notch_freqs=notch_freqs,
                                  **kws)
            eeg.append(_eeg.tolist())
        eeg = np.array(eeg)
        n_chans += len(chans)

        # Exclude bad channels.
        if chan_exclusion_thresh > 0:
            freqs, powers = spectral_analysis.timefreq_welch(eeg, sr, fmin=1, fmax=30)
            zpowers = stats.zscore(powers, axis=0)
            keep_iChans = np.where(np.abs(np.mean(zpowers, axis=1)) <= chan_exclusion_thresh)[0]
            keep_chans = chans[keep_iChans]
            eeg = eeg[keep_iChans, :]
            if verbose and (len(chans) != len(keep_chans)):
                exclude_chans = [x for x in chans if x not in keep_chans]
                exclude_iChans = [x for x in np.arange(len(chans)) if x not in keep_iChans]
                print('Excluded {} channel {} (index {})'
                      .format(roi,
                              ', '.join([str(x) for x in exclude_chans]),
                              ', '.join([str(x) for x in exclude_iChans])))
        else:
            keep_chans = chans
        n_chans_kept += len(keep_chans)

        # Make the output dataframe.
        time_eeg[roi] = pd.DataFrame(eeg, index=keep_chans, columns=np.arange(eeg.shape[1]))
    
    if verbose:
        print('Kept {}/{} ({:.0%}) channels across {} regions'
              .format(n_chans_kept, n_chans, n_chans_kept/n_chans, len(time_eeg)))
        print(timer, end='\n'*2)
        
    return time_eeg


def load_event_eeg(subj_sess,
                   regions=None,
                   game_states=['Encoding', 'Retrieval'],
                   buffer=0,
                   l_freq=0.1,
                   h_freq=80,
                   notch_freqs=[60],
                   chan_exclusion_thresh=0,
                   verbose=False,
                   **kws):
    """Return event-epoched EEG for all channels in each region indicated.
    
    Loads processed EEG traces according to the inputs provided. Note:
    currently only supports epoching EEG that was saved at 1000 Hz.
    
    kws are passed to process_eeg()
    
    Parameters
    ----------
    subj_sess : str
        E.g. 'U518_ses0'
    regions : str | list | None
        One or more regions to load EEG for. If None,
        all regions from the montage are returned.
    
    Returns
    -------
    event_eeg : OrderedDict of DataArrays, shape=(gameState, trial, channel, time)
    """
    if verbose:
        timer = Timer()
        print(subj_sess, '-' * len(subj_sess), sep='\n')
    
    sr = 1000
    game_states = np.atleast_1d(game_states)
    
    # Load events.
    event_times = events_proc.load_events(subj_sess, verbose=False).event_times
    gs_durs = events_proc.get_game_state_durs()
    
    # Get start and stop times for each event.
    event_idx = od([])
    for game_state in game_states:
        _event_times = event_times.query("(gameState=='{}')".format(game_state))
        event_idx[game_state] = _event_times.apply(lambda x: [x['time'][0], x['time'][0] +
                                                              gs_durs[game_state]], axis=1).tolist()
    
    # Get the channel x time EEG for the whole session.
    time_eeg = load_time_eeg(subj_sess,
                             regions=regions,
                             sr=sr,
                             l_freq=l_freq,
                             h_freq=h_freq,
                             notch_freqs=notch_freqs,
                             **kws)
    
    # Epoch the EEG data.
    n_chans = 0
    n_chans_kept = 0
    event_eeg = od([])
    for roi, _time_eeg in time_eeg.items():
        chans = _time_eeg.index.values
        n_chans += len(chans)
        # game_state x trial x channel x time
        eeg = np.array([[_time_eeg.values[:, (start-buffer):(stop+buffer)]
                         for start, stop in event_idx[game_state]]
                        for game_state in game_states])

        # Exclude bad channels.
        if chan_exclusion_thresh > 0:
            # Get power at each frequency.
            freqs, powers = spectral_analysis.timefreq_welch(
                eeg[..., buffer:eeg.shape[-1]-buffer], sr, fmin=1, fmax=30)
            
            # Z-score power across all dimensions except frequency.
            zpowers = stats.zscore(powers, axis=(0, 1, 2))
        
            # Remove channels with mean, absolute Z-power above thresh.
            keep_iChans = np.where(np.abs(np.mean(zpowers, axis=(0, 1, 3))) <= chan_exclusion_thresh)[0]
            keep_chans = chans[keep_iChans]
            eeg = eeg[:, :, keep_iChans, :]
                
            if verbose and (len(chans) != len(keep_chans)):
                exclude_chans = [x for x in chans if x not in keep_chans]
                exclude_iChans = [x for x in np.arange(len(chans)) if x not in keep_iChans]
                print('Excluded {} channel {} (index {})'
                      .format(roi,
                              ', '.join([str(x) for x in exclude_chans]),
                              ', '.join([str(x) for x in exclude_iChans])))
        else:
            keep_chans = chans
        n_chans_kept += len(keep_chans)
    
        # Convert to xarray.
        event_eeg[roi] = xarray.DataArray(eeg,
                                          name=(subj_sess, roi),
                                          coords=[('gameState', game_states),
                                                  ('trial', _event_times['trial'].tolist()),
                                                  ('chan', keep_chans),
                                                  ('time', np.arange(eeg.shape[-1]))],
                                          dims=['gameState', 'trial', 'chan', 'time'],
                                          attrs={'sr': sr,
                                                 'buffer': buffer,
                                                 'chan_exclusion_thresh': chan_exclusion_thresh})
            
    if verbose:
        print('Kept {}/{} ({:.0%}) channels across {} regions'
              .format(n_chans_kept, n_chans, n_chans_kept/n_chans, len(time_eeg)))
        print(timer, end='\n'*2)
        
    return event_eeg
