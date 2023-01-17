"""
spike_sorting.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Functions for reading and formatting wave_clus files
when reviewing spike sorting results.

Last Edited
----------- 
10/5/20
"""
<<<<<<< HEAD
import os.path as op
from collections import OrderedDict as od
# import mkl
# mkl.set_num_threads(1)
=======
import os
from collections import OrderedDict as od
import mkl
mkl.set_num_threads(1)
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
import numpy as np
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.formatter.offset_threshold'] = 2
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['figure.subplot.wspace'] = 0.25 
mpl.rcParams['figure.subplot.hspace'] = 0.25 

n_ = 4
c_ = 2
colors = [sns.color_palette('Blues', n_)[c_], 
          sns.color_palette('Reds', n_)[c_], 
          sns.color_palette('Greens', n_)[c_],
          sns.color_palette('Purples', n_)[c_],
          sns.color_palette('Oranges', n_)[c_],
          sns.color_palette('Greys', n_)[c_],
          sns.color_palette('YlOrBr', n_+3)[c_],
          'k']


def compile_wave_clus(subj_sess,
<<<<<<< HEAD
                      proj_dir='/data7/goldmine',
=======
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
                      verbose=True):
    """Return wave_clus output dict for the session.

    Returns
    -------
    wave_clus : OrderedDict
        Contains the wave_clus outputs for a neuron.
    mont : OrderedDict
         Implanted regions and their corresponding electrode channels.
    """
    # Parameters.
    subj, sess = subj_sess.split('_')
<<<<<<< HEAD
    subj_dir = op.join(proj_dir, 'data', subj, sess)
    lfp_dir = op.join(subj_dir, 'micro_lfps')
    spikes_dir = op.join(subj_dir, 'spikes', 'wave_clus3_sortbyhand')
    lfp_dir = op.join(subj_dir, 'micro_lfps')
    lfp_timestamps_f = op.join(lfp_dir, 'lfpTimeStamps_micro.mat')
    elec_montage_f = op.join(lfp_dir, 'anatleads.txt')
    assert op.exists(spikes_dir)
    assert op.exists(lfp_timestamps_f)
    assert op.exists(elec_montage_f)
=======
    subj_dir = os.path.join('/Users/danielschonhaut/penn/lab/projects/time_cells/data', subj, sess)
    lfp_dir = os.path.join(subj_dir, 'micro_lfps')
    spikes_dir = os.path.join(subj_dir, 'spikes', 'wave_clus3_sortbyhand')
    lfp_dir = os.path.join(subj_dir, 'micro_lfps')
    lfp_timestamps_f = os.path.join(lfp_dir, 'lfpTimeStamps_micro.mat')
    elec_montage_f = os.path.join(lfp_dir, 'anatleads.txt')
    assert os.path.exists(spikes_dir)
    assert os.path.exists(lfp_timestamps_f)
    assert os.path.exists(elec_montage_f)
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
    
    # Load LFP timestamps.
    lfp_timestamps, sr = load_lfp_timestamps(lfp_timestamps_f=lfp_timestamps_f, 
                                             verbose=verbose)
    session_dur_ms = lfp_timestamps[-1] - lfp_timestamps[0]

    # Load the electrode montage.
    mont = load_montage(elec_montage_f, verbose)
    rois = list(mont.keys())
    
    # Load wave_clus files and store the outputs
    # for each neuron in a dict.
    wave_clus = od([])
    for roi in rois:  
        chans = mont[roi]
        for chan in chans:
<<<<<<< HEAD
            wave_clusf = op.join(spikes_dir, 'times_CSC{}.mat'.format(chan))
            if not op.exists(wave_clusf):
=======
            wave_clusf = os.path.join(spikes_dir, 'times_CSC{}.mat'.format(chan))
            if not os.path.exists(wave_clusf):
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
                continue
            cluster_class = loadmat(wave_clusf)['cluster_class']
            neurons, spike_inds = np.unique(cluster_class[:, 0], return_inverse=True)
            neurons = np.array([x for x in neurons if x!=0])
            if len(neurons) > 0:
                waveforms = loadmat(wave_clusf)['spikes']
                for iNeuron, neuron in enumerate(neurons):
                    spike_inds_ = np.where(spike_inds==neuron)[0]
                    spike_times = cluster_class[spike_inds_, 1]
                    fr = len(spike_inds_) / (session_dur_ms * 1e-3)
                    wave_clus['{}_ch{}_{}'.format(roi, chan, iNeuron+1)] = od([('spike_times', spike_times),  # ms
                                                                               ('fr', fr),  # Hz
                                                                               ('waveforms', waveforms[spike_inds_, :])])  # spike x time
    neurons = np.array(list(wave_clus.keys()))

    if verbose:
        print('{} neurons'.format(len(wave_clus)))
    
    return wave_clus, neurons, lfp_timestamps, sr, session_dur_ms, mont, rois


def load_lfp_timestamps(subj_sess=None,
                        lfp_timestamps_f=None,
                        conv_factor='infer',
                        start_at_zero=True,
                        proj_dir='/home1/dscho/projects/time_cells',
                        verbose=True):
    """Return the LFP timestamps vector and sampling rate.
    
    LFP timestamps are returned in ms. If conv_factor=='infer', the
    function will infer whether the original data were stored in s or
    ms by examining the sampling rate. If conv_factor!='infer', the
    function caller must specify that number that lfp_timestamps must
    be multiplied by to be returned in ms.

    Either subj_sess (e.g. 'U518_ses1') must be given and the location 
    of the lfp_timestamps file is found, or lfp_timestamps_f is given,
    in which case subj_sess is disregarded.
    """
    if lfp_timestamps_f is None:
        subj, sess = subj_sess.split('_')
<<<<<<< HEAD
        lfp_timestamps_f = op.join(proj_dir, 'data', subj, sess,
                                   'micro_lfps', 'lfpTimeStamps_micro.mat')
    if op.exists(lfp_timestamps_f):
=======
        lfp_timestamps_f = os.path.join(proj_dir, 'data', subj, sess,
                                        'micro_lfps', 'lfpTimeStamps_micro.mat')
    if os.path.exists(lfp_timestamps_f):
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
        try:
            lfp_timestamps = np.squeeze(loadmat(lfp_timestamps_f)['timeStamps'])
        except NotImplementedError:
            with h5py.File(lfp_timestamps_f, 'r') as f:
                lfp_timestamps = np.squeeze(f['timeStamps'])
        
        sr_ = np.int(np.rint(1000 / np.mean(np.diff(lfp_timestamps))))
        if conv_factor == 'infer':
            if sr_ > 5e4:
                conv_factor = 1e3
            else:
                conv_factor = 1
        lfp_timestamps *= conv_factor

        if start_at_zero:
            lfp_timestamps -= lfp_timestamps[0]

        sr = np.int(np.rint(1000 / np.mean(np.diff(lfp_timestamps))))
        dur = lfp_timestamps[-1] - lfp_timestamps[0]
        if verbose:
            print('Conversion factor is {}'.format(conv_factor))
            print('{} timestamps over {} min and {:.1f} s'.format(len(lfp_timestamps), 
                                                                  np.int(dur / 6e4), 
                                                                  (dur % 6e4) * 1e-3))
            print('Sampling rate is {} Hz'.format(sr))
        return lfp_timestamps, sr
    else:
        print('File does not exist.')
        return None


def load_montage(elec_montage_f,
                 verbose=True):
    """Return an OrderedDict of channels for each region."""
    with open(elec_montage_f, 'r') as f:
        mont = f.readlines()
    mont = [line.strip('\n').split(', ') for line in mont]
    mont = od([(x[0], np.arange(int(x[1].split('-')[0]), int(x[1].split('-')[1])+1))
               for x in mont])
    
    if verbose:
        rois = list(mont.keys())
        n_chans = mont[rois[-1]][-1]
        print('{} electrodes in {} regions.'.format(n_chans, len(rois)))
        for k, v in mont.items():
            print('{}: {}'.format(k, v))
            
    return mont


def plot_isi_hist(neuron_dat,
                  ax,
                  isi_thresh=3,
                  bins=60,
                  xscale='log',
                  color='k',
                  label_axes=True,
                  annot_xy='infer', # otherwise (x, y) fractions
                  fontsize={'tick': 12, 'label': 14, 'annot': 14}):
    """Plot a log-histrogram of inter-spike intervals.
    
    Limits are from 1 ms to 100 s.
    
    Parameters
    ----------
    neuron_dat : dict that must contain
        1. 'spike_times' : np.ndarray
           A vector of spike times in ms.
           
    Returns
    -------
    The matplotlib axis.
    """
    isis = np.diff(neuron_dat['spike_times'])
    if xscale == 'log':
        ax.hist(isis, bins=10**np.linspace(0, 5, bins+1), 
                color=color, edgecolor=color)
        ax.set_xscale('log')
    elif xscale == 'wave_clus':
        ax.hist(isis, bins=np.linspace(0, 1e2, bins+1), 
                color=color, edgecolor=color)
    elif xscale == 'linear':
        ax.hist(isis, bins=bins+1, color=color, edgecolor=color)

    ax.tick_params(axis='both', which='major', labelsize=fontsize['tick'])
    if label_axes:
        ax.set_xlabel('ISI time (ms)', fontsize=fontsize['label'], labelpad=8)
        ax.set_ylabel('No. spikes', fontsize=fontsize['label'], labelpad=8)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')

    if annot_xy == 'infer':
        if (xscale=='log') & (np.median(isis) > 316):
            annot_xy = (0.1, 0.9)
        else:
            annot_xy = (0.65, 0.9)
    
    ax.annotate('{:,d} ISI\nviolations'.format(np.sum(isis<isi_thresh)),
                annot_xy, xycoords='axes fraction', fontsize=fontsize['annot'])
    
    return ax


def plot_neuron3(wave_clus, 
                 neuron, 
                 sr, 
                 session_dur_ms,
                 isi_xscale='log',
                 fontsize=None,
                 fig=None,
                 ax=None):
    """Generate three plots for the neuron.
    
    Spike waveforms, inter-spike intervals,
    and spike stability.
    
    Uses default parameters from the functions called.
    
    Parameters
    ----------
    wave_clus : dict
        Created by compile_wave_clus()
    neuron : str
        Key to spikes.
        
    Returns
    -------
    fig, ax
    """
    neuron_dat = wave_clus[neuron]

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        ax = np.ravel(ax)
    if fontsize is None:
        fontsize = od([('tick', 12),
                       ('label', 14),
                       ('annot', 12),
                       ('fig', 16)])

    i = 0
    ax[i] = plot_waveforms(neuron_dat, sr=sr, ax=ax[i], fontsize=fontsize)
    ax[i].set_title(neuron, fontsize=fontsize['fig'], y=1.02, loc='left')

    i = 1
    ax[i] = plot_isi_hist(neuron_dat, ax=ax[i], xscale=isi_xscale,
                          color=colors[0], fontsize=fontsize)

    i = 2
    ax[i] = plot_spike_stability(neuron_dat, 
                                 session_dur_ms, 
                                 ax[i],
                                 fontsize=fontsize)

    fig.tight_layout(w_pad=1.12)
    
    return fig, ax


def plot_spike_stability(neuron_dat,
                         session_dur_ms,
                         ax,
                         color='k',
                         label_axes=True,
                         fontsize={'tick': 12, 'label': 14}):
    """Plot a histrogram of spikes over time.
    
    From start to end of the recording session.
    
    Parameters
    ----------
    neuron_dat : dict that must contain
        1. 'spike_times' : np.ndarray
           A vector of spike times in ms.
           
    Returns
    -------
    The matplotlib axis.
    """
    spike_times = neuron_dat['spike_times']
    ax.hist(spike_times, bins=np.linspace(0, session_dur_ms, 61), 
            color=color, edgecolor=color)
    xticks = np.arange(0, session_dur_ms, 6e5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(10 * np.arange(len(xticks)))
    ax.tick_params(axis='both', which='major', labelsize=fontsize['tick'])
    if label_axes:
        ax.set_xlabel('Session time (min)', fontsize=fontsize['label'], labelpad=8)
        ax.set_ylabel('No. spikes', fontsize=fontsize['label'], labelpad=8)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    return ax


def plot_waveforms(neuron_dat,
                   sr,
                   ax,
                   samp_spikes=200,
                   color='tomato',
                   label_axes=True,
                   annot_xy=(0.5, 0.9),
                   fontsize={'tick': 12, 'label': 14, 'annot': 14}):
    """Plot a neuron's spike waveform.
    
    Plots the mean spike waveform, standard deviation
    error lines, and samp_spikes individual spikes
    sampled evenly across the recording session.
    
    Parameters
    ----------
    neuron_dat : dict that must contain
        1. 'waveforms' : np.ndarray
           spikes x timepoints.
        2. 'fr' : int or float
           The neuron's firing rate in Hz.
    sr : int or float
        The sampling rate in Hz.
           
    Returns
    -------
    The matplotlib axis.
    """
    mean_waveform = np.mean(neuron_dat['waveforms'], 0)
    std_waveform = np.std(neuron_dat['waveforms'], 0)
    fr = neuron_dat['fr']
    n_spikes = neuron_dat['waveforms'].shape[0]
    n_samples = len(mean_waveform)
    x_units = np.arange(n_samples) / (sr/1000)  # ms
    if n_spikes > samp_spikes:
        waves = neuron_dat['waveforms'][np.linspace(0, n_spikes-1, samp_spikes, dtype=np.int), :]
    else:
        waves = neuron_dat['waveforms']
    
    for iSpike in range(waves.shape[0]):
        ax.plot(x_units, waves[iSpike, :], color=color, alpha=0.67, linewidth=0.6)
    ax.plot(x_units, mean_waveform - std_waveform, color='k', linewidth=1.2, linestyle='--')
    ax.plot(x_units, mean_waveform + std_waveform, color='k', linewidth=1.2, linestyle='--')
    ax.plot(x_units, mean_waveform, color='k', linewidth=2)
    ax.set_xticks(np.arange(0, x_units[-1], 0.5))
    ax.tick_params(axis='both', which='major', labelsize=fontsize['tick'])
    if label_axes:
        ax.set_xlabel('Time (ms)', fontsize=fontsize['label'], labelpad=8)
        ax.set_ylabel('$\mu$V', fontsize=fontsize['label'], labelpad=4)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.annotate('{:,d} spikes\n({:.1f}Hz)'.format(n_spikes, fr), 
                annot_xy, xycoords='axes fraction', fontsize=fontsize['annot'])
    return ax
