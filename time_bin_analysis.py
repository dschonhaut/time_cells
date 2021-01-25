"""
time_bin_analysis.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com
    
Description
----------- 
Functions for analyzing firing rate data within event windows.

Last Edited
----------- 
11/27/20
"""
import sys
import os
from collections import OrderedDict as od
import warnings
import mkl
mkl.set_num_threads(1)
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
sys.path.append('/home1/dscho/code/general')
import data_io as dio


def calc_fr_by_time_bin(fr_train, 
                        event_times, 
                        game_states, 
                        n_time_bins,
                        random_shift=False):
    """Return mean firing rate within each time bin, within each event.
    
    Parameters
    ----------
    fr_train : numpy.ndarray
        Vector of firing rates over the session, in ms.
    event_times : pandas DataFrame
    game_states : list or tuple
        E.g. ['Delay1', 'Delay2']
    n_time_bins : int
        Firing rates within each event window will be evenly split into
        this many time bins.
    random_shift : bool
        If True, firing rates will be randomly, circularly shifted
        within each event window.
    
    Returns
    -------
    fr_by_time_bin : np.ndarray
        n_event x n_time_bin array with the mean firing
        rate during each time bin.
    """
    def start_stop(time_window):
        start, stop = time_window
        start = int(np.round(start, 0))
        stop = int(np.round(stop, 0))
        return start, stop
    
    def event_mean_frs(time_window, 
                       fr_train, 
                       n_time_bins, 
                       random_shift=False):
        start, stop = time_window
        fr_train_ = fr_train[start:stop]
        if random_shift:
            roll_by = int(np.random.rand() * (stop - start))
            fr_train_ = np.roll(fr_train_, roll_by)
        return [np.nanmean(x) for x in np.array_split(fr_train_, n_time_bins)]
        
    # Get time windows for the game states of interest.    
    time_wins = (event_times
                 .query("(gameState=={})".format(game_states))['time']
                 .apply(lambda x: start_stop(x)))
    
    # Get the mean firing rate in each time bin, for each event.
    fr_by_time_bin = np.array(time_wins.apply(lambda x: 
                                              event_mean_frs(x, 
                                                             fr_train, 
                                                             n_time_bins, 
                                                             random_shift))
                                       .tolist())
    return fr_by_time_bin


def calc_mean_fr_by_time(fr_by_time_bin_f,
                         proj_dir='/home1/dscho/projects/time_cells',
                         overwrite=False,
                         save_output=True,
                         verbose=True):
    """Calculate mean FR across trial phases of a given type."""
    neuron = '-'.join(os.path.basename(fr_by_time_bin_f).split('-')[:3])

    # Load the output file if it exists.
    output_f = os.path.join(proj_dir, 'analysis', 'fr_by_time_bin', 
                            '{}-mean_fr_by_time.pkl'.format(neuron))
    if os.path.exists(output_f) and not overwrite:
        return dio.open_pickle(output_f)

    subj_sess, chan, unit = neuron.split('-')
    chan = chan[3:]
    subj, sess = subj_sess.split('_')
    event_times = dio.open_pickle(fr_by_time_bin_f)['event_times']
    n_perm = event_times['fr_null'].iloc[0].shape[0]
    # game_states = [['Prepare1'], ['Delay1'], ['Encoding'], ['Prepare2'], ['Delay2'], ['Retrieval'],
    #                ['Prepare1', 'Prepare2'], ['Delay1', 'Delay2'], ['Encoding', 'Retrieval']]
    game_states = [['Delay1'], ['Delay2']]
    cols = ['subj_sess', 'subj', 'sess', 'chan', 'unit', 'gameState', 
            'fr', 'z_fr', 'z_fr_max', 'z_fr_max_ind', 'tis', 'z_tis', 'pval']
    output = []
    for game_state in game_states:
        if not np.all(np.isin(game_state, event_times['gameState'].unique())):
            continue
        obs = np.mean(event_times
                      .query("(gameState=={})".format(game_state))['fr']
                      .tolist(), axis=0)
        null_mean = np.mean(np.mean(event_times
                                    .query("(gameState=={})".format(game_state))['fr_null']
                                    .tolist(), axis=0), axis=0)
        null_std = np.std(np.mean(event_times
                                  .query("(gameState=={})".format(game_state))['fr_null']
                                  .tolist(), axis=0), axis=0)
        obs_z = (obs - null_mean) / null_std
        
        tis = info_rate(obs)  # temporal information score
        null_tis = np.array([info_rate(np.mean(event_times
                                               .query("(gameState=={})".format(game_state))['fr_null']
                                               .tolist(), axis=0)[iPerm, :])
                             for iPerm in range(n_perm)])
        tis_z = (tis - np.mean(null_tis)) / np.std(null_tis)
        pval = (np.sum(null_tis >= tis) + 1) / (n_perm + 1)
        
        output.append([subj_sess, subj, sess, chan, unit, '_'.join(game_state), 
                       obs, obs_z, obs_z.max(), obs_z.argmax(), tis, tis_z, pval])
        
    output = pd.DataFrame(output, columns=cols)

    # Save output.
    if save_output:
        dio.save_pickle(output, output_f, verbose)

    return output


def classify_time_bins(spikes, 
                       event_times, 
                       game_states, 
                       n_time_bins, 
                       C=0.001, 
                       k=6, 
                       n_perms=1000, 
                       save_as=None,
                       verbose=False):
    """Train multiclass linear SVM to predict time bins from firing rates.
    
    """
    do_shift = [False] + ([True] * n_perms)
    
    output = od([('obs', []), ('null', [])])
    for iPerm in range(n_perms+1):
        # Get firing rate data.
        fr = np.array([calc_fr_by_time_bin(row['fr_train'], 
                                           event_times.query("(subj_sess=='{}')".format(row['subj_sess'])), 
                                           game_states, 
                                           n_time_bins,
                                           random_shift=do_shift[iPerm])
                       for idx, row in spikes.iterrows()])
        shp = fr.shape
        n_neurons, n_trials, n_time_bins = shp

        # Z-score firing rates for each cell across time bins from all trials.
        zfr = stats.zscore(fr.reshape((n_neurons, np.prod(shp[1:]))), axis=1).reshape(shp)

        # Setup the the indepedent variable matrix and the dependent variable vector.
        X = np.swapaxes(zfr.reshape([n_neurons, np.prod((n_trials, n_time_bins))]), 0, 1)
        n_samp, n_feat = X.shape
        y = np.tile(np.arange(n_time_bins), n_trials).reshape((n_trials, n_time_bins)).flatten()

        # Perform k-fold cross-validation.
        kf = KFold(n_splits=k, shuffle=False)
        y_true = []
        y_pred = []
        for train, test in kf.split(X, y):
            clf = OneVsRestClassifier(LinearSVC(C=C, dual=False, fit_intercept=True))
            clf.fit(X[train, :], y[train])
            y_true.extend(y[test])
            y_pred.extend(clf.predict(X[test, :]))
        
        if iPerm == 0:
            output['obs'] = np.array(list(zip(y_true, y_pred)))
        else:
            output['null'].append(np.array(list(zip(y_true, y_pred))))
            
    # Save output.
    if save_as:
        dio.save_pickle(output, save_as, verbose)

    return output


# def create_null_fr_trains(fr_train, event_times, n_perms=1):
#     """Create a null distribution of firing rates for a neuron.
    
#     The firing rates within each event window are randomly circularly shifted
#     up to the length of the event window. This procedure is done n_perms times
#     to generate a null distribution in which firing rates are randomized across
#     trials, while the firing rate distribution and temporal structure is 
#     retained within each event.
    
#     By default we only do this once per function call (n_perms=1), since 
#     generating the whole null distribution at once would hit memory errors.
    
#     Returns
#     -------
#     fr_train_null : np.ndarray
#         n_timepoints vector of circ-shifted firing rates if n_perms == 1 
#         or n_perms x n_timepoints array if n_perms > 1.
#     """
#     fr_train_null = np.array([np.concatenate(event_times['time'].apply(lambda x: shift_fr_train(x, fr_train)))
#                               for i in range(n_perms)]) # n_perms x len(fr_train)
                              
#     if n_perms == 1:
#         fr_train_null = np.squeeze(fr_train_null) # len(fr_train)
    
#     return fr_train_null


def fr_by_time_vs_null(fr_train,
                       event_times,
                       game_states,
                       n_time_bins,
                       n_perms=1000,
                       save_as=None,
                       verbose=False):
    """Return the firing rate vector and its temporal information.
    
    Temporal information is tested against a null distribution of
    circ-shifted firing rates to derive an empirical p-value.
    
    Returns
    -------
    OrderedDict with the keys:
    
    fr_vec : np.ndarray
        The mean firing rate within each time bin, across trials.
    fr_vec_z : np.ndarray
        fr_vec Z-scored against the null distribution (separately for
        each time window).
    temporal_info : float
        The amount of temporal information contained in the firing
        rate vector.
    temporal_info_z : float
        temporal_info Z-scored against the null distribution.
    pval : float
        The p-value for the hypothesis that the temporal information score is
        not greater than chance (tested against the null distribution).
    """
    fr_vec = np.nanmean(calc_fr_by_time_bin(fr_train,
                                            event_times,
                                            game_states,
                                            n_time_bins,
                                            random_shift=False),
                        axis=0)
    temporal_info = info_rate(fr_vec)
    
    # Generate a surrogate distribution by circ-shifting the firing rates
    # for each event by a random interval up to the duration of the
    # event window.
    null_fr_vecs = []
    null_temporal_info = []
    for iPerm in range(n_perms):
        fr_vec_null = np.nanmean(calc_fr_by_time_bin(fr_train,
                                                     event_times,
                                                     game_states,
                                                     n_time_bins,
                                                     random_shift=True),
                                 axis=0)
        null_fr_vecs.append(fr_vec_null)
        null_temporal_info.append(info_rate(fr_vec_null))
    null_fr_vecs = np.array(null_fr_vecs) # iPerm x iTime
    null_temporal_info = np.array(null_temporal_info)
    
    # Compare observed values to the null distribution.
    fr_vec_z = (fr_vec - np.nanmean(null_fr_vecs, axis=0)) / np.nanstd(null_fr_vecs, axis=0)
    temporal_info_z = (temporal_info - np.nanmean(null_temporal_info)) / np.nanstd(null_temporal_info)
    pval_ind = np.nansum(null_temporal_info >= temporal_info)
    pval = (pval_ind + 1) / (n_perms + 1)
    
    output = od([('fr_vec', fr_vec),
                 ('fr_vec_z', fr_vec_z),
                 ('temporal_info', temporal_info),
                 ('temporal_info_z', temporal_info_z),
                 ('temporal_info_pval', pval)])
                 
    # Save output.
    if save_as:
        dio.save_pickle(output, save_as, verbose)
    
    return output


def info_rate(fr_given_x, 
              prob_x=None,
              scale_minmax=False):
    """Return the information rate of a cell in bits/spike.
    
    From Skaggs et al., 1993.
    
    Parameters
    ----------
    fr_given_x : numpy.ndarray
        A vector of firing rates for each state x in the 
        range len(fr_given_x).
    prob_x : numpy.ndarray
        A vector of probability densities for each state 
        x in the range len(fr_given_x). If prob_x is
        None then uniform probabilities are assumed.
    """
    if prob_x is None:
        n_states = len(fr_given_x)
        prob_x = np.ones(n_states) / n_states
    if scale_minmax:
        fr_given_x = minmax_scale(fr_given_x)
    mean_fr = np.dot(prob_x, fr_given_x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bits_per_spike = np.nansum(prob_x * (fr_given_x/mean_fr) * np.log2(fr_given_x/mean_fr))
    return bits_per_spike


# def shift_fr_train(event_window, fr_train):
#     """Circ-shift firing rates within an event window.
    
#     Circ-shifts firing rates by a random number of 
#     timepoints up to the duration of the event window.
        
#     Parameters
#     ----------
#     event_window : list or tuple
#         Contains [start, stop] times in ms.
#     fr_train : numpy.ndarray
#         Vector of firing rates over the session, in ms.
    
#     Returns:
#     --------
#     fr_train_shifted : numpy.ndarray
#         Vector of shifted spike times within the event window.
#     """
#     start, stop = event_window
#     roll_by = int(np.random.rand() * (stop - start))
#     start = int(np.round(start, 0))
#     stop = int(np.round(stop, 0))
#     fr_train_shifted = np.roll(fr_train[start:stop], roll_by)
#     return fr_train_shifted
