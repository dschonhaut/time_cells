"""
remapping.py

Author:
    Daniel Schonhaut
    
Dependencies:
    Python 3.6, numpy, pandas

Description: 
    Functions for analyzing time and place cell remapping.

Last Edited: 
    6/14/21
"""
import sys
from collections import OrderedDict as od
import numpy as np
import pandas as pd
import scipy.stats as stats
sys.path.append('/home1/dscho/code/general')
import array_operations as aop


def trial_pair_remapping(twise_spikes_by_time,
                         trial_pairs=None,
                         sim_func='cos_sim'):
    """Firing similarity for within- and between-condition trial pairs.
    
    Parameters
    ----------
    twise_spikes_by_time : dict
        Has two keys for each trial condition type.
        twise_spikes_by_time['key1'] contains a (trial, time) array of
        spike counts or firing rates, and twise_spikes_by_time['key1']
        must be an array with the same shape as
        twise_spikes_by_time['key2'].
    trial_pairs : dict
        By default, trial pairs are all possible within-condition pairs,
        and the same trial pairs are used for each condition.
        If passed explicitly, must be a dict whose keys match those of
        twise_spikes_by_time and have an additional key, 'between'
        Each dictionary value is a list of two-length tuples containing
        each pair of trials to calculate cosine simiarlity between.
    sim_func : str
        'cos_sim' or 'pearson'
    
    Returns
    -------
    sims : dict
        Cosine similaries between firing vectors for all trial
        pairs, for each key in twise_spikes_by_time, along with
        cosine similarities for the same trial pairs for the
        between-condition comparison: twise_spikes_by_time['key1'] vs.
        twise_spikes_by_time['key2'].
    """
    
    # Identify all within-trial and between-trial pairs.
    assert len(twise_spikes_by_time) == 2
    k1, k2 = twise_spikes_by_time.keys()
    assert twise_spikes_by_time[k1].shape[0] == twise_spikes_by_time[k1].shape[0]
    for _k in twise_spikes_by_time:
        if type(twise_spikes_by_time[_k]) == pd.core.frame.DataFrame:
            twise_spikes_by_time[_k] = twise_spikes_by_time[_k].values
    n_trials = twise_spikes_by_time[k1].shape[0]
    
    if trial_pairs is None:
        _pairs = pairwise_combs(n_trials, keep_matching=False)
        trial_pairs = od([(k1, _pairs),
                          (k2, _pairs),
                          ('between', _pairs)])

    # Calculate the cosine similarity between firing vectors for each
    # trial pair.
    sims = od([(k1, []),
               (k2, []),
               ('between', [])])
    
    # Within-condition.
    for _k in twise_spikes_by_time:
        for iPair in range(len(trial_pairs[_k])):
            t1, t2 = trial_pairs[_k][iPair]
            if sim_func == 'cos_sim':
                sims[_k].append(aop.cos_sim(twise_spikes_by_time[_k][t1, :], 
                                            twise_spikes_by_time[_k][t2, :]))
            elif sim_func == 'pearson':
                sims[_k].append(stats.pearsonr(twise_spikes_by_time[_k][t1, :], 
                                               twise_spikes_by_time[_k][t2, :])[0])
        sims[_k] = np.array(sims[_k])[np.isfinite(sims[_k])]

    # Between-condition.
    _k = 'between'
    for iPair in range(len(trial_pairs[_k])):
        t1, t2 = trial_pairs[_k][iPair]
        if sim_func=='cos_sim':
            sims[_k].append(aop.cos_sim(twise_spikes_by_time[k1][t1, :], 
                                        twise_spikes_by_time[k2][t2, :]))
        elif sim_func == 'pearson':
            sims[_k].append(stats.pearsonr(twise_spikes_by_time[k1][t1, :], 
                                           twise_spikes_by_time[k2][t2, :])[0])
    sims[_k] = np.array(sims[_k])[np.isfinite(sims[_k])]
    
    return sims


def pairwise_combs(n,
                   keep_matching=False):
    """Return a list with all pairwise combinations."""
    if keep_matching:
        pairs = [(x, y)
                 for x in range(n)
                 for y in range(n)
                 if (x<=y)]
    else:
        pairs = [(x, y)
                 for x in range(n)
                 for y in range(n)
                 if (x<y)]
    return pairs
