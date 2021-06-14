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
sys.path.append('/home1/dscho/code/general')
import array_operations as aop


def trial_pair_remapping(twise_spikes_by_time,
                         trial_pairs=None):
    """Firing similarity for within- and between-condition trial pairs.
    
    Parameters
    ----------
    twise_spikes_by_time : dict
        Has two keys for each trial condition type.
        twise_spikes_by_time['key1'] contains a (trial, time) array of
        spike counts or firing rates, and twise_spikes_by_time['key1']
        must be an array with the same shape as
        twise_spikes_by_time['key2'].
    
    Returns
    -------
    sims : dict
        Mean cosine similary betwen firing vectors across all trial
        pairs for each key in twise_spikes_by_time, along with the mean
        cosine similarity across the same trial pairs for
        twise_spikes_by_time['key1'] vs. twise_spikes_by_time['key2'].
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
        _pairs = [(x, y)
                  for x in range(n_trials)
                  for y in range(n_trials)
                  if (x<y)]
        trial_pairs = od([(k1, _pairs),
                          (k2, _pairs),
                          ('between', _pairs)])

    # Calculate the cosine similarity between firing vectors for each
    # trial pair.
    sims = od([(k1, []),
               (k2, []),
               ('between', [])])
    for _k in sims:
        for iPair in range(len(trial_pairs[_k])):
            t1, t2 = trial_pairs[iPair]
            sims[_k].append(aop.cos_sim(twise_spikes_by_time[_k][t1, :], 
                                        twise_spikes_by_time[_k][t2, :]))
        sims[_k] = np.array(sims[_k])[np.isfinite(sims[_k])]
    
    return sims
