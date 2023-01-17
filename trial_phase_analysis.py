"""
trial_phase_analysis.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com
    
Description
----------- 
Functions for analyzing firing rate data across trial phases.

Last Edited
----------- 
3/19/21
"""
import sys
import os.path as op
from collections import OrderedDict as od
import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
sys.path.append('/home1/dscho/code/general')
import data_io as dio
sys.path.append('/home1/dscho/code/projects')
from time_cells.time_bin_analysis import load_event_spikes, lr_test, _shift_spikes, _shuffle_spikes


def game_state_unit_fr_diff(neuron,
                            event_spikes=None,
                            subj_sess=None,
                            n_perm=0,
                            regress_trial=True,
                            perm_method='circshift',
                            alpha=0.05,
                            proj_dir='/home1/dscho/projects/time_cells',
                            filename=None,
                            overwrite=False,
                            save_output=True,
                            verbose=False):
    """Fit firing rates during each game state using OLS regression.
    
    First asks if the mean firing rate differs between any of the
    four, timed game states (Delay1, Encoding, Delay2, Retrieval).
    
    Then performs posthoc comparisons between all possible game
    state pairs.
    
    For each of these comparisons, a likelihood ratio is calculated
    between models with and without game state terms. Significance
    is assessed by comparing the likelihood ratio to a null
    distribution in which firing rate vectors are randomized within
    each trial.
    
    Parameters
    ----------
    neuron : str
        e.g. '5-2' would be channel 5, unit 2
    event_spikes : pd.DataFrame
        EventSpikes instance that contains the event_spikes dataframe,
        an expanded version of the behav_events dataframe with columns 
        added for each neuron.
    subj_sess : str
        e.g. 'U518_ses0'. This parameter is used to load event_spikes 
        only if event_spikes is not passed in the function call. If
        event_spikes is passed, subj_sess is ignored.
    n_perm : positive int
        The number of permutations to draw for the null distribution.
    regress_trial : bool
        If True, GLM includes a dummy-coded predictor for each trial.
        If False, no trial predictors are added.
    perm_method : str
        Determines how the firing rate vector will be randomized to 
        obtain a null distribution. 'circshift' or 'shuffle'
    alpha : float between (0, 1)
        Used to determine significance threshold. Post-hoc tests
        are Bonferroni-Holm corrected.
    
    Returns
    -------
    model_pairs : DataFrame
    """
    # Load event_spikes dataframe if it wasn't passed as an input.
    if event_spikes is None:
        event_spikes = load_event_spikes(subj_sess,
                                         proj_dir=proj_dir,
                                         verbose=verbose)
    subj_sess = event_spikes.subj_sess

    # Load the output file if it exists.
    if filename is None:
        filename = op.join(proj_dir, 'analysis', 'unit_to_behav', 
                           '{}-{}-ols-game_state-model_pairs.pkl'
                           .format(subj_sess, neuron))
    
    if op.exists(filename) and not overwrite:
        if verbose:
            print('Loading from pickle.')
        return dio.open_pickle(filename)
    
    # Select rows for the chosen game state.
    game_states = ['Delay1', 'Encoding', 'Delay2', 'Retrieval']
    keep_cols = ['trial', 'gameState', neuron]
    df = event_spikes.event_spikes.query("(gameState=={})".format(game_states))[keep_cols].copy()
    del event_spikes
    df['gameState'] = df['gameState'].astype(str)
    
    # Calculate the mean firing rate within each trial phase, within each trial.
    df = df.groupby(['trial', 'gameState'])[neuron].mean().reset_index()

    # Define the model formulas.
    trial_term = ' + C(trial)' if regress_trial else ''
    formulas = od([])
    formulas['game_state'] = "Q('{}') ~ 1 + C(gameState){}".format(neuron, trial_term)
    formulas['icpt']       = "Q('{}') ~ 1               {}".format(neuron, trial_term)

    # Define pairwise comparisons and initialize the model_pairs dataframe.
    pairs = [(x1, x2) 
             for x1 in range(len(game_states)) 
             for x2 in range(len(game_states)) 
             if x1 < x2]
    posthocs = [(game_states[pair[0]], game_states[pair[1]]) 
                for pair in pairs]
    model_pairs = pd.DataFrame(['omnibus'] + ['_'.join(pair) for pair in posthocs],
                               columns=['testvar'])
    model_pairs.insert(0, 'subj_sess', subj_sess)
    model_pairs.insert(1, 'neuron', neuron)

    # Fit the models.
    model_output_cols = ['llf_full', 'lr', 'z_lr', 'df', 'pval', 'null_hist']
    model_output = []
    for testvar in model_pairs['testvar']:
        # Select a subset of df rows.
        if testvar == 'omnibus':
            _df = df.copy()
        else:
            _game_states = testvar.split('_')
            _df = df.query("(gameState=={})".format(_game_states)).copy()

        # Fit the real data.
        res = od([])
        for mod_name in formulas.keys():
            res[mod_name] = ols(formulas[mod_name], data=_df).fit()
        llf_full = res['game_state'].llf
        lr, dof, _ = lr_test(res['icpt'], res['game_state'])
        lr = np.max((lr, 0))

        # Fit the same models to shuffled spike counts.
        res_null = od({k: [] for k in formulas.keys()})
        null_lrs = []
        for iPerm in range(n_perm):
            # Permute the firing rate vector.
            if perm_method == 'circshift':
                _df[neuron] = np.concatenate(_df.groupby('trial')[neuron]
                                                .apply(lambda x: list(_shift_spikes(x))).tolist())
            elif perm_method == 'shuffle':
                _df[neuron] = np.concatenate(_df.groupby('trial')[neuron]
                                                .apply(lambda x: list(_shuffle_spikes(x))).tolist())
            else:
                raise RuntimeError("Permutation method '{}' not recognized".format(perm_method))

            # Fit the model.
            for mod_name in formulas.keys():
                res_null[mod_name].append(ols(formulas[mod_name], data=_df).fit())
            null_lrs.append(lr_test(res_null['icpt'][-1], res_null['game_state'][-1])[0])

        # Get likelihood ratios from the null distribution, 
        # and use these to obtain an empirical p-value.
        null_mean = np.mean(null_lrs)
        null_std = np.std(null_lrs)
        null_hist = np.histogram(null_lrs, bins=31)
        z_lr = (lr - null_mean) / null_std
        pval_ind = np.sum(null_lrs >= lr)
        emp_pval = (pval_ind + 1) / (n_perm + 1)

        # Add results to the output dataframe.
        model_output.append([llf_full, lr, z_lr, dof, emp_pval, null_hist])

    model_output = pd.DataFrame(model_output, columns=model_output_cols)
    model_pairs = pd.concat((model_pairs, model_output), axis=1)
    model_pairs = model_pairs[np.isfinite(model_pairs['lr'])]

    # Determine significance.
    model_pairs.insert(8, 'sig', np.nan)
    omni_sig = model_pairs.loc[model_pairs['testvar']=='omnibus', 'pval'] < alpha
    phoc_sig = sm.stats.multipletests(model_pairs.loc[model_pairs['testvar']!='omnibus', 'pval'],
                                      alpha, method='holm')[0]
    model_pairs.loc[model_pairs['testvar']=='omnibus', 'sig'] = omni_sig
    model_pairs.loc[model_pairs['testvar']!='omnibus', 'sig'] = phoc_sig
    
    # Save the model_pairs dataframe.
    if save_output:
        dio.save_pickle(model_pairs, filename, verbose)

    return model_pairs
