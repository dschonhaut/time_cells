"""
place_analysis.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
-----------
Analyzing spike associations with place and navigation.

Last Edited
-----------
12/11/20
"""
import sys
import os.path as op
from collections import OrderedDict as od
import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
sys.path.append('/home1/dscho/code/general')
import data_io as dio
sys.path.append('/home1/dscho/code/projects')
from time_cells.spike_preproc import apply_spike_times_to_fr, roi_mapping
from time_cells.time_bin_analysis import info_rate


def calc_fr_by_time_and_pos(neuron,
                            event_times=None,
                            pos_time_idx=None,
                            pos_time_dur=None,
                            tbin_time_idx=None,
                            game_states=['Delay1', 'Encoding', 'Delay2', 'Retrieval'],
                            compute_null=True,
                            compute_stats=True,
                            proj_dir='/home1/dscho/projects/time_cells',
                            overwrite=False,
                            save_output=True,
                            verbose=True):
    """Calculate firing rates by trial phase time and subject position.
    
    Parameters
    ----------
    neuron : pd.Series
        Contains spike times and metainfo for a unit, as saved by
        format_spikes() or add_null_to_spikes() in spike_prepoc.py.
        Requires that add_null_to_spikes() has been run if compute_null
        is True.
    """
    def calc_mean_frs(spike_times,
                      return_fr_vecs=False):
        """
        Convolve spikes with a Gaussian to obtain a vector of firing rates
        at every ms, separately for each trial phase window. Compare the 
        flattened firing rate vector to the flattened subject position and
        trial phase time bin vector indices to calculate mean firing rate
        at each position and trial phase time bin.
        """
        fr_by_time = od([])
        fr_by_pos = od([])
        fr_by_time_subpos = od([])
        fr_by_pos_subtime = od([])
        if return_fr_vecs:
            fr_vecs = od([])
        for i, game_state in enumerate(game_states):
            # Firing rate each ms from start to end of each trial phase,
            # unraveled into a vector.
            fr_vec = np.concatenate(
                event_times.query("(gameState=='{}')".format(game_state))['time']
                           .apply(lambda x: apply_spike_times_to_fr(x, spike_times))
                           .tolist())

            mean_fr_vec = np.mean(fr_vec)

            # Mean firing rate within each time bin, across trials.
            fr_by_time[game_state] = {k: np.mean(fr_vec[tbin_time_idx[game_state][k]])
                                      for k in tbin_time_idx[game_state]}

            if game_state in ('Delay1', 'Delay2'):
                # Store the firing rate vectors.
                if return_fr_vecs:
                    fr_vecs[game_state] = od([('fr_vec', fr_vec)])
                continue

            # Mean firing rate at each position index, across trials.
            fr_by_pos[game_state] = {k: np.mean(fr_vec[pos_time_idx[game_state][k]])
                                     for k in pos_time_idx[game_state]}

            # Calculate the expected firing rate from time alone.
            # 1) Get the residual firing rate each ms after subtracting
            #    the mean firing rate for each time bin.
            # 2) Add back the mean firing rate for the game state.
            # 3) Set a floor value of 0.
            fr_vec_subtime = fr_vec.copy()
            for k in tbin_time_idx[game_state]:
                fr_vec_subtime[tbin_time_idx[game_state][k]] -= fr_by_time[game_state][k]
            fr_vec_subtime += mean_fr_vec
            fr_vec_subtime[fr_vec_subtime<0] = 0

            # Calculate the expected firing rate from subject position alone.
            # 1) Get the residual firing rate each ms after subtracting
            #    the mean firing rate for each subject position.
            # 2) Add back the mean firing rate for the game state.
            # 3) Set a floor value of 0.
            fr_vec_subpos = fr_vec.copy()
            for k in pos_time_idx[game_state]:
                fr_vec_subpos[pos_time_idx[game_state][k]] -= fr_by_pos[game_state][k]
            fr_vec_subpos += mean_fr_vec
            fr_vec_subpos[fr_vec_subpos<0] = 0

            # Mean firing rate within each time bin, across trials,
            # subtracting the expected firing rate based on subject position alone.
            fr_by_time_subpos[game_state] = {k: np.mean(fr_vec_subpos[tbin_time_idx[game_state][k]])
                                             for k in tbin_time_idx[game_state]}

            # Mean firing rate at each position index, subtracting the 
            # expected firing rate based on time alone.
            # adding back the mean firing rate across all fr_vec values,
            # and setting floor values to 0.
            fr_by_pos_subtime[game_state] = {k: np.mean(fr_vec_subtime[pos_time_idx[game_state][k]])
                                             for k in pos_time_idx[game_state]}

            # Store the firing rate vectors.
            if return_fr_vecs:
                fr_vecs[game_state] = od([('fr_vec', fr_vec),
                                          ('fr_vec_subtime', fr_vec_subtime),
                                          ('fr_vec_subpos', fr_vec_subpos)])

        mean_frs = {'fr_by_time': fr_by_time,
                    'fr_by_pos': fr_by_pos,
                    'fr_by_time_subpos': fr_by_time_subpos,
                    'fr_by_pos_subtime': fr_by_pos_subtime}

        if return_fr_vecs:
            return mean_frs, fr_vecs
        else:
            return mean_frs

    # Load the output file if it exists.
    output_f = op.join(proj_dir, 'analysis', 'fr_by_time_and_pos',
                       '{}-CSC{}-unit{}-fr_by_time_and_pos.pkl'
                       .format(neuron['subj_sess'], neuron['chan'], neuron['unit']))
    if op.exists(output_f) and not overwrite:
        if verbose:
            print('Found {}'.format(output_f))
        return dio.open_pickle(output_f)

    # Load event objects.
    subj_sess = neuron.subj_sess
    if np.any((event_times is None, 
               pos_time_idx is None, 
               pos_time_dur is None, 
               tbin_time_idx is None)):
        events_f = op.join(proj_dir, 'analysis', 'events', 'time_and_pos',
                           '{}_event_objs.pkl'.format(subj_sess))
        events_d = dio.open_pickle(events_f)
        event_times = events_d['event_times']
        pos_time_idx = events_d['pos_time_idx']
        pos_time_dur = events_d['pos_time_dur']
        tbin_time_idx = events_d['tbin_time_idx']

    # Calculate mean firing rate vectors by trial phase time bin
    # and subject position, with and without each variable controlling
    # for the other. 
    neuron['mean_frs_obs'], neuron['fr_vecs'] = calc_mean_frs(neuron['spike_times'], 
                                                              return_fr_vecs=True)

    if compute_null:
        neuron['mean_frs_null'] = {fr_cat: {} for fr_cat in neuron['mean_frs_obs']}
        n_perm = neuron['spike_times_null_trial_phase'].shape[0]
        for iPerm in range(n_perm):
            d = calc_mean_frs(neuron['spike_times_null_trial_phase'][iPerm, :])
            for fr_cat in d:
                for game_state in d[fr_cat]:
                    if game_state not in neuron['mean_frs_null'][fr_cat]:
                        neuron['mean_frs_null'][fr_cat][game_state] = []
                    neuron['mean_frs_null'][fr_cat][game_state].append(d[fr_cat][game_state])

    # Save outputs.
    if save_output:
        dio.save_pickle(neuron, output_f, verbose)

    # Call the next function in the pipeline.
    if compute_stats:
        mean_fr = compare_fr_by_time_and_pos(neuron,
                                             proj_dir=proj_dir,
                                             overwrite=overwrite,
                                             save_output=save_output,
                                             verbose=verbose)
        return neuron, mean_fr
    else:
        return neuron


def compare_fr_by_time_and_pos(neuron,
                               proj_dir='/home1/dscho/projects/time_cells',
                               overwrite=False,
                               save_output=True,
                               verbose=True):
    """Compare observed vs. null model firing rate vectors.
    
    Quantifies an information rate in bits/spike for each 
    firing rate vector based on the formula in Skaggs et al., 1993.
    
    Parameters
    ----------
    neuron : pd.Series
        This is the output of
        place_analaysis.calc_fr_by_time_and_pos()
    
    Returns
    -------
    pd.DataFrame
    """
    # Load the output file if it exists.
    output_f = op.join(proj_dir, 'analysis', 'fr_by_time_and_pos',
                       '{}-CSC{}-unit{}-info_rates.pkl'
                       .format(neuron['subj_sess'], neuron['chan'], neuron['unit']))
    if op.exists(output_f) and not overwrite:
        if verbose:
            print('Found {}'.format(output_f))
        return dio.open_pickle(output_f)
    

    cols = ['neuron', 'subj_sess', 'subj', 'sess', 
            'chan', 'unit', 'fr', 'hemroi', 'roi', 
            'fr_cat', 'game_state', 'obs_fr', 'obs_fr_max_ind', 'obs_fr_max', 
            'obs_fr_z', 'obs_fr_z_max_ind', 'obs_fr_z_max', 
            'obs_info', 'obs_info_z', 'pval_ind', 'pval']
    neuron_id = '{}-{}-{}'.format(neuron['subj_sess'], neuron['chan'], neuron['unit'])
    roi = roi_mapping().get(neuron['hemroi'][1:], 'Unknown')
    events_d = dio.open_pickle(op.join(proj_dir, 'analysis', 'events', 'time_and_pos', 
                                       '{}_event_objs.pkl'.format(neuron['subj_sess'])))
    fr_cats = list(neuron['mean_frs_obs'].keys())
    output = []
    for fr_cat in fr_cats:
        for game_state in neuron['mean_frs_obs'][fr_cat]:
            fr_idx = np.sort(list(neuron['mean_frs_obs'][fr_cat][game_state].keys()))
            if 'fr_by_pos' in fr_cat:
                rel_durs = np.array([events_d['pos_time_dur'][game_state][k] 
                                     for k in fr_idx])
                rel_durs = rel_durs / np.nansum(rel_durs)
            else:
                rel_durs = None
            n_perm = len(neuron['mean_frs_null'][fr_cat][game_state])
            obs_fr = np.array([neuron['mean_frs_obs'][fr_cat][game_state][k] 
                               for k in fr_idx])  # fr_idx
            null_fr = np.array([[neuron['mean_frs_null'][fr_cat][game_state][iPerm][k]  # perm x fr_idx
                                 for k in fr_idx]
                                for iPerm in range(n_perm)])
            obs_fr_max_ind = np.argmax(obs_fr)
            obs_fr_max = np.max(obs_fr)
            obs_fr_z = (obs_fr - np.nanmean(null_fr, axis=0)) / np.nanstd(null_fr, axis=0)
            obs_fr_z_max_ind = np.argmax(obs_fr_z)
            obs_fr_z_max = np.max(obs_fr_z)
            obs_info = info_rate(obs_fr, rel_durs)
            null_info = np.array([info_rate(null_fr[iPerm, :], rel_durs)
                                  for iPerm in range(n_perm)])
            obs_info_z = (obs_info - np.nanmean(null_info)) / np.nanstd(null_info)
            pval_ind = np.nansum(null_info >= obs_info)
            pval = (pval_ind + 1) / (n_perm + 1)
            output.append([neuron_id, neuron['subj_sess'], neuron['subj'], neuron['sess'], 
                           neuron['chan'], neuron['unit'], neuron['fr'], neuron['hemroi'], roi,
                           fr_cat, game_state, obs_fr, obs_fr_max_ind, obs_fr_max, 
                           obs_fr_z, obs_fr_z_max_ind, obs_fr_z_max, 
                           obs_info, obs_info_z, pval_ind, pval])

    output = pd.DataFrame(output, columns=cols).sort_values('game_state').reset_index(drop=True)
    
    # Save outputs.
    if save_output:
        dio.save_pickle(output, output_f, verbose)

    return output
