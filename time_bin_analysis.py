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
3/11/21
"""
import sys
import os.path as op
from glob import glob
from collections import OrderedDict as od
import itertools
import warnings
import mkl
mkl.set_num_threads(1)
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
sys.path.append('/home1/dscho/code/general')
import data_io as dio
sys.path.append('/home1/dscho/code/projects')
from time_cells import events_proc, spike_preproc


class EventSpikes(object):
    """The event_spikes dataframe and its properties and methods."""
    
    def __init__(self, 
                 subj_sess,
                 proj_dir='/home1/dscho/projects/time_cells',
                 filename=None):
        """Initialize event_spikes for a testing session."""
        self.subj_sess = subj_sess
        self.proj_dir = proj_dir
        if filename is None:
            self.filename = op.join(self.proj_dir, 'analysis', 'events', 
                                    '{}-EventSpikes.pkl'.format(self.subj_sess))
        else:
            self.filename = filename
        self._set_event_spikes()
        self._set_column_map()

    def __str__(self):
        _s = '{} neurons\n'.format(self.n_neurons)
        return _s
        
    def get_spike_mat(self,
                      neuron,
                      game_state,
                      index='trial',
                      column='time_bin',
                      qry=None):
        """Return a trial x time_bin dataframe of spikes within game state.
                
        Parameters
        ----------
        neuron : str
            e.g. '17-1' is channel 17, unit 1.
        game_state : str
            Delay1, Encoding, ReturnToBase1, Delay2, Retrieval, or ReturnToBase2.
        index : str
            event_spikes column whose unique values will form the rows of the output dataframe.
        column : str
            event_spikes column whose unique values will form the columns of the output dataframe.
        qry : str
            Gets passed to event_spikes.query() to select a subset of rows within the game state
            (e.g. when time_penalty==1).
        """
        spike_mat = self.event_spikes.query("(gameState=='{}')".format(game_state)).copy()
        if qry is not None:
            spike_mat = spike_mat.query(qry)
        spike_mat = spike_mat.pivot(index=index, columns=column, values=neuron)
        return spike_mat
    
    def _set_column_map(self):
        """Organize event_spikes columns into behavioral and neural ID lists."""
        cols = od({'icpt': ['icpt'],
                   'time': ['time-{}'.format(_i) for _i in range(1, 11)],
                   'place': [col for col in self.event_spikes.columns if col.startswith('place-')],
                   'hd': [col for col in self.event_spikes.columns if col.startswith('hd-')],
                   'is_moving': ['is_moving'],
                   'base_in_view': ['base_in_view'],
                   'gold_in_view': ['gold_in_view'],
                   'dig_performed': ['dig_performed']})

        # Add an unnested version of all behavioral columns.
        cols['behav'] = list(itertools.chain.from_iterable(cols.values()))

        # Add a column for neuron IDs.
        cols['neurons'] = []
        for col in self.event_spikes.columns:
            # Neuron ID columns are stored like 'channel-unit';
            # e.g. '16-2' is the second unit on channel 16.
            try:
                assert len([int(x) for x in col.split('-')]) == 2
                cols['neurons'].append(col)
            except (ValueError, AssertionError):
                continue

        self.column_map = cols

    def _set_event_spikes(self):
        """Create the event spikes dataframe.
        
        event_spikes is an extension of the behav_events dataframe
        (see events_proc.Events.log_events_behav) in which each row
        contains behavioral variables for one 500ms time bin in a 
        testing session. Here we add on a column for each neuron in
        the session in which we count the number of spikes within each
        time bin. 
        
        Also, categorical variable columns are one-hot-coded
        in preparation to use event_spikes for regression model 
        construction.
        """
        events = events_proc.load_events(self.subj_sess,
                                         proj_dir=self.proj_dir,
                                         run_all=True)
        event_spikes = events.events_behav.copy()
        
        # Add an intercept column for regression fitting.
        event_spikes['icpt'] = 1

        # Format column values.
        event_spikes['maze_region'] = event_spikes['maze_region'].apply(lambda x: x.replace(' ', '_'))
        game_states = ['Delay1', 'Encoding', 'ReturnToBase1', 
                       'Delay2', 'Retrieval', 'ReturnToBase2']
        game_state_cat = pd.CategoricalDtype(game_states, ordered=True)
        event_spikes['gameState'] = event_spikes['gameState'].astype(game_state_cat)

        # Convert discrete value columns into one-hot-coded columns.
        time_step_loc, time_step_vals = event_spikes.columns.tolist().index('time_step'), event_spikes['time_step'].tolist()
        maze_region_loc, maze_region_vals = event_spikes.columns.tolist().index('maze_region'), event_spikes['maze_region'].tolist()
        head_direc_loc, head_direc_vals = event_spikes.columns.tolist().index('head_direc'), event_spikes['head_direc'].tolist()
        event_spikes = pd.get_dummies(event_spikes, prefix_sep='-',
                                      prefix={'time_step': 'time', 'maze_region': 'place', 'head_direc': 'hd'}, 
                                      columns=['time_step', 'maze_region', 'head_direc'])
        event_spikes.insert(time_step_loc, 'time_step', time_step_vals)
        event_spikes.insert(maze_region_loc, 'maze_region', maze_region_vals)
        event_spikes.insert(head_direc_loc, 'head_direc', head_direc_vals)
        
        # Only count the base as being in view when player is outside the base.
        event_spikes.loc[(event_spikes['place-Base']==1) & (event_spikes['base_in_view']==1), 'base_in_view'] = 0
        
        # Convert gold_in_view nans at Retrieval to 0.
        event_spikes.loc[(event_spikes['gameState']=='Retrieval') & (np.isnan(event_spikes['gold_in_view'])), 'gold_in_view'] = 0

        # Sort rows.
        event_spikes = event_spikes.sort_values(['trial', 'gameState', 'time_bin']).reset_index(drop=True)
        
        # Get neurons from the session.
        globstr = op.join(self.proj_dir, 'analysis', 'spikes', 
                          '{}-*-spikes.pkl'.format(self.subj_sess))
        spike_files = {spike_preproc.unit_from_file(_f): _f 
                       for _f in glob(globstr)}
        neurons = list(spike_files.keys())
        
        # Add a column with spike counts for each neuron.
        for neuron in neurons:
            spike_times = dio.open_pickle(spike_files[neuron])['spike_times']
            event_spikes[neuron] = spikes_per_timebin(event_spikes, spike_times)
        
        self.n_neurons = len(neurons)
        self.event_spikes = event_spikes


def save_event_spikes(event_spikes,
                      overwrite=False,
                      verbose=True):
    """Pickle an EventSpikes instance."""
    if op.exists(event_spikes.filename) and not overwrite:
        print('Cannot save {} as it already exists'.format(event_spikes.filename))
    else:
        dio.save_pickle(event_spikes, event_spikes.filename, verbose)


def load_event_spikes(subj_sess,
                      proj_dir='/home1/dscho/projects/time_cells',
                      filename=None,
                      overwrite=False,
                      verbose=True):
    """Load pickled EventSpikes for the session, else instantiate."""
    if filename is None:
        filename = op.join(proj_dir, 'analysis', 'events', '{}-EventSpikes.pkl'.format(subj_sess))

    if op.exists(filename) and not overwrite:
        if verbose:
            print('Loading saved EventSpikes file')
        event_spikes = dio.open_pickle(filename)
    else:
        if verbose:
            print('Creating EventSpikes')
        event_spikes = EventSpikes(subj_sess,
                                   proj_dir=proj_dir,
                                   filename=filename)
    return event_spikes


def lr_test(res_reduced, res_full):
    """Run a likelihood ratio test comparing nested models.
    
    Parameters
    ----------
    res_reduced : statsmodels results class
        Results for the reduced model, which
        must be a subset of the full model.
    res_full : statsmodels results class
        Results for the full model.
        
    Returns
    -------
    lr : float
        The likelihood ratio.
    df : int
        Degrees of freedom, defined as the difference between
        the number of parameters in the full vs. reduced models.
    pval : float
        P-value, or the area to the right of the likehood ratio
        on the chi-squared distribution with df degrees of freedom.
    """
    lr = -2 * (res_reduced.llf - res_full.llf)
    df = len(res_full.params) - len(res_reduced.params)
    pval = stats.chi2.sf(lr, df)
    
    return lr, df, pval


def glm_fit_unit(subj_sess,
                 neuron,
                 game_states=['Delay1', 'Delay2', 'Encoding', 'Retrieval'],
                 n_perm=1000,
                 proj_dir='/home1/dscho/projects/time_cells',
                 filename=None,
                 overwrite=False,
                 save_output=True,
                 verbose=False,
                 **kwargs):
    """Fit GLMs to predict firing rates for multiple game_states.
    
    **kwargs are passed to glm_fit().

    Returns
    -------
    glm_res : DataFrame
    """
    # Load the output file if it exists.
    if filename is None:
        filename = op.join(proj_dir, 'analysis', 'behav_glms', 
                           '{}-{}-glm_results.pkl'.format(subj_sess, neuron))
    
    if op.exists(filename) and not overwrite:
        if verbose:
            print('Loading from pickle.')
        return dio.open_pickle(filename)

    # Load the event_spikes dataframe.
    event_spikes = load_event_spikes(subj_sess)

    # Get the GLM fits.
    warnings.filterwarnings('ignore')
    model_fits = od([])
    for game_state in game_states:
        if verbose:
            print('\tfitting {} GLMs...'.format(game_state))
        model_fits[game_state] = glm_fit(neuron, 
                                         event_spikes, 
                                         game_state, 
                                         n_perm=n_perm)
    warnings.resetwarnings()

    # Compare GLM results for preselected model contrasts, and store
    # the results in a dataframe.
    glm_res = _get_glm_res(subj_sess,
                           neuron,
                           model_fits)

    # Save the glm_res dataframe.
    if save_output:
        dio.save_pickle(glm_res, filename, verbose)

    return glm_res


def glm_fit(neuron,
            event_spikes,
            game_state,
            n_perm=0,
            optimizer='lbfgs'):
    """Fit firing rates using Poisson-link GLM.
    
    Parameters
    ----------
    neuron : str
        e.g. '5-2' would be channel 5, unit 2
    event_spikes : pd.DataFrame
        EventSpikes instance that contains the event_spikes dataframe,
        an expanded version of the behav_events dataframe with columns 
        added for each neuron.
    game_state : str
        Delay1, Encoding, Delay2, or Retrieval
    n_perm : positive int
        The number of permutations to include in a null mutation
        spike counts (circ-shifted at random within each trial).
    optimizer : str
        The optimization algorithm to use for fitting model weights.
        Default uses limited-memory BFGS, chosen because it is fast and
        usually quite robust. 'nm' is a good alternative though if 
        'lbfgs' has convergence problems.
    
    Returns
    -------
    model_fits : dict[GLMResultsWrapper]
        Contains model fits from real and shuffled spike data.
    """
    # Get the data relevant time bins.
    df = event_spikes.event_spikes.query("(gameState=='{}')".format(game_state))
    cols = event_spikes.column_map
    
    # Independent variables are the dummy coded time steps and the intercept constant.
    predictors = od([])
    if game_state in ['Delay1', 'Delay2']:
        predictors['time'] = cols['icpt'] + cols['time'] 
        predictors['icpt'] = cols['icpt'] 

    elif game_state in ['Encoding', 'Retrieval']:
        predictors['time_place'] = cols['icpt'] + cols['time'] + cols['place'] 
        predictors['place'] = cols['icpt'] + cols['place'] 
        predictors['time'] = cols['icpt'] + cols['time']
        
        dig_col = []
        if game_state == 'Retrieval':
            dig_col += cols['dig_performed']
        predictors['full']          = cols['icpt'] + cols['time'] + cols['place'] + cols['hd'] + cols['is_moving'] + cols['base_in_view'] + cols['gold_in_view'] + dig_col
        predictors['full_subtime']  = cols['icpt']                + cols['place'] + cols['hd'] + cols['is_moving'] + cols['base_in_view'] + cols['gold_in_view'] + dig_col
        predictors['full_subplace'] = cols['icpt'] + cols['time']                 + cols['hd'] + cols['is_moving'] + cols['base_in_view'] + cols['gold_in_view'] + dig_col
        predictors['full_subhd']    = cols['icpt'] + cols['time'] + cols['place']              + cols['is_moving'] + cols['base_in_view'] + cols['gold_in_view'] + dig_col
        predictors['full_subbiv']   = cols['icpt'] + cols['time'] + cols['place'] + cols['hd'] + cols['is_moving']                        + cols['gold_in_view'] + dig_col
        predictors['full_subgiv']   = cols['icpt'] + cols['time'] + cols['place'] + cols['hd'] + cols['is_moving'] + cols['base_in_view']                        + dig_col

    # Fit Poisson regressions.
    res = od([])
    y = df[neuron]
    for mod_name in predictors.keys():
        X = df[predictors[mod_name]].fillna(0)
        res[mod_name] = sm.GLM(y, X, family=sm.families.Poisson()).fit(method=optimizer)
    
    # Fit the same models to shuffled spike counts.
    res_null = od({k: [] for k in predictors.keys()})
    for iPerm in range(n_perm):
        y = np.concatenate(df.groupby('trial')[neuron].apply(lambda x: list(_shift_spikes(x))).tolist())
        for mod_name in predictors.keys():
            X = df[predictors[mod_name]].fillna(0)
            res_null[mod_name].append(sm.GLM(y, X, family=sm.families.Poisson()).fit(method=optimizer))
        
    model_fits = od({'obs': res, 'null': res_null})
    return model_fits


def spikes_per_timebin(events_behav,
                       spike_times):
    """Count how many spikes are in each time bin.
    
    Returns a pandas Series.
    """
    def count_spikes(row, spike_times):
        start = row['start_time']
        stop = row['stop_time']
        return np.count_nonzero((spike_times >= start) & (spike_times < stop))

    return events_behav.apply(lambda x: count_spikes(x, spike_times), axis=1)


def _get_glm_res(subj_sess,
                 neuron,
                 model_fits):
    """Compare GLM fits between predefined model pairs.
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U518_ses0'
    neuron : str
        e.g. '17-1' is channel 17, unit 1.
    model_fits : dict
        This is the output of glm_fit_unit().

    Returns
    -------
    glm_res : DataFrame
    """
    # Find the names of all independent variables tested across models.
    param_cols = []
    for game_state in model_fits.keys():
        for mod_name in model_fits[game_state]['obs'].keys():
            param_cols += [param for param in model_fits[game_state]['obs'][mod_name].params.index
                           if (param not in param_cols)]
    
    # Iterate over each constrast in the model comparison dataframe
    # and add on the GLM fit variables.
    model_comp = _get_model_comp()
    model_output_cols = ['llf_diff', 'z_llf_diff', 'lr', 'df', 'chi_pval', 'emp_pval', 'null_hist'] + param_cols
    model_output = []
    for iRow, row in model_comp.iterrows():
        game_state = row['gameState']
        full_mod = row['full']
        reduced_mod = row['reduced']

        # Get full model parameter weights.
        params = od({k: np.nan for k in param_cols})
        for k, v in model_fits[game_state]['obs'][full_mod].params.to_dict().items():
            params[k] = v

        # Get the difference between reduced and full model AICs.
        # Better model fits are indicated by positive AIC_diff values.
        llf_diff = (model_fits[game_state]['obs'][reduced_mod].llf - 
                    model_fits[game_state]['obs'][full_mod].llf)
        lr, df, chi_pval = lr_test(model_fits[game_state]['obs'][reduced_mod], 
                                   model_fits[game_state]['obs'][full_mod])

        # Get AIC diffs from the null distribution, 
        # and use these to obtain an empirical p-value.
        n_perm = len(model_fits[game_state]['null'][reduced_mod])
        null_llf_diffs = np.array([model_fits[game_state]['null'][reduced_mod][iPerm].llf -
                                   model_fits[game_state]['null'][full_mod][iPerm].llf
                                   for iPerm in range(n_perm)])
        null_mean = np.mean(null_llf_diffs)
        null_std = np.std(null_llf_diffs)
        null_hist = np.histogram(null_llf_diffs, bins=31)
        z_llf_diff = (llf_diff - null_mean) / null_std
        pval_ind = np.sum(null_llf_diffs >= llf_diff)
        emp_pval = (pval_ind + 1) / (n_perm + 1)

        # Add the results to the output dataframe.
        model_output.append([llf_diff, z_llf_diff, lr, df, chi_pval, emp_pval, list(null_hist)] + list(params.values()))
    
    model_output = pd.DataFrame(model_output, columns=model_output_cols)
    glm_res = pd.concat((model_comp, model_output), axis=1)
    
    # Add subj_sess and neuron identity columns.
    glm_res.insert(0, 'subj_sess', subj_sess)
    glm_res.insert(1, 'neuron', neuron)
    
    return glm_res


def _get_model_comp():
    """Define pairwise comparisons between full and reduced models."""
    cols = ['gameState', 'testvar', 'reduced', 'full']
    model_comp = [
        ['Delay1', 'time', 'icpt', 'time'],
        ['Delay2', 'time', 'icpt', 'time'],
        ['Encoding', 'time', 'place', 'time_place'],
        ['Encoding', 'place', 'time', 'time_place'],
        ['Encoding', 'time', 'full_subtime', 'full'],
        ['Encoding', 'place', 'full_subplace', 'full'],
        ['Encoding', 'head_direc', 'full_subhd', 'full'],
        ['Encoding', 'base_in_view', 'full_subbiv', 'full'],
        ['Encoding', 'gold_in_view', 'full_subgiv', 'full'],
        ['Retrieval', 'time', 'place', 'time_place'],
        ['Retrieval', 'place', 'time', 'time_place'],
        ['Retrieval', 'time', 'full_subtime', 'full'],
        ['Retrieval', 'place', 'full_subplace', 'full'],
        ['Retrieval', 'head_direc', 'full_subhd', 'full'],
        ['Retrieval', 'base_in_view', 'full_subbiv', 'full'],
        ['Retrieval', 'gold_in_view', 'full_subgiv', 'full']
    ]
    model_comp = pd.DataFrame(model_comp, columns=cols)
    return model_comp


def _shift_spikes(spike_vec):
    """Circularly shift spike vector."""
    roll_by = np.random.randint(0, len(spike_vec))
    return np.roll(spike_vec, roll_by)


def calc_mean_fr_by_time(fr_by_time_bin_f,
                         proj_dir='/home1/dscho/projects/time_cells',
                         overwrite=False,
                         save_output=True,
                         verbose=True):
    """Calculate mean FR across trial phases of a given type."""
    neuron = '-'.join(op.basename(fr_by_time_bin_f).split('-')[:3])

    # Load the output file if it exists.
    output_f = op.join(proj_dir, 'analysis', 'fr_by_time_bin', 
                       '{}-mean_fr_by_time.pkl'.format(neuron))
    if op.exists(output_f) and not overwrite:
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
    """Predict time bins from firing rates using multiclass linear SVM."""
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
