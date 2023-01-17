"""
pop_decoding.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Classify behavior from population neural activity.

Last Edited
----------- 
7/17/21
"""
import sys
import os.path as op
from collections import OrderedDict as od
from glob import glob
from time import time
import numpy as np
import pandas as pd
from sklearn.utils.fixes import loguniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
sys.path.append('/home1/dscho/code/general')
import data_io as dio
sys.path.append('/home1/dscho/code/projects')
from time_cells import spike_preproc, events_proc, time_bin_analysis


class PopSpikes(object):
    """The pop_spikes dataframe and its properties and methods."""
    
    def __init__(self,
                 time_bin_size=500,
                 place_adj=False,
                 proj_dir='/home1/dscho/projects/time_cells'):
        """Initialize the pop_spikes dataframe."""
        self.time_bin_size = time_bin_size
        self.place_adj = place_adj
        self.proj_dir = proj_dir
        self.game_states = ['Delay1', 'Encoding', 'Delay2', 'Retrieval']
        self._set_pop_spikes()
        
    def __str__(self):
        """Print how many neurons there are."""
        info = '{} neurons'.format(self.neurons.size) + '\n'
        info += '{} ms time bins'.format(self.time_bin_size) + '\n'
        info += '{} trials'.format(self.pop_spikes['trial'].unique().size) + '\n'
        info += '{} time bins/trial'.format(self.pop_spikes.groupby(['trial']).size().iloc[0]) + '\n'
        info += 'pop_spikes: {}'.format(self.pop_spikes.shape)
        
        return info
    
    def combine_time_bins(self,
                          pop_spikes=None,
                          time_step_size=1000,
                          impute_nans=True):
        """Combine time bins into larger time steps and sum spikes.
        
        Returns
        -------
        _pop_spikes : dataframe
            Same as the normal pop_spikes dataframe but with time bin
            rows aggregated into time steps within each trial and
            game state, and with added columns for game state integer
            order, time step (replacing time bin) and game state order
            x time step.
        """
        if pop_spikes is None:
            _pop_spikes = self.pop_spikes.copy()
        else:
            _pop_spikes = pop_spikes.copy()

        # Add a column for game state order.
        i_game_states = {game_state: idx
                         for (idx, game_state) in enumerate(self.game_states)}
        _pop_spikes.insert(2, 'iGameState', _pop_spikes['gameState'].apply(lambda x: i_game_states[x]).astype(int))

        # Add a column for time step.
        bins_per_step = int(time_step_size / self.time_bin_size)
        _pop_spikes.insert(3, 'time_step', _pop_spikes['time_bin'].apply(lambda x: int(x/bins_per_step)))

        # Median impute NaNs.
        if impute_nans:
            _pop_spikes[self.neurons] = _pop_spikes[self.neurons].fillna(_pop_spikes[self.neurons].median(axis=0), axis=0)

        # For each unit, sum spikes within each time step.
        grp = _pop_spikes.groupby(['trial', 'gameState', 'iGameState', 'time_step'], observed=True)
        _pop_spikes = grp[self.neurons].sum().reset_index()

        # Add a column combining game state and time step.
        game_state_times = _pop_spikes.apply(lambda x: '{}_{:0>2}'.format(int(x['iGameState']), int(x['time_step'])), axis=1)
        i_game_state_times = {game_state_time: idx
                              for (idx, game_state_time) in enumerate(np.unique(game_state_times))}
        _pop_spikes.insert(4, 'iGameStateTime', [i_game_state_times[x] for x in game_state_times])

        return _pop_spikes

    def _set_pop_spikes(self):
        """Set the pop_spikes dataframe and a list of neuron columns.

        Sets
        ----
        pop_spikes : DataFrame
            pop_spikes concatenates spike counts of units across all subject
            sessions within each trial, game state, and time bin. Each row
            is a unique time bin, and each column a unique unit.
        neurons : list
            List of unique neuron names that make up pop_spikes columns.
        """
        # Find all subject sessions.
        sessions = np.unique([op.basename(f).split('-')[0] 
                              for f in glob(op.join(self.proj_dir, 'analysis', 'events', '*-Events.pkl'))])

        # Load the event_spikes dataframe for each session.
        game_states = ['Delay1', 'Encoding', 'Delay2', 'Retrieval']
        event_cols = ['trial', 'gameState', 'time_bin']
        neurons = []
        dfs = od([])
        for subj_sess in sessions:
            if self.place_adj:
                filename = op.join(self.proj_dir, 'analysis', 'events',
                                   '{}-EventSpikes-place_adj.pkl'.format(subj_sess))
                event_spikes = time_bin_analysis.load_event_spikes(subj_sess, filename=filename,
                                                                   verbose=False)
            else:
                event_spikes = time_bin_analysis.load_event_spikes(subj_sess, verbose=False)
            neuron_labels = od({neuron: '{}-{}'.format(subj_sess, neuron) 
                                for neuron in event_spikes.column_map['neurons']})
            dfs[subj_sess] = (event_spikes.event_spikes.query("(gameState=={})".format(game_states))
                                                       .rename(columns=neuron_labels)
                                                       .loc[:, event_cols + list(neuron_labels.values())]
                                                       .set_index(event_cols))
            neurons += list(neuron_labels.values())
        neurons = np.array(neurons)

        # Concatentate spiking data for each unit, within each time bin, across sessions.
        pop_spikes = pd.concat(dfs, axis=1)
        pop_spikes.columns = pop_spikes.columns.get_level_values(1)
        pop_spikes = pop_spikes.reset_index()

        # Sort rows of the output dataframe by trial, game state, and time bin.
        game_state_cat = pd.CategoricalDtype(game_states, ordered=True)
        pop_spikes['gameState'] = pop_spikes['gameState'].astype(game_state_cat)
        pop_spikes = pop_spikes.sort_values(['trial', 'gameState', 'time_bin']).reset_index(drop=True)
        
        self.pop_spikes = pop_spikes
        self.neurons = neurons


def load_pop_spikes(exclude_ctx=True,
                    **kws):
    pop_spikes = PopSpikes(**kws)

    # Exclude cortical units outside MTL, prefrontal, or temporal lobe.
    if exclude_ctx:
<<<<<<< HEAD
        # n_rois = 5
        # roi_map = spike_preproc.roi_mapping(n_rois)
        # keep_idx = np.where(np.array([roi_map[spike_preproc.roi_lookup(x.split('-')[0], x.split('-')[1])[1:]]
        #                               for x in pop_spikes.neurons]) != 'Cortex')[0]
        keep_idx = ~np.isin([spike_preproc.roi_lookup(x.split('-')[0], x.split('-')[1])[1:]
                             for x in pop_spikes.neurons], ['AI', 'O'])
=======
        n_rois = 5
        roi_map = spike_preproc.roi_mapping(n_rois)
        keep_idx = np.where(np.array([roi_map[spike_preproc.roi_lookup(x.split('-')[0], x.split('-')[1])[1:]] for x in pop_spikes.neurons]) != 'Cortex')[0]
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
        pop_spikes.neurons = pop_spikes.neurons[keep_idx]
        keep_cols = ['trial', 'gameState', 'time_bin'] + pop_spikes.neurons.tolist()
        pop_spikes.pop_spikes = pop_spikes.pop_spikes[keep_cols]

    return pop_spikes


def _combine_time_bins(spike_mat, bins=10):
    """Return sum of spikes across trials within each time bin."""
    spike_mat = np.array(spike_mat)
    if len(spike_mat.shape) == 2:
        return np.array([v.sum() for v in np.split(np.sum(spike_mat, axis=0), bins)])
    else:
        return None


<<<<<<< HEAD
def classify_place(subj_sess,
                   game_states=['Encoding', 'Retrieval'],
                   save_clfs=False,
                   search_method='random',
                   vals_per_hyperparam=7,
                   hyperparam_n_iter=100,
                   n_jobs=32,
                   proj_dir='/home1/dscho/projects/time_cells',
                   save_results=True,
                   overwrite=True,
                   verbose=True):
    """Predict time within each game state from population neural activity.
    
    Implements support vector classification (RBF kernel) with repeated,
    nested cross-validation to first, optimize C and gamma hyperparaters
    on a dataset of trainval trials, and then test the best fitting model
    on hold-out test trials.
    
    Parameters
    ----------
    game_states : list[str]
        List of game states to train classifiers on.
    save_clfs : bool
        Determines if the fit observed and null classifiers are
        included in the dataframe that gets saved and returned.
    search_method : str
        Defines how parameter search will be implemented.
        'grid' uses sklearn's GridSearchCV.
        'random' uses sklearn's RandomizedSearchCV.
    vals_per_hyperparam : int
        Only relevant if search_method is 'grid'. Defines how many
        values will be searched within the parameter space
        for C and gamma. 7, # for grid search
    hyperparam_n_iter : int
        Only relevant if search_method is 'random'. Defines how many
        random points in the parameter space will be searched to find
        a best fitting model within each fold of the trainval inner 
        cross-validation.
    n_jobs : int
        Number of jobs to run in parallel for the parameter search inner
        cross-validation (the most the time-consuming part of this function).
    """
    def setup_param_search():
        """Setup parameter search over the inner CV and return an estimator.
        
        Does not return a fitted estimator.
        """
        kfold_inner = 5 # n_trials - 1
        inner_cv = KFold(kfold_inner)

        if search_method == 'grid':    
            param_grid = {'svc__C'     : np.logspace(-5, 5, vals_per_hyperparam),
                          'svc__gamma' : np.logspace(-5, 5, vals_per_hyperparam)}
            clf = GridSearchCV(estimator=pipe,
                               param_grid=param_grid,
                               refit=True,
                               cv=inner_cv,
                               n_jobs=n_jobs)
        elif search_method == 'random':
            param_dist = {'svc__C'     : loguniform(1e-9, 1e9),
                          'svc__gamma' : loguniform(1e-9, 1e9)}
            clf = RandomizedSearchCV(estimator=pipe,
                                     param_distributions=param_dist,
                                     refit=True,
                                     cv=inner_cv,
                                     n_jobs=n_jobs,
                                     n_iter=hyperparam_n_iter)
        return clf

    def get_mean_accuracy(acc_vec, reshp):
        """Return mean accuracy across time_bins.

        Parameters
        ----------
        acc_vec : list
            Accuracy vector comparing y_test to y_test_pred 
            at each time bin, across test trials.
        reshp : list
            (n_test_trials, n_time_bins)
        """
        return np.mean(np.array(acc_vec).reshape(reshp), axis=0).tolist()
    
    start_time = time()
    
    # Define hard-coded parameters.
    y_col = 'maze_region'
    
    # Load the pop_spikes dataframe (spike counts for units across all subject
    # sessions within each trial, game state, and time bin).
    event_spikes = time_bin_analysis.load_event_spikes(subj_sess,
                                                       proj_dir=proj_dir,
                                                       verbose=False)
    neurons = event_spikes.column_map['neurons']
    trials = event_spikes.event_spikes['trial'].unique()
    n_trials = trials.size
    
    # Split trials into trainval and test sets.
    kfold_outer = n_trials
    trainval_test = split_trials(trials, n_splits=kfold_outer)
    
    # Get the output filename and return its contents
    # if filename exists and overwrite is False.
    basename = 'SVC_predicting_{}'.format(y_col)
    basename += '-{}'.format(subj_sess)
    basename += '-{}units'.format(len(neurons))
    basename += '-{}_search'.format(search_method)
    basename += '-{}fold'.format(kfold_outer)
    basename += '.pkl'
    filename = op.join(proj_dir, 'analysis', 'classifiers', basename)
    if op.exists(filename) and not overwrite:
        clf_results = dio.open_pickle(filename)
        return clf_results
    
    # Setup the processing pipeline for classification.
    # 1. Impute missing data by replacing NaNs with their column-wise median.
    # 2. Z-score the values in each column.
    # 3. Train a support vector classifier with RBF kernel.
    pipe = Pipeline(steps=[('impute', SimpleImputer(strategy='median')),
                           ('scale', StandardScaler()),
                           ('svc', SVC(kernel='rbf'))])
    
    clf_results = []
    for game_state in game_states:
        # Select a subset of pop_spikes rows and columns.
        _event_spikes = event_spikes.event_spikes.query("(gameState=='{}')".format(game_state)).reset_index(drop=True)

        # Train classifiers to predict time from population neural activity.
        if verbose:
            print('{} {}: Fitting {} neurons, {:.1f} min'
                  .format(subj_sess, game_state, len(neurons), (time() - start_time) / 60))
        
        # Perform nested cross-validation, splitting trials into
        # test and nested train/val sets.
        for iFold in range(kfold_outer):
            # Select the test and trainval trials.
            trainval, test = trainval_test[iFold]

            # ---------------------------------
            # Observed data:
            #
            # Split trials into trainval and test sets.
            X_trainval = _event_spikes.loc[np.isin(_event_spikes['trial'], trainval)][neurons].values
            y_trainval = _event_spikes.loc[np.isin(_event_spikes['trial'], trainval)][y_col].values
            X_test = _event_spikes.loc[np.isin(_event_spikes['trial'], test)][neurons].values
            y_test = _event_spikes.loc[np.isin(_event_spikes['trial'], test)][y_col].values

            # Setup grid search on the inner CV.
            clf = setup_param_search()

            # Train the model on trainval data.
            clf.fit(X_trainval, y_trainval)
            best_trainval_score = clf.best_score_
            best_trainval_C = clf.best_params_['svc__C']
            best_trainval_gamma = clf.best_params_['svc__gamma']

            # Predict time from neural activity on test data.
            y_test_pred = clf.predict(X_test).tolist()

            # Calculate accuracy.
            accuracy = [y_test_pred[iVal]==y_test[iVal] for iVal in range(len(y_test))]
            mean_acc = np.mean(accuracy)

            # ---------------------------------
            # Null distribution:
            #
            # Circ-shift time steps within each trial to randomize 
            # time_step ~ pop_spiking associations across trials.
            shuf_idx = np.concatenate(_event_spikes.reset_index()
                                                   .groupby('trial')['index']
                                                   .apply(lambda x: np.roll(x, np.random.randint(0, len(x))))
                                                   .tolist())
            _event_spikes_null = _event_spikes.copy()
            _event_spikes_null[y_col] = _event_spikes_null.loc[shuf_idx, y_col].values

            # Split trials into trainval and test sets, using the same split as
            # for the observed data.
            X_test_null = _event_spikes_null.loc[np.isin(_event_spikes_null['trial'], test)][neurons].values
            y_test_null = _event_spikes_null.loc[np.isin(_event_spikes_null['trial'], test)][y_col].values
            X_trainval_null = _event_spikes_null.loc[np.isin(_event_spikes_null['trial'], trainval)][neurons].values
            y_trainval_null = _event_spikes_null.loc[np.isin(_event_spikes_null['trial'], trainval)][y_col].values

            # Setup grid search on the inner CV.
            clf_null = setup_param_search()

            # Train the model on trainval data.
            clf_null.fit(X_trainval_null, y_trainval_null)
            best_trainval_score_null = clf_null.best_score_
            best_trainval_C_null = clf_null.best_params_['svc__C']
            best_trainval_gamma_null = clf_null.best_params_['svc__gamma']

            # Predict time from neural activity on test data.
            y_test_pred_null = clf_null.predict(X_test_null).tolist()

            # Calculate accuracy.
            accuracy_null = [y_test_pred_null[iVal]==y_test_null[iVal] for iVal in range(len(y_test_null))]
            mean_acc_null = np.mean(accuracy_null)

            # Append results to the output dataframe.
            new_row = [game_state,
                       game_state,
                       test,
                       best_trainval_score,
                       best_trainval_C,
                       best_trainval_gamma,
                       y_test,
                       y_test_pred,
                       mean_acc,
                       accuracy,
                       best_trainval_score_null,
                       best_trainval_C_null,
                       best_trainval_gamma_null,
                       y_test_null,
                       y_test_pred_null,
                       mean_acc_null,
                       accuracy_null]
            if save_clfs:
                new_row += [clf, clf_null]
            clf_results.append(new_row)

    cols = ['gameState_train',
            'gameState_test',
            'test_trials',
            'best_trainval_score',
            'best_trainval_C',
            'best_trainval_gamma',
            'y_test',
            'y_test_pred',
            'mean_acc',
            'accuracy',
            'best_trainval_score_null',
            'best_trainval_C_null',
            'best_trainval_gamma_null',
            'y_test_null',
            'y_test_pred_null',
            'mean_acc_null',
            'accuracy_null']
    if save_clfs:
        cols += ['clf', 'clf_null']

    clf_results = pd.DataFrame(clf_results, columns=cols)

    if save_results:
        dio.save_pickle(clf_results, filename, verbose=verbose)
    
    if verbose:
        print('Done in {:.1f} min'.format((time() - start_time) / 60))
    
    return clf_results


def classify_within_subj(subj_sess,
                         y_col='time_bin',
                         game_states=['Delay1', 'Delay2', 'Encoding', 'Retrieval'],
                         save_clfs=False,
                         search_method='random',
                         vals_per_hyperparam=7,
                         hyperparam_n_iter=100,
                         n_jobs=32,
                         proj_dir='/home1/dscho/projects/time_cells',
                         save_results=True,
                         overwrite=True,
                         verbose=True,
                         **kws):
    """Classify within each game state from population neural activity.
    
    Implements support vector classification (RBF kernel) with repeated,
    nested cross-validation to first, optimize C and gamma hyperparaters
    on a dataset of trainval trials, and then test the best fitting model
    on hold-out test trials.
    
    Parameters
    ----------
    game_states : list[str]
        List of game states to train classifiers on.
    save_clfs : bool
        Determines if the fit observed and null classifiers are
        included in the dataframe that gets saved and returned.
    search_method : str
        Defines how parameter search will be implemented.
        'grid' uses sklearn's GridSearchCV.
        'random' uses sklearn's RandomizedSearchCV.
    vals_per_hyperparam : int
        Only relevant if search_method is 'grid'. Defines how many
        values will be searched within the parameter space
        for C and gamma. 7, # for grid search
    hyperparam_n_iter : int
        Only relevant if search_method is 'random'. Defines how many
        random points in the parameter space will be searched to find
        a best fitting model within each fold of the trainval inner 
        cross-validation.
    n_jobs : int
        Number of jobs to run in parallel for the parameter search inner
        cross-validation (the most the time-consuming part of this function).
    """
    def setup_param_search():
        """Setup parameter search over the inner CV and return an estimator.
        
        Does not return a fitted estimator.
        """
        kfold_inner = 5 # n_trials - 1
        inner_cv = KFold(kfold_inner)

        if search_method == 'grid':    
            param_grid = {'svc__C'     : np.logspace(-5, 5, vals_per_hyperparam),
                          'svc__gamma' : np.logspace(-5, 5, vals_per_hyperparam)}
            clf = GridSearchCV(estimator=pipe,
                               param_grid=param_grid,
                               refit=True,
                               cv=inner_cv,
                               n_jobs=n_jobs)
        elif search_method == 'random':
            param_dist = {'svc__C'     : loguniform(1e-9, 1e9),
                          'svc__gamma' : loguniform(1e-9, 1e9)}
            clf = RandomizedSearchCV(estimator=pipe,
                                     param_distributions=param_dist,
                                     refit=True,
                                     cv=inner_cv,
                                     n_jobs=n_jobs,
                                     n_iter=hyperparam_n_iter)
        return clf

    def get_mean_accuracy(acc_vec, reshp):
        """Return mean accuracy across time_bins.

        Parameters
        ----------
        acc_vec : list
            Accuracy vector comparing y_test to y_test_pred 
            at each time bin, across test trials.
        reshp : list
            (n_test_trials, n_time_bins)
        """
        return np.mean(np.array(acc_vec).reshape(reshp), axis=0).tolist()
    
    start_time = time()
    
    # Load the pop_spikes dataframe (spike counts for units across all subject
    # sessions within each trial, game state, and time bin).
    event_spikes = time_bin_analysis.load_event_spikes(subj_sess,
                                                       proj_dir=proj_dir,
                                                       verbose=False)
    neurons = event_spikes.column_map['neurons']
    trials = event_spikes.event_spikes['trial'].unique()
    n_trials = trials.size
    
    # Split trials into trainval and test sets.
    kfold_outer = n_trials
    trainval_test = split_trials(trials, n_splits=kfold_outer)
    
    # Get the output filename and return its contents
    # if filename exists and overwrite is False.
    basename = 'SVC_predicting_{}'.format(y_col)
    basename += '-{}'.format(subj_sess)
    basename += '-{}units'.format(len(neurons))
    basename += '-{}'.format('_'.join(game_states))
    basename += '-{}_search'.format(search_method)
    basename += '-{}fold'.format(kfold_outer)
    basename += '.pkl'
    filename = op.join(proj_dir, 'analysis', 'classifiers', basename)
    if op.exists(filename) and not overwrite:
        clf_results = dio.open_pickle(filename)
        return clf_results
    
    # Setup the processing pipeline for classification.
    # 1. Impute missing data by replacing NaNs with their column-wise median.
    # 2. Z-score the values in each column.
    # 3. Train a support vector classifier with RBF kernel.
    pipe = Pipeline(steps=[('impute', SimpleImputer(strategy='median')),
                           ('scale', StandardScaler()),
                           ('svc', SVC(kernel='rbf'))])
    
    clf_results = []
    for game_state in game_states:
        # Select a subset of pop_spikes rows and columns.
        _event_spikes = event_spikes.event_spikes.query("(gameState=='{}')".format(game_state)).reset_index(drop=True).copy()

        if y_col == 'time_step':
            # Aggregate time bins into a defined number of equal-size, larger time steps.
            game_state_durs = events_proc.get_game_state_durs()
            n_time_steps = kws['n_time_steps']
            n_time_bins = np.unique(_event_spikes['time_bin']).size
            n_time_steps = n_time_steps[game_state]
            bins_per_step = int(n_time_bins / n_time_steps)
            _event_spikes['time_step'] = _event_spikes['time_bin'].apply(lambda time_bin: int(time_bin/bins_per_step))
            
            # For each unit, sum spikes within each time step.
            _event_spikes = _event_spikes.groupby(['gameState', 'trial', 'time_step'], observed=True)[neurons].apply(np.sum).reset_index()
            
        # Train classifiers to predict time from population neural activity.
        if verbose:
            print('{} {}: Fitting {} neurons, {:.1f} min'
                  .format(subj_sess, game_state, len(neurons), (time() - start_time) / 60))
        
        # Perform nested cross-validation, splitting trials into
        # test and nested train/val sets.
        for iFold in range(kfold_outer):
            # Select the test and trainval trials.
            trainval, test = trainval_test[iFold]

            # ---------------------------------
            # Observed data:
            #
            # Split trials into trainval and test sets.
            X_trainval = _event_spikes.loc[np.isin(_event_spikes['trial'], trainval)][neurons].values
            y_trainval = _event_spikes.loc[np.isin(_event_spikes['trial'], trainval)][y_col].values
            X_test = _event_spikes.loc[np.isin(_event_spikes['trial'], test)][neurons].values
            y_test = _event_spikes.loc[np.isin(_event_spikes['trial'], test)][y_col].values

            # Setup grid search on the inner CV.
            clf = setup_param_search()

            # Train the model on trainval data.
            clf.fit(X_trainval, y_trainval)
            best_trainval_score = clf.best_score_
            best_trainval_C = clf.best_params_['svc__C']
            best_trainval_gamma = clf.best_params_['svc__gamma']

            # Predict time from neural activity on test data.
            y_test_pred = clf.predict(X_test).tolist()

            # Calculate accuracy.
            accuracy = [y_test_pred[iVal]==y_test[iVal] for iVal in range(len(y_test))]
            mean_acc = np.mean(accuracy)

            # ---------------------------------
            # Null distribution:
            #
            # Circ-shift time steps within each trial to randomize 
            # time_step ~ pop_spiking associations across trials.
            shuf_idx = np.concatenate(_event_spikes.reset_index()
                                                   .groupby('trial')['index']
                                                   .apply(lambda x: np.roll(x, np.random.randint(0, len(x))))
                                                   .tolist())
            _event_spikes_null = _event_spikes.copy()
            _event_spikes_null[y_col] = _event_spikes_null.loc[shuf_idx, y_col].values

            # Split trials into trainval and test sets, using the same split as
            # for the observed data.
            X_test_null = _event_spikes_null.loc[np.isin(_event_spikes_null['trial'], test)][neurons].values
            y_test_null = _event_spikes_null.loc[np.isin(_event_spikes_null['trial'], test)][y_col].values
            X_trainval_null = _event_spikes_null.loc[np.isin(_event_spikes_null['trial'], trainval)][neurons].values
            y_trainval_null = _event_spikes_null.loc[np.isin(_event_spikes_null['trial'], trainval)][y_col].values

            # Setup grid search on the inner CV.
            clf_null = setup_param_search()

            # Train the model on trainval data.
            clf_null.fit(X_trainval_null, y_trainval_null)
            best_trainval_score_null = clf_null.best_score_
            best_trainval_C_null = clf_null.best_params_['svc__C']
            best_trainval_gamma_null = clf_null.best_params_['svc__gamma']

            # Predict time from neural activity on test data.
            y_test_pred_null = clf_null.predict(X_test_null).tolist()

            # Calculate accuracy.
            accuracy_null = [y_test_pred_null[iVal]==y_test_null[iVal] for iVal in range(len(y_test_null))]
            mean_acc_null = np.mean(accuracy_null)

            # Append results to the output dataframe.
            new_row = [subj_sess,
                       game_state,
                       game_state,
                       y_col,
                       test,
                       best_trainval_score,
                       best_trainval_C,
                       best_trainval_gamma,
                       y_test,
                       y_test_pred,
                       mean_acc,
                       accuracy,
                       best_trainval_score_null,
                       best_trainval_C_null,
                       best_trainval_gamma_null,
                       y_test_null,
                       y_test_pred_null,
                       mean_acc_null,
                       accuracy_null]
            if save_clfs:
                new_row += [clf, clf_null]
            clf_results.append(new_row)

    cols = ['subj_sess',
            'gameState_train',
            'gameState_test',
            'testvar',
            'test_trials',
            'best_trainval_score',
            'best_trainval_C',
            'best_trainval_gamma',
            'y_test',
            'y_test_pred',
            'mean_acc',
            'accuracy',
            'best_trainval_score_null',
            'best_trainval_C_null',
            'best_trainval_gamma_null',
            'y_test_null',
            'y_test_pred_null',
            'mean_acc_null',
            'accuracy_null']
    if save_clfs:
        cols += ['clf', 'clf_null']

    clf_results = pd.DataFrame(clf_results, columns=cols)

    if save_results:
        dio.save_pickle(clf_results, filename, verbose=verbose)
    
    if verbose:
        print('Done in {:.1f} min'.format((time() - start_time) / 60))
    
    return clf_results


=======
>>>>>>> 288ef90488a75ccdf754e8946f1eb37476c7d185
def classify_time(unit_groups=None,
                  n_subset_units=None,
                  exclude_ctx=True,
                  game_states=['Delay1', 'Encoding', 'Delay2', 'Retrieval'],
                  time_steps_per_game_state={'Delay1': 10, 'Encoding': 30, 'Delay2': 10, 'Retrieval': 30},
                  drop_first2=False,
                  save_clfs=False,
                  search_method='random',
                  vals_per_hyperparam=7,
                  hyperparam_n_iter=100,
                  n_jobs=32,
                  proj_dir='/home1/dscho/projects/time_cells',
                  save_results=True,
                  overwrite=True,
                  verbose=True):
    """Predict time within each game state from population neural activity.
    
    Implements support vector classification (RBF kernel) with repeated,
    nested cross-validation to first, optimize C and gamma hyperparaters
    on a dataset of trainval trials, and then test the best fitting model
    on hold-out test trials.
    
    Parameters
    ----------
    unit_groups : dict, 'group_name' : ['unit1', 'unit2', ...]
        Dict containing lists of units that correspond to columns
        of the pop_spikes dataframe.
    n_subset_units : int
        Determines how many units will be drawn at random from
        each unit group. This variable is overwritten if there
        is more than one unit group in unit_groups, in which
        case it is updated to the minimum number of units across
        unit groups.
    game_states : list[str]
        List of game states to train classifiers on.
    time_steps_per_game_state : dict, 'game_state' : int
        Defines how many evenly-sized time steps will divide up
        each game state in game_states. The time steps for each
        game state must evenly divide the number of smaller time
        bins that make up each row of the pop_spikes dataframe.
        E.g., if Delay1 is divided into 20 time bins each 500ms
        long, Delay1 time steps can be 4 or 5 but not 3 or 6, as
        these don't divide into 20.
    drop_first2 : bool
        If True, the first 2s of data for each trial interval
        are dropped from the analysis (so e.g., for Delay only
        seconds 2-10 would be considered).
    save_clfs : bool
        Determines if the fit observed and null classifiers are
        included in the dataframe that gets saved and returned.
    search_method : str
        Defines how parameter search will be implemented.
        'grid' uses sklearn's GridSearchCV.
        'random' uses sklearn's RandomizedSearchCV.
    vals_per_hyperparam : int
        Only relevant if search_method is 'grid'. Defines how many
        values will be searched within the parameter space
        for C and gamma. 7, # for grid search
    hyperparam_n_iter : int
        Only relevant if search_method is 'random'. Defines how many
        random points in the parameter space will be searched to find
        a best fitting model within each fold of the trainval inner 
        cross-validation.
    n_jobs : int
        Number of jobs to run in parallel for the parameter search inner
        cross-validation (the most the time-consuming part of this function).
    """
    def setup_param_search():
        """Setup parameter search over the inner CV and return an estimator.
        
        Does not return a fitted estimator.
        """
        kfold_inner = 5 # n_trials - 1
        inner_cv = KFold(kfold_inner)

        if search_method == 'grid':    
            param_grid = {'svc__C'     : np.logspace(-5, 5, vals_per_hyperparam),
                          'svc__gamma' : np.logspace(-5, 5, vals_per_hyperparam)}
            clf = GridSearchCV(estimator=pipe,
                               param_grid=param_grid,
                               refit=True,
                               cv=inner_cv,
                               n_jobs=n_jobs)
        elif search_method == 'random':
            param_dist = {'svc__C'     : loguniform(1e-9, 1e9),
                          'svc__gamma' : loguniform(1e-9, 1e9)}
            clf = RandomizedSearchCV(estimator=pipe,
                                     param_distributions=param_dist,
                                     refit=True,
                                     cv=inner_cv,
                                     n_jobs=n_jobs,
                                     n_iter=hyperparam_n_iter)
        return clf

    def get_mean_accuracy(acc_vec, reshp):
        """Return mean accuracy across time_bins.

        Parameters
        ----------
        acc_vec : list
            Accuracy vector comparing y_test to y_test_pred 
            at each time bin, across test trials.
        reshp : list
            (n_test_trials, n_time_bins)
        """
        return np.mean(np.array(acc_vec).reshape(reshp), axis=0).tolist()
    
    start_time = time()
    
    # Define hard-coded parameters.
    y_col = 'time_step'
    
    # Load the pop_spikes dataframe (spike counts for units across all subject
    # sessions within each trial, game state, and time bin).
    pop_spikes = load_pop_spikes(exclude_ctx=exclude_ctx)
    trials = pop_spikes.pop_spikes['trial'].unique()
    n_trials = trials.size
    
    # Setup groups of units to train classifiers on.
    if unit_groups is None:
        unit_groups = od([('all', pop_spikes.neurons)])
    
    # If more than one unit group is being processed, find the group with
    # the fewest number of neurons and assign this number to n_subset_units.
    if len(unit_groups) > 1:
        n_subset_units = np.min([len(_neurons) for _neurons in unit_groups.values()])
        
    # Split trials into trainval and test sets.
    kfold_outer = n_trials
    trainval_test = split_trials(trials, n_splits=kfold_outer)
    
    # Get the output filename and return its contents
    # if filename exists and overwrite is False.
    basename = 'SVC_predicting_{}'.format(y_col)
    basename += '-' + '-'.join(['{}units_{}'.format(len(unit_groups[unit_group]), unit_group) for unit_group in unit_groups])
    basename += '-{}_search'.format(search_method)
    basename += '-{}fold'.format(kfold_outer)
    basename += '-{}units_per_subset'.format(n_subset_units) if (n_subset_units is not None) else ''
    basename += '-spike_matched' if (len(unit_groups) > 1) else ''
    basename += '-' + '-'.join(['{}_{}bins'.format(game_state, time_steps_per_game_state[game_state]) for game_state in game_states])
    basename += '-drop_first2s' if drop_first2 else ''
    basename += '.pkl'
    filename = op.join(proj_dir, 'analysis', 'classifiers', basename)
    if op.exists(filename) and not overwrite:
        clf_results = dio.open_pickle(filename)
        return clf_results
    
    # Setup the processing pipeline for classification.
    # 1. Impute missing data by replacing NaNs with their column-wise median.
    # 2. Z-score the values in each column.
    # 3. Train a support vector classifier with RBF kernel.
    pipe = Pipeline(steps=[('impute', SimpleImputer(strategy='median')),
                           ('scale', StandardScaler()),
                           ('svc', SVC(kernel='rbf'))])
    
    # Drop first 2s of data.
    if drop_first2:
        pop_spikes.pop_spikes = pop_spikes.pop_spikes.query("(time_bin!=[0, 1, 2, 3])").reset_index(drop=True)
    
    # Check that the number of time steps evenly divides the number of 
    # time bins for each game state.
    game_state_durs = events_proc.get_game_state_durs()
    for game_state in game_states:
        n_time_bins = np.unique(pop_spikes.pop_spikes.query("(gameState=='{}')".format(game_state))['time_bin']).size
        n_time_steps = time_steps_per_game_state[game_state]
        if drop_first2:
            n_time_steps -= int(2 * (time_steps_per_game_state[game_state] / (game_state_durs[game_state]*1e-3)))
        if (n_time_bins % n_time_steps) != 0:
            raise ValueError('n_time_steps does not evenly divide n_time_bins for {} ({} % {} != 0)'
                             .format(game_state, n_time_bins, n_time_steps))
    
    # Check that all units in unit_groups are columns in pop_spikes.
    for unit_group in unit_groups:
        _neurons = unit_groups[unit_group]
        if not np.all(np.isin(_neurons, pop_spikes.neurons)):
            raise ValueError('Some unit names in {} are not in pop_spikes.pop_spikes.columns'
                             .format(unit_group))
    
    clf_results = []
    for game_state in game_states:
        # Randomly select n_subset_units from neurons in the unit group.
        pop_spike_grps = od()
        for unit_group in unit_groups:
            # Select neurons in the unit group.
            _neurons = unit_groups[unit_group]

            # Randomly select n_subset_units from neurons in the unit group.
            if (n_subset_units is not None) and (len(_neurons) > n_subset_units):
                __neurons = np.random.permutation(_neurons).tolist()[:n_subset_units]
            else:
                __neurons = _neurons
                
            # Select a subset of pop_spikes rows and columns.
            _pop_spikes = pop_spikes.pop_spikes.query("(gameState=='{}')".format(game_state)).copy()
            
            # Aggregate time bins into a defined number of equal-size, larger time steps.
            n_time_bins = np.unique(_pop_spikes['time_bin']).size
            n_time_steps = time_steps_per_game_state[game_state]
            if drop_first2:
                n_time_steps -= int(2 * (time_steps_per_game_state[game_state] / (game_state_durs[game_state]*1e-3)))
            bins_per_step = int(n_time_bins / n_time_steps)
            _pop_spikes.insert(2, 'time_step', _pop_spikes['time_bin'].apply(lambda time_bin: int(time_bin/bins_per_step)))

            # For each unit, sum spikes within each time step.
            _pop_spikes = _pop_spikes.groupby(['gameState', 'trial', 'time_step'], observed=True)[__neurons].apply(np.sum).reset_index()
                
            pop_spike_grps[unit_group] = _pop_spikes
            
        # Match the number of spikes retained between unit groups,
        # sorting the neurons in each unit group by firing rate and then
        # randomly removing spikes from each neuron as needed.
        keep_spikes = np.min(np.concatenate([[pop_spike_grps[unit_group].iloc[:, 3:].sum().sort_values()]
                                             for unit_group in unit_groups], axis=0), axis=0)
        for unit_group in unit_groups:
            _keep_spikes = pd.Series(data=keep_spikes,
                                     index=pop_spike_grps[unit_group].iloc[:, 3:].sum().sort_values().index.values,
                                     dtype=int)
            for neuron, n_spikes in _keep_spikes.iteritems():
                spike_vec = pop_spike_grps[unit_group][neuron].values
                n_remove = spike_vec.sum() - n_spikes
                while n_remove > 0:
                    iRow = np.random.randint(spike_vec.size)
                    if spike_vec[iRow] > 0:
                        spike_vec[iRow] -= 1
                        n_remove -= 1

        # Train classifiers to predict time from population neural activity.
        for unit_group in unit_groups:
            _pop_spikes = pop_spike_grps[unit_group]
            __neurons = _pop_spikes.columns[3:].values
            if verbose:
                print('{}: Fitting {} neurons from {}, {:.1f} min'
                      .format(game_state, len(__neurons), unit_group, (time() - start_time) / 60))
            
            # Perform nested cross-validation, splitting trials into
            # test and nested train/val sets.
            for iFold in range(kfold_outer):
                # Select the test and trainval trials.
                trainval, test = trainval_test[iFold]

                # ---------------------------------
                # Observed data:
                #
                # Split trials into trainval and test sets.
                X_trainval = _pop_spikes.loc[np.isin(_pop_spikes['trial'], trainval)][__neurons].values
                y_trainval = _pop_spikes.loc[np.isin(_pop_spikes['trial'], trainval)][y_col].values
                X_test = _pop_spikes.loc[np.isin(_pop_spikes['trial'], test)][__neurons].values
                y_test = _pop_spikes.loc[np.isin(_pop_spikes['trial'], test)][y_col].values

                # Setup grid search on the inner CV.
                clf = setup_param_search()

                # Train the model on trainval data.
                clf.fit(X_trainval, y_trainval)
                best_trainval_score = clf.best_score_
                best_trainval_C = clf.best_params_['svc__C']
                best_trainval_gamma = clf.best_params_['svc__gamma']

                # Predict time from neural activity on test data.
                y_test_pred = clf.predict(X_test).tolist()

                # Calculate accuracy.
                accuracy = [y_test_pred[iVal]==y_test[iVal] for iVal in range(len(y_test))]
                acc_by_time = get_mean_accuracy(accuracy, reshp=(len(test), n_time_steps))
                mean_acc = np.mean(accuracy)

                # ---------------------------------
                # Null distribution:
                #
                # Circ-shift time steps within each trial to randomize 
                # time_step ~ pop_spiking associations across trials.
                shuf_idx = np.concatenate(_pop_spikes.reset_index()
                                                     .groupby('trial')['index']
                                                     .apply(lambda x: np.roll(x, np.random.randint(0, len(x))))
                                                     .tolist())
                _pop_spikes_null = _pop_spikes.copy()
                _pop_spikes_null[y_col] = _pop_spikes_null.loc[shuf_idx, y_col].values

                # Split trials into trainval and test sets, using the same split as
                # for the observed data.
                X_test_null = _pop_spikes_null.loc[np.isin(_pop_spikes_null['trial'], test)][__neurons].values
                y_test_null = _pop_spikes_null.loc[np.isin(_pop_spikes_null['trial'], test)][y_col].values
                X_trainval_null = _pop_spikes_null.loc[np.isin(_pop_spikes_null['trial'], trainval)][__neurons].values
                y_trainval_null = _pop_spikes_null.loc[np.isin(_pop_spikes_null['trial'], trainval)][y_col].values

                # Setup grid search on the inner CV.
                clf_null = setup_param_search()

                # Train the model on trainval data.
                clf_null.fit(X_trainval_null, y_trainval_null)
                best_trainval_score_null = clf_null.best_score_
                best_trainval_C_null = clf_null.best_params_['svc__C']
                best_trainval_gamma_null = clf_null.best_params_['svc__gamma']

                # Predict time from neural activity on test data.
                y_test_pred_null = clf_null.predict(X_test_null).tolist()

                # Calculate accuracy.
                accuracy_null = [y_test_pred_null[iVal]==y_test_null[iVal] for iVal in range(len(y_test_null))]
                acc_by_time_null = get_mean_accuracy(accuracy_null, reshp=(len(test), n_time_steps))
                mean_acc_null = np.mean(accuracy_null)

                # Append results to the output dataframe.
                new_row = [game_state,
                           unit_group,
                           n_time_steps,
                           test,
                           best_trainval_score,
                           best_trainval_C,
                           best_trainval_gamma,
                           y_test,
                           y_test_pred,
                           mean_acc,
                           acc_by_time,
                           accuracy,
                           best_trainval_score_null,
                           best_trainval_C_null,
                           best_trainval_gamma_null,
                           y_test_null,
                           y_test_pred_null,
                           mean_acc_null,
                           acc_by_time_null,
                           accuracy_null]
                if save_clfs:
                    new_row += [clf, clf_null]
                clf_results.append(new_row)

    cols = ['gameState',
            'unit_group',
            'n_time_steps',
            'test_trials',
            'best_trainval_score',
            'best_trainval_C',
            'best_trainval_gamma',
            'y_test',
            'y_test_pred',
            'mean_acc',
            'acc_by_time',
            'accuracy',
            'best_trainval_score_null',
            'best_trainval_C_null',
            'best_trainval_gamma_null',
            'y_test_null',
            'y_test_pred_null',
            'mean_acc_null',
            'acc_by_time_null',
            'accuracy_null']
    if save_clfs:
        cols += ['clf', 'clf_null']

    clf_results = pd.DataFrame(clf_results, columns=cols)

    if save_results:
        dio.save_pickle(clf_results, filename, verbose=verbose)
    
    if verbose:
        print('Done in {:.1f} min'.format((time() - start_time) / 60))
    
    return clf_results


def split_trials(trials=None,
                 n_trials=36,
                 n_splits=36):
    """Split trials into trainval and test sets.
    
    Parameters
    ----------
    trials : array
        List of trials to split.
    n_trials : int
        Number of sequential trials to generate from 1..n_trials.
        This parameter is only used if trials is None.
    n_splits : int
        Defines the number of cross-validation folds. Must evenly
        divide n_trials.
    Returns
    -------
    trainval_test: list[trainval, test]
        Contains lists of trainval and test trial numbers
        for each cross-validation fold.
    """
    # Get trials to split.
    if trials is None:
        trials = np.arange(1, n_trials+1)
    if type(trials) in (list, tuple):
        trials = np.array(trials)
    
    # Construct the cross-validation generator.
    if len(trials) % n_splits != 0:
        raise ValueError('array split does not result in an equal division')
    cv = KFold(n_splits)
    
    # Get pairs of [test, trainval] trial splits.
    trainval_test = [[trials[idx[0]].tolist(), trials[idx[1]].tolist()]
                     for idx in cv.split(trials)]
    
    return trainval_test
