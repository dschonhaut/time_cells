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
Functions for analyzing firing rate data within time bins.

Last Edited
-----------
3/19/21
"""
import sys
import os.path as op
import copy
from glob import glob
from collections import OrderedDict as od
import itertools
import warnings
import random
import numpy as np
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

sys.path.append("/home1/dscho/code/general")
from helper_funcs import str_replace, strip_space
import data_io as dio

sys.path.append("/home1/dscho/code/projects")
from time_cells import events_proc, spike_preproc


class EventSpikes(object):
    """The event_spikes dataframe and its properties and methods.

    EventSpikes combines the behavioral by time bin info contained in
    events_behav dataframe within the Events class with spike counts
    by time bin for each neuron in a session.
    """

    def __init__(
        self, subj_sess, proj_dir="/home1/dscho/projects/time_cells", filename=None
    ):
        """Initialize event_spikes for a testing session."""
        self.subj_sess = subj_sess
        self.proj_dir = proj_dir
        if filename is None:
            self.filename = op.join(
                self.proj_dir,
                "analysis",
                "events",
                "{}-EventSpikes.pkl".format(self.subj_sess),
            )
        else:
            self.filename = filename
        self._set_event_spikes()
        self._set_column_map()

    def __str__(self):
        _s = "{} neurons\n".format(self.n_neurons)
        return _s

    def get_spike_mat(
        self,
        neuron,
        game_state,
        index="trial",
        column="time_bin",
        apply_f=np.sum,
        qry=None,
    ):
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
        apply_f : func
            The function to apply over grouped values.
        qry : str
            Gets passed to event_spikes.query() to select a subset of rows within the game state
            (e.g. when time_penalty==1).
        """
        spike_mat = self.event_spikes.query(
            "(gameState=='{}')".format(game_state)
        ).copy()
        if qry is not None:
            spike_mat = spike_mat.query(qry)
        spike_mat = (
            spike_mat.groupby([index, column])[neuron].apply(apply_f).unstack(column)
        )
        return spike_mat

    def _set_column_map(self):
        """Organize event_spikes columns into behavioral and neural ID lists."""
        cols = od(
            {
                "icpt": ["icpt"],
                "trial": [
                    col for col in self.event_spikes.columns if col.startswith("trial-")
                ],
                "time": ["time-{}".format(_i) for _i in range(1, 11)],
                "place": [
                    col for col in self.event_spikes.columns if col.startswith("place-")
                ],
                "hd": [
                    col for col in self.event_spikes.columns if col.startswith("hd-")
                ],
                "is_moving": ["is_moving"],
                "base_in_view": ["base_in_view"],
                "gold_in_view": ["gold_in_view"],
                "dig_performed": ["dig_performed"],
            }
        )

        # Add an unnested version of all behavioral columns.
        cols["behav"] = list(itertools.chain.from_iterable(cols.values()))

        # Add a column for neuron IDs.
        cols["neurons"] = []
        for col in self.event_spikes.columns:
            # Neuron ID columns are stored like 'channel-unit';
            # e.g. '16-2' is the second unit on channel 16.
            try:
                assert len([int(x) for x in col.split("-")]) == 2
                cols["neurons"].append(col)
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
        events = events_proc.load_events(
            self.subj_sess, proj_dir=self.proj_dir, verbose=False, run_all=True
        )
        event_spikes = events.events_behav.copy()

        # Add an intercept column for regression fitting.
        event_spikes["icpt"] = 1

        # Format column values.
        event_spikes["maze_region"] = event_spikes["maze_region"].apply(
            lambda x: x.replace(" ", "_")
        )
        game_states = [
            "Delay1",
            "Encoding",
            "ReturnToBase1",
            "Delay2",
            "Retrieval",
            "ReturnToBase2",
        ]
        game_state_cat = pd.CategoricalDtype(game_states, ordered=True)
        event_spikes["gameState"] = event_spikes["gameState"].astype(game_state_cat)

        # Convert discrete value columns into one-hot-coded columns.
        time_step_loc, time_step_vals = (
            event_spikes.columns.tolist().index("time_step"),
            event_spikes["time_step"].tolist(),
        )
        maze_region_loc, maze_region_vals = (
            event_spikes.columns.tolist().index("maze_region"),
            event_spikes["maze_region"].tolist(),
        )
        head_direc_loc, head_direc_vals = (
            event_spikes.columns.tolist().index("head_direc"),
            event_spikes["head_direc"].tolist(),
        )
        trial_loc, trial_vals = (
            event_spikes.columns.tolist().index("trial"),
            event_spikes["trial"].tolist(),
        )
        event_spikes = pd.get_dummies(
            event_spikes,
            prefix_sep="-",
            prefix={
                "time_step": "time",
                "maze_region": "place",
                "head_direc": "hd",
                "trial": "trial",
            },
            columns=["time_step", "maze_region", "head_direc", "trial"],
        )
        event_spikes.insert(time_step_loc, "time_step", time_step_vals)
        event_spikes.insert(maze_region_loc, "maze_region", maze_region_vals)
        event_spikes.insert(head_direc_loc, "head_direc", head_direc_vals)
        event_spikes.insert(trial_loc, "trial", trial_vals)

        # Only count the base as being in view when player is outside the base.
        event_spikes.loc[
            (event_spikes["place-Base"] == 1) & (event_spikes["base_in_view"] == 1),
            "base_in_view",
        ] = 0

        # Convert gold_in_view nans at Retrieval to 0.
        event_spikes.loc[
            (event_spikes["gameState"] == "Retrieval")
            & (np.isnan(event_spikes["gold_in_view"])),
            "gold_in_view",
        ] = 0

        # Sort rows.
        event_spikes = event_spikes.sort_values(
            ["trial", "gameState", "time_bin"]
        ).reset_index(drop=True)

        # Get neurons from the session.
        globstr = op.join(
            self.proj_dir,
            "analysis",
            "spikes",
            "{}-*-spikes.pkl".format(self.subj_sess),
        )
        spike_files = {spike_preproc.unit_from_file(_f): _f for _f in glob(globstr)}
        neurons = list(spike_files.keys())

        # Add a column with spike counts for each neuron.
        for neuron in neurons:
            spike_times = dio.open_pickle(spike_files[neuron])["spike_times"]
            event_spikes[neuron] = spikes_per_timebin(event_spikes, spike_times)

        self.n_neurons = len(neurons)
        self.event_spikes = event_spikes


def save_event_spikes(event_spikes, overwrite=False, verbose=True):
    """Pickle an EventSpikes instance."""
    if op.exists(event_spikes.filename) and not overwrite:
        print("Cannot save {} as it already exists".format(event_spikes.filename))
    else:
        dio.save_pickle(event_spikes, event_spikes.filename, verbose)


def load_event_spikes(
    subj_sess,
    proj_dir="/home1/dscho/projects/time_cells",
    filename=None,
    overwrite=False,
    verbose=True,
):
    """Return EventSpikes from saved file or by instantiating anew."""
    if filename is None:
        filename = op.join(
            proj_dir, "analysis", "events", "{}-EventSpikes.pkl".format(subj_sess)
        )

    if op.exists(filename) and not overwrite:
        if verbose:
            print("Loading saved EventSpikes file")
        event_spikes = dio.open_pickle(filename)
    else:
        if verbose:
            print("Creating EventSpikes")
        event_spikes = EventSpikes(subj_sess, proj_dir=proj_dir, filename=filename)
    return event_spikes


def get_ols_delay_formulas(neuron, df):
    """Define model formulas for single-unit to behavior comparisons.

    Parameters
    ----------
    neuron : str
        e.g. '5-2' would be channel 5, unit 2
    df : DataFrame
        Contains the dependent variable and all independent variables.

    Returns
    -------
    Xy : dataframe
        Contains the dependent variable and independent variables, with
        all categorical variables deviation-coded.
    Xycols : dict[list]
        Xy column names, grouped by variable type.
    formulas : dict[str]
        Full and reduced model formulas.
    """
    # Setup the full model.
    param_map = {
        "neuron": neuron,
        "gameState": "C(gameState, Sum)",
        "time": "C(time_step, Sum)",
    }
    formula = strip_space(
        "Q('{_[neuron]}') ~ 1 + {_[gameState]} + {_[time]} + {_[gameState]}:{_[time]}".format(
            _=param_map
        )
    )
    full_mod = ols(formula, data=df)

    # Get the expanded predictor matrix of deviation-coded parameters,
    # and insert the dependent variable column.
    Xy = pd.concat(
        (
            pd.Series(full_mod.endog, name=neuron),
            pd.DataFrame(full_mod.exog, columns=full_mod.exog_names),
        ),
        axis=1,
    )
    Xy.drop(columns=["Intercept"], inplace=True)

    # Rename columns for the Xy dataframe.
    Xycols_old = od(
        [
            ("neuron", [neuron]),
            (
                "gameState",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "time",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "gameState:time",
                [
                    col
                    for col in Xy.columns
                    if np.all([("time" in col), ("gameState" in col), (":" in col)])
                ],
            ),
        ]
    )

    Xycols_new = od([])
    param_map = od([])
    for cols in Xycols_old:
        Xycols_new[cols] = str_replace(
            Xycols_old[cols],
            {
                "C(gameState, Sum)[S.": "gameState_",
                "C(time_step, Sum)[S.": "time_",
                "]": "",
            },
        )
        Xy.rename(
            columns=pd.Series(index=Xycols_old[cols], data=Xycols_new[cols]).to_dict(),
            inplace=True,
        )
        param_map[cols] = " + ".join(Xycols_new[cols])
    Xycols = Xycols_new

    # --------------------------------------------
    # --------------------------------------------
    # Define formulas for full and reduced models.
    formulas = od([])

    # ---------------------------------------------------------
    # full models firing rate as a function of gameState, time,
    # and their interaction.
    formulas["full"] = strip_space(
        "Q('{_[neuron]}') ~ 1 + {_[gameState]} + {_[time]} + {_[gameState:time]}".format(
            _=param_map
        )
    )
    # Fixed firing rate differences between encoding and retrieval, holding constant time and its interaction with gameState.
    formulas["full-gameState"] = strip_space(
        "Q('{_[neuron]}') ~ 1                  + {_[time]} + {_[gameState:time]}".format(
            _=param_map
        )
    )
    # Time cells, with or without remapping, holding constant fixed differences between encoding and retrieval firing.
    formulas["full-time,gameState:time"] = strip_space(
        "Q('{_[neuron]}') ~ 1 + {_[gameState]}                                  ".format(
            _=param_map
        )
    )
    # Context-invariant time cells, holding constant gameState and its interaction with time.
    formulas["full-time"] = strip_space(
        "Q('{_[neuron]}') ~ 1 + {_[gameState]}             + {_[gameState:time]}".format(
            _=param_map
        )
    )
    # Trial state remapping time cells, holding constant time and gameState main effects.
    formulas["full-gameState:time"] = strip_space(
        "Q('{_[neuron]}') ~ 1 + {_[gameState]} + {_[time]}                      ".format(
            _=param_map
        )
    )

    return Xy, Xycols, formulas, param_map


def run_ols_delay(
    subj_sess_unit,
    formula_func=get_ols_delay_formulas,
    game_states=["Delay1", "Delay2"],
    n_perm=1000,
    alpha=0.05,
    mod_names=None,
    mult_by=1,
    proj_dir="/home1/dscho/projects/time_cells",
    save_output=True,
    overwrite=True,
    verbose=False,
):
    """Run time regression analysis for one neuron.

    Returns
    -------
    mod_pairs : dataframe
        Likelihood ratio and empirical p-value for each model comparison.
    ols_weights : dataframe
        Beta weights and Z-scored weights for full model parameters.
    """
    # Load event and neuron info.
    subj_sess, *neuron = subj_sess_unit.split("-")
    neuron = "-".join(neuron)
    event_spikes = load_event_spikes(subj_sess, proj_dir=proj_dir, verbose=verbose)
    hem, *roi = spike_preproc.roi_lookup(
        subj_sess, neuron.split("-")[0], proj_dir=proj_dir
    )
    roi = "".join(roi)

    # List the models that we want to extract weights for.
    if mod_names is None:
        mod_names = ["full"]

    # Fit models to real data.
    ols_mods, Xy, Xycols, *_ = get_ols_delay_mods(
        neuron,
        event_spikes,
        formula_func=formula_func,
        game_states=game_states,
        circshift_frs=False,
    )
    paired_comps = _get_paired_comps("delay")
    mod_pairs = get_ols_mod_pairs(ols_mods, paired_comps)
    ols_weights = get_ols_weights(
        ols_mods, Xycols, mod_names=mod_names, mult_by=mult_by
    )

    # Fit null models.
    mod_pairs_null = []
    ols_weights_null = []
    for iPerm in range(n_perm):
        _ols_mods_null, *_ = get_ols_delay_mods(
            neuron,
            event_spikes,
            formula_func=formula_func,
            game_states=game_states,
            circshift_frs=True,
        )
        mod_pairs_null.append(get_ols_mod_pairs(_ols_mods_null, paired_comps))
        ols_weights_null.append(
            get_ols_weights(
                _ols_mods_null, Xycols, mod_names=mod_names, mult_by=mult_by
            )
        )
    mod_pairs_null = pd.concat(mod_pairs_null, axis=0).reset_index(drop=True)
    ols_weights_null = pd.concat(ols_weights_null, axis=0).reset_index(drop=True)

    # Test significance and Z-score weights against the null distribution.
    mod_pairs[["z_lr", "emp_pval"]] = mod_pairs.apply(
        lambda x: _get_emp_pval(x, mod_pairs_null), axis=1
    ).tolist()
    mod_pairs["sig"] = mod_pairs["emp_pval"].apply(lambda x: x < alpha)
    ols_weights["z_weight"] = ols_weights.apply(
        lambda x: _zscore_weights(x, ols_weights_null), axis=1
    )

    # Add unique identifiers.
    for _df in (mod_pairs, ols_weights):
        _df.insert(0, "subj_sess_unit", subj_sess_unit)
        _df.insert(1, "hem", hem)
        _df.insert(2, "roi", roi)
        _df.insert(3, "gameState", "-".join(game_states))

    # Save outputs.
    if save_output:
        mod_pairs_file = op.join(
            proj_dir,
            "analysis",
            "unit_to_behav",
            "{}-{}-{}-ols_model_pairs.pkl".format(
                subj_sess, neuron, "_".join(game_states)
            ),
        )
        ols_weights_file = op.join(
            proj_dir,
            "analysis",
            "unit_to_behav",
            "{}-{}-{}-ols_weights.pkl".format(subj_sess, neuron, "_".join(game_states)),
        )
        if overwrite or not op.exists(mod_pairs_file):
            dio.save_pickle(mod_pairs, mod_pairs_file, verbose)
        if overwrite or not op.exists(ols_weights_file):
            dio.save_pickle(ols_weights, ols_weights_file, verbose)

    return mod_pairs, ols_weights


def get_ols_delay_mods(
    neuron,
    event_spikes,
    formula_func,
    game_states=["Delay1", "Delay2"],
    event_spikes_idx=None,
    circshift_frs=False,
):
    """Find the most important time bin.
    Fit firing rates using OLS regression, iteratively removing
    each time step from the data and recording

    Parameters
    ----------
    neuron : str
        e.g. '5-2' would be channel 5, unit 2
    event_spikes : pd.DataFrame
        EventSpikes instance that contains the event_spikes dataframe,
        an expanded version of the behav_events dataframe with columns
        added for each neuron.
    game_states : list[str]
        Delay1, Encoding, Delay2, or Retrieval.
    event_spikes_idx : list
        Only event_spikes rows that correspond to the provided index
        labels are used in the regression model. Overrides querying
        by gameState.

    Returns
    -------
    ols_mods : dict[statsmodels.regression.linear_model.OLS]
    Xy : dataframe
        Contains the dependent variable and independent variables, with
        all categorical variables deviation-coded.
    Xycols : dict[list]
        Xy column names, grouped by variable type.
    formulas : dict[str]
        Full and reduced model formulas.
    """
    # Select rows for the chosen game states.
    if type(game_states) == str:
        game_states = [game_states]
    if event_spikes_idx is None:
        _event_spikes = event_spikes.event_spikes.query(
            "(gameState=={})".format(game_states)
        ).copy()
    else:
        _event_spikes = event_spikes.event_spikes.loc[event_spikes_idx].copy()
    _event_spikes["gameState"] = _event_spikes["gameState"].astype(str)

    # For delays, sum spikes across time bins within each time step, within each trial.
    _event_spikes = (
        _event_spikes.groupby(["gameState", "trial", "time_step"])
        .agg({neuron: np.sum})
        .reset_index()
    )

    # Circ-shift firing rates within each trial interval,
    # then shuffle trial intervals across game states.
    if circshift_frs:
        _event_spikes[neuron] = _circshift_shuffle(
            _event_spikes, neuron, ["trial", "gameState"]
        )

    # Fit the model.
    Xy, Xycols, formulas, param_map = formula_func(neuron, _event_spikes)
    ols_mods = od([])
    for mod, formula in formulas.items():
        ols_mods[mod] = ols(formula, data=Xy)

    return ols_mods, Xy, Xycols, formulas, param_map


def get_ols_nav_formulas(neuron, df):
    """Define model formulas for single-unit to behavior comparisons.

    Parameters
    ----------
    neuron : str
        e.g. '5-2' would be channel 5, unit 2
    df : DataFrame
        Contains the dependent variable and all independent variables.

    Returns
    -------
    Xy : dataframe
        Contains the dependent variable and independent variables, with
        all categorical variables deviation-coded.
    Xycols : dict[list]
        Xy column names, grouped by variable type.
    formulas : dict[str]
        Full and reduced model formulas.
    """
    # ------------------------------------------------------------------------
    # Setup a model with all parameters that we want to use across all models.
    param_map = {
        "neuron": neuron,
        "gameState": "C(gameState, Sum)",
        "time": "C(time_step, Sum)",
        "place": "C(maze_region, Sum)",
        "headDirec": "C(head_direc, Sum)",
        "isMoving": "C(is_moving, Sum(0))",
        "baseView": "C(base_in_view, Sum(0))",
        "goldView": "C(gold_in_view, Sum(0))",
        "digAction": "C(dig_performed, Sum(0))",
        "penalty": "C(time_penalty, Sum(0))",
        "goldDug": "gold_dug",
    }
    formula = strip_space(
        "Q('{_[neuron]}') ~ 1 + {_[gameState]}                      + {_[time]}                             + {_[place]}                             + "
        "                       {_[penalty]}                        + {_[goldDug]}                          + {_[headDirec]}                         + "
        "                       {_[isMoving]}                       + {_[baseView]}                         + {_[goldView]}                          + "
        "                                                                                                     {_[digAction]}                         + "
        "                       {_[gameState]}:{_[time]}            + {_[gameState]}:{_[place]}             + {_[time]}:{_[place]}                   + "
        "                       {_[gameState]}:{_[penalty]}         + {_[time]}:{_[penalty]}                + {_[place]}:{_[penalty]}                + "
        "                       {_[gameState]}:{_[goldDug]}         + {_[time]}:{_[goldDug]}                + {_[place]}:{_[goldDug]}                + "
        "                       {_[gameState]}:{_[headDirec]}       + {_[gameState]}:{_[isMoving]}          + {_[gameState]}:{_[baseView]}             ".format(
            _=param_map
        )
    )
    mod = ols(formula, data=df)

    # Get the expanded predictor matrix of deviation-coded parameters,
    # and insert the dependent variable column.
    Xy = pd.concat(
        (
            pd.Series(mod.endog, name=neuron),
            pd.DataFrame(mod.exog, columns=mod.exog_names),
        ),
        axis=1,
    )
    Xy.drop(columns=["Intercept"], inplace=True)

    # ------------------------------------
    # Rename columns for the Xy dataframe.
    Xycols_old = od(
        [
            ("neuron", [neuron]),
            (
                "gameState",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "time",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "place",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "penalty",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "goldDug",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "headDirec",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "isMoving",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "baseView",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "goldView",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "digAction",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" not in col),
                        ]
                    )
                ],
            ),
            (
                "gameState:time",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "gameState:place",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" not in col),
                            ("maze_region" in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "time:place",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" in col),
                            ("maze_region" in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "gameState:penalty",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "time:penalty",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "place:penalty",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "gameState:goldDug",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "time:goldDug",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "place:goldDug",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" not in col),
                            ("time_step" not in col),
                            ("maze_region" in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "gameState:headDirec",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" in col),
                            ("is_moving" not in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "gameState:isMoving",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" in col),
                            ("base_in_view" not in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
            (
                "gameState:baseView",
                [
                    col
                    for col in Xy.columns
                    if np.all(
                        [
                            ("gameState" in col),
                            ("time_step" not in col),
                            ("maze_region" not in col),
                            ("head_direc" not in col),
                            ("is_moving" not in col),
                            ("base_in_view" in col),
                            ("gold_in_view" not in col),
                            ("dig_performed" not in col),
                            ("time_penalty" not in col),
                            ("gold_dug" not in col),
                            (":" in col),
                        ]
                    )
                ],
            ),
        ]
    )

    Xycols_new = od([])
    param_map = od([])
    for cols in Xycols_old:
        Xycols_new[cols] = str_replace(
            Xycols_old[cols],
            {
                "C(gameState, Sum)[S.": "gameState_",
                "C(time_step, Sum)[S.": "time_",
                "C(maze_region, Sum)[S.": "place_",
                "C(time_penalty, Sum(0))[S.": "penalty_",
                "gold_dug": "goldDug",
                "C(head_direc, Sum)[S.": "headDirec_",
                "C(is_moving, Sum(0))[S.": "isMoving_",
                "C(base_in_view, Sum(0))[S.": "baseView_",
                "C(gold_in_view, Sum(0))[S.": "goldView_",
                "C(dig_performed, Sum(0))[S.": "digAction_",
                "]": "",
            },
        )
        Xy.rename(
            columns=pd.Series(index=Xycols_old[cols], data=Xycols_new[cols]).to_dict(),
            inplace=True,
        )
        param_map[cols] = " + ".join(Xycols_new[cols])
    Xycols = Xycols_new

    # --------------------------------------------
    # --------------------------------------------
    # Define formulas for full and reduced models.
    formulas = od([])

    # ----------------------------------------------------------------
    # full models firing rate as a function of gameState, time, place,
    # and their first-order interactions.
    formulas["full"] = strip_space(
        "Q('{_[neuron]}') ~ 1 + {_[gameState]}      + {_[time]}            + {_[place]}             + "
        "                       {_[gameState:time]} + {_[gameState:place]} + {_[time:place]}          ".format(
            _=param_map
        )
    )
    # Fixed firing rate differences between encoding and retrieval, holding constant
    # time, place, and their interactions with gameState.
    formulas["full-gameState"] = " + ".join(
        [
            val
            for val in formulas["full"].split(" + ")
            if (val not in param_map["gameState"].split(" + "))
        ]
    )
    # Time cells, with or without remapping, holding constant place and time:place interactions.
    formulas["full-time,gameState:time"] = " + ".join(
        [
            val
            for val in formulas["full"].split(" + ")
            if (
                val
                not in (
                    param_map["time"].split(" + ")
                    + param_map["gameState:time"].split(" + ")
                )
            )
        ]
    )
    # Context-invariant time cells, holding constant place, gameState, and their interactions with time.
    formulas["full-time"] = " + ".join(
        [
            val
            for val in formulas["full"].split(" + ")
            if (val not in param_map["time"].split(" + "))
        ]
    )
    # Trial state remapping time cells, holding constant place and time:place interactions.
    formulas["full-gameState:time"] = " + ".join(
        [
            val
            for val in formulas["full"].split(" + ")
            if (val not in param_map["gameState:time"].split(" + "))
        ]
    )
    # Place cells, with or without remapping, holding constant time and time:place interactions.
    formulas["full-place,gameState:place"] = " + ".join(
        [
            val
            for val in formulas["full"].split(" + ")
            if (
                val
                not in (
                    param_map["place"].split(" + ")
                    + param_map["gameState:place"].split(" + ")
                )
            )
        ]
    )
    # Context-invariant place cells, holding constant time, gameState, and their interactions with place.
    formulas["full-place"] = " + ".join(
        [
            val
            for val in formulas["full"].split(" + ")
            if (val not in param_map["place"].split(" + "))
        ]
    )
    # Trial state remapping place cells, holding constant time and time:place interactions.
    formulas["full-gameState:place"] = " + ".join(
        [
            val
            for val in formulas["full"].split(" + ")
            if (val not in param_map["gameState:place"].split(" + "))
        ]
    )
    # Joint time x place coding.
    formulas["full-time:place"] = " + ".join(
        [
            val
            for val in formulas["full"].split(" + ")
            if (val not in param_map["time:place"].split(" + "))
        ]
    )

    # ---------------------------------------------------------------------
    # full+penalty is the same as full but adds parameters for time penalty
    # and associated interactions of interest.
    formulas["full+penalty"] = strip_space(
        formulas["full"]
        + " +                                                     {_[penalty]}              + "
        "{_[gameState:penalty]} + {_[time:penalty]}           + {_[place:penalty]}          ".format(
            _=param_map
        )
    )
    # Interaction between time and penalty.
    formulas["full+penalty-time:penalty"] = " + ".join(
        [
            val
            for val in formulas["full+penalty"].split(" + ")
            if (val not in param_map["time:penalty"].split(" + "))
        ]
    )
    # Interaction between place and penalty.
    formulas["full+penalty-place:penalty"] = " + ".join(
        [
            val
            for val in formulas["full+penalty"].split(" + ")
            if (val not in param_map["place:penalty"].split(" + "))
        ]
    )

    # ----------------------------------------------------------------------
    # full+goldDug is the same as full but adds parameters for the number of
    # golds dug per trial and associated interactions of interest.
    formulas["full+goldDug"] = strip_space(
        formulas["full"]
        + " +                                                     {_[goldDug]}              + "
        "{_[gameState:goldDug]} + {_[time:goldDug]}           + {_[place:goldDug]}          ".format(
            _=param_map
        )
    )
    # Interaction between time and goldDug.
    formulas["full+goldDug-time:goldDug"] = " + ".join(
        [
            val
            for val in formulas["full+goldDug"].split(" + ")
            if (val not in param_map["time:goldDug"].split(" + "))
        ]
    )
    # Interaction between place and goldDug.
    formulas["full+goldDug-place:goldDug"] = " + ".join(
        [
            val
            for val in formulas["full+goldDug"].split(" + ")
            if (val not in param_map["place:goldDug"].split(" + "))
        ]
    )

    # -------------------------------------------------------------------
    # fullMax models firing rate as a function of gameState, time, place,
    # and their first-order interactions, along with
    # nuissance covariates and nuissance interactions.
    formulas["fullMax"] = strip_space(
        "Q('{_[neuron]}') ~ 1 + {_[gameState]}           + {_[time]}               + {_[place]}              + "
        "                       {_[headDirec]}           + {_[isMoving]}           + {_[baseView]}           + "
        "                                                  {_[goldView]}           + {_[digAction]}          + "
        "                       {_[gameState:time]}      + {_[gameState:place]}    + {_[time:place]}         + "
        "                       {_[gameState:headDirec]} + {_[gameState:isMoving]} + {_[gameState:baseView]}   ".format(
            _=param_map
        )
    )
    # Fixed firing rate differences between encoding and retrieval, holding constant
    # time, place, and their interactions with gameState.
    formulas["fullMax-gameState"] = " + ".join(
        [
            val
            for val in formulas["fullMax"].split(" + ")
            if (val not in param_map["gameState"].split(" + "))
        ]
    )
    # Time cells, with or without remapping, holding constant place and time:place interactions.
    formulas["fullMax-time,gameState:time"] = " + ".join(
        [
            val
            for val in formulas["fullMax"].split(" + ")
            if (
                val
                not in (
                    param_map["time"].split(" + ")
                    + param_map["gameState:time"].split(" + ")
                )
            )
        ]
    )
    # Context-invariant time cells, holding constant place, gameState, and their interactions with time.
    formulas["fullMax-time"] = " + ".join(
        [
            val
            for val in formulas["fullMax"].split(" + ")
            if (val not in param_map["time"].split(" + "))
        ]
    )
    # Trial state remapping time cells, holding constant place and time:place interactions.
    formulas["fullMax-gameState:time"] = " + ".join(
        [
            val
            for val in formulas["fullMax"].split(" + ")
            if (val not in param_map["gameState:time"].split(" + "))
        ]
    )
    # Place cells, with or without remapping, holding constant time and time:place interactions.
    formulas["fullMax-place,gameState:place"] = " + ".join(
        [
            val
            for val in formulas["fullMax"].split(" + ")
            if (
                val
                not in (
                    param_map["place"].split(" + ")
                    + param_map["gameState:place"].split(" + ")
                )
            )
        ]
    )
    # Context-invariant place cells, holding constant time, gameState, and their interactions with place.
    formulas["fullMax-place"] = " + ".join(
        [
            val
            for val in formulas["fullMax"].split(" + ")
            if (val not in param_map["place"].split(" + "))
        ]
    )
    # Trial state remapping place cells, holding constant time and time:place interactions.
    formulas["fullMax-gameState:place"] = " + ".join(
        [
            val
            for val in formulas["fullMax"].split(" + ")
            if (val not in param_map["gameState:place"].split(" + "))
        ]
    )
    # Joint time x place coding.
    formulas["fullMax-time:place"] = " + ".join(
        [
            val
            for val in formulas["fullMax"].split(" + ")
            if (val not in param_map["time:place"].split(" + "))
        ]
    )

    # ---------------------------------------------------------------------
    # fullMax+penalty is the same as fullMax but adds parameters for time penalty
    # and associated interactions of interest.
    formulas["fullMax+penalty"] = strip_space(
        formulas["fullMax"]
        + " +                                                     {_[penalty]}              + "
        "{_[gameState:penalty]} + {_[time:penalty]}           + {_[place:penalty]}          ".format(
            _=param_map
        )
    )
    # Interaction between time and penalty.
    formulas["fullMax+penalty-time:penalty"] = " + ".join(
        [
            val
            for val in formulas["fullMax+penalty"].split(" + ")
            if (val not in param_map["time:penalty"].split(" + "))
        ]
    )
    # Interaction between place and penalty.
    formulas["fullMax+penalty-place:penalty"] = " + ".join(
        [
            val
            for val in formulas["fullMax+penalty"].split(" + ")
            if (val not in param_map["place:penalty"].split(" + "))
        ]
    )

    # ----------------------------------------------------------------------
    # fullMax+goldDug is the same as fullMax but adds parameters for the number of
    # golds dug per trial and associated interactions of interest.
    formulas["fullMax+goldDug"] = strip_space(
        formulas["fullMax"]
        + " +                                                     {_[goldDug]}              + "
        "{_[gameState:goldDug]} + {_[time:goldDug]}           + {_[place:goldDug]}          ".format(
            _=param_map
        )
    )
    # Interaction between time and goldDug.
    formulas["fullMax+goldDug-time:goldDug"] = " + ".join(
        [
            val
            for val in formulas["fullMax+goldDug"].split(" + ")
            if (val not in param_map["time:goldDug"].split(" + "))
        ]
    )
    # Interaction between place and goldDug.
    formulas["fullMax+goldDug-place:goldDug"] = " + ".join(
        [
            val
            for val in formulas["fullMax+goldDug"].split(" + ")
            if (val not in param_map["place:goldDug"].split(" + "))
        ]
    )

    return Xy, Xycols, formulas, param_map


def run_ols_nav(
    subj_sess_unit,
    game_states=["Encoding", "Retrieval"],
    formula_func=get_ols_nav_formulas,
    n_perm=1000,
    alpha=0.05,
    mod_names=None,
    mult_by=2,
    proj_dir="/home1/dscho/projects/time_cells",
    save_output=True,
    overwrite=True,
    verbose=False,
):
    """Run time x place regression analysis for one neuron.

    Returns
    -------
    mod_pairs : dataframe
        Likelihood ratio and empirical p-value for each model comparison.
    ols_weights : dataframe
        Beta weights and Z-scored weights for full model parameters.
    """
    # Load event and neuron info.
    subj_sess, *neuron = subj_sess_unit.split("-")
    neuron = "-".join(neuron)
    events = events_proc.load_events(subj_sess, proj_dir=proj_dir, verbose=verbose)
    event_spikes = load_event_spikes(subj_sess, proj_dir=proj_dir, verbose=verbose)
    hem, *roi = spike_preproc.roi_lookup(
        subj_sess, neuron.split("-")[0], proj_dir=proj_dir
    )
    roi = "".join(roi)

    # List the models that we want to extract weights for.
    if mod_names is None:
        mod_names = [
            "full",
            "full+penalty",
            "full+goldDug",
            "fullMax",
            "fullMax+penalty",
            "fullMax+goldDug",
        ]

    # Fit models to real data.
    ols_mods, Xy, Xycols, *_ = get_ols_nav_mods(
        neuron,
        events,
        event_spikes,
        formula_func=formula_func,
        game_states=game_states,
        circshift_frs=False,
    )
    paired_comps = _get_paired_comps("nav")
    mod_pairs = get_ols_mod_pairs(ols_mods, paired_comps)
    ols_weights = get_ols_weights(
        ols_mods, Xycols, mod_names=mod_names, mult_by=mult_by
    )

    # Fit null models.
    mod_pairs_null = []
    ols_weights_null = []
    for iPerm in range(n_perm):
        _ols_mods_null, *_ = get_ols_nav_mods(
            neuron,
            events,
            event_spikes,
            formula_func=formula_func,
            game_states=game_states,
            circshift_frs=True,
        )
        mod_pairs_null.append(get_ols_mod_pairs(_ols_mods_null, paired_comps))
        ols_weights_null.append(
            get_ols_weights(
                _ols_mods_null, Xycols, mod_names=mod_names, mult_by=mult_by
            )
        )
    mod_pairs_null = pd.concat(mod_pairs_null, axis=0).reset_index(drop=True)
    ols_weights_null = pd.concat(ols_weights_null, axis=0).reset_index(drop=True)

    # Test significance and Z-score weights against the null distribution.
    mod_pairs[["z_lr", "emp_pval"]] = mod_pairs.apply(
        lambda x: _get_emp_pval(x, mod_pairs_null), axis=1
    ).tolist()
    mod_pairs["sig"] = mod_pairs["emp_pval"].apply(lambda x: x < alpha)
    ols_weights["z_weight"] = ols_weights.apply(
        lambda x: _zscore_weights(x, ols_weights_null), axis=1
    )

    # Add unique identifiers.
    for _df in (mod_pairs, ols_weights):
        _df.insert(0, "subj_sess_unit", subj_sess_unit)
        _df.insert(1, "hem", hem)
        _df.insert(2, "roi", roi)
        _df.insert(3, "gameState", "-".join(game_states))

    # Save outputs.
    if save_output:
        mod_pairs_file = op.join(
            proj_dir,
            "analysis",
            "unit_to_behav",
            "{}-{}-{}-ols_model_pairs.pkl".format(
                subj_sess, neuron, "_".join(game_states)
            ),
        )
        ols_weights_file = op.join(
            proj_dir,
            "analysis",
            "unit_to_behav",
            "{}-{}-{}-ols_weights.pkl".format(subj_sess, neuron, "_".join(game_states)),
        )
        if overwrite or not op.exists(mod_pairs_file):
            dio.save_pickle(mod_pairs, mod_pairs_file, verbose)
        if overwrite or not op.exists(ols_weights_file):
            dio.save_pickle(ols_weights, ols_weights_file, verbose)

    return mod_pairs, ols_weights


def get_ols_nav_mods(
    neuron,
    events,
    event_spikes,
    formula_func,
    game_states=["Encoding", "Retrieval"],
    event_spikes_idx=None,
    circshift_frs=False,
):
    """Find the most important time bin.
    Fit firing rates using OLS regression, iteratively removing
    each time step from the data and recording

    Parameters
    ----------
    neuron : str
        e.g. '5-2' would be channel 5, unit 2
    event_spikes : pd.DataFrame
        EventSpikes instance that contains the event_spikes dataframe,
        an expanded version of the behav_events dataframe with columns
        added for each neuron.
    game_states : list[str]
        Delay1, Encoding, Delay2, or Retrieval.
    event_spikes_idx : list
        Only event_spikes rows that correspond to the provided index
        labels are used in the regression model. Overrides querying
        by gameState.

    Returns
    -------
    ols_mods : dict[statsmodels.regression.linear_model.OLS]
    Xy : dataframe
        Contains the dependent variable and independent variables, with
        all categorical variables deviation-coded.
    Xycols : dict[list]
        Xy column names, grouped by variable type.
    formulas : dict[str]
        Full and reduced model formulas.
    """
    warnings.filterwarnings("ignore")

    # Select rows for the chosen game states.
    if type(game_states) == str:
        game_states = [game_states]
    if event_spikes_idx is None:
        _event_spikes = (
            event_spikes.event_spikes.query("(gameState=={})".format(game_states))
            .reset_index(drop=True)
            .copy()
        )
    else:
        _event_spikes = (
            event_spikes.event_spikes.loc[event_spikes_idx]
            .reset_index(drop=True)
            .copy()
        )

    # Set retrieval gold views to 0.
    _event_spikes.loc[
        (_event_spikes["gameState"] == "Retrieval")
        & (_event_spikes["gold_in_view"] == 1),
        "gold_in_view",
    ] = 0

    # Add a column to track how many golds were successfully dug on each trial.
    golds_dug_per_trial = (
        events.dig_events.query("(dig_success==True)").groupby("trial").size()
    )
    _event_spikes.insert(
        4,
        "gold_dug",
        _event_spikes["trial"].apply(lambda x: golds_dug_per_trial.get(x, 0)),
    )

    # Select columns to keep.
    keep_cols = [
        "trial",
        "gameState",
        "time_step",
        "maze_region",
        "head_direc",
        "is_moving",
        "base_in_view",
        "gold_in_view",
        "dig_performed",
        "time_penalty",
        "gold_dug",
        neuron,
    ]
    _event_spikes = _event_spikes[keep_cols]

    # Set nans to 0.
    fill_cols = [
        "is_moving",
        "base_in_view",
        "gold_in_view",
        "dig_performed",
        "time_penalty",
    ]
    _event_spikes[fill_cols] = _event_spikes[fill_cols].fillna(0)

    # Reset column datatypes.
    _event_spikes["gameState"] = _event_spikes["gameState"].astype(str)
    reset_cols = [
        "is_moving",
        "base_in_view",
        "gold_in_view",
        "dig_performed",
        "time_penalty",
    ]
    for col in reset_cols:
        _event_spikes[col] = _event_spikes[col].astype(int)

    # Circ-shift firing rates within each trial interval,
    # then shuffle trial intervals across game states.
    if circshift_frs:
        _event_spikes[neuron] = _circshift_shuffle(
            _event_spikes, neuron, ["trial", "gameState"]
        )

    # Fit the model.
    Xy, Xycols, formulas, param_map = formula_func(neuron, _event_spikes)
    ols_mods = od([])
    for mod, formula in formulas.items():
        ols_mods[mod] = ols(formula, data=Xy)

    warnings.resetwarnings()

    return ols_mods, Xy, Xycols, formulas, param_map


def get_ols_mod_pairs(ols_mods, paired_comps):
    """Return a dataframe of likehood ratios between nested model pairs."""
    # Fit models.
    mods_to_fit = np.unique(
        [c for comps in paired_comps for c in comps if (c in ols_mods)]
    )
    mod_fits = od([])
    for mod_name in mods_to_fit:
        mod_fits[mod_name] = ols_mods[mod_name].fit()

    # Get likelihood ratios between model pairs.
    mod_pairs = []
    for full, red in paired_comps:
        llf_full = mod_fits[full].llf
        llf_red = mod_fits[red].llf
        lr = get_lr(mod_fits[red], mod_fits[full])
        lr, df, chi_pval = lr_test(mod_fits[red], mod_fits[full])
        lr = np.max((lr, 0))
        mod_pairs.append([full, red, llf_full, llf_red, df, lr])
    mod_pairs = pd.DataFrame(
        mod_pairs, columns=["full", "red", "llf_full", "llf_red", "df", "lr"]
    )
    return mod_pairs


def get_ols_weights(ols_mods, Xycols, mod_names=["full"], include_icpt=True, mult_by=1):
    """Return a dataframe of weights for the chosen models."""

    # Hard-code the levels for time and place.
    times = [
        "time_1",
        "time_2",
        "time_3",
        "time_4",
        "time_5",
        "time_6",
        "time_7",
        "time_8",
        "time_9",
        "time_10",
    ]
    places = [
        "place_Base",
        "place_C_Hall",
        "place_NE_Hall",
        "place_NE_Room",
        "place_NW_Hall",
        "place_NW_Room",
        "place_N_Passage",
        "place_SE_Hall",
        "place_SE_Room",
        "place_SW_Hall",
        "place_SW_Room",
        "place_S_Passage",
    ]

    mod_names = [m for m in mod_names if (m in ols_mods)]
    mod_weights = []
    for mod_name in mod_names:
        param_map = _get_param_map(ols_mods[mod_name], Xycols)
        mod_fit = ols_mods[mod_name].fit()
        params = mod_fit.params
        if include_icpt:
            mod_weights.append([mod_name, "icpt", "icpt", params["Intercept"]])
        for factor, levels in param_map.items():
            for level in levels:
                if level in params:
                    weight = params[level]
                else:
                    if factor == "time:place":
                        _time, _place = level.split(":")
                        if _time != "time_10":
                            contrast_levels = [
                                "{}:{}".format(_time, x) for x in places[:-1]
                            ]
                            weight = -np.sum(params[contrast_levels])
                        elif _place != "place_S_Passage":
                            contrast_levels = [
                                "{}:{}".format(x, _place) for x in times[:-1]
                            ]
                            weight = -np.sum(params[contrast_levels])
                        else:
                            contrast_levels = [x for x in levels if x in params]
                            weight = np.sum(params[contrast_levels])
                    else:
                        contrast_levels = [x for x in levels if (x != level)]
                        weight = -np.sum(params[contrast_levels])

                mod_weights.append([mod_name, factor, level, weight])
    mod_weights = pd.DataFrame(
        mod_weights, columns=["model", "factor", "level", "weight"]
    )
    mod_weights["weight"] = mod_weights["weight"].astype(float)
    mod_weights["weight"] *= mult_by  # convert to firing rate in Hz

    return mod_weights


def _get_paired_comps(compare):
    """Return tuple pairs of full, reduced models to compare.

    Parameters
    ----------
    compare : str
        'delay' or 'nav'
    """
    if compare == "delay":
        paired_comps = [
            ("full", "full-gameState"),
            ("full", "full-time,gameState:time"),
            ("full", "full-time"),
            ("full", "full-gameState:time"),
        ]
    elif compare == "nav":
        paired_comps = [
            ("full", "full-gameState"),
            ("full", "full-time,gameState:time"),
            ("full", "full-time"),
            ("full", "full-gameState:time"),
            ("full", "full-place,gameState:place"),
            ("full", "full-place"),
            ("full", "full-gameState:place"),
            ("full", "full-time:place"),
            ("full+penalty", "full+penalty-time:penalty"),
            ("full+penalty", "full+penalty-place:penalty"),
            ("full+goldDug", "full+goldDug-time:goldDug"),
            ("full+goldDug", "full+goldDug-place:goldDug"),
            ("fullMax", "fullMax-gameState"),
            ("fullMax", "fullMax-time,gameState:time"),
            ("fullMax", "fullMax-time"),
            ("fullMax", "fullMax-gameState:time"),
            ("fullMax", "fullMax-place,gameState:place"),
            ("fullMax", "fullMax-place"),
            ("fullMax", "fullMax-gameState:place"),
            ("fullMax", "fullMax-time:place"),
            ("fullMax+penalty", "fullMax+penalty-time:penalty"),
            ("fullMax+penalty", "fullMax+penalty-place:penalty"),
            ("fullMax+goldDug", "fullMax+goldDug-time:goldDug"),
            ("fullMax+goldDug", "fullMax+goldDug-place:goldDug"),
        ]

    return paired_comps


def _get_param_map(ols_mod, Xycols):
    """Add dropped columns for value-coded variables.

    We're just hard-coding in these columns based on knowing the
    event_spikes dataframe structure.
    """
    param_map = copy.deepcopy(Xycols)
    param_map = od({k: v for k, v in param_map.items() if (v[0] in ols_mod.exog_names)})

    # Hard-code the levels for time and place.
    times = [
        "time_1",
        "time_2",
        "time_3",
        "time_4",
        "time_5",
        "time_6",
        "time_7",
        "time_8",
        "time_9",
        "time_10",
    ]
    places = [
        "place_Base",
        "place_C_Hall",
        "place_NE_Hall",
        "place_NE_Room",
        "place_NW_Hall",
        "place_NW_Room",
        "place_N_Passage",
        "place_SE_Hall",
        "place_SE_Room",
        "place_SW_Hall",
        "place_SW_Room",
        "place_S_Passage",
    ]

    if "time" in param_map:
        param_map["time"].append("time_10")
    if "place" in param_map:
        param_map["place"].append("place_S_Passage")
    if "headDirec" in param_map:
        param_map["headDirec"].append("headDirec_W")
    if "gameState:time" in param_map:
        param_map["gameState:time"].append(
            "{}:time_10".format(param_map["gameState"][0])
        )
    if "gameState:place" in param_map:
        param_map["gameState:place"].append(
            "{}:place_S_Passage".format(param_map["gameState"][0])
        )
    if "time:place" in param_map:
        param_map["time:place"] = []
        for _place in places:
            for _time in times:
                time_place = "{}:{}".format(_time, _place)
                param_map["time:place"].append(time_place)
    if "time:penalty" in param_map:
        param_map["time:penalty"].append("time_10:penalty_1")
    if "place:penalty" in param_map:
        param_map["place:penalty"].append("place_S_Passage:penalty_1")
    if "time:goldDug" in param_map:
        param_map["time:goldDug"].append("time_10:gold_dug")
    if "place:goldDug" in param_map:
        param_map["place:goldDug"].append("place_S_Passage:gold_dug")
    if "gameState:headDirec" in param_map:
        param_map["gameState:headDirec"].append(
            "{}:headDirec_W".format(param_map["gameState"][0])
        )

    return param_map


def _get_emp_pval(x, mod_pairs_null):
    null_lrs = mod_pairs_null.query(
        "(full=='{}') & (red=='{}')".format(x["full"], x["red"])
    )["lr"].values
    n_perm = null_lrs.size
    z_lr = (x["lr"] - np.nanmean(null_lrs)) / np.nanstd(null_lrs)
    pval_ind = np.sum(null_lrs >= x["lr"])
    emp_pval = (pval_ind + 1) / (n_perm + 1)
    return z_lr, emp_pval


def _zscore_weights(x, ols_weights_null):
    null_weights = ols_weights_null.query(
        "(model=='{}') & (factor=='{}') & (level=='{}')".format(
            x["model"], x["factor"], x["level"]
        )
    )["weight"].values
    null_mean = np.nanmean(null_weights)
    null_std = np.nanstd(null_weights)
    if null_std < 1e-12:
        z_weight = 0
    else:
        z_weight = (x["weight"] - null_mean) / null_std
    return z_weight


def model_unit_fr(
    subj_sess,
    neuron,
    game_states=["Delay1", "Delay2", "Encoding", "Retrieval"],
    model="ols",
    n_perm=1000,
    proj_dir="/home1/dscho/projects/time_cells",
    filename=None,
    overwrite=False,
    save_output=True,
    verbose=False,
    **kwargs
):
    """Predict a neuron's firing rate from behavioral variables.

    kwargs are passed to the model fitting function.

    Parameters
    ----------
    subj_sess : str
        e.g. 'U518-ses0'
    neuron : str
        e.g. '17-1' is channel 17, unit 1.
    game_states : list
        One or more game_states that you want to fit models for.
    model : str
        Determines the type of regression model used. 'ols' or 'glm'
    n_perm : positive int
        The number of permutations to draw for the null distribution.

    Returns
    -------
    model_pairs : DataFrame
    """
    # Load the output file if it exists.
    if filename is None:
        filename = op.join(
            proj_dir,
            "analysis",
            "unit_to_behav",
            "{}-{}-{}-time_bin-model_pairs.pkl".format(subj_sess, neuron, model),
        )

    if op.exists(filename) and not overwrite:
        if verbose:
            print("Loading from pickle.")
        return dio.open_pickle(filename)

    # Load the event_spikes dataframe.
    event_spikes = load_event_spikes(subj_sess)

    # Reduce dataframe size to just the columns we need.
    if model == "ols":
        keep_cols = [
            "trial",
            "gameState",
            "time_step",
            "maze_region",
            "head_direc",
            "is_moving",
            "base_in_view",
            "gold_in_view",
            "dig_performed",
            neuron,
        ]
        event_spikes.event_spikes = event_spikes.event_spikes[keep_cols]

    # Define the model fitting function.
    if model == "ols":
        model_func = ols_fit
    elif model == "glm":
        model_func = glm_fit
        warnings.filterwarnings("ignore")

    # Fit models for the chosen game states.
    model_fits = od([])
    for game_state in game_states:
        if verbose:
            print("\tfitting {} {}...".format(game_state, model.upper()))
        model_fits[game_state] = model_func(
            neuron, event_spikes, game_state, n_perm=n_perm, **kwargs
        )
    warnings.resetwarnings()

    # Compare results for preselected model contrasts, and store
    # the results in a dataframe.
    model_pairs = get_model_pairs(model_fits)

    # Save the model_pairs dataframe.
    if save_output:
        dio.save_pickle(model_pairs, filename, verbose)

    return model_pairs


def ols_fit(
    neuron,
    event_spikes,
    game_state,
    n_perm=0,
    regress_trial=False,
    perm_method="circshift",
    event_spikes_idx=None,
    mod_names=None,
):
    """Fit firing rates using OLS regression.

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
        The number of permutations to draw for the null distribution.
    regress_trial : bool
        If True, GLM includes a dummy-coded predictor for each trial.
        If False, no trial predictors are added.
    perm_method : str
        Determines how the firing rate vector will be randomized to
        obtain a null distribution. 'circshift' or 'shuffle'
    event_spikes_idx : list
        Only event_spikes rows that correspond to the provided index
        labels are used in the regression model.
    mod_names : list
        Will run only models defined in both mod_names and formulas;
        otherwise runs models for all formulas.

    Returns
    -------
    model_fits : dict[RegressionResultsWrapper]
        Contains model fits from real and shuffled spike data.
    """
    # Select rows for the chosen game state.
    if event_spikes_idx is None:
        df = event_spikes.event_spikes.query(
            "(gameState=='{}')".format(game_state)
        ).copy()
    else:
        df = event_spikes.event_spikes.loc[event_spikes_idx].copy()

    # For delays, sum spikes across time bins within each time step, within each trial.
    if game_state in ["Delay1", "Delay2"]:
        df = df.groupby(["trial", "time_step"]).agg({neuron: np.sum}).reset_index()

    # Make sure there are no nans in the dummy columns.
    if game_state in ["Encoding", "Retrieval"]:
        dummys = ["is_moving", "base_in_view", "gold_in_view", "dig_performed"]
        df[dummys] = df[dummys].fillna(0)

    # Define the model formulas.
    trial_term = " + C(trial)" if regress_trial else ""
    formulas = od([])
    if game_state in ["Delay1", "Delay2"]:
        formulas["full"] = "Q('{}') ~ 1 + C(time_step){}".format(neuron, trial_term)
        formulas["full_subtime"] = "Q('{}') ~ 1               {}".format(
            neuron, trial_term
        )
    elif game_state == "Encoding":
        formulas[
            "full"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region) + C(head_direc) + is_moving + base_in_view + gold_in_view{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subtime"
        ] = "Q('{}') ~ 1                + C(maze_region) + C(head_direc) + is_moving + base_in_view + gold_in_view{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subplace"
        ] = "Q('{}') ~ 1 + C(time_step)                  + C(head_direc) + is_moving + base_in_view + gold_in_view{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subhd"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region)                 + is_moving + base_in_view + gold_in_view{}".format(
            neuron, trial_term
        )
        formulas[
            "full_submvmt"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region) + C(head_direc)             + base_in_view + gold_in_view{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subbiv"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region) + C(head_direc) + is_moving                + gold_in_view{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subgiv"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region) + C(head_direc) + is_moving + base_in_view               {}".format(
            neuron, trial_term
        )
    elif game_state == "Retrieval":
        formulas[
            "full"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region) + C(head_direc) + is_moving + base_in_view + dig_performed{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subtime"
        ] = "Q('{}') ~ 1                + C(maze_region) + C(head_direc) + is_moving + base_in_view + dig_performed{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subplace"
        ] = "Q('{}') ~ 1 + C(time_step)                  + C(head_direc) + is_moving + base_in_view + dig_performed{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subhd"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region)                 + is_moving + base_in_view + dig_performed{}".format(
            neuron, trial_term
        )
        formulas[
            "full_submvmt"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region) + C(head_direc)             + base_in_view + dig_performed{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subbiv"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region) + C(head_direc) + is_moving                + dig_performed{}".format(
            neuron, trial_term
        )
        formulas[
            "full_subdig"
        ] = "Q('{}') ~ 1 + C(time_step) + C(maze_region) + C(head_direc) + is_moving + base_in_view                {}".format(
            neuron, trial_term
        )

    # Fit the models.
    if mod_names is None:
        mod_names = list(formulas.keys())
    else:
        mod_names = [k for k in mod_names if k in formulas.keys()]

    res = od([])
    for mod_name in mod_names:
        res[mod_name] = ols(formulas[mod_name], data=df).fit()

    # Fit the same models to shuffled spike counts.
    res_null = od({k: [] for k in mod_names})
    for iPerm in range(n_perm):
        # Permute the firing rate vector.
        if perm_method == "circshift":
            df[neuron] = np.concatenate(
                df.groupby("trial")[neuron]
                .apply(lambda x: list(_shift_spikes(x)))
                .tolist()
            )
        elif perm_method == "shuffle":
            df[neuron] = np.concatenate(
                df.groupby("trial")[neuron]
                .apply(lambda x: list(_shuffle_spikes(x)))
                .tolist()
            )
        else:
            raise RuntimeError(
                "Permutation method '{}' not recognized".format(perm_method)
            )

        # Fit the model.
        for mod_name in mod_names:
            res_null[mod_name].append(ols(formulas[mod_name], data=df).fit())

    model_fits = od(
        {
            "subj_sess": event_spikes.subj_sess,
            "neuron": neuron,
            "obs": res,
            "null": res_null,
        }
    )
    return model_fits


def glm_fit(
    neuron,
    event_spikes,
    game_state,
    n_perm=0,
    regress_trial=True,
    perm_method="circshift",
    optimizer="lbfgs",
):
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
    regress_trial : bool
        If True, GLM includes a dummy-coded predictor for each trial.
        If False, no trial predictors are added.
    perm_method : str
        Determines how the firing rate vector will be scrambled to
        obtain the null distribution. 'circshift' or 'shuffle'
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
    cm = event_spikes.column_map.copy()

    # For categorical variables with >2 levels, drop the first level.
    cm["time"] = cm["time"][1:]
    cm["place"] = cm["place"][1:]
    cm["hd"] = cm["hd"][1:]
    cm["trial"] = cm["trial"][1:]

    # For delays, sum spikes across time bins within each time step, within each trial.
    if game_state in ["Delay1", "Delay2"]:
        df = pd.concat(
            (
                df.groupby(["trial", "time_step"]).agg({neuron: np.sum}),
                df.groupby(["trial", "time_step"]).agg(
                    {icpt: np.mean for icpt in event_spikes.column_map["icpt"]}
                ),
                df.groupby(["trial", "time_step"]).agg(
                    {time: np.mean for time in event_spikes.column_map["time"]}
                ),
                df.groupby(["trial", "time_step"]).agg(
                    {trial: np.mean for trial in event_spikes.column_map["trial"]}
                ),
            ),
            axis=1,
        )

    # Independent variables are the dummy coded time steps and the intercept constant.
    trial_cols = cm["trial"] if regress_trial else []
    predictors = od([])
    if game_state in ["Delay1", "Delay2"]:
        predictors["time"] = cm["icpt"] + cm["time"] + trial_cols
        predictors["icpt"] = cm["icpt"] + trial_cols

    elif game_state in ["Encoding", "Retrieval"]:
        predictors["time_place"] = cm["icpt"] + cm["time"] + cm["place"] + trial_cols
        predictors["place"] = cm["icpt"] + cm["place"] + trial_cols
        predictors["time"] = cm["icpt"] + cm["time"] + trial_cols

        dig_col = cm["dig_performed"] if (game_state == "Retrieval") else []
        predictors["full"] = (
            cm["icpt"]
            + cm["time"]
            + cm["place"]
            + cm["hd"]
            + cm["is_moving"]
            + cm["base_in_view"]
            + cm["gold_in_view"]
            + dig_col
            + trial_cols
        )
        predictors["full_subtime"] = (
            cm["icpt"]
            + cm["place"]
            + cm["hd"]
            + cm["is_moving"]
            + cm["base_in_view"]
            + cm["gold_in_view"]
            + dig_col
            + trial_cols
        )
        predictors["full_subplace"] = (
            cm["icpt"]
            + cm["time"]
            + cm["hd"]
            + cm["is_moving"]
            + cm["base_in_view"]
            + cm["gold_in_view"]
            + dig_col
            + trial_cols
        )
        predictors["full_subhd"] = (
            cm["icpt"]
            + cm["time"]
            + cm["place"]
            + cm["is_moving"]
            + cm["base_in_view"]
            + cm["gold_in_view"]
            + dig_col
            + trial_cols
        )
        predictors["full_subbiv"] = (
            cm["icpt"]
            + cm["time"]
            + cm["place"]
            + cm["hd"]
            + cm["is_moving"]
            + cm["gold_in_view"]
            + dig_col
            + trial_cols
        )
        predictors["full_subgiv"] = (
            cm["icpt"]
            + cm["time"]
            + cm["place"]
            + cm["hd"]
            + cm["is_moving"]
            + cm["base_in_view"]
            + dig_col
            + trial_cols
        )

    # Fit Poisson regressions.
    res = od([])
    y = df[neuron]
    for mod_name in predictors.keys():
        X = df[predictors[mod_name]].fillna(0)
        res[mod_name] = sm.GLM(y, X, family=sm.families.Poisson()).fit(method=optimizer)

    # Fit the same models to shuffled spike counts.
    res_null = od({k: [] for k in predictors.keys()})
    for iPerm in range(n_perm):
        # Permute the firing rate vector.
        if perm_method == "circshift":
            y = np.concatenate(
                df.groupby("trial")[neuron]
                .apply(lambda x: list(_shift_spikes(x)))
                .tolist()
            )
        elif perm_method == "shuffle":
            y = np.concatenate(
                df.groupby("trial")[neuron]
                .apply(lambda x: list(_shuffle_spikes(x)))
                .tolist()
            )
        else:
            raise RuntimeError(
                "Permutation method '{}' not recognized".format(perm_method)
            )

        # Fit the model.
        for mod_name in predictors.keys():
            X = df[predictors[mod_name]].fillna(0)
            res_null[mod_name].append(
                sm.GLM(y, X, family=sm.families.Poisson()).fit(method=optimizer)
            )

    model_fits = od(
        {
            "subj_sess": event_spikes.subj_sess,
            "neuron": neuron,
            "obs": res,
            "null": res_null,
        }
    )
    return model_fits


def get_model_pairs(model_fits):
    """Compare fits between predefined model pairs.

    Parameters
    ----------
    subj_sess : str
        e.g. 'U518_ses0'
    neuron : str
        e.g. '17-1' is channel 17, unit 1.
    model_fits : dict
        This is the output of ols_fit() and glm_fit().

    Returns
    -------
    model_pairs : DataFrame
    """
    # Find the names of all independent variables tested across models.
    param_cols = []
    for game_state in model_fits.keys():
        for mod_name in model_fits[game_state]["obs"].keys():
            param_cols += [
                param
                for param in model_fits[game_state]["obs"][mod_name].params.index
                if (param not in param_cols)
            ]
    replace_vals = {
        "time_step": "time",
        "maze_region": "place",
        "head_direc": "hd",
        "C(": "",
        ")[T.": "-",
        "]": "",
    }
    param_col_names = str_replace(param_cols, replace_vals)

    # Iterate over each constrast in the model comparison dataframe
    # and add on the GLM fit variables.
    model_comp = _get_model_comp()
    model_output_cols = [
        "llf_full",
        "lr",
        "z_lr",
        "df",
        "chi_pval",
        "emp_pval",
        "null_hist",
    ] + param_col_names
    model_output = []
    for iRow, row in model_comp.iterrows():
        game_state = row["gameState"]
        if game_state not in model_fits.keys():
            continue

        full_mod = row["full"]
        reduced_mod = row["reduced"]

        # Get full model parameter weights and Z-scored weights relative
        # to the null distribution.
        obs_params = model_fits[game_state]["obs"][full_mod].params
        params = od([])
        for k in param_cols:
            params[k] = obs_params.get(k, np.nan)

        # Peform a likelihood ratio test on the full vs. reduced models.
        llf_full = model_fits[game_state]["obs"][full_mod].llf
        lr, df, chi_pval = lr_test(
            model_fits[game_state]["obs"][reduced_mod],
            model_fits[game_state]["obs"][full_mod],
        )
        lr = np.max((lr, 0))

        # Get likelihood ratios from the null distribution,
        # and use these to obtain an empirical p-value.
        n_perm = len(model_fits[game_state]["null"][reduced_mod])
        null_lrs = np.array(
            [
                get_lr(
                    model_fits[game_state]["null"][reduced_mod][iPerm],
                    model_fits[game_state]["null"][full_mod][iPerm],
                )
                for iPerm in range(n_perm)
            ]
        )
        null_lrs[np.isnan(null_lrs)] = 0
        null_lrs[null_lrs < 0] = 0
        null_mean = np.mean(null_lrs)
        null_std = np.std(null_lrs)
        null_hist = np.histogram(null_lrs, bins=31)
        z_lr = (lr - null_mean) / null_std
        pval_ind = np.sum(null_lrs >= lr)
        emp_pval = (pval_ind + 1) / (n_perm + 1)

        # Add results to the output dataframe.
        model_output.append(
            [llf_full, lr, z_lr, df, chi_pval, emp_pval, list(null_hist)]
            + list(params.values())
        )

    model_output = pd.DataFrame(model_output, columns=model_output_cols)
    model_pairs = pd.concat((model_comp, model_output), axis=1)
    model_pairs = model_pairs[np.isfinite(model_pairs["lr"])]

    # Add subj_sess and neuron identity columns.
    model_pairs.insert(0, "subj_sess", list(model_fits.values())[0]["subj_sess"])
    model_pairs.insert(1, "neuron", list(model_fits.values())[0]["neuron"])

    return model_pairs


def _get_model_comp():
    """Define pairwise comparisons between full and reduced models."""
    cols = ["gameState", "testvar", "reduced", "full"]
    model_comp = [
        ["Delay1", "time", "full_subtime", "full"],
        ["Delay2", "time", "full_subtime", "full"],
        ["Encoding", "time", "full_subtime", "full"],
        ["Encoding", "place", "full_subplace", "full"],
        ["Encoding", "head_direc", "full_subhd", "full"],
        ["Encoding", "is_moving", "full_submvmt", "full"],
        ["Encoding", "base_in_view", "full_subbiv", "full"],
        ["Encoding", "gold_in_view", "full_subgiv", "full"],
        ["Retrieval", "time", "full_subtime", "full"],
        ["Retrieval", "place", "full_subplace", "full"],
        ["Retrieval", "head_direc", "full_subhd", "full"],
        ["Retrieval", "is_moving", "full_submvmt", "full"],
        ["Retrieval", "base_in_view", "full_subbiv", "full"],
        ["Retrieval", "dig_performed", "full_subdig", "full"],
    ]
    model_comp = pd.DataFrame(model_comp, columns=cols)
    return model_comp


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
    lr = get_lr(res_reduced, res_full)
    df = len(res_full.params) - len(res_reduced.params)
    pval = stats.chi2.sf(lr, df)

    return lr, df, pval


def get_lr(res_reduced, res_full):
    """Return the likelihood ratio.

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
    """
    lr = -2 * (res_reduced.llf - res_full.llf)
    return lr


def spikes_per_timebin(events_behav, spike_times):
    """Count how many spikes are in each time bin.

    Returns a pandas Series.
    """

    def count_spikes(row, spike_times):
        start = row["start_time"]
        stop = row["stop_time"]
        return np.count_nonzero((spike_times >= start) & (spike_times < stop))

    return events_behav.apply(lambda x: count_spikes(x, spike_times), axis=1)


def _circshift_shuffle(df, output_col, grpby=["trial", "gameState"]):
    """Circ-shift col values within each group, and shuffle grouped rows.

    Both operations are randomized.

    Parameters
    ----------
    df : dataframe
        Contains the grouping columns and output column.
    output_col : str
        The column whose values get shifted around.
    grpby : str or list[str]
        The grouping column[s].

    Returns
    -------
    null_vals : np.array
        Vector of shifted output_col values.
    """
    null_vals = np.concatenate(
        np.random.permutation(
            df.groupby(grpby, sort=False)[output_col]
            .apply(lambda x: list(_shift_spikes(x)))
            .tolist()
        )
    )

    return null_vals


def _shift_spikes(spike_vec):
    """Circularly shift spike vector."""
    roll_by = np.random.randint(0, len(spike_vec))
    return np.roll(spike_vec, roll_by)


def _shuffle_spikes(spike_vec):
    """Shuffle spike vector."""
    idx = np.arange(len(spike_vec))
    np.random.shuffle(idx)
    return np.array(list(spike_vec))[idx]


def calc_mean_fr_by_time(
    fr_by_time_bin_f,
    proj_dir="/home1/dscho/projects/time_cells",
    overwrite=False,
    save_output=True,
    verbose=True,
):
    """Calculate mean FR across trial phases of a given type."""
    neuron = "-".join(op.basename(fr_by_time_bin_f).split("-")[:3])

    # Load the output file if it exists.
    output_f = op.join(
        proj_dir, "analysis", "fr_by_time_bin", "{}-mean_fr_by_time.pkl".format(neuron)
    )
    if op.exists(output_f) and not overwrite:
        return dio.open_pickle(output_f)

    subj_sess, chan, unit = neuron.split("-")
    chan = chan[3:]
    subj, sess = subj_sess.split("_")
    event_times = dio.open_pickle(fr_by_time_bin_f)["event_times"]
    n_perm = event_times["fr_null"].iloc[0].shape[0]
    # game_states = [['Prepare1'], ['Delay1'], ['Encoding'], ['Prepare2'], ['Delay2'], ['Retrieval'],
    #                ['Prepare1', 'Prepare2'], ['Delay1', 'Delay2'], ['Encoding', 'Retrieval']]
    game_states = [["Delay1"], ["Delay2"]]
    cols = [
        "subj_sess",
        "subj",
        "sess",
        "chan",
        "unit",
        "gameState",
        "fr",
        "z_fr",
        "z_fr_max",
        "z_fr_max_ind",
        "tis",
        "z_tis",
        "pval",
    ]
    output = []
    for game_state in game_states:
        if not np.all(np.isin(game_state, event_times["gameState"].unique())):
            continue
        obs = np.mean(
            event_times.query("(gameState=={})".format(game_state))["fr"].tolist(),
            axis=0,
        )
        null_mean = np.mean(
            np.mean(
                event_times.query("(gameState=={})".format(game_state))[
                    "fr_null"
                ].tolist(),
                axis=0,
            ),
            axis=0,
        )
        null_std = np.std(
            np.mean(
                event_times.query("(gameState=={})".format(game_state))[
                    "fr_null"
                ].tolist(),
                axis=0,
            ),
            axis=0,
        )
        obs_z = (obs - null_mean) / null_std

        tis = info_rate(obs)  # temporal information score
        null_tis = np.array(
            [
                info_rate(
                    np.mean(
                        event_times.query("(gameState=={})".format(game_state))[
                            "fr_null"
                        ].tolist(),
                        axis=0,
                    )[iPerm, :]
                )
                for iPerm in range(n_perm)
            ]
        )
        tis_z = (tis - np.mean(null_tis)) / np.std(null_tis)
        pval = (np.sum(null_tis >= tis) + 1) / (n_perm + 1)

        output.append(
            [
                subj_sess,
                subj,
                sess,
                chan,
                unit,
                "_".join(game_state),
                obs,
                obs_z,
                obs_z.max(),
                obs_z.argmax(),
                tis,
                tis_z,
                pval,
            ]
        )

    output = pd.DataFrame(output, columns=cols)

    # Save output.
    if save_output:
        dio.save_pickle(output, output_f, verbose)

    return output


def classify_time_bins(
    spikes,
    event_times,
    game_states,
    n_time_bins,
    C=0.001,
    k=6,
    n_perms=1000,
    save_as=None,
    verbose=False,
):
    """Predict time bins from firing rates using multiclass linear SVM."""
    do_shift = [False] + ([True] * n_perms)

    output = od([("obs", []), ("null", [])])
    for iPerm in range(n_perms + 1):
        # Get firing rate data.
        fr = np.array(
            [
                calc_fr_by_time_bin(
                    row["fr_train"],
                    event_times.query("(subj_sess=='{}')".format(row["subj_sess"])),
                    game_states,
                    n_time_bins,
                    random_shift=do_shift[iPerm],
                )
                for idx, row in spikes.iterrows()
            ]
        )
        shp = fr.shape
        n_neurons, n_trials, n_time_bins = shp

        # Z-score firing rates for each cell across time bins from all trials.
        zfr = stats.zscore(fr.reshape((n_neurons, np.prod(shp[1:]))), axis=1).reshape(
            shp
        )

        # Setup the the indepedent variable matrix and the dependent variable vector.
        X = np.swapaxes(
            zfr.reshape([n_neurons, np.prod((n_trials, n_time_bins))]), 0, 1
        )
        n_samp, n_feat = X.shape
        y = (
            np.tile(np.arange(n_time_bins), n_trials)
            .reshape((n_trials, n_time_bins))
            .flatten()
        )

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
            output["obs"] = np.array(list(zip(y_true, y_pred)))
        else:
            output["null"].append(np.array(list(zip(y_true, y_pred))))

    # Save output.
    if save_as:
        dio.save_pickle(output, save_as, verbose)

    return output


def fr_by_time_vs_null(
    fr_train,
    event_times,
    game_states,
    n_time_bins,
    n_perms=1000,
    save_as=None,
    verbose=False,
):
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
    fr_vec = np.nanmean(
        calc_fr_by_time_bin(
            fr_train, event_times, game_states, n_time_bins, random_shift=False
        ),
        axis=0,
    )
    temporal_info = info_rate(fr_vec)

    # Generate a surrogate distribution by circ-shifting the firing rates
    # for each event by a random interval up to the duration of the
    # event window.
    null_fr_vecs = []
    null_temporal_info = []
    for iPerm in range(n_perms):
        fr_vec_null = np.nanmean(
            calc_fr_by_time_bin(
                fr_train, event_times, game_states, n_time_bins, random_shift=True
            ),
            axis=0,
        )
        null_fr_vecs.append(fr_vec_null)
        null_temporal_info.append(info_rate(fr_vec_null))
    null_fr_vecs = np.array(null_fr_vecs)  # iPerm x iTime
    null_temporal_info = np.array(null_temporal_info)

    # Compare observed values to the null distribution.
    fr_vec_z = (fr_vec - np.nanmean(null_fr_vecs, axis=0)) / np.nanstd(
        null_fr_vecs, axis=0
    )
    temporal_info_z = (temporal_info - np.nanmean(null_temporal_info)) / np.nanstd(
        null_temporal_info
    )
    pval_ind = np.nansum(null_temporal_info >= temporal_info)
    pval = (pval_ind + 1) / (n_perms + 1)

    output = od(
        [
            ("fr_vec", fr_vec),
            ("fr_vec_z", fr_vec_z),
            ("temporal_info", temporal_info),
            ("temporal_info_z", temporal_info_z),
            ("temporal_info_pval", pval),
        ]
    )

    # Save output.
    if save_as:
        dio.save_pickle(output, save_as, verbose)

    return output


def calc_fr_by_time_bin(
    fr_train, event_times, game_states, n_time_bins, random_shift=False
):
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

    def event_mean_frs(time_window, fr_train, n_time_bins, random_shift=False):
        start, stop = time_window
        fr_train_ = fr_train[start:stop]
        if random_shift:
            roll_by = int(np.random.rand() * (stop - start))
            fr_train_ = np.roll(fr_train_, roll_by)
        return [np.nanmean(x) for x in np.array_split(fr_train_, n_time_bins)]

    # Get time windows for the game states of interest.
    time_wins = event_times.query("(gameState=={})".format(game_states))["time"].apply(
        lambda x: start_stop(x)
    )

    # Get the mean firing rate in each time bin, for each event.
    fr_by_time_bin = np.array(
        time_wins.apply(
            lambda x: event_mean_frs(x, fr_train, n_time_bins, random_shift)
        ).tolist()
    )
    return fr_by_time_bin


def info_rate(fr_given_x, prob_x=None, scale_minmax=False):
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
        warnings.simplefilter("ignore")
        bits_per_spike = np.nansum(
            prob_x * (fr_given_x / mean_fr) * np.log2(fr_given_x / mean_fr)
        )
    return bits_per_spike


def get_mean_frs(spike_mat):
    """Return mean firing rate across trials, for each time bin.

    Parameters
    ----------
    spike_mat : DataFrame or array
        A trial x time_bin matrix of spike counts.

    Returns
    -------
    mean_frs : vector
        Each value is the mean number of spikes for a given time bin.
        For a 500ms time bin, you can get firing rate in Hz by
        multiplying the output of this function by 2.
    """
    mean_frs = np.mean(spike_mat, axis=0)
    return mean_frs


def get_sem_frs(spike_mat):
    """Return mean firing rate across trials, for each time bin.

    Parameters
    ----------
    spike_mat : DataFrame or array
        A trial x time_bin matrix of spike counts.

    Returns
    -------
    sem_frs : vector
        Each value is the standard error of the mean number of
        spikes for a given time bin. For a 500ms time bin, you
        can get SEM firing rate in Hz by multiplying the output
        of this function by 2.
    """
    sem_frs = stats.sem(spike_mat, axis=0)
    return sem_frs


def get_sparsity(spike_mat):
    """Return sparsity for a trial x time_bin spike matrix.

    Ranges from 0 to 1, with smaller values reflecting finer tuning.

    See Jung, Wiener & McNaughton, J Neurosci 1994
    """
    mean_frs = get_mean_frs(spike_mat)
    numer = np.square(np.sum(mean_frs) / mean_frs.size)
    denom = np.sum(np.square(mean_frs) / mean_frs.size)
    return numer / denom


def bootstrap_time_fields(
    subj_sess_unit,
    game_states,
    mult=2,
    smooth=0,
    n_perm=1000,
    thresh=1.96,
    max_skips=1,
    proj_dir="/home1/dscho/projects/time_cells",
    save_output=True,
    overwrite=False,
    verbose=True,
):
    """Return a dataframe of time fields."""
    # Load the saved file if it exists.
    filename = op.join(
        proj_dir,
        "analysis",
        "time_fields",
        "{}-{}-smooth{}-{}perm-zthresh{}-max_skips{}-time_fields.pkl".format(
            subj_sess_unit, "_".join(game_states), smooth, n_perm, thresh, max_skips
        ),
    )
    if op.exists(filename) and not overwrite:
        time_fields = dio.open_pickle(filename)
        return time_fields

    # Load the data.
    subj_sess = subj_sess_unit.split("-")[0]
    neuron = "-".join(subj_sess_unit.split("-")[1:])
    event_spikes = load_event_spikes(subj_sess, proj_dir=proj_dir, verbose=False)
    spike_mat = {
        game_state: event_spikes.get_spike_mat(neuron, game_state).values * float(mult)
        for game_state in game_states
    }  # trial x time_bin

    # Combine the game states.
    game_states.append("".join(game_states))
    spike_mat[game_states[-1]] = np.concatenate(list(spike_mat.values()))

    # Smooth spike counts over time, separately for each trial and game_state.
    if smooth > 0:
        spike_mat_obs = {
            game_state: np.array(
                [
                    gaussian_filter1d(spike_mat[game_state][iTrial, :], smooth)
                    for iTrial in range(spike_mat[game_state].shape[0])
                ]
            )
            for game_state in game_states
        }
    else:
        spike_mat_obs = spike_mat.copy()

    # Get the mean firing rate in each time bin, across trials.
    mean_frs = {
        game_state: np.mean(spike_mat_obs[game_state], axis=0)
        for game_state in game_states
    }

    # Get the null distribution of mean firing rates
    # by circ-shifting spikes within each trial.
    mean_frs_null = {game_state: [] for game_state in game_states}
    for iPerm in range(n_perm):
        if smooth > 0:
            spike_mat_null = {
                game_state: np.array(
                    [
                        gaussian_filter1d(
                            _shift_spikes(spike_mat[game_state][iTrial, :]), smooth
                        )
                        for iTrial in range(spike_mat[game_state].shape[0])
                    ]
                )
                for game_state in game_states
            }
        else:
            spike_mat_null = {
                game_state: np.array(
                    [
                        _shift_spikes(spike_mat[game_state][iTrial, :])
                        for iTrial in range(spike_mat[game_state].shape[0])
                    ]
                )
                for game_state in game_states
            }
        for game_state in game_states:
            mean_frs_null[game_state].append(
                np.mean(spike_mat_null[game_state], axis=0)
            )  # perm x time

    # Z-score firing rates.
    _null_mean = np.mean(mean_frs_null[game_state], axis=0)
    _null_std = np.std(mean_frs_null[game_state], axis=0)
    z_frs = {
        game_state: (mean_frs[game_state] - _null_mean) / _null_std
        for game_state in game_states
    }

    # Find max and min Z-scores for each permutation of the null distribution.
    z_frs_null = {
        game_state: (mean_frs_null[game_state] - _null_mean) / _null_std
        for game_state in game_states
    }  # perm x time
    z_frs_null_max = {
        game_state: np.max(z_frs_null[game_state], axis=1) for game_state in game_states
    }
    z_frs_null_min = {
        game_state: np.min(z_frs_null[game_state], axis=1) for game_state in game_states
    }

    # Find time fields.
    time_fields = []
    for game_state in game_states:
        _z_frs = z_frs[game_state]
        assert len(_z_frs) == len(np.unique(_z_frs))

        # Identify positive time fields.
        keep = np.where(np.isfinite(_z_frs))[0]
        while len(keep) > 0:
            peak_z = np.max(_z_frs[keep])
            field_peak = np.where(_z_frs == peak_z)[0][0]
            field_idx = [field_peak]
            if peak_z > thresh:
                # Scan backwards.
                _idx = field_peak - 1
                skips = 0
                while (_idx in keep) & (skips <= max_skips):
                    if _z_frs[_idx] > thresh:
                        field_idx.append(_idx)
                    else:
                        skips += 1
                    _idx -= 1

                # Scan forwards.
                _idx = field_peak + 1
                skips = 0
                while (_idx in keep) & (skips <= max_skips):
                    if _z_frs[_idx] > thresh:
                        field_idx.append(_idx)
                    else:
                        skips += 1
                    _idx += 1

                # Append the field.
                field_idx = list(np.sort(field_idx))
                mean_z = np.mean(_z_frs[field_idx])
                time_fields.append(
                    [
                        subj_sess_unit,
                        game_state,
                        "pos",
                        field_peak,
                        peak_z,
                        mean_z,
                        len(field_idx),
                        field_idx,
                    ]
                )

                # Remove the field from consideration.
                keep = [x for x in keep if x not in field_idx]
            else:
                break

        # Identify negative time fields.
        keep = np.where(np.isfinite(_z_frs))[0]
        while len(keep) > 0:
            peak_z = np.min(_z_frs[keep])
            field_peak = np.where(_z_frs == peak_z)[0][0]
            field_idx = [field_peak]
            if peak_z < -thresh:
                # Scan backwards.
                _idx = field_peak - 1
                skips = 0
                while (_idx in keep) & (skips <= max_skips):
                    if _z_frs[_idx] < -thresh:
                        field_idx.append(_idx)
                    else:
                        skips += 1
                    _idx -= 1

                # Scan forwards.
                _idx = field_peak + 1
                skips = 0
                while (_idx in keep) & (skips <= max_skips):
                    if _z_frs[_idx] < -thresh:
                        field_idx.append(_idx)
                    else:
                        skips += 1
                    _idx += 1

                # Append the field.
                field_idx = list(np.sort(field_idx))
                mean_z = np.mean(_z_frs[field_idx])
                time_fields.append(
                    [
                        subj_sess_unit,
                        game_state,
                        "neg",
                        field_peak,
                        peak_z,
                        mean_z,
                        len(field_idx),
                        field_idx,
                    ]
                )

                # Remove the field from consideration.
                keep = [x for x in keep if x not in field_idx]
            else:
                break

    cols = [
        "subj_sess_unit",
        "gameState",
        "field_type",
        "field_peak",
        "peak_z",
        "mean_z",
        "field_size",
        "field_idx",
    ]
    time_fields = pd.DataFrame(time_fields, columns=cols)

    output = od(
        [
            ("time_fields", time_fields),
            ("z_frs", z_frs),
            ("z_frs_null_max", z_frs_null_max),
            ("z_frs_null_min", z_frs_null_min),
        ]
    )

    # Save the output data.
    if save_output:
        dio.save_pickle(output, filename, verbose)

    return output


def bootstrap_time_fields2(spike_mat, n_perm=10000):
    """Identify time fields using a bootstrap estimation method.

    n_trial values are drawn from spike_mat with replacement, and
    the mean across them is calculated. This is repeated 500 times
    to obtain a sample distribution. Actual mean firing rates across
    trials are then Z-scored against this distribution, and time fields
    are defined as any time bin with empirical p-values < 0.05,
    Bonferroni-Holm corrected.

    Parameters
    ----------
    spike_mat : DataFrame
        A trial x time_bin matrix of spike counts.

    Returns
    -------
    z_mean_frs : Series
        Z-scored firing rates over time.
    time_field : array[int]
        All time bins with empirical p-values < 0.05,
        Bonferroni-Holm corrected.
    time_field_size : int
        The number of time bins with significantly elevated firing.
    """
    n_trial = spike_mat.shape[0]
    mean_frs = np.mean(spike_mat, axis=0)
    spike_vals = spike_mat.values.ravel()
    sample_mean_frs = np.array(
        [np.mean(random.choices(spike_vals, k=n_trial)) for _ in range(n_perm)]
    )
    sample_mean = np.mean(sample_mean_frs)
    sample_std = np.std(sample_mean_frs)
    z_mean_frs = (mean_frs - sample_mean) / sample_std
    alpha = 0.05
    pvals = (1 + np.array([np.sum(sample_mean_frs >= fr) for fr in mean_frs])) / (
        1 + n_perm
    )
    sig = sm.stats.multipletests(pvals, alpha, method="holm")[0]
    time_field = np.where(sig)[0]
    time_field_size = len(time_field)
    return z_mean_frs, time_field, time_field_size
