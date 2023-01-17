"""
events_proc.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Functions to consolidate Goldmine events into time bins for analysis.

Last Edited
----------- 
3/26/20
"""
import sys
import os.path as op
from collections import OrderedDict as od
# import mkl
# mkl.set_num_threads(1)
import numpy as np
import pandas as pd
sys.path.append('/home1/dscho/code/general')
import data_io as dio
sys.path.append('/home1/dscho/code/projects')
from time_cells import events_preproc


class Events(object):
    """Analyze the Goldmine behavioral log."""
    
    def __init__(self,
                 subj_sess,
                 remove_bad_trials=True,
                 time_bin_dur=500,
                 n_head_direc=8,
                 run_all=False,
                 proj_dir='/home1/dscho/projects/time_cells',
                 filename=None,
                 verbose=False):
        """Initialize events for a testing session.

        Events must have already been aligned to EEG and preprocessed
        (see events_preproc.py functions).
        
        Parameters
        ----------
        subj_sess : str
            E.g. U518_ses0
        remove_bad_trials : bool
            If True, trials with bad_trial notes or that are manually
            listed in get_excluded_trials() are removed from all 
            class dataframes *except for* self.events.
        time_bin_dur : int
            Exact ms duration of each time bin within each trial phase.
        run_all : bool
            If True, functions are called to generate the 
            self.events_behav dataframe and the dataframes it requires.
            Otherwise, only events and event_times dataframes are made,
            and the user can run the rest by calling self.log*
            functions.
        proj_dir : str
            Top-level directory within which preprocessed events and
            maze dataframes are to be found.
        filename : str
            Full file path to the location where this class instance
            will be saved. Does not have to be within proj_dir, but will
            be by default if no filename is given. See save_events()
            and load_events().
        """
        self.subj_sess = subj_sess
        self.remove_bad_trials = remove_bad_trials
        self.time_bin_dur = time_bin_dur
        self.n_direc = n_head_direc
        self.proj_dir = proj_dir
        if filename is None:
            self.filename = op.join(self.proj_dir, 'analysis', 'events', 
                                    '{}-Events.pkl'.format(self.subj_sess))
        else:
            self.filename = filename
        
        # Load the pre-processed events dataframe.
        self.events = events_preproc.format_events(self.subj_sess,
                                                   overwrite=False,
                                                   save_output=False,
                                                   proj_dir=self.proj_dir,
                                                   verbose=verbose)
        
        # Load the main experiment maze.
        self.maze_name = self.events.iloc[0]['scene']
        self.maze = Maze(self.maze_name, proj_dir=self.proj_dir)
        
        # Get a list of all trials completed, and lists of trials with
        # vs. without comments in the self.events['bad_trials'] column.
        self.trials = list(self.events['trial'].unique())
        self.bad_trials = list(np.unique(self.events.query("(bad_trials!='')")['trial'].unique().tolist() + 
                               get_excluded_trials(self.subj_sess)))
        self.keep_trials = [trial for trial in self.trials if trial not in self.bad_trials]
        self.bad_trial_phases = (self.events.query("(bad_trials!='')")[['trial', 'gameState']]
                                            .drop_duplicates()
                                            .groupby('trial')['gameState']
                                            .apply(lambda x: list(np.unique(x)))
                                            .reset_index().values.tolist()) # e.g. [[1, ['Delay1']], [4, ['Delay2', 'Retrieval']]]
        
        # Store a mapping between trial number and whether or not a time 
        # penalty was applied.
        self.time_penalty = self.events.query("(key=='trialComplete')").set_index('trial')['time_penalty']
        
        # Create the event_times dataframe.
        self.log_event_times()
        
        # Create the events_behav dataframe and its dependencies.
        if run_all:
            self.log_events_behav()

    def __str__(self):
        """Print basic session info."""
        _s = ('-' * 4 * len(self.subj_sess)) + '\n'
        _s += self.subj_sess + '\n'
        _s += '{} trials completed, {} good\n\n'.format(len(self.trials), len(self.keep_trials))
        
        report_gold = np.all((hasattr(self, 'gold_spawned'), 
                              hasattr(self, 'gold_events'), 
                              hasattr(self, 'dig_events')))
        if report_gold:
            gold_spawned = len(self.gold_spawned)
            gold_viewed = len(np.unique(self.gold_events.query("(in_view==True)")['gold_id']))
            dig_attempts = len(self.dig_events)
            gold_dug = len(self.dig_events.query("(dig_success==True)"))
            _s += '{} gold spawned\n'.format(gold_spawned)
            _s += '{} gold viewed ({:.0%})\n'.format(gold_viewed, gold_viewed/gold_spawned)
            _s += '{} gold dug ({:.0%})\n'.format(gold_dug, gold_dug/gold_spawned)
            _s += '{} dig attempts ({:.0%} accuracy)\n\n'.format(dig_attempts, gold_dug/dig_attempts)

        _s += 'dataframe sizes:\n'
        for _k, _v in self.__dict__.items():
            if type(_v) == pd.core.frame.DataFrame:
                _s += '\t{}: {}\n'.format(_k, _v.shape)

        return _s

    def log_event_times(self):
        """Break up event windows into evenly-sized time bins.

        Time bins are 500 ms each by default and extend up to the
        duration specified for each game state by get_game_state_durs().

        Parameters
        ----------
        game_states : list[str]
            Game states to include in the output DataFrame.
        """
        # Get the event time windows for each game state.
        game_states = ['Delay1', 'Encoding', 'ReturnToBase1', 
                       'Delay2', 'Retrieval', 'ReturnToBase2']
        dfs = []
        for iState, game_state in enumerate(game_states):
            dfs.append(self._game_state_intervals(self.events, game_state=game_state, cols=['time']))
            dfs[-1].insert(1, 'trial_phase', iState + 1)
            dfs[-1].insert(2, 'gameState', game_state)
        event_times = pd.concat(dfs, axis=0)
        event_times.insert(0, 'subj_sess', self.subj_sess)
        event_times['duration'] = event_times['time'].apply(lambda x: x[-1]-x[0])

        # Divide each trial phase into equal-length time bins that are exactly 
        # time_bin_dur long and go up to the trial phase duration specified by 
        # get_game_state_durs().
        game_state_durs = get_game_state_durs()
        time_bins = []
        for idx, row in event_times.iterrows():
            if row['gameState'] in game_state_durs:
                assert row['duration'] >= game_state_durs[row['gameState']]
                time_bins.append(np.arange(row['time'][0], 
                                           row['time'][0] + game_state_durs[row['gameState']] + 1, 
                                           step=self.time_bin_dur))
            else:
                time_bins.append(np.arange(row['time'][0], row['time'][1], step=self.time_bin_dur))
        event_times['time_bins'] = time_bins
        event_times['time_bin_dur'] = event_times['time_bins'].apply(lambda x: 0 if len(x)==0 else x[-1] - x[0])
        event_times['n_time_bins'] = event_times['time_bins'].apply(lambda x: np.max((len(x) - 1, 0)))

        # Assign game_state categories a specific order.
        game_state_cat = pd.CategoricalDtype(game_states, ordered=True)
        event_times['gameState'] = event_times['gameState'].astype(game_state_cat)

        # Remove excluded trials.
        _trials = self.keep_trials if self.remove_bad_trials else self.trials
        event_times = (event_times.query("(trial=={})".format(_trials))
                                  .sort_values(['trial', 'trial_phase'])
                                  .reset_index(drop=True))
        
        self.event_times = event_times

    def log_events_behav(self):
        """Log behavioral events for every time bin in the session.
        
        This dataframe can be used to analyze behavioral/neural 
        correlations for events that occur at time resolutions defined
        by the duration of self.time_bin_dur
        """
        # Run dependencies.
        if not hasattr(self, 'positions'):
            self.log_positions()
        if not hasattr(self, 'base_events'):
            self.log_base_view_events()
        if not hasattr(self, 'gold_events'):
            self.log_gold_view_events()
        if not hasattr(self, 'gold_spawned'):
            self.log_gold_spawned()
        if not hasattr(self, 'dig_events'):
            self.log_dig_events()

        # Define how many time bins (each of which is self.time_bin_dur 
        # ms) make up a single time step.
        time_step_size = od([('Delay1', 2),
                             ('Encoding', 6),
                             ('ReturnToBase1', 2),
                             ('Delay2', 2),
                             ('Retrieval', 6),
                             ('ReturnToBase2', 2)])

        cols = ['subj', 'sess', 'trial', 'time_penalty', 'gameState',
                'start_time', 'stop_time', 'time_bin', 'time_step', 'maze_region',
                'head_direc', 'is_moving', 'base_in_view', 'gold_in_view', 'gold_in_view_ids',
                'dig_performed', 'dig_success', 'all_digs', 'dig_ids', 'dig_dists']

        # Create behav_events rows.
        events_behav = []
        subj, sess = self.subj_sess.split('_')
        for idx, row in self.event_times.iterrows():
            trial = row['trial']
            time_penalty = self.time_penalty.loc[trial]
            game_state = row['gameState']
            game_state_dfs = {'positions': self.positions.query("(trial=={}) & (gameState=='{}')".format(trial, game_state)),
                              'base_events': self.base_events.query("(trial=={}) & (gameState=='{}')".format(trial, game_state)),
                              'gold_events': self.gold_events.query("(trial=={}) & (gameState=='{}')".format(trial, game_state)),
                              'dig_events': self.dig_events.query("(trial=={}) & (gameState=='{}')".format(trial, game_state))}
            time_bounds = row['time_bins']

            # Get behavioral vars for each time bin.
            for time_bin, (start, stop) in enumerate(zip(time_bounds[:-1], time_bounds[1:])):
                if game_state in ['Delay1', 'Delay2']:
                    time_step = 1 + int(time_bin / time_step_size[game_state])
                    maze_region = 'Base'
                    head_direc = 'N'
                    is_moving = 0
                    base_in_view = 1
                    gold_in_view = np.nan
                    gold_in_view_ids = np.nan
                    dig_performed = np.nan
                    dig_success = np.nan
                    all_digs = np.nan
                    dig_ids = np.nan
                    dig_dists = np.nan
                elif game_state in ['Encoding', 'ReturnToBase1', 'Retrieval', 'ReturnToBase2']:
                    time_step = 1 + int(time_bin / time_step_size[game_state])
                    maze_region, head_direc, is_moving = self._get_position(start, stop, game_state_dfs['positions'])
                    base_in_view = self._get_base_in_view(start, stop, game_state_dfs['base_events'])
                    gold_in_view, gold_in_view_ids = self._get_gold_in_view(start, stop, game_state_dfs['gold_events'])
                    if game_state == 'Retrieval':
                        dig_performed, dig_success, all_digs, dig_ids, dig_dists = self._get_digs(start, stop, game_state_dfs['dig_events'])
                    else:
                        dig_performed, dig_success, all_digs, dig_ids, dig_dists = np.nan, np.nan, np.nan, np.nan, np.nan
                
                # Add values to the output dataframe.
                events_behav.append([subj, sess, trial, time_penalty, game_state, 
                                     start, stop, time_bin, time_step, maze_region,
                                     head_direc, is_moving, base_in_view, gold_in_view, gold_in_view_ids,
                                     dig_performed, dig_success, all_digs, dig_ids, dig_dists])
                
        events_behav = (pd.DataFrame(events_behav, columns=cols)
                        .sort_values('start_time')
                        .reset_index(drop=True))

        # Assign game_state categories a specific order.
        game_states = ['Delay1', 'Encoding', 'ReturnToBase1', 
                       'Delay2', 'Retrieval', 'ReturnToBase2']
        game_state_cat = pd.CategoricalDtype(game_states, ordered=True)
        events_behav['gameState'] = events_behav['gameState'].astype(game_state_cat)

        self.events_behav = events_behav

    def log_positions(self):
        """Store a DataFrame of all recorded player positions."""
        game_states = ['Encoding', 'ReturnToBase1', 'Retrieval', 'ReturnToBase2']

        cols = ['trial', 'gameState', 'time_penalty',
                'start_time', 'stop_time', 'dur',
                'pos', 'speed', 'maze_region', 'rotation', 'head_direc',
                'moved_pos', 'moved_region', 'moved_rot', 'moved_hd']

        positions = od([])
        for col in cols:
            positions[col] = []

        _trials = self.keep_trials if self.remove_bad_trials else self.trials
        qry = ("(trial=={}) & (gameState=={}) & (key=='playerTransform')"
               .format(_trials, game_states))
        for idx, df in self.events.query(qry).groupby(['trial', 'gameState']):
            trial, gameState = idx
            
            # Calculate the duration (ms) between each position measurement and the next.
            qry = "(gameState=='{}') & (trial=={})".format(gameState, trial)
            event_start, event_stop = self.event_times.query(qry)['time'].iloc[0]
            start_time, stop_time, dur = self._timestamp_intervals(df['time'], event_start, event_stop)

            # Obtain the player's position, speed, and head direction.
            # Speed is defined as the Euclidean distance between the current position and the next,
            # divided by the duration between the current position measurement and the next.
            pos = df['value'].apply(lambda x: (x['positionX'], x['positionZ'])).tolist()
            speed = list((1e3 * np.array(list(np.linalg.norm(np.diff(pos, axis=0), axis=1)) + [0])) / np.array(dur)) # m/s
            maze_region = (df['value']
                           .apply(lambda x: self.maze.point_to_maze_row((x['positionX'], 
                                                                         x['positionZ'])))
                           .tolist())
            rotation = df['value'].apply(lambda x: x['rotationY'] % 360).tolist()
            head_direc = [self._head_direction(x, self.n_direc) for x in rotation]
            
            # Figure out when the player changes position/rotation.
            moved_pos = [x>0 for x in speed]
            moved_region = [False] + [maze_region[i]!=maze_region[i+1] for i in range(len(maze_region)-1)]
            moved_rot = [False] + list(np.diff(rotation)!=0)
            moved_hd = [False] + [head_direc[i]!=head_direc[i+1] for i in range(len(head_direc)-1)]

            # Fill values.
            positions['trial'].extend([trial] * len(df))
            positions['gameState'].extend([gameState for x in range(len(df))])
            positions['time_penalty'].extend([df['time_penalty'].iloc[0]] * len(df))
            positions['start_time'].extend(start_time)
            positions['stop_time'].extend(stop_time)
            positions['dur'].extend(dur)
            positions['pos'].extend(pos)
            positions['speed'].extend(speed)
            positions['maze_region'].extend(maze_region)
            positions['rotation'].extend(rotation)
            positions['head_direc'].extend(head_direc)
            positions['moved_pos'].extend(moved_pos)
            positions['moved_region'].extend(moved_region)
            positions['moved_rot'].extend(moved_rot)
            positions['moved_hd'].extend(moved_hd)
        
        positions = pd.DataFrame(positions).sort_values('start_time').reset_index(drop=True)
        
        self.positions = positions

    def log_base_view_events(self):
        """Create a dataframe for all baseInView events."""
        _trials = self.keep_trials if self.remove_bad_trials else self.trials
        base_events = self.events.query("(trial=={}) & (key=='baseInView')".format(_trials)).copy()
        
        base_events['in_view'] = base_events['value'].apply(lambda x: x['inView'])
        in_view = base_events['in_view'].tolist()
        view_change = [False] + [x[0]!=x[1] for x in zip(in_view[:-1], in_view[1:])]
        base_events['view_change'] = view_change
        
        drop_cols = ['key', 'value', 'scene']
        base_events = base_events.drop(columns=drop_cols).sort_values('time').reset_index(drop=True)
        
        self.base_events = base_events

    def log_gold_view_events(self):
        """Create a dataframe for all goldInView events."""
        _trials = self.keep_trials if self.remove_bad_trials else self.trials
        gold_events = self.events.query("(trial=={}) & (key=='goldInView')".format(_trials)).copy()
        
        gold_events['in_view'] = gold_events['value'].apply(lambda x: x['inView'])
        gold_events['gold_id'] = gold_events['value'].apply(lambda x: x['reportingId'])
        in_view = gold_events['in_view'].tolist()
        view_change = [False] + [x[0]!=x[1] for x in zip(in_view[:-1], in_view[1:])]
        gold_events['view_change'] = view_change
        
        drop_cols = ['key', 'value', 'scene']
        gold_events = gold_events.drop(columns=drop_cols).sort_values('time').reset_index(drop=True)
        
        self.gold_events = gold_events

    def log_gold_spawned(self):
        """Return a dataframe with info on all golds spawned."""
        _trials = self.keep_trials if self.remove_bad_trials else self.trials
        gold_spawned = self.events.query("(trial=={}) & (key=='goldLocation')".format(_trials)).copy()
        
        gold_spawned['gold_id'] = gold_spawned['value'].apply(lambda x: x['reportingId'])
        gold_spawned['gold_x'] = gold_spawned['value'].apply(lambda x: x['positionX'])
        gold_spawned['gold_z'] = gold_spawned['value'].apply(lambda x: x['positionZ'])
        
        drop_cols = ['key', 'value', 'scene']
        gold_spawned = gold_spawned.drop(columns=drop_cols).sort_values('gold_id').reset_index(drop=True)
        
        self.gold_spawned = gold_spawned
    
    def log_dig_events(self):
        """Create a dataframe with info on all dig events."""
        _trials = self.keep_trials if self.remove_bad_trials else self.trials
        dig_events = self.events.query("(trial=={}) & (key=='dig')".format(_trials)).copy()
        
        dig_events['gold_id']= dig_events.apply(
            lambda x: self._nearest_gold_id(x['trial'],
                                            x['value']['nearestGoldPositionX'],
                                            x['value']['nearestGoldPositionZ']),
            axis=1)
        dig_events['dig_success']=dig_events['value'].apply(lambda x: x['successful'])
        dig_events['dist_from_gold']=dig_events['value'].apply(lambda x: x['distanceFromNearestGold'])
        
        drop_cols = ['key', 'value', 'scene']
        dig_events = dig_events.drop(columns=drop_cols).sort_values('time').reset_index(drop=True)
        
        self.dig_events = dig_events

    def _game_state_intervals(self,
                              events,
                              game_state,
                              cols=['time']):
        """Return trial-wise start and stop values for a game state.

        Values are determined by the column names in cols and are
        referenced against the index, with a trial phase running
        from the first index of the trial to the first index of
        the next trial phase.

        Returns
        -------
        pandas.core.frame.DataFrame
        """
        def _first_last(row):
            """Return first and last values in the col iterable."""
            vals = row.index.tolist()
            return [vals[0], vals[-1] + 1]

        # Format inputs correctly.
        if type(cols) == str:
            cols = [cols]

        # Ensure that all indices are consecutive (i.e. we are not accidentally
        # including another gameState in between values for the desired gameState)
        assert np.all([np.all(np.diff(x) == 1)
                       for x in events.query("(gameState=='{}')".format(game_state))
                       .groupby('trial').indices.values()])

        # Group by trial and get the first and last indices for the gameState.
        output_df = (events.query("(gameState=='{}')".format(game_state))
                           .groupby('trial')
                           .apply(lambda x: _first_last(x))
                           .reset_index()
                           .rename(columns={0: 'index'}))

        # Apply the indices to each column that we want to grab values for.
        for col in cols:
            output_df[col] = output_df['index'].apply(lambda x: [events.loc[x[0], col],
                                                                 events.loc[x[1], col]])

        return output_df

    def _head_direction(self,
                        phi,
                        n=4):
        """Return head direction given a phase angle."""
        phi %= 360

        if n == 4:
            if (phi>=315) or (phi<45):
                return 'N'
            elif (phi>=45) and (phi<135):
                return 'E'
            elif (phi>=135) and (phi<225):
                return 'S'
            else:
                return 'W'
        elif n == 8:
            if (phi>=337.5) or (phi<22.5):
                return 'N'
            elif (phi>=22.5) and (phi<67.5):
                return 'NE'
            elif (phi>=67.5) and (phi<112.5):
                return 'E'
            elif (phi>=112.5) and (phi<157.5):
                return 'SE'
            elif (phi>=157.5) and (phi<202.5):
                return 'S'
            elif (phi>=202.5) and (phi<247.5):
                return 'SW'
            elif (phi>=247.5) and (phi<292.5):
                return 'W'
            elif (phi>=292.5) and (phi<337.5):
                return 'NW'

    def _timestamp_intervals(self,
                             timestamps,
                             start=None, 
                             stop=None):
        """Infer time intervals from a vector of ordered timestamps.
        
        Parameters
        ----------
        timestamps : list[int]
            Times should be in ascending order.
        start : int
            If not None, the first timestamp is overwritten with the 
            start value instead.
        stop : int
            If not None, a final interval is added between the last
            timestamp and the stop value. 

        Returns
        -------
        start_time : list[int]
            Start times for each time interval.
        stop_time : list[int]
            Stop times for each time interval.
        dur : list[int]
            Interval durations.
        """
        timestamps = list(timestamps)
        
        if start is not None:
            timestamps[0] = start
        if stop is not None:
            timestamps += [stop]
        
        start_time = timestamps[:-1]
        stop_time = timestamps[1:]
        dur = list(np.array(stop_time) - np.array(start_time))
        
        return start_time, stop_time, dur

    def _nearest_gold_id(self, 
                         trial, 
                         gold_x, 
                         gold_z,
                         tol=0.1):
        """Return unique gold ID from trial number and gold position.
        
        Returns
        -------
        gold_id: id of gold based on table
        """
        if not hasattr(self, 'gold_spawned'):
            self.log_gold_spawned()

        _gold_table = self.gold_spawned.loc[self.gold_spawned['trial']==trial]
        dists = _gold_table.apply(lambda x: np.linalg.norm((x['gold_x'] - gold_x, 
                                                            x['gold_z'] - gold_z)), 
                                  axis=1).tolist()

        xmin = np.argmin(dists)
        if dists[xmin] <= tol:
            return _gold_table.iloc[xmin]['gold_id']
        else:
            return np.nan

    def _get_position(self,
                      start,
                      stop,
                      positions):
        """Return positional info for a specified time window.
        
        Parameters
        ----------
        start : int
        stop : int
        positions : pd.DataFrame
        
        Returns
        -------
        maze_region : str
            Maze region where the most time was spent.
        head_direc : str
            Direction in which the most time was spent facing.
        is_moving : 0 or 1
            1 if the subject moved in space or rotated their head
            at any time during the time window, else 0.
        """
        # Select dataframe rows that fall within the start, stop window.
        pos_idx = np.arange(positions.loc[(positions['start_time']>start)].index[0] - 1,
                            positions.loc[(positions['start_time']<stop)].index[-1] + 1)
        _positions = positions.loc[pos_idx].copy()
        
        # Check that we actually found positions.
        if len(_positions) == 0:
            return np.nan, np.nan, np.nan

        # Rewrite durations to match the beginning and end of the start, stop window.
        _positions.loc[pos_idx[0], 'start_time'] = start
        _positions.loc[pos_idx[-1], 'stop_time'] = stop
        _positions['dur'] = _positions['stop_time'] - _positions['start_time']

        # Get behavior variables.
        maze_region = _positions.groupby('maze_region')['dur'].apply(np.sum).sort_values().index[-1]
        head_direc = _positions.groupby('head_direc')['dur'].apply(np.sum).sort_values().index[-1]
        is_moving = np.any(_positions['moved_pos'] | _positions['moved_rot']).astype(int)
        
        return maze_region, head_direc, is_moving

    def _get_base_in_view(self,
                          start,
                          stop,
                          base_events):
        """Return whether the base was viewed during a specified time window.
        
        Parameters
        ----------
        start : int
        stop : int
        base_events : pd.DataFrame
        
        Returns
        -------
        base_in_view : 0 or 1
            1 if the base was viewed at any time during the 
            time window, else 0.
        """
        _base_events = base_events.loc[(base_events['time']>=start) & (base_events['time']<=stop)]
        
        # Check that we actually found base events.
        if len(_base_events) == 0:
            return np.nan
        
        # Return view info.
        base_in_view = np.any(_base_events['in_view']).astype(int)
        
        return base_in_view

    def _get_gold_in_view(self,
                          start,
                          stop,
                          gold_events):
        """Return whether gold was viewed during a specified time window.
        
        Parameters
        ----------
        start : int
        stop : int
        gold_events : pd.DataFrame
        
        Returns
        -------
        gold_in_view : 0 or 1
            1 if gold was viewed at any time during the 
            time window, else 0.
        gold_in_view_ids : list[str]
            Which golds were viewed, in no particular order.
        """
        _gold_events = gold_events.loc[(gold_events['time']>=start) & (gold_events['time']<=stop)]
        
        # Check that we actually found gold events.
        if len(_gold_events) == 0:
            return np.nan, np.nan
        
        # Return view info.
        gold_in_view = np.any(_gold_events['in_view']).astype(int)
        if gold_in_view:
            gold_in_view_ids = list(np.unique(_gold_events.loc[_gold_events['in_view']==1, 'gold_id']))
        else:
            gold_in_view_ids = np.nan
            
        return gold_in_view, gold_in_view_ids

    def _get_digs(self,
                  start,
                  stop,
                  dig_events):
        """Return dig info for a specified time window.
        
        Parameters
        ----------
        start : int
        stop : int
        dig_events : pd.DataFrame
        
        Returns
        -------        
        dig_performed : 0 or 1
            1 if a dig action occurred at any time during the 
            time window, else 0.
        dig_success : 0 or 1
            1 if any dig was successful during the
            time window, else 0.
        all_digs : list[int]
            Dig outcomes (1: successful; 0: unsuccesful),
            in order of occurrence, during the time window.
        dig_ids : list[str]
            Gold IDs that match the closest gold to each
            dig location. Same order as all_digs.
        dig_dists : list[float]
            Euclidean distances between each dig location
            and the closest gold. Same order as all_digs.
        """
        _dig_events = dig_events.loc[(dig_events['time']>=start) & (dig_events['time']<=stop)]
        
        # Check that we actually found dig events.
        if len(_dig_events) == 0:
            return 0, np.nan, np.nan, np.nan, np.nan
        
        # Return dig info.
        dig_performed = 1
        dig_success = np.any(_dig_events['dig_success']).astype(int)
        all_digs = _dig_events['dig_success'].astype(int).tolist()
        dig_ids = _dig_events['gold_id'].tolist()
        dig_dists = _dig_events['dist_from_gold'].tolist()
        
        return dig_performed, dig_success, all_digs, dig_ids, dig_dists


class Maze(object):
    """Map positional info to a maze configuration."""
    
    def __init__(self, 
                 maze_name,
                 proj_dir='/home1/dscho/projects/time_cells'):
        """Load the maze DataFrame.
        
        Parameters
        ----------
        maze : pd.DataFrame
            This DataFrame contains all of the information needed 
            to reconstruct a maze in (x, y) coordinates.
            Each row contains the coordinates for one rectangle
            in the maze (formatted like [(x1, y1), (x2, y2)]), along 
            with labels that tell us what the rectangle is (wall, room, 
            passageway, etc.) and where it is relative to the player
            spawn point (left, right, center).
        proj_dir : str
            Top level directory for the Goldmine time cell project.
        """
        self.maze_name = maze_name
        self.proj_dir = proj_dir
        self.maze = dio.open_pickle(op.join(proj_dir, 'unity', 'maps', 
                                            '{}.pkl'.format(maze_name)))
        self.maze.reset_index(drop=True, inplace=True)
        coords = np.array(self.maze['coords'].tolist())
        points = np.concatenate((coords[:, 0, :], coords[:, 1, :]))
        self.origin = np.min(points, axis=0)
        self.shape = np.max(points, axis=0) - self.origin # in 2D maze coords
    
    def __str__(self):
        base_tiles = self.maze.query("(region=='base')")
        mine_tiles = self.maze.query("(region!=['base', 'wall'])")
        iwall_tiles = self.maze.query("(region=='wall') & (proxim=='inner')")
        owall_tiles = self.maze.query("(region=='wall') & (proxim=='outer')")
        
        s = '{}\n{}\n'.format(self.maze_name, '-' * len(self.maze_name))
        s += '{} x {} m\n'.format(self.shape[0], self.shape[1])
        s += '{} tiles ({} m^2)\n'.format(len(self.maze), self.maze['area'].sum())
        s += '{} floor tiles in the base ({} m^2)\n'.format(len(base_tiles), 
                                                            base_tiles['area'].sum())
        s += '{} floor tiles in the mine ({} m^2)\n'.format(len(mine_tiles), 
                                                            mine_tiles['area'].sum())
        s += '{} inner wall tiles ({} m^2)\n'.format(len(iwall_tiles), iwall_tiles['area'].sum())
        s += '{} outer wall tiles ({} m^2)'.format(len(owall_tiles), owall_tiles['area'].sum())
        
        return s

    def point_to_maze_row(self, 
                          point,
                          col='region_precise'):
        """Return column value for the maze row that contains point.

        If col=='index', returns the maze row index position instead of
        a column value.
        
        Parameters
        ----------
        point : list
            [x, y]
        """
        def _point_in_coord(p, coord):
            x, y = p
            [(x1, y1), (x2, y2)] = coord
            return (x1<=x<x2) & (y1<=y<y2)
        
        contains_point = self.maze['coords'].apply(lambda x: _point_in_coord(point, x)).tolist()
        if np.sum(contains_point) == 1:
            idx = contains_point.index(True)
            if col == 'index':
                return idx
            else:
                return self.maze.iloc[idx][col]
        else:
            return None


def save_events(events,
                overwrite=False,
                verbose=True):
    """Pickle an Events instance."""
    if op.exists(events.filename) and not overwrite:
        print('Cannot save {} as it already exists'.format(events.filename))
    else:
        dio.save_pickle(events, events.filename, verbose)


def load_events(subj_sess,
                proj_dir='/home1/dscho/projects/time_cells',
                filename=None,
                overwrite=False,
                verbose=True,
                **kwargs):
    """Load pickled Events for the session, else instantiate Events.

    All kwargs are passed to Events.__init__
    """
    if filename is None:
        filename = op.join(proj_dir, 'analysis', 'events', '{}-Events.pkl'.format(subj_sess))

    if op.exists(filename) and not overwrite:
        if verbose:
            print('Loading saved Events file')
        events = dio.open_pickle(filename)
    else:
        if verbose:
            print('Creating Events')
        events = Events(subj_sess,
                        proj_dir=proj_dir,
                        filename=filename,
                        verbose=verbose,
                        **kwargs)
    
    # Do any extra formatting.
    if hasattr(events, 'events_behav'):
        game_states = ['Delay1', 'Encoding', 'ReturnToBase1', 
                       'Delay2', 'Retrieval', 'ReturnToBase2']
        game_state_cat = pd.CategoricalDtype(game_states, ordered=True)
        events.events_behav['gameState'] = events.events_behav['gameState'].astype(game_state_cat)

    return events


def get_game_state_durs():
    """Return the duration of each game state in ms."""
    game_state_durs = {'Prepare1': 2000,
                       'Delay1': 10000,
                       'Encoding': 30000,
                       'Prepare2': 2000,
                       'Delay2': 10000,
                       'Retrieval': 30000}
    
    return game_state_durs


def get_excluded_trials(subj_sess):
    """Store hard-coded trials to remove from all analysis."""
    remove = {'U518_ses1': [31]}
    
    return remove.get(subj_sess, [])
