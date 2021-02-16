"""
events_proc.py

Author
------
Daniel Schonhaut, Jason Chou
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Functions to consolidate Goldmine events into time bins for analysis.

Last Edited
----------- 
2/15/20
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
from time_cells import events_preproc


# def time_bin_interval_align(time_bins_i,time_bins_e, interval_df,labels_extract_list):
#     '''
#     Aligns interval_df to a time_bins list
#     intput: 
#     time_bins_i: list of left edges of time bins
#     time_bins_e: list of right edges of time bins
#     interval_df: interval type dataframe 
#     labels_extract_list: list of column names to extract from the interval
#     output:
#     populated_time_bins_df: df ['start time','end time','column names that were extracted']
#     '''
#     binned_activity = []
#     binned_activity_cols = ['bin_i','bin_e']+labels_extract_list
#     for bin_edges in zip(time_bins_i,time_bins_e):
#         bin_i = bin_edges[0]
#         bin_e = bin_edges[1]
#         df_start = interval_df.query('(start_time < @bin_e) & (start_time>= @bin_i) & (end_time>@bin_e)')
#         df_end   =  interval_df.query('(end_time > @bin_i) & (end_time <= @bin_e) & (start_time<@bin_i)')
#         df_both_in = interval_df.query('(end_time<=@bin_e) & (start_time>=@bin_i)')
#         df_both_out = interval_df.query('(end_time>=@bin_e) & (start_time<=@bin_i)')
#         df_start['overlap']=0
#         df_end['overlap']=0
#         df_both_in['overlap']=0
#         df_both_out['overlap']=0
#         if len(df_start)>0:
#             for iRow in range(len(df_start)):
#                 df_start['overlap'].iloc[iRow]=bin_e - df_start['start_time'].iloc[iRow]
#         if len(df_end)>0:
#             for iRow in range(len(df_end)):
#                 df_end['overlap'].iloc[iRow] = df_end['end_time'].iloc[iRow]-bin_i
#         if len(df_both_in)>0:
#             for iRow in range(len(df_both_in)):
#                 df_both_in['overlap'].iloc[iRow] = df_both_in['end_time'].iloc[iRow]-df_both_in['start_time'].iloc[iRow]
#         if len(df_both_out)>0:
#             for iRow in range(len(df_both_out)):
#                 df_both_out['overlap'].iloc[iRow] = bin_e-bin_i
#         df_merged = pd.concat([df_start,df_end,df_both_in,df_both_out])
#         df_merged = df_merged.sort_values(by=['start_time'])
#         row_max = df_merged['overlap'].argmax()
#         binned_activity.append([bin_i,bin_e]+df_merged[labels_extract_list].iloc[row_max].tolist())
#     binned_activity_df = pd.DataFrame(binned_activity, columns=binned_activity_cols)
#     return binned_activity_df


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
        def point_in_coord(p, coord):
            x, y = p
            [(x1, y1), (x2, y2)] = coord
            return (x1<=x<x2) & (y1<=y<y2)
        contains_point = self.maze['coords'].apply(lambda x: point_in_coord(point, x)).tolist()
        if np.sum(contains_point) == 1:
            idx = contains_point.index(True)
            if col == 'index':
                return idx
            else:
                return self.maze.iloc[idx, col]
        else:
            return None


class Events(object):
    """Analyze the Goldmine behavioral log."""
    
    def __init__(self,
                 subj_sess,
                 proj_dir='/home1/dscho/projects/time_cells'):
        self.subj_sess = subj_sess
        self.proj_dir = proj_dir
        self.events = events_preproc.format_events(self.subj_sess,
                                                   overwrite=False,
                                                   save_output=False,
                                                   proj_dir=self.proj_dir,
                                                   verbose=False)
        self.trials = list(self.events['trial'].unique())
        self.bad_trials = list(np.unique(self.events.query("(bad_trials!='')")['trial'].unique().tolist() + 
                                         events_preproc.get_excluded_trials(self.subj_sess)))
        self.keep_trials = [trial for trial in self.trials if trial not in self.bad_trials]
        self.bad_trial_phases = (self.events.query("(bad_trials!='')")[['trial', 'gameState']]
                                            .drop_duplicates()
                                            .groupby('trial')['gameState']
                                            .apply(lambda x: list(np.unique(x)))
                                            .reset_index().values.tolist()) # e.g. [[1, ['Delay1']], [4, ['Delay2', 'Retrieval']]]
        self.event_times = events_preproc.create_event_time_bins(self.subj_sess,
                                                                 events=self.events,
                                                                 remove_trials=self.bad_trials,
                                                                 verbose=False)
        self.time_penalty = (self.events.query("(key=='trialComplete')")
                                        .set_index('trial')['time_penalty'])
        self.maze_name = self.events.iloc[0]['scene']
        self.maze = Maze(self.maze_name)

    # def add_constant_time_bins(self, bin_size):
    #     '''
    #     Creates constant width time bins given bin size for all navigation phases
    #     input: 
    #     bin_size: width of time bin in 
        
    #     output:
    #     stores a extra column in the event_times df for this list of time bin edges 
    #     '''
    #     def calc_time_bins(time, bin_size):
    #         dur = time[1] - time[0]
    #         bin_num = np.round(dur / bin_size).astype(int)
    #         return np.linspace(time[1], time[1] + (bin_num * bin_size), bin_num + 1)
    #     self.event_times['time_bins_locked_{}ms'.format(bin_size)] = event_times['time'].apply(
    #         lambda x: calc_time_bins(x, bin_size))
    
    def log_positions(self, 
                      game_states=['Encoding', 'ReturnToBase1', 'Retrieval', 'ReturnToBase2'],
                      remove_bad_trials=True):
        """Store a DataFrame of all recorded player positions."""
        cols = ['trial', 'gameState', 'time_penalty',
                'start_time', 'stop_time', 'dur',
                'pos', 'speed', 'maze_region', 'rotation', 'head_direc',
                'moved_pos', 'moved_region', 'moved_rot', 'moved_hd']

        positions = od([])
        for col in cols:
            positions[col] = []

        _trials = self.keep_trials if remove_bad_trials else self.trials
        qry = ("(trial=={}) & (gameState=={}) & (key=='playerTransform')"
               .format(_trials, game_states))
        for idx, df in self.events.query(qry).groupby(['trial', 'gameState']):
            trial, gameState = idx
            
            # Calculate the duration (ms) between each position measurement and the next.
            start_time = []
            stop_time = []
            dur = []
            qry = "(gameState=='{}') & (trial=={})".format(gameState, trial)
            event_start, event_stop = self.event_times.query(qry)['time'].iloc[0]
            for iRow in range(len(df)):
                if iRow == 0:
                    pos_start, pos_stop = event_start, df.iloc[iRow+1]['time']
                elif iRow == len(df) - 1:
                    pos_start, pos_stop = df.iloc[iRow]['time'], event_stop
                else:
                    pos_start, pos_stop = df.iloc[iRow]['time'], df.iloc[iRow+1]['time']
                start_time.append(pos_start)
                stop_time.append(pos_stop)
                dur.append(pos_stop - pos_start)

            # Obtain the player's position and head direction.
            pos = df['value'].apply(lambda x: (x['positionX'], x['positionZ'])).tolist()
            speed = list((1e3 * np.insert(np.linalg.norm(np.diff(pos, axis=0), axis=1), 0, 0)) / np.array(dur)) # m/s
            maze_region = (df['value']
                           .apply(lambda x: self.maze.point_to_maze_row((x['positionX'], 
                                                                         x['positionZ'])))
                           .tolist())
            rotation = df['value'].apply(lambda x: x['rotationY'] % 360).tolist()
            head_direc = [head_direction(x) for x in rotation]
            
            # Figure out when the player changes position/rotation.
            moved_pos = [x>0 for x in speed]
            moved_region = [False] + list(np.diff(maze_region)!=0)
            moved_rot = [False] + list(np.diff(rotation)!=0)
            moved_hd = [False] + list(np.diff(head_direc)!=0)

            # Fill values.
            positions['trial'].extend([trial for x in range(len(df))])
            positions['gameState'].extend([gameState for x in range(len(df))])
            positions['time_penalty'].extend([df['time_penalty'].iloc[0] for x in range(len(df))])
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
        
        self.positions = pd.DataFrame(positions)

    def get_gold_table(self,
                       remove_bad_trials=True):
        """Return a dataframe with info on all golds spawned."""
        _trials = self.keep_trials if remove_bad_trials else self.trials
        gold_table = self.events.query("(trial=={}) & (key=='goldLocation')".format(_trials)).copy()
        gold_table['gold_id'] = gold_table['value'].apply(lambda x: x['reportingId'])
        gold_table['gold_x'] = gold_table['value'].apply(lambda x: x['positionX'])
        gold_table['gold_z'] = gold_table['value'].apply(lambda x: x['positionZ'])
        drop_cols = ['key', 'value', 'scene']
        gold_table = gold_table.drop(columns=drop_cols).reset_index(drop=True)
        return gold_table
    
    def nearest_gold_id(self, 
                        trial, 
                        gold_x, 
                        gold_z,
                        tol=0.1):
        """Return unique gold ID from trial number and gold position.
        
        Returns
        -------
        gold_id: id of gold based on table
        """
        def get_dist(row):
            return np.linalg.norm(np.array((row['gold_x'], row['gold_z'])) - 
                                  np.array((gold_x, gold_z)))

        gold_table = self.get_gold_table()
        gold_table = gold_table.loc[gold_table['trial']==trial]

        gold_table['dist'] = gold_table.apply(lambda x: get_dist(x))
        gold_table.sort_values('dist', ascending=True, inplace=True)
        if gold_table.iloc[0]['dist'] <= tol:
            return gold_table.iloc[0]['gold_id']
        else:
            return np.nan

    def log_dig_events(self,
                       remove_bad_trials=True):
        """Create a dataframe with info on all dig events."""
        _trials = self.keep_trials if remove_bad_trials else self.trials
        dig_events = self.events.query("(trial=={}) & (key=='dig')".format(_trials)).copy()
        dig_events['gold_id']= dig_events.apply(
            lambda x: self.nearest_gold_id(x['trial'],
                                           x['value']['nearestGoldPositionX'],
                                           x['value']['nearestGoldPositionZ']),
            axis=1)
        dig_events['dig_success']=dig_events['value'].apply(lambda x: x['successful'])
        dig_events['dist_from_gold']=dig_events['value'].apply(lambda x: x['distanceFromNearestGold'])
        drop_cols = ['key', 'value', 'scene']
        dig_events = dig_events.drop(columns=drop_cols).reset_index(drop=True)
        self.dig_events = dig_events
    
    def log_gold_view_events(self,
                             remove_bad_trials=True):
        """Create a dataframe for all goldInView events."""
        _trials = self.keep_trials if remove_bad_trials else self.trials
        gold_events = self.events.query("(trial=={}) & (key=='goldInView')".format(_trials)).copy()
        gold_events['in_view'] = gold_events['value'].apply(lambda x: x['inView'])
        gold_events['gold_id'] = gold_events['value'].apply(lambda x: x['reportingId'])
        in_view = gold_events['in_view'].tolist()
        view_change = [False] + [x[0]!=x[1] for x in zip(in_view[:-1], in_view[1:])]
        gold_events['view_change'] = view_change
        drop_cols = ['key', 'value', 'scene']
        gold_events = gold_events.drop(columns=drop_cols).reset_index(drop=True)
        self.gold_events = gold_events
    
    def log_base_view_events(self,
                             remove_bad_trials=True):
        """Create a dataframe for all baseInView events."""
        _trials = self.keep_trials if remove_bad_trials else self.trials
        base_events = self.events.query("(trial=={}) & (key=='baseInView')".format(_trials)).copy()
        base_events['in_view'] = base_events['value'].apply(lambda x: x['inView'])
        in_view = base_events['in_view'].tolist()
        view_change = [False] + [x[0]!=x[1] for x in zip(in_view[:-1], in_view[1:])]
        base_events['view_change'] = view_change
        drop_cols = ['key', 'value', 'scene']
        base_events = base_events.drop(columns=drop_cols).reset_index(drop=True)
        self.base_events = base_events

    # def log_position_intervals(self, 
    #                            idx_on='moved_region'):
    #     """Store a DataFrame of all recorded player position intevals.

    #     Each row gives the game state, trial, player position, head
    #     direction, time window, rotation angle, and duration for one player position
    #     interval, which begins when the player enters a maze tile and 
    #     ends when they leave it. The first position of a trial phase 
    #     begins with the trial phase start time, and the last position 
    #     ends with the trial phase stop time.
    #     """
    #     if not hasattr(self, 'positions'):
    #         self.log_positions()

    #     cols = ['trial', 'gameState', 'pos', 'rotation', 'maze_region', 'head_direc', 'maze_region_hd', 'time', 'dur','time_penalty']
    #     pos_intervals = []
    #     grp = self.positions2.query("({}==True)".format(idx_on)).groupby(['gameState', 'trial'])
    #     for idx, df in grp:
    #         gameState, trial = idx
    #         qry = "(gameState=='{}') & (trial=={})".format(gameState, trial)
    #         event_start, event_stop = self.event_times.query(qry)['time'].iloc[0]
    #         for iRow in range(len(df)):
    #             if iRow == 0:
    #                 if len(df) == 1:
    #                     pos_start, pos_stop = event_start, event_stop
    #                 else:
    #                     pos_start, pos_stop = event_start, df.iloc[iRow+1]['time']
    #             elif iRow == len(df) - 1:
    #                 pos_start, pos_stop = df.iloc[iRow]['time'], event_stop
    #             else:
    #                 pos_start, pos_stop = df.iloc[iRow]['time'], df.iloc[iRow+1]['time']

    #             pos_intervals.append([trial,
    #                                   gameState,
    #                                   df.iloc[iRow]['pos'],
    #                                   df.iloc[iRow]['rotation'],
    #                                   df.iloc[iRow]['maze_region'], 
    #                                   df.iloc[iRow]['head_direc'],
    #                                   df.iloc[iRow]['maze_region_hd'],
    #                                   (pos_start, pos_stop), 
    #                                   pos_stop - pos_start,df.iloc[iRow]['time_penalty']])
    #     self.pos_intervals = pd.DataFrame(pos_intervals, columns=cols)
    #     self.pos_intervals['start_time'] = self.pos_intervals['time'].apply(lambda x: x[0])
    #     self.pos_intervals['end_time'] = self.pos_intervals['time'].apply(lambda x: x[1])
    
    # def log_gold_view_intervals(self):
    #     """Log start and stop times for each baseInView interval."""
    #     #!! need to check whether gold_events was generated
    #     #!! need to add return to base to event_times
    #     gold_intervals = []
    #     gold_cols = ['gameState', 'trial', 'gold_id', 'in_view', 'time', 'dur']
    #     grp = self.gold_events.query('view_change==True').groupby(['gameState', 'trial'])
    #     for idx, df in grp:
    #         gameState, trial = idx
    #         qry = "(gameState=='{}') & (trial=={})".format(gameState, trial)
    #         event_start, event_stop = self.event_times.query(qry)['time'].iloc[0]
    #         for iRow in range(len(df)):
    #             if iRow == 0:
    #                 if len(df) == 1:
    #                     pos_start, pos_stop = event_start, event_stop
    #                 else:
    #                     pos_start, pos_stop = event_start, df.iloc[iRow+1]['time']
    #             elif iRow == len(df) - 1:
    #                 pos_start, pos_stop = df.iloc[iRow]['time'], event_stop
    #             else:
    #                 pos_start, pos_stop = df.iloc[iRow]['time'], df.iloc[iRow+1]['time']
    #             gold_intervals.append([gameState, 
    #                                    trial,
    #                                    df.iloc[iRow]['gold_id'],
    #                                    df.iloc[iRow]['in_view'], 
    #                                    (pos_start, pos_stop), 
    #                                    pos_stop - pos_start])

    #     gold_intervals_df = pd.DataFrame(gold_intervals, columns=gold_cols)
    #     gold_intervals_df['start_time'] = gold_intervals_df['time'].apply(lambda x: x[0])
    #     gold_intervals_df['end_time'] = gold_intervals_df['time'].apply(lambda x: x[1])
    #     self.gold_intervals = gold_intervals_df
        
    # def log_base_view_intervals(self):
    #     """Log start and stop times for each baseInView interval."""
    #     #!! check whether base_events was generated
    #     #!! need to add return to base to event_times
    #     base_intervals = []
    #     base_cols = ['gameState','trial','in_view','time','dur']
    #     grp = self.base_events.query('view_change==True').groupby(['gameState','trial'])
    #     for idx,df in grp:
    #         gameState, trial = idx
    #         qry = "(gameState=='{}') & (trial=={})".format(gameState, trial)
    #         event_start, event_stop = self.event_times.query(qry)['time'].iloc[0]
    #         for iRow in range(len(df)):
    #                 if iRow == 0:
    #                     if len(df) == 1:
    #                         pos_start, pos_stop = event_start, event_stop
    #                     else:
    #                         pos_start, pos_stop = event_start, df.iloc[iRow+1]['time']
    #                 elif iRow == len(df) - 1:
    #                     pos_start, pos_stop = df.iloc[iRow]['time'], event_stop
    #                 else:
    #                     pos_start, pos_stop = df.iloc[iRow]['time'], df.iloc[iRow+1]['time']
    #                 base_intervals.append([gameState, 
    #                                        trial,
    #                                        df.iloc[iRow]['in_view'], 
    #                                        (pos_start, pos_stop), 
    #                                        pos_stop - pos_start])

    #     base_intervals_df = pd.DataFrame(base_intervals, columns=base_cols)
    #     base_intervals_df['start_time'] = base_intervals_df['time'].apply(lambda x: x[0])
    #     base_intervals_df['end_time'] = base_intervals_df['time'].apply(lambda x: x[1])
    #     self.base_intervals = base_intervals_df


# def gen_binned_fr_per_neuron(event_times,spike_df_list):
#     '''
#     Takes in a event_times df from events object and new dataframe containing firing raet for each neuron for each game phase
#     for navigation phases
#     Input:
#     - event_times: event_times df containing bin edges, subj_sess, trial,game_phase
#     - spike_df_list: list of spike_time dfs 
#     output: 
#     - binned_fr: fr for time bins in event_times
#     '''
#     #!! should be integrated as a method too; depends on the function below
#     for spike_df in spike_df_list:
#         spike_times = spike_df['spike_times']
#         event_times['chan'+str(spike_df['chan'])+'_'+'unit'+str(spike_df['unit'])] = event_times['time_bins'].apply(lambda x: bin_fr_nav_output(spike_times,x))
    
#     return event_times

# def bin_fr_nav_output(spike_times,bin_edges):
#     '''
#     Converts spike time for an experiment into binned firing rates given binned edges
#     input: 
#     - spike_times: spike time over the whole experiment for one neuron
#     - bin_edges: time edges for time bins to calculate firing rate rate
    
#     output: 
#     - binned_fr: mean fr per time bin in Hz for each trial
#     '''
#     time_init = bin_edges[0]
#     time_termin = bin_edges[-1]
#     spike_times_filt = spike_times[(spike_times>=time_init) & (spike_times<(time_termin))]
#     spikes_per_bin, _ = np.histogram(spike_times_filt,bin_edges)
#     bin_size = bin_edges[1]-bin_edges[0]
#     binned_fr = spikes_per_bin/bin_size*1000 # convert to Hz
#     return binned_fr


# #!! maze categeorization depends on the following lists
# maze_idx_sets_old = [set([0,1,2,57]),
#                     set([3,4,5,61]),
#                     set([48,49,50,51,60]),
#                     set([52,53,54,55,64]),
#                     set(list(range(6,15))+[58]),
#                     set(list(range(21,30))+[62]),
#                     set(range(30,39)),
#                     set(range(39,48)),
#                     set([56]),
#                     set([66,65,20]),
#                     set(range(15,19)),
#                     set([59,63,19])]
# maze_idx_sets_new = [set([9,10,11,15]),
#                 set([12,13,14,16]),
#                 set([51,52,53,54,49]),
#                 set([55,56,57,58,50]),
#                 set([17,18,19,20,21,22]),
#                 set([23,24,25,26,27,28]),
#                 set([37,38,39,40,41,42]),
#                 set([43,44,45,46,47,48]),
#                 set([0,1,2,3,4,5]),
#                 set([33,34,35,36]),
#                 set([6,7]),
#                 set([29,30,31,32,8])]
# maze_idx_labels = ['SW Hall',
#                   'SE Hall',
#                   'NW Hall',
#                   'NE Hall',
#                   'SW Room',
#                   'SE Room',
#                   'NW Room',
#                   'NE Room',
#                   'Base',
#                   'N Hall',
#                   'S Hall',
#                   'C Hall']

# #!! function for alignment still under construction
# # write function for aligning time bins with arbitrary intervals
# pos_intervals2 = events_sample.pos_intervals2
# time_bins = [865670,865685,867350,867350,867397] # based on row 1,2
# for bin_edges in zip(time_bins[:,-1],time_bins[1:]):
#     bin_i = bin_edges[0]
#     bin_e = bin_edges[1]
#     df_start = pos_intervals2.query('start_time < @bin_e & start_time> @bin_i @ end_time>@bin_e')
#     df_end   =  pos_intervals2.query('end_time > @bin_i & end_time < @bin_e & start_time<@bin_i')
#     df_both = pos_intervals2.query('end_time<@bin_e & start_time>@bin_i')
#     overlap_t = 0
    
# #    if len(df_start)>0:
        
# #    for iRow in range(len(df)):
# #        start_time = df['start_time'].iloc[iRow]
# #        end_time = df['end_time'].iloc[iRow]


def head_direction(phi,
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
