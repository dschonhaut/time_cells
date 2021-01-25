"""
goldmine_behavior.py

Author:
    Daniel Schonhaut
    
Dependencies: 
    Python 3.6, numpy, matplotlib

Description: 
    Analyze and plot Goldmine behavioral results from event logs.

Last Edited: 
    11/29/20
"""
import sys
import os.path as op
from collections import OrderedDict as od
import mkl
mkl.set_num_threads(1)
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sys.path.append('/home1/dscho/code/general')
import data_io as dio
sys.path.append('/home1/dscho/code/projects')
from time_cells import events_preproc


mpl.rcParams['grid.linewidth'] = 0.1
mpl.rcParams['grid.alpha'] = 0.75
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15 
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
colors = ['1f77b4', 'd62728', '2ca02c', 'ff7f0e', '9467bd', 
          '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', colors)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.formatter.offset_threshold'] = 2
mpl.rcParams['axes.labelsize'] = 17
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['axes.titlesize'] = 19
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['legend.loc'] = 'upper right'
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.titlesize'] = 19
mpl.rcParams['figure.figsize'] = (6.85039, 4.79527) 
mpl.rcParams['figure.subplot.wspace'] = 0.25 
mpl.rcParams['figure.subplot.hspace'] = 0.25 
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['pdf.fonttype'] = 42


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
                          point):
        """Return the maze row containing a given (x, y) point.
        
        For a returned value x, access the corresponding
        maze row using maze.iloc[x]
        
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
            return contains_point.index(True)
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
        self.event_times = events_preproc.create_event_time_bins(self.subj_sess,
                                                                 events=self.events,
                                                                 overwrite=True,
                                                                 save_output=False,
                                                                 verbose=False)
        self.trials = list(self.events['trial'].unique())
        self.time_penalty = self.events("(key=='trialComplete')").set_index('trial')['time_penalty']
        self.maze_name = self.events.iloc[0]['scene']
        self.maze = Maze(self.maze_name)
        self.set_plot_params(reset=True)


    def draw_maze(self,
                  maze_vals,
                  show_nav=True,
                  game_state=['Encoding', 'Retrieval'],
                  draw_base=True,
                  ax=None,
                  **kws):
        """Convert maze_vals to a 2D map of the maze environment.."""
        _origin = np.tile(self.maze.origin, 2).reshape((2, 2))

        maze_map = np.zeros(self.maze.shape)
        for idx, val in maze_vals.items():
            maze_tile = np.array(self.maze.maze.iloc[int(idx)]['coords']) - _origin
            maze_map[np.meshgrid(range(maze_tile[0][0], maze_tile[1][0]), 
                                 range(maze_tile[0][1], maze_tile[1][1]))] = val
            
        mask = np.zeros(self.maze.shape)
        for idx, row in self.maze.maze.query("(region=='wall')").iterrows():
            maze_tile = np.array(row['coords']) - _origin
            mask[np.meshgrid(range(maze_tile[0][0], maze_tile[1][0]), 
                             range(maze_tile[0][1], maze_tile[1][1]))] = 1

        # Generate the heatmap.
        if ax is None:
            ax = plt.gca()
        kws['vmin'] = kws.get('vmin', 0)
        kws['vmax'] = kws.get('vmax', np.nanmax(maze_map))
        kws['cmap'] = kws.get('cmap', self.plot_params['cmap'])
        facecolor = kws.pop('facecolor', '#40291c')
        color_nav = kws.pop('color_nav', '#b2babf')
        alpha_nav = kws.pop('alpha_nav', 0.25)
        lw_nav = kws.pop('lw_nav', 0.3)
        color_base = kws.pop('color_base', '#b2babf')
        lw_base = kws.pop('lw_base', 1)
        cbar_label = kws.pop('cbar_label', '')
        cbar_labelpad = kws.pop('labelpad', self.plot_params['labelpad'])

        ax = sns.heatmap(maze_map.T, mask=mask.T, ax=ax, **kws)
        ax.set_facecolor(facecolor)
        if show_nav:
            if isinstance(game_state, str):
                game_state = [game_state]
            all_positions = (self.positions.query("(gameState=={})".format(game_state))
                                           .groupby(['trial', 'gameState'])['pos']
                                           .apply(lambda x: np.array(list(x))).to_dict())
            for k, v in all_positions.items():
                ax.plot(v[:, 0] - self.maze.origin[0],
                        v[:, 1] - self.maze.origin[1],
                        color=color_nav, alpha=alpha_nav, lw=lw_nav)
        if draw_base:
            base_coords = np.array(self.maze.maze.query("(region=='base')")['coords'].tolist())
            base_points = np.concatenate((base_coords[:, 0, :], base_coords[:, 1, :]))
            base_min = (np.min(base_points, axis=0) - self.maze.origin)
            base_max = (np.max(base_points, axis=0) - self.maze.origin)
            base_width, base_height = base_max - base_min
            rect = patches.Rectangle(base_min, base_width, base_height, 
                                     fill=False, lw=lw_base, ec=color_base)
            ax.add_patch(rect)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=self.plot_params['font']['tick'])
        cbar_ticks = np.linspace(kws['vmin'], kws['vmax'], 5)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(['{:g}'.format(x) for x in np.round(cbar_ticks, 1)])
        cbar.set_label(cbar_label, 
                       fontsize=self.plot_params['font']['label'], 
                       labelpad=cbar_labelpad)

        return ax, maze_map, mask
    
    def log_position_intervals(self, 
                               idx_on='moved_pos_or_hd'):
        """Store a DataFrame of all recorded player position intevals.

        Each row gives the game state, trial, player position, head
        direction, time window, and duration for one player position
        interval, which begins when the player enters a maze tile and 
        ends when they leave it. The first position of a trial phase 
        begins with the trial phase start time, and the last position 
        ends with the trial phase stop time.
        """
        if not hasattr(self, 'positions'):
            self.log_positions()

        cols = ['gameState', 'trial', 'pos', 'maze_idx', 'head_direc', 'maze_idx_hd', 'time', 'dur']
        pos_intervals = []
        grp = self.positions.query("({}==True)".format(idx_on)).groupby(['gameState', 'trial'])
        for idx, df in grp:
            gameState, trial = idx
            qry = "(gameState=='{}') & (trial=={})".format(gameState, trial)
            event_start, event_stop = self.event_times.query(qry)['time'].iloc[0]
            for iRow in range(len(df)):
                if iRow == 0:
                    if len(df) == 1:
                        pos_start, pos_stop = event_start, event_stop
                    else:
                        pos_start, pos_stop = event_start, df.iloc[iRow+1]['time']
                elif iRow == len(df) - 1:
                    pos_start, pos_stop = df.iloc[iRow]['time'], event_stop
                else:
                    pos_start, pos_stop = df.iloc[iRow]['time'], df.iloc[iRow+1]['time']
                pos_intervals.append([gameState, 
                                      trial,
                                      df.iloc[iRow]['pos'],
                                      df.iloc[iRow]['maze_idx'], 
                                      df.iloc[iRow]['head_direc'],
                                      df.iloc[iRow]['maze_idx_hd'],
                                      (pos_start, pos_stop), 
                                      pos_stop - pos_start])
        self.pos_intervals = pd.DataFrame(pos_intervals, columns=cols)
        
    def log_positions(self,
                      game_states=['Encoding', 'Retrieval']):
        """Store a DataFrame of all recorded player positions.

        Logs the game state, trial, and time of each position recording, 
        (x, y) position (technically x, z in Unity coordinates), head 
        rotation and direction, and whether the player has moved or 
        rotated their head since the last position recording.
        """
        cols = ['gameState', 'trial', 'time', 'time_penalty', 'dur', 'pos',
                'maze_idx', 'rotation', 'head_direc', 'maze_idx_hd',
                'moved_rawpos', 'moved_pos', 'moved_hd', 'moved_pos_or_hd']

        positions = od([])
        for col in cols:
            positions[col] = []
        for idx, df in (self.events
                        .query("(key=='playerTransform') & (gameState=={})".format(game_states))
                        .groupby(['gameState', 'trial'])):
            gameState, trial = idx
            
            # Calculate the duration (ms) between each position measurement and the next.
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
                dur.append(pos_stop - pos_start)

            # Obtain the player's position and head direction.
            pos = df['value'].apply(lambda x: (x['positionX'], x['positionZ'])).tolist()
            maze_idx = (df['value']
                        .apply(lambda x: self.maze.point_to_maze_row((x['positionX'], 
                                                                      x['positionZ']))).tolist())
            rotation = df['value'].apply(lambda x: x['rotationY'] % 360).tolist()
            head_direc = [head_direction(x) for x in rotation]
            hd_key = {'N': 0.0,
                      'E': 0.25,
                      'S': 0.5,
                      'W': 0.75}
            maze_idx_hd = list(map(lambda x: x[0] + hd_key[x[1]], zip(maze_idx, head_direc)))
            
            # Fill values.
            positions['gameState'].extend([gameState for x in range(len(df))])
            positions['trial'].extend([trial for x in range(len(df))])
            positions['time'].extend(df['time'].tolist())
            positions['time_penalty'].extend([df['time_penalty'].iloc[0] for x in range(len(df))])
            positions['dur'].extend(dur)
            positions['pos'].extend(pos)
            positions['maze_idx'].extend(maze_idx)
            positions['rotation'].extend(rotation)
            positions['head_direc'].extend(head_direc)
            positions['maze_idx_hd'].extend(maze_idx_hd)
            positions['moved_rawpos'].extend([False] + [tuple(x[0])!=tuple(x[1]) for x in 
                                                        zip(np.array(pos).astype(np.int)[:-1], 
                                                            np.array(pos).astype(np.int)[1:])])
            positions['moved_pos'].extend([False] + [x[0]!=x[1] for x in zip(maze_idx[:-1], 
                                                                             maze_idx[1:])])
            positions['moved_hd'].extend([False] + [x[0]!=x[1] for x in zip(head_direc[:-1], 
                                                                            head_direc[1:])])
            positions['moved_pos_or_hd'].extend([False] + [x[0]!=x[1] for x in zip(maze_idx_hd[:-1], 
                                                                                   maze_idx_hd[1:])])
        self.positions = pd.DataFrame(positions)

    def plot_digging_accuracy(self,
                              add_gaussian=True,
                              ax=None,
                              savefig=False,
                              **kws):
        """Calculate digging accuracy by trial.
        
        Returns
        -------
        ax : AxesSubplot
        digacc_by_trial : np.ndarray
        """
        def get_digacc_by_trial():
            dig_events = self.events.query("(key=='dig')")['value']
            select_pos = list(dig_events[dig_events.apply(lambda x: x['successful'])].index)
            select_neg = list(dig_events[dig_events.apply(lambda x: not x['successful'])].index)

            digacc_by_trial = []
            for trial in self.trials:
                n_succ = len(self.events.loc[select_pos].query("(trial=={})".format(trial)))
                n_fail = len(self.events.loc[select_neg].query("(trial=={})".format(trial)))
                if n_succ + n_fail > 0:
                    digacc_by_trial.append(100 * (n_succ / (n_succ+n_fail)))
                else:
                    digacc_by_trial.append(-100)
            digacc_by_trial = np.array(digacc_by_trial)
            return digacc_by_trial

        digacc_by_trial = get_digacc_by_trial()

        if add_gaussian:
            digacc_by_trial_ = digacc_by_trial.copy()
            digacc_by_trial_[np.where(digacc_by_trial_==-100)[0]] = 0
            digacc_by_trial_smooth = gaussian_filter1d(digacc_by_trial_, 3, mode='mirror')
        
        # Generate the plot.
        if ax is None:
            ax = plt.gca()
        ax.plot(np.arange(len(self.trials)), digacc_by_trial, marker='X',
                markeredgewidth=0.4, markeredgecolor='k', 
                markersize=10, linewidth=0, **kws)
        if add_gaussian:
            ax.plot(np.arange(len(self.trials)), digacc_by_trial_smooth, color='k', linewidth=1.2, 
                    zorder=0)
        ax.set_xticks(np.arange(0, len(self.trials)+1, 6, dtype=int))
        ax.set_xticklabels(np.arange(0, len(self.trials)+1, 6, dtype=int), 
                           fontsize=self.plot_params['font']['tick'])
        ax.set_ylim([-5, 105])
        ax.set_yticks(np.linspace(0, 100, 6))
        ax.set_yticklabels(np.linspace(0, 100, 6, dtype=int), 
                           fontsize=self.plot_params['font']['tick'])
        ax.set_xlabel('Trial number', fontsize=self.plot_params['font']['label'], 
                      labelpad=self.plot_params['labelpad'])
        ax.set_ylabel('Digging accuracy (%)', fontsize=self.plot_params['font']['label'], 
                      labelpad=self.plot_params['labelpad'])

        if savefig:
            plt.savefig(op.join(self.proj_dir, 'figs',
                                '{}-behavior-digging_accuracy.pdf'.format(self.subj_sess)),
                        format='pdf', bbox_inches='tight')

        return ax, digacc_by_trial

    def plot_golds_dug(self,
                       ax=None,
                       savefig=False,
                       **kws):
        """
        Plot the number of gold dug by trial.

        Returns
        -------
        ax : AxesSubplot
        gold_by_trial : np.ndarray
        """
        def get_gold_by_trial():
            gold_by_trial = []
            for trial in self.trials:
                gold_by_trial.append(len(self.events.loc[select].query("(trial=={})".format(trial))))
            gold_by_trial = np.cumsum(gold_by_trial)
            return gold_by_trial

        events_score = self.events.query("(key=='score')")['value'] 
        select = list(events_score[events_score.apply(lambda x: x['scoreChange']==10)].index)

        ## ADD CODE FOR GOLD FOUND
        
        gold_by_trial = get_gold_by_trial()

        # gold_by_trial = []
        # for trial in self.trials:
        #     gold_by_trial.append(len(self.events.loc[select].query("(trial=={})".format(trial))))
        # gold_by_trial = np.cumsum(gold_by_trial)

        # Generate the plot.
        if ax is None:
            ax = plt.gca()
        ax.plot(np.arange(len(self.trials)), gold_by_trial, linewidth=2, **kws)
        ax.set_xticks(np.arange(0, len(self.trials)+1, 6, dtype=int))
        ax.set_xticklabels(np.arange(0, len(self.trials)+1, 6, dtype=int), 
                           fontsize=self.plot_params['font']['tick'])
        ax.set_ylim([0, 50])
        ax.set_yticks(np.arange(0, 51, 10, dtype=np.int))
        ax.set_yticklabels(np.arange(0, 51, 10, dtype=np.int), 
                           fontsize=self.plot_params['font']['tick'])
        ax.set_xlabel('Trial number', fontsize=self.plot_params['font']['label'], 
                      labelpad=self.plot_params['labelpad'])
        ax.set_ylabel('Gold collected (cum.)', fontsize=self.plot_params['font']['label'], 
                      labelpad=self.plot_params['labelpad'])

        if savefig:
            plt.savefig(op.join(self.proj_dir, 'figs',
                                '{}-behavior-gold_collected.pdf'.format(self.subj_sess)),
                        format='pdf', bbox_inches='tight')

        return ax, gold_by_trial

    def save_time_indices(self,
                          idx_on=('moved_pos_or_hd', 'maze_idx_hd'),
                          pos_states=['Encoding', 'Retrieval'],
                          tbin_states=['Delay1', 'Encoding', 'Delay2', 'Retrieval'],
                          overwrite=False,
                          save_output=True,
                          verbose=True):
        """Save vectorized indices for each time bin and position."""
        # Load the output file if it exists.
        output_f = op.join(self.proj_dir, 'analysis', 'events', 'time_and_pos', 
                           '{}_event_objs.pkl'.format(self.subj_sess))
        if op.exists(output_f) and not overwrite:
            return dio.open_pickle(output_f)
        
        # Determine when the player was at each position.
        if not hasattr(self, 'positions'):
            self.log_position_intervals(idx_on[0])
            
        # Store indices for each time bin and position.
        self.set_pos_time_idx(idx_on[1], pos_states)
        self.set_tbin_time_idx(tbin_states)

        # Save outputs.
        if save_output:
            _d = {'event_times': self.event_times,
                  'pos_time_idx': self.pos_time_idx,
                  'pos_time_dur': self.pos_time_dur,
                  'tbin_time_idx': self.tbin_time_idx}
            dio.save_pickle(_d, output_f, verbose)

        return _d

    def set_plot_params(self,
                        reset=False,
                        **kws):
        if reset:
            self.plot_params = {}
            self.plot_params['font'] = {'tick': 12,
                                        'label': 14,
                                        'annot': 12,
                                        'fig': 16}
            n_ = 4
            c_ = 2
            self.plot_params['colors'] = [sns.color_palette('Blues', n_)[c_], 
                                          sns.color_palette('Reds', n_)[c_], 
                                          sns.color_palette('Greens', n_)[c_],
                                          sns.color_palette('Purples', n_)[c_],
                                          sns.color_palette('Oranges', n_)[c_],
                                          sns.color_palette('Greys', n_)[c_],
                                          sns.color_palette('YlOrBr', n_+3)[c_],
                                          'k']
            self.plot_params['cmap'] = ['#b2babf'] + sns.color_palette('rocket', 500)
            self.plot_params['labelpad'] = 8
        else:
            for k, v in kws.items():
                self.plot_params['k'] = v

    def set_pos_time_idx(self,
                         idx_on='maze_idx_hd',                        
                         game_states=['Encoding', 'Retrieval']):
        """Store indices for each position in the mine.
        
        Positions are taken from self.position_intervals
        
        pos_time_idx[gameState] is a dict for which:
            - Keys are unique positions or behavioral states.
            - Values are indices to the flattened firing rate vector
              for gameState (from time windows in self.event_times).
              
        pos_time_dur[gameState] is a dict for which:
            - Keys are unique positions or behavioral states.
            - Values are ms for which each position was occupied.
        """
        pos_time_idx = od([])
        pos_time_dur = od([])
        for game_state in game_states:
            pos_intervals_ = self.pos_intervals.query("(gameState=='{}')".format(game_state))
            pos_keys = list(pos_intervals_[idx_on].unique())
            pos_vec = np.concatenate(pos_intervals_.apply(lambda x: x[idx_on] * np.ones(x['dur']), 
                                                          axis=1).tolist())
            pos_time_idx[game_state] = {pos_key: np.where(pos_vec==pos_key)[0] 
                                        for pos_key in pos_keys}
            pos_time_dur[game_state] = {k: len(v) 
                                        for k, v in pos_time_idx[game_state].items()}
        
        self.pos_time_idx = pos_time_idx
        self.pos_time_dur = pos_time_dur
        
    def set_tbin_time_idx(self,
                          game_states=['Encoding', 'Retrieval']):
        """Store indices for each trial phase time bin.
        
        Number of bins is taken from self.event_times
        
        tbin_time_idx[gameState] is a dict for which:
            - Keys are unique time bins that evenly divide
              the trial phase from 0, 1, ..., n_bins-1.
            - Values are indices to the flattened firing rate vector
              for gameState (from time windows in self.event_times).
        """
        def _f(x):
            split_inds = np.array_split(np.arange(x['duration']), x['n_time_bins'])
            return np.concatenate([iBin * np.ones(len(split_inds[iBin]), dtype=np.uint8)
                                   for iBin in range(x['n_time_bins'])])
        
        tbin_time_idx = od([])
        for game_state in game_states:
            event_times_ = self.event_times.query("(gameState=='{}')".format(game_state))
            tbin_vec = np.concatenate(event_times_.apply(lambda x: _f(x), axis=1).tolist())
            tbin_keys = list(np.unique(tbin_vec))
            tbin_time_idx[game_state] = {tbin_key: np.where(tbin_vec==tbin_key)[0] 
                                         for tbin_key in tbin_keys}
        
        self.tbin_time_idx = tbin_time_idx

    def set_trialwise_tbin_time_idx(self,
                                    game_states=['Encoding', 'Retrieval']):
        """Store indices for each trial phase time bin.
        
        Number of bins is taken from self.event_times
        
        tbin_time_idx[gameState] is a dict for which:
            - Keys are unique time bins that evenly divide
              the trial phase from 0, 1, ..., n_bins-1.
            - Values are indices to the flattened firing rate vector
              for gameState (from time windows in self.event_times).
        """
        def _f(x):
            split_inds = np.array_split(np.arange(x['duration']), x['n_time_bins'])
            return np.concatenate([iBin * np.ones(len(split_inds[iBin]), dtype=np.uint8)
                                   for iBin in range(x['n_time_bins'])])
        
        tbin_time_idx = od([])
        for game_state in game_states:
            for trial in self.event_times['trial'].unique():
                event_times_ = self.event_times.query("(gameState=='{}') & (trial=={})"
                                                      .format(game_state, trial))
                tbin_vec = np.concatenate(event_times_.apply(lambda x: _f(x), axis=1).tolist())
                tbin_keys = list(np.unique(tbin_vec))
                tbin_time_idx[game_state] = {tbin_key: np.where(tbin_vec==tbin_key)[0] 
                                             for tbin_key in tbin_keys}
        
        self.tbin_time_idx = tbin_time_idx


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
