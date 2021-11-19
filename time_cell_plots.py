"""
time_cell_plots.py

Author:
    Daniel Schonhaut
    
Dependencies: 
    Python 3.6, numpy, matplotlib

Description: 
    Functions for generating plots for the time cell project.

Last Edited: 
    4/5/21
"""
# General
import sys
from collections import OrderedDict as od

# Scientific
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d

# Plots
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as patches
mpl.rcParams['grid.linewidth'] = 0.1
mpl.rcParams['grid.alpha'] = 0.75
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
colors = ['1f77b4', 'd62728', '2ca02c', 'ff7f0e', '9467bd', 
          '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', colors)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.formatter.offset_threshold'] = 2
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.labelpad'] = 8
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['legend.loc'] = 'upper right'
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.figsize'] = (10, 4) 
mpl.rcParams['figure.subplot.wspace'] = 0.25 
mpl.rcParams['figure.subplot.hspace'] = 0.25 
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['pdf.fonttype'] = 42

# Personal
sys.path.append('/home1/dscho/code/projects')
from time_cells import spike_preproc, events_proc, time_bin_analysis


def time_raster(subj_sess,
                neuron,
                game_states=['Delay1', 'Encoding', 'Delay2', 'Retrieval'],
                plot_vlines=True,
                plot_labels=True,
                plot_game_states=True,
                plot_title=True,
                proj_dir='/home1/dscho/projects/time_cells',
                ax=None,
                **kws):
    """Plot spike rasters for each trial, and return ax."""
    # Load data to make the plot.
    if isinstance(game_states, str):
        game_states = [game_states]
    game_state_durs = od({game_state: events_proc.get_game_state_durs()[game_state]
                          for game_state in game_states})
    events = events_proc.load_events(subj_sess, proj_dir=proj_dir, verbose=False)
    spikes = spike_preproc.load_spikes(subj_sess, neuron, proj_dir=proj_dir)
    
    # Get user-defined params.
    font = kws.pop('font', {'tick': 6, 'label': 7, 'annot': 7, 'fig': 9})
    labelpad = kws.pop('labelpad', 1)
    xtick_inc = kws.pop('xtick_inc', 10)
    rastersize = kws.pop('rastersize', 1)
    rasterwidth = kws.pop('rasterwidth', 0.1)
    ax_linewidth = kws.pop('ax_linewidth', 0.5)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.25 * 0.5), dpi=1200)
        
    # For each trial, convert spike times to a downsampled spike train.
    for iTrial, trial in enumerate(events.keep_trials):
        spike_train = []
        for game_state in game_states:
            qry = "(trial=={}) & (gameState=='{}')".format(trial, game_state)
            row = events.event_times.query(qry).sort_values('trial', ascending=True)
            assert len(row) == 1
            row = row.iloc[0]
            
            # Find spike times within each event window.
            event_start, event_stop = row['time_bins'][0], row['time_bins'][-1]
            spike_times = spikes['spike_times'][(spikes['spike_times']>=event_start) & 
                                                (spikes['spike_times']<event_stop)] - event_start
                        
            # Convert spike times to spike train.
            spike_train_ = np.zeros(game_state_durs[game_state])
            spike_train_[spike_times] = iTrial + 1
            spike_train_[spike_train_==0] = np.nan
            spike_train.append(spike_train_)
                
        spike_train = np.concatenate(spike_train)
        ax.plot(spike_train, linewidth=0, marker='|', 
                markerfacecolor='k', markeredgecolor='k',
                markersize=rastersize, markeredgewidth=rasterwidth)

    # Concentate spikes across trial phases.
    v_lines = np.cumsum([0] + list(game_state_durs.values()))
    if plot_vlines:
        for x in v_lines[1:-1]:
            ax.axvline(x=x, color='k', alpha=1, linewidth=ax_linewidth)
    for axis in ['left', 'bottom']:
        ax.spines[axis].set_linewidth(ax_linewidth)
    ax.tick_params(axis='both', which='both', length=2, width=ax_linewidth, pad=1)
    xticks = np.arange(0, v_lines[-1] + 1, xtick_inc * 1000)
    ax.set_xlim([-1000, v_lines[-1] + 1000])
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.arange(0, (v_lines[-1]/1000)+1, xtick_inc, dtype=np.int), 
                       fontsize=font['tick'], rotation=0)
    if 'ymin' in kws and 'ymax' in kws:
        ax.set_ylim([kws['ymin'], kws['ymax']+1])
    ax.invert_yaxis()
    # yticks = np.arange(12, len(events.keep_trials)+1, 12, dtype=np.int)
    yticks = [10, 20, 30]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=font['tick'], rotation=0)
    if plot_labels:
        ax.set_xlabel('Time (s)', fontsize=font['label'], labelpad=labelpad)
        ax.set_ylabel('Trial', fontsize=font['label'], labelpad=labelpad)
    if plot_game_states:
        annot_x = v_lines[:-1] + (np.diff(v_lines) / 2)
        for ii in range(len(annot_x)):
            ax.text(annot_x[ii]/v_lines[-1], 1.1, game_states[ii].replace('Delay', 'Delay '),
                    ha='center', va='center', fontsize=font['annot'], transform=ax.transAxes)
    if plot_title:
        ax_title = '{}-{} ({})'.format(subj_sess, neuron, spikes['hemroi'])
        if len(game_states) == 1:
            ax_title += ', {}'.format(game_states[0])
        ax.set_title(ax_title, loc='left', pad=6, fontsize=font['fig'])
    
    return ax


def firing_rate_over_time(subj_sess,
                          neuron,
                          game_states=['Delay1', 'Encoding', 'Delay2', 'Retrieval'],
                          overlap=False,
                          smooth=0,
                          plot_grand_mean=False,
                          plot_vlines=True,
                          plot_labels=True,
                          plot_game_states=True,
                          plot_title=True,
                          proj_dir='/home1/dscho/projects/time_cells',
                          ax=None,
                          **kws):
    """Plot mean ± SEM firing rates over time for neuron, and return ax.
    
    Uses 500ms time bins.
    """
    # Get user-defined params.
    font = kws.pop('font', {'tick': 6, 'label': 7, 'annot': 7, 'fig': 9})
    if overlap:
        linecolor = kws.pop('linecolor', ['#e10600', '#296eb4'])
        linestyle = kws.pop('linestyle', ['-', '--'])
    else:
        linecolor = kws.pop('linecolor', '#e10600')
        linestyle = kws.pop('linestyle', '-')
    linewidth = kws.pop('linewidth', 0.6)
    alpha = kws.pop('alpha', 0.15)
    grand_mean_color = kws.pop('grand_mean_color', '#296eb4')
    grand_mean_linestyle = kws.pop('grand_mean_linestyle', '--')
    labelpad = kws.pop('labelpad', 1)
    xtick_inc = kws.pop('xtick_inc', 10)
    ax_linewidth = kws.pop('ax_linewidth', 0.5)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.25 * 0.5), dpi=1200)
        
    # Get some other params we'll need for plotting.
    timebin_size = 500
    mult = 1000 / timebin_size
    if isinstance(game_states, str):
        game_states = [game_states]
    game_state_durs = od({game_state: events_proc.get_game_state_durs()[game_state]
                          for game_state in game_states})
    if overlap:
        v_lines = (np.array([0, np.max(list(game_state_durs.values()))]) / timebin_size).astype(int)
    else:
        v_lines = (np.cumsum([0] + list(game_state_durs.values())) / timebin_size).astype(int)
    xticks = np.arange(0, v_lines[-1] + 1, int((xtick_inc*1000)/timebin_size))
    xticklabels = (xticks * (timebin_size/1000)).astype(int)
    
    # Get spikes for each trial, in each time bin.
    event_spikes = time_bin_analysis.load_event_spikes(subj_sess, verbose=0)
    spike_mat = [event_spikes.get_spike_mat(neuron, game_state).values * mult
                 for game_state in game_states] # [trial x time_bin,]

    # Smooth spike counts.
    if smooth > 0:
        for ii in range(len(spike_mat)):
            spike_mat[ii] = np.array([gaussian_filter1d(spike_mat[ii][iTrial, :].astype(float), smooth)
                                      for iTrial in range(spike_mat[ii].shape[0])])

    # Calculate mean and SEM firing rates across trials.
    mean_frs = [np.nanmean(_spike_mat, axis=0) for _spike_mat in spike_mat]
    sem_frs = [stats.sem(_spike_mat, axis=0, nan_policy='omit') for _spike_mat in spike_mat]
    grand_mean = [np.mean(np.nanmean(_spike_mat, axis=1)) for _spike_mat in spike_mat]
    grand_sem = [stats.sem(np.nanmean(_spike_mat, axis=1)) for _spike_mat in spike_mat]

    # Make plot.
    for ii in range(len(game_states)):
        if overlap:
            xvals = np.arange(mean_frs[ii].size) + 0.5
            _linecolor = linecolor[ii]
            _linestyle = linestyle[ii]
        else:
            xvals = np.arange(v_lines[ii], v_lines[ii+1]) + 0.5
            _linecolor = linecolor
            _linestyle = linestyle
        if plot_grand_mean:
            ax.fill_between(xvals,
                            ([grand_mean[ii]] * xvals.size) + grand_sem[ii],
                            ([grand_mean[ii]] * xvals.size) - grand_sem[ii],
                            color=grand_mean_color, linewidth=0, alpha=alpha)
        ax.fill_between(xvals,
                        mean_frs[ii] + sem_frs[ii],
                        mean_frs[ii] - sem_frs[ii],
                        color=_linecolor, linewidth=0, alpha=alpha)
        if plot_grand_mean:
            ax.plot(xvals, [grand_mean[ii]] * xvals.size,
                    color=grand_mean_color, linewidth=linewidth, linestyle=grand_mean_linestyle)
        ax.plot(xvals, mean_frs[ii],
                color=_linecolor, linewidth=linewidth, linestyle=_linestyle)
    
    if plot_vlines & (not overlap):
        for x in v_lines[1:-1]:
            ax.axvline(x=x, color='k', alpha=1, linewidth=ax_linewidth)

    # Configure plot params.
    xmin = xticks[0]-int(1000/timebin_size)
    xmax = xticks[-1]+int(1000/timebin_size)
    mean_frs = np.concatenate(mean_frs)
    sem_frs = np.concatenate(sem_frs)
    ymin = kws.pop('ymin', np.max((0, np.floor(np.min(mean_frs - sem_frs)))))
    ymax = kws.pop('ymax', np.ceil(np.max(mean_frs + sem_frs)))
    yticks = kws.pop('yticks', np.round(np.linspace(ymin, ymax, 3), 1))
    for axis in ['left', 'bottom']:
        ax.spines[axis].set_linewidth(ax_linewidth)
    ax.tick_params(axis='both', which='both', length=2, width=ax_linewidth, pad=1)
    ax.set_xlim([xmin, xmax])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=font['tick'], rotation=0)
    ax.set_ylim([ymin, ymax])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=font['tick'], rotation=0)
    if plot_labels:
        ax.set_xlabel('Time (s)', fontsize=font['label'], labelpad=labelpad)
        ax.set_ylabel('Firing rate (Hz)', fontsize=font['label'], labelpad=labelpad)
    if plot_game_states:
        annot_x = v_lines[:-1] + (np.diff(v_lines) / 2)
        for ii in range(len(annot_x)):
            ax.text(annot_x[ii]/v_lines[-1], 1.1, game_states[ii].replace('Delay', 'Delay '),
                    ha='center', va='center', fontsize=font['annot'], transform=ax.transAxes)
    if plot_title:
        hemroi = spike_preproc.roi_lookup(subj_sess, neuron.split('-')[0])
        ax_title = '{}-{} ({})'.format(subj_sess, neuron, hemroi)
        if len(game_states) == 1:
            ax_title += ', {}'.format(game_states[0])
        ax.set_title(ax_title, loc='left', pad=6, fontsize=font['fig'])
        
    return ax


def plot_firing_maze2(subj_sess,
                      neuron,
                      scale_by=10,
                      draw_base=True,
                      show_nav=True,
                      show_spikes=True,
                      only_show_spikes_when_moving=False,
                      proj_dir='/home1/dscho/projects/time_cells',
                      **kws):
    """Plot unit firing by 2D maze position."""
    # -------------------------------------------------
    # Get user-defined plot params.
    colws = kws.pop('colws', {1: 2.05, 2: 3.125, 3: 6.45})
    grid_shp = (50, 120)
    figsize = (colws[1], colws[1] * (grid_shp[0]/grid_shp[1]))
    dpi = kws.pop('dpi', 1200)
    plt.close()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = [plt.subplot2grid(grid_shp, (0, 0), rowspan=50, colspan=50),
          plt.subplot2grid(grid_shp, (0, 51), rowspan=50, colspan=3),
          plt.subplot2grid(grid_shp, (0, 66), rowspan=50, colspan=50),
          plt.subplot2grid(grid_shp, (0, 117), rowspan=50, colspan=3)]
    game_states = ['Encoding', 'Retrieval']
    font = kws.pop('font', {'tick': 5, 'label': 6, 'fig': 7})
    wall_cmap = kws.pop('wall_cmap', 'binary')
    wall_vmin = kws.pop('wall_vmin', 0)
    wall_vmax = kws.pop('wall_vmax', 1)
    fr_cmap = kws.pop('fr_cmap', 'binary_r')
    base_lw = kws.pop('base_lw', 0.2)
    base_color = kws.pop('base_color', '#538d89')
    background_color = kws.pop('background_color', 'k')
    nav_alpha = kws.pop('nav_alpha', 0.25)
    nav_lw = kws.pop('nav_lw', 0.06)
    nav_color = kws.pop('nav_color', '#f0b2b2')
    spike_marker = kws.pop('spike_marker', 'x')
    spike_fill_color = kws.pop('spike_fill_color', '#e10600')
    spike_edge_color = kws.pop('spike_edge_color', 'w')
    ticklength = kws.pop('ticklength', 1)
    tickwidth = kws.pop('tickwidth', 0.25)
    tickpad = kws.pop('tickpad', 1)
    labelpad = kws.pop('labelpad', 1.5)
    cbar_label = kws.pop('cbar_label', None)
    spike_alpha = kws.pop('spike_alpha', 0.5)
    spike_markersize = kws.pop('spike_markersize', 0.75)
    spike_mew = kws.pop('spike_mew', 0.15)

    # Load player position and spiking data.
    events = events_proc.load_events(subj_sess, proj_dir=proj_dir, verbose=False)
    events.maze.maze.loc[:, 'region_precise'] = events.maze.maze['region_precise'].apply(lambda x: x.replace(' ', '_'))
    _origin = events.maze.origin * scale_by
    event_spikes = time_bin_analysis.load_event_spikes(subj_sess, proj_dir=proj_dir, verbose=False)

    # Calculate mean firing rate (Hz) in each maze region.
    fr_pos = od([])
    for game_state in game_states:
        fr_pos[game_state] = (event_spikes.event_spikes.query("(gameState=='{}')".format(game_state))
                                                       .groupby('maze_region')[neuron].mean() * 2)

    _min = np.min([fr_pos[game_state][np.isfinite(fr_pos[game_state])].min() for game_state in game_states])
    _max = np.max([fr_pos[game_state][np.isfinite(fr_pos[game_state])].max() for game_state in game_states])
    kws['vmin'] = kws.get('vmin', np.max((0, _min)))
    kws['vmax'] = kws.get('vmax', _max)

    # Make the plot.
    for game_state in game_states:
        if game_state == 'Encoding':
            iax = 0
        else:
            iax = 2

        # Find the most recent position to each spike.
        spike_times = spike_preproc.load_spikes(subj_sess, neuron)['spike_times']
        spike_loc = []
        for trial in events.keep_trials:
            qry = "(trial=={}) & (gameState=='{}')".format(trial, game_state)
            _positions = events.positions.query(qry).reset_index(drop=True)
            keep_spikes = np.any(events.event_times.query(qry)['time_bins']
                                                   .apply(lambda x: [x[0]<=spike<x[-1] for spike in spike_times])
                                                   .tolist(), axis=0)
            _spike_times = spike_times[keep_spikes]
            spike_pos_arr = np.array(_positions['start_time']
                                     .apply(lambda x: [spike - x for spike in _spike_times]).tolist()) # pos x spike
            spike_pos_arr[spike_pos_arr<0] = 1e6
            _spike_loc = (_positions.loc[np.argmin(spike_pos_arr, axis=0), 'pos']
                                    .apply(lambda x: tuple([((x[_i] * scale_by) - _origin[_i]) 
                                                            for _i in range(len(x))])))
            if only_show_spikes_when_moving:
                keep_spikes = _positions.loc[np.argmin(spike_pos_arr, axis=0), 'moved_pos']
                _spike_loc = _spike_loc[keep_spikes]
            spike_loc += _spike_loc.tolist()
        spike_loc = np.array(spike_loc) # spike x (xPos, yPos)

        # Get mean firing at each location.
        _fr_pos = fr_pos[game_state]
        shp = events.maze.shape * scale_by
        maze_fr_mat = np.zeros(shp) * np.nan
        for idx, maze_row in events.maze.maze.iterrows():
            if maze_row['region_precise'] in _fr_pos:
                coords = (np.array(maze_row['coords']) * scale_by) - _origin
                mesh_coords = tuple(np.meshgrid(range(coords[0][0], coords[1][0]), 
                                                range(coords[0][1], coords[1][1])))
                maze_fr_mat[mesh_coords] = _fr_pos[maze_row['region_precise']]

        # Get a mask of the maze walls.
        mask = np.zeros(shp) * np.nan
        for idx, maze_row in events.maze.maze.query("(region=='wall')").iterrows():
            coords = (np.array(maze_row['coords']) * scale_by) - _origin
            mesh_coords = tuple(np.meshgrid(range(coords[0][0], coords[1][0]), 
                                            range(coords[0][1], coords[1][1])))
            mask[mesh_coords] = 1

        # Draw the walls.
        _ = ax[iax].imshow(mask.T, cmap=wall_cmap, vmin=wall_vmin, vmax=wall_vmax)

        # Draw a border around the base.
        if draw_base:
            coords = (np.array(events.maze.maze.query("(region=='base')")['coords'].tolist()) * scale_by) - _origin
            coords = np.concatenate((coords[:, 0, :], coords[:, 1, :]))
            base_min = np.min(coords, axis=0)
            base_max = np.max(coords, axis=0)
            base_width, base_height = base_max - base_min
            rect = patches.Rectangle(base_min, base_width, base_height, 
                                     fill=False, lw=base_lw, ec=base_color)
            ax[iax].add_patch(rect)

        # Make firing rate by maze region heatmap.
        ax[iax] = sns.heatmap(maze_fr_mat.T, ax=ax[iax], cbar=True, 
                              cbar_ax=ax[iax+1], cmap=fr_cmap, **kws)

        # Overlay position trajectories.
        if show_nav:
            # all_pos : pd.Series
            #     keys: trial numbers
            #     values: (time, (xPos, yPos)) array of position recordings
            all_pos = (events.positions.query("(gameState=='{}')".format(game_state))
                                       .groupby(['trial'])['pos']
                                       .apply(lambda x: (np.array(x.tolist()) * scale_by) - _origin))
            for trial in all_pos.index:
                ax[iax].plot(all_pos[trial][:, 0], 
                             all_pos[trial][:, 1],
                             lw=nav_lw,
                             alpha=nav_alpha,
                             color=nav_color)

        # Overlay spikes.
        if show_spikes:
            ax[iax].plot(spike_loc[:, 0], spike_loc[:, 1], lw=0, alpha=spike_alpha, 
                         marker=spike_marker, ms=spike_markersize, mew=spike_mew,
                         mfc=spike_fill_color, mec=spike_edge_color)
        
        # Configure the colorbar.
        cbar = ax[iax].collections[0].colorbar
        cbar.ax.tick_params(labelsize=font['tick'], length=ticklength, width=tickwidth, pad=tickpad)
        cbar_ticks = np.linspace(kws['vmin'], kws['vmax'], 5)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(['{:g}'.format(x) for x in np.round(cbar_ticks, 1)])
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=font['label'], labelpad=labelpad)
        
        # Tweak other plot params.
        ax[iax].set_facecolor(background_color)
        ax[iax].invert_yaxis()
        ax[iax].set_xticks([])
        ax[iax].set_yticks([])
    
    return fig, ax


def plot_firing_maze(subj_sess,
                     neuron,
                     game_states,
                     scale_by=10,
                     draw_base=True,
                     show_nav=True,
                     show_spikes=True,
                     only_show_spikes_when_moving=False,
                     proj_dir='/home1/dscho/projects/time_cells',
                     **kws):
    """Plot unit firing by 2D maze position."""
    # Load player position and spiking data.
    events = events_proc.load_events(subj_sess, proj_dir=proj_dir, verbose=False)
    events.maze.maze.loc[:, 'region_precise'] = events.maze.maze['region_precise'].apply(lambda x: x.replace(' ', '_'))
    _origin = events.maze.origin * scale_by
    event_spikes = time_bin_analysis.load_event_spikes(subj_sess, proj_dir=proj_dir, verbose=False)
    if isinstance(game_states, str):
        game_states = [game_states]

    # Find the most recent position to each spike.
    spike_times = spike_preproc.load_spikes(subj_sess, neuron)['spike_times']
    spike_loc = []
    for trial in events.keep_trials:
        qry = "(trial=={}) & (gameState=={})".format(trial, game_states)
        _positions = events.positions.query(qry).reset_index(drop=True)
        keep_spikes = np.any(events.event_times.query(qry)['time_bins']
                                               .apply(lambda x: [x[0]<=spike<x[-1] for spike in spike_times])
                                               .tolist(), axis=0)
        _spike_times = spike_times[keep_spikes]
        spike_pos_arr = np.array(_positions['start_time']
                                 .apply(lambda x: [spike - x for spike in _spike_times]).tolist()) # pos x spike
        spike_pos_arr[spike_pos_arr<0] = 1e6
        _spike_loc = (_positions.loc[np.argmin(spike_pos_arr, axis=0), 'pos']
                                .apply(lambda x: tuple([((x[_i] * scale_by) - _origin[_i]) 
                                                        for _i in range(len(x))])))
        if only_show_spikes_when_moving:
            keep_spikes = _positions.loc[np.argmin(spike_pos_arr, axis=0), 'moved_pos']
            _spike_loc = _spike_loc[keep_spikes]
        spike_loc += _spike_loc.tolist()
    spike_loc = np.array(spike_loc) # spike x (xPos, yPos)

    # Calculate mean firing rate (Hz) in each maze region.
    fr_pos = (event_spikes.event_spikes.query("(gameState=={})".format(game_states))
                                       .groupby('maze_region')[neuron].mean() * 2)  
    shp = events.maze.shape * scale_by
    maze_fr_mat = np.zeros(shp) * np.nan
    for idx, maze_row in events.maze.maze.iterrows():
        if maze_row['region_precise'] in fr_pos:
            coords = (np.array(maze_row['coords']) * scale_by) - _origin
            mesh_coords = tuple(np.meshgrid(range(coords[0][0], coords[1][0]), 
                                            range(coords[0][1], coords[1][1])))
            maze_fr_mat[mesh_coords] = fr_pos[maze_row['region_precise']]

    # Get a mask of the maze walls.
    mask = np.zeros(shp) * np.nan
    for idx, maze_row in events.maze.maze.query("(region=='wall')").iterrows():
        coords = (np.array(maze_row['coords']) * scale_by) - _origin
        mesh_coords = tuple(np.meshgrid(range(coords[0][0], coords[1][0]), 
                                        range(coords[0][1], coords[1][1])))
        mask[mesh_coords] = 1
    
    # -------------------------------------------------
    # Get user-defined plot params.
    if 'vmin' not in kws:
        if fr_pos.max() < 6:
            kws['vmin'] = 0
        else:
            kws['vmin'] = np.max((0, int(fr_pos.min() - (0.2 * (fr_pos.max() - fr_pos.min())))))
    kws['vmax'] = kws.get('vmax', np.max(maze_fr_mat[np.isfinite(maze_fr_mat)]))
    colws = kws.pop('colws', {1: 2.05, 2: 3.125, 3: 6.45})
    grid_shp = (100, 107)
    figsize = (colws[1], colws[1] * (grid_shp[0]/grid_shp[1]))
    dpi = kws.pop('dpi', 1200)
    font = kws.pop('font', {'tick': 6, 'label': 7, 'fig': 9})
    wall_cmap = kws.pop('wall_cmap', 'binary')
    wall_vmin = kws.pop('wall_vmin', 0)
    wall_vmax = kws.pop('wall_vmax', 1)
    fr_cmap = kws.pop('fr_cmap', 'binary_r')
    base_lw = kws.pop('base_lw', 0.4)
    base_color = kws.pop('base_color', '#538d89')
    nav_alpha = kws.pop('nav_alpha', 0.25)
    nav_lw = kws.pop('nav_lw', 0.12)
    nav_color = kws.pop('nav_color', '#f0b2b2')
    spike_marker = kws.pop('spike_marker', 'x')
    spike_fill_color = kws.pop('spike_fill_color', '#e10600')
    spike_edge_color = kws.pop('spike_edge_color', 'w')
    ticklength = kws.pop('ticklength', 2)
    tickwidth = kws.pop('tickwidth', 0.5)
    tickpad = kws.pop('tickpad', 2)
    labelpad = kws.pop('labelpad', 3)
    cbar_label = kws.pop('cbar_label', None)
    
    # Autoscale spikes if params are not passed.
    if np.any(('spike_alpha' in kws, 'spike_markersize' in kws, 'spike_mew' in kws)):
        spike_alpha = kws.pop('spike_alpha', 0.5)
        spike_markersize = kws.pop('spike_markersize', 0.75)
        spike_mew = kws.pop('spike_mew', 0.15)
    else:
        if len(spike_loc) < 3000:
            spike_alpha = 0.5
            spike_markersize = 0.75
            spike_mew = 0.15
        elif len(spike_loc) < 6000:
            spike_alpha = 0.4
            spike_markersize = 0.7
            spike_mew = 0.14
        elif len(spike_loc) < 10000:
            spike_alpha = 0.3
            spike_markersize = 0.65
            spike_mew = 0.12
        else:
            spike_alpha = 0.2
            spike_markersize = 0.55
            spike_mew = 0.1
    
    # Make the plot.
    plt.close()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = [plt.subplot2grid(grid_shp, (0, 0), rowspan=100, colspan=100),
          plt.subplot2grid(grid_shp, (0, 101), rowspan=100, colspan=6)]
    iax = 0
    
    # Draw the walls.
    _ = ax[iax].imshow(mask.T, cmap=wall_cmap, vmin=wall_vmin, vmax=wall_vmax)

    # Draw a border around the base.
    if draw_base:
        coords = (np.array(events.maze.maze.query("(region=='base')")['coords'].tolist()) * scale_by) - _origin
        coords = np.concatenate((coords[:, 0, :], coords[:, 1, :]))
        base_min = np.min(coords, axis=0)
        base_max = np.max(coords, axis=0)
        base_width, base_height = base_max - base_min
        rect = patches.Rectangle(base_min, base_width, base_height, 
                                 fill=False, lw=base_lw, ec=base_color)
        ax[iax].add_patch(rect)

    # Make firing rate by maze region heatmap.
    ax[iax] = sns.heatmap(maze_fr_mat.T, ax=ax[iax], cbar=True, 
                          cbar_ax=ax[iax+1], cmap=fr_cmap, **kws)

    # Overlay position trajectories.
    if show_nav:
        # all_pos : pd.Series
        #     keys: trial numbers
        #     values: (time, (xPos, yPos)) array of position recordings
        all_pos = (events.positions.query("(gameState=={})".format(game_states))
                                   .groupby(['trial', 'gameState'])['pos']
                                   .apply(lambda x: (np.array(x.tolist()) * scale_by) - _origin))
        for trial in all_pos.index:
            ax[iax].plot(all_pos[trial][:, 0], 
                         all_pos[trial][:, 1],
                         lw=nav_lw,
                         alpha=nav_alpha,
                         color=nav_color)

    # Overlay spikes.
    if show_spikes:
        ax[iax].plot(spike_loc[:, 0], spike_loc[:, 1], lw=0, alpha=spike_alpha, 
                     marker=spike_marker, ms=spike_markersize, mew=spike_mew,
                     mfc=spike_fill_color, mec=spike_edge_color)
    
    # Configure the colorbar.
    cbar = ax[iax].collections[0].colorbar
    cbar.ax.tick_params(labelsize=font['tick'], length=ticklength, width=tickwidth, pad=tickpad)
    cbar_ticks = np.linspace(kws['vmin'], kws['vmax'], 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(['{:g}'.format(x) for x in np.round(cbar_ticks, 1)])
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=font['label'], labelpad=labelpad)
    
    # Tweak other plot params.
    ax[iax].set_facecolor('k')
    ax[iax].invert_yaxis()
    ax[iax].set_xticks([])
    ax[iax].set_yticks([])
    
    return fig, ax


def plot_firing_hd(subj_sess,
                   neuron,
                   game_state,
                   n_direc=8,
                   show_center_rlabel=False,
                   ax=None,
                   proj_dir='/home1/dscho/projects/time_cells',
                   **kws):
    """Plot unit firing by 2D maze position."""
    def update_rpads(ax, rpad):
        angle = ax._r_label_position.to_values()[4]
        rmax = ax.get_rmax()
        ax._r_label_position.clear().translate(angle, rpad * rmax)
        
    # Load player position and spiking data.
    events = events_proc.load_events(subj_sess, proj_dir=proj_dir, verbose=False)
    spike_times = spike_preproc.load_spikes(subj_sess, neuron)['spike_times']
    
    # Find the most recent player transform to each spike.
    spike_rot = []
    for trial in events.keep_trials:
        qry = "(trial=={}) & (gameState=='{}')".format(trial, game_state)
        _positions = events.positions.query(qry).reset_index(drop=True)
        keep_spikes = np.any(events.event_times.query(qry)['time_bins']
                                               .apply(lambda x: [x[0]<=spike<x[-1] for spike in spike_times])
                                               .tolist(), axis=0)
        _spike_times = spike_times[keep_spikes]
        spike_pos_arr = np.array(_positions['start_time']
                                 .apply(lambda x: [spike - x for spike in _spike_times]).tolist()) # pos x spike
        spike_pos_arr[spike_pos_arr<0] = 1e6
        spike_rot += _positions.loc[np.argmin(spike_pos_arr, axis=0), 'rotation'].tolist()
    spike_rot = np.array(spike_rot)
    
    # How many spikes were recorded in each direction?
    spike_hd = np.unique([events._head_direction(x, n_direc) for x in spike_rot], return_counts=True)
    
    # How many positions were recorded in each direction?
    all_hd = np.unique([events._head_direction(x, n_direc)
                        for x in events.positions.query("(trial=={}) & (gameState=='{}')"
                                                        .format(events.keep_trials, game_state))
                        ['rotation']], return_counts=True)
    
    # Combine into a dataframe.
    head_direc = pd.concat((pd.Series(all_hd[1], all_hd[0], name='pos_count'), 
                            pd.Series(spike_hd[1], spike_hd[0], name='spikes')), 
                           axis=1).reset_index().rename(columns={'index': 'hd'})
    hd_angles = {'N':  (2*np.pi)/4,
                 'NE': (1*np.pi)/4,
                 'E':  (0*np.pi)/4,
                 'SE': (7*np.pi)/4,
                 'S':  (6*np.pi)/4,
                 'SW': (5*np.pi)/4,
                 'W':  (4*np.pi)/4,
                 'NW': (3*np.pi)/4}
    head_direc.insert(1, 'hd_angle', head_direc['hd'].apply(lambda x: hd_angles[x]))
    head_direc = head_direc.sort_values('hd_angle').reset_index(drop=True)

    # What proportion of time was spent going in each direction?
    head_direc.insert(3, 'pct_time', head_direc['pos_count'] / np.sum(head_direc['pos_count']))

    # Calculate firing rates (Hz) in each direction.
    qry = "(trial=={}) & (gameState=='{}')".format(events.keep_trials, game_state)
    duration = events.event_times.query(qry)['time_bin_dur'].sum() * 1e-3
    head_direc['fr'] = head_direc['spikes'] / (duration * head_direc['pct_time'])
    
    # -------------------------------------------------
    # Get user-defined plot params.
    if 'vmin' in kws:
        vmin = kws.pop('vmin')
    elif head_direc['fr'].max() < 6:
        vmin = 0
    else:
        vmin = np.max((0, int(head_direc['fr'].min() - (0.2 * (head_direc['fr'].max() - head_direc['fr'].min())))))
    vmax = kws.get('vmax', head_direc['fr'].max() + (0.2 * (head_direc['fr'].max() - head_direc['fr'].min())))
    colws = kws.pop('colws', {1: 2.05, 2: 3.125, 3: 6.45})
    dpi = kws.pop('dpi', 1200)
    font = kws.pop('font', {'tick': 5, 'label': 6.5})
    bar_width = kws.pop('bar_width', 0.785)
    bar_lw = kws.pop('bar_lw', 0.4)
    bar_color = kws.pop('bar_color', '#e10600')
    bar_ec = kws.pop('bar_ec', mpl.rcParams['grid.color'])
    grid_lw = kws.pop('grid_lw', 0.3)
    grid_linestyle = kws.pop('grid_linestyle', 'dotted')
    grid_color = kws.pop('grid_color', 'k')
    rlabel_rot = kws.pop('rlabel_rot', 45)
    rlabel_pad = kws.pop('rlabel_pad', -0.03)
    
    # Make the plot.
    plt.close()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(colws[1]/2, colws[1]/2), 
                               dpi=dpi, subplot_kw={'polar': True})
        ax = np.ravel(ax)

    iax = 0
    ax[iax].grid(alpha=1, lw=grid_lw, linestyle=grid_linestyle, color=grid_color)
    ax[iax].bar(head_direc['hd_angle'], head_direc['fr'], 
                width=bar_width, bottom=0, lw=bar_lw, color=bar_color, ec=bar_ec, zorder=0)
    xticks = [0, (2*np.pi)/4, (4*np.pi)/4, (6*np.pi)/4]
    ax[iax].set_xticks(xticks)
    ax[iax].set_xticklabels(['E', 'N', 'W', 'S'], fontsize=font['label'])
    if show_center_rlabel:
        yticks = np.linspace(vmin, vmax, num=4)
    else:
        yticks = np.linspace(vmin, vmax, num=4)[1:]
    ax[iax].set_ylim([vmin, vmax])
    ax[iax].set_yticks(yticks)
    ax[iax].set_yticklabels(['{:.1f}'.format(x) for x in yticks], 
                            fontsize=font['tick'], color=grid_color)
    ax[iax].tick_params(axis='x', pad=-5)
    ax[iax].set_rlabel_position(rlabel_rot)
    ax[iax].spines['polar'].set_linewidth(0.4)
    if not show_center_rlabel:
        update_rpads(ax[iax], rpad=rlabel_pad)
    
    return fig, ax


def gold_view_raster(subj_sess,
                     neuron,
                     game_state,
                     time_win=3001,
                     proj_dir='/home1/dscho/projects/time_cells',
                     plot_vlines=True,
                     plot_labels=True,
                     plot_title=False,
                     **kws):
    """Show spikes before and after gold view events."""
    # Get input parameters.
    colws = kws.pop('colws', od([('5', 1.19)]))
    figsize = kws.pop('figsize', (colws['5'], colws['5']))
    dpi = kws.pop('dpi', 1200)
    font = kws.pop('font', {'tick': 6, 'label': 7, 'annot': 7, 'fig': 7})
    labelpad = kws.pop('labelpad', 2)
    xtick_step = kws.pop('xtick_step', 750)
    rastersize = kws.pop('rastersize', 1)
    rasterwidth = kws.pop('rasterwidth', 0.1)
    ax_linewidth = kws.pop('ax_linewidth', 0.5)
    vline_color = kws.pop('vline_color', '#bebebe')
    vline_alpha = kws.pop('vline_alpha', 0.15)
    
    # Load data.
    giv_uncertainty = 250 # ~ms between consecutive gold_in_view calls
    half_win = int(time_win/2)
    events = events_proc.load_events(subj_sess, proj_dir=proj_dir, verbose=False)
    spikes = spike_preproc.load_spikes(subj_sess, neuron, proj_dir=proj_dir)
    
    # Identify gold-in-view on and off times for each gold ID. 
    gold_views = []
    for trial in events.keep_trials:
        # Find game state start and stop times.
        start, stop = (events.event_times.query("(trial=={}) & (gameState=='{}')"
                                                .format(trial, game_state))['time_bins']
                                         .apply(lambda x: (x[0], x[-1])).iloc[0])

        # Iterate over gold_in_view entries for each gold ID.
        qry = "(trial=={}) & (gameState=='{}')".format(trial, game_state)
        gold_ids = np.unique(events.gold_events.query(qry)['gold_id'])
        for gold_id in gold_ids:
            _gold_events = events.gold_events.query("(trial=={}) & (gameState=='{}') & (gold_id=='{}')"
                                                    .format(trial, game_state, gold_id))
            view_state = -1
            giv_on = []
            giv_off = []
            for idx, row in _gold_events.iterrows():
                if (row['in_view'] is True) & (view_state == -1):
                    giv_on.append(row['time'])
                    view_state *= -1
                elif (row['in_view'] is False) & (view_state == 1):
                    giv_off.append(row['time'])
                    view_state *= -1

            # Add game state stop time to the last giv_off time 
            # if it wasn't already recorded.
            if len(giv_off) == len(giv_on) - 1:
                giv_off.append(stop)
            assert len(giv_on) == len(giv_off)

            # Append gold view events to the output dataframe.
            for _i in range(len(giv_on)):
                gold_views.append([trial, gold_id, giv_on[_i], giv_off[_i]])

    # Create the gold_views dataframe.
    col_names = ['trial', 'gold_id', 'view_on', 'view_off']
    gold_views = pd.DataFrame(gold_views, columns=col_names)
    gold_views['duration'] = gold_views['view_off'] - gold_views['view_on']
    gold_views = gold_views.sort_values('view_on').reset_index(drop=True)

    # Log refractory time before consecutive gold view events.
    pre_refrac = []
    for trial in gold_views['trial'].unique():
        # Find game state start and stop times.
        start, stop = (events.event_times.query("(trial=={}) & (gameState=='{}')"
                                                .format(trial, game_state))['time_bins']
                                         .apply(lambda x: (x[0], x[-1])).iloc[0])

        _gold_views = gold_views.query("(trial=={})".format(trial))
        prev_time = start
        for idx, row in _gold_views.iterrows():
            pre_refrac.append(row['view_on'] - prev_time)
            prev_time = row['view_off']
    gold_views['pre_refrac'] = pre_refrac

    # Log gold view events that begin >1500ms after the start of the current
    # game state or the end of the last gold view event.
    giv_times = gold_views.query("(pre_refrac>{})".format(half_win))['view_on'].values
    n_gold_views = len(giv_times)

    # Find spikes within 1500ms of each gold-in-view start times.
    # Center spike times around each gold-in-view start time.
    giv_spikes = []
    for giv_time in giv_times:
        giv_win = (giv_time - half_win, giv_time + half_win)
        spike_times = [(x - giv_win[0]) for x in spikes['spike_times']
                       if (x>=giv_win[0]) & (x<=giv_win[1])]
        giv_spikes.append(spike_times)
        
    # Make the raster plot.
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # For each gold view event, convert spike times to a spike train.
    spike_train = np.zeros([n_gold_views, time_win]) * np.nan
    for iView, spike_times in enumerate(giv_spikes):
        spike_train = np.zeros(time_win) * np.nan
        spike_train[spike_times] = iView + 1
        ax.plot(spike_train, linewidth=0, marker='|', 
                markerfacecolor='k', markeredgecolor='k',
                markersize=rastersize, markeredgewidth=rasterwidth)

    # Plot uncertainty lines.
    if plot_vlines:
        vlines = (half_win - giv_uncertainty, half_win + giv_uncertainty)
        ax.fill_between(vlines, ax.get_ylim()[0], ax.get_ylim()[1], 
                        facecolor=vline_color, alpha=vline_alpha, lw=0, zorder=0)
        ax.axvline(half_win, ax.get_ylim()[0], ax.get_ylim()[1],
                   color=vline_color, alpha=1, linestyle='solid',
                   linewidth=rasterwidth*2, zorder=1)
    for axis in ['left', 'bottom']:
        ax.spines[axis].set_linewidth(ax_linewidth)
    ax.tick_params(axis='both', which='both', length=2, width=ax_linewidth, pad=1)

    xticks = np.arange(0, time_win, xtick_step)
    xticklabs = np.arange(-half_win, half_win+1, xtick_step)
    ax.set_xlim([-100, time_win+100])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabs, fontsize=font['tick'], rotation=0)
    yticks = np.linspace(0, n_gold_views, 6, dtype=int)[1:]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=font['tick'], rotation=0)
    ax.set_ylim((0, n_gold_views+1))
    ax.invert_yaxis()
    if plot_labels:
        ax.set_xlabel('Time (ms)', fontsize=font['label'], labelpad=labelpad)
        ax.set_ylabel('Gold view 1..{}'.format(n_gold_views), 
                      fontsize=font['label'], labelpad=labelpad)
    if plot_title:
        ax_title = '{}-{} ({}), {}'.format(subj_sess, neuron, spikes['hemroi'], game_state)
        ax.set_title(ax_title, loc='left', pad=6, fontsize=font['fig'])
        
    return fig, ax
