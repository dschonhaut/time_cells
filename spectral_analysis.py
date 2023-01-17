"""
spectral_analysis.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Spectral decomposition methods.

Last Edited
----------- 
2/3/22
"""
import sys
import os
import os.path as op
from glob import glob
import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
import xarray
import mne
from scipy import stats
from scipy import signal
from fooof import FOOOF, FOOOFGroup
from fooof.objs.utils import combine_fooofs
sys.path.append('/home1/dscho/code/general')
import data_io as dio
from helper_funcs import Timer
sys.path.append('/home1/dscho/code/projects')
from time_cells import spike_preproc


def timefreq_wavelet(eeg,
                     freqs,
                     sr=None,
                     buffer=None,
                     clip_buffer=True,
                     n_cycles=5,
                     zero_mean=True,
                     log_power=False,
                     subj_sess=None,
                     roi=None,
                     output='both',
                     output_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/spectral',
                     save_output=False,
                     overwrite=False,
                     verbose=False):
    """Use wavelets to estimate power and phase at each timepoint.

    Parameters
    ----------
    eeg : array, shape=(event, channel, time)
        EEG in the time domain. Wavelet convolution is done
        over the last dimension.
    sr : float
        The sampling rate, in Hz.
    buffer : int | None
        The number of buffer samples to be clipped from
        either side of the time domain after wavelet convolution.
    freqs : array
        The frequencies of interest, in Hz.
    n_cycles : int
        The wavelet length, in cycles at a given frequency.
    zero_mean : bool
        If True, ensures that the wavelets have a mean of zero.
    output : str
        'both' returns power and phase, in that order.
        'power' returns only power
        'phase' returns only phase
    
    Returns
    -------
    power, phase : tuple of arrays, shape=(event, chan, frequency, time)
        > Phase goes from 0 to 2π.
        > Buffer is clipped out of the time domain.
        > If input EEG is a DataArray, then DataArrays are returned.
          Otherwise numpy arrays are returned.
    """
    assert output in ('both', 'power', 'phase')
    is_xarray = False
    if isinstance(eeg, xarray.core.dataarray.DataArray):
        is_xarray = True
        
    # Get metadata parameters that weren't explicitly passed.
    if is_xarray:
        if subj_sess is None:
            subj_sess = eeg.name[0]
        if roi is None:
            roi = eeg.name[1]
        if sr is None:
            sr = eeg.sr
        if buffer is None:
            if 'buffer' in eeg.attrs:
                buffer = eeg.buffer
            else:
                buffer = 0
            
    # Get the output filenames.
    basename = '{}-{}.pkl'.format(subj_sess, roi)
    power_f = op.join(output_dir, 'power', basename)
    phase_f = op.join(output_dir, 'phase', basename)

    # Return the processed data if it already exists.
    if output == 'both':
        if np.all((op.exists(power_f), op.exists(phase_f))) and not overwrite:
            power = dio.open_pickle(power_f)
            phase = dio.open_pickle(phase_f)
            return power, phase
    elif output == 'power':
        if op.exists(power_f) and not overwrite:
            power = dio.open_pickle(power_f)
            return power
    elif output == 'phase':
        if op.exists(phase_f) and not overwrite:
            phase = dio.open_pickle(phase_f)
            return phase

    # Run wavelet convolution.
    tfr = mne.time_frequency.tfr_array_morlet(eeg,
                                              sr,
                                              freqs,
                                              n_cycles=n_cycles,
                                              zero_mean=zero_mean,
                                              output='complex')
    
    # Clip out the buffer.
    if clip_buffer:
        tfr = tfr[..., buffer:eeg.shape[-1]-buffer]
    
    # Get power and phase from the complex values.
    power = (tfr * tfr.conj()).real
    if log_power:
        power = np.log10(power)
    phase = (np.angle(tfr) + np.pi) % (2 * np.pi) # 0 to 2π
    power = power.astype(np.float32)
    phase = phase.astype(np.float32)
    
    # Format the outputs as DataArrays if the inputs were DataArrays.
    if is_xarray:
        dims = list(eeg.dims[:-1]) + ['freq', 'time']
        coords = {x: eeg.coords[x] for x in eeg.dims[:-1]}
        coords['freq'] = freqs
        coords['time'] = np.arange(power.shape[-1])
        attrs = {'sr': sr,
                 'buffer': buffer,
                 'clip_buffer': clip_buffer,
                 'n_cycles': n_cycles,
                 'zero_mean': zero_mean,
                 'log_power': log_power}
        power = xarray.DataArray(power,
                                 name=eeg.name,
                                 dims=dims,
                                 coords=coords,
                                 attrs=attrs.copy())
        del attrs['log_power']
        phase = xarray.DataArray(phase,
                                 name=eeg.name,
                                 dims=dims,
                                 coords=coords,
                                 attrs=attrs)

    # Save the data.
    if save_output:
        if output == 'both':
            dio.save_pickle(power, power_f, verbose)
            dio.save_pickle(phase, phase_f, verbose)
        elif output == 'power':
            dio.save_pickle(power, power_f, verbose)
        elif output == 'phase':
            dio.save_pickle(phase, phase_f, verbose)
        
    # Return the data.
    if output == 'both':
        return power, phase
    elif output == 'power':
        return power
    elif output == 'phase':
        return phase


def timefreq_welch(eeg,
                   sr,
                   fmin=1,
                   fmax=80,
                   n_fft=None,
                   n_overlap=None,
                   log_transform=True,
                   verbose=False):
    """Uses Welch's method to convert signal from time to frequency domain.
    
    Parameters
    ----------
    eeg : array, shape=(..., n_times)
        Decomposition is done over the last dimension.
    sr : float
        The sampling rate, in Hz.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    n_fft : int
        The length of FFT used.
    n_overlap : int
        The number of overlapping points between segments.
    log_transform: bool
        If true, power values are log10-transformed; otherwise
        the raw powers are returned.
    Returns
    -------
    freqs : array
        frequency vector.
    powers : array
        channel x frequency power values.
    """
    if n_fft is None:
        n_fft = int(sr)
    if n_overlap is None:
        n_overlap = int(n_fft / 8)
    
    powers, freqs = mne.time_frequency.psd_array_welch(eeg,
                                                       sr,
                                                       fmin=fmin,
                                                       fmax=fmax,
                                                       n_fft=n_fft,
                                                       n_overlap=n_overlap,
                                                       verbose=verbose)
    if log_transform:
        powers = np.log10(powers)
        
    return freqs, powers


def run_fooof(subj_sess,
              roi,
              power=None,
              freqs=np.arange(1, 31),
              peak_width_limits=(1, 8),
              min_peak_height=0.2,
              max_n_peaks=4,
              peak_threshold=2,
              aperiodic_mode='fixed',
              data_dir=None,
              output_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/fooof',
              save_output=False,
              overwrite=False,
              verbose=False):
    """Use FOOOF to fit the aperiodic power spectrum and peaks."""
    # Load and return the output file, if it exists.
    basename = '{}-{}.json'.format(subj_sess, roi)
    output_f = op.join(output_dir, basename)
    if op.exists(output_f) and not overwrite:
        fg = FOOOFGroup()
        fg.load(basename, output_dir)
        return fg
    
    # Load inputs.
    if power is None:
        if data_dir is None:
            data_dir = op.abspath(op.join(output_dir, os.pardir))
        power = dio.open_pickle(op.join(data_dir, 'spectral', 'power',
                                        basename.replace('.json', '.pkl')))
        
    # Get the average power at each frequency over time, for each channel and event.
    if power.clip_buffer:
        mean_power = power.mean(dim='time') # event x chan x freq
    else:
        mean_power = power[:, :, :, power.buffer:power.time.size-power.buffer].mean(dim='time')
    
    # Run FOOOF.
    foofs = []
    for iChan in range(len(mean_power.chan)):
        for iEvent in range(len(mean_power.event)):
            fm = FOOOF(peak_width_limits=peak_width_limits,
                       max_n_peaks=max_n_peaks,
                       min_peak_height=min_peak_height,
                       peak_threshold=peak_threshold,
                       aperiodic_mode=aperiodic_mode,
                       verbose=verbose)
            fm.fit(freqs, mean_power.values[iEvent, iChan, :freqs.size])
            foofs.append(fm)
    fg = combine_fooofs(foofs)
    
    # Save output.
    if save_output:
        fg.save(basename, output_dir, save_results=True, save_settings=True, save_data=True)
    
    return fg


def run_p_episode(subj_sess,
                  roi,
                  power=None,
                  phase=None,
                  fooof_group=None,
                  freqs=np.arange(1, 31),
                  cycle_thresh=3,
                  thresh_req='all',
                  chi2_pctl=0.95,
                  # buffer=None,
                  return_ap_thresh=False,
                  data_dir=None,
                  output_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/p_episode',
                  save_output=True,
                  overwrite=False,
                  verbose=True):
    """Run P-episode and return the oscillation mask."""
    assert thresh_req in ('mean', 'all')
    
    # Load the output file if it exists.
    basename = '{}-{}.pkl'.format(subj_sess, roi)
    output_f = op.join(output_dir, basename)
    if op.exists(output_f) and not overwrite:
        return dio.open_pickle(output_f)
    
    # Load inputs.
    if data_dir is None:
        data_dir = op.abspath(op.join(output_dir, os.pardir))
    if power is None:
        power = dio.open_pickle(op.join(data_dir, 'spectral', 'power', basename))
    if phase is None:
        phase = dio.open_pickle(op.join(data_dir, 'spectral', 'phase', basename))
    if fooof_group is None:
        fooof_group = FOOOFGroup()
        fooof_group.load(basename.replace('.pkl', '.json'),
                         op.join(data_dir, 'fooof'))

    # if buffer is None:
    #     try:
    #         if power.clip_buffer:
    #             buffer = 0
    #         else:
    #             buffer = power.buffer
    #     except AttributeError:
    #         buffer = 0
    # mean_power = power[:, :, :, buffer:len(power.time)-buffer].mean(dim='time')
    
    # Get aperiodic line fits for each channel and event.
    ii = 0
    ap_fits = []
    for iChan in range(len(power.chan)):
        ap_fits.append([])
        for iEvent in range(len(power.event)):
            ap_fits[iChan].append(10**fooof_group.get_fooof(ii)._ap_fit)
            ii += 1
    ap_fits = np.array(ap_fits) # chan x event x freq

    # Find the 95th percentile of the chi2 distribution centered
    # on the aperiodic fit for each channel, event, and frequency.
    dof = 2
    chi2_thresh = stats.chi2.ppf(chi2_pctl, dof)
    ap_thresh = np.array([[[chi2_thresh * (ap_fits[iChan, iEvent, iFreq]/2)
                            for iFreq in range(ap_fits.shape[2])]
                           for iEvent in range(ap_fits.shape[1])]
                          for iChan in range(ap_fits.shape[0])])
    # ap_thresh = np.array([[[stats.chi2.ppf(chi2_pctl, dof, loc=ap_fits[iChan, iEvent, iFreq])
    #                         for iFreq in range(ap_fits.shape[2])]
    #                        for iEvent in range(ap_fits.shape[1])]
    #                       for iChan in range(ap_fits.shape[0])])

    # Match ap_thresh and power dimensions.
    ap_thresh = np.swapaxes(ap_thresh, 0, 1) # event x chan x freq

    # Restrict P-episode analysis to frequencies of interest.
    power = power.sel(freq=power.freq<=freqs.max())
    phase = phase.sel(freq=phase.freq<=freqs.max())
    ap_thresh = ap_thresh[:, :, :freqs.size]

    # For each event, channel, and frequency, construct a cycle-by-cycle
    # mask of mean power values above the P-episode threshold.
    pep_vec = np.ones(cycle_thresh)
    osc_win = int((cycle_thresh - 1) / 2)
    osc_mask = np.zeros(power.shape, dtype=bool)
    for iEvent in range(len(power.event)):
        for iChan in range(len(power.chan)):
            for iFreq in range(len(power.freq)):
                trough_idx = np.where(np.abs(np.diff(phase.values[iEvent, iChan, iFreq, :])) > 1)[0]
                _power = np.split(power.values[iEvent, iChan, iFreq, :], trough_idx)
                _cycle_dur = [len(x) for x in _power]
                _thresh = ap_thresh[iEvent, iChan, iFreq]
                if thresh_req == 'mean':
                    _power_above_thresh = np.array([(np.mean(x) > _thresh) for x in _power])
                elif thresh_req == 'all':
                    _power_above_thresh = np.array([np.all(x > _thresh) for x in _power])
                osc_idx = np.where(signal.convolve(_power_above_thresh.astype(float), pep_vec, mode='same') > (cycle_thresh - 0.5))[0]
                osc_idx_full = np.unique(np.concatenate([osc_idx+ii for ii in range(-osc_win, osc_win+1)]))
                _osc_bout = np.zeros(len(_power_above_thresh), dtype=bool)
                _osc_bout[osc_idx_full] = True
                osc_mask[iEvent, iChan, iFreq, :] = np.concatenate([np.repeat(_osc_bout[ii], _cycle_dur[ii])
                                                                    for ii in range(len(_cycle_dur))])
    
    # Format the outputs as DataArrays if the inputs were DataArrays.
    if isinstance(power, xarray.core.dataarray.DataArray):
        osc_mask = power.copy(data=osc_mask)
        osc_mask.attrs.update(cycle_thresh=cycle_thresh,
                              thresh_req=thresh_req,
                              chi2_pctl=chi2_pctl)
    
    # Save outputs.
    if save_output:
        dio.save_pickle(osc_mask, output_f, verbose)
        
    if return_ap_thresh:
        return osc_mask, ap_thresh
    else:
        return osc_mask


def load_p_episode_pct(n_rois=5,
                       data_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2',
                       save_output=True,
                       overwrite=False,
                       verbose=True):
    """Return the mean percent oscillatory time at each frequency, for each session."""
    if verbose:
        timer = Timer()

    # Load outputs if they exist.
    output_file = op.join(data_dir, 'p_episode', 'pep_pct.pkl')
    if op.exists(output_file) and not overwrite:
        pep_pct = dio.open_pickle(output_file)
        return pep_pct
    
    pep_files = glob(op.join(data_dir, 'p_episode', 'U*.pkl'))
    roi_map = spike_preproc.roi_mapping(n=n_rois)

    pep_pct = []
    for fname in pep_files:
        basename = op.basename(fname).replace('.pkl', '')
        subj_sess, roi = basename.split('-')
        subj = subj_sess.split('_')[0]
        osc_mask = dio.open_pickle(fname)
        if (osc_mask.buffer > 0) & (not osc_mask.clip_buffer):
            means = np.mean(osc_mask.values[:, :, :, osc_mask.buffer:osc_mask.time.size-osc_mask.buffer],
                            axis=(0, 1, 3)).tolist()
        else:
            means = np.mean(osc_mask.values, axis=(0, 1, 3)).tolist()
        pep_pct.append(([subj, subj_sess, roi, roi_map[roi[1:]]] + means))
    cols = ['subj', 'subj_sess', 'roi', 'roi_gen'] + osc_mask.freq.values.tolist()
    pep_pct = pd.DataFrame(pep_pct, columns=cols)

    # Save output.
    if save_output:
        dio.save_pickle(pep_pct, output_file, verbose)
    
    if verbose:
        print('pep_pct: {}'.format(pep_pct.shape))
        print(timer)
    
    return pep_pct
