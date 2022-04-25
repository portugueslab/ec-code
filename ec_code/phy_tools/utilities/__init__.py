import numpy as np
import sys
import os
import pandas as pd
import flammkuchen as fl
from scipy.stats import zscore
from scipy.signal import detrend

from numba import jit
from ec_code.phy_tools.utilities.spikes_detection import *


import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import detrend


def butter_highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y



def nanzscore(vect):
    return (vect - np.nanmean(vect)) / np.nanstd(vect)

def nanzscoremedian(vect):
    return (vect - np.nanmedian(vect)) / (np.abs(np.nanpercentile(vect, 1) -
                                                 np.nanpercentile(vect, 99)))

# Old version  
def get_bouts(b, thr=0):
    bouts = np.array(b > thr).astype(int)
    ons = np.where(np.diff(bouts) > 0)[0]
    offs = np.where(np.diff(bouts) < 0)[0]

    space_lim = 160
    bout_len_lim = 300
    for i in np.where((ons[1:] - offs[:-1]) < space_lim)[0]:
        bouts[offs[i]:ons[i + 1] + 1] = 1

    ons = np.where(np.diff(bouts) > 0)[0]
    offs = np.where(np.diff(bouts) < 0)[0]

    for i in np.where(offs - ons < bout_len_lim)[0]:
        bouts[ons[i]:offs[i] + 1] = 0
        
    ons = np.where(np.diff(bouts) > 0)[0]
    offs = np.where(np.diff(bouts) < 0)[0]

    return ons, offs, bouts


def norm_detrend(pd, exclude=[], wnd=1600):
    all_arr = np.array([np.newaxis])
    for i in range(pd.sweep.max()):
        arr = np.array(pd[pd.sweep==i].I_VR.rolling(wnd, center=True).std())

        if i not in exclude:
            arr[~np.isnan(arr)] = nanzscore(detrend(arr[~np.isnan(arr)]))
            arr -= np.nanmedian(arr)
        else:
            arr = np.ones(len(arr))*np.nan
        all_arr = np.concatenate([all_arr, arr]).astype('float')
    return all_arr[1:]


def shuffle(serie):
    diffs = np.diff(serie)
    np.random.shuffle(diffs)
    return np.cumsum(diffs)


def autocorrelation(spike_times, spike_times2=None, bin_width=1e-3,
                    width=1e-1, T=None):
    """Stolen from somewhere else.
    Given the sorted spike train 'spike_times' return the
    autocorrelation histogram, as well as the bin edges (including the
    rightmost one). The bin size is specified by 'bin_width', and lags are
    required to fall within the interval [-width, width]. The algorithm is
    partly inspired on the Brian function with the same name."""

    if spike_times2 is None:
        spike_times2 = spike_times
    d = []                    # Distance between any two spike times
    n_sp = np.alen(spike_times2)  # Number of spikes in the input spike train

    i, j = 0, 0
    for t in spike_times:
        # For each spike we only consider those spikes times that are at most
        # at a 'width' time lag. This requires finding the indices
        # associated with the limiting spikes.
        while i < n_sp and spike_times2[i] < t - width:
            i += 1
        while j < n_sp and spike_times2[j] < t + width:
            j += 1
        # Once the relevant spikes are found, add the time differences
        # to the list
        d.extend(spike_times2[i:j] - t)


    n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
    # Define the edges of the bins (including rightmost bin)
    b = np.linspace(-width, width, 2 * n_b, endpoint=True)
    h = np.histogram(d, bins=b)
    H = h[0] # number of entries per bin

    # Compute the total duration, if it was not given
    # (spike trains are assumed to be sorted sequences)
    if T is None:
        T = spike_times[-1] - spike_times[0] # True for T >> 1/r

    # The sample space gets smaller as spikes are closer to the boundaries.
    # We have to take into account this effect.
    W = T - bin_width * abs( np.arange(n_b - 1, -n_b, -1) )

    auto = H/W - n_sp**2 * bin_width / (T**2)
    auto[np.argmax(auto)] = np.nan
    return (auto , b)


def bouts_from_twitches(twitches, fn=8333, min_dist_s=0.02, max_int_s=0.15,
                        sort=False):
    """ Function to put together successive tail twitches into bouts.
    :param twitches: indexes of twitches
    :param fn: sampling rate
    :param min_dist: minimum distance between twitches (in seconds)
    :param max_int: maximum distance between twitches (in seconds)
    :return:
    """
    twitches = twitches[np.concatenate([[False], np.diff(twitches) > min_dist_s * fn])]
    
    dist_vect = np.concatenate([[False], np.diff(twitches) > max_int_s * fn])
    ons_idx = np.where(dist_vect)[0]
    offs_idx = np.concatenate([ons_idx[1:] - 1,
                               [len(twitches) - 1]])
    ons = twitches[ons_idx]
    offs = twitches[offs_idx]
    
    sel = ons - offs != 0
    ons = ons[sel]
    offs = offs[sel]

    if sort:
        sort_lengths = np.argsort(offs - ons)
        offs = offs[sort_lengths]
        ons = ons[sort_lengths]
    
    return ons, offs


def get_from_folder(input_folder, sel=None):
    selected_files = []
    for (dirpath, dirnames, filenames) in os.walk(input_folder):
        for f in filenames:
            if sel:
                if sel in f:
                    selected_files.append(dirpath + '/' + f)
            else:
                selected_files.append(dirpath + '/' + f)
    return selected_files
    
    
def bin_bouts_from_onsets(ons, offs, length):
    if length:
        bin_vect = np.zeros(length).astype(int)
    else:
        bin_vect = np.zeros(np.max(offs)+1)
    
    for on, off in zip(ons, offs):
        bin_vect[on:off] = 1
    
    return bin_vect
    
    
def bin_from_spk(spk, length=None, smooth=None, fn=8333.33):
    if length:
        bin_vect = np.zeros(length)
    else:
        bin_vect = np.zeros(np.max(spk)+1)
    bin_vect[spk] = 1
    if smooth:
        bin_vect = np.array(pd.Series(bin_vect).rolling(smooth).sum())/(smooth/fn)
    
    return bin_vect 


def df_from_h5(filename):
    data = fl.io.load(str(filename))
    df = data['data_dict']['traces']
    df = df.reset_index()
    df['spikes'] = 0
    df.loc[data['data_dict']['spikes'], 'spikes'] = 1
    df['twitches'] = 0
    df.loc[data['data_dict']['twitches'], 'twitches'] = 1
    #df['bouts'] = bin_bouts_from_onsets(*bouts_from_twitches(data['data_dict']['contr']), length=len(df))
    
    #df['time'] = np.round(df['time']*10000)/10000  # This should be done
    # more elegantly, with xarray?
    
    return df

    

         