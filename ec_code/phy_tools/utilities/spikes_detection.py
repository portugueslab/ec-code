import numpy as np
from numba import jit
import pandas as pd

# #@jit(nopython=True)
# def find_spikes(input_arr, thr, pre_int=10, post_int=10,
#                 crop=False, min_dist=None, invert=True):
#     """
#     Find spikes in an array/ pandas series
#     :param input_arr:
#     :param thr:
#     :param pre_int:
#     :param post_int:
#     :param crop:
#     :return:
#     """
#     if isinstance(input_arr, pd.Series):
#         arr = np.array(input_arr)
#     else:
#         arr = input_arr
#
#     input_arr = input_arr - np.nanmean(input_arr)
#
#     arr[np.isnan(arr)] = 0
#     t = arr > thr
#     t[1:][t[:-1] & t[1:]] = False
#     spikes = np.where(t)[0]
#
#     peak_pos = np.zeros(spikes.shape, dtype=int)
#
#     for i, s in enumerate(spikes):
#         if thr>0:
#             peak_pos[i] = np.int(s - pre_int + np.argmax(arr[np.max([0, s - pre_int]):np.min([len(arr), s + post_int])]))
#         else:
#             peak_pos[i] = np.int(s - pre_int + np.argmin(arr[np.max([0, s - pre_int]):np.min([len(arr), s + post_int])]))
#     peak_pos = np.unique(peak_pos)
#     if min_dist:
#         peak_pos = peak_pos[np.concatenate([np.diff(peak_pos) > (pre_int+post_int),  [True]])]
#
#     if crop:
#         spike_mat = crop_trace(arr, peak_pos, pre_int=pre_int, post_int=post_int)
#         return peak_pos, spike_mat
#     else:
#         return peak_pos


#@jit(nopython=True)
def find_spikes(trace, thr_coef, pre_int=10, post_int=10,
                crop=False, min_dist=0, invert=False):
    """
    Find spikes in an array/ pandas series
    :param input_arr:
    :param thr:
    :param pre_int:
    :param post_int:
    :param crop:
    :return:
    """
    if isinstance(trace, pd.Series):
        trace = trace.as_matrix()

    peak_pos = _find_spikes(trace*((not invert)*2-1), thr_coef, pre_int=pre_int,
                            post_int=post_int, min_dist=min_dist)
    peak_pos = peak_pos[np.concatenate([[1], np.diff(peak_pos)]) != 0]

    if crop:
        spike_mat = _crop_trace(trace, peak_pos, pre_int=pre_int, post_int=post_int)
        return peak_pos, spike_mat
    else:
        return peak_pos


@jit(nopython=True)
def _find_spikes(trace, thr, pre_int=80, post_int=80, min_dist=0):
    """ Optimized code for finding in a trace all events higher then a
    threshold. Looks for crossings of the threshold and then find around the
    crossing point a peak in a window defined by the arguments
    :param trace: numpy array
    :param thr: threshold, same units as the trace
    :param pre_int: pre-interval of the window for finding the peak
    :param post_int: post-interval of the window for finding the peak
    :param min_dist: distance to jump after an event is found
    :return: numpy arrray with spikes indexes

    """

    if min_dist == 0:
        min_dist = post_int  # the minimum distance should be at least this
    times = np.zeros(trace.shape, dtype=np.int32)

    k = 0
    i = 0
    while i < len(trace):
        if trace[i] > thr:
            # Find the peak within a segment:
            segment = trace[max(0, i - pre_int):min(len(trace), i + post_int)]
            seg_peak = np.argmax(segment)
            times[k] = i - pre_int + seg_peak
            i += min_dist
            k += 1
        i += 1
    return times[times > 0]


@jit(nopython=True)
def _crop_trace(trace, events, pre_int=20, post_int=30, dwn=1):
    """ Crop the trace around specified events in a window given by parameters.
    :param trace: trace to be cropped
    :param events: events around which to crop
    :param pre_int: interval to crop before the event, in points
    :param post_int: interval to crop after the event, in points
    :param dwn: downsampling (default=1 i.e. no downsampling)
    :return: events x n_points numpy array
    """

    # Avoid problems with spikes at the borders:
    valid_events = events[(events > pre_int) & (events < len(trace) - post_int)]

    mat = np.empty((int((pre_int + post_int) / dwn), valid_events.shape[0]))

    for i, s in enumerate(valid_events):
        cropped = trace[s - pre_int:s + post_int:dwn].copy()
        mat[:len(cropped), i] = cropped

    return mat


@jit(nopython=True)
def raster_on_evts(spikes, events, fn=8333.3, pre_int=1, post_int=2):
    raster_list = [(0, 0)]

    for i, e in enumerate(events):
        spikes_sub = spikes - e
        spikes_sub = spikes_sub[spikes_sub < post_int * fn]
        spikes_sub = spikes_sub[spikes_sub > -pre_int * fn]

        for s in spikes_sub:
            raster_list.append((s, i))

    return np.array(raster_list)
#
#
# def crop_trace(trace, events, pre_int=20, post_int=30,
#                rebase=None, mean_only=False, dwn=1):
#     trace = np.array(trace)
#     mat = np.ones((events.shape[0], int((pre_int + post_int)/dwn)))*np.nan
#     # TODO handle border cases
#     # TODO buggy behaviour for rebasing, and now it has to be None
#     for i, s in enumerate(events):
#         if s > pre_int and s < len(trace) - post_int:
#             cropped = trace[s - pre_int:s + post_int:dwn].copy()
#             if rebase is not None:
#                 if isinstance(rebase, np.ndarray):
#                     cropped -= np.nanmean(cropped[rebase])
#                 else:
#                     cropped -= np.nanmean(cropped)
#             mat[i, :len(cropped)] = cropped
#
#     if mean_only:
#         return np.nanmean(mat, 0)
#     else:
#         return mat