import numpy as np
import pandas as pd

cols = np.array([[72, 177, 167],
                 [200, 87, 123],
                 [182, 94, 189],
                 [103, 166, 78],
                 [111, 124, 203],
                 [180, 148, 64],
                 [202, 96, 63]]) / 255


def bout_nan_traces(traces, idxs, wnd_pre=2, wnd_post=5):
    """ Set to nan the trace values around (bouts) indexes
    """
    out_traces = traces.copy()
    for idx in idxs:
        out_traces[idx - wnd_pre:idx + wnd_post, :] = np.nan

    return out_traces


def max_amplitude_resp(traces, percentile=80):
    """Calculate max response of a matrix of cropped responses
    using timepoints with the highest absolute deviation from zero.
    This is highly subjected to noise and should be used in combination with
    reliability-based filtering.
    Also, it should probably be rewritten to take continuity into consideration.

    :param: traces: 2D (timepoints x n_cells/responses) array
    :param: percentile: percentile to calculate the time window for the max deviation
    """
    # Find timepoints that are not in the maximum deviation window:
    resp_amplitude_pts = np.abs(traces) < np.nanpercentile(np.abs(traces), percentile,
                                                           axis=0)

    # Set those points to nan, and calculate mean of the rest:
    resp_amplitude_pts[resp_amplitude_pts] = np.nan
    return np.nanmean(traces * resp_amplitude_pts, 0)


