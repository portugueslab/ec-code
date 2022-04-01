import numpy as np
import pandas as pd
import flammkuchen as fl
from bouter.utilities import crop

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


def crop_trace(trace, timepoints, dt, pre_int_s, post_int_s, normalize=False):
    """Crop a trace given timepoints and crop interval in seconds and sampling dt.
    """
    start_idxs = np.round(timepoints / dt).astype(np.int)
    cropped = crop(
        trace, start_idxs, pre_int=int(pre_int_s / dt), post_int=int(post_int_s / dt)
    )
    if normalize:
        cropped = cropped - np.nanmean(cropped[: int(pre_int_s / dt), :], 0)

    return cropped


def crop_f_beh_at_times(
    cid,
    timepoints,
    pre_int_s,
    post_int_s,
    cells_df,
    traces_df,
    exp_df,
    normalize=True,
    beh_trace=None,
):
    """Crop fluorescence trace and behavio trace at given times, and normalize if required.
    """
    fid = cells_df.loc[cid, "fid"]
    cropped = crop_trace(
        traces_df[cid].values,
        timepoints,
        exp_df.loc[fid, "dt"],
        pre_int_s,
        post_int_s,
        normalize=normalize,
    )

    # Crop behavior:
    if beh_trace is not None:
        # beh_trace = fl.load(master_path / "resamp_beh_dict.h5", f"/{fid}")
        dt_beh = np.diff(beh_trace.index[:5]).mean()
        cropped_be = crop_trace(
            beh_trace["vigor"].values,
            timepoints,
            dt_beh,
            pre_int_s,
            post_int_s,
            normalize=False,
        )
        return cropped, cropped_be

    return cropped

