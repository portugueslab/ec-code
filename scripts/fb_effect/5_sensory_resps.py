"""Compute statistics for the cell responsiveness to visual stimuli.
"""

import flammkuchen as fl
import numpy as np
from bouter import utilities
from tqdm import tqdm

from ec_code.analysis_utils import bout_nan_traces, max_amplitude_resp
from ec_code.fb_effect.default_vals import DEF_DT
from ec_code.file_utils import get_dataset_location

master_path = get_dataset_location("fb_effect")

exp_df = fl.load(master_path / "exp_df.h5")
trials_df = fl.load(master_path / "trials_df.h5")
cells_df = fl.load(master_path / "cells_df.h5")
trials_df = fl.load(master_path / "trials_df.h5")
traces_df = fl.load(master_path / "traces_df.h5")
bouts_df = fl.load(master_path / "bouts_df.h5")


# Add columns if necessary:
for new_col in ["forward_rel", "backward_rel", "forward_amp", "backward_amp"]:
    if new_col not in cells_df.columns:
        cells_df[new_col] = np.nan

crop_params_dict = dict(
    forward=dict(pre_int_s=1, post_int_s=9), backward=dict(pre_int_s=1, post_int_s=5)
)

WND_PRE_BOUT_NAN_S = 0.4
WND_POST_BOUT_NAN_S = 6

# percentile of response to be averaged to get amplitude (not to define a window):
AMPLITUDE_PERC = 80

pre_wnd_bout_nan = int(WND_PRE_BOUT_NAN_S / DEF_DT)
post_wnd_bout_nan = int(WND_POST_BOUT_NAN_S / DEF_DT)

for stim in ["forward", "backward"]:
    pre_int_s = crop_params_dict[stim]["pre_int_s"]
    post_int_s = crop_params_dict[stim]["post_int_s"]

    for fid in tqdm(exp_df.index):
        cells_fsel = cells_df["fid"] == fid  # .copy()

        # Traces matrix, and bout-nanned version:
        bout_start_idxs = np.round(
            bouts_df.loc[bouts_df["fid"] == fid, "t_start"] / DEF_DT
        ).values.astype(np.int)

        traces = traces_df.loc[:, cells_df[cells_fsel].index].copy()
        traces_mat = traces.values
        traces_mat_nanbouts = bout_nan_traces(
            traces_mat,
            bout_start_idxs,
            wnd_pre=pre_wnd_bout_nan,
            wnd_post=post_wnd_bout_nan,
        )
        # trials:
        ftrials_df = trials_df.loc[trials_df["fid"] == fid, :]
        # Exclude trials with a leading bout too close to stimulus onset:
        # ftrials_df = ftrials_df.loc[np.isnan(ftrials_df["lead_bout_latency"]), :]

        start_idxs = np.round(
            ftrials_df.loc[ftrials_df["trial_type"] == stim, "t_start"] / DEF_DT
        )
        bout_lat_sort = np.argsort(
            ftrials_df.loc[ftrials_df["trial_type"] == stim, "bout_latency"].values
        )

        cropped_nan = utilities.crop(
            traces_mat_nanbouts,
            start_idxs,
            pre_int=int(pre_int_s / DEF_DT),
            post_int=int(post_int_s / DEF_DT),
        )
        cropped_nan = cropped_nan[:, bout_lat_sort, :]
        cropped_nan = cropped_nan - np.nanmean(
            cropped_nan[: int(pre_int_s / DEF_DT), :, :], 0
        )

        # cells_df.loc[cells_df["fid"]==fid, "motor_rel"] = utilities.reliability(cropped)
        reliabilities = utilities.reliability(cropped_nan)

        # Calculate mean response for all cells:
        mean_resps = np.nanmean(cropped_nan, 1)

        # Calculate amplitude of the response looking at top 20% percentile of the response
        # (response is normalized at pre-stim onset)

        amplitudes = max_amplitude_resp(mean_resps, percentile=AMPLITUDE_PERC)

        cells_df.loc[cells_fsel, f"{stim}_rel"] = reliabilities
        cells_df.loc[cells_fsel, f"{stim}_amp"] = amplitudes

fl.save(master_path / "cells_df.h5", cells_df)
