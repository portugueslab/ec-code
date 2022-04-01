"""Compute responses to bouts for all cells, and statistics over their
different responses to the presence of visual reafference.
"""

import warnings

import flammkuchen as fl
from bouter import utilities
from tqdm import tqdm

from ec_code.analysis_utils import (
    bout_nan_traces,
    crop_trace,
    max_amplitude_resp,
)
from ec_code.fb_effect.default_vals import DEF_DT, POST_INT_BT_S, PRE_INT_BT_S
from ec_code.file_utils import get_dataset_location
from ec_code.plotting_utils import *

master_path = get_dataset_location("fb_effect")

exp_df = fl.load(master_path / "exp_df.h5")
trials_df = fl.load(master_path / "trials_df.h5")
cells_df = fl.load(master_path / "cells_df.h5")
bouts_df = fl.load(master_path / "bouts_df.h5")
traces_df = fl.load(master_path / "traces_df.h5")

fid = exp_df.index[6]

# Analysis parameters:
AMPLITUDE_PERC = 90  # percentile for the calculation of the response amplitude

# Widow for nanning out the bout artefacts:
wnd_pre_bout_nan_s = 0.2
wnd_post_bout_nan_s = 0.2

# Different selection criteria to compute different responses to motor:
selections_dict = dict(
    motor=bouts_df["mindist_included"],
    motor_g0=bouts_df["mindist_included"]
    & (bouts_df["base_vel"] < 0)
    & (bouts_df["gain"] == 0),
    motor_g1=bouts_df["mindist_included"]
    & (bouts_df["base_vel"] < 0)
    & (bouts_df["gain"] == 1),
    motor_spont=bouts_df["mindist_included"] & (bouts_df["base_vel"] > -10),
)


for val in ["rel", "amp"]:
    for sel in selections_dict.keys():
        column_id = f"{sel}_{val}"
        if column_id not in cells_df.columns:
            cells_df[column_id] = np.nan


pre_wnd_bout_nan = int(wnd_pre_bout_nan_s / DEF_DT)
post_wnd_bout_nan = int(wnd_post_bout_nan_s / DEF_DT)

# Loop over criteria for the different reliabilities:
for selection in selections_dict.keys():

    # Loop over fish:
    for fid in tqdm(exp_df.index):
        cells_fsel = cells_df.loc[cells_df["fid"] == fid, :]  # .copy()
        traces = traces_df.loc[:, cells_fsel.index].copy()

        # Nan all bouts:
        start_idxs = np.round(
            bouts_df.loc[bouts_df["fid"] == fid, "t_start"] / DEF_DT
        ).astype(np.int)
        traces = bout_nan_traces(
            traces.values,
            start_idxs,
            wnd_pre=pre_wnd_bout_nan,
            wnd_post=post_wnd_bout_nan,
        )

        beh_df = fl.load(master_path / "beh_dict.h5", f"/{fid}")
        stim_df = fl.load(master_path / "stim_dict.h5", f"/{fid}")

        sel_bouts = bouts_df[(bouts_df["fid"] == fid) & selections_dict[selection]]
        sel_start_idxs = np.round(sel_bouts["t_start"] / DEF_DT).astype(np.int)

        # Crop cell responses around bouts:
        cropped = utilities.crop(
            traces,
            sel_start_idxs,
            pre_int=int(PRE_INT_BT_S / DEF_DT),
            post_int=int(POST_INT_BT_S / DEF_DT),
        )

        # Subtract pre-bout baseline:
        cropped = cropped - np.nanmean(cropped[: int(PRE_INT_BT_S / DEF_DT), :, :], 0)

        # Calculate reliability indexes:
        reliabilities = utilities.reliability(cropped)

        # Calculate mean response for all cells:
        mean_resps = np.nanmean(cropped, 1)

        # Calculate amplitude of the response looking at top 20% percentile of the response
        # (response is normalized at pre-stim onset):
        amplitudes = max_amplitude_resp(mean_resps, percentile=AMPLITUDE_PERC)

        cells_df.loc[cells_fsel.index, f"{selection}_rel"] = reliabilities
        cells_df.loc[cells_fsel.index, f"{selection}_amp"] = amplitudes

fl.save(master_path / "cells_df.h5", cells_df)
