"""Preprocess data to aggregate dataframes: bouts_df, cells_df, exp_df.
"""

import flammkuchen as fl
import numpy as np
import pandas as pd
from bouter import bout_stats, utilities
from bouter.utilities import (
    extract_segments_above_threshold,
    predictive_tail_fill,
)
from scipy.interpolate import interp1d
from tqdm import tqdm

from ec_code.file_utils import get_dataset_location


def merge_bouts(bouts, min_dist):
    bouts = list(bouts)
    i = 1

    while i < len(bouts) - 1:
        current_dist = bouts[i + 1][0] - bouts[i][1]
        if current_dist < min_dist:
            bout_after_to_merge = bouts.pop(i + 1)
            bouts[i][1] = bout_after_to_merge[1]
        else:
            i += 1

    return np.array(bouts)


def get_bout_properties(t_array, tail_sum, vigor, threshold=0.1):
    """Create dataframe with summary of bouts properties.
    TODO: This is probably complete duplicate from bouter function, not sure why we define
    something different, I'd probably try and replace this with bouter defaults.

    :return: a dataframe giving properties for each bout
    """
    # Window for the calculation of bout directionality:
    DIRECTIONALITY_CALC_WND_S = 0.06
    # Min distance between 2 bouts below which they get merged in a single bout:
    MIN_DIST_BOUT_MERGE_S = 0.05

    THETA_OFFSET_DURATION_S = 0.1  # window to subract tail_sum mean before bout:

    behavior_dt = np.nanmedian(np.diff(t_array))
    bout_init_window_pts = int(DIRECTIONALITY_CALC_WND_S / behavior_dt)
    min_dist_bout_merge_pts = int(MIN_DIST_BOUT_MERGE_S / behavior_dt)
    theta_offset_duration_pts = int(THETA_OFFSET_DURATION_S / behavior_dt)

    bouts, _ = extract_segments_above_threshold(vigor, threshold, min_between=0)
    bouts = merge_bouts(bouts, min_dist_bout_merge_pts)

    peak_vig, med_vig, bias, bias_tot = bout_stats.bout_stats(
        vigor, tail_sum, bouts, bout_init_window_pts, theta_offset_duration_pts
    )
    n_pos_peaks, n_neg_peaks = bout_stats.count_peaks_between(
        utilities.bandpass(tail_sum, behavior_dt),
        bouts[:, 0],
        bouts[:, 1],
    )

    t_start, t_end = [t_array[bouts[:, i]] for i in range(2)]
    return pd.DataFrame(
        dict(
            t_start=t_start,
            duration=t_end - t_start,
            peak_vig=peak_vig,
            med_vig=med_vig,
            bias=bias,
            bias_total=bias_tot,
            n_pos_peaks=n_pos_peaks,
            n_neg_peaks=n_neg_peaks,
        )
    )


# TODO read some of those
PAUSE_DUR = 7
VIG_WND_S = 0.05
DT_BE_RESAMP = 0.0025
IMAGING_DT_S = 0.2
N_TAIL_SEGMENTS = 9
DIFF_VEL_THR = 0.025  # threshold for detecting stimulus moving times

master_path = get_dataset_location("fb_effect")
raw_data_file = master_path / "summary_dfs.h5"

USEFUL_STIM_KEYS = [
    f"selfcalib_shuntgrat_clol1D_{k}"
    for k in ["base_vel", "gain", "vel", "x", "fish_swimming"]
]

STIM_ATTRIB_TO_ADD = ["base_vel", "gain"]
raw_data_file = master_path / "summary_dfs.h5"
stim_namebase = "selfcalib_shuntgrat_clol1D"

# Load master dataframes:
exp_df = fl.load(raw_data_file, "/exp_df")
traces_df = fl.load(raw_data_file, "/traces_df")
cells_df = fl.load(raw_data_file, "/cells_df")


################################
# 0. Experiment stuff ##########

# Add genotype to experiments dataframe:
def genotype_from_name(name):
    return name.split("_")[-1] if name.split("_")[-1] in ["PC", "GC", "IO"] else "EC"


exp_df["genotype"] = exp_df["fid"].apply(genotype_from_name)
exp_df = exp_df.set_index("fid")

################################
# 1. Bouts/Trials stuff ########
bouts_df = pd.DataFrame()
trials_df = pd.DataFrame()

################################
# 2. Cells/traces stuff ########

# Check that indexing of cells_df and traces_df columns is the same:
(cells_df["cid"].values == traces_df.columns.values).all()
cells_df = cells_df.set_index("cid")

# Remove cells with sum == 0, which are invalid suite2p ROIs:
traces_df = traces_df.iloc[:, np.nansum(traces_df, 0) != 0]
cells_df = cells_df.loc[traces_df.columns, :]

################################
# 3. Broken fish stuff #########

# Ugly fix for broken data in fish 200829_f6_clol:
cells = cells_df.loc[cells_df["fid"] == "200829_f6_clol", :].index
vals = traces_df.loc[:, cells].values
n_to_fix = 500
traces_df.loc[:, cells] = np.concatenate(
    [np.zeros((n_to_fix, vals.shape[1])), vals[:-n_to_fix, :]], 0
)

################################
# 4. Fish-wise stuff ###########

# Filter dataframes for useful info:
filt_stimlog_dict = dict()
filt_behlog_dict = dict()

# For a bug in microscope acquisition software, there are some funny offsets
# that need to be used for synchronization with stytra temporal information.
# This has been semi/automatically computed looking at motor artefact timing.

# The fact that it is specified in a list makes me cry. This will probably break
# the moment the order of folders is altered for any reason.
# Don't try this at home.
# TODO: probably one of the first things that have to go
offsets = [
    -0.8,
    -0.8,
    -0.8,
    -0.8,
    -0.8,
    -0.8,
    0.2,
    -0.8,
    -0.8,
    -0.8,
    -0.6,
    -0.6,
    -0.8,
    -0.8,
    0.0,
    0.0,
    -0.6,
    0.0,
    -0.4,
    -0.4,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

for i, fid in enumerate(tqdm(exp_df.index)):
    offset = offsets[i]
    # Load stuff:
    stim_log = fl.load(raw_data_file, f"/stimlog_dict/{fid}")
    beh_log = fl.load(raw_data_file, f"/beh_dict/{fid}")

    stim_log.loc[:, "t"] += offset
    beh_log.loc[:, "t"] += offset

    # Remove duplications in the stimulus log dataframe, and set t as index:
    stim_log = stim_log.loc[~stim_log["t"].duplicated(keep="first")].copy()
    stim_log = stim_log.set_index("t")

    # Calculate vigor, tail_sum:
    dt_be = np.nanmedian(np.diff(beh_log["t"]))
    vig_wnd_pts = int(VIG_WND_S / dt_be)
    thetas = beh_log.loc[:, [f"theta_0{i}" for i in range(N_TAIL_SEGMENTS)]].values
    beh_log.loc[
        :, [f"theta_0{i}" for i in range(N_TAIL_SEGMENTS)]
    ] = predictive_tail_fill(thetas)

    beh_log.loc[:, "tail_sum"] = beh_log.loc[:, "theta_08"]
    beh_log["vigor"] = (
        beh_log["tail_sum"].rolling(window=vig_wnd_pts, center=True).std()
    )

    # Remove pre-acquisition start trailing behavior (t < 0):
    beh_log = beh_log[beh_log["t"] > 0]

    # Recalculate bout properties (to be fixed when bouter is corrected)
    new_bouts_df = get_bout_properties(
        beh_log.loc[:, "t"].values,
        beh_log.loc[:, "tail_sum"].values,
        beh_log.loc[:, "vigor"].values,
    )

    new_bouts_df["fid"] = fid
    ends = new_bouts_df["t_start"] + new_bouts_df["duration"]
    inter_bout = new_bouts_df["t_start"][1:].values - ends[:-1].values
    new_bouts_df["inter_bout"] = np.insert(inter_bout, 0, np.nan)

    # Interpolate stimulus parameters to get what was happening
    # at bout start times:
    for k in STIM_ATTRIB_TO_ADD:
        # create interpolating function:
        interp = interp1d(
            stim_log.index, stim_log[f"{stim_namebase}_{k}"], kind="nearest"
        )

        # Get interpolated value at bout onset:
        new_bouts_df.loc[:, k] = interp(new_bouts_df.loc[:, "t_start"])

    # Unique ids for bouts:
    new_bouts_df.index = [f"{fid}_bt{i:>04}" for i in new_bouts_df.index]

    # Add column for the distance of the subsequent bout:
    new_bouts_df["after_interbout"] = np.nan
    new_bouts_df.loc[new_bouts_df.index[:-1], "after_interbout"] = new_bouts_df.loc[
        new_bouts_df.index[1:], "inter_bout"
    ].values  # > min_dist_s

    bouts_df = pd.concat([bouts_df, new_bouts_df])

    # filter behavior log:
    filt_behlog_dict[fid] = beh_log[["t", "tail_sum", "vigor"]]

    # filter stimulus log with the actually useful entries:
    columns_to_take = USEFUL_STIM_KEYS.copy()
    if "moving_gratings_x" in stim_log.keys():
        columns_to_take.append("moving_gratings_x")

    stim_log_filt = stim_log[columns_to_take].copy()
    # Handle fish_swimming variable to use same format:
    ks_to_conv = ["selfcalib_shuntgrat_clol1D_fish_swimming"]
    for k in ks_to_conv:
        stim_log_filt.loc[:, k] = stim_log_filt[k].values.astype(float)

    filt_stimlog_dict[fid] = stim_log_filt

    ################################
    # 5. Trial-wise df #############

    # Backward gratings stimulus:
    # Use derivative to find trial start and trial end from stimulus velocity:
    vel_diff = np.ediff1d(stim_log["selfcalib_shuntgrat_clol1D_base_vel"], to_begin=0)
    trial_s_cl = stim_log[vel_diff < 0].index
    trial_e_cl = stim_log[vel_diff > 0].index
    trial_s_cl = trial_s_cl[: len(trial_e_cl)]

    # Initialize dataframe for closed-loop gratings:
    new_trials_df = pd.DataFrame(
        dict(
            t_start=trial_s_cl,
            fid=fid,
            t_end=trial_e_cl,
            duration=trial_e_cl - trial_s_cl,
            inter_bout_t=np.nan,
            trial_type="forward",
            bout_n=np.nan,
            bout_duration=np.nan,
            bout_latency=np.nan,
        ),
        index=range(len(trial_s_cl)),
    )

    if "moving_gratings_x" in stim_log.keys():
        # Forward gratings stimulus; here we have better-functioning phase log:
        vel_profile = (
            np.ediff1d(stim_log["moving_gratings_x"], to_begin=0) > DIFF_VEL_THR
        ).astype(float)
        vel_diff = np.ediff1d(vel_profile, to_begin=0)
        trial_s_backward = stim_log[vel_diff > 0].index
        trial_e_backward = stim_log[vel_diff < 0].index

        # Initialize dataframe for forward gratings:
        trials_df_backward = pd.DataFrame(
            dict(
                t_start=trial_s_backward,
                t_end=trial_e_backward,
                fid=fid,
                duration=trial_e_backward - trial_s_backward,
                inter_bout_t=np.nan,
                bout_n=np.nan,
                trial_type="backward",
                bout_duration=np.nan,
                bout_latency=np.nan,
            ),
            index=np.arange(len(trial_s_backward)) + len(trial_s_cl),
        )

        new_trials_df = pd.concat([new_trials_df, trials_df_backward])

    # Loop over trials, and fill dataframe with bout statistics:

    for i in new_trials_df.index:
        # Find bouts in temporal boundaries of trial i:
        bout_idxs = new_bouts_df[
            (new_bouts_df["t_start"] > new_trials_df.loc[i, "t_start"])
            & (new_bouts_df["t_start"] < new_trials_df.loc[i, "t_end"])
        ].index
        new_trials_df.loc[i, "bout_n"] = len(bout_idxs)

        # If there are bouts in the trial:
        if len(bout_idxs) > 0:
            new_trials_df.loc[i, "bout_latency"] = (
                new_bouts_df.loc[bout_idxs[0], "t_start"]
                - new_trials_df.loc[i, "t_start"]
            )
            new_trials_df.loc[i, "bout_duration"] = new_bouts_df.loc[
                bout_idxs[0], "duration"
            ]
        # Check for leading bouts:
        trail_bout_idxs = new_bouts_df[
            (new_bouts_df["t_start"] > new_trials_df.loc[i, "t_start"] - PAUSE_DUR)
            & (new_bouts_df["t_start"] < new_trials_df.loc[i, "t_start"])
        ].index
        if len(trail_bout_idxs) > 0:
            new_trials_df.loc[i, "lead_bout_latency"] = (
                new_bouts_df.loc[trail_bout_idxs[0], "t_start"]
                - new_trials_df.loc[i, "t_start"]
            )

            # Check for trailing bouts:
        trail_bout_idxs = new_bouts_df[
            (new_bouts_df["t_start"] > new_trials_df.loc[i, "t_end"])
            & (new_bouts_df["t_start"] < new_trials_df.loc[i, "t_end"]) + PAUSE_DUR
        ].index
        if len(trail_bout_idxs) > 0:
            new_trials_df.loc[i, "trail_bout_latency"] = (
                new_bouts_df.loc[trail_bout_idxs[0], "t_start"]
                - new_trials_df.loc[i, "t_end"]
            )

    new_trials_df.index = [f"{fid}_tr{i:>04}" for i in new_trials_df.index]
    trials_df = pd.concat([trials_df, new_trials_df])

# Imaging dt for each experiment, might have to be changed in the future.
dt = np.full(len(exp_df), IMAGING_DT_S)
exp_df["dt"] = dt

# Add genotype info to cells
cells_df["genotype"] = cells_df["fid"].map(exp_df["genotype"])
bouts_df["genotype"] = bouts_df["fid"].map(exp_df["genotype"])
trials_df["genotype"] = trials_df["fid"].map(exp_df["genotype"])

fl.save(master_path / "exp_df.h5", exp_df)
fl.save(master_path / "traces_df.h5", traces_df)
fl.save(master_path / "cells_df.h5", cells_df)
fl.save(master_path / "bouts_df.h5", bouts_df)
fl.save(master_path / "trials_df.h5", trials_df)
fl.save(master_path / "stim_dict.h5", filt_stimlog_dict)
fl.save(master_path / "beh_dict.h5", filt_behlog_dict)


# Resampled behavior dataset
resamp_beh_dict = dict()
for fid in tqdm(exp_df.index):
    resamp_beh_dict[fid] = utilities.resample(
        fl.load(master_path / "beh_dict.h5", f"/{fid}"), resample_sec=DT_BE_RESAMP
    )

fl.save(master_path / "resamp_beh_dict.h5", resamp_beh_dict)
