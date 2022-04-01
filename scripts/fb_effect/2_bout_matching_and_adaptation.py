"""Perform bout-matching procedure to match bouts in duration. It also compute an
adaptation score for each fish. The bouts_df and exp_df files are then overwritten
with the new columns.
"""

from ec_code.file_utils import get_dataset_location
import flammkuchen as fl
import numpy as np
from tqdm import tqdm

##############################################
# Match bouts by duration & temporal proximity
# Here we select for each fish a subset of bouts in closed and open loop
# that had similar duration and occourred reasonable close to each other
# in the experiment to make sure we can compare responses with and w/o visual
# reafference.

# Maximum difference in bout length (in seconds) allowed in the matching procedure:
BOUT_LENGTH_SIMILARITY_THR_S = 0.05

# Maximum distance in time between two bouts. We do this to avoid the confounding effect
# of slow trends during the experiment of bout durations and fluorescence signal, which
# could produce spurious differences in responses to bouts (when comparing eg
# closed-loop bouts from the beginning of the experiment with open-loop from the end).
BOUT_MAX_TIMEDISTANCE_S = 600

master_path = get_dataset_location("fb_effect")

# Load dataframes:
exp_df = fl.load(master_path / "exp_df.h5")
bouts_df = fl.load(master_path / "bouts_df.h5")

bouts_df["matched"] = False
for fid in tqdm(exp_df.index):
    common_sel = (
        (bouts_df["fid"] == fid) & (bouts_df["mindist_included"]) & ~bouts_df["spont"]
    )
    for b in bouts_df.loc[(bouts_df["gain"] == 1) & common_sel].index:

        time_distances = np.abs(
            bouts_df.loc[bouts_df["fid"] == fid, "t_start"] - bouts_df.loc[b, "t_start"]
        )

        # Candidate bouts to match: gain 1, not matched yet,
        # with minimum spacing from other bouts, and not too far in time:
        selection = (
            (bouts_df["gain"] == 0)
            & ~bouts_df["matched"]
            & (time_distances < BOUT_MAX_TIMEDISTANCE_S)
            & common_sel
        )

        # Calculate all duration differences
        diffs = np.abs(
            bouts_df.loc[selection, "duration"] - bouts_df.loc[b, "duration"]
        )

        # If we have a valid candidate, match it :
        if diffs.min() < BOUT_LENGTH_SIMILARITY_THR_S:
            bouts_df.loc[diffs.sort_values().index[0], "matched"] = True
            bouts_df.loc[b, "matched"] = True