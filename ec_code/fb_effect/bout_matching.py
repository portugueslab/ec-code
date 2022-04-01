##############################################
# Match bouts by duration & temporal proximity
# Here we select for each fish a subset of bouts in closed and open loop
# that had similar duration and occourred reasonable close to each other
# in the experiment to make sure we can compare responses with and w/o visual
# reafference.

bout_length_similarity_thr = 0.05
bout_max_timedistance = 600


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
            & (time_distances < bout_max_timedistance)
            & common_sel
        )

        # Calculate all duration differences
        diffs = np.abs(
            bouts_df.loc[selection, "duration"] - bouts_df.loc[b, "duration"]
        )

        # If we have a valid candidate, match it :
        if diffs.min() < bout_length_similarity_thr:
            bouts_df.loc[diffs.sort_values().index[0], "matched"] = True
            bouts_df.loc[b, "matched"] = True
