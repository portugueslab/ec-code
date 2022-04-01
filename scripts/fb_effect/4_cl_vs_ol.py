import flammkuchen as fl
from scipy.stats import ttest_ind
from tqdm import tqdm

from ec_code.analysis_utils import crop_trace
from ec_code.fb_effect.default_vals import DEF_DT, POST_INT_BT_S, PRE_INT_BT_S
from ec_code.file_utils import get_dataset_location
from ec_code.plotting_utils import *

master_path = get_dataset_location("fb_effect")

exp_df = fl.load(master_path / "exp_df.h5")
trials_df = fl.load(master_path / "trials_df.h5")
cells_df = fl.load(master_path / "cells_df.h5")
bouts_df = fl.load(master_path / "bouts_df.h5")
traces_df = fl.load(master_path / "traces_df.h5")


#############################
# Calculate ol vs cl pvalues:
wnd_s = 2  # Window of average response over which calculate pval
wnd = int(wnd_s / DEF_DT)
perc_excluding_shortbouts = 20
n_pval_intervals = 4
step_pval_intervals = 1

for i in range(step_pval_intervals):
    cells_df[f"pval_clol"] = np.nan
    cells_df[f"int0_clol"] = np.nan
    cells_df[f"int1_clol"] = np.nan
    # cells_df[f"amp_cl"] = np.nan
    # cells_df[f"amp_ol"] = np.nan

for fid in tqdm(exp_df.index):
    cell_idxs = cells_df[cells_df["fid"] == fid].index

    sel = (bouts_df["fid"] == fid) & bouts_df["matched"]

    # Esclude short bouts from p val calculation:
    min_dur = np.percentile(bouts_df.loc[sel, "duration"], perc_excluding_shortbouts)
    sel = sel & (bouts_df["duration"] >= min_dur)

    # Crop bouts:
    timepoints = bouts_df.loc[sel, "t_start"]
    cropped = crop_trace(
        traces_df[cells_df[cells_df["fid"] == fid].index].values,
        timepoints,
        DEF_DT,
        PRE_INT_BT_S,
        POST_INT_BT_S,
        normalize=True,
    )

    for n, cell_idx in enumerate(cell_idxs):
        # Calculate p value over 4 intervals:
        pvals = np.zeros(n_pval_intervals)
        # amps = np.zeros(n_pval_intervals, 2)

        for i in range(n_pval_intervals):
            t_start = PRE_INT_BT_S + i * step_pval_intervals
            i_start = int(t_start / DEF_DT)
            mean_resps = [
                np.nanmean(
                    cropped[i_start : i_start + wnd, bouts_df.loc[sel, "gain"] == g, n],
                    0,
                )
                for g in range(2)
            ]
            pvals[i] = ttest_ind(mean_resps[0], mean_resps[1]).pvalue
            # amps[i, :] =

        best_p_idx = np.argmin(pvals)
        best_t_start = best_p_idx * step_pval_intervals

        cells_df.loc[cell_idx, f"pval_clol"] = pvals[best_p_idx]
        cells_df.loc[cell_idx, f"int0_clol"] = best_t_start
        cells_df.loc[cell_idx, f"int1_clol"] = best_t_start + wnd_s
        # cells_df.loc[cell_idx, f"amp_cl"] = np.nan
        # cells_df.loc[cell_idx, f"amp_ol"] = np.nan

fl.save(master_path / "cells_df.h5", cells_df)
