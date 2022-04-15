"""Create dataframes that include all data from all experiments.
Can be run after from the raw data traces have been extracted and an
"exported.h5" dataframe that contains traces and ROI stack is created for every fish.
"""

import flammkuchen as fl
import numpy as np
import pandas as pd
from bouter import EmbeddedExperiment

from ec_code.file_utils import get_dataset_location

master_path = get_dataset_location("fb_effect")

path_list = [
    f.parent for f in master_path.glob("*_f[0-9]*/exported.h5")
]  # list all valid paths

# Initialize dataframe with all experiments
exp_df = pd.DataFrame(
    dict(
        fid=[path.name for path in path_list],
        location=[str(path) for path in path_list],
    )
)


def _get_centroids_from_stack(stack):
    """Calculate ROI centroids from a stack of ROIs, with -1 or -2 in non-labelled
    voxels.
    """
    n_rois = int(stack.max())
    centroids = np.zeros((n_rois, 3))
    for i in range(n_rois):
        idxs = np.argwhere(imaging_data["stack"] == i + 1)
        centroids[i, :] = np.mean(idxs, 0)

    return centroids


cells_df = pd.DataFrame()  # Dataframe with info for every cell in the dataframe
traces_df = pd.DataFrame()  # Dataframe with all traces pooled together
bouts_df = pd.DataFrame()  # Dataframe of all bouts

stimlog_dict = dict()  # dictionary with all stimulus logs
beh_dict = dict()  # dictionary with all behavior logs

# Run only if folders are found:
if len(path_list) > 0:
    # Load all traces from all fish:
    for path in path_list:
        print(path.stem)
        exp = EmbeddedExperiment(path)
        fid = path.name

        imaging_data = fl.load(path / "exported.h5")
        traces = imaging_data["traces"].T
        traces = (traces - np.nanmean(traces, 0)) / np.nanstd(traces, 0)

        beh_log = exp.behavior_log
        stim_log = exp.stimulus_log
        new_bouts_df = exp.get_bout_properties()
        new_bouts_df["fid"] = fid
        ends = new_bouts_df["t_start"] + new_bouts_df["duration"]
        inter_bout = new_bouts_df["t_start"][1:].values - ends[:-1].values
        new_bouts_df["inter_bout"] = np.insert(inter_bout, 0, np.nan)
        #     bouts_idxs = (bouts_df.start.values / exp.imaging_dt).astype(np.int)

        #     traces = bout_nan_traces(traces, bouts_idxs, wnd_pre=5, wnd_post=15)

        coords = _get_centroids_from_stack(imaging_data["stack"])
        new_cell_df = pd.DataFrame(
            dict(
                fid=fid,
                cell_n=np.arange(traces.shape[1]),
                x=coords[:, 2],
                y=coords[:, 1],
                z=coords[:, 0],
                cid=[f"{fid}_{i}" for i in range(traces.shape[1])],
            )
        )
        new_traces_df = pd.DataFrame(traces, columns=new_cell_df.cid)

        traces_df = pd.concat([traces_df, new_traces_df], axis=1)
        cells_df = pd.concat([cells_df, new_cell_df])
        bouts_df = pd.concat([bouts_df, new_bouts_df])

        stimlog_dict[path.stem] = stim_log
        beh_dict[path.stem] = beh_log

    # There is an inconsistency: here we save a single file, and later we re-save those
    # dfs in different files. Did not have time to fix this.
    fl.save(
        master_path / "summary_dfs.h5",
        dict(
            exp_df=exp_df,
            traces_df=traces_df,
            cells_df=cells_df,
            bouts_df=bouts_df,
            stimlog_dict=stimlog_dict,
            beh_dict=beh_dict,
        ),
    )
