import numpy as np
import sys
import os
from neo import Block
from neo.io import AxonIO, Spike2IO
import pandas as pd
from pathlib import Path
from ec_code.phy_tools.utilities import nanzscore, nanzscoremedian, butter_highpass_filter


def load_traces(filenames, **kwargs):
    """Function to load flexibly either one file or multiple files concatenated.
    """
    if type(filenames) == list:
        if len(filenames) == 1:
            return _load_trace(filenames[0], **kwargs)
        else:
            df = _load_trace(filenames[0], **kwargs)

            for f in filenames[1:]:
                new_df = _load_trace(f, **kwargs)
                new_df["sweep"] += df["sweep"].max() + 1
                df = pd.concat([df, new_df], axis=0)
            return df
    else:
        if Path(filenames).is_dir():
            files = list(Path(filenames).glob("*.abf"))

            if len(files) > 0:
                return load_traces(files, **kwargs)

        else:
            return _load_trace(filenames, **kwargs)


def _load_trace(filename, chan_names=[], verbose=True, artifact_chan=None,
                artifact_thr_sd=20, zscore=False, highpass_coff=None):
    """

    :param filename: .abf file to be loaded;
    :param chan_names: new names for the channels; if "null", data won't be loaded;
    :param verbose: print info abot the trace if true;
    :return:
    """

    # Time at the beginning of trace to be nan because of the artifact shape
    REMOVE_ART_S = 0.70

    filename = str(filename)  # convert to string if Path object
    if verbose:
        print('loading {}...'.format(filename))

    # Filetype from filename:
    filetype = filename.split(".")[-1]
    # Read binary file:
    r = AxonIO(filename=filename)
    bl = r.read_block(lazy=False)

    # To be changed? read infos from the first block:
    read_names = []
    for sig in bl.segments[0].analogsignals:
        read_names.append(sig.name)
    fn = np.float(sig.sampling_rate)
    print("Read info: Channels: {}; Sampling rate: {}".format(read_names, fn))

    # If names are specified, overwrite:
    for i, overwrite_name in enumerate(chan_names):
        read_names[i] = overwrite_name

    # Initialise data dictionary:
    data = {'time': [], 'sweep': []}
    for k in read_names:
        if k is not 'null':
            data[k] = []

    # Iterate over sweeps:
    artifact_positions = []
    for i, seg in enumerate(bl.segments):
        # Calculate sample count
        time_vect = np.array(seg.analogsignals[0].times)

        # If trace has to start from the artifact, find a new start_idx:
        start_idx = 0
        if artifact_chan is not None:
            data_arr = seg.analogsignals[artifact_chan].as_array()
            start_idx = find_artifact(data_arr, fn=fn,
                                      artifact_thr_sd=artifact_thr_sd)

        artifact_positions.append(start_idx/fn)  # To check if all are found

        # Get time and sweep number
        time_vect = time_vect[start_idx:] - time_vect[start_idx]
        data['time'] += [time_vect]
        data['sweep'] += [np.ones(len(time_vect), dtype=int) * i]

        # Append all channel traces, removing artifact if necessary:
        for j, k in enumerate(read_names):
            if k is not 'null':  # if channels are excluded
                data_arr = seg.analogsignals[j].as_array()[start_idx:, 0]

                if highpass_coff is not None:
                    data_arr = butter_highpass_filter(data_arr,
                                                      highpass_coff,
                                                      fn, order=4)

                if artifact_chan is not None:
                    data_arr[:int(REMOVE_ART_S * fn)] = np.nan

                if zscore:
                    data_arr = nanzscoremedian(data_arr)

                data[k] += [data_arr]

    if verbose:
        print("Artifacts not found: {}".format(0 in artifact_positions))

    # Concatenate values in dictionary
    for key in data.keys():
        data[key] = np.squeeze(
            np.concatenate([np.squeeze(x) for x in data[key]]))

    return pd.DataFrame(data)


def find_artifact(trace_in, fn=8333.33333, artifact_thr_sd=20):
    """ Function to find initial artifact that synchronises the stimulus.
    :param trace:
    :param fn:
    :return:
    """
    ART_LEN_MS = 7  # expected artifact length in points
    trace = trace_in.copy().flatten()
    start_idx = 0
    art_len_pts = np.int((ART_LEN_MS / 1000) * fn)
    artifact = np.zeros(art_len_pts * 2)
    artifact[art_len_pts // 2:-art_len_pts // 2] = 1
    convolved = np.convolve(trace, artifact)[
                len(artifact) // 2:len(artifact) // 2 + len(trace)]
    artifact_thr = np.std(convolved) * artifact_thr_sd + np.mean(
        convolved)
    return np.where(convolved > artifact_thr)[0][0]

