import numpy as np
from matplotlib import pyplot as plt

def thr_plot(trace, coef=None, abs_thr=None):
    plt.plot(trace)
    if abs_thr:
        plt.plot(np.ones(len(trace))*abs_thr)
    else:
        plt.plot(np.ones(len(trace))*coef*np.nanstd(trace))


def stim_plot(stim_dict, colors, timedelta=False, **kwargs):
    for k in stim_dict.keys():
        for el in stim_dict[k]:
            plt.axvspan(el[0], el[1], color=colors[el[2]], **kwargs)


def exp_plot(exp_id='Exp022', **kwargs):
    if exp_id == 'Exp022':
        stim_plot(*get_exp22_stim(), **kwargs)
        

def plot_raster(binmat, pre=8000, fn=8333, **kwargs):
    binmat[np.isnan(binmat)] = 0
    idxs = np.where(binmat.flatten())[0]
    plt.scatter((np.mod(idxs, binmat.shape[1])-pre)/fn, -idxs//binmat.shape[1], **kwargs)


def get_exp22_stim():
    omrgo = np.arange(0, 60, 10) + 2.
    omrstop = omrgo + 5
    okrgo = np.array([60, 70, 80])
    okrstop = np.array([70, 80, 90])
    blkon = np.array([0, 2, 4, 6, 8]) + 90 + 1  # white first or black first???
    blkoff = blkon + 1

    colors = ['b', 'g', 'r', (0.8, 0.3, 0.9), 'y', 'k']
    stimdict = {'omr_rev': [[i, j, 0] for i, j in zip(omrgo[:3], omrstop[:3])],
                'omr_for': [[i, j, 1] for i, j in zip(omrgo[3:], omrstop[3:])],
                'okr': [[i, j, k + 2] for i, j, k in zip(okrgo, okrstop, list(range(4)))],
                'blk': [[i, j, 5] for i, j in zip(blkon, blkoff)]}
    return stimdict, colors

def get_exp22_img_stim():
    omrgo = np.arange(0, 60, 10) + 10
    omrstop = omrgo + 5
    okrgo = np.array([60, 70, 80])
    okrstop = np.array([70, 80, 90])
    blkon = np.array([0, 2, 4, 6, 8]) + 90 + 1  # white first or black first???
    blkoff = blkon + 1

    colors = ['b', 'g', 'r', (0.8, 0.3, 0.9), 'y', 'k']
    stimdict = {'omr_rev': [[i, j, 0] for i, j in zip(omrgo[:3], omrstop[:3])],
                'omr_for': [[i, j, 1] for i, j in zip(omrgo[3:], omrstop[3:])],
                'okr': [[i, j, k + 2] for i, j, k in zip(okrgo, okrstop, list(range(4)))],
                'blk': [[i, j, 5] for i, j in zip(blkon, blkoff)]}
    return stimdict, colors

def get_xy(m, pre_int=0, fn=8333):
    if len(m.shape) == 1:
        time = (np.arange(m.shape[0]) - pre_int) / fn
        trace = m
    else:
        trace = np.concatenate((m, np.empty((m.shape[0], 1)) * np.nan), 1).flatten()
        time = np.tile((np.concatenate([np.arange(m.shape[1]), [np.nan]]) - pre_int) / fn, m.shape[0])
    return time, trace