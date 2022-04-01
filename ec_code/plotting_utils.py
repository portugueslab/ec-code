from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Libertinus Sans"]
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["axes.edgecolor"] = "0.3"
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5

def despine(ax, spare=[], ticks=False):
    for side in ["left", "right", "top", "bottom"]:
        if side not in spare:
            ax.axes.spines[side].set_visible(False)
            if ticks and side == "left":
                ax.set_yticks([])
            if ticks and side == "bottom":
                ax.set_xticks([])

def shade_error(block, ax, xarr, c, alpha=0.3, label=""):
    if ax is None:
        ax = plt.gca()

    mn = np.nanmean(block, 1)
    std = np.nanstd(block, 1)

    ax.plot(xarr, mn, c=c, linewidth=2, label=label)
    ax.fill_between(xarr, mn - std, mn + std, facecolor=c, linewidth=0, alpha=alpha,
                    label='_nolegend_')


def bout_lines_arr(start_idxs, vmin=-2, vmax=5):
    if isinstance(start_idxs, pd.DataFrame) or isinstance(start_idxs, pd.Series):
        start_idxs = start_idxs.values
    x_arr = np.repeat(start_idxs, 3)
    x_arr[2::3] = np.nan

    y_arr = np.tile([vmin, vmax, np.nan], len(start_idxs))

    return x_arr, y_arr


def get_xy(cropped):
    bins, nreps = cropped.shape
    nanned = np.vstack([cropped, np.full(nreps, np.nan)]).T
    yarr = np.concatenate(nanned)
    xarr = np.concatenate(np.vstack([np.arange(bins + 1), ] * nreps))

    return xarr, yarr