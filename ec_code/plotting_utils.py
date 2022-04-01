import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Libertinus Sans"]
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["axes.edgecolor"] = "0.3"
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5

cols = (
    np.array(
        [
            [72, 177, 167],
            [200, 87, 123],
            [182, 94, 189],
            [103, 166, 78],
            [111, 124, 203],
            [180, 148, 64],
            [202, 96, 63],
        ]
    )
    / 255
)


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
    ax.fill_between(
        xarr,
        mn - std,
        mn + std,
        facecolor=c,
        linewidth=0,
        alpha=alpha,
        label="_nolegend_",
    )


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
    xarr = np.concatenate(np.vstack([np.arange(bins + 1),] * nreps))

    return xarr, yarr


def plot_crop(data_mat, f=None, bound_box=None, vlim=3, r=0.65):
    """Plot full matrix and individual and average traces for cropped data.
    """
    if f is None:
        f = plt.figure()
    if bound_box is None:
        bound_box = (0.1, 0.1, 0.6, 0.8)

    hp, vp, w, h = bound_box
    ax = f.add_axes((hp, vp + h * (1 - r), w, h * r))
    ax.imshow(
        data_mat.T,
        aspect="auto",
        extent=(-pre_int_s, post_int_s, 0, data_mat.shape[1]),
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
    )
    local_utils.despine(ax, ticks=True)
    ax1 = f.add_axes((hp, vp, w, h * (1 - r)))
    ax1.axvline(0, linewidth=0.5, c=(0.6,) * 3)
    ax1.plot(
        np.linspace(-pre_int_s, post_int_s, data_mat.shape[0]),
        data_mat,
        linewidth=0.1,
        c="b",
    )
    ax1.plot(
        np.linspace(-pre_int_s, post_int_s, data_mat.shape[0]),
        np.nanmean(data_mat, 1),
        linewidth=1.5,
        c="r",
    )
    despine(ax1, spare=["left", "bottom"])
    ax1.set_xlim(-pre_int_s, post_int_s)
    ax1.set_xlabel("time (s)")

    return ax, ax1
