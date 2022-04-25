import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def shuffle_seq(sequence, n_shuffles=5, pre=None, post=None):
    shuffled = []
    for n in range(n_shuffles):
        interval = pre + post
        diff = np.diff(sequence)
        diff = diff[diff > 0]
        np.random.shuffle(diff)
        shuffled.append(np.mod(np.cumsum(diff), interval) - pre)
    return np.concatenate(shuffled)

def gen_shuffled_raster(raster, pre, post, s, n_shuffles=5):
    shuffled = []
    valid_raster = raster[(-pre < raster) & (raster < post)]
    for n in range(n_shuffles):
        interval = pre + post
        diff = np.diff(valid_raster)
        diff = diff[diff > 0]
        np.random.shuffle(diff)
        sequence = np.mod(np.cumsum(diff), interval) - pre

        spk_shuf, bins = np.histogram(sequence, np.arange(-pre, post, s))
        shuffled.append(spk_shuf)
    return np.concatenate(shuffled)


def get_moments(raster_spk, pre, post, s, n_shuffles=50, fn=8333.3):
    n_reps = raster_spk[-1, 1]
    norm_fact = 1/(s*n_reps)
    shuffled = gen_shuffled_raster(raster_spk[:, 0]/fn, pre, post, s, n_shuffles=n_shuffles)*norm_fact
    # fit with curve_fit
    return np.mean(shuffled), np.std(shuffled)



def single_plot(spikes, pre_int_sec, post_int_sec, step, ax=None, events=None,
                fn=8333.3333, n_shuffles=5):
    CONF_INT_COEF = 2.576  # corresponding to 99% confidence
    Y_PAD = 1.5

    norm_fact = 1 / (step * spikes[-1, 1])

    hst_arr = np.arange(-pre_int_sec, post_int_sec, step)
    spk_hst, _ = np.histogram(spikes[:, 0].flatten() / fn, hst_arr)
    spk_hst = spk_hst * norm_fact

    mu, sigma = get_moments(spikes, pre_int_sec, post_int_sec, step,
                            n_shuffles=n_shuffles)
    c1, c2 = mu - CONF_INT_COEF * sigma, mu + CONF_INT_COEF * sigma

    mids = (np.array([hst_arr[1:], hst_arr[:-1]])).mean(0)

    if ax is None:
        f, ax = plt.subplots()

    if events is not None:
        evt_hst, _ = np.histogram(events[:, 0].flatten() / fn, hst_arr)
        evt_hst = evt_hst * norm_fact
        ax.step(mids, evt_hst, where="mid", color=sns.color_palette()[0],
                alpha=0.4, label="tail")
        # ax.fill_between(mids, evt_hst, step="mid", color=(0.9, 0.9, 0.9))
    ax.set_yticks([])
    #ax.set_xticklabels([])

    ax2 = ax.twinx()
    ax2.step(mids, spk_hst, where="mid", color=sns.color_palette()[2],
             label="spikes")
    ax2.axhspan(c1, c2, color="k", alpha=0.1, zorder=-500, label="99% conf.")
    ax2.set_xlabel("Time from bout on (s)")
    ax2.set_ylim(np.max([c1 - Y_PAD, 0, spk_hst.min() - Y_PAD]),
                 np.max([c2 + Y_PAD, spk_hst.max() + Y_PAD]))
    #ax2.set_ylabel("Spikes /s")
    #sns.despine()

