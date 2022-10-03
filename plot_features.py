import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_feature(
    xs,
    sample_rate,
    feats,
    xlabel="time",
    ylabel="",
):
    batch = xs.shape[0]
    duration = xs.shape[1] / sample_rate
    feats /= feats.max(dim=1, keepdim=True).values

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(nrows=batch, ncols=1)
    axs = [fig.add_subplot(gs[b, 0]) for b in range(batch)]

    for i, (x, feat) in enumerate(zip(xs, feats)):
        axs[i].plot(torch.linspace(0, duration, xs.shape[1]), x, alpha=0.5)
        axs[i].plot(torch.linspace(0, duration, feats.shape[1]), feat)
        axs[i].tick_params(labelsize="xx-large")
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_xlim(0.0, duration)
        axs[i].set_ylim(-1.1, 1.1)
        axs[i].minorticks_on()
        axs[i].grid(True, which="major", alpha=1.0, linewidth=1)
        axs[i].grid(True, which="minor", alpha=0.3)

    plt.tight_layout()
    plt.show()
