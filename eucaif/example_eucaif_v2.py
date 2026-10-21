"""
Example of the EuCAIF v2 dataset: one noise-only and one BBH example, H1 and L1, as Q-scans and
whitened time series of the frame, every glitch labelled with its class; and how long it takes.

Usage, from the SAGE repo root:

    PYTHONPATH=. python3 eucaif/example_eucaif_v2.py --seed 2 --out eucaif_v2_example.png [--timing]
"""

import argparse
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eucaif.eucaif_v2 import QTransforms, make_generator

INK2, GRID = "#52514e", "#e4e3df"
BBH, PHYSICAL, NONPHYSICAL = "#2a78d6", "#eb6834", "#6250d6"      # reference palette slots 1, 2, 7


def plot(data, gen, qt, path):
    t = gen.t.numpy() - qt.qpad                                     # time from the frame start
    fr = qt.frame_slice
    n = len(data["y"]) // 2
    n_phys = len(gen.glitches[0].classes)
    tab = data["glitch_table"]
    fig, axes = plt.subplots(2, 2 * gen.D, figsize=(5.2 * gen.D * 2, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.12, wspace=0.18))
    for j, i in enumerate((0, n)):                                  # first noise-only, first BBH example
        scan, q = qt.qscan(data["x"][i].float(), return_q=True)
        for d in range(gen.D):
            col = 2 * j + d
            ax = axes[0, col]
            ax.pcolormesh(qt.scan.times.numpy() - qt.qpad, qt.scan.frequencies.numpy(), scan[d].T.numpy(),
                          shading="auto", vmin=0, vmax=25, rasterized=True)
            what = f"BBH (network SNR {float(data['snr'][i - n]):.1f})" if data["y"][i] else "noise only"
            ax.set_title(f"{gen.detectors[d]}, {what}\nQ-scan, Q = {float(q[d]):.0f}", fontsize=10)
            ax.set(yscale="log")
            if col == 0:
                ax.set_ylabel("frequency [Hz]")

            ax = axes[1, col]
            ax.plot(t[fr], data["x"][i, d].numpy()[fr], lw=0.35, color="0.65", label="whitened strain")
            g = data["glitch_whitened"][i, d].numpy()[fr]
            if np.abs(g).max() > 0:
                ax.plot(t[fr], g, lw=0.8, color=PHYSICAL, alpha=0.9, label="glitches")
            if data["y"][i]:
                ax.plot(t[fr], data["signal_whitened"][i - n, d].numpy()[fr], lw=1.0, color=BBH,
                        label="BBH")
                ax.axvline(float(data["tc"][i - n]) - qt.qpad, color=BBH, ls="--", lw=0.7)
            sel = (tab["example"] == i) & (tab["detector"] == d)
            ymax = max(4.0, float(np.abs(data["x"][i, d].numpy()[fr]).max()))
            outside, last = 0, [-1e9, -1e9, -1e9]       # last label time in each of three label rows
            order = np.argsort(tab["time"][sel])
            for c, tt, snr in zip(tab["class"][sel][order], tab["time"][sel][order] - qt.qpad, tab["snr"][sel][order]):
                color = NONPHYSICAL if c >= n_phys else PHYSICAL
                if 0 <= tt <= qt.frame_s:
                    for a in axes[:, col]:
                        a.axvline(tt, color=color, lw=0.8, ls=":")
                    row = next((r for r in range(3) if tt - last[r] > 0.35), int(np.argmin(last)))
                    last[row] = tt
                    ax.text(tt, ymax * (0.95 - 0.28 * row), f"{gen.glitch_classes[c]}\nSNR {snr:.0f}", fontsize=7,
                            color=color, ha="center", va="top",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
                else:                                   # centre outside the frame: note it, stacked at the lower right
                    ax.text(0.99, 0.04 + 0.07 * outside, f"{gen.glitch_classes[c]} centred at {tt:+.2f} s",
                            transform=ax.transAxes, fontsize=6.5, color=color, ha="right")
                    outside += 1
            ax.set(ylim=(-1.15 * ymax, 1.15 * ymax), xlabel="time from the frame start [s]")
            ax.grid(color=GRID, lw=0.5)
            ax.legend(loc="lower left", fontsize=7)
            if col == 0:
                ax.set_ylabel("whitened strain")
    axes[0, 0].set_xlim(0, qt.frame_s)
    fig.suptitle("EuCAIF v2.1: 2 s frame of whitened H1/L1 strain (orange: O3a glitches, violet: non-physical "
                 "McGAN glitch, blue: BBH)", fontsize=11, y=0.995)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("saved", path)


def timing(config):
    t0 = time.perf_counter()
    gen = make_generator(config, seed=0)
    setup = time.perf_counter() - t0
    gen.generate(1, noise_seed=1)                                   # warm-up
    t0 = time.perf_counter()
    for k in range(5):
        gen.generate(1, noise_seed=10 + k)
    single = (time.perf_counter() - t0) / 5 / 2
    t0 = time.perf_counter()
    big = gen.generate(64, noise_seed=99)
    batched = (time.perf_counter() - t0) / 128
    qt = QTransforms(config)
    x = big["x"].float()
    qt.qscan(x[:4])
    t0 = time.perf_counter()
    qt.qscan(x)
    t_scan = (time.perf_counter() - t0) / len(x)
    t0 = time.perf_counter()
    qt.tiles(x)
    t_tiles = (time.perf_counter() - t0) / len(x)
    print(f"generator set-up (once): {setup:.1f} s")
    print(f"one example (H1 + L1), generated alone: {1e3 * single:.0f} ms; in batches of 128: "
          f"{1e3 * batched:.1f} ms ({1 / batched:.0f} examples/s per CPU process)")
    print(f"Q-scan of one example (H1 + L1) on CPU: {1e3 * t_scan:.1f} ms; invertible tiles (2 Q-planes): "
          f"{1e3 * t_tiles:.1f} ms")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="eucaif/eucaif_v2.yaml")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--out", default="eucaif_v2_example.png")
    p.add_argument("--timing", action="store_true")
    args = p.parse_args()
    gen = make_generator(args.config, seed=args.seed)
    data = gen.generate(1, noise_seed=args.seed, keep_signal=True)    # one noise-only, one BBH example
    plot(data, gen, QTransforms(args.config), args.out)
    if args.timing:
        timing(args.config)


if __name__ == "__main__":
    main()
