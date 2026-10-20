"""
Example: a dense analytic-glitch frame (DeepExtractor-style: 5-30 glitches of random types per
frame) and what SAGE's Q-transform shows of it.

For each detector, one frame of white (already whitened) Gaussian noise gets the glitches drawn
by ``GlitchPopulation`` from ``sage_analytic_glitches_dense.yaml`` (injected at their SNR), and is
Q-transformed with SAGE's gwpy-exact ``QTransform`` (one Q per frame, the plane with the loudest
tile, median-normalised energy). Above each Q-scan, a timeline shows every injected glitch:
class (row), time, and SNR (marker size).

Usage, from the SAGE repo root:

    PYTHONPATH=. python3 eucaif/example_dense_glitches.py --seed 0 --out dense_glitches.png
"""

import argparse
import math

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sage.data.noise import GlitchPopulation
from sage.dsp.qtransform import QTransform

# reference palette, categorical slots in fixed order; ink and surface tokens
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="eucaif/sage_analytic_glitches_dense.yaml")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="dense_glitches.png")
    args = p.parse_args()

    pop = GlitchPopulation(args.config)
    fs, dur = pop.sample_rate, args.duration
    n = int(round(fs * dur))
    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(len(pop.detectors), n, generator=g, dtype=torch.float64)
    asd = torch.full((n // 2 + 1,), math.sqrt(2.0 / fs), dtype=torch.float64)   # unit-variance white noise
    strain, events = pop.inject(noise, asd, g, f_low=10.0)

    qt = QTransform(fs, n, qrange=(4, 64), frange=(10, 1000), logf=True, fres=300, tres=dur / 1000,
                    dtype=torch.float64)
    qscan, q = qt(strain, return_q=True)

    names = list(pop.classes)
    color = {c: SLOTS[i % len(SLOTS)] for i, c in enumerate(names)}
    D = len(pop.detectors)
    fig, axes = plt.subplots(2 * D, 1, figsize=(13, 4.6 * D), facecolor=SURFACE, sharex=True,
                             gridspec_kw=dict(height_ratios=[0.55, 1.6] * D, hspace=0.12))
    for d, det in enumerate(pop.detectors):
        ax = axes[2 * d]
        for ev in events[d]:
            y = names.index(ev["cls"])
            ax.scatter(ev["time"], y, s=12 + 1.6 * ev["snr"], color=color[ev["cls"]], edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
        ax.set_yticks(range(len(names)), names, fontsize=8, color=INK2)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{det}: {len(events[d])} glitches injected (marker size ∝ SNR)", fontsize=10,
                     color=INK, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)

        ax = axes[2 * d + 1]
        mesh = ax.pcolormesh(qt.times.numpy(), qt.frequencies.numpy(), qscan[d].T.numpy(), shading="auto",
                             vmin=0, vmax=25, cmap="viridis", rasterized=True)
        ax.set(yscale="log", ylabel="frequency [Hz]")
        ax.set_title(f"{det}: SAGE Q-transform, Q = {float(q[d]):.1f} (loudest plane), "
                     f"median-normalised energy", fontsize=10, color=INK, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        ax.yaxis.label.set_color(INK2)
        fig.colorbar(mesh, ax=ax, fraction=0.02, pad=0.01, label="normalised energy")
    axes[-1].set_xlabel("time [s]", color=INK2)
    axes[-1].set_xlim(0, dur)
    fig.suptitle(f"Dense analytic glitches ({pop.per_segment[0]}-{pop.per_segment[1]} per frame), "
                 f"{dur:g} s of white noise at {fs:g} Hz", color=INK, fontsize=12, y=0.995)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    for d, det in enumerate(pop.detectors):
        print(det, f"Q={float(q[d]):.1f}", ", ".join(f"{e['cls']}@{e['time']:.2f}s/SNR{e['snr']:.0f}"
                                                     for e in sorted(events[d], key=lambda e: e["time"])))
    print("saved", args.out)


if __name__ == "__main__":
    main()
