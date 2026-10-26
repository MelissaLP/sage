"""
Every glitch model of SAGE, in time and in SAGE's Q-transform.

Two random draws of each generator of ``sage.data.noise.analytic_glitches`` (McGAN's four bursts,
Antiglitch and the scattered-light arch) and of ``gengli_glitch``, each injected with
``GlitchInjector`` at ``--snr`` into 8 s of white noise at 4096 Hz. Each draw shows the data and
the injected glitch, and the SAGE ``QTransform`` image (Q in [4, 64], 10-1000 Hz, Q picked per
image, as gwpy's ``q_transform``).

Usage, from the SAGE repo root (gengli: ``pip install gengli``, else its row is skipped):

    PYTHONPATH=. python eucaif/example_glitch_models.py --outdir eucaif/figures/
    PYTHONPATH=. python eucaif/example_glitch_models.py --snr 30 --seed 1

Writes ``glitch_models_sage.png`` (and ``.pdf``).
"""

import argparse
import math
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sage.data.noise import ANALYTIC_GLITCHES, GlitchInjector, gengli_glitch
from sage.dsp.qtransform import QTransform

FS, DUR, F_LOW, F_HIGH = 4096.0, 8.0, 10.0, 1000.0
N = int(FS * DUR)
ASD = math.sqrt(2.0 / FS)                       # one-sided ASD of unit-variance white noise
NF, NT = 257, 4096
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# generator -> (title, keyword arguments, half-width of the plotted window in s)
MODELS = {
    "sine_gaussian": ("McGAN sine-Gaussian", dict(duration=1.0, t0=0.5), 0.15),
    "ringdown": ("McGAN ringdown", dict(duration=1.0, t0=0.5), 0.15),
    "white_noise_burst": ("McGAN white-noise burst", dict(duration=1.0, t0=0.5), 0.25),
    "gaussian_blip": ("McGAN Gaussian blip", dict(duration=1.0, t0=0.5), 0.25),
    "antiglitch_glitch": ("Antiglitch (Blip ranges)", dict(duration=2.0, t0=1.0, f0=(12.0, 260.0),
                                                           gbw=(0.29, 4.2)), 0.25),
    "scattered_light_arch": ("ArchEnemy scattered-light arch", dict(duration=6.0, t0=3.0), 3.0),
    "gengli_glitch": ("gengli blip (L1)", dict(duration=1.0, t0=0.5), 0.15),
}


def draw(name, kwargs, g):
    gen = gengli_glitch if name == "gengli_glitch" else ANALYTIC_GLITCHES[name]
    h, p = gen(1, FS, generator=g, dtype=torch.float64, return_params=True, **kwargs)
    keys = ("f0", "q", "gbw", "tau", "fringe_frequency", "timeperiod")
    label = ", ".join(f"{k} {float(p[k][0]):.3g}" for k in keys if k in p)
    return h[0], label.replace("fringe_frequency", "f_f").replace("timeperiod", "T")


def inject(glitch, snr, g):
    noise = torch.randn(N, generator=g, dtype=torch.float64)
    inj = GlitchInjector(FS, f_low=F_LOW, f_high=F_HIGH, lowpass_order=None)
    asd = torch.full((N // 2 + 1,), ASD, dtype=torch.float64)
    _, h = inj(torch.zeros(1, 1, N, dtype=torch.float64), glitch.view(1, 1, -1), asd, snr=snr, return_glitch=True)
    return noise + h[0, 0], h[0, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    models = dict(MODELS)
    try:
        import gengli  # noqa: F401
    except ImportError:
        print("gengli not installed: skipping its row")
        models.pop("gengli_glitch")

    qt = QTransform(FS, N, qrange=(4.0, 64.0), frange=(F_LOW, F_HIGH), logf=True, fres=NF, tres=DUR / NT)
    freqs = qt.frequencies.numpy()
    t_img = np.arange(NT) * DUR / NT - DUR / 2
    t = np.arange(N) / FS - DUR / 2

    fig, axes = plt.subplots(len(models), 2 * args.draws, figsize=(3.3 * 2 * args.draws, 1.9 * len(models)),
                             facecolor=SURFACE, squeeze=False)
    for i, (name, (title, kwargs, half)) in enumerate(models.items()):
        g = torch.Generator().manual_seed(args.seed + 100 * i)
        for d in range(args.draws):
            glitch, label = draw(name, kwargs, g)
            x, h = inject(glitch, args.snr, g)
            img, q = qt(x.float()[None], return_q=True)
            img = img[0].T.double().numpy()

            ax = axes[i, 2 * d]
            sel = np.abs(t) <= half
            ax.plot(t[sel], x.numpy()[sel], color="#b9b8b2", lw=0.5, label="data")
            ax.plot(t[sel], h.numpy()[sel], color=INK, lw=0.8, label="injected glitch")
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.5)
            ax.text(0.02, 0.96, label, transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
            if d == 0:
                ax.set_ylabel(title, fontsize=8, fontweight="bold")

            ax = axes[i, 2 * d + 1]
            cs = np.abs(t_img) <= half
            ax.pcolormesh(t_img[cs], freqs, img[:, cs], shading="nearest", cmap="viridis", vmin=0, rasterized=True)
            ax.set_yscale("log")
            ax.grid(False)
            ax.set_ylim(F_LOW, F_HIGH)
            ax.text(0.03, 0.96, f"Q = {float(q[0]):.1f}", transform=ax.transAxes, fontsize=7, color="white", va="top")
            ax.set_ylabel("Hz", fontsize=7)
            for a in axes[i, 2 * d:2 * d + 2]:
                a.tick_params(labelsize=7)
                if i == len(models) - 1:
                    a.set_xlabel("time from glitch centre [s]", fontsize=7)
        print(f"{name:22s} done")
    for d in range(args.draws):
        axes[0, 2 * d].set_title(f"draw {d + 1}: time series", fontsize=9)
        axes[0, 2 * d + 1].set_title(f"draw {d + 1}: SAGE Q-transform", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"SAGE glitch models at SNR {args.snr:g} in white noise (4096 Hz)", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    path = os.path.join(args.outdir, "glitch_models_sage")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print("saved", os.path.abspath(path) + ".png")


if __name__ == "__main__":
    main()
