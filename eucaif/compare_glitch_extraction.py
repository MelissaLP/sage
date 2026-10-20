"""
Glitch extraction in three invertible time-frequency representations, for a gengli blip and a
scattered-light arch:

- DeepExtractor's STFT  (n_fft 512, Hann window 64, hop 32; ``torch.istft`` back)
- QTAM                  (``qtam.SingleQTransform``, complex mode, its own inverse)
- SAGE                  (``QTransform.tiles`` / ``QTransform.inverse``: gwpy's tiling, exact inverse)

Each glitch is injected into 8 s of white noise at 4096 Hz at a given SNR, band-limited to
[10, 1000] Hz (the band SAGE's tiling covers). Two tests:

1. Round trip: x -> transform -> inverse, error within [10, 1000] Hz.
2. Oracle extraction: keep only the pixels where the glitch alone has at least ``kappa`` times
   the mean noise energy of that frequency row (an ideal binary mask, i.e. the best a
   network could hope to learn), invert, and compare the extracted glitch with the true one.
   The residual glitch SNR is the SNR of (extracted - true glitch): 0 is perfect subtraction.
   The Q of QTAM and SAGE is the one SAGE's Q-scan picks for the noisy data.

Usage, from the SAGE repo root (QTAM: ``pip install --no-deps qtam``; the blip needs gengli,
or ``--blip-file`` with a (n, L) .npy of gengli blips at 4096 Hz):

    PYTHONPATH=. python3 eucaif/compare_glitch_extraction.py --outdir figures/
"""

import argparse
import math
import os
import warnings

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sage.data.noise import GlitchInjector, scattered_light_arch
from sage.data.noise.glitch_noise import glitch_snr
from sage.dsp.qtransform import QTransform

warnings.filterwarnings("ignore")
try:
    # qtam 0.3.1 imports its bundled torch_spline_interpolation_1_0_0 as a top-level module
    # (see benchmark_qtransform.py), so put the package's own folder on the path first
    import importlib.util
    import sys
    sys.path.insert(0, os.path.dirname(importlib.util.find_spec("qtam").origin))
    from qtam import SingleQTransform
except Exception:
    SingleQTransform = None

FS, DUR, F_LOW, F_HIGH = 4096.0, 8.0, 10.0, 1000.0
N = int(FS * DUR)
ASD = math.sqrt(2.0 / FS)                       # one-sided ASD of unit-variance white noise
# reference palette: categorical slots 1-3 in fixed order; ink and surface tokens
COLORS = {"STFT": "#2a78d6", "QTAM": "#eb6834", "SAGE": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"STFT": "DeepExtractor STFT", "QTAM": "QTAM", "SAGE": "SAGE (gwpy tiling)"}


# -- transforms: complex coefficients (rows, columns) and their inverse ---------------------------

class Stft:
    name = "STFT"

    def __init__(self, q):
        self.win = torch.hann_window(64, dtype=torch.float64)
        self.freqs = np.fft.rfftfreq(512, 1 / FS)

    def coeffs(self, y):
        return torch.stft(y, n_fft=512, hop_length=32, win_length=64, window=self.win, return_complex=True)

    def inverse(self, c):
        return torch.istft(c, n_fft=512, hop_length=32, win_length=64, window=self.win, length=N)


class Qtam:
    name = "QTAM"

    def __init__(self, q):
        self.t = SingleQTransform(DUR, FS, q=q, frange=[F_LOW, float("inf")])
        f = self.t.freqs
        self.freqs = np.asarray(f.cpu() if torch.is_tensor(f) else f, dtype=float)

    def coeffs(self, y):
        return self.t(y[None, None], polar_mode=False, complex_mode=True)[0, 0]

    def inverse(self, c):
        r = self.t.invert_qtransform(c[None, None], polar_mode=False, complex_mode=True).reshape(-1)[:N]
        return r.real if torch.is_complex(r) else r


class Sage:
    name = "SAGE"

    def __init__(self, q, qt):
        self.qt, self.q = qt, q
        self.freqs = qt.tile_frequencies(q).numpy()

    def coeffs(self, y):
        return self.qt.tiles(y, self.q)

    def inverse(self, c):
        return self.qt.inverse(c, self.q)


def band(x):
    """x restricted to [F_LOW, F_HIGH] Hz (FFT mask)."""
    f = torch.fft.rfftfreq(N, 1 / FS)
    X = torch.fft.rfft(x) * ((f >= F_LOW) & (f <= F_HIGH))
    return torch.fft.irfft(X, n=N)


def residual_snr(e):
    """Optimal SNR of an error signal e against the white-noise ASD."""
    return float(glitch_snr(e / ASD, FS))


# -- one glitch ------------------------------------------------------------------------------------

def run(name, glitch, snr, methods_for_q, kappas, kappa, seed):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(4, N, generator=g, dtype=torch.float64)          # [0] data, [1:] noise floor
    inj = GlitchInjector(FS, f_low=F_LOW, f_high=F_HIGH, lowpass_order=None)
    asd = torch.full((N // 2 + 1,), ASD, dtype=torch.float64)
    _, h = inj(torch.zeros(1, 1, N, dtype=torch.float64), glitch.view(1, 1, -1), asd, snr=snr,
               return_glitch=True)
    h = h[0, 0]
    x = noise[0] + h

    qt = QTransform(FS, N, qrange=(4, 64), frange=(F_LOW, F_HIGH), norm=False, dtype=torch.float64)
    _, q = qt(x, return_q=True)
    q = float(q)
    methods = methods_for_q(q, qt)

    out = {"q": q, "x": x, "h": h, "methods": {}}
    for m in methods:
        floor = torch.stack([m.coeffs(n).abs() ** 2 for n in noise[1:]]).mean(dim=(0, -1))[:, None]
        cx, cg = m.coeffs(x), m.coeffs(h)
        rt = float(band(x - m.inverse(cx)).norm() / band(x).norm())
        sweep = []
        for k in kappas:
            mask = (cg.abs() ** 2) >= k * floor
            est = m.inverse(cx * mask)
            sweep.append(residual_snr(est - h))
        mask = (cg.abs() ** 2) >= kappa * floor
        est = m.inverse(cx * mask)
        out["methods"][m.name] = dict(
            obj=m, roundtrip=rt, sweep=sweep, est=est, kept=float(mask.float().mean()),
            res_snr=residual_snr(est - h),
            match=float(torch.dot(est, h) / (est.norm() * h.norm())),
            image=((cx.abs() ** 2) / floor).float(), mask=mask)
        print(f"{name:16s} {m.name:5s} Q={q:5.1f} round trip {rt:.1e} | kappa={kappa:g}: "
              f"kept {100 * out['methods'][m.name]['kept']:.2f}% of pixels, "
              f"match {out['methods'][m.name]['match']:.3f}, residual glitch SNR "
              f"{out['methods'][m.name]['res_snr']:.2f} (of {snr:g})")
    return out


# -- figures ---------------------------------------------------------------------------------------

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def pooled(img, mask, zoom, ncols=600):
    """Crop to the ``zoom`` window, average-pool energies (max-pool the mask) along time to at most
    ``ncols`` columns, and dilate the mask over ~1% of them so its outline is a smooth contour.
    Returns the times of the columns too."""
    t = np.arange(img.shape[-1]) * DUR / img.shape[-1]
    sel = torch.as_tensor((t >= zoom[0]) & (t <= zoom[1]))
    img, mask, t = img[:, sel], mask[:, sel].float(), t[sel.numpy()]
    if img.shape[-1] > ncols:
        img = F.adaptive_avg_pool1d(img[None], ncols)[0]
        mask = F.adaptive_max_pool1d(mask[None], ncols)[0]
        t = np.linspace(t[0], t[-1], ncols)
    k = 2 * (img.shape[-1] // 200) + 1
    mask = F.max_pool1d(mask[None], k, stride=1, padding=k // 2)[0]
    return img, mask, t


def glitch_figure(res, title, zoom, ylim, snr, kappa, path):
    names = list(res["methods"])
    fig, axes = plt.subplots(3, len(names), figsize=(5.2 * len(names), 10.5), facecolor=SURFACE,
                             gridspec_kw=dict(height_ratios=[1.3, 1, 1], hspace=0.35), sharex=True)
    t = np.arange(N) / FS
    sel = (t >= zoom[0]) & (t <= zoom[1])
    h = res["h"].numpy()
    lim = 1.1 * max(np.abs(h[sel]).max(), max(np.abs((r["est"] - res["h"]).numpy()[sel]).max()
                                                for r in res["methods"].values()))
    for j, name in enumerate(names):
        r = res["methods"][name]
        img, mask, tt = pooled(r["image"], r["mask"], zoom)
        ff = r["obj"].freqs
        ax = axes[0, j]
        mesh = ax.pcolormesh(tt, ff, img.numpy(), shading="auto", vmin=0, vmax=25, cmap="viridis",
                             rasterized=True)
        ax.contour(tt, ff, mask.numpy(), levels=[0.5], colors="white", linewidths=0.8)
        ax.set(yscale="log", ylim=ylim, xlim=zoom)
        ax.set_title(f"{LABELS[name]}" + (f", Q = {res['q']:.1f}" if name != "STFT" else ""), fontsize=11)
        if j == 0:
            ax.set_ylabel("frequency [Hz]")
        style(ax)

        ax = axes[1, j]
        ax.plot(t[sel], h[sel], color=INK, lw=1.0, label="injected glitch")
        ax.plot(t[sel], r["est"].numpy()[sel], color=COLORS[name], lw=1.4, alpha=0.9, label="extracted")
        ax.set(xlim=zoom, ylim=(-lim, lim))
        ax.set_title(f"extracted glitch: match {r['match']:.3f}", fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9)
        if j == 0:
            ax.set_ylabel("whitened strain")
        style(ax)

        ax = axes[2, j]
        ax.plot(t[sel], (r["est"] - res["h"]).numpy()[sel], color=COLORS[name], lw=1.0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set(xlim=zoom, ylim=(-lim, lim), xlabel="time [s]")
        ax.set_title(f"extracted − injected: residual SNR {r['res_snr']:.1f}", fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        if j == 0:
            ax.set_ylabel("error")
        style(ax)
    fig.colorbar(mesh, ax=axes[0, :], fraction=0.015, pad=0.01,
                 label="energy / mean noise energy of the row")
    fig.suptitle(f"{title}: injected at SNR {snr:g} in white noise; white contour = oracle mask "
                 f"(glitch energy ≥ {kappa:g}× noise)", color=INK, fontsize=12, y=0.94)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def summary_figure(results, kappas, kappa, snr, path):
    names = list(next(iter(results.values()))["methods"])
    fig, axes = plt.subplots(1, 1 + len(results), figsize=(5.2 * (1 + len(results)), 4.2), facecolor=SURFACE)
    ax = axes[0]
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        vals = [res["methods"][name]["res_snr"] for res in results.values()]
        xs = np.arange(len(results)) + (j - (len(names) - 1) / 2) * width
        bars = ax.bar(xs, vals, width * 0.92, color=COLORS[name], label=LABELS[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(results)), list(results))
    ax.set(ylabel="residual glitch SNR (lower is better)")
    ax.set_title(f"After oracle extraction (κ = {kappa:g}), SNR {snr:g} injected", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    for ax, (gname, res) in zip(axes[1:], results.items()):
        for name in names:
            ax.plot(kappas, res["methods"][name]["sweep"], color=COLORS[name], lw=2, marker="o", ms=5,
                    label=LABELS[name])
        ax.set(xscale="log", xlabel="mask threshold κ (glitch energy / noise energy)",
               ylabel="residual glitch SNR")
        ax.set_xticks(kappas, [f"{k:g}" for k in kappas])
        ax.minorticks_off()
        ax.set_title(f"{gname}: threshold sweep", fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        ax.legend(fontsize=8, frameon=False)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--kappa", type=float, default=1.0, help="oracle mask threshold")
    p.add_argument("--blip-file", help="(n, L) .npy of gengli blips at 4096 Hz (default: run gengli)")
    p.add_argument("--blip-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.blip_file:
        blip = np.load(args.blip_file)[args.blip_index]
    else:
        import gengli
        blip = np.asarray(gengli.glitch_generator("L1").get_glitch(n_glitches=args.blip_index + 1,
                                                                   srate=FS, seed=3))[args.blip_index]
    blip = torch.tensor(blip, dtype=torch.float64)
    arch = scattered_light_arch(1, FS, DUR, fringe_frequency=40.0, timeperiod=4.0, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(args.seed))[0]

    def methods_for_q(q, qt):
        ms = [Stft(q)]
        if SingleQTransform is not None:
            ms.append(Qtam(q))
        else:
            print("qtam not installed: QTAM skipped")
        return ms + [Sage(q, qt)]

    kappas = [0.25, 0.5, 1, 2, 4, 8]
    results = {}
    for gname, glitch, zoom, ylim, fname in (
            ("gengli blip", blip, (3.85, 4.15), (20, 1000), "extraction_blip.png"),
            ("scattered light", arch, (1.5, 6.5), (10, 200), "extraction_scattered_light.png")):
        results[gname] = run(gname, glitch, args.snr, methods_for_q, kappas, args.kappa, args.seed)
        glitch_figure(results[gname], gname, zoom, ylim, args.snr, args.kappa, os.path.join(args.outdir, fname))
    summary_figure(results, kappas, args.kappa, args.snr, os.path.join(args.outdir, "extraction_summary.png"))
    print("saved figures in", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
