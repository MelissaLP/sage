"""
How well does the pretrained DeepExtractor extract the scattered-light arch (and a gengli blip)?

Same data as ``compare_glitch_extraction.py`` (8 s of white noise at 4096 Hz, glitch at SNR 20 in
[10, 1000] Hz, same seeds). DeepExtractor (UNET2D, ``DeepExtractor_257``, bilby-noise checkpoint)
works on 2 s at 4096 Hz, so the 8 s are processed in 2 s windows with a 1 s stride, keeping the
central second of each. Its ``background()`` is reproduced exactly: standardise with a bundled
scaler, STFT (n_fft 512, Hann 64, hop 32), UNET2D, iSTFT, unstandardise; the glitch is
input - background. The scalers are a mean and a std (scikit-learn not needed):
``scaler.pkl`` (PyCBC whitening, default) and ``scaler_bilby.pkl``.

Our whitened noise has unit variance. PyCBC-whitened Gaussian noise (dividing by the ASD in the
frequency domain) has std sqrt(fs / 2) = 45.25 at 4096 Hz, so with ``scaler.pkl`` the input is
multiplied by that (``--input-scale``). The SNR extracted from pure noise is printed as a check.

Compared with the oracle-mask bounds (STFT, SAGE) and a template fit with the true arch shape.

Usage, from the SAGE repo root:

    PYTHONPATH=. python3 eucaif/test_deepextractor_arch.py \\
        --deepextractor-src ../deepextractor/src \\
        --checkpoint path/to/DeepExtractor_257/checkpoint_best_bilby_noise_base.pth.tar \\
        --blip-file blips.npy --outdir eucaif/figures
"""

import argparse
import math
import os
import sys
import warnings

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from eucaif.compare_glitch_extraction import (ASD, DUR, FS, GRID, INK, INK2, N, SURFACE, Sage, Stft,
                                              residual_snr, run, style)
from sage.data.noise import GlitchInjector, scattered_light_arch

SCALERS = {"pycbc": (-0.00041938, 283.84534957),          # deepextractor/assets/scaler.pkl
           "bilby": (3.05534442e-19, 8.79345113)}         # deepextractor/assets/scaler_bilby.pkl
DE_COLOR = "#6250d6"                                      # reference palette slot 7 (violet)
WIN = int(2 * FS)


class DeepExtractor:
    """DeepExtractorModel.background(), on 2 s windows, for unit-variance whitened input scaled by ``s``."""

    def __init__(self, src, checkpoint, scaler="pycbc"):
        self.mean, self.std = SCALERS[scaler]
        sys.path.insert(0, src)
        from deepextractor.models.architectures import UNET2D
        from deepextractor.utils.stft import apply_istft, apply_stft
        self.apply_stft, self.apply_istft = apply_stft, apply_istft
        self.model = UNET2D(in_channels=2, out_channels=2)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.window = torch.hann_window(64)

    def glitch_2s(self, x, s):
        """(n, 8192) unit-variance segments -> extracted glitch, same units."""
        scaled = ((x * s - self.mean) / self.std).numpy()
        S = self.apply_stft(scaled, 512, 32, 64, self.window)
        with torch.no_grad():
            bg = self.apply_istft(self.model(S), 512, 32, 64, self.window).double()
        bg = bg * self.std + self.mean
        return (x * s - bg) / s

    def glitch(self, x, s):
        """8 s series: 2 s windows every 1 s, keep the central second (the outer half at the ends)."""
        starts = list(range(0, N - WIN + 1, WIN // 2))
        out = self.glitch_2s(torch.stack([x[a:a + WIN] for a in starts]), s)
        est = torch.zeros(N, dtype=torch.float64)
        for i, a in enumerate(starts):
            lo = 0 if i == 0 else WIN // 4
            hi = WIN if i == len(starts) - 1 else 3 * WIN // 4
            est[a + lo:a + hi] = out[i, lo:hi]
        return est


def template_fit(x, seed):
    """Least-squares amplitude and phase of the true arch shape (two quadratures)."""
    asd = torch.full((N // 2 + 1,), ASD, dtype=torch.float64)
    inj = GlitchInjector(FS, f_low=10.0, f_high=1000.0, lowpass_order=None)
    a = scattered_light_arch(1, FS, DUR, fringe_frequency=40.0, timeperiod=4.0, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(seed))[0]
    b = inj(torch.zeros(1, 1, N, dtype=torch.float64), a.view(1, 1, -1), asd, snr=10.0, return_glitch=True)[1][0, 0]
    A = torch.stack([b, torch.fft.irfft(torch.fft.rfft(b) * -1j, n=N)], 1)
    return (A @ torch.linalg.lstsq(A, x[:, None]).solution)[:, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deepextractor-src", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--blip-file", required=True)
    p.add_argument("--scaler", choices=sorted(SCALERS), default="pycbc")
    p.add_argument("--input-scale", type=float, default=math.sqrt(FS / 2),
                   help="factor applied to the unit-variance data (default: PyCBC whitening, sqrt(fs/2))")
    p.add_argument("--tag", default="", help="suffix for the figure name")
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    de = DeepExtractor(args.deepextractor_src, args.checkpoint, args.scaler)
    s_best = args.input_scale

    # check: what it extracts from pure noise (0 is ideal; the noise itself has SNR ~181 here)
    noise = torch.randn(3, N, generator=torch.Generator().manual_seed(1234), dtype=torch.float64)
    print(f"{os.path.basename(args.checkpoint)}, {args.scaler} scaler, input scale {s_best:.2f}: SNR extracted "
          f"from pure noise {np.mean([residual_snr(de.glitch(n, s_best)) for n in noise]):.1f}")

    blip = torch.tensor(np.load(args.blip_file)[0], dtype=torch.float64)
    stats = {g: {k: [] for k in ("DeepExtractor", "oracle STFT mask", "oracle SAGE mask", "template fit")}
             for g in ("scattered light", "gengli blip")}
    examples = {}
    for seed in range(args.seeds):
        arch = scattered_light_arch(1, FS, DUR, fringe_frequency=40.0, timeperiod=4.0, dtype=torch.float64,
                                    generator=torch.Generator().manual_seed(seed))[0]
        for gname, glitch in (("scattered light", arch), ("gengli blip", blip)):
            res = run(gname, glitch, args.snr, lambda q, qt: [Stft(q), Sage(q, qt)], [1.0], 1.0, seed)
            x, h = res["x"], res["h"]
            est = de.glitch(x, s_best)
            stats[gname]["DeepExtractor"].append(residual_snr(est - h))
            stats[gname]["oracle STFT mask"].append(res["methods"]["STFT"]["res_snr"])
            stats[gname]["oracle SAGE mask"].append(res["methods"]["SAGE"]["res_snr"])
            if gname == "scattered light":
                stats[gname]["template fit"].append(residual_snr(template_fit(x, seed) - h))
            if seed == 0:
                examples[gname] = (x, h, est, float(torch.dot(est, h) / (est.norm() * h.norm())))

    print(f"\nresidual SNR of (estimate - injected), injected SNR {args.snr:g}, {args.seeds} seeds:")
    for gname, d in stats.items():
        for k, v in d.items():
            if v:
                print(f"  {gname:16s} {k:18s} mean {np.mean(v):5.2f} (min {np.min(v):5.2f}, max {np.max(v):5.2f})")

    # figure: examples (seed 0) + summary bars
    fig = plt.figure(figsize=(15, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1], hspace=0.38, wspace=0.28)
    t = np.arange(N) / FS
    for i, (gname, zoom) in enumerate((("scattered light", (1.5, 6.5)), ("gengli blip", (3.85, 4.15)))):
        x, h, est, match = examples[gname]
        sel = (t >= zoom[0]) & (t <= zoom[1])
        lim = 1.15 * max(float(h[sel].abs().max()), float((est - h)[sel].abs().max()))
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t[sel], h.numpy()[sel], color=INK, lw=1.0, label="injected glitch")
        ax.plot(t[sel], est.numpy()[sel], color=DE_COLOR, lw=1.2, alpha=0.9, label="DeepExtractor")
        ax.set(xlim=zoom, ylim=(-lim, lim), ylabel="whitened strain", xlabel="time [s]")
        ax.set_title(f"{gname}: DeepExtractor extraction (match {match:.3f})", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9)
        ax.grid(color=GRID, lw=0.6)
        style(ax)
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t[sel], (est - h).numpy()[sel], color=DE_COLOR, lw=1.0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set(xlim=zoom, ylim=(-lim, lim), xlabel="time [s]", ylabel="error")
        ax.set_title(f"extracted − injected: residual SNR {residual_snr(est - h):.1f}", fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        style(ax)
        ax = fig.add_subplot(gs[i, 2])
        names = [k for k, v in stats[gname].items() if v]
        means = [np.mean(stats[gname][k]) for k in names]
        lo = [np.mean(stats[gname][k]) - np.min(stats[gname][k]) for k in names]
        hi = [np.max(stats[gname][k]) - np.mean(stats[gname][k]) for k in names]
        colors = [DE_COLOR if k == "DeepExtractor" else "#b9b8b3" for k in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, means, color=colors, height=0.6, xerr=[lo, hi], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for yy, m in zip(y, means):
            ax.text(m, yy + 0.36, f"{m:.1f}", va="center", ha="center", fontsize=8, color=INK2)
        ax.set_yticks(y, names, fontsize=9)
        ax.set(xlabel="residual glitch SNR (lower is better)", xlim=(0, args.snr * 1.05))
        ax.axvline(args.snr, color=INK2, lw=0.6, ls=":")
        ax.set_title(f"{gname}: mean over {args.seeds} seeds (range)", fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        style(ax)
    ck = "real-noise" if "real" in os.path.basename(args.checkpoint) else "bilby-noise"
    fig.suptitle(f"Pretrained DeepExtractor ({ck} checkpoint, {args.scaler} scaler) on 8 s of white noise, "
                 f"glitch SNR {args.snr:g}\n"
                 f" (dotted line = no subtraction; grey = oracle bounds / known-shape fit)", color=INK, fontsize=11)
    path = os.path.join(args.outdir, f"deepextractor_arch_blip{args.tag}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print("saved", path)


if __name__ == "__main__":
    main()
