"""
Example spectrograms of every glitch model, in each transform of ``benchmark_qtransform.py``.

One glitch of each class (the analytic classes of ``sage_analytic_glitches.yaml`` and
``sage_analytic_glitches_dense.yaml``, plus a gengli blip) is injected at ``--snr`` into 8 s of
white noise at 4096 Hz, as in ``residual_snr_by_class.py``. Each row of the figure shows the time
series and the Q-scan image of gwpy (the reference), SAGE ``forward``, ml4gw ``QScan`` (bicubic),
QTAM ``QScanMulti`` and DeepExtractor's STFT magnitude, around the glitch. All Q-scans use the
benchmark's settings: Q in [4, 64], 10-1000 Hz, log frequencies, 257 rows. Each method picks its
own Q, which is printed in the panel. The panels of SAGE and ml4gw also print their largest
difference from gwpy's image, relative to its peak (the benchmark's ``scan`` check).

gwpy, SAGE and ml4gw share one colour scale per row (median-normalised energy, as gwpy). QTAM
(no median normalisation) and the STFT (magnitude) are scaled to their own peak.

Usage, from the SAGE repo root (ml4gw needs Python >= 3.11; QTAM: ``pip install --no-deps qtam``;
the gengli blip needs gengli, or ``--blip-file`` with an (n, L) .npy of gengli blips at 4096 Hz):

    PYTHONPATH=. python eucaif/example_glitch_transforms.py --outdir eucaif/figures/
    PYTHONPATH=. python eucaif/example_glitch_transforms.py --classes Blip ScatteredLight --snr 30

Writes ``glitch_transforms.png`` (and ``.pdf``).
"""

import argparse
import contextlib
import io
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from compare_glitch_extraction import ASD, DUR, F_HIGH, F_LOW, FS, GRID, INK2, N, SURFACE, band, style
from sage.data.noise import GlitchInjector, GlitchPopulation
from sage.dsp.qtransform import QTransform

try:
    from ml4gw.transforms import QScan
except Exception:  # ml4gw needs Python >= 3.11
    QScan = None
try:
    from qtam import QScanMulti, SingleQTransform   # compare_glitch_extraction put QTAM's folder on the path
except Exception:
    QScanMulti = None
from gwpy.timeseries import TimeSeries

HERE = Path(__file__).resolve().parent
CONFIGS = [HERE / "sage_analytic_glitches.yaml", HERE / "sage_analytic_glitches_dense.yaml"]
QRANGE, NF, NT = (4.0, 64.0), 257, 4096
STFT_KW = dict(n_fft=512, hop_length=32, win_length=64)
INK = "#0b0b0b"


def peak_error(out, ref):
    return float(np.abs(out - ref).max() / np.abs(ref).max())


def inject(glitch, snr, seed):
    """White noise + the glitch at ``snr``, centred; returns (x, h) as float64 (N,)."""
    noise = torch.randn(N, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    inj = GlitchInjector(FS, f_low=F_LOW, f_high=F_HIGH, lowpass_order=None)
    asd = torch.full((N // 2 + 1,), ASD, dtype=torch.float64)
    _, h = inj(torch.zeros(1, 1, N, dtype=torch.float64), glitch.view(1, 1, -1), asd, snr=snr, return_glitch=True)
    return noise + h[0, 0], h[0, 0]


def transforms(x):
    """name -> (image (n_freqs, n_times), freqs (n_freqs,), times (n_times,), Q or None)."""
    fr = (F_LOW, F_HIGH)
    t_img = np.arange(NT) * DUR / NT - DUR / 2
    f_log = np.logspace(math.log10(F_LOW), math.log10(F_HIGH), NF)
    out = {}
    spec = TimeSeries(x.numpy(), sample_rate=FS).q_transform(whiten=False, qrange=QRANGE, frange=fr, logf=True,
                                                             fres=NF, tres=DUR / NT)
    out["gwpy"] = (spec.value.T, spec.frequencies.value, t_img, getattr(spec, "q", None))

    qt = QTransform(FS, N, qrange=QRANGE, frange=fr, logf=True, fres=NF, tres=DUR / NT)
    img, q = qt(x.float()[None], return_q=True)
    out["SAGE"] = (img[0].T.double().numpy(), out["gwpy"][1], t_img, float(q[0]))

    if QScan is not None:
        scan = QScan(DUR, FS, (NF, NT), qrange=list(QRANGE), frange=list(fr), interpolation_method="bicubic")
        img = scan(x.float()[None, None])
        k = int(torch.argmax(torch.Tensor([t.get_max_energy() for t in scan.q_transforms])))
        out["ml4gw"] = (img.reshape(NF, NT).double().numpy(), f_log, t_img, float(scan.qs[k]))
    if QScanMulti is not None:
        scan = QScanMulti(DUR, FS, (NF, NT), qrange=list(QRANGE), frange=list(fr), logf=True)
        with contextlib.redirect_stdout(io.StringIO()):                  # QTAM prints aliasing reports
            res = scan(x.float()[None, None], num_time=NT)
        q = float(res[2][0][0][0])
        f = SingleQTransform(DUR, FS, q=q, frange=list(fr)).freqs        # QTAM keeps its own rows at that Q
        img = res[0][0][0].reshape(len(f), NT)
        out["QTAM"] = (img.double().numpy(), np.asarray(f.cpu(), dtype=float), t_img, q)

    s = torch.stft(x, window=torch.hann_window(64, dtype=torch.float64), return_complex=True, **STFT_KW).abs()
    f = np.fft.rfftfreq(STFT_KW["n_fft"], 1 / FS)
    keep = (f > 0) & (f <= F_HIGH + f[1])                                 # the axes clip to [F_LOW, F_HIGH]
    t = np.arange(s.shape[-1]) * STFT_KW["hop_length"] / FS - DUR / 2
    out["STFT"] = (s.numpy()[keep], f[keep], t, None)
    return out


def load_glitches(classes, blip_file, seed):
    """name -> (glitch (L,) float64, half-width of the plotted window in s)."""
    glitches = {}
    for path in CONFIGS:
        pop = GlitchPopulation(path, sample_rate=FS)
        for i, c in enumerate(pop.classes):
            if c in glitches or (classes and c not in classes):
                continue
            g, _ = pop.sample(c, 1, torch.Generator().manual_seed(seed + 1000 * i), dtype=torch.float64)
            d = pop.classes[c]["duration"]
            glitches[c] = (g[0], max(0.1, 0.3 * d) if d < 1.5 else d / 2 + 0.2)
    if not classes or "gengli blip" in classes:
        if blip_file:
            blip = np.load(blip_file)[0]
        else:
            import gengli
            # gengli squeezes its output: one glitch comes back as (L,), not (1, L)
            blip = np.atleast_2d(gengli.glitch_generator("L1").get_glitch(n_glitches=1, srate=FS, seed=3))[0]
        glitches["gengli blip"] = (torch.tensor(blip, dtype=torch.float64), 0.1)
    return glitches


def figure(rows, snr, path):
    cols = ["time series", "gwpy", "SAGE", "ml4gw", "QTAM", "STFT"]
    cols = [c for c in cols if c == "time series" or any(c in r["tf"] for r in rows.values())]
    title = {"gwpy": "gwpy q_transform", "SAGE": "SAGE forward", "ml4gw": "ml4gw QScan (bicubic)",
             "QTAM": "QTAM QScanMulti", "STFT": "DeepExtractor STFT |S|"}
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.75 * len(cols), 1.9 * len(rows)),
                             facecolor=SURFACE, squeeze=False, subplot_kw=dict(axes_class=Axes))
    for i, (name, r) in enumerate(rows.items()):
        half, tf = r["half"], r["tf"]
        ax = axes[i, 0]
        t = np.arange(N) / FS - DUR / 2
        sel = np.abs(t) <= half
        ax.plot(t[sel], band(r["x"]).numpy()[sel], color="#b9b8b2", lw=0.6, label="data (10-1000 Hz)")
        ax.plot(t[sel], band(r["h"]).numpy()[sel], color=INK, lw=0.9, label="injected glitch")
        ax.set_ylabel(name, fontsize=9, fontweight="bold")
        ax.grid(color=GRID, lw=0.5)
        style(ax)
        vmax = tf["gwpy"][0][:, np.abs(tf["gwpy"][2]) <= half].max()
        for j, c in enumerate(cols[1:], start=1):
            ax = axes[i, j]
            if c not in tf:
                ax.axis("off")
                continue
            img, f, tt, q = tf[c]
            cs = np.abs(tt) <= half
            vis = (f >= F_LOW) & (f <= F_HIGH)
            shared = c in ("gwpy", "SAGE", "ml4gw")
            ax.pcolormesh(tt[cs], f, img[:, cs], shading="nearest", cmap="viridis", vmin=0,
                          vmax=vmax if shared else img[vis][:, cs].max(), rasterized=True)
            ax.set_yscale("log")
            ax.grid(False)
            ax.set_ylim(F_LOW, F_HIGH)
            label = f"Q = {q:.1f}" if q is not None else ""
            if c in ("SAGE", "ml4gw"):
                label += f"\ngwpy err {peak_error(img, tf['gwpy'][0]):.0e}"
            if label:
                ax.text(0.03, 0.96, label, transform=ax.transAxes, fontsize=7, color="white", va="top")
            ax.tick_params(labelsize=7)
            if j > 1:
                ax.set_yticklabels([], minor=True)
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Hz", fontsize=7)
        for ax in axes[i]:
            ax.tick_params(labelsize=7)
            if i == len(rows) - 1:
                ax.set_xlabel("time from glitch centre [s]", fontsize=7)
    for j, c in enumerate(cols):
        axes[0, j].set_title(title.get(c, c), fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Every glitch model at SNR {snr:g} in white noise: gwpy, SAGE and ml4gw share a colour scale "
                 f"per row; QTAM and the STFT use their own", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--classes", nargs="+", help="default: every class of both configs and the gengli blip")
    p.add_argument("--blip-file", help="(n, L) .npy of gengli blips at 4096 Hz (default: run gengli)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = {}
    for k, (name, (glitch, half)) in enumerate(load_glitches(args.classes, args.blip_file, args.seed).items()):
        x, h = inject(glitch, args.snr, args.seed + k)
        rows[name] = dict(x=x, h=h, half=half, tf=transforms(x))
        tf = rows[name]["tf"]
        print(f"{name:16s} " + "  ".join(f"{c} Q={v[3]:.1f}" for c, v in tf.items() if v[3] is not None)
              + "".join(f"  {c} err {peak_error(tf[c][0], tf['gwpy'][0]):.1e}" for c in ("SAGE", "ml4gw") if c in tf))
    figure(rows, args.snr, os.path.join(args.outdir, "glitch_transforms"))
    print("saved", os.path.join(os.path.abspath(args.outdir), "glitch_transforms.png"))


if __name__ == "__main__":
    main()
