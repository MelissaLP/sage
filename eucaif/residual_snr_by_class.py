"""
Residual glitch SNR per glitch class, for the three invertible representations of
``compare_glitch_extraction.py`` (DeepExtractor's STFT, QTAM, SAGE's tiles).

For every class of ``sage_analytic_glitches.yaml`` (the Antiglitch classes, the ArchEnemy arch and
McGAN's shapes), ``--n`` glitches are drawn from the population (its shape parameters; the SNR is
fixed to ``--snr`` so the classes compare), each injected into 8 s of white noise and extracted
with the oracle mask of ``compare_glitch_extraction.run`` (pixels where the glitch alone has at
least ``--kappa`` times the noise energy of its row). The residual SNR is the optimal SNR of
(extracted - injected) within the analysis band [10, 1000] Hz, where the injected glitch lives:
0 is perfect subtraction, ``--snr`` is no subtraction at all. The CSV also has the full-band value:
after masking, an inverse can put power below 10 Hz, outside every row, where its window
coverage is ~0 (QTAM's inverse has no cutoff there and can blow up; SAGE's has ``rcond``). A
high-pass, such as the whitening's, removes that, so it is not counted as glitch left over.

Usage, from the SAGE repo root (QTAM: ``pip install --no-deps qtam``):

    PYTHONPATH=. python3 eucaif/residual_snr_by_class.py --outdir figures/
    PYTHONPATH=. python3 eucaif/residual_snr_by_class.py --n 10 --snr 20 --classes Blip KoiFish

Writes ``residual_snr_by_class.png`` (all samples, per class and method),
``residual_snr_examples.png`` (one example per class) and ``residual_snr_by_class.csv``.
"""

import argparse
import contextlib
import csv
import io
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_glitch_extraction import (COLORS, DUR, FS, GRID, INK2, LABELS, N, SURFACE, Qtam, Sage,
                                       SingleQTransform, Stft, band, residual_snr, run, style)
from sage.data.noise import GlitchPopulation

CONFIG = Path(__file__).resolve().parent / "sage_analytic_glitches.yaml"


def methods_for_q(q, qt):
    return [Stft(q)] + ([Qtam(q)] if SingleQTransform is not None else []) + [Sage(q, qt)]


def strip_figure(res, classes, methods, snr, kappa, path):
    """Residual SNR of every sample: one column per class, the methods side by side."""
    fig, ax = plt.subplots(figsize=(1.35 * len(classes) + 2, 4.2), facecolor=SURFACE)
    width = 0.8 / len(methods)
    rng = np.random.default_rng(0)
    for j, m in enumerate(methods):
        for i, c in enumerate(classes):
            v = np.array(res[c][m])
            x0 = i + (j - (len(methods) - 1) / 2) * width
            ax.scatter(x0 + rng.uniform(-0.25, 0.25, len(v)) * width, v, s=16, color=COLORS[m], alpha=0.75,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3, label=LABELS[m] if i == 0 else None)
            ax.plot([x0 - 0.4 * width, x0 + 0.4 * width], [np.median(v)] * 2, color=COLORS[m], lw=2.2, zorder=4)
    ax.axhline(snr, color=INK2, lw=0.8, ls="--")
    ax.text(len(classes) - 0.5, snr, f"injected SNR {snr:g}", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(range(len(classes)), classes, rotation=25, ha="right")
    ax.set(ylabel="residual SNR in 10–1000 Hz", xlim=(-0.6, len(classes) - 0.4),
           ylim=(0, 1.08 * max(snr, max(max(res[c][m]) for c in classes for m in methods))))
    ax.set_title(f"Oracle-mask extraction, kappa = {kappa:g}, {len(res[classes[0]][methods[0]])} glitches per class "
                 f"(dots), median (bar); lower is better", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=len(methods), loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def examples_figure(examples, classes, methods, durations, path):
    """One sample per class: the injected glitch and, for each method, extracted - injected."""
    ncol = 4
    nrow = -(-len(classes) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.6 * nrow), facecolor=SURFACE, squeeze=False)
    t = np.arange(N) / FS - DUR / 2
    for ax, c in zip(axes.flat, classes):
        out = examples[c]
        half = max(0.06, 0.6 * durations[c] / 2) if durations[c] < 1.5 else durations[c] / 2 + 0.2
        sel = np.abs(t) <= half
        ax.plot(t[sel], out["h"].numpy()[sel], color="#b9b8b2", lw=1.6, label="injected glitch")
        for m in methods:
            r = out["methods"][m]
            dashed = m == "QTAM"                                 # drawn dashed on top: it tracks SAGE closely
            ax.plot(t[sel], band(r["est"] - out["h"]).numpy()[sel], color=COLORS[m], lw=1.1 if dashed else 0.9,
                    ls=(0, (3, 2)) if dashed else "-", zorder=4 if dashed else 3, label=f"{LABELS[m]} residual")
        txt = "  ".join(f"{m} {out['methods'][m]['res_snr']:.1f}" for m in methods)
        ax.set_title(f"{c} (Q = {out['q']:.0f})\nresidual SNR: {txt}", fontsize=8.5)
        ax.set_xlabel("time from glitch centre [s]", fontsize=8)
        ax.grid(color=GRID, lw=0.5)
        style(ax)
    for ax in list(axes.flat)[len(classes):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10, help="glitches per class")
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--kappa", type=float, default=1.0, help="oracle mask threshold")
    p.add_argument("--classes", nargs="+", help="default: every class of the config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--q", type=float, help="fixed Q (nearest plane) for every glitch; default: the best Q of each")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pop = GlitchPopulation(CONFIG, sample_rate=FS)
    classes = args.classes or list(pop.classes)
    res, examples, rows, methods = {}, {}, [], None
    for ci, c in enumerate(classes):
        glitches, params = pop.sample(c, args.n, torch.Generator().manual_seed(args.seed + 1000 * ci),
                                      dtype=torch.float64)
        res[c] = {}
        for i in range(args.n):
            with contextlib.redirect_stdout(io.StringIO()):          # run() prints one line per method
                out = run(c, glitches[i], args.snr, methods_for_q, [args.kappa], args.kappa, args.seed + i,
                          q=args.q)
            methods = list(out["methods"])
            for m in methods:
                r = out["methods"][m]
                r["res_snr_full"] = r["res_snr"]
                r["res_snr"] = residual_snr(band(r["est"] - out["h"]))
                res[c].setdefault(m, []).append(r["res_snr"])
                rows.append(dict(cls=c, sample=i, method=m, q=round(out["q"], 2), residual_snr=round(r["res_snr"], 3),
                                 residual_snr_full_band=round(r["res_snr_full"], 3),
                                 match=round(r["match"], 4), kept_pixels=round(r["kept"], 5),
                                 roundtrip_error=f"{r['roundtrip']:.2e}"))
            if i == 0:
                examples[c] = out
        print(f"{c:18s} " + "  ".join(f"{m} median {np.median(res[c][m]):5.2f} "
                                      f"[{min(res[c][m]):.2f}, {max(res[c][m]):.2f}]" for m in methods))

    with open(os.path.join(args.outdir, "residual_snr_by_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    strip_figure(res, classes, methods, args.snr, args.kappa, os.path.join(args.outdir, "residual_snr_by_class.png"))
    durations = {c: pop.classes[c]["duration"] for c in classes}
    examples_figure(examples, classes, methods, durations, os.path.join(args.outdir, "residual_snr_examples.png"))
    print("saved figures and csv in", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
