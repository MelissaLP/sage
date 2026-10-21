"""
Oracle glitch extraction for every glitch class SAGE can generate: residual glitch SNR in
QTAM, SAGE (gwpy tiling) and DeepExtractor's STFT.

Same test as ``compare_glitch_extraction.py`` (8 s of white noise at 4096 Hz, glitch band-limited to
[10, 1000] Hz, ideal binary mask keeping the pixels where the glitch alone has at least ``kappa``
times the mean noise energy of the row, inverse, residual SNR of extracted - true glitch; 0 is
perfect), run on ``--draws`` glitches of each class of ``sage_analytic_glitches.yaml`` (O3a classes
and McGAN's shapes), plus a gengli blip if gengli is installed. Every glitch is injected at the
same ``--snr`` so the classes are comparable. Q is the one SAGE's Q-scan picks for the noisy data
and is used by both QTAM and SAGE.

Usage, from the SAGE repo root (``pip install --no-deps qtam``):

    PYTHONPATH=. python3 eucaif/compare_glitch_classes.py --draws 20 --snr 20
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_glitch_extraction as cge  # noqa: E402
from sage.data.noise import GlitchInjector, GlitchPopulation  # noqa: E402
from sage.dsp.qtransform import QTransform  # noqa: E402

warnings.filterwarnings("ignore")
N, FS = cge.N, cge.FS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", default=os.path.join(os.path.dirname(__file__), "sage_analytic_glitches.yaml"))
    p.add_argument("--draws", type=int, default=20, help="glitches per class")
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--kappa", type=float, default=1.0, help="oracle mask threshold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-stft", action="store_true")
    p.add_argument("--csv", help="write one row per glitch and method")
    args = p.parse_args()
    if cge.SingleQTransform is None:
        sys.exit("qtam is not installed (pip install --no-deps qtam)")

    pop = GlitchPopulation(args.population, sample_rate=FS)
    gen = torch.Generator().manual_seed(args.seed)
    classes = {name: pop.sample(name, args.draws, gen, dtype=torch.float64)[0] for name in pop.classes}
    try:
        import gengli
        g = gengli.glitch_generator("L1")
        classes["gengli blip"] = torch.tensor(np.stack([np.asarray(g.get_glitch(n_glitches=1, srate=FS, seed=s))[0]
                                                        for s in range(args.draws)]), dtype=torch.float64)
    except Exception as ex:
        print(f"gengli blip skipped ({type(ex).__name__}: {ex})")

    qt = QTransform(FS, N, qrange=(4, 64), frange=(cge.F_LOW, cge.F_HIGH), norm=False, dtype=torch.float64)
    inj = GlitchInjector(FS, f_low=cge.F_LOW, f_high=cge.F_HIGH, lowpass_order=None)
    asd = torch.full((N // 2 + 1,), cge.ASD, dtype=torch.float64)
    floor_noise = torch.randn(3, N, generator=torch.Generator().manual_seed(10_000 + args.seed), dtype=torch.float64)
    cache = {}                                    # q -> [(method, noise floor)]

    def methods(q):
        if q not in cache:
            ms = ([] if args.no_stft else [cge.Stft(q)]) + [cge.Qtam(q), cge.Sage(q, qt)]
            cache[q] = [(m, torch.stack([m.coeffs(n).abs() ** 2 for n in floor_noise]).mean(dim=(0, -1))[:, None])
                        for m in ms]
        return cache[q]

    names = (["STFT"] if not args.no_stft else []) + ["QTAM", "SAGE"]
    rows = []
    for cname, glitches in classes.items():
        res = {m: [] for m in names}
        qs = []
        for i, glitch in enumerate(glitches):
            noise = torch.randn(N, generator=torch.Generator().manual_seed(args.seed * 100_003 + i), dtype=torch.float64)
            _, h = inj(torch.zeros(1, 1, N, dtype=torch.float64), glitch.view(1, 1, -1), asd, snr=args.snr,
                       return_glitch=True)
            h = h[0, 0]
            x = noise + h
            _, q = qt(x, return_q=True)
            q = float(q)
            qs.append(q)
            for m, floor in methods(q):
                mask = (m.coeffs(h).abs() ** 2) >= args.kappa * floor
                r = cge.residual_snr(m.inverse(m.coeffs(x) * mask) - h)
                res[m.name].append(r)
                rows.append((cname, i, q, m.name, r))
        line = " | ".join(f"{m} {np.median(res[m]):5.2f} [{np.percentile(res[m], 16):5.2f}, "
                          f"{np.percentile(res[m], 84):5.2f}]" for m in names)
        print(f"{cname:16s} Q {np.median(qs):5.1f} | {line}", flush=True)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("class,draw,q,method,residual_snr\n")
            f.writelines(f"{c},{i},{q:.3f},{m},{r:.4f}\n" for c, i, q, m, r in rows)


if __name__ == "__main__":
    main()
