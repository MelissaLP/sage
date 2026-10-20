"""
Example: scattered-light arches (ArchEnemy model) injected into noise and seen in a Q-scan.

1. draw a few arches with ``scattered_light_arch`` (fringe frequency 20-80 Hz, period 1.8-5.5 s),
2. inject them into white Gaussian noise at a chosen SNR with ``GlitchInjector``,
3. compute the Q-scan with SAGE's ``QTransform`` (gwpy-exact) and plot waveform + Q-scan.

Usage, from the SAGE repo root:

    PYTHONPATH=. python3 eucaif/example_scattered_light.py                  # 3 arches, SNR 20
    PYTHONPATH=. python3 eucaif/example_scattered_light.py --n 4 --snr 30 --seed 1 --out arches.png

White noise keeps the example simple: its one-sided ASD is sqrt(2 / fs) for unit variance, and
no whitening is needed before the Q-scan. For coloured noise, inject with the noise's ASD and
whiten (``sage.dsp.whiten.FIRWhitening``) before the Q-scan.
"""

import argparse
import math

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sage.data.noise import GlitchInjector, scattered_light_arch
from sage.dsp.qtransform import QTransform


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3, help="number of examples")
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--sample-rate", type=float, default=2048.0)
    p.add_argument("--duration", type=float, default=8.0, help="seconds; longer than the longest arch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="scattered_light_example.png")
    args = p.parse_args()

    fs, dur = args.sample_rate, args.duration
    n = int(round(fs * dur))
    g = torch.Generator().manual_seed(args.seed)

    # 1. arches, centred in the window (default t0 = duration / 2)
    arch, par = scattered_light_arch(args.n, fs, duration=dur, generator=g, dtype=torch.float64,
                                     return_params=True)

    # 2. white noise (unit variance) and its ASD; inject at the requested SNR
    noise = torch.randn(args.n, 1, n, generator=g, dtype=torch.float64)
    asd = torch.full((n // 2 + 1,), math.sqrt(2.0 / fs), dtype=torch.float64)
    injector = GlitchInjector(fs, f_low=15.0)
    strain, glitch = injector(noise, arch.view(args.n, 1, n), asd, snr=args.snr, return_glitch=True)

    # 3. Q-scan (low frequencies, where arches live)
    qt = QTransform(fs, n, frange=(10.0, 200.0), logf=True, fres=200, tres=0.01)
    qscan, q = qt(strain, return_q=True)                     # (n, 1, times, freqs)

    t = torch.arange(n) / fs
    fig, axes = plt.subplots(args.n, 2, figsize=(12, 3 * args.n), squeeze=False)
    for i in range(args.n):
        f_f, T = float(par["fringe_frequency"][i]), float(par["timeperiod"][i])
        ax = axes[i, 0]
        ax.plot(t, strain[i, 0], lw=0.3, color="0.6", label="noise + arch")
        ax.plot(t, glitch[i, 0], lw=0.8, color="C3", label="arch")
        ax.set(xlabel="time [s]", ylabel="strain", title=f"f_fringe = {f_f:.1f} Hz, T = {T:.2f} s")
        ax.legend(loc="upper right", fontsize=8)
        ax = axes[i, 1]
        ax.pcolormesh(qt.times, qt.frequencies, qscan[i, 0].T, shading="auto", vmax=25)
        ax.set(yscale="log", xlabel="time [s]", ylabel="frequency [Hz]",
               title=f"Q-scan (Q = {float(q[i, 0]):.1f}), SNR {args.snr:g}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
