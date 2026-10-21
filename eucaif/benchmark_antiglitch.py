"""
Benchmark the Antiglitch glitch model against the sine-Gaussian Blip SAGE uses now:

- ``sine_gaussian``      time domain, exp * sin on (n, L)              (analytic_glitches.py)
- ``antiglitch_glitch``  frequency domain, log-normal bump + one irfft (antiglitch.py)
- Antiglitch reference   its numpy formula (``fglitch_from_sample``), one glitch at a time

each alone and through ``GlitchPopulation.sample`` with the Blip class of
``sage_analytic_glitches.yaml`` and the same class with ``generator: antiglitch_glitch`` (ranges
from Antiglitch's O3 Blip fits). Defaults: 0.5 s glitches at 4096 Hz, as the Blip class.

Usage:

    python eucaif/benchmark_antiglitch.py                       # cpu, plus cuda if present
    python eucaif/benchmark_antiglitch.py --devices cuda --batches 1 1024 16384

Times are the best of ``--repeats`` runs after a warm-up, with device synchronisation.
No MPS: the analytic generators all work in float64, which MPS doesn't have.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sage.data.noise import GlitchPopulation, antiglitch_glitch, sine_gaussian

CONFIG = Path(__file__).resolve().parent / "sage_analytic_glitches.yaml"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def best_time(fn, device, repeats):
    fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t)
    return min(times)


def antiglitch_reference(n, sample_rate, duration, rng):
    """Antiglitch's numpy formula, one glitch per call as in its fits."""
    length = int(round(duration * sample_rate))
    freqs = np.fft.rfftfreq(length, 1.0 / sample_rate)
    out = np.empty((n, length))
    for i in range(n):
        f0, gbw, t0, phi = rng.uniform(10, 600), rng.uniform(0.25, 8), rng.uniform(0.4, 0.6) * duration, rng.uniform(0, 2 * np.pi)
        with np.errstate(divide="ignore"):
            fsignal = np.exp(-0.5 * gbw * (np.log(freqs) - np.log(f0)) ** 2)
        out[i] = np.fft.irfft(np.exp(1j * (phi - np.pi / 2)) * fsignal * np.exp(-2j * np.pi * t0 * freqs), length)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", nargs="+", default=None)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 64, 1024, 8192])
    p.add_argument("--sample-rate", type=float, default=4096.0)
    p.add_argument("--duration", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    devices = args.devices or ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    fs, dur = args.sample_rate, args.duration
    cfg = yaml.safe_load(CONFIG.read_text())
    blip = cfg["classes"]["Blip"]
    anti = {k: v for k, v in blip.items() if k not in ("generator", "q")}
    anti.update(generator="antiglitch_glitch", f0=[12.0, 260.0], gbw=[0.29, 4.2], log_uniform=["f0", "gbw"])
    pop = GlitchPopulation({"sample_rate": fs, "classes": {"Blip": blip, "BlipAntiglitch": anti}})

    print(f"{dur:g} s at {fs:g} Hz ({int(dur * fs)} samples); best of {args.repeats}, milliseconds")
    print(f"{'device':6s} {'batch':>6s} {'sine_gaussian':>14s} {'antiglitch':>11s} {'ratio':>6s} "
          f"{'pop Blip':>9s} {'pop Anti':>9s} {'numpy ref':>10s}")
    for name in devices:
        device = torch.device(name)
        dtype = torch.float64
        for b in args.batches:
            g = torch.Generator().manual_seed(0)
            t_sg = best_time(lambda: sine_gaussian(b, fs, duration=dur, f0=(50.0, 500.0), q=(2.0, 6.0),
                                                   generator=g, dtype=dtype, device=device), device, args.repeats)
            t_ag = best_time(lambda: antiglitch_glitch(b, fs, duration=dur, generator=g, dtype=dtype,
                                                       device=device), device, args.repeats)
            t_pb = best_time(lambda: pop.sample("Blip", b, g, dtype=dtype, device=device), device, args.repeats)
            t_pa = best_time(lambda: pop.sample("BlipAntiglitch", b, g, dtype=dtype, device=device), device,
                             args.repeats)
            ref = ""
            if device.type == "cpu" and b <= 1024:
                rng = np.random.default_rng(0)
                ref = f"{1e3 * best_time(lambda: antiglitch_reference(b, fs, dur, rng), device, 2):10.2f}"
            print(f"{name:6s} {b:6d} {1e3 * t_sg:14.2f} {1e3 * t_ag:11.2f} {t_ag / t_sg:6.2f} "
                  f"{1e3 * t_pb:9.2f} {1e3 * t_pa:9.2f} {ref:>10s}")


if __name__ == "__main__":
    main()
