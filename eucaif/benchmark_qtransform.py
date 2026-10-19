"""
Benchmark time-frequency transforms on batches of whitened strain:

- SAGE  ``sage.dsp.qtransform.QTransform``      (gwpy-exact Q-scan, Q chosen per series)
- ml4gw ``ml4gw.transforms.QScan``               (bicubic default and spline; one Q per batch)
- DeepExtractor's STFT                           (``torch.stft``, magnitude + phase, and the
                                                  ``torch.istft`` back to the time domain)
- gwpy ``TimeSeries.q_transform``                (reference, numpy loop, CPU only)

Defaults are DeepExtractor's setup: 2 s at 4096 Hz, and the Q-scans are interpolated onto the
same 257 x 257 grid as DeepExtractor_257's STFT (n_fft=512, win_length=64, hop_length=32), so
every method returns an image of the same size.

Usage (needs Python >= 3.11 for ml4gw; gwpy and ml4gw are optional):

    python eucaif/benchmark_qtransform.py                       # cpu, plus cuda/mps if present
    python eucaif/benchmark_qtransform.py --devices cuda --batches 1 64 256 1024
    python eucaif/benchmark_qtransform.py --duration 8 --sample-rate 2048

Notes:
- Times are the best of ``--repeats`` runs after a warm-up, with device synchronisation.
- "acc" is the largest difference from gwpy over a few series, relative to the peak (Q-scans
  only, same grid; the STFT is a different transform, so it has no gwpy reference).
- ml4gw picks one Q for the whole batch; SAGE and gwpy pick one per series.
"""

import argparse
import time
import warnings

import numpy as np
import torch

from sage.dsp.qtransform import QTransform

warnings.filterwarnings("ignore")

try:
    from ml4gw.transforms import QScan
except Exception:  # ml4gw needs Python >= 3.11
    QScan = None
try:
    from gwpy.timeseries import TimeSeries
except Exception:
    TimeSeries = None


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench(fn, device, repeats):
    fn()
    sync(device)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


def make_data(batch, n, fs, seed=0):
    """Whitened-like noise, a sine-Gaussian glitch in every other series."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n, generator=g)
    t = torch.arange(n) / fs
    for i in range(0, batch, 2):
        f0 = 40 + 400 * torch.rand(1, generator=g).item()
        tau = 0.005 + 0.1 * torch.rand(1, generator=g).item()
        t0 = n / fs / 2
        x[i] += 6 * torch.sin(2 * np.pi * f0 * (t - t0)) * torch.exp(-((t - t0) / tau) ** 2)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-rate", type=float, default=4096.0)
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    p.add_argument("--devices", nargs="+", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--qrange", type=float, nargs=2, default=(4, 64))
    p.add_argument("--frange", type=float, nargs=2, default=(20, 1000))
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--win-length", type=int, default=64)
    p.add_argument("--hop-length", type=int, default=32)
    args = p.parse_args()

    fs, dur = args.sample_rate, args.duration
    n = int(round(fs * dur))
    devices = args.devices or (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
                               + (["mps"] if torch.backends.mps.is_available() else []))
    n_frames = n // args.hop_length + 1                     # torch.stft, center=True
    n_freq_stft = args.n_fft // 2 + 1
    nf, nt = n_freq_stft, n_frames                          # same image size for the Q-scans
    qr, fr = tuple(args.qrange), tuple(args.frange)

    sage_kw = dict(qrange=qr, frange=fr, logf=True, fres=nf, tres=dur / nt)
    t0 = time.perf_counter()
    QTransform(fs, n, **sage_kw)
    t_init_sage = time.perf_counter() - t0
    t0 = time.perf_counter()
    if QScan is not None:
        QScan(dur, fs, (nf, nt), qrange=list(qr), frange=list(fr))
    t_init_ml4gw = time.perf_counter() - t0
    window = torch.hann_window(args.win_length)

    print(f"{n} samples ({dur:g} s @ {fs:g} Hz) | image {nf} x {nt} | Q {qr} | f {fr} Hz | "
          f"STFT n_fft={args.n_fft} win={args.win_length} hop={args.hop_length}")
    print(f"torch {torch.__version__} | construction: SAGE {t_init_sage:.2f} s, ml4gw {t_init_ml4gw:.2f} s"
          f"{'' if QScan else ' | ml4gw not installed'}{'' if TimeSeries else ' | gwpy not installed'}\n")

    # accuracy against gwpy on the common grid (CPU, 4 series)
    acc = {}
    if TimeSeries is not None:
        xa = make_data(4, n, fs, seed=1).double()
        ref = np.stack([TimeSeries(xi.numpy(), sample_rate=fs).q_transform(
            whiten=False, qrange=qr, frange=fr, logf=True, fres=nf, tres=dur / nt).value.T for xi in xa])
        rel = lambda o: float(np.max(np.abs(o - ref).max(axis=(1, 2)) / ref.max(axis=(1, 2))))
        acc["SAGE"] = rel(QTransform(fs, n, **sage_kw)(xa.float()).transpose(-1, -2).double().numpy())
        if QScan is not None:
            for m in ("bicubic", "spline"):
                qs = QScan(dur, fs, (nf, nt), qrange=list(qr), frange=list(fr), interpolation_method=m)
                acc[f"ml4gw {m}"] = rel(qs(xa.float()[:, None]).reshape(4, nf, nt).double().numpy())

    methods = {}
    methods["SAGE QTransform"] = lambda d: (lambda m: lambda x: m(x))(QTransform(fs, n, **sage_kw).to(d))
    methods["SAGE QTransform + grad"] = lambda d: (lambda m: lambda x: m(x.requires_grad_()).sum().backward())(
        QTransform(fs, n, differentiable=True, **sage_kw).to(d))
    if QScan is not None:
        for m in ("bicubic", "spline"):
            methods[f"ml4gw QScan {m}"] = (lambda m: lambda d: (lambda q: lambda x: q(x[:, None]))(
                QScan(dur, fs, (nf, nt), qrange=list(qr), frange=list(fr), interpolation_method=m).to(d)))(m)
    stft_kw = dict(n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)

    def stft(d):
        w = window.to(d)
        def f(x):
            s = torch.stft(x, window=w, return_complex=True, **stft_kw)
            return torch.stack([s.abs(), s.angle()], dim=1)          # DeepExtractor's apply_stft
        return f

    def stft_istft(d):
        w, fwd = window.to(d), stft(d)
        def f(x):
            mp = fwd(x)                                              # + DeepExtractor's apply_istft
            z = torch.complex(mp[:, 0] * torch.cos(mp[:, 1]), mp[:, 0] * torch.sin(mp[:, 1]))
            return torch.istft(z, window=w, **stft_kw)
        return f
    methods["DeepExtractor STFT"] = stft
    methods["DeepExtractor STFT + iSTFT"] = stft_istft

    header = f"{'method':28s} {'device':6s} " + " ".join(f"{'B=' + str(b):>10s}" for b in args.batches) \
        + f" {'ms/series':>10s} {'acc vs gwpy':>12s}"
    print(header + "\n" + "-" * len(header))
    for name, build in methods.items():
        for d in devices:
            row = []
            try:
                fn = build(d)
                for b in args.batches:
                    x = make_data(b, n, fs).to(d)
                    row.append(bench(lambda: fn(x.clone()), d, args.repeats))
                per = 1e3 * row[-1] / args.batches[-1]
                a = next((v for k, v in acc.items() if name.startswith(k.replace("ml4gw ", "ml4gw QScan "))
                          or (k == "SAGE" and name.startswith("SAGE"))), None)
                print(f"{name:28s} {d:6s} " + " ".join(f"{t:9.4f}s" for t in row)
                      + f" {per:10.3f} {('%.1e' % a) if a is not None else '-':>12s}")
            except Exception as ex:
                print(f"{name:28s} {d:6s} failed: {type(ex).__name__}: {str(ex)[:80]}")
    if TimeSeries is not None:
        x = make_data(8, n, fs).double().numpy()
        t0 = time.perf_counter()
        for xi in x:
            TimeSeries(xi, sample_rate=fs).q_transform(whiten=False, qrange=qr, frange=fr,
                                                       logf=True, fres=nf, tres=dur / nt)
        per = (time.perf_counter() - t0) / 8
        print(f"{'gwpy q_transform (loop)':28s} {'cpu':6s} "
              + " ".join(f"{per * b:9.4f}s" for b in args.batches) + f" {1e3 * per:10.3f} {'reference':>12s}")


if __name__ == "__main__":
    main()
