"""
Benchmark time-frequency transforms on batches of whitened strain, one task at a time, so every
library does the same job with the same settings (``--task``, see ``README_qtransform_benchmark.md``):

- ``scan``      Q-scan image: every Q-plane, median-normalised energies, the loudest plane,
                interpolated onto one image (gwpy ``q_transform``; default task)
- ``singleq``   the same image at one fixed Q (``--q``)
- ``tiles``     raw tiles at one fixed Q: no normalisation, no interpolation
- ``roundtrip`` tiles at one fixed Q and back to the time domain (invertible methods only)
- ``bestq``     only the choice of the loudest Q-plane, no image
- ``all``       all of the above

Libraries (gwpy, ml4gw and QTAM are optional):

- SAGE  ``sage.dsp.qtransform.QTransform``  (gwpy's tiling; ``forward``, ``tiles``/``inverse``, ``best_q``)
- ml4gw ``ml4gw.transforms``                (``QScan`` / ``SingleQTransform``; one Q per batch)
- QTAM  ``qtam``                            (invertible Q-transform with its own frequency rows,
                                             no median normalisation)
- DeepExtractor's STFT                      (``torch.stft`` / ``torch.istft``, scan and roundtrip)
- gwpy  ``TimeSeries.q_transform`` & co.    (reference: numpy, CPU only, one series at a time)

Defaults are DeepExtractor's setup: 2 s at 4096 Hz, and images are interpolated onto the same
257 x 257 grid as DeepExtractor_257's STFT (n_fft=512, win_length=64, hop_length=32).

Usage (needs Python >= 3.11 for ml4gw):

    python eucaif/benchmark_qtransform.py                                # scan, cpu (+ cuda/mps)
    python eucaif/benchmark_qtransform.py --task all --devices cpu cuda --outdir results/
    python eucaif/benchmark_qtransform.py --task roundtrip --q 22.6 --devices cpu mps
    python eucaif/benchmark_qtransform.py --task all --duration 8 --sample-rate 8192 --frange 10 2000 \\
        --batches 1 16 64 --skip spline --outdir results_2khz/
    python eucaif/benchmark_qtransform.py --plot-only results/qtransform_benchmark.json

With ``--outdir``, the results are saved to ``qtransform_benchmark.json`` and drawn as one figure
per device (``qtransform_benchmark_<device>.png`` and ``.pdf``): for every task, the time per
series at the largest batch, and each method's check.

Notes:
- Times are the best of ``--repeats`` runs after a warm-up, with device synchronisation; gwpy
  is timed on 8 series on the CPU and scaled to each batch size (it is drawn in every figure).
- The check, by task, on the device being timed: ``scan``/``singleq``: largest difference from
  gwpy's image, relative to its peak; ``tiles``: the same for the raw tile energies; ``bestq``:
  fraction of series whose Q matches gwpy's; ``roundtrip``: relative reconstruction error within
  [2 f_low, 0.8 f_high] (gwpy has no inverse). QTAM and the STFT are different transforms, so
  they have no gwpy check except for the round trip.
- ml4gw picks one Q for the whole batch; SAGE and gwpy pick one per series.
- QTAM's rows are its own (their number depends on Q), so its images are not on the common grid;
  its time axis is set to the same number of columns (``num_time``).
"""

import argparse
import contextlib
import gc
import io
import json
import os
import time
import warnings

import numpy as np
import torch

from sage.dsp.qtransform import QTransform

warnings.filterwarnings("ignore")

try:
    from ml4gw.transforms import QScan, SingleQTransform as Ml4gwSingleQ
except Exception:  # ml4gw needs Python >= 3.11
    QScan = Ml4gwSingleQ = None
try:
    # qtam 0.3.1 imports its bundled torch_spline_interpolation_1_0_0 as a top-level module
    # (and the "full" extra that would provide it is not on PyPI), so put the package's own
    # folder on the path first
    import importlib.util
    import sys
    sys.path.insert(0, os.path.dirname(importlib.util.find_spec("qtam").origin))
    from qtam import QScanMulti, SingleQTransform as QtamSingleQ
except Exception:
    QScanMulti = QtamSingleQ = None
try:
    from gwpy.signal.qtransform import QPlane, q_scan
    from gwpy.timeseries import TimeSeries
except Exception:
    TimeSeries = None

TASKS = ["scan", "singleq", "tiles", "roundtrip", "bestq"]
CHECK_LABEL = {"scan": "max |image - gwpy| / peak", "singleq": "max |image - gwpy| / peak",
               "tiles": "max |energy - gwpy| / peak", "roundtrip": "relative error in band",
               "bestq": "fraction of Qs = gwpy's"}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def free(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


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


def quiet(fn):
    """Call ``fn`` with its prints (QTAM's aliasing reports) silenced."""
    def f(*a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)
    return f


def band_error(x, y, fs, frange):
    """Relative error of y against x within [2 f_low, 0.8 f_high]."""
    f = torch.fft.rfftfreq(x.shape[-1], 1 / fs)
    m = (f >= 2 * frange[0]) & (f <= 0.8 * frange[1])
    X, Y = torch.fft.rfft(x.double())[..., m], torch.fft.rfft(y.double())[..., m]
    return float((X - Y).abs().norm() / X.abs().norm())


def peak_error(out, ref):
    """Largest |out - ref| per series relative to the series' peak, worst series; (B, ...) arrays."""
    axes = tuple(range(1, ref.ndim))
    return float(np.max(np.abs(out - ref).max(axis=axes) / np.abs(ref).max(axis=axes)))


# -- the methods of one task ----------------------------------------------------------------------

def methods_for(task, cfg):
    """name -> (build(device) -> fn(x), check(fn, x) -> float or None) for one task.

    ``check`` gets the method built on the device being timed and the check series (4, n) on
    that device; it returns the task's check value (see ``CHECK_LABEL``).
    """
    fs, dur, n, q, fr, qr = cfg["fs"], cfg["dur"], cfg["n"], cfg["q"], cfg["frange"], cfg["qrange"]
    nf, nt = cfg["nf"], cfg["nt"]
    qr_task = (q, q) if task in ("singleq", "tiles", "roundtrip") else qr
    sage_kw = dict(qrange=qr_task, frange=fr, logf=True, fres=nf, tres=dur / nt)
    window = torch.hann_window(cfg["win_length"])
    stft_kw = dict(n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], win_length=cfg["win_length"])
    xa = make_data(4, n, fs, seed=1)                       # the check series
    ts = [TimeSeries(xi.double().numpy(), sample_rate=fs) for xi in xa] if TimeSeries is not None else None
    methods = {}

    def qtam_lossless():
        t = QtamSingleQ(dur, fs, q=q, frange=list(fr))
        return int((2 * torch.floor(t.freqs.cpu() / (q / 11 ** 0.5) * dur) + 1).max())

    if task in ("scan", "singleq"):
        ref = (np.stack([t.q_transform(whiten=False, qrange=qr_task, frange=fr, logf=True, fres=nf,
                                       tres=dur / nt).value.T for t in ts]) if ts else None)

        def img_check(time_first):
            # SAGE returns (time, frequency) like gwpy's Spectrogram, ml4gw (frequency, time)
            def c(fn, x):
                if ref is None:
                    return None
                out = fn(x).detach().cpu()
                out = (out.transpose(-1, -2) if time_first else out).reshape(4, nf, nt)
                return peak_error(out.double().numpy(), ref)
            return c
        methods["SAGE forward"] = (lambda d: QTransform(fs, n, **sage_kw).to(d), img_check(True))
        if task == "scan":
            methods["SAGE forward + grad"] = (
                lambda d: (lambda m: lambda x: m(x.requires_grad_()).sum().backward())(
                    QTransform(fs, n, differentiable=True, **sage_kw).to(d)), None)
        for m in ("bicubic", "spline"):
            if task == "scan" and QScan is not None:
                build = (lambda m: lambda d: (lambda t: lambda x: t(x[:, None]))(
                    QScan(dur, fs, (nf, nt), qrange=list(qr), frange=list(fr), interpolation_method=m).to(d)))(m)
            elif task == "singleq" and Ml4gwSingleQ is not None:
                build = (lambda m: lambda d: (lambda t: lambda x: t(x[:, None]))(
                    Ml4gwSingleQ(dur, fs, (nf, nt), q=q, frange=list(fr), interpolation_method=m).to(d)))(m)
            else:
                continue
            methods[f"ml4gw {m}"] = (build, img_check(False))
        if task == "scan" and QScanMulti is not None:
            methods["QTAM QScanMulti"] = (lambda d: (lambda t: quiet(lambda x: t(x[:, None], num_time=nt)))(
                QScanMulti(dur, fs, (nf, nt), qrange=list(qr), frange=list(fr), logf=True, device=d).to(d)), None)
        if task == "singleq" and QtamSingleQ is not None:
            methods["QTAM energy"] = (lambda d: (lambda t: quiet(lambda x: t(
                x[:, None], energy_mode=True, phase_mode=False, num_time=nt)))(
                QtamSingleQ(dur, fs, q=q, frange=list(fr)).to(d)), None)
        if task == "scan":
            def stft(d):
                w = window.to(d)
                def f(x):
                    s = torch.stft(x, window=w, return_complex=True, **stft_kw)
                    return torch.stack([s.abs(), s.angle()], dim=1)      # DeepExtractor's apply_stft
                return f
            methods["DeepExtractor STFT"] = (stft, None)

    elif task == "tiles":
        ref = ([[np.asarray(e.value) for e in QPlane(q, fr, dur, fs).transform(t.fft(), norm=False, epoch=0.0).energies]
                for t in ts] if ts else None)                    # (series, row) -> energies at the row's own times

        def tiles_check(rows_of):
            def c(fn, x):
                if ref is None:
                    return None
                rows = rows_of(fn(x))
                return max(float(np.abs(rows[b][r] - ref[b][r]).max()) / max(np.abs(e).max() for e in ref[b])
                           for b in range(4) for r in range(len(ref[b])))
            return c

        def sage_rows(z):
            # |tiles|^2 on the shared grid; a row's own tiles are every (ncols / ntiles)-th column
            e = (z.abs() ** 2).cpu().double().numpy()
            return [[e[b, r, ::e.shape[-1] // len(ref[b][r])] for r in range(e.shape[1])] for b in range(4)]
        methods["SAGE tiles"] = (lambda d: (lambda t: lambda x: t.tiles(x, q))(
            QTransform(fs, n, qrange=qr_task, frange=fr).to(d)), tiles_check(sage_rows))
        if Ml4gwSingleQ is not None:
            def ml4gw_tiles(d):
                t = Ml4gwSingleQ(dur, fs, (nf, nt), q=q, frange=list(fr)).to(d)
                def f(x):
                    t.compute_qtiles(x[:, None], norm=None)
                    return t.qtiles                              # list over rows of (B, 1, ntiles) energies
                return f
            methods["ml4gw compute_qtiles"] = (ml4gw_tiles, tiles_check(
                lambda rows: [[r[b, 0].cpu().double().numpy() for r in rows] for b in range(4)]))
        if QtamSingleQ is not None:
            lossless = qtam_lossless()
            for label, nt_q in (("full", None), (f"num_time={lossless}", lossless)):
                methods[f"QTAM complex {label}"] = ((lambda nt_q: lambda d: (lambda t: quiet(lambda x: t(
                    x[:, None], polar_mode=False, complex_mode=True, num_time=nt_q)))(
                    QtamSingleQ(dur, fs, q=q, frange=list(fr)).to(d)))(nt_q), None)

    elif task == "roundtrip":
        rt_check = lambda fn, x: band_error(x.cpu(), fn(x).detach().cpu(), fs, fr)

        def sage_rt(d):
            t = QTransform(fs, n, qrange=qr_task, frange=fr).to(d)
            return lambda x: t.inverse(t.tiles(x, q), q)
        methods["SAGE tiles + inverse"] = (sage_rt, rt_check)
        if QtamSingleQ is not None:
            lossless = qtam_lossless()
            for label, nt_q in (("full", None), (f"num_time={lossless}", lossless)):
                def qtam_rt(d, nt_q=nt_q):
                    t = QtamSingleQ(dur, fs, q=q, frange=list(fr)).to(d)
                    kw = dict(polar_mode=False, complex_mode=True)
                    def f(x):
                        r = t.invert_qtransform(t(x[:, None], num_time=nt_q, **kw), **kw).reshape(x.shape[0], -1)[:, :n]
                        return r.real if torch.is_complex(r) else r
                    return quiet(f)
                methods[f"QTAM {label} + invert"] = (qtam_rt, rt_check)

        def stft_rt(d):
            w = window.to(d)
            def f(x):
                s = torch.stft(x, window=w, return_complex=True, **stft_kw)
                mp = torch.stack([s.abs(), s.angle()], dim=1)            # DeepExtractor's apply_stft
                z = torch.complex(mp[:, 0] * torch.cos(mp[:, 1]), mp[:, 0] * torch.sin(mp[:, 1]))
                return torch.istft(z, window=w, length=n, **stft_kw)     # + apply_istft
            return f
        methods["DeepExtractor STFT + iSTFT"] = (stft_rt, rt_check)

    elif task == "bestq":
        ref_q = np.array([q_scan(t, qrange=qr, frange=fr)[0].plane.q for t in ts]) if ts else None

        def q_check(fn, x):
            return None if ref_q is None else float(np.mean(np.isclose(np.asarray(fn(x), dtype=float), ref_q)))
        methods["SAGE best_q"] = (lambda d: (lambda t: lambda x: t.best_q(x).numpy())(
            QTransform(fs, n, qrange=qr, frange=fr).to(d)), q_check)
        if QScan is not None:
            def ml4gw_q(d):
                t = QScan(dur, fs, (nf, nt), qrange=list(qr), frange=list(fr)).to(d)
                def f(x):                                                # QScan.forward without the image
                    for tr in t.q_transforms:
                        tr.compute_qtiles(x[:, None], "median")
                    i = int(torch.argmax(torch.tensor([float(tr.get_max_energy()) for tr in t.q_transforms])))
                    return np.full(x.shape[0], t.qs[i])
                return f
            methods["ml4gw QScan planes"] = (ml4gw_q, q_check)
    return methods


def gwpy_reference(task, cfg):
    """(name, seconds per series) of gwpy's loop for one task, or None."""
    if TimeSeries is None or task == "roundtrip":
        return None
    fs, dur, n, q, fr, qr, nf, nt = (cfg[k] for k in ("fs", "dur", "n", "q", "frange", "qrange", "nf", "nt"))
    xs = [TimeSeries(xi, sample_rate=fs) for xi in make_data(8, n, fs).double().numpy()]
    if task in ("scan", "singleq"):
        qr_task = (q, q) if task == "singleq" else qr
        name, run = "gwpy q_transform", lambda t: t.q_transform(whiten=False, qrange=qr_task, frange=fr,
                                                                 logf=True, fres=nf, tres=dur / nt)
    elif task == "tiles":
        plane = QPlane(q, fr, dur, fs)
        name, run = "gwpy QPlane.transform", lambda t: plane.transform(t.fft(), norm=False, epoch=0.0)
    else:
        name, run = "gwpy q_scan", lambda t: q_scan(t, qrange=qr, frange=fr)
    run(xs[0])
    t0 = time.perf_counter()
    for t in xs:
        run(t)
    return name, (time.perf_counter() - t0) / len(xs)


# -- run ----------------------------------------------------------------------------------------------

def run_task(task, cfg, devices, batches, repeats, skip):
    """Time every method of one task on every device; prints a table, returns the records."""
    records = []
    print(f"\n== task {task}: {cfg['jobs'][task]}")
    header = f"{'method':32s} {'device':6s} " + " ".join(f"{'B=' + str(b):>10s}" for b in batches) \
        + f" {'ms/series':>10s}  {CHECK_LABEL[task]}"
    print(header + "\n" + "-" * len(header))
    for name, (build, chk) in methods_for(task, cfg).items():
        if any(k in name for k in skip):
            continue
        for d in devices:
            rec = dict(task=task, method=name, device=d, seconds={}, check=None, error=None)
            try:
                fn = build(d)
                for b in batches:
                    x = make_data(b, n=cfg["n"], fs=cfg["fs"]).to(d)
                    rec["seconds"][b] = bench(lambda: fn(x.clone()), d, repeats)
                    del x
                try:
                    rec["check"] = chk(fn, make_data(4, cfg["n"], cfg["fs"], seed=1).to(d)) if chk else None
                except Exception as ex:                           # keep the timing if only the check fails
                    print(f"   check failed: {type(ex).__name__}: {str(ex)[:80]}")
                c = "-" if rec["check"] is None else (f"{rec['check']:.2f}" if task == "bestq" else f"{rec['check']:.1e}")
                print(f"{name:32s} {d:6s} " + " ".join(f"{rec['seconds'][b]:9.4f}s" for b in batches)
                      + f" {1e3 * rec['seconds'][batches[-1]] / batches[-1]:10.3f}  {c}")
            except Exception as ex:
                rec["error"] = "out of memory" if "out of memory" in str(ex).lower() else f"{type(ex).__name__}"
                print(f"{name:32s} {d:6s} failed: {type(ex).__name__}: {str(ex)[:80]}")
            records.append(rec)
            fn = None
            free(d)
    if "gwpy" not in skip:
        ref = gwpy_reference(task, cfg)
        if ref is not None:
            name, per = ref
            records.append(dict(task=task, method=name, device="cpu", seconds={b: per * b for b in batches},
                                check="reference", error=None))
            print(f"{name + ' (loop)':32s} {'cpu':6s} " + " ".join(f"{per * b:9.4f}s" for b in batches)
                  + f" {1e3 * per:10.3f}  reference")
    return records


# -- figures ----------------------------------------------------------------------------------------

# reference palette, categorical slots 1-4 (STFT, QTAM, SAGE as in compare_glitch_extraction.py) + gray
COLORS = {"DeepExtractor": "#2a78d6", "QTAM": "#eb6834", "SAGE": "#1baf7a", "ml4gw": "#eda100", "gwpy": "#8a8984"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
TASK_TITLE = {"scan": "scan - Q-scan image (all planes, median, best Q)", "singleq": "singleq - image at Q = {q:g}",
              "tiles": "tiles - raw tiles at Q = {q:g}", "roundtrip": "roundtrip - tiles at Q = {q:g} and back",
              "bestq": "bestq - choice of the best Q only"}
CHECK_SHORT = {"scan": "gwpy err", "singleq": "gwpy err", "tiles": "gwpy err", "roundtrip": "recon. err",
               "bestq": "Q = gwpy"}


def plot_device(results, device, path):
    """One figure for one device: per task, ms per series at the largest batch and each method's check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    cfg, records = results["config"], results["records"]
    batch = str(max(int(b) for b in cfg["batches"]))
    tasks = [t for t in TASKS if any(r["task"] == t for r in records)]
    rows = {t: [r for r in records if r["task"] == t and (r["device"] == device or r["method"].startswith("gwpy"))]
            for t in tasks}
    heights = [max(len(rows[t]), 1) for t in tasks]
    fig = plt.figure(figsize=(8.3, 1.15 + 0.3 * sum(heights) + 0.55 * len(tasks)), facecolor=SURFACE)
    gs = fig.add_gridspec(len(tasks), 1, height_ratios=heights)
    # matplotlib's own Axes: importing gwpy replaces the default one
    axes = np.array([[fig.add_subplot(gs[i], axes_class=Axes)] for i in range(len(tasks))])
    times = [1e3 * r["seconds"][batch] / int(batch) for r in records if r["seconds"].get(batch)]
    lo, hi = 10 ** np.floor(np.log10(min(times))), 10 ** np.ceil(np.log10(max(times)) + 0.9)
    for ax, t in zip(axes[:, 0], tasks):
        ax.set_facecolor(SURFACE)
        names = []
        for i, r in enumerate(rows[t]):
            y = len(rows[t]) - 1 - i
            lib = r["method"].split()[0]
            label = r["method"] + (" (CPU)" if lib == "gwpy" and device != "cpu" else "")
            names.append((y, label))
            sec = r["seconds"].get(batch)
            if r["error"] or sec is None:
                msg = r["error"] or "-"
                ax.text(lo * 1.2, y, msg if msg in ("-", "out of memory") else f"fails ({msg.lstrip('_')})",
                        va="center", fontsize=7.5, color=MUTED)
                continue
            ms = 1e3 * sec / int(batch)
            ax.barh(y, ms, height=0.62, color=COLORS.get(lib, MUTED))
            c = r["check"]
            if c == "reference":
                c = "reference"
            elif c is None:
                c = ""
            else:
                c = f"{CHECK_SHORT[t]} {c:.0%}" if t == "bestq" else f"{CHECK_SHORT[t]} {c:.0e}"
            ax.text(ms * 1.12, y, f"{ms:.3g} ms" + (f"  |  {c}" if c else ""), va="center", fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, len(rows[t]) - 0.4)
        ax.set_yticks([y for y, _ in names], [l for _, l in names], fontsize=8)
        ax.tick_params(axis="y", length=0, colors=INK)
        ax.tick_params(axis="x", labelsize=7.5, colors=INK2)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_title(TASK_TITLE[t].format(q=cfg["q"]), loc="left", fontsize=9.5, weight="bold", color=INK, pad=4)
    axes[-1, 0].set_xlabel(f"milliseconds per series at batch {batch} (log scale)", fontsize=8.5, color=INK2)
    fig.suptitle(f"Q-transform benchmark on {device.upper()}: {cfg['dur']:g} s at {cfg['fs']:g} Hz, "
                 f"{cfg['frange'][0]:g}-{cfg['frange'][1]:g} Hz, image {cfg['nf']} x {cfg['nt']}",
                 x=0.02, ha="left", fontsize=11, weight="bold", color=INK)
    fig.text(0.02, 0.005, f"{results['machine']}; torch {results['torch']}; float32, best of {cfg['repeats']}. "
             "gwpy err: max |x - gwpy| / peak; recon. err: relative error in band; Q = gwpy: series whose Q "
             "matches gwpy's.", fontsize=6.5, color=MUTED, ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def machine():
    import platform
    name = platform.processor() or platform.machine()
    if torch.cuda.is_available():
        name += f", {torch.cuda.get_device_name(0)}"
    return f"{name}, {os.cpu_count()} CPUs, {torch.get_num_threads()} torch threads"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", nargs="+", default=["scan"], choices=TASKS + ["all"])
    p.add_argument("--q", type=float, default=11.31, help="Q of the fixed-Q tasks (singleq, tiles, roundtrip)")
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
    p.add_argument("--skip", nargs="+", default=[],
                   help="leave out methods whose name contains any of these (e.g. spline: ml4gw's spline "
                        "interpolation needs a lot of memory for long inputs; gwpy: the reference loop)")
    p.add_argument("--outdir", help="save the results (JSON) and one figure per device here")
    p.add_argument("--plot-only", metavar="JSON", help="only draw the figures of a saved results file")
    args = p.parse_args()

    if args.plot_only:
        results = json.load(open(args.plot_only))
        outdir = os.path.dirname(os.path.abspath(args.plot_only))
        for d in results["devices"]:
            plot_device(results, d, os.path.join(outdir, f"qtransform_benchmark_{d}"))
        print("figures in", outdir)
        return

    fs, dur = args.sample_rate, args.duration
    n = int(round(fs * dur))
    devices = args.devices or (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
                               + (["mps"] if torch.backends.mps.is_available() else []))
    tasks = TASKS if "all" in args.task else args.task
    cfg = dict(fs=fs, dur=dur, n=n, q=args.q, qrange=tuple(args.qrange), frange=tuple(args.frange),
               nf=args.n_fft // 2 + 1, nt=n // args.hop_length + 1,     # the STFT's grid (center=True)
               n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length,
               batches=args.batches, repeats=args.repeats)
    cfg["jobs"] = dict(scan="Q-scan image: all planes, median, loudest plane, interpolated",
                       singleq=f"image at Q = {args.q:g}: median, interpolated",
                       tiles=f"raw tiles at Q = {args.q:g}: no normalisation, no interpolation",
                       roundtrip=f"tiles at Q = {args.q:g} and back to the time domain",
                       bestq="choice of the loudest Q-plane only")

    print(f"{n} samples ({dur:g} s @ {fs:g} Hz) | image {cfg['nf']} x {cfg['nt']} | Q {cfg['qrange']}, "
          f"fixed Q {args.q:g} | f {cfg['frange']} Hz | devices {devices}")
    print(f"torch {torch.__version__} | {machine()}" + ("" if QScan else " | ml4gw not installed")
          + ("" if QScanMulti else " | qtam not installed") + ("" if TimeSeries else " | gwpy not installed"))
    if QtamSingleQ is not None and any(t in tasks for t in ("singleq", "tiles", "roundtrip")):
        print(f"QTAM at Q = {args.q:g}: {len(QtamSingleQ(dur, fs, q=args.q, frange=list(cfg['frange'])).freqs)} "
              "frequency rows (its own spacing)")

    records = []
    for t in tasks:
        records += run_task(t, cfg, devices, args.batches, args.repeats, args.skip)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        results = dict(config={k: v for k, v in cfg.items() if k != "jobs"}, devices=devices, torch=torch.__version__,
                       machine=machine(), records=records)
        path = os.path.join(args.outdir, "qtransform_benchmark.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=1)
        results = json.load(open(path))                    # same types as --plot-only (batch keys as str)
        for d in devices:
            plot_device(results, d, os.path.join(args.outdir, f"qtransform_benchmark_{d}"))
        print(f"\nsaved {path} and one figure per device in {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
