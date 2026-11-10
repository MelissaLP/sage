"""
Benchmark time-frequency transforms on batches of whitened strain, one task at a
time, so every library does the same job with the same settings (``--task``, see
``README.md`` in this folder):

- ``scan``      Q-scan image: every Q-plane, median-normalised energies, the loudest
                plane, interpolated onto one image (gwpy ``q_transform``; default)
- ``singleq``   the same image at one fixed Q (``--q``)
- ``tiles``     raw tiles at one fixed Q: no normalisation, no interpolation
- ``roundtrip`` tiles at one fixed Q and back to the time domain (invertible
                methods only)
- ``bestq``     only the choice of the loudest Q-plane, no image
- ``all``       all of the above

Libraries (gwpy, ml4gw and QTAM are optional):

- SAGE  ``sage.dsp.qtransform.QTransform`` (gwpy's tiling; ``forward``,
  ``tiles``/``inverse``, ``best_q``)
- ml4gw ``ml4gw.transforms`` (``QScan`` / ``SingleQTransform``; one Q per batch)
- QTAM  ``qtam`` (invertible Q-transform with its own frequency rows, no median
  normalisation)
- DeepExtractor's STFT (``torch.stft`` / ``torch.istft``, scan and roundtrip)
- gwpy  ``TimeSeries.q_transform`` & co. (reference: numpy, CPU only, one series
  at a time)

Defaults are DeepExtractor's setup: 2 s at 4096 Hz, and images are interpolated
onto the same 257 x 257 grid as DeepExtractor_257's STFT (n_fft=512, win_length=64,
hop_length=32).

Usage (needs Python >= 3.11 for ml4gw):

    python eucaif/benchmark/benchmark_qtransform.py          # scan, cpu (+ cuda/mps)
    python eucaif/benchmark/benchmark_qtransform.py --task all --devices cpu cuda \\
        --outdir results/
    python eucaif/benchmark/benchmark_qtransform.py --task roundtrip --q 22.6 \\
        --devices cpu mps
    python eucaif/benchmark/benchmark_qtransform.py --task all --duration 8 \\
        --sample-rate 8192 --frange 10 2000 --batches 1 16 64 --skip spline \\
        --outdir results_2khz/
    python eucaif/benchmark/benchmark_qtransform.py \\
        --plot-only results/qtransform_benchmark.json

With ``--outdir``, the results are saved to ``qtransform_benchmark.json`` and drawn
as one figure per device (``qtransform_benchmark_<device>.png`` and ``.pdf``): for
every task, the time per series at the largest batch, and each method's check.

Notes:

- Times are the best of ``--repeats`` runs after a warm-up, with device
  synchronisation; gwpy is timed on 8 series on the CPU and scaled to each batch
  size (it is drawn in every figure).
- The check, by task, on the device being timed: ``scan``/``singleq``: largest
  difference from gwpy's image, relative to its peak; ``tiles``: the same for the
  raw tile energies; ``bestq``: fraction of series whose Q matches gwpy's;
  ``roundtrip``: relative reconstruction error within [2 f_low, 0.8 f_high] (gwpy
  has no inverse). QTAM and the STFT are different transforms, so they have no
  gwpy check except for the round trip.
- ml4gw picks one Q for the whole batch; SAGE and gwpy pick one per series.
- QTAM's rows are its own (their number depends on Q), so its images are not on
  the common grid; its time axis is set to the same number of columns
  (``num_time``).
"""

import argparse
import contextlib
import dataclasses
import gc
import importlib.util
import io
import json
import os
import platform
import sys
import time
import warnings
from functools import partial
from typing import Callable, Optional

import numpy as np
import torch

from sage.dsp.qtransform import QTransform

warnings.filterwarnings("ignore")

TASKS = ["scan", "singleq", "tiles", "roundtrip", "bestq"]
FIXED_Q_TASKS = ("singleq", "tiles", "roundtrip")
TASK_JOBS = {
    "scan": "Q-scan image: all planes, median, loudest plane, interpolated",
    "singleq": "image at Q = {q:g}: median, interpolated",
    "tiles": "raw tiles at Q = {q:g}: no normalisation, no interpolation",
    "roundtrip": "tiles at Q = {q:g} and back to the time domain",
    "bestq": "choice of the loudest Q-plane only",
}
CHECK_LABEL = {
    "scan": "max |image - gwpy| / peak",
    "singleq": "max |image - gwpy| / peak",
    "tiles": "max |energy - gwpy| / peak",
    "roundtrip": "relative error in band",
    "bestq": "fraction of Qs = gwpy's",
}
N_CHECK = 4          # series used for the checks (seed 1)
N_GWPY_TIMING = 8    # series gwpy is timed on (seed 0)


# -- optional libraries -----------------------------------------------------------

MISSING = {}         # library -> why it could not be imported (printed at start)


def _import_ml4gw():
    from ml4gw.transforms import QScan, SingleQTransform
    return QScan, SingleQTransform


def _import_qtam():
    # qtam 0.3.1 imports its bundled torch_spline_interpolation_1_0_0 as a
    # top-level module (and the "full" extra that would provide it is not on
    # PyPI), so its own folder goes on the path first
    spec = importlib.util.find_spec("qtam")
    if spec is None:
        raise ImportError("qtam is not installed")
    sys.path.insert(0, os.path.dirname(spec.origin))
    from qtam import QScanMulti, SingleQTransform
    return QScanMulti, SingleQTransform


def _import_gwpy():
    from gwpy.signal.qtransform import QPlane, q_scan
    from gwpy.timeseries import TimeSeries
    return QPlane, q_scan, TimeSeries


def _optional(library, importer, n):
    """``importer()``, or ``n`` Nones if the library is unavailable."""
    try:
        return importer()
    except Exception as ex:      # e.g. ml4gw needs Python >= 3.11
        MISSING[library] = f"{type(ex).__name__}: {ex}"
        return (None,) * n


QScan, Ml4gwSingleQ = _optional("ml4gw", _import_ml4gw, 2)
QScanMulti, QtamSingleQ = _optional("qtam", _import_qtam, 2)
QPlane, q_scan, TimeSeries = _optional("gwpy", _import_gwpy, 3)


# -- configuration ----------------------------------------------------------------

@dataclasses.dataclass
class Config:
    """Benchmark settings; saved as the ``config`` of the results file."""

    fs: float
    dur: float
    n: int
    q: float
    qrange: tuple
    frange: tuple
    nf: int              # image grid: the STFT's (n_fft // 2 + 1) x (n // hop + 1)
    nt: int
    n_fft: int
    win_length: int
    hop_length: int
    batches: list
    repeats: int

    def task_qrange(self, task):
        return (self.q, self.q) if task in FIXED_Q_TASKS else self.qrange

    @property
    def tres(self):
        return self.dur / self.nt

    @property
    def stft_kwargs(self):
        return dict(n_fft=self.n_fft, hop_length=self.hop_length,
                    win_length=self.win_length)

    def tiling_frange(self, task):
        """The frequency range the Q tiling actually covers (as gwpy's)."""
        qt = QTransform(self.fs, self.n, qrange=self.task_qrange(task),
                        frange=self.frange)
        return qt.frange


@dataclasses.dataclass
class Method:
    """One way of doing a task: ``build(device) -> fn(x)`` and an optional
    ``check(fn, x) -> float`` against the reference (see ``CHECK_LABEL``)."""

    name: str
    build: Callable
    check: Optional[Callable] = None


# -- helpers ----------------------------------------------------------------------

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
    """Best time of ``repeats`` calls of ``fn`` after one warm-up call."""
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
    t0 = n / fs / 2
    for i in range(0, batch, 2):
        f0 = 40 + 400 * torch.rand(1, generator=g).item()
        tau = 0.005 + 0.1 * torch.rand(1, generator=g).item()
        envelope = torch.exp(-((t - t0) / tau) ** 2)
        x[i] += 6 * torch.sin(2 * np.pi * f0 * (t - t0)) * envelope
    return x


def quiet(fn):
    """``fn`` with its prints (QTAM's aliasing reports) silenced."""
    def silenced(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)
    return silenced


def band_error(x, y, fs, frange):
    """Relative error of ``y`` against ``x`` within [2 f_low, 0.8 f_high]."""
    f = torch.fft.rfftfreq(x.shape[-1], 1 / fs)
    band = (f >= 2 * frange[0]) & (f <= 0.8 * frange[1])
    X = torch.fft.rfft(x.double())[..., band]
    Y = torch.fft.rfft(y.double())[..., band]
    return float((X - Y).abs().norm() / X.abs().norm())


def peak_error(out, ref):
    """Largest |out - ref| per series relative to the series' peak, worst series."""
    axes = tuple(range(1, ref.ndim))
    errors = np.abs(out - ref).max(axis=axes) / np.abs(ref).max(axis=axes)
    return float(np.max(errors))


def qtam_lossless_num_time(cfg):
    """QTAM's number of time samples that keeps every tile of the top row."""
    t = QtamSingleQ(cfg.dur, cfg.fs, q=cfg.q, frange=list(cfg.frange))
    return int((2 * torch.floor(t.freqs.cpu() / (cfg.q / 11 ** 0.5) * cfg.dur)
                + 1).max())


# -- builders: device -> fn(x) ----------------------------------------------------

def sage_image(cfg, task, device, differentiable=False):
    qt = QTransform(cfg.fs, cfg.n, qrange=cfg.task_qrange(task), frange=cfg.frange,
                    logf=True, fres=cfg.nf, tres=cfg.tres,
                    differentiable=differentiable).to(device)
    if not differentiable:
        return qt

    def forward_backward(x):
        qt(x.requires_grad_()).sum().backward()
    return forward_backward


def ml4gw_image(cfg, task, interpolation, device):
    shape = (cfg.nf, cfg.nt)
    if task == "scan":
        t = QScan(cfg.dur, cfg.fs, shape, qrange=list(cfg.qrange),
                  frange=list(cfg.frange), interpolation_method=interpolation)
    else:
        t = Ml4gwSingleQ(cfg.dur, cfg.fs, shape, q=cfg.q, frange=list(cfg.frange),
                         interpolation_method=interpolation)
    t = t.to(device)
    return lambda x: t(x[:, None])


def qtam_scan(cfg, device):
    t = QScanMulti(cfg.dur, cfg.fs, (cfg.nf, cfg.nt), qrange=list(cfg.qrange),
                   frange=list(cfg.frange), logf=True, device=device).to(device)
    return quiet(lambda x: t(x[:, None], num_time=cfg.nt))


def qtam_energy(cfg, device):
    t = QtamSingleQ(cfg.dur, cfg.fs, q=cfg.q, frange=list(cfg.frange)).to(device)
    return quiet(lambda x: t(x[:, None], energy_mode=True, phase_mode=False,
                             num_time=cfg.nt))


def stft_magnitude_phase(x, window, cfg):
    """DeepExtractor's ``apply_stft``: (B, 2, freq, time) magnitude and phase."""
    s = torch.stft(x, window=window, return_complex=True, **cfg.stft_kwargs)
    return torch.stack([s.abs(), s.angle()], dim=1)


def stft_scan(cfg, device):
    window = torch.hann_window(cfg.win_length).to(device)
    return partial(stft_magnitude_phase, window=window, cfg=cfg)


def stft_roundtrip(cfg, device):
    window = torch.hann_window(cfg.win_length).to(device)

    def roundtrip(x):            # apply_stft, then apply_istft
        mp = stft_magnitude_phase(x, window, cfg)
        z = torch.polar(mp[:, 0], mp[:, 1])
        return torch.istft(z, window=window, length=cfg.n, **cfg.stft_kwargs)
    return roundtrip


def sage_tiles(cfg, device):
    qt = QTransform(cfg.fs, cfg.n, qrange=(cfg.q, cfg.q), frange=cfg.frange)
    qt = qt.to(device)
    return lambda x: qt.tiles(x, cfg.q)


def ml4gw_tiles(cfg, device):
    t = Ml4gwSingleQ(cfg.dur, cfg.fs, (cfg.nf, cfg.nt), q=cfg.q,
                     frange=list(cfg.frange)).to(device)

    def tiles(x):                # list over rows of (B, 1, ntiles) energies
        t.compute_qtiles(x[:, None], norm=None)
        return t.qtiles
    return tiles


def qtam_tiles(cfg, num_time, device):
    t = QtamSingleQ(cfg.dur, cfg.fs, q=cfg.q, frange=list(cfg.frange)).to(device)
    return quiet(lambda x: t(x[:, None], polar_mode=False, complex_mode=True,
                             num_time=num_time))


def sage_roundtrip(cfg, device):
    qt = QTransform(cfg.fs, cfg.n, qrange=(cfg.q, cfg.q), frange=cfg.frange)
    qt = qt.to(device)
    return lambda x: qt.inverse(qt.tiles(x, cfg.q), cfg.q)


def qtam_roundtrip(cfg, num_time, device):
    t = QtamSingleQ(cfg.dur, cfg.fs, q=cfg.q, frange=list(cfg.frange)).to(device)
    mode = dict(polar_mode=False, complex_mode=True)

    def roundtrip(x):
        tiles = t(x[:, None], num_time=num_time, **mode)
        r = t.invert_qtransform(tiles, **mode).reshape(x.shape[0], -1)[:, :cfg.n]
        return r.real if torch.is_complex(r) else r
    return quiet(roundtrip)


def sage_best_q(cfg, device):
    qt = QTransform(cfg.fs, cfg.n, qrange=cfg.qrange, frange=cfg.frange)
    qt = qt.to(device)
    return lambda x: qt.best_q(x).numpy()


def ml4gw_best_q(cfg, device):
    t = QScan(cfg.dur, cfg.fs, (cfg.nf, cfg.nt), qrange=list(cfg.qrange),
              frange=list(cfg.frange)).to(device)

    def best_q(x):               # QScan.forward without the image
        for plane in t.q_transforms:
            plane.compute_qtiles(x[:, None], "median")
        energies = [float(plane.get_max_energy()) for plane in t.q_transforms]
        return np.full(x.shape[0], t.qs[int(np.argmax(energies))])
    return best_q


# -- checks against gwpy: (fn, x) -> float ----------------------------------------

def check_image(fn, x, reference, cfg, time_first):
    # SAGE returns (time, frequency) as gwpy's Spectrogram, ml4gw (frequency, time)
    out = fn(x).detach().cpu()
    if time_first:
        out = out.transpose(-1, -2)
    out = out.reshape(N_CHECK, cfg.nf, cfg.nt)
    return peak_error(out.double().numpy(), reference)


def sage_tile_rows(z, reference):
    """|tiles|^2 on SAGE's shared grid, as each row's own tiles (every
    (ncols / ntiles)-th column)."""
    e = (z.abs() ** 2).cpu().double().numpy()
    return [[e[b, r, ::e.shape[-1] // len(reference[b][r])]
             for r in range(e.shape[1])] for b in range(N_CHECK)]


def ml4gw_tile_rows(rows, reference):
    return [[r[b, 0].cpu().double().numpy() for r in rows] for b in range(N_CHECK)]


def check_tiles(fn, x, reference, rows_of):
    rows = rows_of(fn(x), reference)
    worst = 0.0
    for b, ref_rows in enumerate(reference):
        peak = max(np.abs(e).max() for e in ref_rows)
        for r, ref in enumerate(ref_rows):
            worst = max(worst, float(np.abs(rows[b][r] - ref).max()) / peak)
    return worst


def check_roundtrip(fn, x, cfg):
    return band_error(x.cpu(), fn(x).detach().cpu(), cfg.fs, cfg.frange)


def check_best_q(fn, x, reference):
    q = np.asarray(fn(x), dtype=float)
    return float(np.mean(np.isclose(q, reference)))


def gwpy_series(n_series, cfg, seed):
    data = make_data(n_series, cfg.n, cfg.fs, seed=seed).double().numpy()
    return [TimeSeries(x, sample_rate=cfg.fs) for x in data]


def gwpy_image(series, cfg, task):
    return series.q_transform(whiten=False, qrange=cfg.task_qrange(task),
                              frange=cfg.frange, logf=True, fres=cfg.nf,
                              tres=cfg.tres)


def gwpy_tiles(series, cfg):
    """gwpy's raw tile energies, on the frequency range of the tiling."""
    plane = QPlane(cfg.q, cfg.tiling_frange("tiles"), cfg.dur, cfg.fs)
    return plane.transform(series.fft(), norm=False, epoch=0.0)


def gwpy_best_q(series, cfg):
    return q_scan(series, qrange=cfg.qrange, frange=cfg.frange)[0].plane.q


# -- the methods of each task -----------------------------------------------------

def image_methods(task, cfg, series):
    """``scan`` and ``singleq``: images on the common grid."""
    image_check = None
    if series:
        reference = np.stack([gwpy_image(s, cfg, task).value.T for s in series])
        image_check = partial(check_image, reference=reference, cfg=cfg)
    methods = [Method("SAGE forward", partial(sage_image, cfg, task),
                      image_check and partial(image_check, time_first=True))]
    if task == "scan":
        methods.append(Method("SAGE forward + grad",
                              partial(sage_image, cfg, task, differentiable=True)))
    has_ml4gw = QScan is not None if task == "scan" else Ml4gwSingleQ is not None
    if has_ml4gw:
        for interpolation in ("bicubic", "spline"):
            methods.append(Method(f"ml4gw {interpolation}",
                                  partial(ml4gw_image, cfg, task, interpolation),
                                  image_check and partial(image_check,
                                                          time_first=False)))
    if task == "scan" and QScanMulti is not None:
        methods.append(Method("QTAM QScanMulti", partial(qtam_scan, cfg)))
    if task == "singleq" and QtamSingleQ is not None:
        methods.append(Method("QTAM energy", partial(qtam_energy, cfg)))
    if task == "scan":
        methods.append(Method("DeepExtractor STFT", partial(stft_scan, cfg)))
    return methods


def tiles_methods(cfg, series):
    tiles_check = None
    if series:
        # (series, row) -> energies at the row's own times
        reference = [[np.asarray(e.value) for e in gwpy_tiles(s, cfg).energies]
                     for s in series]
        tiles_check = partial(check_tiles, reference=reference)
    methods = [Method("SAGE tiles", partial(sage_tiles, cfg),
                      tiles_check and partial(tiles_check, rows_of=sage_tile_rows))]
    if Ml4gwSingleQ is not None:
        methods.append(Method("ml4gw compute_qtiles", partial(ml4gw_tiles, cfg),
                              tiles_check and partial(tiles_check,
                                                      rows_of=ml4gw_tile_rows)))
    if QtamSingleQ is not None:
        lossless = qtam_lossless_num_time(cfg)
        for label, num_time in (("full", None), (f"num_time={lossless}", lossless)):
            methods.append(Method(f"QTAM complex {label}",
                                  partial(qtam_tiles, cfg, num_time)))
    return methods


def roundtrip_methods(cfg, series):
    check = partial(check_roundtrip, cfg=cfg)
    methods = [Method("SAGE tiles + inverse", partial(sage_roundtrip, cfg), check)]
    if QtamSingleQ is not None:
        lossless = qtam_lossless_num_time(cfg)
        for label, num_time in (("full", None), (f"num_time={lossless}", lossless)):
            methods.append(Method(f"QTAM {label} + invert",
                                  partial(qtam_roundtrip, cfg, num_time), check))
    methods.append(Method("DeepExtractor STFT + iSTFT",
                          partial(stft_roundtrip, cfg), check))
    return methods


def bestq_methods(cfg, series):
    check = None
    if series:
        reference = np.array([gwpy_best_q(s, cfg) for s in series])
        check = partial(check_best_q, reference=reference)
    methods = [Method("SAGE best_q", partial(sage_best_q, cfg), check)]
    if QScan is not None:
        methods.append(Method("ml4gw QScan planes", partial(ml4gw_best_q, cfg),
                              check))
    return methods


def methods_for(task, cfg):
    series = gwpy_series(N_CHECK, cfg, seed=1) if TimeSeries is not None else None
    if task in ("scan", "singleq"):
        return image_methods(task, cfg, series)
    return {"tiles": tiles_methods, "roundtrip": roundtrip_methods,
            "bestq": bestq_methods}[task](cfg, series)


def gwpy_timing(task, cfg):
    """(name, seconds per series) of gwpy's loop for one task, or None."""
    if TimeSeries is None or task == "roundtrip":
        return None
    if task in ("scan", "singleq"):
        name, run = "gwpy q_transform", partial(gwpy_image, cfg=cfg, task=task)
    elif task == "tiles":
        name, run = "gwpy QPlane.transform", partial(gwpy_tiles, cfg=cfg)
    else:
        name, run = "gwpy q_scan", partial(gwpy_best_q, cfg=cfg)
    series = gwpy_series(N_GWPY_TIMING, cfg, seed=0)
    run(series[0])
    t0 = time.perf_counter()
    for s in series:
        run(s)
    return name, (time.perf_counter() - t0) / len(series)


# -- running ----------------------------------------------------------------------

def format_check(task, value):
    if value is None:
        return "-"
    return f"{value:.2f}" if task == "bestq" else f"{value:.1e}"


def format_times(seconds, batches):
    return " ".join(f"{seconds[b]:9.4f}s" for b in batches)


def time_method(method, task, cfg, device):
    """Time one method on one device and run its check; prints a table row."""
    record = dict(task=task, method=method.name, device=device, seconds={},
                  check=None, error=None)
    try:
        fn = method.build(device)
        for b in cfg.batches:
            x = make_data(b, cfg.n, cfg.fs).to(device)
            record["seconds"][b] = bench(lambda: fn(x.clone()), device, cfg.repeats)
            del x
        if method.check is not None:
            x = make_data(N_CHECK, cfg.n, cfg.fs, seed=1).to(device)
            try:
                record["check"] = method.check(fn, x)
            except Exception as ex:      # keep the timing if only the check fails
                print(f"   check failed: {type(ex).__name__}: {str(ex)[:80]}")
        per_series = 1e3 * record["seconds"][cfg.batches[-1]] / cfg.batches[-1]
        print(f"{method.name:32s} {device:6s} "
              f"{format_times(record['seconds'], cfg.batches)} "
              f"{per_series:10.3f}  {format_check(task, record['check'])}")
    except Exception as ex:
        oom = "out of memory" in str(ex).lower()
        record["error"] = "out of memory" if oom else type(ex).__name__
        print(f"{method.name:32s} {device:6s} failed: {type(ex).__name__}: "
              f"{str(ex)[:80]}")
    return record


def run_task(task, cfg, devices, skip):
    """Time every method of one task on every device; returns the records."""
    print(f"\n== task {task}: {TASK_JOBS[task].format(q=cfg.q)}")
    header = (f"{'method':32s} {'device':6s} "
              + " ".join(f"{'B=' + str(b):>10s}" for b in cfg.batches)
              + f" {'ms/series':>10s}  {CHECK_LABEL[task]}")
    print(header + "\n" + "-" * len(header))
    records = []
    for method in methods_for(task, cfg):
        if any(word in method.name for word in skip):
            continue
        for device in devices:
            records.append(time_method(method, task, cfg, device))
            free(device)
    timing = None if "gwpy" in skip else gwpy_timing(task, cfg)
    if timing is not None:
        name, per = timing
        seconds = {b: per * b for b in cfg.batches}
        records.append(dict(task=task, method=name, device="cpu", seconds=seconds,
                            check="reference", error=None))
        print(f"{name + ' (loop)':32s} {'cpu':6s} "
              f"{format_times(seconds, cfg.batches)} {1e3 * per:10.3f}  reference")
    return records


# -- figures ----------------------------------------------------------------------

# reference palette, categorical slots 1-4 (STFT, QTAM, SAGE as in
# compare_glitch_extraction.py) + gray
COLORS = {"DeepExtractor": "#2a78d6", "QTAM": "#eb6834", "SAGE": "#1baf7a",
          "ml4gw": "#eda100", "gwpy": "#8a8984"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
TASK_TITLE = {
    "scan": "scan - Q-scan image (all planes, median, best Q)",
    "singleq": "singleq - image at Q = {q:g}",
    "tiles": "tiles - raw tiles at Q = {q:g}",
    "roundtrip": "roundtrip - tiles at Q = {q:g} and back",
    "bestq": "bestq - choice of the best Q only",
}
CHECK_SHORT = {"scan": "gwpy err", "singleq": "gwpy err", "tiles": "gwpy err",
               "roundtrip": "recon. err", "bestq": "Q = gwpy"}
FOOTNOTE = ("gwpy err: max |x - gwpy| / peak; recon. err: relative error in band; "
            "Q = gwpy: series whose Q matches gwpy's.")


def bar_annotation(task, check):
    if check == "reference":
        return "reference"
    if check is None:
        return ""
    value = f"{check:.0%}" if task == "bestq" else f"{check:.0e}"
    return f"{CHECK_SHORT[task]} {value}"


def draw_task(ax, task, rows, device, batch, xlim, q):
    """One task's horizontal bars: ms per series at ``batch``, and the checks."""
    ax.set_facecolor(SURFACE)
    labels = []
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        library = r["method"].split()[0]
        suffix = " (CPU)" if library == "gwpy" and device != "cpu" else ""
        labels.append((y, r["method"] + suffix))
        seconds = r["seconds"].get(batch)
        if r["error"] or seconds is None:
            msg = r["error"] or "-"
            if msg not in ("-", "out of memory"):
                msg = f"fails ({msg.lstrip('_')})"
            ax.text(xlim[0] * 1.2, y, msg, va="center", fontsize=7.5, color=MUTED)
            continue
        ms = 1e3 * seconds / int(batch)
        ax.barh(y, ms, height=0.62, color=COLORS.get(library, MUTED))
        note = bar_annotation(task, r["check"])
        ax.text(ms * 1.12, y, f"{ms:.3g} ms" + (f"  |  {note}" if note else ""),
                va="center", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([y for y, _ in labels], [name for _, name in labels], fontsize=8)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", labelsize=7.5, colors=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(TASK_TITLE[task].format(q=q), loc="left", fontsize=9.5,
                 weight="bold", color=INK, pad=4)


def plot_device(results, device, path):
    """One figure for one device: per task, ms per series at the largest batch
    and each method's check. ``results`` as read back from the JSON file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    cfg, records = results["config"], results["records"]
    batch = str(max(int(b) for b in cfg["batches"]))
    tasks = [t for t in TASKS if any(r["task"] == t for r in records)]
    rows = {t: [r for r in records if r["task"] == t
                and (r["device"] == device or r["method"].startswith("gwpy"))]
            for t in tasks}
    times = [1e3 * r["seconds"][batch] / int(batch)
             for r in records if r["seconds"].get(batch)]
    xlim = (10 ** np.floor(np.log10(min(times))),
            10 ** np.ceil(np.log10(max(times)) + 0.9))

    heights = [max(len(rows[t]), 1) for t in tasks]
    fig = plt.figure(figsize=(8.3, 1.15 + 0.3 * sum(heights) + 0.55 * len(tasks)),
                     facecolor=SURFACE)
    grid = fig.add_gridspec(len(tasks), 1, height_ratios=heights)
    for i, task in enumerate(tasks):
        # matplotlib's own Axes: importing gwpy replaces the default one
        ax = fig.add_subplot(grid[i], axes_class=Axes)
        draw_task(ax, task, rows[task], device, batch, xlim, cfg["q"])
    ax.set_xlabel(f"milliseconds per series at batch {batch} (log scale)",
                  fontsize=8.5, color=INK2)
    fig.suptitle(f"Q-transform benchmark on {device.upper()}: {cfg['dur']:g} s at "
                 f"{cfg['fs']:g} Hz, {cfg['frange'][0]:g}-{cfg['frange'][1]:g} Hz, "
                 f"image {cfg['nf']} x {cfg['nt']}",
                 x=0.02, ha="left", fontsize=11, weight="bold", color=INK)
    fig.text(0.02, 0.005, f"{results['machine']}; torch {results['torch']}; float32, "
             f"best of {cfg['repeats']}. {FOOTNOTE}",
             fontsize=6.5, color=MUTED, ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_all(results, outdir):
    for device in results["devices"]:
        plot_device(results, device,
                    os.path.join(outdir, f"qtransform_benchmark_{device}"))


# -- command line -----------------------------------------------------------------

def machine():
    name = platform.processor() or platform.machine()
    if torch.cuda.is_available():
        name += f", {torch.cuda.get_device_name(0)}"
    return f"{name}, {os.cpu_count()} CPUs, {torch.get_num_threads()} torch threads"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", nargs="+", default=["scan"], choices=TASKS + ["all"])
    p.add_argument("--q", type=float, default=11.31,
                   help="Q of the fixed-Q tasks (singleq, tiles, roundtrip)")
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
                   help="leave out methods whose name contains any of these (e.g. "
                        "spline: ml4gw's spline interpolation needs a lot of memory "
                        "for long inputs; gwpy: the reference loop)")
    p.add_argument("--outdir",
                   help="save the results (JSON) and one figure per device here")
    p.add_argument("--plot-only", metavar="JSON",
                   help="only draw the figures of a saved results file")
    return p.parse_args(argv)


def make_config(args):
    n = int(round(args.sample_rate * args.duration))
    return Config(fs=args.sample_rate, dur=args.duration, n=n, q=args.q,
                  qrange=tuple(args.qrange), frange=tuple(args.frange),
                  nf=args.n_fft // 2 + 1, nt=n // args.hop_length + 1,
                  n_fft=args.n_fft, win_length=args.win_length,
                  hop_length=args.hop_length, batches=args.batches,
                  repeats=args.repeats)


def default_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def print_header(cfg, devices, tasks):
    print(f"{cfg.n} samples ({cfg.dur:g} s @ {cfg.fs:g} Hz) | image {cfg.nf} x "
          f"{cfg.nt} | Q {cfg.qrange}, fixed Q {cfg.q:g} | f {cfg.frange} Hz | "
          f"devices {devices}")
    print(f"torch {torch.__version__} | {machine()}")
    for library, why in MISSING.items():
        print(f"{library} unavailable, skipped ({why[:150]})")
    if "gwpy" in MISSING:
        print("  -> no gwpy reference: no gwpy timings and no gwpy checks")
    if QtamSingleQ is not None and any(t in tasks for t in FIXED_Q_TASKS):
        rows = len(QtamSingleQ(cfg.dur, cfg.fs, q=cfg.q,
                               frange=list(cfg.frange)).freqs)
        print(f"QTAM at Q = {cfg.q:g}: {rows} frequency rows (its own spacing)")


def save_results(cfg, devices, records, outdir):
    os.makedirs(outdir, exist_ok=True)
    results = dict(config=dataclasses.asdict(cfg), devices=devices,
                   torch=torch.__version__, machine=machine(), records=records)
    path = os.path.join(outdir, "qtransform_benchmark.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    with open(path) as f:        # as --plot-only reads it (batch keys as str)
        plot_all(json.load(f), outdir)
    print(f"\nsaved {path} and one figure per device in {os.path.abspath(outdir)}")


def main(argv=None):
    args = parse_args(argv)
    if args.plot_only:
        with open(args.plot_only) as f:
            results = json.load(f)
        outdir = os.path.dirname(os.path.abspath(args.plot_only))
        plot_all(results, outdir)
        print("figures in", outdir)
        return

    cfg = make_config(args)
    devices = args.devices or default_devices()
    tasks = TASKS if "all" in args.task else args.task
    print_header(cfg, devices, tasks)
    records = []
    for task in tasks:
        records += run_task(task, cfg, devices, args.skip)
    if args.outdir:
        save_results(cfg, devices, records, args.outdir)


if __name__ == "__main__":
    main()
