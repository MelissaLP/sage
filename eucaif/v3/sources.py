"""
Real strain for EuCAIF v3: where it comes from (GWOSC or frames), resampling to the target rate, and a local cache.

**Sources** (``sources`` in the YAML, tried in order: the next one is used if one fails, e.g. GWOSC is down):

- ``gwosc``: ``gwpy.timeseries.TimeSeries.fetch_open_data`` (GWOSC open data), at ``sample_rate``: 16384 Hz by
  default (the same decimation as frames, x8 to 2048 Hz). 4096 Hz files can be downloaded too (4x less traffic),
  but they change the decimation: GWOSC's own downsampling filter, then x2 here. GWOSC serves
  offline-calibrated strain.
- ``frames``: ``TimeSeries.get(channel, frametype=...)`` on a LIGO/Virgo cluster, with ``channels`` and
  ``frametypes`` per detector in the YAML. Their calibration version need not be GWOSC's: the two sources are not
  identical (see the v3 README), so every cached piece records which source, channel and frame type it came from.

**Resampling**: native rate -> ``sample_rate`` with ``sage.dsp.decimate`` (half-band FIR, the factor must be a power
of 2), the same for both sources.

**Cache**: strain is stored decimated, as float32 ``.npy`` pieces with a ``.json`` of provenance each, one piece per
good segment inside each GPS-aligned chunk of ``chunk_seconds`` (4096 s: GWOSC's file length, so one chunk is one
GWOSC file). Pieces start on whole GPS seconds (segments are rounded inward), so all cached samples sit on one
global grid and windows spanning several pieces line up. float32 storage changes whitened data by ~2e-5 of their
peak (whitening itself must be float64). ``margin_seconds`` (default 0) fetches extra data around each piece so the
decimation's edge effects (~5 ms) stay outside it; with GWOSC a margin across a file boundary costs a second
file, so by default the few windows that straddle a chunk edge carry ~5 ms of edge effect.

**Download once, then train from disk.** GWOSC's API serves whole 4096 s files (~0.5 GB at 16 kHz) even for a
few seconds of data, so it is far too slow to call per window. The ``download`` command fetches every chunk the
network's segments need, once each, and is split into parts for a batch system (HTCondor:
``eucaif/v3/download.sub``); training then reads only the cache (a missing piece is an error, unless
``cache.fetch_missing`` is true):

    PYTHONPATH=. python -m eucaif.v3.sources download eucaif/eucaif_v3.yaml --dry-run          # files and GB
    PYTHONPATH=. python -m eucaif.v3.sources download eucaif/eucaif_v3.yaml --parts 100 --part 7

    cache = StrainCache.from_config(cfg)          # cfg: the v3 YAML
    x, gps0, prov = cache.window("H1", t0, 32.0, segment)   # (n,) float64, GPS of x[0], provenance per piece
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sage.dsp import decimate


class Source:
    """One way to get strain: ``fetch(detector, start, end) -> (float64 array, native sample rate, provenance)``."""

    name = None

    def fetch(self, detector, start, end):
        raise NotImplementedError


class GWOSCSource(Source):
    name = "gwosc"

    def __init__(self, sample_rate=16384):
        self.sample_rate = int(sample_rate)

    def fetch(self, detector, start, end):
        from gwpy.timeseries import TimeSeries
        ts = TimeSeries.fetch_open_data(detector, start, end, sample_rate=self.sample_rate, cache=False)
        return (np.asarray(ts.value, dtype=np.float64), float(ts.sample_rate.value),
                dict(source="gwosc", detector=detector, native_sample_rate=float(ts.sample_rate.value),
                     gps_start=float(ts.t0.value)))


class FramesSource(Source):
    name = "frames"

    def __init__(self, channels, frametypes):
        self.channels, self.frametypes = dict(channels), dict(frametypes)

    def fetch(self, detector, start, end):
        from gwpy.timeseries import TimeSeries
        if detector not in self.channels or detector not in self.frametypes:
            raise ValueError(f"frames: no channel/frametype for {detector} in the config")
        ts = TimeSeries.get(self.channels[detector], start, end, frametype=self.frametypes[detector])
        return (np.asarray(ts.value, dtype=np.float64), float(ts.sample_rate.value),
                dict(source="frames", detector=detector, channel=self.channels[detector],
                     frametype=self.frametypes[detector], native_sample_rate=float(ts.sample_rate.value),
                     gps_start=float(ts.t0.value)))


def make_sources(config):
    """Sources in fallback order from the ``sources`` block:
    ``{order: [gwosc, frames], gwosc: {...}, frames: {...}}``."""
    out = []
    for name in config.get("order", ["gwosc"]):
        opts = config.get(name, {}) or {}
        if name == "gwosc":
            out.append(GWOSCSource(opts.get("sample_rate", 16384)))
        elif name == "frames":
            out.append(FramesSource(opts.get("channels", {}), opts.get("frametypes", {})))
        else:
            raise ValueError(f"unknown source {name!r}; choose gwosc or frames")
    return out


def resample(x, native_rate, target_rate):
    """``sage.dsp.decimate`` from ``native_rate`` to ``target_rate`` (a power-of-2 ratio)."""
    ratio = native_rate / target_rate
    factor = int(round(ratio))
    if not math.isclose(ratio, factor) or factor < 1 or factor & (factor - 1):
        raise ValueError(f"cannot decimate {native_rate} Hz to {target_rate} Hz: the ratio must be a power of 2")
    return decimate(torch.as_tensor(x, dtype=torch.float64), factor).numpy() if factor > 1 else np.asarray(x)


class StrainCache:
    """Decimated strain pieces on disk, fetched on first use (see the module documentation)."""

    def __init__(self, directory, sources, sample_rate=2048, chunk_seconds=4096, margin_seconds=0.0,
                 fetch_missing=True, log=print):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sources = list(sources)
        self.fs = float(sample_rate)
        self.chunk = float(chunk_seconds)
        self.margin = float(margin_seconds)
        self.fetch_missing = bool(fetch_missing)
        self.log = log

    @classmethod
    def from_config(cls, cfg, log=print):
        """From the v3 YAML; a relative ``cache.directory`` is relative to the YAML's folder (``cfg["_dir"]``)."""
        c = cfg["cache"]
        directory = Path(c["directory"])
        if not directory.is_absolute() and cfg.get("_dir") is not None:
            directory = Path(cfg["_dir"]) / directory
        return cls(directory, make_sources(cfg["sources"]), sample_rate=cfg["data"]["sample_rate"],
                   chunk_seconds=c.get("chunk_seconds", 4096), margin_seconds=c.get("margin_seconds", 0.0),
                   fetch_missing=c.get("fetch_missing", False), log=log)

    def _fetch(self, detector, start, end):
        """Native strain of [start, end) from the first source that can provide it."""
        errors = []
        for src in self.sources:
            try:
                x, rate, prov = src.fetch(detector, start, end)
                return x, rate, prov, errors
            except Exception as e:                            # e.g. GWOSC down: try the next source
                errors.append(f"{src.name}: {type(e).__name__}: {str(e)[:200]}")
        raise RuntimeError(f"no source could provide {detector} [{start}, {end}): " + " | ".join(errors))

    def _store(self, detector, lo, hi, x, rate, fetched_start, prov, errors):
        """Decimate the native strain ``x`` (starting at ``fetched_start``), keep [lo, hi), save it."""
        n_native = int(round((hi - lo) * rate))
        i0 = int(round((lo - fetched_start) * rate))
        m = int(round(min(self.margin, lo - fetched_start) * rate))           # margin kept for the decimation
        seg = x[i0 - m:i0 + n_native + int(round(self.margin * rate))]
        if not np.isfinite(seg[m:m + n_native]).all():
            raise RuntimeError(f"{detector} [{lo}, {hi}): {prov['source']} returned non-finite samples")
        y = resample(np.nan_to_num(seg), rate, self.fs)
        y = y[int(round(m * self.fs / rate)):][:int(round((hi - lo) * self.fs))].astype(np.float32)
        if y.size != int(round((hi - lo) * self.fs)):
            raise RuntimeError(f"{detector} [{lo}, {hi}): got {y.size} samples from {prov['source']}")
        prov = dict(prov, gps_start=lo, gps_end=hi, sample_rate=self.fs, fetched_start=fetched_start,
                    fetched_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), failed_sources=errors)
        path = self._path(detector, lo, hi)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), y)
        path.with_suffix(".json").write_text(json.dumps(prov))
        return y, prov

    def fill_chunk(self, detector, pieces):
        """Fetch the span of ``pieces`` (all in one chunk) once and store every missing piece of it; if no
        source serves the whole span, fetch the pieces one by one. Returns the number of pieces written."""
        todo = [(lo, hi) for lo, hi in pieces if not self._path(detector, lo, hi).with_suffix(".npy").exists()]
        if not todo:
            return 0
        fa, fb = todo[0][0], todo[-1][1]
        try:
            x, rate, prov, errors = self._fetch(detector, fa, fb)
            for lo, hi in todo:
                self._store(detector, lo, hi, x, rate, fa, prov, errors)
        except RuntimeError:
            for lo, hi in todo:
                if not self._path(detector, lo, hi).with_suffix(".npy").exists():
                    x, rate, prov, errors = self._fetch(detector, lo, hi)
                    self._store(detector, lo, hi, x, rate, lo, prov, errors)
        return len(todo)

    def pieces(self, start, end, segment):
        """The cache pieces (GPS-aligned, clipped to ``segment``) covering [start, end)."""
        a, b = segment
        out = []
        for k in range(int(math.floor(start / self.chunk)), int(math.floor((end - 1e-9) / self.chunk)) + 1):
            lo, hi = max(k * self.chunk, a), min((k + 1) * self.chunk, b)
            if hi > lo:
                out.append((lo, hi))
        return out

    def _path(self, detector, lo, hi):
        return self.dir / detector / f"{detector}-{int(lo)}-{int(hi - lo)}"

    def piece(self, detector, lo, hi, segment):
        """Decimated float32 strain of [lo, hi) and its provenance (fetched and stored on first use)."""
        path = self._path(detector, lo, hi)
        if path.with_suffix(".npy").exists():
            return np.load(path.with_suffix(".npy")), json.loads(path.with_suffix(".json").read_text())
        if not self.fetch_missing:
            raise FileNotFoundError(f"{path}.npy is not in the cache: run `python -m eucaif.v3.sources download` "
                                    f"first (or set cache.fetch_missing: true)")
        fa, fb = max(lo - self.margin, segment[0]), min(hi + self.margin, segment[1])
        x, rate, prov, errors = self._fetch(detector, fa, fb)
        return self._store(detector, lo, hi, x, rate, fa, prov, errors)

    def window(self, detector, start, length, segment):
        """``length`` s of decimated strain from ``start`` (snapped to the sample grid), inside ``segment``.

        Returns (float64 array, GPS time of its first sample, [provenance of each piece used])."""
        n = int(round(length * self.fs))
        i_start = int(math.ceil(start * self.fs - 1e-6))                    # first grid sample at or after start
        if (i_start + n) / self.fs > segment[1] + 1e-9:
            i_start = int(math.floor(segment[1] * self.fs)) - n
        t0 = i_start / self.fs
        if t0 < segment[0] - 1e-9:
            raise ValueError(f"a {length} s window does not fit in segment {segment}")
        parts, provs = [], []
        for lo, hi in self.pieces(t0, t0 + length, segment):
            y, prov = self.piece(detector, lo, hi, segment)
            parts.append((lo, y))
            provs.append(prov)
        x = np.concatenate([y for _, y in parts]).astype(np.float64)
        i = int(round((t0 - parts[0][0]) * self.fs))
        return x[i:i + n], t0, provs


def plan(segments, chunk_seconds):
    """The cache's work: ``[(chunk index, [pieces in it])]`` for {run: [(start, end), ...]} (pieces are the good
    segments clipped to the chunk)."""
    chunks = {}
    for seg in segments.values():
        for a, b in seg:
            for k in range(int(math.floor(a / chunk_seconds)), int(math.floor((b - 1e-9) / chunk_seconds)) + 1):
                lo, hi = max(k * chunk_seconds, a), min((k + 1) * chunk_seconds, b)
                if hi > lo:
                    chunks.setdefault(k, []).append((lo, hi))
    return sorted((k, sorted(v)) for k, v in chunks.items())


def main(argv=None):
    import yaml
    from eucaif.v3 import segments as S
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["download"])
    p.add_argument("config")
    p.add_argument("--parts", type=int, default=1, help="split the work into this many parts (batch jobs)")
    p.add_argument("--part", type=int, default=0, help="which part to do (0 .. parts - 1)")
    p.add_argument("--detectors", nargs="+", help="default: the config's")
    p.add_argument("--dry-run", action="store_true", help="only print how many files and GB the parts need")
    args = p.parse_args(argv)
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["_dir"] = cfg_path.resolve().parent
    seg = S.load(cfg["_dir"] / cfg["segments"]["file"])["segments"]
    cache = StrainCache.from_config(cfg)
    work = plan(seg, cache.chunk)
    dets = args.detectors or cfg["data"]["detectors"]
    jobs = [(det, k, pieces) for k, pieces in work for det in dets]
    mine = jobs[args.part::args.parts]
    gb = sum(hi - lo for _, _, pieces in mine for lo, hi in pieces) * cache.fs * 4 / 1e9
    print(f"{len(jobs)} (detector, chunk) downloads in total; part {args.part}/{args.parts}: {len(mine)}, "
          f"{gb:.1f} GB in the cache at {cache.fs:g} Hz float32", flush=True)
    if args.dry_run:
        return
    for i, (det, k, pieces) in enumerate(mine):
        tic = time.time()
        n = cache.fill_chunk(det, pieces)
        print(f"[{i + 1}/{len(mine)}] {det} chunk {int(k * cache.chunk)}: {n} pieces written "
              f"({time.time() - tic:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
