"""
Real-data time for EuCAIF v3: good segments per network, event vetoes, splits and glitch tags.

1. **Good time** per observing run and detector: GWOSC ``DATA`` minus time failing CBC Category 2 or holding
   hardware injections, exactly as ``sage.data.noise.data_quality.bad_segments`` (used by SAGE ``main``).
2. **Event vetoes**: every event in GWOSC's event list (all catalogues, marginal ones included) is removed over
   ``[t - before, t + after]``.
3. **Network**: the good time of its detectors, intersected, minus the vetoes; only stretches at least
   ``min_length`` long are kept (the context needed to estimate the PSD and crop the frame).
4. **Splits**, two modes:

   - ``gps_blocks``: the network time is cut into blocks of ``block_seconds`` (GPS-aligned), each block assigned
     to train / validation / test with the given fractions (seeded). Windows never cross a block edge, so the
     splits share no data and no PSD context.
   - ``random``: no separation; every split draws windows anywhere in the network time, as ggwd FM production
     does. Splits may then share data (and PSD contexts); use different seeds per split.

   Either way every window keeps its GPS start time.

The segments are built once from the GWOSC timeline (``build``) and saved to JSON (``save`` / ``load``), so
training never needs the GWOSC API.

**Glitch tags** (``GlitchTags``): Gravity Spy tables (H1/L1 from Zenodo 5649212, other detectors from any CSV
with a GPS column) give the labelled glitches inside a window; windows are tagged, not removed: real data are
never certified glitch-free.

Command line (writes the segments JSON of a v3 config):

    PYTHONPATH=. python -m eucaif.v3.segments eucaif/eucaif_v3.yaml --out eucaif/v3/segments_HL.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

DQ_FLAGS = ["CBC_CAT2", "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_DETCHAR_HW_INJ", "NO_STOCH_HW_INJ"]
EVENTS_URL = "https://gwosc.org/eventapi/json/allevents/"
SPLITS = ("train", "validation", "test")


# -- interval arithmetic (sorted, non-overlapping [start, end) lists) ---------------------------------

def merge(intervals):
    """Sort and merge overlapping or touching intervals."""
    out = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def intersect(x, y):
    """Intersection of two sorted interval lists."""
    out, i, j = [], 0, 0
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if a < b:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(x, y):
    """Parts of ``x`` outside ``y`` (as ``sage.data.noise.data_quality.subtract``)."""
    from sage.data.noise.data_quality import subtract as _subtract
    return [(float(a), float(b)) for a, b in _subtract(list(x), list(y))]


def total(intervals):
    return float(sum(b - a for a, b in intervals))


# -- building from GWOSC -------------------------------------------------------------------------------

def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(total=6, connect=6, read=6, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET", "HEAD"]), respect_retry_after_header=True, raise_on_status=False)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def good_segments(run, detector, session=None):
    """GWOSC DATA minus CBC Category 2 and hardware-injection time (``data_quality.bad_segments``)."""
    from gwosc.datasets import run_segment
    from sage.data.noise.data_quality import bad_segments, get_segments
    session = session or _session()
    start, end = run_segment(run)
    data = merge(get_segments(session, run, f"{detector}_DATA", start, end))
    return subtract(data, bad_segments(run, detector))


def event_times(session=None):
    """GPS times of every event in GWOSC's event list (all catalogues)."""
    session = session or _session()
    events = session.get(EVENTS_URL, timeout=300).json()["events"]
    return sorted({float(v["GPS"]) for v in events.values() if v.get("GPS")})


def network_segments(per_detector, vetoes, min_length):
    """Intersection over the detectors, minus vetoes, rounded inward to whole GPS seconds (so cached data sit on
    one global sample grid), keeping stretches >= ``min_length`` s."""
    seg = None
    for s in per_detector:
        seg = list(s) if seg is None else intersect(seg, s)
    seg = [(float(math.ceil(a)), float(math.floor(b))) for a, b in subtract(seg, vetoes)]
    return [(a, b) for a, b in seg if b - a >= min_length]


def build(runs, detectors, min_length, veto_before=30.0, veto_after=10.0, session=None, log=print):
    """Segments of a network over several runs, from the GWOSC timeline and event list.

    Returns a dict ready for ``save``: per-run segments, the vetoes, and how much time each step removed.
    """
    session = session or _session()
    events = event_times(session)
    vetoes = merge([(t - veto_before, t + veto_after) for t in events])
    out = dict(detectors=list(detectors), runs=list(runs), min_length=float(min_length),
               veto=dict(before=veto_before, after=veto_after, n_events=len(events), source=EVENTS_URL),
               dq_flags=DQ_FLAGS, built=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), segments={},
               hours={})
    for run in runs:
        per_det = []
        for det in detectors:
            log(f"{run} {det}: GWOSC timeline")
            per_det.append(good_segments(run, det, session))
        seg = network_segments(per_det, vetoes, min_length)
        out["segments"][run] = seg
        out["hours"][run] = dict(per_detector={d: total(s) / 3600 for d, s in zip(detectors, per_det)},
                                 network=total(seg) / 3600)
        log(f"{run} {''.join(d[0] for d in detectors)}: {total(seg) / 3600:.0f} h in {len(seg)} segments "
            f">= {min_length:g} s")
    return out


def save(segments, path):
    Path(path).write_text(json.dumps(segments, indent=1))


def load(path):
    d = json.loads(Path(path).read_text())
    d["segments"] = {run: [tuple(s) for s in seg] for run, seg in d["segments"].items()}
    return d


# -- splits ---------------------------------------------------------------------------------------------

def gps_block_split(segments, block_seconds, fractions, seed=0):
    """Assign GPS-aligned blocks of ``block_seconds`` to the splits; each split gets the segment pieces
    inside its blocks (windows drawn from them never cross a block edge).

    ``segments``: {run: [(start, end), ...]}; ``fractions``: {split: fraction}, summing to 1.
    Returns {split: [(start, end), ...]} (all runs together, sorted).
    """
    names = list(fractions)
    p = np.array([fractions[k] for k in names], dtype=float)
    if (p < 0).any() or not math.isclose(p.sum(), 1.0, rel_tol=1e-9):
        raise ValueError(f"split fractions must be >= 0 and sum to 1, got {fractions}")
    out = {k: [] for k in names}
    for run in sorted(segments):
        for a, b in segments[run]:
            for k0 in range(int(math.floor(a / block_seconds)), int(math.ceil(b / block_seconds))):
                # the block's split is a function of (seed, block index) alone: reproducible and independent
                # of which other segments exist
                rng = np.random.default_rng([seed, k0])
                split = names[int(rng.choice(len(names), p=p))]
                lo, hi = max(a, k0 * block_seconds), min(b, (k0 + 1) * block_seconds)
                if hi > lo:
                    out[split].append((lo, hi))
    # pieces stay separate even when adjacent: a window must not cross a block edge
    return {k: sorted(v) for k, v in out.items()}


def split_segments(segments, config):
    """{split: [(start, end), ...]} for a ``split`` config: ``{mode: gps_blocks, block_seconds, fractions, seed}``
    or ``{mode: random}`` (every split sees all the network time)."""
    mode = config.get("mode", "gps_blocks")
    if mode == "gps_blocks":
        return gps_block_split(segments, float(config.get("block_seconds", 4096)),
                               config.get("fractions", dict(train=0.8, validation=0.1, test=0.1)),
                               int(config.get("seed", 0)))
    if mode == "random":
        every = sorted(s for seg in segments.values() for s in seg)
        return {k: list(every) for k in SPLITS}
    raise ValueError(f"split mode must be 'gps_blocks' or 'random', got {mode!r}")


class WindowSampler:
    """Draws windows of ``length`` s uniformly over the time of a list of segments (duration-weighted), each
    returned with its GPS start; segments shorter than ``length`` are never used."""

    def __init__(self, segments, length):
        seg = [(a, b) for a, b in segments if b - a >= length]
        if not seg:
            raise ValueError(f"no segment is at least {length} s long")
        self.length = float(length)
        self.starts = np.array([a for a, _ in seg])
        self.room = np.array([b - a - length for a, b in seg])            # start positions available
        self.cum = np.cumsum(self.room + 1e-9)
        self.hours = total(seg) / 3600

    def sample(self, n, rng, return_segments=False):
        """``n`` GPS start times (float64), uniform over all available start positions; with ``return_segments``
        also the (start, end) of the segment of each."""
        u = rng.random(n) * self.cum[-1]
        i = np.searchsorted(self.cum, u, side="right")
        i = np.minimum(i, len(self.cum) - 1)
        offset = u - (self.cum[i] - self.room[i] - 1e-9)
        t = self.starts[i] + np.clip(offset, 0.0, self.room[i])
        if return_segments:
            return t, [(float(self.starts[k]), float(self.starts[k] + self.room[k] + self.length)) for k in i]
        return t


# -- glitch tags ----------------------------------------------------------------------------------------

class GlitchTags:
    """Labelled glitches from Gravity Spy-like tables, looked up per detector and time window.

    ``sources``: list of dicts ``{path, detector | detector_column, gps_column, label_column,
    confidence_column (optional), min_confidence (optional)}``. Gravity Spy's Zenodo tables:
    ``detector_column: ifo, gps_column: event_time, label_column: ml_label, confidence_column: ml_confidence``.
    """

    def __init__(self, sources):
        import pandas as pd
        self.tables = {}
        for s in sources or []:
            df = pd.read_csv(s["path"])
            if s.get("confidence_column") and s.get("min_confidence") is not None:
                df = df[df[s["confidence_column"]] >= float(s["min_confidence"])]
            dets = df[s["detector_column"]].astype(str) if s.get("detector_column") else [s["detector"]] * len(df)
            conf = df[s["confidence_column"]].to_numpy(float) if s.get("confidence_column") else \
                np.full(len(df), np.nan)
            for det, gps, label, c in zip(dets, df[s["gps_column"]].to_numpy(float),
                                          df[s["label_column"]].astype(str), conf):
                self.tables.setdefault(det, []).append((gps, label, c))
        for det in self.tables:
            rows = sorted(self.tables[det])
            self.tables[det] = (np.array([r[0] for r in rows]), [r[1] for r in rows], np.array([r[2] for r in rows]))

    def detectors(self):
        return sorted(self.tables)

    def in_window(self, detector, start, end):
        """[(gps, label, confidence)] of the tagged glitches of ``detector`` with start <= gps < end; ``None``
        if no table covers the detector (unknown, not 'no glitch')."""
        if detector not in self.tables:
            return None
        t, labels, conf = self.tables[detector]
        i, j = np.searchsorted(t, start, side="left"), np.searchsorted(t, end, side="left")
        return [(float(t[k]), labels[k], float(conf[k])) for k in range(i, j)]


# -- command line ---------------------------------------------------------------------------------------

def main(argv=None):
    import yaml
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    d, s = cfg["data"], cfg["segments"]
    out = build(d["runs"], d["detectors"], float(s["min_length"]), float(s["veto"]["before"]),
                float(s["veto"]["after"]))
    save(out, args.out)
    print(f"saved {args.out}: {sum(total(v) for v in out['segments'].values()) / 3600:.0f} h")


if __name__ == "__main__":
    main()
