#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : glitch_population.py
Description     : A population of analytic glitches from a config file, injected at given SNRs.

GitHub Repository: NULL

Documentation:

    Modelled on GWForge's glitch configuration (https://github.com/koustavchandra/gwforge,
    ``GWForge/glitch``): each class has a rate per hour per detector, parameter ranges and a
    power-law SNR distribution. See ``eucaif/sage_analytic_glitches.yaml`` for the format.

        pop = GlitchPopulation("eucaif/sage_analytic_glitches.yaml")
        strain, events = pop.inject(strain, asd, generator=g)       # strain: (D, T) or (B, D, T)

    As in GWForge:

    - the number of glitches of a class in a segment is Poisson with mean rate * duration,
      their times uniform in the segment (so the whole glitch array fits in it);
    - the optimal SNR follows p(snr) ~ snr**-snr_index on snr_range (log-uniform for index 1,
      uniform for index 0), drawn by inverse CDF;
    - ``phases`` are in GWForge's cosine convention, h ~ cos(2 pi f0 (t - t0) + phase). SAGE's
      sine-Gaussian, ringdown and Antiglitch shapes use a sine, so phi = phase + pi / 2 is passed to them.

    Shapes come from ``analytic_glitches`` and ``scattered_light``, centred in an array of
    ``duration`` seconds. ``inject`` does what ``add_glitch`` does for each glitch (taper, place,
    band-limit and scale to its SNR, colour with the ASD), vectorised: each class is sampled once
    for the whole batch, glitches of the same length are processed together in the frequency domain
    (one FFT per glitch), and since colouring is linear, each series is coloured and brought back
    to the time domain once, with the sum of its scaled glitches.

    Parameters listed in a class's ``log_uniform`` are drawn log-uniformly on their range (as
    GWForge draws f0); the others uniformly.

    Dense mode (DeepExtractor's training data: 1-30 glitches of random types per 2 s sample): with
    ``glitches_per_segment: [low, high]`` at the top of the config, a segment of a detector gets
    glitches with probability ``segment_probability`` (default 1), then a number uniform in
    [low, high], each of a class drawn with probability proportional to its ``weight`` in that
    detector. Without ``weight``, a class's ``rate`` is its weight, so the class mix follows the
    rates (per detector if they are given per detector); classes with neither have weight 1.
"""

import inspect
import math
from pathlib import Path

import torch
from torch import Tensor

from .analytic_glitches import ANALYTIC_GLITCHES
from .antiglitch import antiglitch_glitch
from .glitch_noise import _band_mask, _resample_asd, place_glitch, tukey_window
from .scattered_light import scattered_light_arch

GENERATORS = dict(ANALYTIC_GLITCHES, antiglitch_glitch=antiglitch_glitch, scattered_light_arch=scattered_light_arch)
CLASS_KEYS = {"generator", "rate", "weight", "snr_range", "snr_index", "phases", "duration", "log_uniform"}
PHASED = {"sine_gaussian", "ringdown", "antiglitch_glitch"}


def power_law_snr(n: int, snr_range, snr_index: float, generator: torch.Generator = None) -> Tensor:
    """``n`` SNRs from p(snr) ~ snr**-snr_index on ``snr_range`` (log-uniform for index 1)."""
    low, high = float(snr_range[0]), float(snr_range[1])
    u = torch.rand(n, generator=generator, dtype=torch.float64)
    if math.isclose(snr_index, 1.0):
        return low * (high / low) ** u
    k = 1.0 - snr_index
    return (low ** k + u * (high ** k - low ** k)) ** (1.0 / k)


class GlitchPopulation:
    """Analytic glitch classes with rates, parameter ranges and SNR distributions, from a config.

    Parameters
    ----------
    config : str, Path or dict
        YAML file (or its content) with ``detectors``, ``sample_rate``, ``classes`` and, for the
        dense mode, ``glitches_per_segment``.
    sample_rate : float, optional
        Overrides the config's ``sample_rate``.
    """

    def __init__(self, config, sample_rate: float = None):
        if isinstance(config, (str, Path)):
            import yaml
            with open(config) as f:
                config = yaml.safe_load(f)
        self.config = config
        self.detectors = list(config.get("detectors", []))
        self.sample_rate = float(sample_rate or config["sample_rate"])
        self.classes = {name: self._check(name, dict(c)) for name, c in config["classes"].items()}
        self.per_segment = config.get("glitches_per_segment")
        self.segment_probability = float(config.get("segment_probability", 1.0))
        if not 0 <= self.segment_probability <= 1:
            raise ValueError(f"segment_probability must be in [0, 1], got {self.segment_probability}")
        if self.per_segment is not None:
            low, high = (int(v) for v in self.per_segment)
            if not 0 <= low <= high:
                raise ValueError(f"glitches_per_segment must be 0 <= low <= high, got {self.per_segment}")
            self.per_segment = (low, high)
            for det in (self.detectors or [None]):
                w = self.weights(det)
                if (w < 0).any() or w.sum() <= 0:
                    raise ValueError(f"class weights{' in ' + det if det else ''} must be >= 0 and not all 0")

    def _check(self, name, c):
        gen = c.get("generator")
        if gen not in GENERATORS:
            raise ValueError(f"{name}: unknown generator {gen!r}; choose from {sorted(GENERATORS)}")
        allowed = set(inspect.signature(GENERATORS[gen]).parameters) - {"n", "sample_rate", "duration",
                                                                        "generator", "dtype", "device",
                                                                        "return_params"}
        unknown = set(c) - CLASS_KEYS - allowed
        if unknown:
            raise ValueError(f"{name}: unknown keys {sorted(unknown)} for {gen} (parameters: {sorted(allowed)})")
        if "phases" in c and gen not in PHASED:
            raise ValueError(f"{name}: 'phases' only applies to {sorted(PHASED)}")
        if "phases" in c and "phi" in c:
            raise ValueError(f"{name}: give 'phases' or 'phi', not both")
        for key in ("rate", "weight"):
            v = c.get(key, 0.0)
            if isinstance(v, dict) and self.detectors and set(v) - set(self.detectors):
                raise ValueError(f"{name}: {key}s for unknown detectors {sorted(set(v) - set(self.detectors))}")
        for key in c.get("log_uniform", []):
            v = c.get(key)
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and 0 < v[0] < v[1]):
                raise ValueError(f"{name}: log_uniform parameter {key!r} needs a range [low, high] with 0 < low < high")
        low, high = c.setdefault("snr_range", [5.0, 200.0])
        if not 0 < low < high:
            raise ValueError(f"{name}: snr_range must be 0 < low < high, got {c['snr_range']}")
        c.setdefault("snr_index", 1.0)
        if c.setdefault("duration", 1.0) <= 0:
            raise ValueError(f"{name}: duration must be positive")
        # [low, high] -> (low, high) ranges for the generators
        for key in allowed & set(c):
            if isinstance(c[key], list):
                c[key] = tuple(float(v) for v in c[key])
        return c

    def rate(self, name: str, detector: str = None) -> float:
        """Mean number per hour of class ``name`` in ``detector``."""
        r = self.classes[name].get("rate", 0.0)
        return float(r.get(detector, 0.0)) if isinstance(r, dict) else float(r)

    def parameter_names(self) -> list:
        """Names of all parameters the classes' glitches carry (``params`` of ``sample``)."""
        names = set()
        for name in self.classes:
            names |= set(self.sample(name, 1, torch.Generator().manual_seed(0))[1])
        return sorted(names)

    def weights(self, detector: str = None) -> Tensor:
        """Dense-mode class weights in ``detector``: ``weight``, else ``rate``, else 1."""
        out = []
        for name, c in self.classes.items():
            if "weight" in c:
                w = c["weight"]
                out.append(float(w.get(detector, 0.0)) if isinstance(w, dict) else float(w))
            elif "rate" in c:
                out.append(self.rate(name, detector))
            else:
                out.append(1.0)
        return torch.tensor(out, dtype=torch.float64)

    def sample(self, name: str, n: int, generator: torch.Generator = None, dtype=torch.float32,
               device=None):
        """``n`` glitches of class ``name``: shapes (n, duration * sample_rate) and their parameters
        (the generator's, plus ``snr`` and, with ``phases``, ``phase`` in GWForge's convention)."""
        c = self.classes[name]
        gen = c["generator"]
        kwargs = {k: v for k, v in c.items() if k not in CLASS_KEYS}
        for key in c.get("log_uniform", []):
            low, high = (math.log(v) for v in c[key])
            kwargs[key] = torch.exp(low + (high - low) * torch.rand(n, generator=generator, dtype=torch.float64))
        if gen != "scattered_light_arch":
            kwargs.setdefault("t0", c["duration"] / 2)          # centred: injection time = glitch time
        phase = None
        if "phases" in c:
            choice = torch.randint(len(c["phases"]), (n,), generator=generator)
            phase = torch.tensor(c["phases"], dtype=torch.float64)[choice]
            kwargs["phi"] = phase + math.pi / 2                 # cos(x + phase) = sin(x + phase + pi / 2)
        glitch, params = GENERATORS[gen](n, self.sample_rate, duration=c["duration"], generator=generator,
                                         dtype=dtype, device=device, return_params=True, **kwargs)
        params["snr"] = power_law_snr(n, c["snr_range"], c["snr_index"], generator).to(device)
        if phase is not None:
            params["phase"] = phase.to(device)
        return glitch, params

    def _counts(self, segment_duration: float, detector, generator) -> list:
        """Number of glitches of each class in one segment of one detector."""
        if self.per_segment is None:
            return [int(torch.poisson(torch.tensor(self.rate(name, detector) * segment_duration / 3600.0,
                                                   dtype=torch.float64), generator=generator))
                    for name in self.classes]
        low, high = self.per_segment
        if self.segment_probability < 1 and float(torch.rand(1, generator=generator)) >= self.segment_probability:
            return [0] * len(self.classes)
        total = int(torch.randint(low, high + 1, (1,), generator=generator))
        if total == 0:
            return [0] * len(self.classes)
        picks = torch.multinomial(self.weights(detector), total, replacement=True, generator=generator)
        return torch.bincount(picks, minlength=len(self.classes)).tolist()

    def draw_batch(self, segment_duration: float, detectors: list, generator: torch.Generator = None,
                   dtype=torch.float32, device=None, window=None) -> list:
        """Glitches for one segment per entry of ``detectors``: for each, a list of events, each a dict
        with ``cls``, ``detector``, ``time`` (s from the segment start), ``snr``, ``glitch`` and
        ``params``. Each class is sampled once for all segments.

        ``window = (start, stop)`` (s from the segment start) restricts the glitch arrays to that part
        of the segment, e.g. to keep them out of the edges a whitening filter discards; rates then
        apply to its length.
        """
        start, stop = (0.0, segment_duration) if window is None else (float(window[0]), float(window[1]))
        if not 0 <= start < stop <= segment_duration:
            raise ValueError(f"window {window} must lie inside [0, {segment_duration}] s")
        length = stop - start
        counts = [self._counts(length, det, generator) for det in detectors]
        events = [[] for _ in detectors]
        for j, (name, c) in enumerate(self.classes.items()):
            per_series = [cnt[j] for cnt in counts]
            m = sum(per_series)
            if m == 0:
                continue
            if c["duration"] > length:
                raise ValueError(f"{name}: glitches of {c['duration']} s don't fit in {length} s segments")
            glitch, params = self.sample(name, m, generator, dtype, device)
            half = c["duration"] / 2
            times = start + half + (length - 2 * half) * torch.rand(m, generator=generator, dtype=torch.float64)
            values = {k: v.tolist() for k, v in params.items()}
            i = 0
            for s, k in enumerate(per_series):
                for _ in range(k):
                    events[s].append(dict(cls=name, detector=detectors[s], time=float(times[i]),
                                          snr=values["snr"][i], glitch=glitch[i],
                                          params={key: v[i] for key, v in values.items()}))
                    i += 1
        return events

    def draw(self, segment_duration: float, detector: str = None, generator: torch.Generator = None,
             dtype=torch.float32, device=None, window=None) -> list:
        """Glitches of all classes in one segment of one detector (see ``draw_batch``)."""
        return self.draw_batch(segment_duration, [detector], generator, dtype, device, window)[0]

    def apply(self, strain: Tensor, asd: Tensor, events: list, taper_alpha: float = 0.1, f_low: float = None,
              f_high: float = None, lowpass_order=3, max_rows: int = 512) -> Tensor:
        """Add the glitches of ``events`` (one list per series of ``strain``, flattened leading index)
        to coloured strain (..., T): for each, exactly ``add_glitch`` with its ``time`` and ``snr``."""
        T = strain.shape[-1]
        flat = strain.reshape(-1, T)
        if len(events) != flat.shape[0]:
            raise ValueError(f"{len(events)} event lists for {flat.shape[0]} series")
        work = torch.float64 if strain.dtype == torch.float64 else torch.float32
        dev = strain.device
        groups = {}                                    # glitch length -> rows
        for s, evs in enumerate(events):
            for e in evs:
                g = groups.setdefault(e["glitch"].shape[-1], ([], [], [], []))
                for lst, v in zip(g, (s, e["glitch"], e["time"], e["snr"])):
                    lst.append(v)
        if not groups:
            return strain.clone()
        # as rescale_glitch + colouring, in the frequency domain: band-limit, SNR
        # rho^2 = 4 df sum |W / fs|^2, scale, sum per series; colour and invert once per series
        fs = self.sample_rate
        mask = _band_mask(T, fs, f_low, f_high, dev, work, lowpass_order)
        white = torch.zeros(flat.shape[0], T // 2 + 1, dtype=torch.complex128 if work == torch.float64
                            else torch.complex64, device=dev)
        for L, (series, glitches, times, snrs) in groups.items():
            win = tukey_window(L, taper_alpha, device=dev, dtype=work) if taper_alpha else None
            for a in range(0, len(series), max_rows):
                b = a + max_rows
                g = torch.stack(glitches[a:b]).to(dtype=work, device=dev)
                if win is not None:
                    g = g * win
                t = torch.tensor(times[a:b], dtype=torch.float64, device=dev)
                W = torch.fft.rfft(place_glitch(g, T, torch.round(t * fs).long() - L // 2), dim=-1) * mask
                current = torch.sqrt(4.0 * (fs / T) * ((W.abs() / fs) ** 2).sum(-1))
                target = torch.tensor(snrs[a:b], dtype=work, device=dev)
                scale = torch.where(current > 0, target / current.clamp_min(torch.finfo(work).tiny),
                                    torch.zeros_like(current))
                white.index_add_(0, torch.tensor(series[a:b], device=dev), W * scale[:, None])
        asd = torch.as_tensor(asd, device=dev).to(work)
        asd = _resample_asd(asd.expand(*strain.shape[:-1], asd.shape[-1]).reshape(-1, asd.shape[-1]), T // 2 + 1)
        h = torch.fft.irfft(white * asd, n=T, dim=-1)
        return (flat + h.to(strain.dtype)).reshape(strain.shape)

    def inject(self, strain: Tensor, asd: Tensor, generator: torch.Generator = None, taper_alpha: float = 0.1,
               f_low: float = None, f_high: float = None, lowpass_order=3, window=None):
        """Draw and add glitches to coloured strain (..., D, T), the detectors in config order.

        ``asd`` is the strain's one-sided ASD, (F,) or broadcastable to (..., D, F), as for
        ``add_glitch``. ``window`` restricts the glitches to part of the series (see ``draw_batch``).
        Returns the strain with glitches and, for each series (flattened leading index), its list
        of events.
        """
        T = strain.shape[-1]
        D = strain.shape[-2] if strain.dim() > 1 else 1
        if self.detectors and D != len(self.detectors):
            raise ValueError(f"strain has {D} detectors, the config {len(self.detectors)}: {self.detectors}")
        S = strain.numel() // T
        dets = [self.detectors[i % D] if self.detectors else None for i in range(S)]
        events = self.draw_batch(T / self.sample_rate, dets, generator, dtype=strain.dtype, device=strain.device,
                                 window=window)
        out = self.apply(strain, asd, events, taper_alpha=taper_alpha, f_low=f_low, f_high=f_high,
                         lowpass_order=lowpass_order)
        return out, events
