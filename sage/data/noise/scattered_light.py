#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : scattered_light.py
Description     : Scattered-light arches, the ArchEnemy model in torch.

GitHub Repository: NULL

Documentation:

    Model: A. Tolley et al., "ArchEnemy: Removing scattered-light glitches from
    gravitational wave data", arXiv:2301.10491 (2023); reference code at
    https://github.com/ArthurTolley/ArchEnemy, ``archenemy/search/artefact_template.py``.

    LICENCE NOTE (to discuss with the SAGE maintainers): ArchEnemy is GPL-3.0 and SAGE is MIT.
    This module is written from the paper's equation below and copies no ArchEnemy code.

    One arch of fringe frequency f_f and time period T, for t in [-T/2, T/2):

        h(t) = sin((f_f / f_rep) sin(2 pi f_rep t) + phi),    f_rep = 1 / (2 T)

    Its instantaneous frequency is f_f cos(pi t / T): it rises from 0 to f_f at the centre
    and falls back to 0, the arch seen in Q-scans. As in ArchEnemy, each arch is tapered with
    a Tukey window (alpha = 0.2) over its own samples, and the defaults are the ranges of
    ArchEnemy's template bank example: f_f in [20, 80] Hz, T in [1.8, 5.5] s (at 2048 Hz).

    The arches are meant for ``GlitchInjector``, which sets their amplitude by SNR:

        arch = scattered_light_arch(B * D, sample_rate=2048, duration=8.0).view(B, D, -1)
        strain = GlitchInjector(2048, f_low=15.0)(strain, arch, asd, snr, t_inj=t_inj)

    Arches last seconds: use a strain window longer than the longest T (plus what the whitening
    discards at the edges), or the arch is cut.

    Differences from ArchEnemy's code: batched, any sample rate, one random phase per arch,
    and the arch placed at ``t0`` inside an array of ``duration`` seconds (ArchEnemy's padding
    step does not run as written). Unit amplitude: the SNR sets it.
"""

import math

import torch

from .analytic_glitches import _uniform


def _upper(bounds) -> float:
    return float(bounds) if isinstance(bounds, (int, float)) else float(bounds[1])


def scattered_light_arch(n: int, sample_rate: float, duration: float = None,
                         fringe_frequency=(20.0, 80.0), timeperiod=(1.8, 5.5), t0=None,
                         taper_alpha: float = 0.2, generator: torch.Generator = None,
                         dtype=torch.float32, device=None, return_params: bool = False):
    """``n`` scattered-light arches, shape (n, duration * sample_rate).

    Parameters
    ----------
    n : int
        Number of arches.
    sample_rate : float
        In Hz.
    duration : float, optional
        Length of the output, in seconds. Default: the longest ``timeperiod``.
    fringe_frequency, timeperiod : (low, high) or float
        Peak frequency f_f (Hz) and duration T (s) of each arch, drawn uniformly per arch
        (a single number is fixed).
    t0 : (low, high) or float, optional
        Centre of each arch, in seconds from the start. Default: ``duration / 2``.
    taper_alpha : float
        Tukey taper over each arch (ArchEnemy: 0.2). 0 for none.
    generator : torch.Generator, optional
        For reproducible draws.
    return_params : bool
        Also return the drawn ``fringe_frequency``, ``timeperiod``, ``t0`` and ``phi``.
    """
    work = torch.float64
    duration = _upper(timeperiod) if duration is None else duration
    t0 = duration / 2 if t0 is None else t0
    f_f, period, t0 = (_uniform(b, n, generator, work) for b in (fringe_frequency, timeperiod, t0))
    phi = _uniform((0.0, 2 * math.pi), n, generator, work)

    # each arch on its own samples, as numpy.arange(-T/2, T/2, 1/fs)
    n_arch = torch.ceil(period * sample_rate).long()
    start = torch.round((t0 - period / 2) * sample_rate).long()
    n_out = int(round(duration * sample_rate))
    if (start < 0).any() or (start + n_arch > n_out).any():
        raise ValueError(f"Arches (T up to {float(period.max()):.2f} s) don't fit in {duration} s "
                         f"around t0; increase duration or narrow t0")

    k = torch.arange(n_out, device=device).unsqueeze(0) - start.to(device).unsqueeze(-1)
    inside = (k >= 0) & (k < n_arch.to(device).unsqueeze(-1))
    period_, f_f_, phi_ = (p.to(device).unsqueeze(-1) for p in (period, f_f, phi))
    t = -period_ / 2 + k.to(work) / sample_rate
    f_rep = 1.0 / (2.0 * period_)
    h = torch.sin(f_f_ / f_rep * torch.sin(2 * math.pi * f_rep * t) + phi_)

    if taper_alpha > 0:
        # scipy.signal.windows.tukey(n_arch, taper_alpha) on each arch's samples
        x = k.to(work) / (n_arch.to(device).unsqueeze(-1) - 1).to(work)
        a = min(taper_alpha, 1.0)
        rise = 0.5 * (1 + torch.cos(math.pi * (-1 + 2.0 * x / a)))
        fall = 0.5 * (1 + torch.cos(math.pi * (-2.0 / a + 1 + 2.0 * x / a)))
        width = torch.floor(a * (n_arch.to(device).unsqueeze(-1) - 1) / 2.0)
        w = torch.where(k <= width, rise, torch.where(k >= n_arch.to(device).unsqueeze(-1) - 1 - width,
                                                      fall, torch.ones_like(x)))
        h = h * w

    h = torch.where(inside, h, torch.zeros_like(h)).to(dtype)
    if return_params:
        params = dict(fringe_frequency=f_f, timeperiod=period, t0=t0, phi=phi)
        return h, {key: v.to(device) for key, v in params.items()}
    return h
