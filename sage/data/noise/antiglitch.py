#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : antiglitch.py
Description     : Antiglitch's quasi-physical glitch model, ported to torch from the Antiglitch repository.

GitHub Repository: NULL

Documentation:

    Ported from the Antiglitch repository by Andrew Lundgren,
    https://github.com/andrew-lundgren/antiglitch, ``antiglitch/model.py`` (``fsignal`` and
    ``fglitch_from_sample``) at commit de319c4 (MIT licence). Model paper: arXiv:2309.06594.
    The parameter ranges in the table below come from the same repository, ``data/all_PE_v3.json``.

    Antiglitch fits short glitches (Gravity Spy Blip, Tomte, Koi_Fish, Blip_Low_Frequency,
    Extremely_Loud) with a frequency-domain shape: a Gaussian bump in log frequency, with a
    complex amplitude and a time shift,

        h(f) = A exp(i psi) exp(-gbw / 2 (ln f - ln f0)^2) exp(-2 pi i f t0)

    f0 is the centre of the bump in *strain* (before whitening; the whitened peak sits higher,
    where the detector is most sensitive) and gbw its inverse width: the bump's standard
    deviation in ln f is 1 / sqrt(gbw). The defaults are Antiglitch's priors (``new_model``):
    f0 in [10, 600] Hz, gbw in [0.25, 8]. Its fits to O3 glitches (``data/all_PE_v3.json``,
    5th-95th percentiles):

        class                f0 (Hz)     gbw
        Blip                 12 - 260    0.29 - 4.2
        Blip_Low_Frequency   12 - 48     0.61 - 6.4
        Koi_Fish             10 - 168    0.31 - 7.5
        Tomte                12 - 26     1.4 - 7.0

    The shapes are meant for ``GlitchInjector``, which sets their amplitude by SNR:

        glitch = antiglitch_glitch(B * D, sample_rate=4096).view(B, D, -1)
        strain = GlitchInjector(4096, f_low=20.0)(strain, glitch, asd, snr, t_inj=t_inj)

    Differences from Antiglitch:

    - Any sample rate and duration (Antiglitch: 1/8 s at 8192 Hz, 513 frequencies).
    - The phase ``phi`` follows ``analytic_glitches``: it multiplies a sine, so phi = pi / 2
      is a glitch symmetric about t0 (Antiglitch's real amplitude, psi = 0) and works with
      ``GlitchPopulation``'s ``phases``. Here psi = phi - pi / 2.
    - The glitch is placed at ``t0`` from the start of the array (Antiglitch: from its centre).
    - Peak |h| of 1, like the other analytic shapes; the SNR sets the amplitude.

    The shape is built in the frequency domain, so it wraps around the array: a narrow bump
    (large gbw) at low f0 rings for long, so give it a long enough ``duration``.
"""

import math

import torch

from .analytic_glitches import _finish, _uniform


def antiglitch_glitch(n: int, sample_rate: float, duration: float = 1.0, f0=(10.0, 600.0),
                      gbw=(0.25, 8.0), t0=(0.4, 0.6), phi=(0.0, 2 * math.pi),
                      generator: torch.Generator = None, dtype=torch.float32, device=None,
                      return_params: bool = False):
    """``n`` Antiglitch glitches (log-normal bump in frequency), shape (n, duration * sample_rate)."""
    work = torch.float64
    f0, gbw, t0 = (_uniform(b, n, generator, work) for b in (f0, gbw, t0))
    phi = _uniform(phi, n, generator, work)
    n_out = int(round(duration * sample_rate))
    f = torch.fft.rfftfreq(n_out, d=1.0 / sample_rate, dtype=work, device=device)
    f0_, gbw_, t0_, phi_ = (p.to(device).unsqueeze(-1) for p in (f0, gbw, t0, phi))
    log_f = torch.log(f.clamp_min(f[1]))                    # f = 0 is set to 0 below
    bump = torch.exp(-0.5 * gbw_ * (log_f - torch.log(f0_)) ** 2)
    bump = torch.where(f > 0, bump, torch.zeros_like(bump))
    # irfft(S exp(i theta)) ~ sum S cos(2 pi f t + theta): theta = phi - pi / 2 gives sin(... + phi)
    h = torch.fft.irfft(bump * torch.exp(1j * (phi_ - math.pi / 2 - 2 * math.pi * f * t0_)), n=n_out, dim=-1)
    return _finish(h, dict(f0=f0, gbw=gbw, t0=t0, phi=phi), dtype, device, return_params)
