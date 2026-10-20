#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : analytic_glitches.py
Description     : Analytic glitch shapes, a torch port of McGAN's training-data generators.

GitHub Repository: NULL

Documentation:

    Source: https://github.com/jmcginn/McGANn, ``cGAN/GAN_training_data.py`` at commit
    28b335d (MIT licence). Cite:

        J. McGinn et al., "Generalised gravitational wave burst generation with generative
        adversarial networks", Class. Quantum Grav. 38, 155005 (2021),
        doi:10.1088/1361-6382/ac09cc.

    Four burst-like shapes, with McGAN's parameter ranges as defaults (McGAN draws 1 s at
    1024 Hz, centred at t0 in [0.4, 0.6] s):

    - ``sine_gaussian``     A exp(-(t - t0)^2 / tau^2) sin(2 pi f0 (t - t0) + phi),
                            f0 in [70, 250] Hz, Q in [3, 20], tau = Q / (sqrt(2) pi f0)
    - ``ringdown``          A exp(-(t - t0) / tau) sin(2 pi f0 (t - t0) + phi) for t > t0,
                            f0 in [70, 250] Hz, Q in [9, 20]
    - ``white_noise_burst`` exp(-(t - t0)^2 / tau) times white noise band-passed to
                            [70, 250] Hz (5th-order Butterworth), tau in [1e-4, 1e-2] s^2
                            (McGAN divides by tau, not tau^2: widths of 0.01 to 0.1 s)
    - ``gaussian_blip``     exp(-(t - t0)^2 / tau^2), tau in [0.01, 0.05] s

    McGAN's BBH inspiral class is a signal, not a glitch, and is not ported.

    The shapes are meant for ``GlitchInjector``, which sets their amplitude by SNR:

        glitch = sine_gaussian(B * D, sample_rate=2048).view(B, D, -1)
        strain = GlitchInjector(2048, f_low=20.0)(strain, glitch, asd, snr, t_inj=t_inj)

    ``GlitchInjector`` centres the glitch *array* at ``t_inj``; the glitch itself sits at t0
    inside it, so pass ``t0=duration / 2`` to put it exactly at ``t_inj``.

    Differences from McGAN:

    - Any sample rate and duration. McGAN's time grid is ``linspace(0, 1, sample_rate)``
      (spacing 1 / (sample_rate - 1)); here it is ``arange(n) / sample_rate``. McGAN's white
      noise burst only works at 1024 Hz (it draws 1024 noise samples).
    - One phase per glitch; McGAN draws one phase per call, shared by the whole batch.
    - Every shape is scaled to a peak |h| of 1. McGAN divides by max(h) (not max |h|) and
      leaves the Gaussian blip unscaled (its peak is 1 anyway). Irrelevant with
      ``GlitchInjector``, which rescales to the requested SNR.
    - The white noise burst's band-pass is applied as the Butterworth filters' frequency
      response (the same gain and phase as McGAN's ``lfilter`` calls), circularly, so there
      is no start-up transient.

    Each generator takes ranges as ``(low, high)`` (drawn uniformly per glitch) or a single
    number (fixed), a ``torch.Generator`` for reproducibility, and returns (n, L), plus the
    drawn parameters with ``return_params=True``.
"""

import math

import numpy as np
import scipy.signal
import torch
from torch import Tensor


def _uniform(bounds, n: int, generator, dtype) -> Tensor:
    """n draws from U(low, high) for ``bounds = (low, high)``; a single number is fixed."""
    if isinstance(bounds, (int, float)):
        return torch.full((n,), float(bounds), dtype=dtype)
    low, high = bounds
    return low + (high - low) * torch.rand(n, generator=generator, dtype=dtype)


def _time(n_glitch: int, sample_rate: float, duration: float, device, dtype) -> Tensor:
    n = int(round(duration * sample_rate))
    return (torch.arange(n, device=device, dtype=dtype) / sample_rate).expand(n_glitch, n)


def _peak_normalise(h: Tensor) -> Tensor:
    peak = h.abs().amax(dim=-1, keepdim=True)
    return h / peak.clamp_min(torch.finfo(h.dtype).tiny)


def quality_factor_to_tau(q, f0):
    """McGAN's ``quality_factor_conversion``: tau = Q / (sqrt(2) pi f0)."""
    return q / (math.sqrt(2) * math.pi * f0)


def _finish(h, params, dtype, device, return_params):
    h = _peak_normalise(h).to(dtype)
    if return_params:
        return h, {k: v.to(device) for k, v in params.items()}
    return h


def sine_gaussian(n: int, sample_rate: float, duration: float = 1.0, f0=(70.0, 250.0),
                  q=(3.0, 20.0), t0=(0.4, 0.6), generator: torch.Generator = None,
                  dtype=torch.float32, device=None, return_params: bool = False):
    """``n`` sine-Gaussians (McGAN's ``sinegaussian``), shape (n, duration * sample_rate)."""
    work = torch.float64
    f0, q, t0 = (_uniform(b, n, generator, work) for b in (f0, q, t0))
    phi = _uniform((0.0, 2 * math.pi), n, generator, work)
    tau = quality_factor_to_tau(q, f0)
    t = _time(n, sample_rate, duration, device, work)
    f0_, t0_, tau_, phi_ = (p.to(device).unsqueeze(-1) for p in (f0, t0, tau, phi))
    h = torch.exp(-((t - t0_) / tau_) ** 2) * torch.sin(2 * math.pi * f0_ * (t - t0_) + phi_)
    return _finish(h, dict(f0=f0, q=q, tau=tau, t0=t0, phi=phi), dtype, device, return_params)


def ringdown(n: int, sample_rate: float, duration: float = 1.0, f0=(70.0, 250.0),
             q=(9.0, 20.0), t0=(0.4, 0.6), generator: torch.Generator = None,
             dtype=torch.float32, device=None, return_params: bool = False):
    """``n`` damped sinusoids starting at t0 (McGAN's ``ringdown``)."""
    work = torch.float64
    f0, q, t0 = (_uniform(b, n, generator, work) for b in (f0, q, t0))
    phi = _uniform((0.0, 2 * math.pi), n, generator, work)
    tau = quality_factor_to_tau(q, f0)
    t = _time(n, sample_rate, duration, device, work)
    f0_, t0_, tau_, phi_ = (p.to(device).unsqueeze(-1) for p in (f0, t0, tau, phi))
    dt = t - t0_
    h = torch.where(dt > 0, torch.exp(-dt / tau_) * torch.sin(2 * math.pi * f0_ * dt + phi_),
                    torch.zeros_like(dt))
    return _finish(h, dict(f0=f0, q=q, tau=tau, t0=t0, phi=phi), dtype, device, return_params)


def _bandpass_response(n: int, sample_rate: float, band, order: int) -> Tensor:
    """Frequency response of McGAN's ``butter_filter``: low-pass at band[1], then high-pass at band[0]."""
    f = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    low = scipy.signal.butter(order, band[1], btype="low", fs=sample_rate, output="sos")
    high = scipy.signal.butter(order, band[0], btype="high", fs=sample_rate, output="sos")
    _, h_low = scipy.signal.sosfreqz(low, worN=f, fs=sample_rate)
    _, h_high = scipy.signal.sosfreqz(high, worN=f, fs=sample_rate)
    return torch.from_numpy(h_low * h_high)


def white_noise_burst(n: int, sample_rate: float, duration: float = 1.0, tau=(1e-4, 1e-2),
                      t0=(0.4, 0.6), band=(70.0, 250.0), order: int = 5,
                      generator: torch.Generator = None, dtype=torch.float32, device=None,
                      return_params: bool = False):
    """``n`` band-passed white noise bursts (McGAN's ``whitenoiseburst``).

    The envelope is McGAN's exp(-(t - t0)^2 / tau): ``tau`` is in s^2, the Gaussian width
    is sqrt(tau / 2).
    """
    work = torch.float64
    tau, t0 = (_uniform(b, n, generator, work) for b in (tau, t0))
    t = _time(n, sample_rate, duration, device, work)
    noise = torch.randn(n, t.shape[-1], generator=generator, dtype=work)
    response = _bandpass_response(t.shape[-1], sample_rate, band, order)
    noise = torch.fft.irfft(torch.fft.rfft(noise, dim=-1) * response, n=t.shape[-1], dim=-1)
    t0_, tau_ = (p.to(device).unsqueeze(-1) for p in (t0, tau))
    h = torch.exp(-(t - t0_) ** 2 / tau_) * noise.to(device)
    return _finish(h, dict(tau=tau, t0=t0), dtype, device, return_params)


def gaussian_blip(n: int, sample_rate: float, duration: float = 1.0, tau=(0.01, 0.05),
                  t0=(0.4, 0.6), generator: torch.Generator = None, dtype=torch.float32,
                  device=None, return_params: bool = False):
    """``n`` Gaussian pulses exp(-(t - t0)^2 / tau^2) (McGAN's ``gaussianblip``)."""
    work = torch.float64
    tau, t0 = (_uniform(b, n, generator, work) for b in (tau, t0))
    t = _time(n, sample_rate, duration, device, work)
    t0_, tau_ = (p.to(device).unsqueeze(-1) for p in (t0, tau))
    h = torch.exp(-((t - t0_) / tau_) ** 2)
    return _finish(h, dict(tau=tau, t0=t0), dtype, device, return_params)


ANALYTIC_GLITCHES = {
    "sine_gaussian": sine_gaussian,
    "ringdown": ringdown,
    "white_noise_burst": white_noise_burst,
    "gaussian_blip": gaussian_blip,
}


def analytic_glitch(kind: str, n: int, sample_rate: float, **kwargs):
    """``n`` glitches of one of ``ANALYTIC_GLITCHES`` (keyword arguments as that generator's)."""
    if kind not in ANALYTIC_GLITCHES:
        raise ValueError(f"Unknown glitch kind {kind!r}; choose from {sorted(ANALYTIC_GLITCHES)}")
    return ANALYTIC_GLITCHES[kind](n, sample_rate, **kwargs)
