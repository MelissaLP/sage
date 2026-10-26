#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : analytic_glitches.py
Description     : Every analytic glitch shape in torch: McGAN's bursts, Antiglitch and ArchEnemy's scattered-light arch.

GitHub Repository: NULL

Documentation:

    All generators share one interface: they take ranges as ``(low, high)`` (drawn uniformly
    per glitch), a single number (fixed) or a tensor of n per-glitch values, a
    ``torch.Generator`` for reproducibility, and return (n, duration * sample_rate), plus the
    drawn parameters with ``return_params=True``. ``ANALYTIC_GLITCHES`` lists them by name
    (the names ``GlitchPopulation`` configs use as ``generator``). The phase ``phi`` multiplies
    a sine: phi = pi / 2 is a glitch symmetric about t0.

    The shapes are meant for ``GlitchInjector``, which sets their amplitude by SNR:

        glitch = sine_gaussian(B * D, sample_rate=2048).view(B, D, -1)
        strain = GlitchInjector(2048, f_low=20.0)(strain, glitch, asd, snr, t_inj=t_inj)

    ``GlitchInjector`` centres the glitch *array* at ``t_inj``; the glitch itself sits at t0
    inside it, so pass ``t0=duration / 2`` to put it exactly at ``t_inj``.

    gengli's GAN blips are not analytic: they are in ``gengli_glitches``.

    1. McGAN's bursts
    -----------------

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

    2. Antiglitch (``antiglitch_glitch``)
    -------------------------------------

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

    Differences from Antiglitch:

    - Any sample rate and duration (Antiglitch: 1/8 s at 8192 Hz, 513 frequencies).
    - The phase ``phi`` follows the sine convention above, so it works with
      ``GlitchPopulation``'s ``phases``. Antiglitch's real amplitude (psi = 0) is phi = pi / 2;
      in general psi = phi - pi / 2.
    - The glitch is placed at ``t0`` from the start of the array (Antiglitch: from its centre).
    - Peak |h| of 1, like the other shapes; the SNR sets the amplitude.

    The shape is built in the frequency domain, so it wraps around the array: a narrow bump
    (large gbw) at low f0 rings for long, so give it a long enough ``duration``.

    3. Scattered-light arches (``scattered_light_arch``)
    ----------------------------------------------------

    Model: A. Tolley et al., "ArchEnemy: Removing scattered-light glitches from
    gravitational wave data", arXiv:2301.10491 (2023); reference code at
    https://github.com/ArthurTolley/ArchEnemy, ``archenemy/search/artefact_template.py``.

    Written from the paper's equation below; it copies no ArchEnemy code (GPL-3.0).

    One arch of fringe frequency f_f and time period T, for t in [-T/2, T/2):

        h(t) = sin((f_f / f_rep) sin(2 pi f_rep t) + phi),    f_rep = 1 / (2 T)

    Its instantaneous frequency is f_f cos(pi t / T): it rises from 0 to f_f at the centre
    and falls back to 0, the arch seen in Q-scans. As in ArchEnemy, each arch is tapered with
    a Tukey window (alpha = 0.2) over its own samples, and the defaults are the ranges of
    ArchEnemy's template bank example: f_f in [20, 80] Hz, T in [1.8, 5.5] s (at 2048 Hz).

    Arches last seconds: use a strain window longer than the longest T (plus what the whitening
    discards at the edges), or the arch is cut.

    Differences from ArchEnemy's code: batched, any sample rate, one random phase per arch,
    and the arch placed at ``t0`` inside an array of ``duration`` seconds (ArchEnemy's padding
    step does not run as written). Unit amplitude, not peak-normalised: the SNR sets it.
"""

import math

import numpy as np
import scipy.signal
import torch
from torch import Tensor


def _uniform(bounds, n: int, generator, dtype) -> Tensor:
    """n draws from U(low, high) for ``bounds = (low, high)``; a single number is fixed; a tensor
    of n values is used as given (per-glitch values, e.g. drawn by a population)."""
    if isinstance(bounds, Tensor):
        if bounds.numel() != n:
            raise ValueError(f"Expected {n} per-glitch values, got {bounds.numel()}")
        return bounds.reshape(n).to(dtype)
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


# -- McGAN's bursts --------------------------------------------------------------------------

def sine_gaussian(n: int, sample_rate: float, duration: float = 1.0, f0=(70.0, 250.0),
                  q=(3.0, 20.0), t0=(0.4, 0.6), phi=(0.0, 2 * math.pi), generator: torch.Generator = None,
                  dtype=torch.float32, device=None, return_params: bool = False):
    """``n`` sine-Gaussians (McGAN's ``sinegaussian``), shape (n, duration * sample_rate)."""
    work = torch.float64
    f0, q, t0 = (_uniform(b, n, generator, work) for b in (f0, q, t0))
    phi = _uniform(phi, n, generator, work)
    tau = quality_factor_to_tau(q, f0)
    t = _time(n, sample_rate, duration, device, work)
    f0_, t0_, tau_, phi_ = (p.to(device).unsqueeze(-1) for p in (f0, t0, tau, phi))
    h = torch.exp(-((t - t0_) / tau_) ** 2) * torch.sin(2 * math.pi * f0_ * (t - t0_) + phi_)
    return _finish(h, dict(f0=f0, q=q, tau=tau, t0=t0, phi=phi), dtype, device, return_params)


def ringdown(n: int, sample_rate: float, duration: float = 1.0, f0=(70.0, 250.0),
             q=(9.0, 20.0), t0=(0.4, 0.6), phi=(0.0, 2 * math.pi), generator: torch.Generator = None,
             dtype=torch.float32, device=None, return_params: bool = False):
    """``n`` damped sinusoids starting at t0 (McGAN's ``ringdown``)."""
    work = torch.float64
    f0, q, t0 = (_uniform(b, n, generator, work) for b in (f0, q, t0))
    phi = _uniform(phi, n, generator, work)
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


# -- Antiglitch ------------------------------------------------------------------------------

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


# -- ArchEnemy scattered-light arches --------------------------------------------------------

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


ANALYTIC_GLITCHES = {
    "sine_gaussian": sine_gaussian,
    "ringdown": ringdown,
    "white_noise_burst": white_noise_burst,
    "gaussian_blip": gaussian_blip,
    "antiglitch_glitch": antiglitch_glitch,
    "scattered_light_arch": scattered_light_arch,
}


def analytic_glitch(kind: str, n: int, sample_rate: float, **kwargs):
    """``n`` glitches of one of ``ANALYTIC_GLITCHES`` (keyword arguments as that generator's)."""
    if kind not in ANALYTIC_GLITCHES:
        raise ValueError(f"Unknown glitch kind {kind!r}; choose from {sorted(ANALYTIC_GLITCHES)}")
    return ANALYTIC_GLITCHES[kind](n, sample_rate, **kwargs)
