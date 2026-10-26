#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : glitch_noise.py
Description     : Glitch injection in PyTorch, a port of gengli's ``add_glitch``.

Created on 2026-01-19 21:51:33

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:

    Inject whitened glitch shapes (from gengli, or a DeepExtractor glitch bank) into
    *coloured* strain at a chosen SNR, so that the pipeline's own whitening
    (e.g. ``sage.dsp.whiten.FIRWhitening``) is applied to glitch and noise alike:

        injector = GlitchInjector(sample_rate=2048, f_low=20.0)
        strain = injector(strain, glitches, asd, snr, t_inj=t_inj)   # (B, D, T)

    For each glitch:

    1. taper its ends with a Tukey window (as gengli's ``get_glitch``),
    2. place it at ``t_inj`` in a zero series as long as the strain,
    3. cut below ``f_low``, low-pass above ``f_high`` (zero-phase Butterworth, as gengli's
       ``low_pass_glitch``) and scale it to the requested optimal SNR
       (gengli's ``rescale_glitch``: rho^2 = 4 df sum |dt * rfft(w)|^2),
    4. colour it with the strain's ASD: h = irfft(rfft(w) * ASD),
    5. add it to the strain.

    Why the SNR is set on the whitened shape: for h = irfft(rfft(w) * ASD), the optimal
    SNR of h against that ASD, rho^2 = 4 df sum |dt * rfft(h)|^2 / S, equals
    4 df sum |dt * rfft(w)|^2 because ASD^2 / S = 1. The SNR is therefore exact in any
    whitening convention: PyCBC-whitened shapes (gengli, DeepExtractor, unnormalised)
    and gwpy-whitened data (``FIRWhitening``, unit variance) differ by a constant
    factor that the SNR scaling removes.

    Differences from gengli's ``examples/inject_glitch.py::add_glitch``:

    - gengli whitens the *noise* (PyCBC, PSD from the same noise), adds the glitch
      in the whitened domain and optionally recolours. Here the glitch is coloured and
      added to the strain directly, so no PSD is estimated and the result goes
      through the same whitening as everything else.
    - gengli tapers the whitened noise (Tukey, alpha=0.07). Here the noise is left
      untouched; keep ``t_inj`` away from the edges your whitening discards
      (``fduration / 2`` for ``FIRWhitening``).
    - gengli's optional low-pass (``fhigh``) is ``scipy.signal.filtfilt`` with a 3rd-order
      Butterworth filter. filtfilt is zero-phase with gain |H(f)|^2, and for scipy's
      digital Butterworth |H(f)|^2 = 1 / (1 + (tan(pi f / fs) / tan(pi f_c / fs))^(2N)),
      so here it is applied as that gain in the Fourier domain of the placed glitch (no
      filtfilt edge padding, nothing cut off by the end of the short glitch array).
      ``lowpass_order=None`` gives a hard cutoff instead. ``f_low`` is always a hard cutoff.
      Both are applied before the SNR is computed, so the SNR is that of the filtered glitch,
      as in gengli (which low-passes before ``rescale_glitch``).
    - gengli low-passes the raw glitch and tapers it afterwards; here the taper comes first,
      so the filter's response is not truncated by the window.

    All functions work on batches, broadcast over leading dimensions, and run on
    whatever device the inputs are on.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def tukey_window(n: int, alpha: float, device=None, dtype=torch.float64) -> Tensor:
    """Symmetric Tukey window, identical to ``scipy.signal.windows.tukey(n, alpha, sym=True)``.

    ``alpha = 0`` is a rectangular window, ``alpha = 1`` a Hann window.
    """
    if n == 1:
        return torch.ones(1, device=device, dtype=dtype)
    if alpha <= 0:
        return torch.ones(n, device=device, dtype=dtype)
    if alpha >= 1:
        alpha = 1.0
    x = torch.arange(n, device=device, dtype=dtype)
    width = int(math.floor(alpha * (n - 1) / 2.0))
    w = torch.ones(n, device=device, dtype=dtype)
    left = x[: width + 1]
    w[: width + 1] = 0.5 * (1 + torch.cos(math.pi * (-1 + 2.0 * left / alpha / (n - 1))))
    right = x[n - width - 1:]
    w[n - width - 1:] = 0.5 * (1 + torch.cos(math.pi * (-2.0 / alpha + 1 + 2.0 * right / alpha / (n - 1))))
    return w


def resample_glitch(glitch: Tensor, sample_rate: float, new_sample_rate: float) -> Tensor:
    """Change the sampling rate of glitch(es) along the last axis.

    Port of gengli's ``resample_glitch``, i.e. ``scipy.signal.resample`` (Fourier method):
    the new length is ``int(L * new_sample_rate / sample_rate)``.
    """
    n_old = glitch.shape[-1]
    n_new = int(n_old * new_sample_rate / sample_rate)
    if n_new == n_old:
        return glitch
    X = torch.fft.rfft(glitch, dim=-1)
    Y = torch.zeros(*X.shape[:-1], n_new // 2 + 1, dtype=X.dtype, device=X.device)
    n = min(n_new, n_old)
    nyq = n // 2 + 1
    Y[..., :nyq] = X[..., :nyq]
    if n % 2 == 0:
        # the shorter length's Nyquist bin, handled as scipy does
        if n_new < n_old:
            Y[..., n // 2] *= 2.0
        elif n_new > n_old:
            Y[..., n // 2] *= 0.5
    return torch.fft.irfft(Y, n=n_new, dim=-1) * (float(n_new) / float(n_old))


def lowpass_response(n: int, sample_rate: float, f_high: float, order=3, device=None,
                     dtype=torch.float64) -> Tensor:
    """Gain of a zero-phase low-pass on the ``rfftfreq(n, 1 / sample_rate)`` grid.

    With ``order`` N, the gain of ``scipy.signal.filtfilt`` with ``scipy.signal.butter(N, f_high,
    fs=sample_rate)``: |H(f)|^2 = 1 / (1 + (tan(pi f / fs) / tan(pi f_high / fs))^(2N)), which is
    1/4 (-6 dB) at ``f_high``. ``order=None`` is a hard cutoff (1 up to ``f_high``, 0 above).
    """
    nyq = sample_rate / 2.0
    if not 0 < f_high < nyq:
        raise ValueError(f"f_high must be between 0 and Nyquist ({nyq} Hz), got {f_high}")
    f = torch.fft.rfftfreq(n, d=1.0 / sample_rate, device=device, dtype=dtype)
    if order is None:
        return (f <= f_high).to(dtype)
    # tan(pi f / fs) -> inf at Nyquist, where the gain is 0
    ratio = torch.tan(math.pi * f / sample_rate) / math.tan(math.pi * f_high / sample_rate)
    gain = 1.0 / (1.0 + ratio ** (2 * int(order)))
    return torch.where(f >= nyq, torch.zeros_like(gain), gain)


def _band_mask(n: int, sample_rate: float, f_low, f_high, device, dtype, lowpass_order=None) -> Tensor:
    f = torch.fft.rfftfreq(n, d=1.0 / sample_rate, device=device, dtype=dtype)
    mask = torch.ones_like(f)
    if f_low is not None:
        mask = mask * (f >= f_low)
    if f_high is not None:
        mask = mask * lowpass_response(n, sample_rate, f_high, lowpass_order, device=device, dtype=dtype)
    return mask


def lowpass_glitch(glitch: Tensor, f_high: float, sample_rate: float, order=3) -> Tensor:
    """Zero-phase low-pass of glitch(es). Port of gengli's ``low_pass_glitch``.

    Applies the gain of ``filtfilt`` with an ``order``-th order Butterworth filter (gengli: 3)
    in the Fourier domain, i.e. as a circular filter: pad short glitches with zeros first if
    the filter's response must not wrap around. ``f_high=None`` returns the glitch unchanged.
    """
    if f_high is None:
        return glitch
    n = glitch.shape[-1]
    W = torch.fft.rfft(glitch, dim=-1)
    W = W * lowpass_response(n, sample_rate, f_high, order, device=glitch.device, dtype=W.real.dtype)
    return torch.fft.irfft(W, n=n, dim=-1)


def glitch_snr(glitch: Tensor, sample_rate: float, f_low: float = None, f_high: float = None) -> Tensor:
    """Optimal SNR of whitened glitch(es): rho^2 = 4 df sum |dt * rfft(w)|^2 over [f_low, f_high].

    Same formula as gengli's ``rescale_glitch`` (which agrees with PyCBC's ``sigmasq``),
    with an optional frequency band (hard cutoffs: this measures, it does not filter).
    """
    n = glitch.shape[-1]
    W = torch.fft.rfft(glitch, dim=-1) / sample_rate
    mask = _band_mask(n, sample_rate, f_low, f_high, glitch.device, W.real.dtype)
    df = sample_rate / n
    return torch.sqrt(4.0 * df * (W.abs() ** 2 * mask).sum(dim=-1))


def rescale_glitch(glitch: Tensor, snr, sample_rate: float, f_low: float = None, f_high: float = None,
                   lowpass_order=3) -> Tensor:
    """Scale whitened glitch(es) to the given optimal SNR. Port of gengli's ``rescale_glitch``.

    ``snr`` is a number or a tensor broadcastable to ``glitch.shape[:-1]``; an SNR of 0
    gives a zero glitch. Content below ``f_low`` is removed and content above ``f_high``
    low-passed first (see ``lowpass_response``; ``lowpass_order=None`` for a hard cutoff),
    so the returned glitch has exactly the requested SNR after filtering.
    """
    n = glitch.shape[-1]
    W = torch.fft.rfft(glitch, dim=-1)
    if f_low is not None or f_high is not None:
        W = W * _band_mask(n, sample_rate, f_low, f_high, glitch.device, W.real.dtype, lowpass_order)
        glitch = torch.fft.irfft(W, n=n, dim=-1)
    current = glitch_snr(glitch, sample_rate)
    snr = torch.as_tensor(snr, dtype=glitch.dtype, device=glitch.device)
    scale = torch.where(current > 0, snr / current.clamp_min(torch.finfo(glitch.dtype).tiny), torch.zeros_like(current))
    return glitch * scale.unsqueeze(-1)


def _resample_asd(asd: Tensor, n_freq: int) -> Tensor:
    """Linearly resample an ASD on a uniform [0, Nyquist] grid onto ``n_freq`` bins.

    Same as ``FIRWhitening.resample_asd``, so the colouring and the whitening see the same ASD.
    """
    if asd.shape[-1] == n_freq:
        return asd
    shape = asd.shape
    flat = asd.reshape(-1, 1, shape[-1])
    out = F.interpolate(flat, size=n_freq, mode="linear", align_corners=True)
    return out.reshape(*shape[:-1], n_freq)


def place_glitch(glitch: Tensor, length: int, start: Tensor) -> Tensor:
    """Put each glitch into a zero series of ``length`` samples, starting at sample ``start``.

    ``glitch``: (..., L); ``start``: integer tensor broadcastable to ``glitch.shape[:-1]``.
    """
    L = glitch.shape[-1]
    start = torch.as_tensor(start, device=glitch.device).long().expand(glitch.shape[:-1])
    if (start < 0).any() or (start + L > length).any():
        raise ValueError(f"Glitch of {L} samples placed outside the series of {length} samples "
                         f"(start between {int(start.min())} and {int(start.max())})")
    out = torch.zeros(*glitch.shape[:-1], length, dtype=glitch.dtype, device=glitch.device)
    idx = start.unsqueeze(-1) + torch.arange(L, device=glitch.device)
    return out.scatter_(-1, idx, glitch)


def add_glitch(
    strain: Tensor,
    glitch: Tensor,
    asd: Tensor,
    snr,
    sample_rate: float,
    t_inj=None,
    taper_alpha: float = 0.1,
    f_low: float = None,
    f_high: float = None,
    lowpass_order=3,
    return_glitch: bool = False,
):
    """Add whitened glitch(es) to coloured strain at a given SNR. Torch port of gengli's ``add_glitch``.

    Parameters
    ----------
    strain : Tensor, shape (..., T)
        Coloured strain, e.g. (B, D, T).
    glitch : Tensor, shape (..., L)
        Whitened glitch shapes at ``sample_rate``, one per series (same leading shape as
        ``strain``). Any normalisation: only the shape matters, the SNR sets the amplitude.
    asd : Tensor, shape (F,) or broadcastable to (..., F)
        One-sided ASD of the strain, in the strain's units / sqrt(Hz), on a uniform grid from
        0 Hz to Nyquist (any number of bins: it is interpolated onto the strain's grid).
    snr : float or Tensor broadcastable to ``strain.shape[:-1]``
        Optimal SNR of each injected glitch against ``asd``. 0 means no glitch in that series,
        e.g. to inject into one detector only.
    sample_rate : float
        Sample rate of both strain and glitch, in Hz.
    t_inj : float or Tensor broadcastable to ``strain.shape[:-1]``, optional
        Time of the glitch centre, in seconds from the start of the strain. Default: the
        middle of the strain, as in gengli.
    taper_alpha : float
        Tukey taper applied to each glitch before injection (gengli's default ``alpha``
        in ``get_glitch`` is 0.25, ``add_glitch`` uses 0.1). 0 for none.
    f_low : float, optional
        Hard high-pass cutoff (Hz) applied to the glitch before setting its SNR.
    f_high : float, optional
        Low-pass cutoff (Hz) applied to the glitch before setting its SNR, as gengli's
        ``fhigh`` (gengli suggests 250 Hz for blips).
    lowpass_order : int or None
        Order of the zero-phase Butterworth low-pass at ``f_high`` (gengli: 3). None for a
        hard cutoff.
    return_glitch : bool
        Also return the coloured glitch that was added.

    Returns
    -------
    Tensor, shape (..., T): strain + coloured glitch (and the coloured glitch if requested).
    """
    T, L = strain.shape[-1], glitch.shape[-1]
    if glitch.shape[:-1] != strain.shape[:-1]:
        raise ValueError(f"glitch leading shape {tuple(glitch.shape[:-1])} must match "
                         f"strain leading shape {tuple(strain.shape[:-1])}")
    work = torch.float64 if strain.dtype == torch.float64 else torch.float32
    g = glitch.to(dtype=work, device=strain.device)

    if taper_alpha:
        g = g * tukey_window(L, taper_alpha, device=g.device, dtype=work)

    # 1. placement: the glitch centre at t_inj (default: middle of the strain)
    if t_inj is None:
        start = torch.full(strain.shape[:-1], T // 2 - L // 2, device=strain.device)
    else:
        t_inj = torch.as_tensor(t_inj, dtype=torch.float64, device=strain.device)
        start = torch.round(t_inj * sample_rate).long() - L // 2
    w = place_glitch(g, T, start)

    # 2. band and SNR, on the whitened shape (exact in any whitening convention, see module doc)
    w = rescale_glitch(w, snr, sample_rate, f_low=f_low, f_high=f_high, lowpass_order=lowpass_order)

    # 3. colour with the strain's ASD on the strain's frequency grid
    asd = torch.as_tensor(asd, device=strain.device).to(work)
    asd = _resample_asd(asd, T // 2 + 1)
    h = torch.fft.irfft(torch.fft.rfft(w, dim=-1) * asd, n=T, dim=-1)

    out = strain + h.to(strain.dtype)
    return (out, h.to(strain.dtype)) if return_glitch else out


class GlitchInjector(torch.nn.Module):
    """``add_glitch`` with fixed settings, as a module.

    Example
    -------
    >>> injector = GlitchInjector(sample_rate=2048, f_low=20.0)
    >>> x = injector(strain, glitches, asd, snr=torch.tensor([[12.0, 0.0]]), t_inj=1.3)
    """

    def __init__(self, sample_rate: float, taper_alpha: float = 0.1,
                 f_low: float = None, f_high: float = None, lowpass_order=3):
        super().__init__()
        self.sample_rate = sample_rate
        self.taper_alpha = taper_alpha
        self.f_low = f_low
        self.f_high = f_high
        self.lowpass_order = lowpass_order

    def forward(self, strain: Tensor, glitch: Tensor, asd: Tensor, snr, t_inj=None,
                return_glitch: bool = False):
        return add_glitch(strain, glitch, asd, snr, self.sample_rate, t_inj=t_inj,
                          taper_alpha=self.taper_alpha, f_low=self.f_low, f_high=self.f_high,
                          lowpass_order=self.lowpass_order, return_glitch=return_glitch)
