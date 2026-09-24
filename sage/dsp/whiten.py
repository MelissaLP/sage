#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : whiten.py
Description   : Short description of the file

Created on 2026-01-19 16:26:37

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# LOCAL
from sage.data.psd import get_fiducial_psds
from sage.core.config import get_cfg, get_data_cfg
from sage.core.pipeline import GWBatch, Grid, ProcessingState


class FiducialWhitening(torch.nn.Module):
    """
    Whiten frequency-domain strain using fixed, detector-specific fiducial PSDs.

    The whitening kernel is derived once from pre-computed fiducial ASDs and
    stored as a registered buffer so it moves to the correct device
    automatically and is included in ``torch.compile`` graphs.

    Pipeline (per sample)
    ---------------------
    1. Multiply FD strain by the whitening kernel:
       ``X_white = X_fd * whitening``  where
       ``whitening[d, f] = 2 Δf / (√0.5 · ASD[d, f])``.
    2. Convert back to time domain via inverse real FFT.
    3. Strip the corrupted edge samples introduced by the Welch PSD
       estimation window (``padding_nsamples`` on each side).

    The ``@torch.no_grad()`` decorator on :meth:`forward` means this
    module **severs the autograd graph**.  Adversarial perturbations or any
    gradient-based optimisation must therefore operate on the *output* of
    this module, not on its FD input.

    Parameters
    ----------
    **kwargs
        Forwarded to ``nn.Module.__init__``.

    Attributes
    ----------
    whitening : torch.Tensor, shape ``(D, F)``
        Per-detector, per-frequency whitening kernel (registered buffer).
    corrupted_len : int
        Number of samples removed from each end of the whitened time series.

    Input / Output
    --------------
    forward(X_fd) : (B, D, F) complex64 → (B, D, T_valid) float32
        where ``T_valid = seq_len - 2 * corrupted_len``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setup configs
        cfg = get_cfg()
        data_cfg = get_data_cfg()

        # Get fiducial psds
        fiducial_psds = get_fiducial_psds()

        self.device = cfg.device

        self.seq_len = data_cfg.padded_length_in_nsamples
        self.sample_rate = data_cfg.sample_rate
        self.corrupted_len = data_cfg.padding_nsamples

        # Frequency resolution
        delta_f = data_cfg.sample_rate / self.seq_len
        self.delta_f = torch.tensor(delta_f).to(device=cfg.device)

        # Whitening
        whitening = 2 * self.delta_f / (math.sqrt(0.5) * fiducial_psds)
        # Final whitening moved to device
        whitening = whitening.to(device=cfg.device)

        # Register as buffer for compile friendliness
        self.register_buffer("whitening", whitening)  # (D, F)

    def remove_corrupted(self, x):
        """
        Strip edge samples corrupted by the Welch PSD estimation window.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, T)``
            Whitened time-domain strain (full length, including corrupted ends).

        Returns
        -------
        torch.Tensor, shape ``(B, D, T - 2 * corrupted_len)``
            Valid central samples only.
        """
        # x_td_white or x: (B, D, T)
        T = x.shape[-1]
        start = self.corrupted_len
        end = T - self.corrupted_len
        return x[..., start:end]

    @torch.no_grad()
    def forward(self, input):
        """
        Whiten frequency-domain strain.

        Accepts either a raw tensor (legacy path) or a :class:`GWBatch`
        (state-tracked path).  The behaviour depends on the grid type:

        * **FD_UNIFORM** — whiten → IFFT → strip corrupted edges → return
          ``GWBatch`` with ``TD_UNIFORM`` state (real float32, shape
          ``(B, D, T_valid)``).
        * **FD_COARSE** — whiten at the coarse frequency indices using
          ``batch.coarse_indices`` → return ``GWBatch`` with ``FD_COARSE``
          whitened state (complex, shape ``(B, D, N_coarse)``).
          No IFFT is applied — the non-uniform grid cannot be IFFTed.
        * **Raw tensor** (no GWBatch) — treated as FD_UNIFORM and the raw
          whitened TD tensor is returned for backward compatibility.

        Parameters
        ----------
        input : torch.Tensor or GWBatch
            FD strain ``(B, D, F)`` complex, or a GWBatch wrapping it.

        Returns
        -------
        GWBatch or torch.Tensor
            GWBatch when input is a GWBatch; raw float32 tensor otherwise.
        """
        if isinstance(input, GWBatch):
            return self._forward_batch(input)
        # Legacy raw-tensor path: FD → whitened TD (backward compatible)
        return self._whiten_to_td(input)

    def _whiten_to_td(self, X_fd: torch.Tensor) -> torch.Tensor:
        """Whiten FD strain and convert to valid TD float32."""
        X_white = X_fd * self.whitening.unsqueeze(0)
        x_td    = torch.fft.irfft(X_white, dim=-1, norm="forward") * self.delta_f
        return self.remove_corrupted(x_td)

    def _forward_batch(self, batch: GWBatch) -> GWBatch:
        if batch.state.grid == Grid.FD_COARSE:
            # Non-uniform grid: whiten at the exact coarse indices only.
            # coarse_indices are integer offsets into the full 0→Nyquist
            # whitening buffer — guaranteed to be exact integer multiples of
            # delta_f, so no interpolation is needed.
            idx = batch.coarse_indices                         # (N_coarse,)
            whitening_coarse = self.whitening[:, idx]          # (D, N_coarse)
            X_white = batch.data * whitening_coarse.unsqueeze(0)
            new_state = batch.state.after_whiten()
            return GWBatch(X_white, new_state, batch.freqs, batch.coarse_indices)

        # FD_UNIFORM: existing whiten → IFFT → strip path, wrapped in GWBatch
        x_td      = self._whiten_to_td(batch.data)
        new_state = batch.state.after_whiten().after_ifft()
        return GWBatch(x_td, new_state, freqs=None, coarse_indices=None)


class FIRWhitening(torch.nn.Module):
    """
    Whiten a time-domain strain via inverse spectrum truncation.

    This reproduces gwpy's *current* ``TimeSeries.whiten()`` — FIR filter
    design via ``gwpy.signal.filter_design.fir_from_transfer`` followed by
    ``TimeSeries.convolve`` — which is a structurally different algorithm
    from the legacy Welch/overlap-add whitener that ``SelfASDWhitening``
    reproduces. The two share a name in gwpy's history but not a method.

    Pipeline (per sample), mirroring gwpy's
    ``whiten`` / ``fir_from_transfer`` / ``truncate_transfer`` /
    ``truncate_impulse`` / ``convolve`` exactly:

    1. Take the supplied ASD and resample it (linearly, if not already
       on that grid — see ``resample_asd``) onto a grid with resolution
       ``Δf = sample_rate / T`` (``T`` = input length in samples) — the
       same grid as a full-length rFFT of the input itself. This is far
       finer than any Welch segment grid, and is recomputed per call
       since it depends on ``T``.
    2. Build the transfer function ``H = 1 / ASD`` (invalid/near-zero
       bins excised rather than amplified — see ``eps``; gwpy's own code
       has no such guard and would propagate ``inf``/``nan`` on a
       literally-zero ASD bin, exactly as diagnosed earlier for
       zero/out-of-band ASD values).
    3. Smoothly zero/taper ``H`` (``_truncate_transfer``): hard-zero the
       first ``ncorner`` bins (``ncorner = int(highpass * T /
       sample_rate)`` if ``highpass`` is given, else 0), then taper the
       remainder with a Planck window (``nleft=nright=5``, exactly
       gwpy's ``truncate_transfer`` / ``gwpy.signal.window.planck``).
    4. Inverse-FFT the tapered transfer into a length-``T`` impulse
       response, then keep only the outer ``ntaps // 2`` samples on each
       end (``_truncate_impulse``, using a fixed length-``ntaps`` window
       precomputed at construction), zeroing everything in between —
       exactly gwpy's ``truncate_impulse``.
    5. Re-center via ``torch.roll`` into a causal length-``ntaps`` FIR
       kernel (``_fir_from_transfer``) — one filter per detector (and
       per batch element, if the supplied ASD varies per sample).
    6. Detrend the *whole* input once (single global mean removal, not
       per-frame — unlike ``SelfASDWhitening``), taper its first/last
       ``ntaps // 2`` samples with the same length-``ntaps`` window, and
       convolve with the FIR kernel via a full linear convolution
       followed by scipy/gwpy's ``mode="same"`` centering (implemented
       with a grouped ``conv1d``). Note: gwpy's ``convolve`` switches to
       a *chunked* overlap-save algorithm for very long inputs purely as
       a memory optimisation — it computes the identical linear
       convolution, so it is not separately reproduced here.
    7. Scale the result by ``sqrt(2 / sample_rate)`` and strip
       ``ntaps // 2`` samples from each end — the filter settle-in
       region, matching gwpy's stated ``0.5 * fduration`` corruption
       exactly.

    This module currently supports only an externally supplied ASD
    (``forward(x, asd=...)``). gwpy's own default self-estimation inside
    ``whiten()`` uses a *median*-averaged periodogram (not a mean/Welch
    average) with a specific bias-correction factor — a separate,
    non-trivial addition not implemented here.

    Parameters
    ----------
    sample_rate : float
        Sample rate of the input time series, in Hz.
    window : str, optional, default: "hann"
        Window name, built with the *periodic* (``fftbins=True``)
        convention that ``scipy.signal.get_window`` (and hence gwpy)
        uses by default — note this differs from the symmetric
        convention ``SelfASDWhitening`` uses.
    detrend : {"constant", None}, optional, default: "constant"
        Detrending applied once to the *whole* input before convolving
        (not per-frame, unlike ``SelfASDWhitening``). Only mean removal
        is supported.
    fduration : float, optional, default: 2.0
        Duration in seconds of the FIR whitening filter.
        ``ntaps = fduration * sample_rate`` must be even.
    highpass : float, optional
        Highpass corner frequency in Hz. ``None`` disables highpassing.
    eps : float, optional
        Validity threshold for inverting the ASD — see
        ``SelfASDWhitening``'s ``eps`` docstring for the identical
        rationale (excise, don't amplify, invalid/out-of-band bins).
    **kwargs
        Forwarded to ``nn.Module.__init__``.

    Attributes
    ----------
    ntaps : int
        Number of taps in the FIR whitening filter,
        ``round(fduration * sample_rate)``.
    pad : int
        ``ntaps // 2`` — samples tapered at each input edge before
        convolving, and stripped from each output edge afterward.

    Input / Output
    --------------
    forward(x, asd) : (B, D, T) float32, (D, F_in) or (B, D, F_in) float32
        → (B, D, T - 2 * pad) float32
        ``asd`` is auto-resampled (linearly) onto ``F = T // 2 + 1`` bins
        if it isn't already on that grid — see ``resample_asd``.
    """

    def __init__(
        self,
        sample_rate: float,
        window: str = "hann",
        detrend: str = "constant",
        fduration: float = 2.0,
        highpass: float = None,
        eps: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if detrend not in ("constant", None):
            raise NotImplementedError(
                f"detrend={detrend!r} is not supported; only 'constant' or None."
            )
        self.detrend = detrend
        self.eps = eps
        self.sample_rate = sample_rate
        self.highpass = highpass

        self.ntaps = int(round(fduration * sample_rate))
        if self.ntaps % 2 != 0:
            raise ValueError("fduration * sample_rate must be even (ntaps).")
        self.pad = self.ntaps // 2  # == ceil(ntaps / 2) since ntaps is even

        # Length-ntaps window, PERIODIC convention (scipy/gwpy default),
        # reused for both _truncate_impulse and the input edge-taper in
        # forward -- exactly as gwpy reuses get_window(window, fir.size)
        # in both truncate_impulse and convolve.
        win_ntaps = self._build_window(window, self.ntaps)
        self.register_buffer("_ntaps_window", win_ntaps)

    @staticmethod
    def _build_window(name: str, n: int) -> torch.Tensor:
        """Build a 1-D window with the PERIODIC convention (matches
        scipy.signal.get_window's default fftbins=True), not the
        symmetric convention used elsewhere."""
        if name in (None, "boxcar", "rectangular"):
            return torch.ones(n)
        try:
            window_fn = getattr(torch, f"{name}_window")
        except AttributeError as exc:
            raise ValueError(f"Unsupported window {name!r}.") from exc
        return window_fn(n, periodic=True)

    def _invert_asd(self, asd: torch.Tensor) -> torch.Tensor:
        """
        Invert an ASD, excising (rather than amplifying) invalid bins.

        Deviates deliberately from gwpy's raw ``1 / asd.value`` (which
        has no such guard): a literal 0 or negative ASD bin would
        otherwise produce ``inf``, which the subsequent ``irfft``
        spreads into ``nan`` across the whole impulse response.

        Parameters
        ----------
        asd : torch.Tensor

        Returns
        -------
        torch.Tensor, same shape as ``asd``
        """
        threshold = self.eps if self.eps is not None else torch.finfo(asd.dtype).tiny
        valid = asd > threshold
        invasd = torch.zeros_like(asd)
        invasd[valid] = 1.0 / asd[valid]
        return invasd

    @staticmethod
    def _planck_taper(nsamp: int, nleft: int, nright: int, device, dtype) -> torch.Tensor:
        """
        Vectorised reproduction of ``gwpy.signal.window.planck``.

        Parameters
        ----------
        nsamp : int
            Length of the output window.
        nleft, nright : int
            Number of samples tapered at the left/right ends.

        Returns
        -------
        torch.Tensor, shape ``(nsamp,)``
        """
        w = torch.ones(nsamp, device=device, dtype=dtype)
        if nleft:
            w[0] = 0.0
            if nleft > 1:
                k = torch.arange(1, nleft, device=device, dtype=dtype)
                zleft = nleft * (1.0 / k + 1.0 / (k - nleft))
                w[1:nleft] = w[1:nleft] * torch.sigmoid(-zleft)
        if nright:
            w[nsamp - 1] = 0.0
            if nright > 1:
                k = torch.arange(1, nright, device=device, dtype=dtype)
                zright = -nright * (1.0 / (k - nright) + 1.0 / k)
                w[nsamp - nright:nsamp - 1] = (
                    w[nsamp - nright:nsamp - 1] * torch.sigmoid(-zright)
                )
        return w

    def _truncate_transfer(self, transfer: torch.Tensor, ncorner: int) -> torch.Tensor:
        """
        Smoothly zero the edges of a (complex) transfer function.

        Parameters
        ----------
        transfer : torch.Tensor, shape ``(..., F)``
        ncorner : int
            Number of low-frequency bins to hard-zero.

        Returns
        -------
        torch.Tensor, same shape as ``transfer``
        """
        nsamp = transfer.shape[-1]
        out = transfer.clone()
        if ncorner:
            out[..., :ncorner] = 0
        taper = self._planck_taper(
            nsamp - ncorner, nleft=5, nright=5,
            device=transfer.device, dtype=transfer.real.dtype,
        )
        out[..., ncorner:nsamp] = out[..., ncorner:nsamp] * taper
        return out

    def _truncate_impulse(self, impulse: torch.Tensor) -> torch.Tensor:
        """
        Keep only the outer ``ntaps // 2`` samples of an impulse
        response on each end, tapered, zeroing everything in between —
        exactly gwpy's ``truncate_impulse``.

        Parameters
        ----------
        impulse : torch.Tensor, shape ``(..., T)``

        Returns
        -------
        torch.Tensor, same shape as ``impulse``
        """
        out = impulse.clone()
        trunc_start = self.ntaps // 2
        trunc_stop = out.shape[-1] - trunc_start
        win = self._ntaps_window
        out[..., 0:trunc_start] = out[..., 0:trunc_start] * win[trunc_start:self.ntaps]
        out[..., trunc_stop:] = out[..., trunc_stop:] * win[0:trunc_start]
        out[..., trunc_start:trunc_stop] = 0
        return out

    def _fir_from_transfer(self, transfer: torch.Tensor, ncorner: int) -> torch.Tensor:
        """
        Design a length-``ntaps`` FIR filter from a transfer function,
        reproducing gwpy's ``fir_from_transfer`` exactly.

        Parameters
        ----------
        transfer : torch.Tensor, shape ``(..., F)``
            Target transfer function (e.g. ``1 / ASD``), on a grid with
            ``F = T // 2 + 1`` bins matching the full input length ``T``.
        ncorner : int
            Number of low-frequency bins to hard-zero (highpass).

        Returns
        -------
        torch.Tensor, shape ``(..., ntaps)``
        """
        transfer = self._truncate_transfer(transfer, ncorner)
        T = 2 * (transfer.shape[-1] - 1)
        impulse = torch.fft.irfft(transfer, n=T, dim=-1)
        impulse = self._truncate_impulse(impulse)
        fir = torch.roll(impulse, shifts=self.ntaps // 2 - 1, dims=-1)[..., : self.ntaps]
        return fir

    @staticmethod
    def resample_asd(asd: torch.Tensor, n_freq: int) -> torch.Tensor:
        """
        Linearly resample an ASD onto a different uniform frequency grid
        spanning the same ``[0, Nyquist]`` range (i.e. same sample rate).

        Identical to ``SelfASDWhitening.resample_asd``, duplicated here so
        ``FIRWhitening`` has no cross-class dependency. Called
        automatically by ``forward`` whenever the supplied ASD isn't
        already on the ``T // 2 + 1``-bin grid it needs.

        Parameters
        ----------
        asd : torch.Tensor, shape ``(..., F_in)``
            ASD sampled on a uniform grid from 0 Hz to Nyquist with
            ``F_in`` bins. Must share the same Nyquist frequency (i.e.
            the same underlying sample rate) as the target grid.
        n_freq : int
            Number of bins in the output grid, also spanning 0 to Nyquist.

        Returns
        -------
        torch.Tensor, shape ``(..., n_freq)``
        """
        orig_shape = asd.shape
        flat = asd.reshape(-1, 1, orig_shape[-1])  # (N, 1, F_in)
        resampled = F.interpolate(flat, size=n_freq, mode="linear", align_corners=True)
        return resampled.reshape(*orig_shape[:-1], n_freq)

    def forward(self, x: torch.Tensor, asd: torch.Tensor) -> torch.Tensor:
        """
        Whiten ``x`` against a supplied ASD via inverse spectrum truncation.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, T)``
        asd : torch.Tensor, shape ``(D, F_in)`` or ``(B, D, F_in)``
            One-sided ASD on any uniform ``[0, Nyquist]`` grid sharing
            this module's ``sample_rate``. If ``F_in != T // 2 + 1``, it
            is automatically resampled (linearly, via ``resample_asd``)
            onto the grid this forward pass actually needs — no manual
            pre-resampling required.

        Returns
        -------
        torch.Tensor, shape ``(B, D, T - 2 * pad)``
        """
        B, D, T = x.shape
        expected_bins = T // 2 + 1
        asd = asd.to(device=x.device, dtype=x.dtype)
        if asd.shape[-1] != expected_bins:
            asd = self.resample_asd(asd, expected_bins)
        transfer = self._invert_asd(asd)  # H = 1 / ASD

        duration = T / self.sample_rate
        new_df = 1.0 / duration
        ncorner = int(self.highpass / new_df) if self.highpass else 0

        fir = self._fir_from_transfer(transfer, ncorner)  # (D, ntaps) or (B, D, ntaps)
        if fir.dim() == 2:
            fir = fir.unsqueeze(0).expand(B, -1, -1)  # shared across batch

        # -- condition the input: single global detrend + edge taper --
        if self.detrend == "constant":
            x = x - x.mean(dim=-1, keepdim=True)
        pad = self.pad
        x = x.clone()
        x[..., :pad] = x[..., :pad] * self._ntaps_window[:pad]
        x[..., -pad:] = x[..., -pad:] * self._ntaps_window[-pad:]

        # -- full linear convolution via grouped conv1d, then scipy's
        # mode="same" centering (mathematically identical to gwpy's
        # convolve, whether or not IT took the chunked overlap-save path) --
        kernel = torch.flip(fir, dims=[-1])  # conv1d is cross-correlation
        x_flat = x.reshape(1, B * D, T)
        kernel_flat = kernel.reshape(B * D, 1, self.ntaps)
        full = F.conv1d(x_flat, kernel_flat, padding=self.ntaps - 1, groups=B * D)
        full = full.reshape(B, D, -1)  # length T + ntaps - 1

        start = (self.ntaps - 1) // 2
        conv = full[..., start:start + T]  # scipy "same"-mode centering

        out = conv * math.sqrt(2.0 / self.sample_rate)
        return out[..., pad:T - pad]
