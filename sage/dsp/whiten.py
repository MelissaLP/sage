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



class TimeSeriesWhitener(torch.nn.Module):
    """
    Whiten a time-domain strain against an ASD estimated from itself.

    Unlike a fixed-fiducial whitener, this module has no pre-computed PSD
    buffer: for every ``forward`` call it (1) estimates a Welch-averaged
    ASD directly from the input segment, then (2) whitens the *same*
    segment against that self-estimated ASD using windowed, overlapping
    FFT frames reconstructed via overlap-add (OLA). This mirrors
    ``gwpy.timeseries.TimeSeries.whiten`` frame-for-frame, but batched
    and vectorised across ``(B, D)`` in torch.

    Pipeline (per sample)
    ---------------------
    1. Frame the input into overlapping windows of length ``nfft`` at
       stride ``nstride = nfft - noverlap`` (``self._frame``).
    2. Detrend + window every frame and rFFT them **once** (shared
       between steps 3 and 4 below — no transform is ever computed twice).
    3. If self-estimating: Welch-average the scaled periodograms of that
       one rFFT across frames to obtain a PSD, then ``ASD = sqrt(PSD)``
       (``self._psd_from_Xf``). If an ``asd`` was supplied instead, skip
       this step entirely.
    4. Whiten every frame in the frequency domain by ``1 / ASD``
       (invalid/out-of-band bins excised rather than amplified, see
       ``eps``), inverse-FFT back to the time domain, and reassemble the
       full-length signal via a single fused overlap-add
       (``self._overlap_add``, implemented with ``torch.nn.functional.fold``)
       — numerically the same accumulation as gwpy's own
       ``out[i0:i1] += irfft(...)`` loop, just executed as one vectorised,
       ``torch.compile``-friendly op instead of a Python loop over segments.

    This module does **not** wrap ``forward`` in ``@torch.no_grad()``.
    Because the ASD is estimated from the very input being whitened, the
    whole operation (framing → Welch ASD → inverse-ASD multiply →
    overlap-add) is left differentiable end to end so that gradients can
    flow back to the raw input if desired. Pass ``detach_asd=True`` to
    ``forward`` to stop gradients through the ASD estimate only (i.e.
    treat the ASD as fixed for that call), which is the closer analogue
    to a fiducial whitener's severed graph.

    Parameters
    ----------
    fftlength : float
        Length in seconds of each Welch/whitening segment.
    sample_rate : float
        Sample rate of the input time series, in Hz.
    overlap : float, optional, default: 0.0
        Overlap in seconds between neighbouring segments.
    window : str, optional, default: "hann"
        Name of a window available via ``torch.<window>_window``
        (e.g. ``"hann"``, ``"hamming"``), or ``"boxcar"`` for no window.
    detrend : {"constant", None}, optional, default: "constant"
        Per-frame detrending applied before windowing. Only mean removal
        is currently supported; anything else raises ``NotImplementedError``.
    corrupted_len : int, optional
        Number of samples to strip from each end of the reconstructed
        output to remove edge effects from the finite-length whitening
        filter (the first/last frames only have one-sided overlap
        support). Defaults to ``nfft // 2``, the standard half-filter-
        length convention.
    eps : float, optional
        Validity threshold applied to any ASD (self-estimated or
        externally supplied) before inverting it. Bins with
        ``asd <= eps`` are treated as *out of band* (e.g. a literal 0 at
        DC, or a PSD that is only defined over some analysis band and
        zero-padded elsewhere) and are excised — given ``invasd = 0`` —
        rather than divided, since dividing by a near-zero value would
        instead hugely *amplify* whatever signal power happens to sit at
        that frequency, which is almost never what you want and can
        blow up into ``inf``/``nan`` once summed across segments in the
        overlap-add. Defaults to ``torch.finfo(dtype).tiny`` at call
        time if left as ``None``, which only excises exact zeros/
        negatives; pass something larger (e.g. matching your ASD's
        analysis-band cutoff) if your "zero" region isn't exactly zero.
    **kwargs
        Forwarded to ``nn.Module.__init__``.

    Attributes
    ----------
    window : torch.Tensor, shape ``(nfft,)``
        Analysis/synthesis window (registered buffer).
    nfft : int
        Samples per segment, ``round(fftlength * sample_rate)``.
    noverlap : int
        Samples of overlap between segments, ``round(overlap * sample_rate)``.
    nstride : int
        Hop size between segment starts, ``nfft - noverlap``.
    corrupted_len : int
        Number of samples removed from each end of the OLA output.

    Input / Output
    --------------
    forward(x) : (B, D, T) float32 → (B, D, T_valid) float32
        where ``T_valid = nsteps * nstride + noverlap - 2 * corrupted_len``
        and ``nsteps = 1 + (T - nfft) // nstride``.
    """

    def __init__(
        self,
        fftlength: float,
        sample_rate: float,
        overlap: float = 0.0,
        window: str = "hann",
        detrend: str = "constant",
        corrupted_len: int = None,
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
        self.nfft = int(round(fftlength * sample_rate))
        self.noverlap = int(round(overlap * sample_rate))
        self.nstride = self.nfft - self.noverlap

        if self.nstride <= 0:
            raise ValueError("overlap must be smaller than fftlength.")

        self.corrupted_len = (
            self.nfft // 2 if corrupted_len is None else int(corrupted_len)
        )

        win = self._build_window(window, self.nfft)
        # Registered as a buffer for device/dtype/compile friendliness.
        self.register_buffer("window", win)

        # Welch one-sided PSD scale factor: 2 / (fs * sum(window**2)),
        # with DC/Nyquist bins *not* doubled (handled in _welch_asd).
        scale = 1.0 / (self.sample_rate * torch.sum(win**2))
        self.register_buffer("_psd_scale", scale)

    @staticmethod
    def _build_window(name: str, nfft: int) -> torch.Tensor:
        """Build a 1-D analysis window by name."""
        if name in (None, "boxcar", "rectangular"):
            return torch.ones(nfft)
        try:
            window_fn = getattr(torch, f"{name}_window")
        except AttributeError as exc:
            raise ValueError(f"Unsupported window {name!r}.") from exc
        return window_fn(nfft, periodic=False)

    def _frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slice ``x`` into overlapping frames.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, T)``

        Returns
        -------
        torch.Tensor, shape ``(B, D, nsteps, nfft)``
        """
        return x.unfold(-1, self.nfft, self.nstride)

    def _detrend(self, frames: torch.Tensor) -> torch.Tensor:
        """Remove the per-frame mean (only supported detrend mode)."""
        if self.detrend == "constant":
            return frames - frames.mean(dim=-1, keepdim=True)
        return frames

    def _invert_asd(self, asd: torch.Tensor) -> torch.Tensor:
        """
        Invert an ASD, excising (rather than amplifying) invalid bins.

        Bins with ``asd <= self.eps`` are treated as out of band and get
        ``invasd = 0``; all other bins get the ordinary ``1 / asd``. See
        the ``eps`` parameter docstring for why this is preferred over
        clamping the ASD to a small floor before dividing.

        Parameters
        ----------
        asd : torch.Tensor
            One-sided ASD, any shape.

        Returns
        -------
        torch.Tensor, same shape as ``asd``
        """
        threshold = self.eps if self.eps is not None else torch.finfo(asd.dtype).tiny
        valid = asd > threshold
        invasd = torch.zeros_like(asd)
        invasd[valid] = 1.0 / asd[valid]
        return invasd

    def _psd_from_Xf(self, Xf: torch.Tensor) -> torch.Tensor:
        """
        Welch-average an already-computed rFFT into a one-sided PSD.

        Split out from the old ``_welch_asd`` so ``forward`` can compute
        the windowed ``rfft`` exactly once and reuse it both for
        self-estimating the ASD and for the actual whitening multiply,
        instead of transforming the same frames twice.

        Parameters
        ----------
        Xf : torch.Tensor, shape ``(B, D, nsteps, F)``
            rFFT of detrended, windowed frames.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)``
        """
        periodogram = self._psd_scale * Xf.abs() ** 2
        # One-sided doubling of all bins except DC and (if nfft even) Nyquist.
        periodogram[..., 1:-1] = periodogram[..., 1:-1] * 2.0
        return periodogram.mean(dim=-2)  # average over segments -> (B, D, F)

    def _overlap_add(self, frames: torch.Tensor, out_len: int) -> torch.Tensor:
        """
        Reassemble whitened frames into a full-length signal via OLA.

        Uses ``torch.nn.functional.fold`` — the exact inverse of
        ``_frame``'s ``unfold`` — to sum overlapping frames back into
        place in a single fused op, rather than a Python loop over
        segments. This produces bit-identical results to a literal
        ``for i in range(nsteps): out[i0:i1] += frames[i]`` loop (no
        window-based normalization is applied, matching gwpy's own
        un-normalized ``+=`` accumulation), but as one vectorised,
        ``torch.compile``-friendly call instead of ``nsteps`` sequential
        Python-level slice-adds.

        Parameters
        ----------
        frames : torch.Tensor, shape ``(B, D, nsteps, nfft)``
            Whitened time-domain frames to be summed back into place.
        out_len : int
            Length of the reconstructed output, ``nsteps * nstride + noverlap``.

        Returns
        -------
        torch.Tensor, shape ``(B, D, out_len)``
        """
        B, D, nsteps, nfft = frames.shape
        # fold expects (N, C * kernel_size, L_patches); we have C=1, so
        # C * kernel_size = nfft. Treat the signal as a "1 x out_len" image
        # and each frame as a "1 x nfft" patch.
        patches = frames.reshape(B * D, nsteps, nfft).transpose(1, 2)  # (B*D, nfft, nsteps)
        folded = torch.nn.functional.fold(
            patches,
            output_size=(1, out_len),
            kernel_size=(1, nfft),
            stride=(1, self.nstride),
        )  # (B*D, 1, 1, out_len)
        return folded.reshape(B, D, out_len)

    def remove_corrupted(self, x: torch.Tensor) -> torch.Tensor:
        """
        Strip edge samples corrupted by the finite-length whitening filter.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, T)``
            Whitened time-domain strain (full OLA-reconstructed length).

        Returns
        -------
        torch.Tensor, shape ``(B, D, T - 2 * corrupted_len)``
            Valid central samples only.
        """
        if self.corrupted_len == 0:
            return x
        T = x.shape[-1]
        start = self.corrupted_len
        end = T - self.corrupted_len
        return x[..., start:end]

    @staticmethod
    def resample_asd(asd: torch.Tensor, n_freq: int) -> torch.Tensor:
        """
        Linearly resample an ASD onto a different uniform frequency grid
        spanning the same ``[0, Nyquist]`` range (i.e. same sample rate).

        Use this when your ASD was estimated with a different segment
        length (hence a different number of frequency bins) than this
        module's ``nfft``, before passing it to ``forward(x, asd=...)``.

        Parameters
        ----------
        asd : torch.Tensor, shape ``(..., F_in)``
            ASD sampled on a uniform grid from 0 Hz to Nyquist with
            ``F_in`` bins. Must share the same Nyquist frequency (i.e.
            the same underlying sample rate) as the target grid.
        n_freq : int
            Number of bins in the output grid, also spanning 0 to
            Nyquist (typically ``nfft // 2 + 1`` of the target module).

        Returns
        -------
        torch.Tensor, shape ``(..., n_freq)``
        """
        orig_shape = asd.shape
        flat = asd.reshape(-1, 1, orig_shape[-1])  # (N, 1, F_in)
        resampled = torch.nn.functional.interpolate(
            flat, size=n_freq, mode="linear", align_corners=True
        )
        return resampled.reshape(*orig_shape[:-1], n_freq)

    def forward(
        self,
        x: torch.Tensor,
        asd: torch.Tensor = None,
        detach_asd: bool = False,
    ) -> torch.Tensor:
        """
        Whiten ``x`` against either a self-estimated or a supplied ASD.

        This mirrors gwpy's ``asd=`` keyword on ``TimeSeries.whiten``:
        when ``asd`` is given, the Welch self-estimation step
        (``self._psd_from_Xf``) is skipped entirely and every frame is
        whitened against that fixed spectrum instead. Edge samples
        corrupted by the finite-length whitening filter are then
        stripped from both ends of the result (``self.remove_corrupted``).

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, T)``
            Time-domain strain.
        asd : torch.Tensor, shape ``(D, F)``, optional
            Externally supplied one-sided amplitude spectral density,
            with ``F = nfft // 2 + 1`` frequency bins (i.e. matching the
            rFFT of one ``nfft``-length frame at this module's
            ``sample_rate``/``fftlength``). When given, this overrides
            self-estimation and is broadcast across the batch and across
            every segment. When ``None`` (default), the ASD is estimated
            from ``x`` itself via Welch's method.
        detach_asd : bool, optional, default: False
            If ``True``, stop gradients from flowing back through the
            ASD used for whitening — whether self-estimated or supplied
            — treating it as a constant for this call, analogous to a
            fiducial whitener's severed graph.

        Returns
        -------
        torch.Tensor, shape ``(B, D, T_valid)``
            Whitened time series, reconstructed by overlap-add with
            corrupted edges removed.
        """
        raw_frames = self._frame(x)  # (B, D, nsteps, nfft)
        windowed = self._detrend(raw_frames) * self.window
        Xf = torch.fft.rfft(windowed, dim=-1)  # (B, D, nsteps, F) — computed ONCE

        if asd is not None:
            expected_bins = self.nfft // 2 + 1
            if asd.shape[-1] != expected_bins:
                raise ValueError(
                    f"asd has {asd.shape[-1]} frequency bins; expected "
                    f"{expected_bins} for nfft={self.nfft}. If your ASD was "
                    f"estimated at a different segment length, resample it "
                    f"first with SelfASDWhitening.resample_asd(asd, "
                    f"{expected_bins})."
                )
            asd = asd.to(device=x.device, dtype=raw_frames.dtype)
            invasd = self._invert_asd(asd)  # (D, F)
            invasd = invasd.unsqueeze(0).unsqueeze(-2)  # (1, D, 1, F) -> broadcast (B, D, nsteps, F)
        else:
            psd = self._psd_from_Xf(Xf)  # (B, D, F), reusing Xf — no second transform
            asd = psd.clamp_min(0.0).sqrt()
            invasd = self._invert_asd(asd).unsqueeze(-2)  # (B, D, 1, F) -> broadcast over nsteps

        if detach_asd:
            invasd = invasd.detach()

        Xw = Xf * invasd
        xw_frames = torch.fft.irfft(Xw, n=self.nfft, dim=-1)  # (B, D, nsteps, nfft)

        nsteps = raw_frames.shape[-2]
        out_len = nsteps * self.nstride + self.noverlap
        x_white = self._overlap_add(xw_frames, out_len)
        return self.remove_corrupted(x_white)