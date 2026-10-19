#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : qtransform.py
Description   : Multi-Q transform (Q-scan) in PyTorch, reproducing gwpy's ``TimeSeries.q_transform``.

GitHub Repository: NULL

Documentation:

    ``QTransform`` reproduces ``gwpy.timeseries.TimeSeries.q_transform(whiten=False)``
    (gwpy's port of the Omega-pipeline Q-scan, ``gwpy.signal.qtransform``) on batches:

        qt = QTransform(sample_rate=2048, seq_len=x.shape[-1], qrange=(4, 64), frange=(10, 1024))
        spec = qt(x)                       # (B, D, n_times, n_freqs), like gwpy's Spectrogram
        spec = qt(x, gps=1.0)              # loudest Q-plane searched within gps +/- search / 2

    Whitening is not part of the module: whiten first, e.g. with ``sage.dsp.whiten.FIRWhitening``.
    ``QTransform`` applied to ``FIRWhitening(x, asd)`` equals gwpy's
    ``TimeSeries(FIRWhitening(x, asd)).q_transform(whiten=False)``. (gwpy's ``q_transform(whiten=asd)``
    whitens the full series and keeps the corrupted ``fduration / 2`` at each edge, which then enter
    the median normalisation; ``FIRWhitening`` crops them.)

    Pipeline, mirroring gwpy step by step:

    1. Tiling (``QTiling`` / ``QPlane`` / ``QTile``), computed once at construction: Q-planes
       log-spaced in ``qrange`` with the given ``mismatch``, and for each plane frequency rows
       log-spaced in ``frange``, each row with ``ntiles`` (a power of 2) time tiles and a bi-square
       frequency-domain window.
    2. ``X = rfft(x) / T``, doubled above DC (gwpy's ``TimeSeries.fft``).
    3. Per row: window the bins around its frequency, zero-pad to ``ntiles``, ``ifftshift``,
       ``ifft``, and take ``|.|^2``: the tile energies. Normalised by their median (numpy's median,
       i.e. the mean of the two middle values) or mean (``norm``).
    4. For each series, pick the Q-plane with the loudest tile (inside the search window if ``gps``
       is given); ties go to the first plane, as in gwpy.
    5. Interpolate the chosen plane onto a regular grid, as ``QGram.interpolate``: each row in time
       with a cubic interpolating spline (``InterpolatedUnivariateSpline``, extrapolated past its last
       tile), then in frequency with ``RectBivariateSpline`` (clamped outside the plane's frequencies).

    Both interpolations are linear in the energies and depend only on the tiling, so they are built
    once at construction with scipy's own splines and applied as matrix products: a sparse matrix
    per row length in time (the spline's weights decay by 2 - sqrt(3) per sample, so only the
    weights above ~1e-16 of the largest are kept), a dense one per Q-plane in frequency.

    Differences from gwpy:

    - gwpy casts the normalised energies and the output to float32. Here everything is in the
      module's ``dtype``: float32 by default (~1e-6 from gwpy), float64 for validation (gwpy's own
      float32 rounding, ~1e-7, is then the difference).
    - ``fres=None`` (no frequency interpolation) is not supported: the number of rows differs
      between Q-planes, so a batch could not be stacked.
    - Times are in seconds from the start of the series (gwpy uses the series' GPS epoch).

    Gradients: off by default, like SAGE's other processing modules. With ``differentiable=True``
    the output is differentiable with respect to the input (e.g. for adversarial perturbations or
    a loss on the Q-scan). The choice of Q-plane is discrete, so it carries no gradient and is made
    without autograd; the gradient flows through the chosen plane's tiles, normalisation and
    interpolation.
"""

import contextlib
import math
import warnings

import numpy as np
import torch
from torch import Tensor

DEFAULT_FRANGE = (0, float("inf"))
DEFAULT_MISMATCH = 0.2
DEFAULT_QRANGE = (4, 64)


# -- tiling: plain-python port of gwpy.signal.qtransform, evaluated once ------------------------

def _round_to_power_upper(x: float, base: int = 2) -> int:
    """gwpy.utils.round_to_power(x, base, which='upper')."""
    return int(base ** math.ceil(math.log(x, base)))


def _deltam(mismatch: float) -> float:
    return 2 * (mismatch / 3.0) ** (1 / 2.0)


def _iter_qs(qrange, mismatch):
    cumum = math.log(qrange[1] / qrange[0]) / 2 ** (1 / 2.0)
    nplanes = int(max(math.ceil(cumum / _deltam(mismatch)), 1))
    dq = cumum / nplanes
    for i in range(nplanes):
        yield qrange[0] * math.exp(2 ** (1 / 2.0) * dq * (i + 0.5))


def _tiling_frange(qs, frange, duration, sampling):
    """QTiling's frequency range: non-zero lower bound, upper bound capped by the lowest Q."""
    frange = [float(frange[0]), float(frange[1])]
    if frange[0] == 0:
        frange[0] = 50 * max(qs) / (2 * math.pi * duration)
    maxf = sampling / 2 / (1 + 11 ** (1 / 2.0) / min(qs))
    if math.isinf(frange[1]):
        frange[1] = maxf
    elif frange[1] > maxf:
        warnings.warn("upper frequency of %.2f is too high for the given Q range, resetting to %.2f"
                      % (frange[1], maxf))
        frange[1] = maxf
    return frange


def _plane_frequencies(q, frange, duration, mismatch):
    """QPlane._iter_frequencies (unique row frequencies, floored to multiples of 1 / duration)."""
    minf, maxf = frange
    fcum_mismatch = math.log(maxf / minf) * (2 + q ** 2) ** (1 / 2.0) / 2.0
    nfreq = int(max(1, math.ceil(fcum_mismatch / _deltam(mismatch))))
    fstep = fcum_mismatch / nfreq
    fstepmin = 1 / duration
    out, last = [], None
    for i in range(nfreq):
        this = minf * math.exp(2 / (2 + q ** 2) ** (1 / 2.0) * (i + 0.5) * fstep) // fstepmin * fstepmin
        if this != last:
            out.append(this)
        last = this
    return out


def _tile_row(q, frequency, duration, sampling, mismatch):
    """QTile: (ntiles, rfft bin indices, bi-square window, padding) for one row."""
    qprime = q / 11 ** (1 / 2.0)
    tcum_mismatch = duration * 2 * math.pi * frequency / q
    ntiles = _round_to_power_upper(tcum_mismatch / _deltam(mismatch))
    windowsize = 2 * int(frequency / qprime * duration) + 1
    half = int((windowsize - 1) / 2)
    indices = np.arange(-half, half + 1)
    xfrequencies = indices / duration * qprime / frequency
    norm = ntiles / (duration * sampling) * (315 * qprime / (128 * frequency)) ** (1 / 2.0)
    window = (1 - xfrequencies ** 2) ** 2 * norm
    data_indices = np.round(indices + 1 + frequency * duration).astype(int)
    pad = ntiles - windowsize
    padding = (int((pad - 1) / 2.0), int((pad + 1) / 2.0))
    return ntiles, data_indices, window, padding


# -- interpolation operators, built with scipy's own splines --------------------------------------

def _time_operator(n: int, duration: float, xout: np.ndarray, rtol: float = 1e-16):
    """Sparse (len(xout), n) matrix of ``InterpolatedUnivariateSpline(xrow, .)(xout)``.

    ``xrow`` are the tile times of a row with ``n`` tiles, as gwpy builds them. FITPACK's
    interpolating cubic spline (s=0) has the not-a-knot knots ``x[2:-2]``, so it is the B-spline
    with those knots that interpolates the data, extrapolated with its end polynomials (ext=0).
    """
    import scipy.sparse.linalg
    from scipy.interpolate import BSpline

    dx = duration / n
    xrow = np.arange(0.0, n * dx, dx)
    if xrow.size != n or n < 4:
        raise ValueError(f"Cannot build the time interpolation for a row of {n} tiles")
    t = np.r_[[xrow[0]] * 4, xrow[2:-2], [xrow[-1]] * 4]
    colloc = BSpline.design_matrix(xrow, t, 3)                    # (n, n): data = colloc @ coef
    evaluate = BSpline.design_matrix(xout, t, 3, extrapolate=True)  # (nx, n): out = evaluate @ coef
    # out = evaluate @ colloc^-1 @ data; solve the transposed system for the (n, nx) operator
    op_t = scipy.sparse.linalg.splu(colloc.T.tocsc()).solve(evaluate.T.toarray())
    keep = np.abs(op_t) > rtol * np.abs(op_t).max(axis=0, keepdims=True)
    rows, cols = np.nonzero(keep.T)                              # (output time, tile) pairs
    return np.stack([rows, cols]), op_t.T[rows, cols], (xout.size, n)


def _frequency_operator(freqs: np.ndarray, xout: np.ndarray, outfreq: np.ndarray) -> np.ndarray:
    """Dense (len(outfreq), len(freqs)) matrix of gwpy's ``RectBivariateSpline`` step.

    gwpy fits ``RectBivariateSpline(xout, freqs, out)`` and evaluates it at ``(xout, outfreq)``. At
    the time nodes the time interpolation is exact, so this is a 1-D spline in frequency per time
    sample; evaluating the same routine on the identity gives its matrix (clamping included).
    """
    from scipy.interpolate import RectBivariateSpline

    nf = freqs.size
    if nf < 4:
        raise ValueError(f"A Q-plane with {nf} frequency rows cannot be interpolated (need >= 4)")
    eye = np.eye(nf)
    nodes = np.arange(nf, dtype=float)
    return RectBivariateSpline(nodes, freqs, eye)(nodes, np.asarray(outfreq, dtype=float)).T


def _median(x: Tensor, dim: int = -1) -> Tensor:
    """numpy's median: the mean of the two middle values for an even count (torch.median
    returns the lower one)."""
    n = x.shape[dim]
    s = torch.sort(x, dim=dim).values
    if n % 2:
        return s.narrow(dim, n // 2, 1)
    return 0.5 * (s.narrow(dim, n // 2 - 1, 1) + s.narrow(dim, n // 2, 1))


class QTransform(torch.nn.Module):
    """
    Multi-Q transform of a batch of time series, reproducing gwpy's ``TimeSeries.q_transform``
    with ``whiten=False`` (whiten beforehand, e.g. with ``FIRWhitening``).

    Parameters
    ----------
    sample_rate : float
        Sample rate of the input, in Hz.
    seq_len : int
        Length ``T`` of the inputs, in samples. The tiling depends on the duration, so the module
        is built for one input length.
    qrange : (float, float)
        Range of Q to scan (gwpy default (4, 64)).
    frange : (float, float)
        Frequency range to scan, in Hz. 0 and inf select gwpy's automatic limits.
    mismatch : float
        Maximum fractional mismatch between neighbouring tiles (gwpy default 0.2).
    tres : float, optional
        Time resolution of the output, in s. Default: ``(outseg[1] - outseg[0]) / 1000``.
    fres : float or int
        Frequency resolution in Hz (default 0.5), or the number of log-spaced frequencies if
        ``logf``.
    logf : bool
        Log-spaced output frequencies (gwpy default False).
    outseg : (float, float), optional
        Output time span ``[start, stop)`` in s from the start of the series. Default: all of it.
    norm : {"median", "mean", True, False}
        Normalise each row's energies by their median (``True``/``"median"``, gwpy default) or
        mean; ``False`` for the raw energies.
    search : float
        Width (s) of the window around ``gps`` in which the loudest Q-plane is searched
        (gwpy default 0.5). Only used when ``gps`` is given to ``forward``.
    dtype : {torch.float32, torch.float64}
        Precision of the computation and of the output.
    differentiable : bool
        If True, gradients flow from the output back to the input (when autograd is enabled);
        if False (default), ``forward`` runs under ``torch.no_grad()``. Can be changed later via
        the ``differentiable`` attribute.

    Attributes
    ----------
    qs : Tensor (P,)
        Q of each plane (float64, on the CPU, like ``times`` and ``frequencies``).
    times : Tensor (n_times,)
        Output times, in s from the start of the series.
    frequencies : Tensor (n_freqs,)
        Output frequencies, in Hz.
    frange : (float, float)
        Frequency range actually scanned.

    Input / Output
    --------------
    forward(x, gps=None, return_q=False) : (..., T) -> (..., n_times, n_freqs)
        Same orientation as gwpy's ``Spectrogram`` (time first); ``.transpose(-1, -2)`` for
        frequency-time images. With ``return_q``, also the Q of the plane chosen for each series.
    """

    def __init__(
        self,
        sample_rate: float,
        seq_len: int,
        qrange=DEFAULT_QRANGE,
        frange=DEFAULT_FRANGE,
        mismatch: float = DEFAULT_MISMATCH,
        tres: float = None,
        fres=0.5,
        logf: bool = False,
        outseg=None,
        norm="median",
        search: float = 0.5,
        dtype: torch.dtype = torch.float32,
        differentiable: bool = False,
    ):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")
        if fres is None:
            raise NotImplementedError("fres=None is not supported (rows differ between Q-planes).")
        norm = norm.lower() if isinstance(norm, str) else norm
        if norm not in (True, False, "median", "mean"):
            raise ValueError(f"Invalid normalisation {norm!r}")
        self.norm = "median" if norm is True else norm
        self.sample_rate = float(sample_rate)
        self.seq_len = int(seq_len)
        self.search = search
        self.differentiable = differentiable
        # gwpy: duration = abs(span) = N * dt
        self.duration = duration = self.seq_len * (1.0 / self.sample_rate)

        qrange = (float(qrange[0]), float(qrange[1]))
        qs = list(_iter_qs(qrange, mismatch))
        self.frange = tuple(_tiling_frange(qs, frange, duration, self.sample_rate))

        # output grid, exactly as QGram.interpolate builds it
        outseg = (0.0, duration) if outseg is None else (float(outseg[0]), float(outseg[1]))
        if tres is None:
            tres = abs(outseg[1] - outseg[0]) / 1000.0
        xout = np.arange(*outseg, step=tres)
        energy_np_dtype = np.float32 if self.norm else np.float64
        if logf:
            outfreq = np.geomspace(self.frange[0], self.frange[1], num=int(fres))
        else:
            outfreq = np.arange(self.frange[0], self.frange[1], fres, dtype=energy_np_dtype)

        # float64 metadata stays on the CPU (not buffers): MPS has no float64
        self.qs = torch.tensor(qs, dtype=torch.float64)
        self.times = torch.tensor(xout, dtype=torch.float64)
        self.frequencies = torch.tensor(np.asarray(outfreq, dtype=np.float64))
        self.register_buffer("_dtype_ref", torch.zeros(0, dtype=dtype))

        # per plane: rows grouped by ntiles, each group with its gather indices and windows
        # (already placed at their ifftshifted positions), plus the frequency operator
        self._planes = []
        time_ops = {}
        for p, q in enumerate(qs):
            freqs = _plane_frequencies(q, self.frange, duration, mismatch)
            groups = {}
            for r, f in enumerate(freqs):
                ntiles, idx, win, (pad_left, _) = _tile_row(q, f, duration, self.sample_rate, mismatch)
                pos = (pad_left + np.arange(idx.size) - ntiles // 2) % ntiles  # after ifftshift
                gidx = np.zeros(ntiles, dtype=np.int64)
                gwin = np.zeros(ntiles)
                gidx[pos], gwin[pos] = idx, win
                groups.setdefault(ntiles, ([], [], []))
                groups[ntiles][0].append(r)
                groups[ntiles][1].append(gidx)
                groups[ntiles][2].append(gwin)
            plane_groups = []
            for g, (ntiles, (rows, gidx, gwin)) in enumerate(sorted(groups.items())):
                name = f"_p{p}_g{g}"
                self.register_buffer(name + "_rows", torch.tensor(rows))
                self.register_buffer(name + "_idx", torch.tensor(np.stack(gidx)))
                self.register_buffer(name + "_win", torch.tensor(np.stack(gwin), dtype=dtype))
                plane_groups.append((name, ntiles))
                if ntiles not in time_ops:
                    time_ops[ntiles] = _time_operator(ntiles, duration, xout)
            fop = _frequency_operator(np.asarray(freqs), xout, outfreq)
            self.register_buffer(f"_p{p}_fop", torch.tensor(fop, dtype=dtype))
            self._planes.append((plane_groups, len(freqs)))

        for ntiles, (indices, values, shape) in time_ops.items():
            self.register_buffer(f"_t{ntiles}_indices", torch.tensor(indices))
            self.register_buffer(f"_t{ntiles}_values", torch.tensor(values, dtype=dtype))
        self._time_shapes = {n: shape for n, (_, _, shape) in time_ops.items()}

    @property
    def dtype(self) -> torch.dtype:
        """Working precision; follows ``.to(dtype)`` / ``.double()`` / ``.float()``."""
        return self._dtype_ref.dtype

    # -- building blocks -----------------------------------------------------------------------

    def _fft(self, x: Tensor) -> Tensor:
        """gwpy's TimeSeries.fft: rfft / T, doubled above DC."""
        X = torch.fft.rfft(x, dim=-1) / x.shape[-1]
        return torch.cat([X[..., :1], 2.0 * X[..., 1:]], dim=-1)

    def _group_energy(self, X: Tensor, name: str) -> Tensor:
        """Normalised tile energies of one group of rows: (M, F) -> (M, R, ntiles)."""
        idx = getattr(self, name + "_idx")
        win = getattr(self, name + "_win")
        tiles = torch.fft.ifft(X[:, idx] * win, dim=-1)        # (M, R, ntiles)
        energy = tiles.real ** 2 + tiles.imag ** 2
        if self.norm == "median":
            energy = energy / _median(energy)
        elif self.norm == "mean":
            energy = energy / energy.mean(dim=-1, keepdim=True)
        return energy

    def _search_mask(self, ntiles: int, gps, device) -> Tensor:
        """Tiles inside gwpy's search segment, (M or 1, ntiles) bool; None if no search."""
        if gps is None:
            return None
        # tile indices in float64 on the CPU (exact floor at segment edges, MPS has no float64)
        dx = self.duration / ntiles
        k = torch.arange(ntiles)
        gps = gps.reshape(-1, 1)
        # Segment(gps - search / 2, gps + search / 2) & span, then Series.crop's floor indices
        start = torch.clamp(gps - self.search / 2, min=0.0)
        end = torch.clamp(gps + self.search / 2, max=self.duration)
        i0 = torch.floor(start / dx)
        i1 = torch.where(end >= self.duration, torch.full_like(end, ntiles), torch.floor(end / dx))
        return ((k >= i0) & (k < i1)).to(device)

    def _plane_peak(self, X: Tensor, p: int, gps) -> Tensor:
        groups, _ = self._planes[p]
        peak = None
        for name, ntiles in groups:
            energy = self._group_energy(X, name)
            mask = self._search_mask(ntiles, gps, X.device)
            if mask is not None:
                energy = torch.where(mask.unsqueeze(1), energy, torch.zeros((), dtype=energy.dtype,
                                                                            device=energy.device))
            m = energy.amax(dim=(-2, -1))
            peak = m if peak is None else torch.maximum(peak, m)
        return peak

    def _plane_spectrogram(self, X: Tensor, p: int) -> Tensor:
        """Interpolated spectrogram of plane ``p``: (M, F) -> (M, n_times, n_freqs)."""
        groups, nrows = self._planes[p]
        M = X.shape[0]
        nx = self.times.numel()
        rows_t = torch.zeros(M, nrows, nx, dtype=self.dtype, device=X.device)
        for name, ntiles in groups:
            energy = self._group_energy(X, name)                  # (M, R, ntiles)
            R = energy.shape[1]
            op = torch.sparse_coo_tensor(getattr(self, f"_t{ntiles}_indices"),
                                         getattr(self, f"_t{ntiles}_values"),
                                         self._time_shapes[ntiles])
            interp = torch.sparse.mm(op, energy.reshape(M * R, ntiles).T)   # (nx, M * R)
            rows_t[:, getattr(self, name + "_rows")] = interp.T.reshape(M, R, nx)
        fop = getattr(self, f"_p{p}_fop")                          # (n_freqs, nrows)
        return torch.einsum("fr,mrt->mtf", fop, rows_t)

    # -- forward -------------------------------------------------------------------------------

    def forward(self, x: Tensor, gps=None, return_q: bool = False):
        """
        Q-transform ``x``.

        Parameters
        ----------
        x : Tensor, shape (..., T)
            Time series (normally whitened), ``T == seq_len``.
        gps : float or Tensor broadcastable to ``x.shape[:-1]``, optional
            Time (s from the start of the series) around which the loudest Q-plane is searched,
            within ``+/- search / 2``. Default: the whole series.
        return_q : bool
            Also return the Q of the plane chosen for each series, shape ``x.shape[:-1]``.

        Returns
        -------
        Tensor, shape (..., n_times, n_freqs)
        """
        grad = contextlib.nullcontext() if self.differentiable else torch.no_grad()
        with grad:
            return self._forward(x, gps, return_q)

    def _forward(self, x: Tensor, gps, return_q: bool):
        if x.shape[-1] != self.seq_len:
            raise ValueError(f"Input length {x.shape[-1]} doesn't match seq_len={self.seq_len}")
        lead = x.shape[:-1]
        x = x.to(dtype=self.dtype).reshape(-1, self.seq_len)
        if not torch.isfinite(x.detach()).all():
            raise ValueError("Input signal contains non-numerical values")
        X = self._fft(x)
        if gps is not None:
            gps = torch.as_tensor(gps, dtype=torch.float64, device="cpu").expand(lead).reshape(-1)

        # 1st pass: loudest tile of every plane (energies are recomputed for the chosen plane
        # only, rather than keeping every plane's tiles in memory)
        # (a discrete choice: no gradient, so no autograd graph either)
        with torch.no_grad():
            Xd = X.detach()
            peaks = torch.stack([self._plane_peak(Xd, p, gps) for p in range(len(self._planes))])
            choice = torch.argmax(peaks, dim=0)                      # first maximum, as gwpy

        # 2nd pass: the chosen plane of each series (index_copy keeps the autograd graph)
        out = torch.zeros(x.shape[0], self.times.numel(), self.frequencies.numel(),
                          dtype=self.dtype, device=x.device)
        for p in range(len(self._planes)):
            sel = torch.nonzero(choice == p).squeeze(-1)
            if sel.numel():
                out = out.index_copy(0, sel, self._plane_spectrogram(X[sel], p))
        out = out.reshape(*lead, *out.shape[-2:])
        if return_q:
            return out, self.qs[choice.cpu()].reshape(lead)
        return out
