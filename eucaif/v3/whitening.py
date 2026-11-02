"""
Whitening methods for EuCAIF v3, behind one interface so that new ones plug in with a class and a name.

Every whitener is a ``Whitener``:

    w = whitener(x, asd)        # x: (T,), (D, T) or (B, D, T) strain; asd: (F,), (D, F) or (B, D, F), or None
                                # -> same leading shape, T - 2 * whitener.pad samples

- ``pad``: samples dropped at each end (the edges the method corrupts);
- ``needs_asd``: True if the method uses the ASD of each window (passed as ``asd``); False if it uses a fixed one
  given at construction (``asd`` is then ignored);
- output: whitened Gaussian noise has unit variance, in ``output_dtype`` (float32 by default); the filtering itself
  is float64 (real strain needs it, see ``sage.dsp.whiten.FIRWhitening``).

Methods (``whitening.method`` in the YAML):

- ``fir``: ``sage.dsp.whiten.FIRWhitening``, gwpy's ``TimeSeries.whiten`` (gwpy >= 4): an FIR inverse-spectrum-
  truncation filter of ``fduration`` s from each window's own ASD, high-pass at ``highpass``. Drops fduration / 2.
- ``fiducial``: SAGE ``main``'s ``FiducialWhitening`` algorithm: division by a fixed ASD per detector in the
  frequency domain (circular, no filter truncation), then ``pad_seconds`` dropped at each end. The ASD is read from
  ``main``'s fiducial files (``fiducial_dir`` with ``fiducial_{det}_psd.bin`` / ``.json``) or from an ``.npz``
  (``asd_file`` with ``detectors``, ``frequencies``, ``asd``; ``save_asd_file`` writes one). Unlike ``main``'s class
  it needs no SAGE run config, and it is normalised like ``fir`` (unit-variance whitened noise).

Adding a method: subclass ``Whitener``, set ``name`` and ``needs_asd``, implement ``from_config`` and ``whiten``,
and decorate with ``@register``.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from sage.dsp.whiten import FIRWhitening

REGISTRY = {}
DTYPES = {"float32": torch.float32, "float64": torch.float64}


def register(cls):
    """Class decorator: make a ``Whitener`` available under ``cls.name``."""
    if not cls.name or cls.name in REGISTRY:
        raise ValueError(f"whitener name {cls.name!r} missing or already registered")
    REGISTRY[cls.name] = cls
    return cls


def make_whitener(config, sample_rate, detectors):
    """The whitener of a ``whitening`` config block (``method`` and that method's options)."""
    config = dict(config)
    method = config.pop("method", "fir")
    if method not in REGISTRY:
        raise ValueError(f"unknown whitening method {method!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[method].from_config(config, float(sample_rate), list(detectors))


class Whitener(torch.nn.Module):
    """Base class: shape handling and output dtype; subclasses implement ``whiten`` on (B, D, T) float64."""

    name = None
    needs_asd = True

    def __init__(self, sample_rate, pad, output_dtype=torch.float32):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.pad = int(pad)
        self.output_dtype = output_dtype

    @classmethod
    def from_config(cls, config, sample_rate, detectors):
        raise NotImplementedError

    def whiten(self, x, asd):
        """(B, D, T) float64, asd (as given or None) -> (B, D, T - 2 * pad)."""
        raise NotImplementedError

    def forward(self, x, asd=None):
        x = torch.as_tensor(x)
        if not 1 <= x.dim() <= 3:
            raise ValueError(f"x must be (T,), (D, T) or (B, D, T), got shape {tuple(x.shape)}")
        if x.shape[-1] <= 2 * self.pad:
            raise ValueError(f"input of {x.shape[-1]} samples is not longer than 2 * pad = {2 * self.pad}")
        if self.needs_asd and asd is None:
            raise ValueError(f"the {self.name!r} whitener needs the window's ASD")
        lead = x.shape[:-1]
        x3 = x.to(torch.float64).reshape((1,) * (3 - x.dim()) + tuple(x.shape))
        if asd is not None:
            asd = torch.as_tensor(asd).to(device=x3.device, dtype=torch.float64)
        out = self.whiten(x3, asd)
        return out.reshape(*lead, out.shape[-1]).to(self.output_dtype)


@register
class FIRWhitener(Whitener):
    """``FIRWhitening`` (gwpy's whitening) with each window's ASD."""

    name = "fir"
    needs_asd = True

    def __init__(self, sample_rate, fduration=4.0, highpass=15.0, taper="hann", output_dtype=torch.float32):
        fir = FIRWhitening(sample_rate, fduration=fduration, highpass=highpass, taper=taper, dtype=torch.float64)
        super().__init__(sample_rate, fir.pad, output_dtype)
        self.fir = fir

    @classmethod
    def from_config(cls, config, sample_rate, detectors):
        return cls(sample_rate, fduration=float(config.get("fduration", 4.0)),
                   highpass=config.get("highpass", 15.0), taper=config.get("taper", "hann"),
                   output_dtype=DTYPES[config.get("output_dtype", "float32")])

    def whiten(self, x, asd):
        return self.fir(x, asd)


def load_fiducial_dir(path, detectors):
    """SAGE ``main``'s fiducial ASDs: ``fiducial_{det}_psd.bin`` (float32) + ``.json`` (bins, ``delta_f``).
    Returns (frequencies (F,), asd (D, F)), all detectors on the grid of the first."""
    path = Path(path)
    freqs, rows = None, []
    for det in detectors:
        meta = json.loads((path / f"fiducial_{det}_psd.json").read_text())
        n = int(meta.get("num_freq_bins") or meta.get("n_fd"))
        asd = np.fromfile(path / f"fiducial_{det}_psd.bin", dtype=np.float32).astype(np.float64)
        if asd.size != n:
            raise ValueError(f"{det}: {asd.size} ASD bins, the json says {n}")
        df = float(meta.get("delta_f") or float(meta["sample_rate"]) / (2.0 * (n - 1)))
        f = np.arange(n) * df
        if freqs is None:
            freqs = f
        rows.append(asd if f.size == freqs.size and np.allclose(f, freqs) else np.interp(freqs, f, asd))
    return freqs, np.stack(rows)


def save_asd_file(path, detectors, frequencies, asd):
    """Write an ASD file for the ``fiducial`` whitener: detectors, frequencies (F,) Hz, asd (D, F)."""
    np.savez(path, detectors=np.array(list(detectors)), frequencies=np.asarray(frequencies, dtype=np.float64),
             asd=np.asarray(asd, dtype=np.float64))


def load_asd_file(path, detectors):
    d = np.load(path)
    names = [str(s) for s in d["detectors"]]
    missing = [det for det in detectors if det not in names]
    if missing:
        raise ValueError(f"{path}: no ASD for {missing} (has {names})")
    return d["frequencies"], np.stack([d["asd"][names.index(det)] for det in detectors])


@register
class FiducialWhitener(Whitener):
    """Division by a fixed ASD per detector in the frequency domain (SAGE ``main``'s FiducialWhitening)."""

    name = "fiducial"
    needs_asd = False

    def __init__(self, sample_rate, frequencies, asd, pad_seconds=2.0, highpass=15.0, output_dtype=torch.float32):
        super().__init__(sample_rate, int(round(pad_seconds * sample_rate)), output_dtype)
        self.highpass = highpass
        self.src_freqs = np.asarray(frequencies, dtype=np.float64)
        self.src_asd = np.asarray(asd, dtype=np.float64)                    # (D, F_src)
        self._grids = {}

    @classmethod
    def from_config(cls, config, sample_rate, detectors):
        if config.get("fiducial_dir"):
            freqs, asd = load_fiducial_dir(config["fiducial_dir"], detectors)
        elif config.get("asd_file"):
            freqs, asd = load_asd_file(config["asd_file"], detectors)
        else:
            raise ValueError("the 'fiducial' whitener needs fiducial_dir (SAGE main's files) or asd_file (.npz)")
        return cls(sample_rate, freqs, asd, pad_seconds=float(config.get("pad_seconds", 2.0)),
                   highpass=config.get("highpass", 15.0), output_dtype=DTYPES[config.get("output_dtype", "float32")])

    def _inverse_asd(self, n, device):
        """1 / ASD on the rfft grid of n samples, 0 where the ASD is not positive or below the high-pass."""
        key = (n, str(device))
        if key not in self._grids:
            f = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
            asd = np.stack([np.interp(f, self.src_freqs, row, left=0.0, right=0.0) for row in self.src_asd])
            inv = np.where(asd > 0, 1.0 / np.where(asd > 0, asd, 1.0), 0.0)
            if self.highpass:
                inv[:, f < self.highpass] = 0.0
            self._grids[key] = torch.tensor(inv, device=device)
        return self._grids[key]

    def whiten(self, x, asd=None):
        n = x.shape[-1]
        inv = self._inverse_asd(n, x.device)
        if inv.shape[0] != x.shape[-2]:
            raise ValueError(f"{inv.shape[0]} fiducial ASDs for {x.shape[-2]} detectors")
        # sqrt(2 / fs): unit-variance output for Gaussian noise with that ASD (as FIRWhitening)
        w = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * inv, n=n, dim=-1) * math.sqrt(2.0 / self.sample_rate)
        return w[..., self.pad:n - self.pad]
