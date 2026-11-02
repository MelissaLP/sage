"""Tests for eucaif/v3/whitening.py: the Whitener interface, the 'fir' and 'fiducial' methods and the registry."""

import json
import math

import numpy as np
import pytest
import torch

from eucaif.v3 import whitening as W
from sage.dsp.whiten import FIRWhitening

FS = 2048.0


def _asd(f):
    f = np.maximum(f, 5.0)
    return 1e-23 * np.sqrt((30.0 / f) ** 8 + 1.0 + (f / 300.0) ** 2)


def _coloured_noise(n, d=2, seed=0):
    """Gaussian noise with one-sided ASD _asd (white unit-variance noise has PSD 2 / fs)."""
    white = torch.randn(d, n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    f = np.fft.rfftfreq(n, 1 / FS)
    colour = torch.tensor(_asd(f) * math.sqrt(FS / 2))
    return torch.fft.irfft(torch.fft.rfft(white, dim=-1) * colour, n=n, dim=-1), f


def test_registry():
    assert {"fir", "fiducial"} <= set(W.REGISTRY)
    with pytest.raises(ValueError):
        W.make_whitener(dict(method="median_filter"), FS, ["H1"])
    with pytest.raises(ValueError):
        W.make_whitener(dict(method="fiducial"), FS, ["H1"])                  # no ASD source given

    @W.register
    class Identity(W.Whitener):                                            # plugging in a new method
        name = "identity_for_test"
        needs_asd = False

        @classmethod
        def from_config(cls, config, sample_rate, detectors):
            return cls(sample_rate, pad=0)

        def whiten(self, x, asd):
            return x
    try:
        out = W.make_whitener(dict(method="identity_for_test"), FS, ["H1"])(torch.ones(3, 8))
        assert out.shape == (3, 8) and out.dtype == torch.float32
    finally:
        del W.REGISTRY["identity_for_test"]


def test_fir_is_firwhitening():
    x, f = _coloured_noise(16 * 2048)
    asd = torch.tensor(_asd(f))
    wh = W.make_whitener(dict(method="fir", fduration=4.0, highpass=15.0, output_dtype="float64"), FS, ["H1", "L1"])
    ref = FIRWhitening(FS, fduration=4.0, highpass=15.0, dtype=torch.float64)(x, asd)
    torch.testing.assert_close(wh(x, asd), ref)
    assert wh.pad == 4096 and wh.needs_asd
    with pytest.raises(ValueError):
        wh(x)                                                               # needs the window's ASD


def test_fiducial_and_fir_agree_on_gaussian_noise(tmp_path):
    n = 64 * 2048
    x, f = _coloured_noise(n)
    asd = _asd(f)
    W.save_asd_file(tmp_path / "asd.npz", ["H1", "L1"], f, np.stack([asd, asd]))
    fid = W.make_whitener(dict(method="fiducial", asd_file=str(tmp_path / "asd.npz"), pad_seconds=2.0,
                               highpass=15.0), FS, ["H1", "L1"])
    fir = W.make_whitener(dict(method="fir", fduration=4.0, highpass=15.0), FS, ["H1", "L1"])
    a, b = fid(x), fir(x, torch.tensor(asd))
    assert not fid.needs_asd and a.shape == b.shape == (2, n - 2 * 4096) and a.dtype == torch.float32
    for w in (a, b):
        assert float(w.std()) == pytest.approx(1.0, abs=0.03)              # unit variance (minus the < 15 Hz part)
    assert float(np.corrcoef(a.flatten(), b.flatten())[0, 1]) > 0.99      # same whitened noise


def test_fiducial_reads_sage_main_files(tmp_path):
    n_fd, df = 4097, 0.25                                                  # main's format: .bin + .json per detector
    f = np.arange(n_fd) * df
    for det in ("H1", "L1"):
        _asd(f).astype(np.float32).tofile(tmp_path / f"fiducial_{det}_psd.bin")
        (tmp_path / f"fiducial_{det}_psd.json").write_text(json.dumps(dict(n_fd=n_fd, delta_f=df, sample_rate=FS)))
    W.save_asd_file(tmp_path / "asd.npz", ["H1", "L1"], f, np.stack([_asd(f).astype(np.float32)] * 2))
    x, _ = _coloured_noise(16 * 2048)
    a = W.make_whitener(dict(method="fiducial", fiducial_dir=str(tmp_path)), FS, ["H1", "L1"])(x)
    b = W.make_whitener(dict(method="fiducial", asd_file=str(tmp_path / "asd.npz")), FS, ["H1", "L1"])(x)
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError):
        W.make_whitener(dict(method="fiducial", asd_file=str(tmp_path / "asd.npz")), FS, ["V1"])


@pytest.mark.parametrize("method", ["fir", "fiducial"])
def test_shapes(method, tmp_path):
    n = 16 * 2048
    x, f = _coloured_noise(n)
    W.save_asd_file(tmp_path / "asd.npz", ["H1", "L1"], f, np.stack([_asd(f)] * 2))
    wh = W.make_whitener(dict(method=method, asd_file=str(tmp_path / "asd.npz")), FS, ["H1", "L1"])
    asd = torch.tensor(_asd(f))
    out_bdt = wh(x[None], asd)
    assert out_bdt.shape == (1, 2, n - 2 * wh.pad)
    torch.testing.assert_close(wh(x, asd), out_bdt[0])                                   # (D, T)
    if method == "fir":
        torch.testing.assert_close(wh(x[0], asd), out_bdt[0, 0])                          # (T,)
    with pytest.raises(ValueError):
        wh(torch.zeros(1, 1, 2, n), asd)
