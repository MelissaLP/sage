"""Tests for sage.dsp.qtransform: the torch port of gwpy's TimeSeries.q_transform.

Compares QTransform with gwpy's ``q_transform(whiten=False)`` on noise with sine-Gaussian
glitches, for several settings, and checks the chain FIRWhitening -> QTransform against gwpy's
whiten -> q_transform. gwpy rounds normalised energies to float32, so with ``norm`` the
float64 module differs from gwpy by ~1e-7; without normalisation the two agree to ~1e-15.
"""

import warnings

import numpy as np
import pytest
import torch

from sage.dsp.qtransform import QTransform, _median

gwpy_ts = pytest.importorskip("gwpy.timeseries")

FS = 2048.0
T = 4096


def _data(seed=0):
    """(2, 2, T) white noise; a loud short glitch in [0, 0] and a longer one in [1, 1]."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FS
    x = rng.normal(size=(2, 2, T))
    x[0, 0] += 6 * np.sin(2 * np.pi * 150 * (t - 1.2)) * np.exp(-((t - 1.2) / 0.01) ** 2)
    x[1, 1] += 4 * np.sin(2 * np.pi * 400 * (t - 0.7)) * np.exp(-((t - 0.7) / 0.05) ** 2)
    return x


def _gwpy(x, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gwpy_ts.TimeSeries(x, sample_rate=FS).q_transform(whiten=False, **kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(),
    dict(qrange=(4, 64), frange=(10, 1000)),
    dict(frange=(20, 500), tres=0.002, fres=1.0),
    dict(logf=True, fres=300, outseg=(0.5, 1.5)),
    dict(norm="mean"),
    dict(norm=False),
    dict(mismatch=0.1, qrange=(8, 32)),
], ids=["default", "frange", "tres-fres", "logf-outseg", "mean", "nonorm", "mismatch"])
def test_matches_gwpy_q_transform(kwargs):
    x = _data()
    qt = QTransform(FS, T, dtype=torch.float64, **kwargs)
    out, q = qt(torch.tensor(x), return_q=True)
    rtol = 1e-12 if kwargs.get("norm", True) is False else 1e-6   # gwpy's float32 rounding
    for i in range(2):
        for d in range(2):
            ref = _gwpy(x[i, d], **kwargs)
            assert out.shape[-2:] == ref.shape
            assert float(q[i, d]) == pytest.approx(ref.q)
            np.testing.assert_allclose(out[i, d].numpy(), ref.value, atol=rtol * np.abs(ref.value).max())
    np.testing.assert_allclose(qt.times.numpy(), ref.times.value)
    np.testing.assert_allclose(qt.frequencies.numpy(), ref.frequencies.value)


def test_search_window_and_per_series_gps():
    """gps selects the loudest plane inside gps +/- search / 2, per series if gps is a tensor."""
    x = _data()
    qt = QTransform(FS, T, search=0.3, dtype=torch.float64)
    gps = torch.tensor([[1.2, 0.7], [1.2, 0.7]], dtype=torch.float64)
    out, q = qt(torch.tensor(x), gps=gps, return_q=True)
    for i in range(2):
        for d in range(2):
            ref = _gwpy(x[i, d], gps=float(gps[i, d]), search=0.3)
            assert float(q[i, d]) == pytest.approx(ref.q)
            np.testing.assert_allclose(out[i, d].numpy(), ref.value, atol=1e-6 * np.abs(ref.value).max())


def test_float32_and_leading_dims():
    x = torch.tensor(_data())
    out64 = QTransform(FS, T, dtype=torch.float64)(x)
    qt32 = QTransform(FS, T)
    out32 = qt32(x.float())
    assert out32.dtype == torch.float32
    np.testing.assert_allclose(out32.numpy(), out64.numpy(), atol=1e-4 * out64.abs().max().item())
    # any leading shape; a single series gives (n_times, n_freqs)
    assert qt32(x[0, 0].float()).shape == out32.shape[-2:]
    assert qt32(x.reshape(4, T).float()).shape == (4, *out32.shape[-2:])
    # .double() switches the module's precision
    assert qt32.double()(x).dtype == torch.float64


def test_after_fir_whitening_matches_gwpy_whiten():
    """FIRWhitening -> QTransform == gwpy's whiten (edges cropped) -> q_transform."""
    from gwpy.frequencyseries import FrequencySeries
    from sage.dsp.whiten import FIRWhitening

    fs, n = 2048.0, 8 * 2048
    f = np.linspace(0, fs / 2, n // 2 + 1)
    asd = 1e-23 * np.sqrt((20.0 / np.maximum(f, 5.0)) ** 8 + 1.0 + (np.maximum(f, 5.0) / 300.0) ** 2)
    rng = np.random.default_rng(3)
    colour = np.fft.irfft(np.fft.rfft(rng.normal(size=n)) * asd, n=n) * np.sqrt(fs / 2)
    wh = FIRWhitening(fs, fduration=2.0, dtype=torch.float64)
    white = wh(torch.tensor(colour).reshape(1, 1, n), asd=torch.tensor(asd).reshape(1, -1))[0, 0]

    ref_white = gwpy_ts.TimeSeries(colour, sample_rate=fs).whiten(
        asd=FrequencySeries(asd, df=f[1]), fduration=2.0).value[wh.pad:n - wh.pad]
    qt = QTransform(fs, white.numel(), frange=(10, 900), dtype=torch.float64)
    ref = _gwpy(ref_white, frange=(10, 900))
    np.testing.assert_allclose(qt(white).numpy(), ref.value, atol=1e-6 * np.abs(ref.value).max())


def test_median_is_numpys():
    x = torch.randn(3, 5, 8, dtype=torch.float64)
    np.testing.assert_allclose(_median(x).squeeze(-1).numpy(), np.median(x.numpy(), axis=-1))
    y = torch.randn(3, 7, dtype=torch.float64)
    np.testing.assert_allclose(_median(y).squeeze(-1).numpy(), np.median(y.numpy(), axis=-1))


def test_errors():
    qt = QTransform(FS, T)
    with pytest.raises(ValueError):
        qt(torch.zeros(2, T - 1))
    bad = torch.zeros(T)
    bad[10] = float("nan")
    with pytest.raises(ValueError):
        qt(bad)
    with pytest.raises(NotImplementedError):
        QTransform(FS, T, fres=None)
    with pytest.raises(ValueError):
        QTransform(FS, T, norm="max")


def test_gradients_off_by_default():
    x = torch.tensor(_data()[0, 0], requires_grad=True)
    out = QTransform(FS, T, dtype=torch.float64)(x)
    assert not out.requires_grad


@pytest.mark.parametrize("gps", [None, 1.0])
def test_gradients_match_finite_differences(gps):
    """differentiable=True: autograd gradient == numerical gradient (small setup, float64)."""
    fs, n = 256.0, 512
    rng = np.random.default_rng(5)
    t = np.arange(n) / fs
    x0 = rng.normal(size=(2, n))
    x0[0] += 5 * np.sin(2 * np.pi * 40 * (t - 1.0)) * np.exp(-((t - 1.0) / 0.03) ** 2)
    qt = QTransform(fs, n, qrange=(4, 16), frange=(8, 100), tres=0.05, fres=2.0,
                    dtype=torch.float64, differentiable=True)
    x = torch.tensor(x0, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: qt(v, gps=gps), (x,), fast_mode=True)
    # same values as without gradients, and a usable gradient for e.g. adversarial updates
    out = qt(x, gps=gps)
    qt.differentiable = False
    np.testing.assert_array_equal(out.detach().numpy(), qt(x, gps=gps).numpy())
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max() > 0
