"""Tests for sage.data.noise.glitch_noise: the torch port of gengli's add_glitch.

Checks the ports against scipy / gengli's own formulas, that the injected glitch has the
requested SNR against the ASD it was coloured with, and that SAGE's FIRWhitening recovers the
glitch shape at the right time. If gengli and PyCBC are installed, also checks that the
injection reproduces gengli's ``add_glitch`` exactly when both use the same PSD.
"""

import numpy as np
import pytest
import scipy.signal
import torch

from sage.data.noise.glitch_noise import (
    GlitchInjector,
    add_glitch,
    glitch_snr,
    lowpass_glitch,
    lowpass_response,
    resample_glitch,
    rescale_glitch,
    tukey_window,
)

FS = 2048.0


def _asd(fs=FS, n_freq=4097):
    """A smooth, LIGO-like ASD on a uniform [0, Nyquist] grid (strain / sqrt(Hz))."""
    f = np.linspace(0, fs / 2, n_freq)
    f = np.maximum(f, 5.0)
    return torch.tensor(1e-23 * np.sqrt((20.0 / f) ** 8 + 1.0 + (f / 300.0) ** 2), dtype=torch.float64)


def _sine_gaussian(n=512, fs=FS, f0=150.0, tau=0.02):
    t = (np.arange(n) - n // 2) / fs
    return torch.tensor(np.sin(2 * np.pi * f0 * t) * np.exp(-(t / tau) ** 2), dtype=torch.float64)


def _snr_against_asd(h, asd, fs, f_low=None):
    """Independent optimal SNR of coloured h: rho^2 = 4 df sum |dt rfft(h)|^2 / S(f) (as PyCBC sigmasq)."""
    h = np.asarray(h, dtype=np.float64)
    n = h.shape[-1]
    f = np.fft.rfftfreq(n, 1 / fs)
    a = np.interp(f, np.linspace(0, fs / 2, asd.shape[-1]), np.asarray(asd))
    H = np.fft.rfft(h, axis=-1) / fs
    m = np.ones_like(f) if f_low is None else (f >= f_low)
    return np.sqrt(4 * (fs / n) * np.sum(np.abs(H) ** 2 / a ** 2 * m, axis=-1))


@pytest.mark.parametrize("n", [1, 2, 7, 100, 469, 1024])
@pytest.mark.parametrize("alpha", [0.0, 0.07, 0.1, 0.25, 0.5, 1.0])
def test_tukey_matches_scipy(n, alpha):
    ours = tukey_window(n, alpha).numpy()
    ref = scipy.signal.windows.tukey(n, alpha=alpha, sym=True)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("n_old,fs_old,fs_new", [(938, 4096, 2048), (937, 4096, 2048),
                                                 (469, 2048, 4096), (1000, 4096, 1024), (512, 4096, 4096)])
def test_resample_matches_scipy(n_old, fs_old, fs_new):
    x = np.random.default_rng(0).normal(size=(3, n_old))
    ours = resample_glitch(torch.tensor(x), fs_old, fs_new).numpy()
    ref = scipy.signal.resample(x, int(n_old * fs_new / fs_old), axis=-1)
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_snr_and_rescale_match_gengli_formula():
    g = torch.tensor(np.random.default_rng(1).normal(size=(4, 469)))
    # gengli.utils.rescale_glitch, numpy
    G = np.fft.rfft(g.numpy(), axis=-1) / FS
    ref = np.sqrt(4.0 * (FS / g.shape[-1]) * np.sum(np.abs(G) ** 2, axis=-1))
    np.testing.assert_allclose(glitch_snr(g, FS).numpy(), ref, rtol=1e-12)
    target = torch.tensor([5.0, 10.0, 0.0, 50.0], dtype=torch.float64)
    np.testing.assert_allclose(glitch_snr(rescale_glitch(g, target, FS), FS).numpy(), target.numpy(), atol=1e-9)


@pytest.mark.parametrize("order", [1, 3, 6])
@pytest.mark.parametrize("f_high", [100.0, 250.0, 900.0])
def test_lowpass_response_matches_scipy_butter_filtfilt(order, f_high):
    n = 4096
    sos = scipy.signal.butter(order, f_high, btype="low", fs=FS, output="sos")
    _, h = scipy.signal.sosfreqz(sos, worN=np.fft.rfftfreq(n, 1 / FS), fs=FS)
    np.testing.assert_allclose(lowpass_response(n, FS, f_high, order).numpy(), np.abs(h) ** 2, atol=1e-12)
    hard = lowpass_response(n, FS, f_high, None).numpy()
    np.testing.assert_array_equal(hard, (np.fft.rfftfreq(n, 1 / FS) <= f_high).astype(float))


def test_lowpass_glitch_matches_gengli_low_pass_glitch():
    """gengli.utils.low_pass_glitch (butter(3) + filtfilt) on a glitch padded with zeros."""
    g = _sine_gaussian(f0=300.0, tau=0.005).numpy() * scipy.signal.windows.tukey(512, alpha=0.25)
    x = np.pad(g, 2048)
    b, a = scipy.signal.butter(3, 250.0 / (0.5 * FS), btype="low", analog=False)
    ref = scipy.signal.filtfilt(b, a, x, axis=-1)
    ours = lowpass_glitch(torch.tensor(x), 250.0, FS).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-9 * np.abs(ref).max())
    xt = torch.tensor(x)
    assert lowpass_glitch(xt, None, FS) is xt


def test_lowpass_sets_snr_after_filtering():
    B, D, T = 2, 2, int(4 * FS)
    asd = _asd()
    g = _sine_gaussian(f0=400.0, tau=0.01).expand(B, D, -1).clone()
    snr = torch.tensor([[8.0, 20.0], [12.0, 0.0]], dtype=torch.float64)
    _, h = add_glitch(torch.zeros(B, D, T, dtype=torch.float64), g, asd, snr, FS, t_inj=2.0,
                      f_low=20.0, f_high=250.0, return_glitch=True)
    np.testing.assert_allclose(_snr_against_asd(h.numpy(), asd, FS), snr.numpy(), rtol=1e-6, atol=1e-9)
    # the content above f_high is suppressed by the Butterworth gain (>= 1/4 per bin at 250 Hz)
    f = np.fft.rfftfreq(T, 1 / FS)
    W = np.abs(np.fft.rfft(h.numpy(), axis=-1)) / np.interp(f, np.linspace(0, FS / 2, asd.shape[-1]), asd.numpy())
    assert W[..., f > 500].max() < 0.02 * W.max()
    with pytest.raises(ValueError):
        add_glitch(torch.zeros(B, D, T, dtype=torch.float64), g, asd, 5.0, FS, f_high=FS)


def test_injected_snr_is_exact_against_the_asd():
    B, D, T = 3, 2, int(8 * FS)
    asd = _asd()
    g = _sine_gaussian().expand(B, D, -1).clone()
    snr = torch.tensor([[8.0, 20.0], [12.0, 0.0], [30.0, 5.0]], dtype=torch.float64)
    t_inj = torch.tensor([[2.0, 4.0], [5.5, 4.0], [3.25, 6.0]], dtype=torch.float64)
    strain = torch.zeros(B, D, T, dtype=torch.float64)
    for f_low in (None, 20.0):
        out, h = add_glitch(strain, g, asd, snr, FS, t_inj=t_inj, f_low=f_low, return_glitch=True)
        np.testing.assert_allclose(_snr_against_asd(h, asd, FS, f_low), snr.numpy(), rtol=1e-6, atol=1e-9)
        assert torch.equal(out, strain + h)
    # SNR 0 leaves that series untouched
    assert torch.count_nonzero(h[1, 1]) == 0


def test_whitening_recovers_the_glitch_at_t_inj():
    """Coloured injection + SAGE's FIRWhitening gives back the glitch shape, centred at t_inj."""
    from sage.dsp.whiten import FIRWhitening

    T, fduration = int(16 * FS), 2.0
    asd = _asd()
    g = _sine_gaussian(f0=150.0, tau=0.02)[None, None]
    t_inj = 6.5
    _, h = add_glitch(torch.zeros(1, 1, T, dtype=torch.float64), g, asd, 20.0, FS,
                      t_inj=t_inj, taper_alpha=0.1, return_glitch=True)
    white = FIRWhitening(FS, fduration=fduration, dtype=torch.float64)(h, asd=asd[None])[0, 0]
    pad = int(fduration * FS) // 2
    t = (torch.arange(white.shape[-1]) + pad) / FS                  # FIRWhitening strips pad samples
    env = torch.from_numpy(np.abs(scipy.signal.hilbert(white.numpy())))
    assert abs(float(t[torch.argmax(env)]) - t_inj) < 2 / FS          # peak at the injection time
    # shape: compare with the tapered template around t_inj
    L = g.shape[-1]
    i0 = int(round(t_inj * FS)) - L // 2 - pad
    seg = white[i0:i0 + L]
    tmpl = g[0, 0] * tukey_window(L, 0.1)
    corr = float(torch.dot(seg, tmpl) / seg.norm() / tmpl.norm())
    assert corr > 0.99, corr


def test_module_float32_and_errors():
    inj = GlitchInjector(sample_rate=FS, f_low=20.0)
    T = int(4 * FS)
    strain = torch.randn(2, 2, T, dtype=torch.float32) * 1e-21
    g = _sine_gaussian().to(torch.float32).expand(2, 2, -1).clone()
    out = inj(strain, g, _asd().float(), snr=10.0, t_inj=2.0)
    assert out.dtype == torch.float32 and out.shape == strain.shape
    with pytest.raises(ValueError):                    # glitch would run off the end
        inj(strain, g, _asd(), snr=10.0, t_inj=3.95)
    with pytest.raises(ValueError):                    # leading shapes must match
        inj(strain, g[:1], _asd(), snr=10.0)


def test_matches_gengli_add_glitch():
    """With the same PSD, coloured injection + PyCBC-style whitening == gengli's whitened add_glitch."""
    gengli = pytest.importorskip("gengli")
    pytest.importorskip("pycbc")
    from pycbc.noise.gaussian import noise_from_psd
    import pycbc.psd
    from pycbc.types import TimeSeries as PyCBCTimeSeries

    fs, T_sec, t_inj = 4096.0, 4, 2.5
    n = int(T_sec * fs)
    psd_model = pycbc.psd.aLIGOZeroDetHighPower(n // 2 + 1, 1.0 / T_sec, 10.0)
    noise = np.asarray(noise_from_psd(n, 1.0 / fs, psd_model, seed=3))
    glitch = gengli.glitch_generator("L1").get_glitch(1, srate=fs, snr=25.0, seed=7)

    # gengli's add_glitch (examples/inject_glitch.py), whitened output, written out step by step
    white_noise, psd = PyCBCTimeSeries(noise, delta_t=1.0 / fs).whiten(2, 1, remove_corrupted=False, return_psd=True)
    L = glitch.shape[-1]
    i0 = int(t_inj * fs) - L // 2
    gengli_glitch = np.zeros(n)
    gengli_glitch[i0:i0 + L] = glitch * scipy.signal.windows.tukey(L, alpha=0.1)

    # torch: colour with the same PSD, keep gengli's amplitude. The SNR is taken on the placed
    # series: gengli's formula weights DC and Nyquist like every other bin, so the SNR of the
    # short glitch array differs from that of the same glitch in the long series by ~1e-4.
    asd = torch.tensor(np.sqrt(np.asarray(psd)))
    snr = glitch_snr(torch.tensor(gengli_glitch), fs)
    _, h = add_glitch(torch.zeros(n, dtype=torch.float64), torch.tensor(glitch), asd, snr, fs,
                      t_inj=t_inj, taper_alpha=0.1, return_glitch=True)
    ours_white = np.asarray((PyCBCTimeSeries(h.numpy(), delta_t=1.0 / fs).to_frequencyseries() / psd ** 0.5).to_timeseries())
    np.testing.assert_allclose(ours_white, gengli_glitch, atol=1e-6 * np.abs(gengli_glitch).max())



def test_lowpass_matches_gengli_low_pass_glitch_on_gengli_glitches():
    """gengli's own low_pass_glitch (fhigh) on real gengli glitches, padded with zeros."""
    gengli = pytest.importorskip("gengli")
    from gengli.utils import low_pass_glitch

    fs = 4096.0
    glitch = gengli.glitch_generator("H1").get_glitch(4, srate=fs, snr=20.0, seed=11)
    x = np.pad(glitch, ((0, 0), (4096, 4096)))
    for f_high in (150.0, 250.0, 500.0):
        ref = low_pass_glitch(x, f_high, fs)
        ours = lowpass_glitch(torch.tensor(x), f_high, fs).numpy()
        np.testing.assert_allclose(ours, ref, atol=1e-8 * np.abs(ref).max())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
