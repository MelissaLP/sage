"""Tests for the glitch sources fed to GlitchInjector.

- sage.data.noise.glitch_bank: the DeepExtractor glitch bank, on a tiny fake bank written in
  the same format (no 2.3 GB download); the download itself against a stub huggingface_hub.
- sage.data.noise.analytic_glitches: the port of McGAN's analytic shapes, against McGAN's
  formulas evaluated with the same parameters.
"""

import math
import sys
import types

import numpy as np
import pytest
import torch

from sage.data.noise import glitch_bank
from sage.data.noise.analytic_glitches import (
    ANALYTIC_GLITCHES,
    analytic_glitch,
    gaussian_blip,
    ringdown,
    sine_gaussian,
    white_noise_burst,
)
from sage.data.noise.glitch_bank import DeepExtractorGlitchBank, FILES
from sage.data.noise.glitch_noise import GlitchInjector, glitch_snr

CLASSES = ["Blip", "Fast_Scattering", "Koi_Fish", "Low_Frequency_Burst", "Scattered_Light",
           "Tomte", "Whistle"]
L = 256


def _write_bank(path, per_class=6, n_unique=4, seed=0):
    """A bank like the real one: per class, ``n_unique`` glitches bootstrapped to ``per_class``."""
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for c in range(len(CLASSES)):
        unique = rng.normal(size=(n_unique, L))
        samples.append(unique[np.r_[np.arange(n_unique), rng.integers(n_unique, size=per_class - n_unique)]])
        labels += [c] * per_class
    np.save(path / FILES["samples"], np.concatenate(samples))
    np.save(path / FILES["labels"], np.eye(len(CLASSES))[labels])
    np.save(path / FILES["label_order"], np.array(CLASSES))
    return np.concatenate(samples), np.array(labels)


# -- DeepExtractor glitch bank ---------------------------------------------------------------

def test_bank_loads_and_deduplicates(tmp_path):
    samples, labels = _write_bank(tmp_path)
    bank = DeepExtractorGlitchBank(data_dir=tmp_path, download=False)
    assert bank.classes == tuple(CLASSES)
    assert len(bank) == 4 * len(CLASSES)
    assert len(np.unique(samples[bank.indices], axis=0)) == len(bank)
    np.testing.assert_array_equal(bank.labels, labels)
    full = DeepExtractorGlitchBank(data_dir=tmp_path, download=False, deduplicate=False)
    assert len(full) == len(samples)


def test_bank_get_sample_and_split(tmp_path):
    samples, _ = _write_bank(tmp_path)
    bank = DeepExtractorGlitchBank(data_dir=tmp_path, download=False)
    rows = np.array([5, 0, 17])
    np.testing.assert_allclose(bank.get(rows, dtype=torch.float64).numpy(), samples[rows])
    assert bank.get(rows, sample_rate=bank.sample_rate / 2).shape == (3, L // 2)

    g = torch.Generator().manual_seed(1)
    glitches, lab = bank.sample(50, classes=["Koi_Fish", 6], generator=g)
    assert glitches.shape == (50, L) and glitches.dtype == torch.float32
    assert set(lab.tolist()) <= {2, 6}

    train, test = bank.split(0.5, seed=0)
    assert len(np.intersect1d(train, test)) == 0
    assert len(train) + len(test) == len(bank)
    # no glitch on both sides, even though the bank repeats glitches
    rows_train = {samples[i].tobytes() for i in train}
    assert not any(samples[i].tobytes() in rows_train for i in test)
    _, lab = bank.sample(20, indices=test, classes="Blip")
    assert (lab == 0).all()


def test_bank_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeepExtractorGlitchBank(data_dir=tmp_path, download=False)


def test_bank_downloads_pinned_revision(tmp_path, monkeypatch):
    src = tmp_path / "hub"
    src.mkdir()
    _write_bank(src)
    calls = []

    def hf_hub_download(repo_id, filename, repo_type, revision, cache_dir, local_dir):
        calls.append((repo_id, filename, repo_type, revision))
        return str(src / filename)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download))
    bank = DeepExtractorGlitchBank()
    assert len(bank) == 4 * len(CLASSES)
    assert {c[1] for c in calls} == set(FILES.values())
    assert all(c[0] == glitch_bank.HF_REPO_ID and c[2] == "dataset" and c[3] == glitch_bank.HF_REVISION
               for c in calls)


# -- McGAN analytic glitches -----------------------------------------------------------------

FS = 1024.0


def _t(fs=FS, duration=1.0):
    return np.arange(int(duration * fs)) / fs


def _peak1(h):
    return h / np.abs(h).max(axis=-1, keepdims=True)


@pytest.mark.parametrize("kind", sorted(ANALYTIC_GLITCHES))
def test_shapes_peak_and_reproducible(kind):
    a = analytic_glitch(kind, 5, 2048.0, duration=2.0, generator=torch.Generator().manual_seed(3))
    b = analytic_glitch(kind, 5, 2048.0, duration=2.0, generator=torch.Generator().manual_seed(3))
    assert a.shape == (5, 4096) and a.dtype == torch.float32
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a.abs().amax(-1), torch.ones(5))


def test_sine_gaussian_formula():
    h, p = sine_gaussian(4, FS, generator=torch.Generator().manual_seed(0), dtype=torch.float64,
                         return_params=True)
    f0, q, t0, phi = (p[k].numpy()[:, None] for k in ("f0", "q", "t0", "phi"))
    assert ((70 <= f0) & (f0 <= 250)).all() and ((3 <= q) & (q <= 20)).all()
    assert ((0.4 <= t0) & (t0 <= 0.6)).all()
    assert len(np.unique(phi)) == 4                     # one phase per glitch
    tau = q / (np.sqrt(2) * np.pi * f0)
    t = _t()
    ref = np.exp(-1.0 * (t - t0) ** 2 / (tau ** 2)) * np.sin(2 * np.pi * f0 * (t - t0) + phi)
    np.testing.assert_allclose(h.numpy(), _peak1(ref), atol=1e-12)


def test_ringdown_formula():
    h, p = ringdown(4, FS, generator=torch.Generator().manual_seed(0), dtype=torch.float64,
                    return_params=True)
    f0, q, t0, phi = (p[k].numpy()[:, None] for k in ("f0", "q", "t0", "phi"))
    assert ((9 <= q) & (q <= 20)).all()
    tau = q / (np.sqrt(2) * np.pi * f0)
    t = _t()
    ref = ((t - t0) > 0) * np.exp(-1.0 * ((t - t0) / tau)) * np.sin(2 * np.pi * f0 * (t - t0) + phi)
    np.testing.assert_allclose(h.numpy(), _peak1(ref), atol=1e-12)


def test_gaussian_blip_formula():
    h, p = gaussian_blip(4, FS, generator=torch.Generator().manual_seed(0), dtype=torch.float64,
                         return_params=True)
    t0, tau = p["t0"].numpy()[:, None], p["tau"].numpy()[:, None]
    # McGAN leaves the blip unscaled; its sampled peak is just below 1 when t0 is off the grid
    np.testing.assert_allclose(h.numpy(), _peak1(np.exp(-(_t() - t0) ** 2 / tau ** 2)), atol=1e-12)


def test_white_noise_burst_band_and_envelope():
    h, p = white_noise_burst(64, FS, tau=1e-3, t0=0.5, generator=torch.Generator().manual_seed(0),
                             dtype=torch.float64, return_params=True)
    t = _t()
    # McGAN's envelope exp(-(t - t0)^2 / tau): nothing beyond a few widths sqrt(tau / 2)
    far = np.abs(t - 0.5) > 6 * math.sqrt(1e-3 / 2)
    assert h[:, far].abs().max() < 1e-6
    # power concentrated in McGAN's [70, 250] Hz band (envelope-broadened at the edges)
    power = (torch.fft.rfft(h, dim=-1).abs() ** 2).mean(0).numpy()
    f = np.fft.rfftfreq(h.shape[-1], 1 / FS)
    assert power[(f > 50) & (f < 300)].sum() > 0.95 * power.sum()


def test_analytic_glitch_injects_at_snr():
    fs, T = 2048.0, 8192
    glitch = sine_gaussian(2, fs, t0=0.5, f0=150.0, q=10.0, dtype=torch.float64).view(1, 2, -1)
    strain = torch.zeros(1, 2, T, dtype=torch.float64)
    asd = torch.ones(T // 2 + 1, dtype=torch.float64)
    out = GlitchInjector(fs, f_low=20.0)(strain, glitch, asd, snr=torch.tensor([[8.0, 12.0]]))
    torch.testing.assert_close(glitch_snr(out, fs), torch.tensor([[8.0, 12.0]], dtype=torch.float64))
    assert out[0, 0].abs().argmax().item() == pytest.approx(T // 2, abs=5)


def test_unknown_kind():
    with pytest.raises(ValueError):
        analytic_glitch("koi_fish", 1, FS)
