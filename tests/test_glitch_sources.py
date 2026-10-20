"""Tests for the glitch sources fed to GlitchInjector.

- sage.data.noise.glitch_bank: the DeepExtractor glitch bank, on a tiny fake bank written in
  the same format (no 2.3 GB download); the download itself against a stub huggingface_hub.
- sage.data.noise.analytic_glitches: the port of McGAN's analytic shapes, against McGAN's
  formulas evaluated with the same parameters.
- sage.data.noise.scattered_light: ArchEnemy's arch model, against the paper's equation on
  ArchEnemy's time grid.
"""

import collections
import math
import sys
import types

import numpy as np
import pytest
import scipy.signal
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
from sage.data.noise.scattered_light import scattered_light_arch

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


# -- ArchEnemy scattered-light arches --------------------------------------------------------

def test_scattered_light_matches_archenemy_model():
    fs = 2048.0
    h, p = scattered_light_arch(3, fs, duration=8.0, t0=4.0, generator=torch.Generator().manual_seed(0),
                                dtype=torch.float64, return_params=True)
    assert h.shape == (3, 16384)
    for i in range(3):
        f_f, T, phi = (float(p[k][i]) for k in ("fringe_frequency", "timeperiod", "phi"))
        assert 20 <= f_f <= 80 and 1.8 <= T <= 5.5
        # the paper's model on ArchEnemy's grid, with its Tukey(0.2) taper
        t = np.arange(-T / 2.0, T / 2.0, 1.0 / fs)
        f_rep = 1.0 / (2.0 * T)
        ref = np.sin((f_f / f_rep) * np.sin(2 * np.pi * f_rep * t) + phi)
        ref = ref * scipy.signal.windows.tukey(len(ref), 0.2)
        start = int(round((4.0 - T / 2) * fs))
        np.testing.assert_allclose(h[i, start:start + len(ref)].numpy(), ref, atol=1e-9)
        assert h[i, :start].abs().max() == 0 and h[i, start + len(ref):].abs().max() == 0


def test_scattered_light_injects_and_checks_fit():
    fs = 2048.0
    arch = scattered_light_arch(2, fs, duration=8.0, fringe_frequency=40.0, timeperiod=3.0,
                                dtype=torch.float64).view(1, 2, -1)
    strain = torch.zeros(1, 2, 16384, dtype=torch.float64)
    asd = torch.ones(8193, dtype=torch.float64)
    out = GlitchInjector(fs, f_low=15.0)(strain, arch, asd, snr=torch.tensor([[10.0, 20.0]]))
    torch.testing.assert_close(glitch_snr(out, fs), torch.tensor([[10.0, 20.0]], dtype=torch.float64))
    with pytest.raises(ValueError):
        scattered_light_arch(1, fs, duration=4.0, timeperiod=5.0, t0=1.0)


# -- glitch population (GWForge-style config) ------------------------------------------------

from pathlib import Path

from sage.data.noise.glitch_population import GlitchPopulation, power_law_snr

CONFIG = Path(__file__).resolve().parents[1] / "eucaif" / "sage_analytic_glitches.yaml"


def test_power_law_snr():
    g = torch.Generator().manual_seed(0)
    s = power_law_snr(200000, (5.0, 200.0), 2.3, g)
    assert s.min() >= 5.0 and s.max() <= 200.0
    # median of p ~ s^-2.3 on [5, 200]: inverse CDF at u = 1/2
    k = 1 - 2.3
    median = (0.5 * (5.0 ** k + 200.0 ** k)) ** (1 / k)
    assert float(s.median()) == pytest.approx(median, rel=0.02)
    lu = power_law_snr(200000, (5.0, 200.0), 1.0, g)
    assert float(torch.log(lu).mean()) == pytest.approx(0.5 * (math.log(5) + math.log(200)), rel=0.01)


def test_population_config_classes():
    pop = GlitchPopulation(CONFIG)
    assert pop.detectors == ["H1", "L1"] and pop.sample_rate == 4096.0
    assert pop.rate("Blip", "H1") == 1.3 and pop.rate("Tomte", "L1") == 6.2 and pop.rate("Ringdown", "H1") == 0.5
    g = torch.Generator().manual_seed(1)
    for name, c in pop.classes.items():
        glitch, params = pop.sample(name, 3, g, dtype=torch.float64)
        assert glitch.shape == (3, int(c["duration"] * 4096)) and torch.isfinite(glitch).all()
        assert ((params["snr"] >= 5) & (params["snr"] <= 200)).all()
    # Blip phase 0 or pi in GWForge's cosine convention: symmetric about the centre
    glitch, params = pop.sample("Blip", 4, g, dtype=torch.float64)
    assert set(params["phase"].tolist()) <= {0.0, math.pi}
    c = glitch.shape[-1] // 2
    torch.testing.assert_close(glitch[:, c - 200:c], glitch[:, c + 1:c + 201].flip(-1), atol=1e-3, rtol=0)
    assert (torch.sign(glitch[:, c]) == torch.where(params["phase"] == 0, 1.0, -1.0).double()).all()


def test_population_counts_and_injection():
    fs, T = 2048.0, 8.0
    n = int(fs * T)
    cfg = dict(detectors=["H1", "L1"], sample_rate=fs, classes=dict(
        Tomte=dict(generator="sine_gaussian", rate={"H1": 0.0, "L1": 2 * 3600 / T}, f0=[30, 65], q=[4, 12],
                   phases=[math.pi], snr_range=[10, 50], snr_index=2.2, duration=1.0)))
    pop = GlitchPopulation(cfg)
    g = torch.Generator().manual_seed(3)
    counts = [len(pop.draw(T, "L1", g)) for _ in range(400)]
    assert np.mean(counts) == pytest.approx(2.0, rel=0.1)
    assert all(len(pop.draw(T, "H1", g)) == 0 for _ in range(20))
    # a single glitch has exactly its drawn SNR; H1 untouched
    asd = torch.ones(n // 2 + 1, dtype=torch.float64)
    while True:
        out, events = pop.inject(torch.zeros(2, n, dtype=torch.float64), asd, g)
        if len(events[1]) == 1:
            break
    assert events[0] == [] and out[0].abs().max() == 0
    ev = events[1][0]
    assert float(glitch_snr(out[1], fs)) == pytest.approx(ev["snr"], rel=1e-6)
    assert ev["cls"] == "Tomte" and ev["detector"] == "L1" and 0.5 <= ev["time"] <= T - 0.5
    assert out[1].abs().argmax().item() / fs == pytest.approx(ev["time"], abs=0.05)


def test_population_rejects_bad_configs():
    base = dict(detectors=["H1"], sample_rate=2048.0)
    for bad in (dict(generator="koi_fish"),
                dict(generator="gaussian_blip", f0=[1, 2]),
                dict(generator="gaussian_blip", phases=[0.0]),
                dict(generator="sine_gaussian", rate={"V1": 1.0}),
                dict(generator="sine_gaussian", snr_range=[10, 5])):
        with pytest.raises(ValueError):
            GlitchPopulation(dict(base, classes=dict(X=bad)))


def test_population_dense_mode():
    pop = GlitchPopulation(CONFIG.with_name("sage_analytic_glitches_dense.yaml"))
    assert pop.per_segment == (5, 30)
    g = torch.Generator().manual_seed(4)
    counts, classes = [], collections.Counter()
    for _ in range(200):
        ev = pop.draw(8.0, "H1", g)
        counts.append(len(ev))
        classes.update(e["cls"] for e in ev)
        assert all(0.0 <= e["time"] <= 8.0 and 5.0 <= e["snr"] <= 100.0 for e in ev)
    assert min(counts) >= 5 and max(counts) <= 30 and np.mean(counts) == pytest.approx(17.5, rel=0.1)
    # weights: ScatteredLight 0.5, the six others 1
    frac = classes["ScatteredLight"] / sum(classes.values())
    assert frac == pytest.approx(0.5 / 6.5, rel=0.25)
    cfg = dict(detectors=["H1"], sample_rate=2048.0, glitches_per_segment=[3, 3], classes=dict(
        A=dict(generator="gaussian_blip", weight=0.0), B=dict(generator="ringdown", weight=2.0)))
    ev = GlitchPopulation(cfg).draw(4.0, "H1", g)
    assert len(ev) == 3 and {e["cls"] for e in ev} == {"B"}
    with pytest.raises(ValueError):
        GlitchPopulation(dict(cfg, glitches_per_segment=[5, 2]))


def test_population_window():
    cfg = dict(detectors=["H1"], sample_rate=2048.0, glitches_per_segment=[20, 20],
               classes=dict(B=dict(generator="sine_gaussian", duration=0.5)))
    pop = GlitchPopulation(cfg)
    ev = pop.draw(7.0, "H1", torch.Generator().manual_seed(0), window=(1.0, 6.0))
    assert len(ev) == 20 and all(1.25 <= e["time"] <= 5.75 for e in ev)
    with pytest.raises(ValueError):
        pop.draw(7.0, "H1", window=(1.0, 8.0))


def test_population_apply_equals_add_glitch():
    """The vectorised injection is add_glitch applied glitch by glitch."""
    from sage.data.noise.glitch_noise import add_glitch
    pop = GlitchPopulation(CONFIG.with_name("sage_analytic_glitches_dense.yaml"))
    fs = pop.sample_rate
    n = int(6 * fs)
    g = torch.Generator().manual_seed(7)
    strain = torch.randn(4, 1, n, generator=g, dtype=torch.float64)
    asd = torch.rand(1, 1000, generator=g, dtype=torch.float64) + 0.5
    out, events = pop.inject(strain, asd, g, f_low=10.0, f_high=900.0, window=(1.0, 5.0))
    ref = strain.reshape(-1, n).clone()
    asd_s = asd.expand(4, 1, 1000).reshape(-1, 1000)
    for s, evs in enumerate(events):
        for e in evs:
            ref[s:s + 1] = add_glitch(ref[s:s + 1], e["glitch"][None], asd_s[s], e["snr"], fs, t_inj=e["time"],
                                      f_low=10.0, f_high=900.0)
    assert sum(map(len, events)) > 30
    torch.testing.assert_close(out.reshape(-1, n), ref, rtol=0, atol=1e-9 * float(ref.abs().max()))
