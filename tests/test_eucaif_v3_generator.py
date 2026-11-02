"""Tests for eucaif/v3/generator.py, offline: a stub source of coloured Gaussian noise feeds the v3 cache."""

import copy
import math

import numpy as np
import pytest
import torch

from eucaif.v3 import segments as S
from eucaif.v3 import sources as SRC
from eucaif.v3.generator import EucaifV3, load_config

FS = 2048.0
SEG = (1_240_000_000.0, 1_240_000_600.0)


def _asd(f):
    f = np.maximum(f, 10.0)
    return 1e-23 * np.sqrt((20.0 / f) ** 8 + 1.0 + (f / 300.0) ** 2)


class ColouredNoise(SRC.Source):
    """Gaussian noise with ASD _asd, at 2048 Hz, deterministic in (detector, start)."""

    name = "stub"

    def fetch(self, detector, start, end):
        n = int(round((end - start) * FS))
        rng = np.random.default_rng([int(start), {"H1": 1, "L1": 2}[detector]])
        white = rng.normal(size=n)
        f = np.fft.rfftfreq(n, 1 / FS)
        x = np.fft.irfft(np.fft.rfft(white) * _asd(f) * math.sqrt(FS / 2), n=n)
        return x, FS, dict(source="stub", detector=detector, native_sample_rate=FS)


def _generator(tmp_path, seed=0, **changes):
    cfg = load_config("eucaif/eucaif_v3.yaml")
    cfg = copy.deepcopy(cfg)
    S.save(dict(detectors=["H1", "L1"], segments={"O3a": [SEG]}), tmp_path / "seg.json")
    cfg["segments"]["file"] = str(tmp_path / "seg.json")
    cfg["cache"] = dict(directory=str(tmp_path / "cache"), chunk_seconds=128, fetch_missing=True)
    cfg["split"] = dict(mode="random")
    for k, v in changes.items():
        cfg[k] = dict(cfg[k], **v)
    gen = EucaifV3(cfg, split="train", seed=seed, batch_size=8, log=lambda *a: None)
    gen.cache.sources = [ColouredNoise()]
    return gen


def test_batch_shapes_labels_gps_and_sources(tmp_path):
    gen = _generator(tmp_path)
    b = gen.batch(8)
    assert b["x"].shape == b["signal"].shape == b["glitch"].shape == (8, 2, 4 * 2048)
    assert b["x"].dtype == torch.float32 and int(b["y"].sum()) == 4
    pos = b["y"] == 1
    assert torch.isfinite(b["snr"][pos]).all() and torch.isnan(b["snr"][~pos]).all()
    assert ((b["tc"][pos] >= 1.5) & (b["tc"][pos] <= 2.5)).all()                 # q_padding + tc_in_frame
    assert ((b["gps"] >= SEG[0]) & (b["gps"] + 4 <= SEG[1])).all()
    assert all(s == ["stub", "stub"] for s in b["sources"])
    assert (b["signal"][~pos] == 0).all()
    noise = (b["x"] - b["signal"] - b["glitch"]).double()
    assert float(noise.std()) == pytest.approx(1.0, abs=0.08)                    # whitened real-ish noise
    assert b["asd"].shape[:2] == (8, 2)


def test_snr_range_sets_the_network_snr_and_seed_reproduces(tmp_path):
    gen = _generator(tmp_path, signal=dict(snr_range=[12.0, 12.0]))
    b = gen.batch(8)
    pos = b["y"] == 1
    torch.testing.assert_close(b["snr"][pos], torch.full((4,), 12.0, dtype=torch.float64))
    again = _generator(tmp_path, signal=dict(snr_range=[12.0, 12.0])).batch(8)
    torch.testing.assert_close(b["x"], again["x"])
    assert b["glitch_events"] and all(abs(e["gps"] - (g + e["time"] - 14.0)) < 1e-6
                                      for ev, g in zip(b["glitch_events"], b["gps"].tolist()) for e in ev)


def test_exclude_centre_psd(tmp_path):
    gen = _generator(tmp_path, psd=dict(exclude_centre=True))
    b = gen.batch(4)
    assert torch.isfinite(b["x"]).all()


def test_network_population_keeps_only_the_network_detectors(tmp_path):
    from eucaif.v3.generator import _network_population
    cfg = load_config("eucaif/eucaif_v3.yaml")
    cfg["data"]["detectors"] = ["H1"]
    import yaml
    pop = yaml.safe_load(_network_population(cfg, tmp_path).read_text())
    assert pop["detectors"] == ["H1"] and set(pop["classes"]["Blip"]["rate"]) == {"H1"}
    cfg["data"]["detectors"] = ["H1", "V1"]
    with pytest.raises(ValueError, match="no rate for"):
        _network_population(cfg, tmp_path)


def test_antenna_factors_delays_and_in_frame_glitch_snr(tmp_path):
    det = pytest.importorskip("pycbc.detector")
    gen = _generator(tmp_path, seed=2)
    b = gen.batch(8)
    pos = (b["y"] == 1).numpy()
    assert torch.isnan(b["fplus"][~pos]).all() and torch.isfinite(b["delay"][pos]).all()
    centre = b["gps"].numpy() + 2.0                                              # stored window centre
    p = {k: b["params"][k].numpy() for k in ("ra", "dec", "polarization")}
    for i in np.nonzero(pos)[0]:
        for j, name in enumerate(gen.detectors):
            d = det.Detector(name)
            fp, fc = d.antenna_pattern(p["ra"][i], p["dec"][i], p["polarization"][i], centre[i])
            assert float(b["fplus"][i, j]) == pytest.approx(fp, abs=2e-3)
            assert float(b["fcross"][i, j]) == pytest.approx(fc, abs=2e-3)
            assert float(b["delay"][i, j]) == pytest.approx(d.time_delay_from_earth_center(
                p["ra"][i], p["dec"][i], centre[i]), abs=1e-5)
    events = [e for ev in b["glitch_events"] for e in ev]
    assert events and all(0 <= e["frame_fraction"] <= 1 + 1e-9 and e["snr_in_frame"] <= e["snr"] + 1e-9
                          for e in events)
