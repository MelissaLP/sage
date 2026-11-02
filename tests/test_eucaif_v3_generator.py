"""Tests for eucaif/v3/generator.py, offline: a stub source of coloured Gaussian noise feeds the v3 cache."""

import copy
import math

import numpy as np
import pytest
import torch

from eucaif.v3 import segments as S
from eucaif.v3 import sources as SRC
from eucaif.v3.generator import EucaifV3, load_config, without

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
        cfg[k] = v if v is None or isinstance(v, list) else dict(cfg.get(k) or {}, **v)
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


def test_replay_regenerates_an_example_exactly(tmp_path):
    gen = _generator(tmp_path, seed=5)
    b = gen.batch(8, key=(3, 1))
    for g in (gen, _generator(tmp_path, seed=5)):                                # also a fresh generator
        ex = g.replay(b["recipes"][5])
        torch.testing.assert_close(ex["x"], b["x"][5:6], rtol=0, atol=0)
        assert ex["glitch_events"][0] == b["glitch_events"][5] and float(ex["gps"]) == float(b["gps"][5])
    assert not torch.equal(gen.batch(8, key=(3, 2))["x"], b["x"])                 # another key, other data
    with pytest.raises(ValueError, match="another config or seed"):
        _generator(tmp_path, seed=6).replay(b["recipes"][0])


def test_matched_pairs_are_the_example_without_the_component(tmp_path):
    b = _generator(tmp_path, seed=7).batch(8, key=(0, 0))
    off = _generator(tmp_path, seed=7, glitches=None).batch(8, key=(0, 0))          # same key, glitches off
    torch.testing.assert_close(off["y"], b["y"])
    torch.testing.assert_close(off["x"], without(b, "glitch"), rtol=0, atol=1e-4)
    noise = _generator(tmp_path, seed=7, glitches=None, signal=dict(fraction=0.0)).batch(8, key=(0, 0))
    pos = b["y"] == 1                                                              # noise-only twins of the BBHs
    torch.testing.assert_close(noise["x"][pos], without(b, "glitch", "signal")[pos], rtol=0, atol=1e-4)


def test_observed_glitches_centre_windows_on_tagged_glitches(tmp_path):
    import pandas as pd
    t0 = SEG[0]
    pd.DataFrame(dict(ifo=["H1", "H1", "H1", "H1", "L1"], ml_label=["Blip", "Blip", "None_of_the_Above", "Koi_Fish",
                                                                    "Tomte"],
                      event_time=[t0 + 100.25, t0 + 300.5, t0 + 400.0, t0 + 500.0, t0 + 10.0],   # Tomte: too early
                      ml_confidence=[0.95, 0.97, 0.99, 0.5, 0.99])).to_csv(tmp_path / "gs.csv", index=False)
    tags = [dict(path=str(tmp_path / "gs.csv"), detector_column="ifo", gps_column="event_time",
                 label_column="ml_label", confidence_column="ml_confidence")]
    gen = _generator(tmp_path, seed=8, glitch_tags=tags,
                     observed_glitches=dict(fraction=0.5, min_confidence=0.9, time_in_frame=[0.5, 1.5]))
    assert sorted(gen.obs_pool["gps"]) == [t0 + 100.25, t0 + 300.5]                 # labels, confidence, fit
    b = gen.batch(8)
    obs = [i for i, o in enumerate(b["observed"]) if o is not None]
    assert len(obs) == 4
    for i in obs:
        o = b["observed"][i]
        assert o["label"] == "Blip" and o["detector"] == "H1" and 1.5 - 1e-3 <= o["time"] <= 2.5 + 1e-3  # in frame
        assert any(abs(g - o["gps"]) < 1e-6 for g, _, _ in b["tags"][i][0])        # and tagged
        assert (b["glitch"][i] == 0).all() and b["glitch_events"][i] == []          # the real glitch alone
    ex = gen.replay(b["recipes"][obs[0]])
    assert ex["observed"][0] == b["observed"][obs[0]]
