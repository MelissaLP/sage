"""Tests for eucaif/eucaif_v2.py: H1/L1 noise + BBH + O3a analytic glitches, on the fly and stored."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pycbc")          # analytic ASDs and the v1 waveform stack
h5py = pytest.importorskip("h5py")

from eucaif.eucaif_v2 import EucaifV2OnTheFly, QTransforms, load_config, make_generator

CONFIG = Path(__file__).resolve().parents[1] / "eucaif" / "eucaif_v2.yaml"


def test_generator_mixes_noise_bbh_and_glitches():
    gen = make_generator(CONFIG, seed=1)
    data = gen.generate(6, noise_seed=11, keep_signal=True)
    assert data["x"].shape == (12, 2, 8192) and gen.frame_slice == (2048, 6144)
    assert data["y"].tolist() == [0] * 6 + [1] * 6
    # tc inside the frame: q_padding + [0.5, 1.5]
    assert ((data["tc"] >= 1.5) & (data["tc"] <= 2.5)).all()
    noise = data["x"] - data["glitch_whitened"]
    noise[6:] -= data["signal_whitened"]
    assert 0.9 < float(noise.std()) < 1.1
    tab = data["glitch_table"]
    assert len(tab["example"]) == int(data["n_glitches"].sum())
    assert ((tab["time"] >= 0) & (tab["time"] <= 4.0)).all()
    # O3a classes, then the non-physical McGAN group: exactly one per (example, detector), Blip-like SNR
    assert gen.glitch_classes == ["Blip", "Tomte", "BlipLowFrequency", "ScatteredLight",
                                  "Ringdown", "WhiteNoiseBurst", "GaussianBlip"]
    nonphys = tab["class"] >= 4
    per_pair = np.zeros((12, 2), dtype=int)
    np.add.at(per_pair, (tab["example"][nonphys], tab["detector"][nonphys]), 1)
    assert (per_pair == 1).all()
    assert ((tab["snr"][nonphys] >= 5) & (tab["snr"][nonphys] <= 200)).all()


def test_on_the_fly_reproducible():
    ds = EucaifV2OnTheFly(CONFIG, batch_size=4, batches_per_epoch=2, seed=3)
    a, b = list(ds), list(ds)
    assert a[0]["x"].shape == (4, 2, 8192) and sorted(a[0]["y"].tolist()) == [0, 0, 1, 1]
    for k in ("x", "glitch", "signal"):
        torch.testing.assert_close(a[1][k], b[1][k])
    assert not torch.equal(a[0]["x"], a[1]["x"])
    # labels match the targets: noise-only examples have no BBH signal
    assert (a[0]["signal"][a[0]["y"] == 0] == 0).all() and (a[0]["signal"][a[0]["y"] == 1].abs().sum(-1) > 0).all()


def test_qtransform_inputs():
    qt = QTransforms(CONFIG)
    x = torch.randn(3, 2, 8192)
    assert qt.frame(x).shape == (3, 2, 4096)
    assert qt.qscan(x).shape == (3, 2, 256, 256)
    tiles = qt.tiles(x)
    assert set(tiles) == {5.7, 45.3}
    rec = qt.inverse(tiles[45.3], 45.3)
    f = torch.fft.rfftfreq(8192, 1 / 2048.0)
    band = (f > 20) & (f < 900)
    X, R = torch.fft.rfft(x), torch.fft.rfft(rec)
    assert float((X - R)[..., band].abs().max() / X[..., band].abs().max()) < 1e-4


def test_written_file(tmp_path):
    from make_whitened_examples import write_dataset
    gen = make_generator(CONFIG, seed=2)
    write_dataset(gen, tmp_path / "v2.h5", n_per_class=3, chunk_per_class=2)
    with h5py.File(tmp_path / "v2.h5", "r") as f:
        n = f["x"].shape[0]
        assert f["x"].shape == (n, 2, 8192) and f["glitch"].shape == (n, 2, 8192) and n == 6
        ex = f["glitches/example"][:]
        assert len(ex) == int(f["n_glitches"][:].sum()) and (ex >= 0).all() and (ex < n).all()
        counts = np.bincount(ex, minlength=n)
        np.testing.assert_array_equal(counts, f["n_glitches"][:].sum(1))
        assert f.attrs["dataset_version"] == "eucaif v2" and "glitches:" in f.attrs["eucaif_v2_config"]


def test_nonphysical_off_and_config_errors():
    cfg = load_config(CONFIG)
    cfg["nonphysical_glitches"]["probability"] = 0.0
    gen = make_generator(cfg, seed=4)
    assert gen.glitch_classes == ["Blip", "Tomte", "BlipLowFrequency", "ScatteredLight"]
    cfg["nonphysical_glitches"].update(probability=1.0, classes=["Koi_Fish"])
    with pytest.raises(ValueError):
        make_generator(cfg, seed=4)
