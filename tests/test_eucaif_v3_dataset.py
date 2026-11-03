"""Tests for eucaif/v3/dataset.py, offline: the cache is filled beforehand from a stub source of coloured noise and
the stream must then read only it."""

import copy
import math
import shutil

import numpy as np
import pytest
import torch

from eucaif.v3 import segments as S
from eucaif.v3 import sources as SRC
from eucaif.v3.dataset import EucaifV3Stream, make_loader
from eucaif.v3.generator import load_config

FS = 2048.0
SEG = (1_240_000_000.0, 1_240_000_640.0)
CHUNK = 128


class ColouredNoise(SRC.Source):
    name = "stub"

    def fetch(self, detector, start, end):
        n = int(round((end - start) * FS))
        rng = np.random.default_rng([int(start), {"H1": 1, "L1": 2}[detector]])
        f = np.fft.rfftfreq(n, 1 / FS)
        asd = 1e-23 * np.sqrt((20.0 / np.maximum(f, 10.0)) ** 8 + 1.0)
        x = np.fft.irfft(np.fft.rfft(rng.normal(size=n)) * asd * math.sqrt(FS / 2), n=n)
        return x, FS, dict(source="stub", detector=detector, native_sample_rate=FS)


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v3")
    cache = SRC.StrainCache(tmp / "cache", [ColouredNoise()], sample_rate=FS, chunk_seconds=CHUNK, log=lambda *a: None)
    for det in ("H1", "L1"):
        for lo, hi in cache.pieces(*SEG, SEG):
            cache.piece(det, lo, hi, SEG)
    cfg = copy.deepcopy(load_config("eucaif/v3/eucaif_v3.yaml"))
    S.save(dict(detectors=["H1", "L1"], segments={"O3a": [SEG]}), tmp / "seg.json")
    cfg["segments"]["file"] = str(tmp / "seg.json")
    cfg["cache"] = dict(directory=str(tmp / "cache"), chunk_seconds=CHUNK, fetch_missing=True)
    cfg["split"] = dict(mode="random")
    return cfg


def _stream(cfg, **kw):
    return EucaifV3Stream(cfg, batch_size=4, log=lambda *a: None, **kw)


def test_noise_only_from_the_cache(config):
    ds = _stream(config, batches=2, overrides=dict(signal=dict(fraction=0.0), glitches=None))
    assert ds.cfg["cache"]["fetch_missing"] is False                     # never the network
    out = list(ds)
    assert len(out) == 2
    for b in out:
        assert b["x"].shape == (4, 2, 4 * 2048) and (b["y"] == 0).all()
        assert (b["signal"] == 0).all() and (b["glitch"] == 0).all() and not any(b["glitch_events"])
        assert ((b["gps"] >= SEG[0]) & (b["gps"] + 4 <= SEG[1])).all()


def test_draws_only_where_the_cache_has_every_detector(config, tmp_path):
    cfg = copy.deepcopy(config)
    cache = tmp_path / "cache"
    shutil.copytree(config["cache"]["directory"], cache)
    hole = (SEG[0] - SEG[0] % CHUNK + 2 * CHUNK, SEG[0] - SEG[0] % CHUNK + 3 * CHUNK)  # one L1 piece missing
    for p in (cache / "L1").glob(f"L1-{int(hole[0])}-*"):
        p.unlink()
    cfg["cache"]["directory"] = str(cache)
    ds = _stream(cfg, batches=4)
    for b in ds:
        gps = b["gps"].numpy()                                          # stored window = context centre +- 2 s
        start, end = gps - 14.0, gps - 14.0 + 32.0
        assert ((end <= hole[0]) | (start >= hole[1])).all()
        assert ((b["y"] == 1).sum() == 2) and torch.isfinite(b["x"]).all()


def test_epochs_differ_and_reproduce(config):
    ds = _stream(config, batches=1, overrides=dict(signal=dict(fraction=0.0), glitches=None))
    a = next(iter(ds))["gps"]
    assert torch.equal(a, next(iter(ds))["gps"])
    ds.set_epoch(1)
    assert not torch.equal(a, next(iter(ds))["gps"])


def test_loader_workers_draw_different_windows(config):
    loader = make_loader(config, batch_size=4, num_workers=2, batches=2,
                         overrides=dict(signal=dict(fraction=0.0), glitches=None))
    out = list(loader)
    assert len(out) == 2 and not torch.equal(out[0]["gps"], out[1]["gps"])
