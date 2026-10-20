"""Tests for eucaif/glitch_frames.py: on-the-fly and stored glitch frames in the same format."""

from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")
pytest.importorskip("pycbc")                 # the analytic ASDs (and provenance) come from PyCBC

from eucaif.glitch_frames import GlitchFramesOnTheFly, StoredGlitchFrames, frame_columns
from eucaif.make_glitch_dataset import GlitchDatasetGenerator
from sage.dsp.qtransform import QTransform

CONFIG = Path(__file__).resolve().parents[1] / "eucaif" / "sage_analytic_glitches_dense.yaml"


def test_on_the_fly_is_reproducible_per_epoch_and_batch():
    ds = GlitchFramesOnTheFly(CONFIG, batch_size=2, batches_per_epoch=2, seed=5)
    first = list(ds)
    assert len(first) == 2 and first[0]["x"].shape == (2, 1, 24576) and first[0]["glitch"].shape == (2, 1, 24576)
    assert ds.frame_slice == (4096, 20480) and float(ds.t[ds.frame_slice[0]]) == 0.0
    again = list(ds)
    torch.testing.assert_close(first[1]["x"], again[1]["x"])
    assert not torch.equal(first[0]["x"], first[1]["x"])
    ds.set_epoch(1)
    assert not torch.equal(list(ds)[0]["x"], first[0]["x"])
    noise = first[0]["x"] - first[0]["glitch"]
    assert 0.9 < float(noise.std()) < 1.1
    assert (first[0]["n_glitches"] >= 5).all() and (first[0]["n_glitches"] <= 30).all()


def test_stored_frames_round_trip(tmp_path):
    from eucaif.make_glitch_dataset import write_dataset
    gen = GlitchDatasetGenerator(CONFIG, seed=3)
    path = write_dataset(gen, tmp_path / "frames.h5", n_frames=5, chunk=2)
    ds = StoredGlitchFrames(path)
    assert len(ds) == 5 and ds.classes == gen.classes and ds.frame_slice == (4096, 20480)
    item = ds[4]
    assert item["x"].shape == (1, 24576) and item["glitch"].shape == (1, 24576)
    rows = ds.glitches(4)
    assert len(rows["frame"]) == int(item["n_glitches"].sum()) and (rows["frame"] == 4).all()
    # the stored frames are those of the generator's chunks (seed * 1_000_003 + chunk index)
    x, glitch, counts, table = gen.generate(1, chunk_seed=3 * 1_000_003 + 2)
    np.testing.assert_allclose(item["x"].numpy(), x[0].numpy(), rtol=0, atol=1e-5)


def test_frame_columns():
    fs, L = 4096.0, 24576
    qt = QTransform(fs, L, qrange=(45, 45), frange=(10, 1000), norm=False)
    cols = frame_columns(qt, 45.0, 1.0, 4.0)
    t = qt.tile_times(45.0)
    assert float(t[cols.start]) >= 1.0 and float(t[cols.start - 1]) < 1.0
    assert float(t[cols.stop - 1]) < 5.0 and float(t[cols.stop]) >= 5.0
