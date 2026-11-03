"""Tests for eucaif/inspect_dataset.py on a small v2 file: a sound file passes, a broken one fails."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pycbc")
h5py = pytest.importorskip("h5py")

from eucaif import inspect_dataset as ins                              # noqa: E402
from eucaif.v2.eucaif_v2 import make_generator                            # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "eucaif" / "v2" / "eucaif_v2.yaml"


@pytest.fixture(scope="module")
def v2_file(tmp_path_factory):
    from eucaif.v1.make_whitened_examples import write_dataset
    path = tmp_path_factory.mktemp("inspect") / "v2.h5"
    write_dataset(make_generator(CONFIG, seed=2), path, n_per_class=4, chunk_per_class=4)
    return path


def test_sound_file_passes_and_has_in_frame_snr(v2_file, capsys):
    assert ins.main(["check", str(v2_file)]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "in-frame glitch SNR <= SNR" in out
    with h5py.File(v2_file, "r") as f:
        assert ins.kind(f) == "v2"
        snr, inf, fr = (f[f"glitches/{k}"][:] for k in ("snr", "snr_in_frame", "frame_fraction"))
        np.testing.assert_allclose(inf, snr * np.sqrt(fr))
    ins.main(["summary", str(v2_file)])
    ins.main(["example", str(v2_file), "1"])
    assert "glitches:" in capsys.readouterr().out


def test_broken_file_fails(v2_file, tmp_path):
    bad = tmp_path / "bad.h5"
    bad.write_bytes(v2_file.read_bytes())
    with h5py.File(bad, "a") as f:
        f["x"][0, 0, 100] = np.nan
        f["n_glitches"][0, 0] += 1                                       # table and counts disagree
    with h5py.File(bad, "r") as f:
        failed = {msg for status, msg in ins.check(f) if status == "FAIL"}
    assert failed == {"x (and glitch) are finite everywhere", "glitch table matches n_glitches"}
    assert ins.main(["check", str(bad)]) == 1
