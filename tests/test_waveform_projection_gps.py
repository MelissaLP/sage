"""Tests for the GPS sidereal time of ConstantProjection: gmst_from_gps against astropy / LAL and the antenna
patterns and delays at a GPS time against PyCBC's Detector."""

import numpy as np
import pytest
import torch

from sage.core import config as sage_config
from sage.data.waveform.project import ConstantProjection, gmst_from_gps

# GW150914, both sides of the 2015-07-01 and 2017-01-01 leap seconds, O3a, O4a
GPS = np.array([1126259462.4, 1119744016.5, 1119744017.5, 1167264017.5, 1167264018.5, 1246322820.0, 1380000000.0])


@pytest.fixture
def projection():
    class CFG:
        dtype = torch.float64
        device = "cpu"
        batch_size = 8
        class_balance = 0.5
        detectors = ["H1", "L1", "V1"]

    class DataCFG:
        sample_rate = 2048.0
        padded_length_in_s = 4.0

    saved = sage_config._CFG, sage_config._DATA_CFG
    sage_config.register_configs(CFG(), DataCFG())
    yield ConstantProjection()
    sage_config._CFG, sage_config._DATA_CFG = saved


def test_gmst_matches_astropy():
    from astropy.time import Time
    ref = Time(GPS, format="gps").sidereal_time("mean", "greenwich").rad
    got = gmst_from_gps(GPS).numpy()
    assert got.dtype == np.float64 and ((got >= 0) & (got < 2 * np.pi)).all()
    np.testing.assert_allclose(got, ref, atol=1e-4)
    assert float(gmst_from_gps(1126259462.4)) == pytest.approx(2.45655, abs=1e-4)


def test_gmst_matches_lal():
    lal = pytest.importorskip("lal")
    ref = [lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(t))) % (2 * np.pi) for t in GPS]
    np.testing.assert_allclose(gmst_from_gps(torch.tensor(GPS)).numpy(), ref, atol=1e-4)


def test_set_gps_matches_pycbc(projection):
    det = pytest.importorskip("pycbc.detector")
    rng = np.random.default_rng(0)
    n = len(GPS)
    ra, dec, pol = rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n)), rng.uniform(0, np.pi, n)
    projection.set_gps(GPS)
    g = projection.gmst_estimate(n)
    fp, fc = projection.antenna_pattern(*map(torch.tensor, (ra, dec, pol)), g)
    dt = projection.time_delay_from_earth_center(torch.tensor(ra), torch.tensor(dec), g)
    for k, name in enumerate(projection.detnames):
        d = det.Detector(name)
        ref = np.array([d.antenna_pattern(ra[i], dec[i], pol[i], GPS[i]) for i in range(n)])
        np.testing.assert_allclose(fp[:, k].numpy(), ref[:, 0], atol=2e-4)
        np.testing.assert_allclose(fc[:, k].numpy(), ref[:, 1], atol=2e-4)
        delay = [d.time_delay_from_earth_center(ra[i], dec[i], GPS[i]) for i in range(n)]
        np.testing.assert_allclose(dt[:, k].numpy(), delay, atol=1e-6)


def test_set_gps_shapes_and_back_to_random(projection):
    projection.set_gps(GPS[0])                                         # one time for the whole batch
    assert torch.equal(projection.gmst_estimate(4), gmst_from_gps(GPS[0]).expand(4))
    projection.set_gps(GPS[:3])
    with pytest.raises(ValueError, match="3 GPS times for a batch of 4"):
        projection.gmst_estimate(4)
    projection.set_gps(None)
    a, b = projection.gmst_estimate(4), projection.gmst_estimate(4)
    assert a.shape == (4,) and not torch.equal(a, b)                   # random again
