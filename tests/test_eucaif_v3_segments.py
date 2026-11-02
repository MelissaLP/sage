"""Tests for eucaif/v3/segments.py: interval arithmetic, network segments with vetoes, both split modes, the
window sampler and glitch tags. GWOSC is stubbed: no network access."""

import numpy as np
import pytest

from eucaif.v3 import segments as S


def test_interval_arithmetic():
    assert S.merge([(5, 6), (0, 2), (1, 3)]) == [(0, 3), (5, 6)]
    assert S.intersect([(0, 10), (20, 30)], [(5, 25)]) == [(5, 10), (20, 25)]
    assert S.subtract([(0, 10), (20, 30)], [(2, 3), (8, 22)]) == [(0, 2), (3, 8), (22, 30)]
    assert S.total([(0, 10), (20, 25)]) == 15


def test_network_segments_vetoes_and_min_length():
    h1 = [(0, 100), (200, 300)]
    l1 = [(50, 250)]
    vetoes = [(60, 70)]
    # H1 & L1 = (50, 100), (200, 250); minus the veto: (50, 60), (70, 100), (200, 250); >= 20 s kept
    assert S.network_segments([h1, l1], vetoes, 20) == [(70, 100), (200, 250)]


def test_build_with_stubbed_gwosc(monkeypatch):
    good = {("O3a", "H1"): [(0, 1000)], ("O3a", "L1"): [(100, 1100)]}
    monkeypatch.setattr(S, "good_segments", lambda run, det, session=None: good[(run, det)])
    monkeypatch.setattr(S, "event_times", lambda session=None: [500.0])
    out = S.build(["O3a"], ["H1", "L1"], min_length=32, veto_before=30, veto_after=10, session=object(),
                  log=lambda *a: None)
    assert out["segments"]["O3a"] == [(100, 470), (510, 1000)]
    assert out["veto"]["n_events"] == 1
    assert out["hours"]["O3a"]["network"] == pytest.approx(860 / 3600)


def test_save_and_load(tmp_path):
    d = dict(segments={"O3a": [(0.0, 10.0)]}, detectors=["H1"])
    S.save(d, tmp_path / "s.json")
    assert S.load(tmp_path / "s.json")["segments"] == {"O3a": [(0.0, 10.0)]}


def test_gps_block_split_disjoint_complete_reproducible():
    seg = {"O3a": [(1000.0, 1000.0 + 50 * 4096 + 123.0)], "O3b": [(9e6, 9e6 + 20 * 4096)]}
    frac = dict(train=0.6, validation=0.2, test=0.2)
    a = S.gps_block_split(seg, 4096, frac, seed=1)
    assert a == S.gps_block_split(seg, 4096, frac, seed=1)
    assert a != S.gps_block_split(seg, 4096, frac, seed=2)
    pieces = sorted(p for v in a.values() for p in v)
    assert S.total(pieces) == pytest.approx(sum(S.total(v) for v in seg.values()))
    for v in a.values():                                   # every piece inside one GPS block
        for lo, hi in v:
            assert np.floor(lo / 4096) == np.floor((hi - 1e-6) / 4096)
    for x in a:                                            # splits disjoint
        for y in a:
            if x < y:
                assert S.intersect(a[x], a[y]) == []
    assert S.total(a["train"]) / S.total(pieces) == pytest.approx(0.6, abs=0.15)
    with pytest.raises(ValueError):
        S.gps_block_split(seg, 4096, dict(train=0.5, test=0.2), seed=0)


def test_random_split_shares_all_time():
    seg = {"O3a": [(0.0, 100.0)], "O3b": [(200.0, 300.0)]}
    r = S.split_segments(seg, dict(mode="random"))
    assert r["train"] == r["validation"] == r["test"] == [(0.0, 100.0), (200.0, 300.0)]
    with pytest.raises(ValueError):
        S.split_segments(seg, dict(mode="shuffle"))


def test_window_sampler_inside_and_duration_weighted():
    seg = [(0.0, 132.0), (1000.0, 1032.0), (2000.0, 2010.0)]          # room: 100 s, 0 s, too short
    ws = S.WindowSampler(seg, 32.0)
    starts = ws.sample(20000, np.random.default_rng(0))
    inside = ((starts >= 0) & (starts + 32 <= 132)) | ((starts >= 1000) & (starts + 32 <= 1032))
    assert inside.all()
    assert (starts < 1000).mean() > 0.99                                # the first segment has all the room
    with pytest.raises(ValueError):
        S.WindowSampler([(0.0, 10.0)], 32.0)


def test_glitch_tags(tmp_path):
    (tmp_path / "gs.csv").write_text("event_time,ifo,ml_label,ml_confidence\n"
                                     "100.5,H1,Blip,0.95\n101.0,L1,Tomte,0.99\n102.0,H1,Koi_Fish,0.5\n")
    (tmp_path / "virgo.csv").write_text("GPStime,label,confidence\n100.2,Blip,0.97\n")
    tags = S.GlitchTags([
        dict(path=tmp_path / "gs.csv", detector_column="ifo", gps_column="event_time", label_column="ml_label",
             confidence_column="ml_confidence", min_confidence=0.9),
        dict(path=tmp_path / "virgo.csv", detector="V1", gps_column="GPStime", label_column="label",
             confidence_column="confidence")])
    assert tags.detectors() == ["H1", "L1", "V1"]
    assert tags.in_window("H1", 100, 104) == [(100.5, "Blip", 0.95)]          # Koi_Fish below 0.9 dropped
    assert tags.in_window("L1", 100, 101) == []                                 # end excluded
    assert tags.in_window("V1", 100, 101)[0][1] == "Blip"
    assert tags.in_window("K1", 0, 1e10) is None                                # no table: unknown
