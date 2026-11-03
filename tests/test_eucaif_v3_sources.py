"""Tests for eucaif/v3/sources.py with stub sources (no network): decimation, the cache's pieces and margins,
GPS alignment, reuse and the fallback between sources."""

import numpy as np
import pytest

from eucaif.v3 import sources as SRC


def _signal(t):
    return np.sin(2 * np.pi * 37.0 * t) + 0.3 * np.sin(2 * np.pi * 211.0 * (t - 1e9))


class StubSource(SRC.Source):
    """Returns _signal at native_rate for any [start, end); counts calls; can be made to fail."""

    def __init__(self, name="stub", native_rate=16384, fail=False):
        self.name, self.rate, self.fail, self.calls = name, native_rate, fail, 0

    def fetch(self, detector, start, end):
        self.calls += 1
        if self.fail:
            raise ConnectionError("source down")
        t = start + np.arange(int(round((end - start) * self.rate))) / self.rate
        prov = dict(source=self.name, detector=detector, native_sample_rate=float(self.rate))
        return _signal(t), float(self.rate), prov


SEG = (1_000_000_000.0, 1_000_001_000.0)


def test_window_is_the_decimated_signal_on_the_gps_grid(tmp_path):
    cache = SRC.StrainCache(tmp_path, [StubSource()], sample_rate=2048, chunk_seconds=64, log=lambda *a: None)
    x, t0, provs = cache.window("H1", SEG[0] + 10.3, 32.0, SEG)                 # inside one 64 s piece
    assert x.shape == (32 * 2048,) and t0 * 2048 == int(t0 * 2048)               # on the 2048 Hz grid
    assert t0 - (SEG[0] + 10.3) < 1 / 2048 and t0 >= SEG[0] + 10.3
    ref = _signal(t0 + np.arange(x.size) / 2048)
    assert np.abs(x - ref).max() < 2e-3                                           # passband of the half-band FIR
    assert provs[0]["source"] == "stub" and provs[0]["sample_rate"] == 2048.0


def test_window_across_pieces_matches_one_decimation(tmp_path):
    src = StubSource()
    cache = SRC.StrainCache(tmp_path, [src], sample_rate=2048, chunk_seconds=64, margin_seconds=1.0,
                            log=lambda *a: None)
    start = SEG[0] + 64 * 3 - 10.0                                                # crosses a piece edge
    x, t0, provs = cache.window("H1", start, 32.0, SEG)
    assert len(provs) == 2
    native, rate, _ = src.fetch("H1", t0 - 4, t0 + 36)
    direct = SRC.resample(native, rate, 2048)[4 * 2048:4 * 2048 + x.size]
    assert np.abs(x - direct).max() < 1e-6                                        # float32 storage only
    calls = src.calls
    cache.window("H1", start, 32.0, SEG)
    assert src.calls == calls                                                     # served from the cache


def test_fallback_to_the_next_source(tmp_path):
    down, frames = StubSource("gwosc", fail=True), StubSource("frames")
    cache = SRC.StrainCache(tmp_path, [down, frames], sample_rate=2048, chunk_seconds=64, log=lambda *a: None)
    _, _, provs = cache.window("L1", SEG[0] + 10, 16.0, SEG)
    assert provs[0]["source"] == "frames" and "gwosc: ConnectionError" in provs[0]["failed_sources"][0]
    dead = SRC.StrainCache(tmp_path / "x", [StubSource(fail=True)], sample_rate=2048, log=lambda *a: None)
    with pytest.raises(RuntimeError):
        dead.window("L1", SEG[0] + 10, 16.0, SEG)


def test_window_edges_and_resample_checks(tmp_path):
    cache = SRC.StrainCache(tmp_path, [StubSource()], sample_rate=2048, chunk_seconds=64, log=lambda *a: None)
    x, t0, _ = cache.window("H1", SEG[1] - 32.0, 32.0, SEG)                      # flush with the segment end
    assert x.size == 32 * 2048 and t0 + 32 <= SEG[1]
    with pytest.raises(ValueError):
        cache.window("H1", SEG[0], 32.0, (SEG[0], SEG[0] + 16))                   # does not fit
    with pytest.raises(ValueError):
        SRC.resample(np.zeros(100), 12000, 2048)                                  # not a power-of-2 ratio
    assert SRC.resample(np.zeros(4096), 4096, 2048).shape == (2048,)


def test_make_sources():
    s = SRC.make_sources(dict(order=["gwosc", "frames"], gwosc=dict(sample_rate=4096),
                              frames=dict(channels={"V1": "V1:Hrec_hoft_16384Hz"}, frametypes={"V1": "HoftOnline"})))
    assert [x.name for x in s] == ["gwosc", "frames"] and s[0].sample_rate == 4096
    with pytest.raises(ValueError):
        s[1].fetch("H1", 0, 1)                                                    # no channel for H1
    with pytest.raises(ValueError):
        SRC.make_sources(dict(order=["nds"]))


def test_without_margin_the_edge_effect_stays_near_the_piece_edge(tmp_path):
    src = StubSource()
    cache = SRC.StrainCache(tmp_path, [src], sample_rate=2048, chunk_seconds=64, log=lambda *a: None)
    x, t0, _ = cache.window("H1", SEG[0] + 64 * 3 - 10.0, 32.0, SEG)
    native, rate, _ = src.fetch("H1", t0 - 4, t0 + 36)
    err = np.abs(x - SRC.resample(native, rate, 2048)[4 * 2048:4 * 2048 + x.size])
    edge = int(round((SEG[0] + 64 * 3 - t0) * 2048))                              # sample of the piece edge
    near = np.abs(np.arange(x.size) - edge) < int(0.01 * 2048)                    # within 10 ms of it
    assert err[~near].max() < 1e-6 and err[near].max() > 1e-6


class CountingSpan(StubSource):
    """Fails for spans longer than ``max_span`` s (as a source with a gap would), counts fetches."""

    def __init__(self, max_span=1e9):
        super().__init__()
        self.max_span, self.spans = max_span, []

    def fetch(self, detector, start, end):
        if end - start > self.max_span:
            raise ValueError("gap in the data")
        self.spans.append((start, end))
        return super().fetch(detector, start, end)


def test_fill_chunk_fetches_each_chunk_once_and_falls_back_per_piece(tmp_path):
    pieces = [(SEG[0] + 10.0, SEG[0] + 200.0), (SEG[0] + 300.0, SEG[0] + 500.0)]      # two segments, one chunk
    src = CountingSpan()
    cache = SRC.StrainCache(tmp_path / "a", [src], sample_rate=2048, chunk_seconds=512, log=lambda *a: None)
    assert cache.fill_chunk("H1", pieces) == 2 and src.spans == [(pieces[0][0], pieces[1][1])]   # one fetch
    assert cache.fill_chunk("H1", pieces) == 0                                                   # already there
    x, _, _ = cache.window("H1", SEG[0] + 350.0, 32.0, pieces[1])
    assert np.abs(x - _signal(SEG[0] + 350.0 + np.arange(x.size) / 2048)).max() < 2e-3
    gappy = CountingSpan(max_span=400.0)                                        # the whole span fails
    cache = SRC.StrainCache(tmp_path / "b", [gappy], sample_rate=2048, chunk_seconds=512, log=lambda *a: None)
    assert cache.fill_chunk("H1", pieces) == 2 and gappy.spans == pieces       # one fetch per piece


def test_training_reads_only_the_cache(tmp_path):
    cache = SRC.StrainCache(tmp_path, [StubSource()], sample_rate=2048, fetch_missing=False, log=lambda *a: None)
    with pytest.raises(FileNotFoundError, match="download"):
        cache.window("H1", SEG[0] + 10, 16.0, SEG)
    small = (SEG[0], SEG[0] + 64.0)
    cache.fill_chunk("H1", [small])
    assert cache.window("H1", SEG[0] + 10, 16.0, small)[0].size == 16 * 2048


def test_plan_groups_pieces_by_chunk():
    seg = {"O3a": [(4096.0 * 2 + 100, 4096.0 * 3 + 50)], "O3b": [(4096.0 * 5 + 10, 4096.0 * 5 + 20)]}
    assert SRC.plan(seg, 4096.0) == [(2, [(4096.0 * 2 + 100, 4096.0 * 3)]),
                                     (3, [(4096.0 * 3, 4096.0 * 3 + 50)]),
                                     (5, [(4096.0 * 5 + 10, 4096.0 * 5 + 20)])]


def test_pieces_are_written_atomically(tmp_path, monkeypatch):
    cache = SRC.StrainCache(tmp_path, [StubSource()], sample_rate=2048, log=lambda *a: None)
    piece = (SEG[0], SEG[0] + 64.0)
    cache.fill_chunk("H1", [piece])
    assert sorted(p.name for p in (tmp_path / "H1").iterdir()) == \
        [f"H1-{int(SEG[0])}-64.json", f"H1-{int(SEG[0])}-64.npy"]                     # no temporary files left
    other = (SEG[0] + 64.0, SEG[0] + 128.0)

    def killed(*a, **k):                                                              # the job dies mid-write
        raise KeyboardInterrupt
    monkeypatch.setattr(SRC.np, "save", killed)
    with pytest.raises(KeyboardInterrupt):
        cache.fill_chunk("H1", [other])
    monkeypatch.undo()
    assert not cache._path("H1", *other).with_suffix(".npy").exists()                 # not taken as cached
    assert cache.cached([piece, other], ["H1"]) == [piece]
    assert cache.fill_chunk("H1", [other]) == 1                                       # a rerun fills it
