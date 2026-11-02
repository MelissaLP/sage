"""
EuCAIF v3 examples: real detector noise from the v3 cache, with BBH signals (SAGE waveforms) and analytic glitches
injected at their SNR against each window's own PSD, whitened by a pluggable ``Whitener``.

Per example (all times from the GPS start of the context):

    context (context_seconds, e.g. 32 s of real strain from the cache; its PSD is estimated from it)
      |-- stored window (q_padding + frame_length + q_padding, e.g. 4 s, centred): what is returned
      |     |-- frame (e.g. 2 s): where mergers fall (tc_in_frame)
      |-- the BBH is generated on stored + 2 * signal_padding (e.g. 12 s), centred

1. a window is drawn from the split's segments (``segments.WindowSampler``) and read from the cache for every
   detector of the network (same GPS time);
2. its PSD per detector: median Welch (``TorchWelch``) over the context, or (``psd.exclude_centre``) over its two
   sides only, averaged, as ggwd FM production does, so a loud glitch in the stored window does not bias it;
3. ``signal.fraction`` of the examples (default half; 0: noise only) get a BBH: SAGE's waveform (default
   ``IMRPhenomPv2``; SAGE also has IMRPhenomD, XAS, XHM, XP and XPHM) from the v1 prior with tc in the frame,
   projected with the sidereal time of the window's real GPS time (``ConstantProjection.set_gps``), physical
   distances or rescaled to ``snr_range`` against this window's PSD;
4. glitches (``GlitchPopulation``, analytic shapes as strain, v2's populations; ``probability: 0`` or no
   ``glitches`` section: none) are injected in the stored window at their SNR against this window's PSD;
5. the sum is whitened with the configured ``Whitener`` (with this window's ASD) and cropped to the stored window;
   the injected BBH and glitches are whitened alone too (the whitening is linear), as targets.

Real data are not clean: the background may hold real glitches (``tags`` from Gravity Spy-like tables mark the
labelled ones; a detector without a table is unknown) and only the *injected* components have exact targets.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from eucaif.eucaif_v2 import _populations, _prior_with_tc, _register
from eucaif.v3 import segments as S
from eucaif.v3.sources import StrainCache
from eucaif.v3.whitening import make_whitener
from sage.data.waveform import ConstantProjection, gmst_from_gps, read_from_config
from sage.dsp.welch import TorchWelch

APPROXIMANTS = ("IMRPhenomPv2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM", "IMRPhenomXP", "IMRPhenomXPHM")


def _waveform_class(name):
    if name not in APPROXIMANTS:
        raise ValueError(f"approximant {name!r} not in SAGE's {APPROXIMANTS}")
    import importlib
    return getattr(importlib.import_module(f"sage.data.waveform.approximants.{name}"), name)


def _network_population(cfg, tmpdir):
    """The glitch population file restricted to the network: per-detector rates/weights of other detectors are
    dropped; a network detector with none (e.g. V1: Gravity Spy covers H1/L1 only) is an error."""
    path = cfg["_dir"] / cfg["glitches"]["population"]
    pop = yaml.safe_load(path.read_text())
    dets = list(cfg["data"]["detectors"])
    pop["detectors"] = dets
    for name, c in pop["classes"].items():
        for key in ("rate", "weight"):
            if isinstance(c.get(key), dict):
                missing = [d for d in dets if d not in c[key]]
                if missing and key == "rate" and "weight" not in c and float(sum(c[key].values())) > 0:
                    raise ValueError(f"{path.name}: class {name} has no rate for {missing}; add one (or a weight) "
                                     f"for the network {dets}")
                c[key] = {d: v for d, v in c[key].items() if d in dets}
    out = Path(tmpdir) / "population_network.yaml"
    out.write_text(yaml.safe_dump(pop, sort_keys=False))
    return out


def _v2_view(cfg):
    """The v3 config in the shape eucaif_v2's helpers expect (whitening settings under data)."""
    d = dict(cfg["data"], whitening_fduration=float(cfg["whitening"].get("fduration", 4.0)),
             whitening_highpass=float(cfg["whitening"].get("highpass", 15.0)))
    return dict(cfg, data=d)


class EucaifV3:
    """Batches of v3 examples from one split. ``cfg``: the v3 YAML as a dict, with ``_dir`` (its folder)."""

    def __init__(self, cfg, split="train", seed=0, batch_size=64, device="cpu", log=print):
        self.cfg, self.log = cfg, log
        d = cfg["data"]
        self.detectors = list(d["detectors"])
        self.fs = float(d["sample_rate"])
        self.context = float(d["context_seconds"])
        self.stored = float(d["frame_length"] + 2 * d["q_padding"])
        self.q_padding, self.frame_length = float(d["q_padding"]), float(d["frame_length"])
        self.stored_offset = (self.context - self.stored) / 2                     # stored window in the context
        self.L = int(round(self.stored * self.fs))
        self.N = int(round(self.context * self.fs))
        sig_len = self.stored + 2 * float(d["signal_padding"])
        self.signal_offset = (self.context - sig_len) / 2
        self.whitener = make_whitener(cfg["whitening"], self.fs, self.detectors)
        if self.whitener.pad / self.fs > self.stored_offset:
            raise ValueError("the whitening drops more than the context leaves around the stored window")
        if self.signal_offset < 0:
            raise ValueError("context_seconds is shorter than the BBH grid (stored + 2 * signal_padding)")
        self.seg_file = S.load(cfg["_dir"] / cfg["segments"]["file"])
        if list(self.seg_file["detectors"]) != self.detectors:
            raise ValueError(f"segments are for {self.seg_file['detectors']}, the config for {self.detectors}")
        self.split_segments = S.split_segments(self.seg_file["segments"], cfg["split"])[split]
        self.cache = StrainCache.from_config(cfg, log=log)
        if not self.cache.fetch_missing:                     # read only the local cache: draw where it has data
            cached = self.cache.cached(self.split_segments, self.detectors)
            log(f"{split}: {S.total(cached) / 3600:.1f} h of {S.total(self.split_segments) / 3600:.1f} h in the cache")
            self.split_segments = cached
        self.sampler = S.WindowSampler(self.split_segments, self.context)
        p = cfg.get("psd", {})
        self.welch = TorchWelch(delta_t=1 / self.fs, seg_len=int(round(p.get("segment_seconds", 4.0) * self.fs)),
                                seg_stride=int(round(p.get("segment_seconds", 4.0) * self.fs / 2)))
        self.exclude_centre = bool(p.get("exclude_centre", False))
        self.tags = S.GlitchTags([dict(t, path=cfg["_dir"] / t["path"]) for t in cfg.get("glitch_tags") or []])
        # SAGE waveforms read SAGE's global configs (as v2): one call gives batch_size / 2 signals
        v2 = _v2_view(cfg)
        self._tmp = tempfile.TemporaryDirectory()
        self.signal_fraction = float(cfg["signal"].get("fraction", 0.5))
        if not 0 <= self.signal_fraction <= 1:
            raise ValueError(f"signal.fraction must be in [0, 1], got {self.signal_fraction}")
        self.cols = {}
        if self.signal_fraction > 0:
            _register(v2, batch_size=batch_size, device=device)
            self.param_sampler = read_from_config(_prior_with_tc(v2, self._tmp.name), seed=seed)
            self.projection = ConstantProjection()
            self.signal_sampler = _waveform_class(cfg["signal"].get("approximant", "IMRPhenomPv2"))(
                self.param_sampler, self.projection)
            self.cols = self.param_sampler.param_index
        self.snr_range = cfg["signal"].get("snr_range")
        self.f_low_signal = float(d["signal_low_frequency_cutoff"])
        if cfg.get("glitches"):
            v2["glitches"] = dict(cfg["glitches"], population=str(_network_population(cfg, self._tmp.name)))
        self.glitches = _populations(v2) if cfg.get("glitches") else []
        self.glitch_f_low = (cfg.get("glitches") or {}).get("f_low")
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    # -- pieces ----------------------------------------------------------------------------------------

    def _psd(self, x):
        """(D, N) context -> (F,) frequencies, (D, F) PSD."""
        out = []
        for row in torch.as_tensor(x):
            if self.exclude_centre:
                a = int(round(self.stored_offset * self.fs))
                b = int(round((self.stored_offset + self.stored) * self.fs))
                f, p1 = self.welch(row[:a], return_freqs=True)
                _, p2 = self.welch(row[b:], return_freqs=True)
                out.append((p1 + p2) / 2)
            else:
                f, p = self.welch(row, return_freqs=True)
                out.append(p)
        return f, torch.stack(out)

    def _signals(self, gps_centres):
        """BBHs for len(gps_centres) examples: (n, D, N) strain on the context grid, and their parameters."""
        n = len(gps_centres)
        per_call = self.batch_size // 2
        hs, thetas = [], []
        for k in range(0, n, per_call):
            gps = np.asarray(gps_centres[k:k + per_call])
            pad = per_call - len(gps)
            self.projection.set_gps(np.concatenate([gps, np.full(pad, gps[-1])]))
            hf, _, theta = self.signal_sampler(return_theta=True)
            hs.append(hf[:len(gps)].cpu())
            thetas.append(theta[:len(gps)].cpu())
        hf = torch.cat(hs).to(torch.complex128)
        theta = torch.cat(thetas).double()
        sig_n = 2 * (hf.shape[-1] - 1)
        h = torch.fft.irfft(hf, n=sig_n, dim=-1, norm="forward")                    # physical strain, its grid
        out = torch.zeros(n, len(self.detectors), self.N, dtype=torch.float64)
        i0 = int(round(self.signal_offset * self.fs))
        out[..., i0:i0 + sig_n] = h
        return out, theta

    @staticmethod
    def _snr(h, f_psd, psd, fs, f_low):
        """Optimal SNR per detector of strain h (n, D, N) against PSD (n, D, F) given on f_psd."""
        n = h.shape[-1]
        f = np.fft.rfftfreq(n, 1 / fs)
        H = torch.fft.rfft(h, dim=-1) / fs
        p = torch.tensor(np.stack([[np.interp(f, f_psd, row) for row in ex] for ex in psd.numpy()]))
        m = torch.tensor(f >= f_low) & (p > 0)
        rho2 = 4 * (fs / n) * torch.where(m, H.abs() ** 2 / torch.where(p > 0, p, torch.ones_like(p)), 0).sum(-1)
        return rho2.sqrt()

    # -- a batch ---------------------------------------------------------------------------------------

    def batch(self, n=None):
        n = int(n or self.batch_size)
        starts, segs = self.sampler.sample(n, self.rng, return_segments=True)
        D = len(self.detectors)
        x = np.zeros((n, D, self.N))
        gps0, prov = np.zeros(n), []
        for i, (t, seg) in enumerate(zip(starts, segs)):
            row = []
            for j, det in enumerate(self.detectors):
                x[i, j], gps0[i], pv = self.cache.window(det, t, self.context, seg)
                row.append(pv[0].get("source", "?"))
            prov.append(row)
        x = torch.tensor(x)
        psd = []
        for i in range(n):
            f_psd, p = self._psd(x[i])
            psd.append(p)
        psd = torch.stack(psd)                                                        # (n, D, F)
        asd = psd.sqrt()

        # BBHs in signal.fraction of the examples
        y = torch.zeros(n, dtype=torch.long)
        y[self.rng.permutation(n)[: int(round(self.signal_fraction * n))]] = 1
        sig = torch.zeros_like(x)
        snr_det = torch.full((n, D), float("nan"), dtype=torch.float64)
        fplus, fcross, delay = (torch.full((n, D), float("nan"), dtype=torch.float64) for _ in range(3))
        theta_full = torch.full((n, len(self.cols)), float("nan"), dtype=torch.float64)
        idx = torch.nonzero(y).squeeze(-1)
        if idx.numel():
            centre = gps0[idx.numpy()] + self.context / 2
            h, theta = self._signals(centre)
            rho = self._snr(h, f_psd.numpy(), psd[idx], self.fs, self.f_low_signal)
            if self.snr_range is not None:
                target = torch.tensor(self.rng.uniform(*self.snr_range, size=idx.numel()))
                scale = target / rho.norm(dim=-1)
                h = h * scale[:, None, None]
                theta[:, self.cols["distance"]] = theta[:, self.cols["distance"]] / scale
                rho = rho * scale[:, None]
            sig[idx] = h
            snr_det[idx] = rho
            theta_full[idx] = theta
            # the projection's antenna factors and delays (same GPS time and sky position as the signals)
            pr = self.projection
            gmst = gmst_from_gps(centre, pr.device, pr.dtype)
            ra, dec, pol = (theta[:, self.cols[k]].to(device=pr.device, dtype=pr.dtype)
                            for k in ("ra", "dec", "polarization"))
            fp, fc = pr.antenna_pattern(ra, dec, pol, gmst)
            fplus[idx], fcross[idx] = fp.cpu().double(), fc.cpu().double()
            delay[idx] = pr.time_delay_from_earth_center(ra, dec, gmst).cpu().double()

        # glitches in the stored window, at their SNR against this window's PSD (analytic shapes as strain)
        glitch = torch.zeros_like(x)
        events = [[] for _ in range(n)]
        window = (self.stored_offset, self.stored_offset + self.stored)
        frame = (self.stored_offset + self.q_padding, self.stored_offset + self.q_padding + self.frame_length)
        g = torch.Generator().manual_seed(int(self.rng.integers(2 ** 31)))
        for pop in self.glitches:
            asd_grid = asd.reshape(n * D, -1)
            add, ev = pop.inject(torch.zeros_like(x), asd_grid.reshape(n, D, -1), g, f_low=self.glitch_f_low,
                                 window=window, frame=frame)
            glitch = glitch + add
            for s, evs in enumerate(ev):                                             # s: example * D + detector
                for e in evs:
                    events[s // D].append(dict(e, detector=self.detectors[s % D], glitch=None,
                                               gps=float(gps0[s // D] + e["time"])))

        # whiten the sum and the injected components with this window's ASD; crop the stored window
        a = int(round(self.stored_offset * self.fs)) - self.whitener.pad
        crop = slice(a, a + self.L)
        out = {k: self.whitener(v, asd)[..., crop] for k, v in (("x", x + sig + glitch), ("signal", sig),
                                                                ("glitch", glitch))}
        gps_stored = gps0 + self.stored_offset
        tags = [[self.tags.in_window(det, gs, gs + self.stored) for det in self.detectors] for gs in gps_stored]
        return dict(out, y=y, snr=snr_det.norm(dim=-1), snr_det=snr_det, fplus=fplus, fcross=fcross, delay=delay,
                    tc=theta_full[:, self.cols["tc"]] if "tc" in self.cols else None,
                    params={k: theta_full[:, i] for k, i in self.cols.items()},
                    gps=torch.tensor(gps_stored), sources=prov, glitch_events=events, tags=tags,
                    asd_frequencies=f_psd, asd=asd.float())


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_dir"] = path.resolve().parent
    cfg["_text"] = path.read_text()
    return cfg
