"""
Generate whitened positive (noise + signal) and negative (noise only)
examples for H1/L1, following the setup of ``testing_sage.ipynb``, and
optionally write them to an HDF5 dataset (see ``README_dataset.md``).

Pipeline
--------
1. Noise: ``sample_synthetic_noise`` in the time domain, one realisation per
   example, over the same window the waveform generator uses
   (``padded_length_in_s``).
2. Signals: ``IMRPhenomPv2`` returns the projected strain in the FREQUENCY
   domain, in sage's ``rfft(norm="forward")`` convention (LAL's h(f) times
   df). It is brought to the time domain with
   ``irfft(hf, n=N, norm="forward")`` before being added to the noise.
3. Signal amplitude: by default the physical amplitude implied by the
   sampled luminosity distance is kept (as ``IMRPhenomPv2`` does without an
   ``augment``). Optionally (``snr_range``) each signal is instead rescaled to
   a network optimal SNR drawn uniformly from that range, and its distance is
   updated to ``distance / scale`` so amplitude and distance stay consistent
   -- the same convention as sage's ``OptimalSNRRescaler`` augment. The
   optimal SNR (against the ASD used to colour the noise) is always stored.
4. Whitening: ``FIRWhitening`` (gwpy-equivalent) with the known ASD, cached
   once, over the whole padded window (default 4 s filter, as ggwd's
   ``whitening_max_filter_duration``). The whitened series is then cropped
   to the ``sample_length_in_s`` window, which must lie at least
   ``fduration / 2`` from either end (the filter's corrupted edges).

The padding should also be long enough that the whole signal fits in the
generated window: the waveform is periodic in it, so an inspiral longer than
``tc + padding_length_in_s`` wraps around to the end. (10+10 Msun from 20 Hz
lasts ~6 s, hence the 4 s padding around a 2 s sample below.)

Timing convention: ``IMRPhenomPv2`` puts the merger at
``tc + padding_length_in_s`` inside the padded window, i.e. ``tc`` is the
merger time measured from the start of the (unpadded) sample. So the config
must satisfy ``padded_length_in_s == sample_length_in_s + 2 *
padding_length_in_s`` and the ``tc`` prior must lie in
``[0, sample_length_in_s]``.

Usage from the notebook, after ``register_configs(...)``::

    from make_whitened_examples import make_whitened_examples, plot_examples
    data = make_whitened_examples("./sage_eucaif_waveform.yaml", n_per_class=50)
    plot_examples(data)

Writing a dataset from the command line (registers the notebook's configs)::

    python make_whitened_examples.py sage_eucaif_waveform.yaml \\
        --n-per-class 25000 --out eucaif_whitened.h5 [--store-dtype float64]
"""
import sys
sys.path.insert(0, '/data/gravwav/lopezm/Projects/EuCAIF/sage/')
import datetime
import subprocess
from pathlib import Path

import numpy as np
import torch

from sage.core.config import get_cfg, get_data_cfg
from sage.data.noise import sample_synthetic_noise
from sage.data.waveform import ConstantProjection, IMRPhenomPv2, read_from_config
from sage.dsp.whiten import FIRWhitening

METADATA_FIELDS = ["tc", "mchirp", "ra", "dec", "class"]


class _RecordingProjection(ConstantProjection):
    """``ConstantProjection`` that remembers the random GMST it drew.

    The projection picks a uniformly random Greenwich Mean Sidereal Time per
    signal (instead of deriving it from a GPS time) using torch's global RNG.
    The detector response depends on ``ra`` only through the hour angle
    ``gmst - ra``, so the GMST is needed to interpret ``ra``.
    """

    def random_gmst_estimate(self, B=None):
        gmst = super().random_gmst_estimate(B)
        self.last_gmst = gmst.detach()
        return gmst


def optimal_snr(hf, asd, df, f_low):
    """
    Optimal SNR per detector and network SNR.

    ``hf`` is in sage's ``rfft(norm="forward")`` convention, i.e. LAL's h(f)
    times df, so ``rho^2 = 4 df sum |h(f)|^2 / S = (4 / df) sum |hf|^2 / S``.

    Parameters
    ----------
    hf : (B, D, F) complex
    asd : (D, F) real, on the same grid as ``hf``
    df : float
    f_low : float
        Bins below this frequency (and bins with ASD <= 0) are excluded.

    Returns
    -------
    per_det : (B, D), network : (B,)
    """
    freqs = torch.arange(hf.shape[-1], dtype=torch.float64, device=hf.device) * df
    asd = asd.to(device=hf.device, dtype=torch.float64)
    valid = (asd > 0) & (freqs >= f_low)
    inv_psd = torch.where(valid, 1.0 / asd.clamp_min(1e-300) ** 2, torch.zeros_like(asd))
    rho2 = (4.0 / df) * (hf.abs().to(torch.float64) ** 2 * inv_psd).sum(-1)
    return rho2.sqrt(), rho2.sum(-1).sqrt()


class WhitenedExampleGenerator:
    """
    Set up the waveform sampler and whitener once, then produce any number
    of chunks of whitened examples with :meth:`generate`.

    Parameters
    ----------
    waveform_yaml : str
        Prior config for ``read_from_config`` (e.g. ``sage_eucaif_waveform.yaml``).
    asd_names : sequence of str
        One analytic ASD per detector (see ``sage.data.noise.available_asds``).
    snr_range : (float, float) or None
        ``None`` (default) keeps the physical amplitudes implied by the
        sampled distances. A range rescales each signal to a network optimal
        SNR drawn uniformly from it, and updates the stored distance to
        ``distance / scale``.
    fduration : float
        FIR whitening filter length in seconds (default 4, as ggwd's
        ``whitening_max_filter_duration``). Must satisfy
        ``fduration / 2 <= padding_length_in_s``; the whitened output is
        cropped to the ``sample_length_in_s`` window.
    highpass : float or None
        Whitening highpass (Hz). Defaults to ``noise_low_frequency_cutoff``.
    dtype : torch.float32 or torch.float64
        Whitening precision.
    seed : int
        Seeds the waveform prior sampler (chunks then continue its stream).
    merger_margin : float
        Every merger (peak of |h(t)| in each detector) must lie at least this
        many seconds inside the whitened sample window, else ValueError.
    """

    def __init__(
        self,
        waveform_yaml,
        asd_names=("aLIGOZeroDetHighPower", "aLIGOZeroDetHighPower"),
        snr_range=None,
        fduration=4.0,
        highpass=None,
        dtype=torch.float32,
        seed=150914,
        merger_margin=0.25,
    ):
        data_cfg = get_data_cfg()
        self.waveform_yaml = str(waveform_yaml)
        self.asd_names = list(asd_names)
        self.detectors = list(get_cfg().detectors)
        self.D = len(self.asd_names)
        self.fs = float(data_cfg.sample_rate)
        self.snr_range = snr_range
        self.highpass = (
            highpass if highpass is not None
            else getattr(data_cfg, "noise_low_frequency_cutoff", None)
        )
        self.f_low_signal = float(getattr(data_cfg, "signal_low_frequency_cutoff", 20.0))
        self.sample_length_s = float(data_cfg.sample_length_in_s)
        self.padding_s = float(data_cfg.padding_length_in_s)
        expected_window = self.sample_length_s + 2 * self.padding_s
        if abs(float(data_cfg.padded_length_in_s) - expected_window) > 1e-9:
            raise ValueError(
                f"Inconsistent data config: padded_length_in_s="
                f"{data_cfg.padded_length_in_s} but sample_length_in_s + 2 * "
                f"padding_length_in_s = {expected_window}. IMRPhenomPv2 puts the "
                f"merger at tc + padding_length_in_s, so with this config mergers "
                f"land outside the window and wrap around."
            )
        self.fduration = float(fduration)
        if self.fduration / 2 > self.padding_s + 1e-9:
            raise ValueError(
                f"fduration / 2 = {self.fduration / 2} s exceeds padding_length_in_s = "
                f"{self.padding_s} s: the whitening-corrupted edges would reach into "
                f"the sample window."
            )
        self.dtype = dtype
        self.seed = int(seed)
        self.merger_margin = float(merger_margin)

        # -- waveform generator (FD, on its own padded grid) --
        self.param_sampler = read_from_config(self.waveform_yaml, seed=self.seed)
        self.projection = _RecordingProjection()
        self.signal_sampler = IMRPhenomPv2(self.param_sampler, self.projection)
        pidx = self.param_sampler.param_index
        self._cols = {k: pidx[k] for k in ("tc", "mchirp", "ra", "dec", "distance")}

        self.window_s = float(self.signal_sampler.sample_length_in_s)  # padded_length_in_s
        self.N = int(round(self.window_s * self.fs))
        self.F = self.N // 2 + 1
        self.df = 1.0 / self.window_s

        # -- ASD and whitener: take the ASD from a tiny noise draw, design once --
        asd = sample_synthetic_noise(
            self.window_s, self.asd_names, detectors=self.D, batch=1,
            sample_rate=self.fs, seed=0, is_asd=True,
        )[1]
        self.asd = torch.as_tensor(asd, dtype=torch.float64)
        self.asd_grid = (
            FIRWhitening.resample_asd(self.asd, self.F)
            if self.asd.shape[-1] != self.F else self.asd
        )
        self.whitener = FIRWhitening(
            sample_rate=self.fs, fduration=self.fduration, highpass=self.highpass,
            asd=self.asd, seq_len=self.N, dtype=self.dtype,
        )
        # The whitened series starts at sample `pad` of the padded window;
        # keep only the sample window [padding, padding + sample_length).
        pad = self.whitener.pad
        start = int(round(self.padding_s * self.fs)) - pad
        self.L = int(round(self.sample_length_s * self.fs))
        self._crop = slice(start, start + self.L)
        # time from the start of the sample window (the reference tc uses)
        self.t = torch.arange(self.L, dtype=torch.float64) / self.fs

    def _draw_signals(self, n):
        """IMRPhenomPv2 returns batch_size * class_balance signals per call."""
        hfs, targets, thetas, gmsts = [], [], [], []
        while sum(h.shape[0] for h in hfs) < n:
            hf, tg, th = self.signal_sampler(return_theta=True)
            hfs.append(hf.cpu())
            targets.append(tg.cpu())
            thetas.append(th.cpu())
            gmsts.append(self.projection.last_gmst.cpu())
        return (torch.cat(hfs)[:n], torch.cat(targets)[:n], torch.cat(thetas)[:n],
                torch.cat(gmsts)[:n])

    def generate(self, n_per_class, noise_seed, keep_signal=True):
        """
        One chunk: ``n_per_class`` negatives followed by ``n_per_class``
        positives (not shuffled).

        Returns
        -------
        dict with
            x          : (2n, D, L) whitened strain, negatives first
            y          : (2n,) labels, 0 = noise, 1 = signal + noise
            metadata   : dict of (2n,) float64 tensors, keys METADATA_FIELDS
                         plus "gmst" and "distance" (physical units; NaN for
                         negatives except "class")
            targets    : (n, 5) standardised targets as returned by the
                         sampler (positives only), same order as metadata
            snr, snr_det : (n,), (n, D) optimal SNRs of the injected signals
            tc, t_merger : (n,), (n, D) sampled tc / measured peak time, s
            snr_from_whitened : (n,) sqrt(sum w^2) of the whitened signal
            signal_whitened   : (n, D, L), only if ``keep_signal``
            t          : (L,) time axis, s from the start of the sample window
        """
        n = int(n_per_class)
        # ConstantProjection draws its random GMST from torch's global RNG
        torch.manual_seed(int(noise_seed))
        hf, targets, theta, gmst = self._draw_signals(n)
        tc = theta[:, self._cols["tc"]].double()
        distance = theta[:, self._cols["distance"]].double()

        # -- noise (TD), one independent realisation per example --
        noise = sample_synthetic_noise(
            self.window_s, self.asd_names, detectors=self.D, batch=2 * n,
            sample_rate=self.fs, seed=int(noise_seed), is_asd=True,
        )[0]
        noise = torch.as_tensor(noise, dtype=torch.float64)
        if noise.shape[-1] != self.N:
            raise ValueError(f"noise has {noise.shape[-1]} samples, expected {self.N}")

        # -- optional SNR rescaling --
        snr_det, snr = optimal_snr(hf, self.asd_grid, self.df, self.f_low_signal)
        if self.snr_range is not None:
            gen = torch.Generator().manual_seed(int(noise_seed) + 1)
            target_snr = torch.empty(n, dtype=torch.float64).uniform_(
                *self.snr_range, generator=gen)
            scale = target_snr / snr
            hf = hf * scale.to(torch.float32)[:, None, None]
            # strain ~ 1 / distance (same convention as OptimalSNRRescaler)
            distance = distance / scale
            snr_det, snr = optimal_snr(hf, self.asd_grid, self.df, self.f_low_signal)

        h_td = torch.fft.irfft(hf.to(torch.complex128), n=self.N, dim=-1, norm="forward")

        # -- check every merger lands inside the whitened sample window --
        t_merger = h_td.abs().argmax(dim=-1).double() / self.fs - self.padding_s  # (n, D)
        lo, hi = self.merger_margin, self.sample_length_s - self.merger_margin
        outside = (t_merger < lo) | (t_merger > hi)
        if outside.any():
            bad = torch.nonzero(outside.any(-1)).flatten().tolist()
            raise ValueError(
                f"{len(bad)} merger(s) outside [{lo:.2f}, {hi:.2f}] s of the sample "
                f"window, e.g. tc={tc[bad[0]].item():.3f} s -> measured "
                f"{t_merger[bad[0]].tolist()} s. Keep the tc prior inside "
                f"[{lo:.2f}, {hi:.2f}] (tc is measured from the start of the sample)."
            )

        # -- assemble and whiten: negatives first, then positives --
        x = noise
        x[n:] += h_td
        x_w = self.whitener(x)[..., self._crop]
        y = torch.cat([torch.zeros(n), torch.ones(n)]).long()

        nan = torch.full((n,), float("nan"), dtype=torch.float64)
        metadata = {
            k: torch.cat([nan, theta[:, self._cols[k]].double()])
            for k in ("tc", "mchirp", "ra", "dec")
        }
        metadata["gmst"] = torch.cat([nan, gmst.double()])
        metadata["distance"] = torch.cat([nan, distance])
        metadata["class"] = y.double()

        out = {
            "x": x_w, "y": y, "metadata": metadata, "targets": targets,
            "snr": snr.float(), "snr_det": snr_det.float(),
            "tc": tc.float(), "t_merger": t_merger.float(), "t": self.t,
        }
        if keep_signal:
            sig_w = self.whitener(h_td)[..., self._crop]
            out["signal_whitened"] = sig_w
            # whitened noise is ~unit variance, so sqrt(sum w^2) of the
            # whitened signal alone recovers the optimal SNR
            out["snr_from_whitened"] = sig_w.double().pow(2).sum(-1).sum(-1).sqrt().float()
        return out

    def attrs(self):
        """Dataset-level description, stored as HDF5 attributes."""
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()
        except Exception:
            commit = ""
        return {
            "detectors": np.array(self.detectors[: self.D], dtype="S"),
            "asd_names": np.array(self.asd_names, dtype="S"),
            "sample_rate": self.fs,
            "sample_length_in_s": self.sample_length_s,
            "padding_length_in_s": self.padding_s,
            "generated_window_in_s": self.window_s,
            "n_samples": self.L,
            "t_start": float(self.t[0]),
            "whitening": "FIRWhitening (gwpy 3.0.14 TimeSeries.whiten equivalent)",
            "whitening_fduration": self.fduration,
            "whitening_highpass": float(self.highpass) if self.highpass is not None else -1.0,
            "whitening_window": "hann",
            "whitening_dtype": str(self.dtype),
            "approximant": "IMRPhenomPv2",
            "projection": "ConstantProjection (uniformly random GMST per signal, stored in metadata/gmst)",
            "signal_low_frequency_cutoff": self.f_low_signal,
            "snr_rescaling": (
                f"uniform network optimal SNR in {list(self.snr_range)}; distance = distance / scale"
                if self.snr_range is not None else "none (physical distances)"
            ),
            "snr_definition": "network optimal SNR against the colouring ASD",
            "tc_reference": "seconds from the start of the stored window",
            "waveform_prior_yaml": Path(self.waveform_yaml).read_text(),
            "seed": self.seed,
            "sage_commit": commit,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }


def make_whitened_examples(waveform_yaml, n_per_class=50, seed=150914, **kwargs):
    """Single in-memory chunk (for notebooks). See ``WhitenedExampleGenerator``."""
    gen = WhitenedExampleGenerator(waveform_yaml, seed=seed, **kwargs)
    data = gen.generate(n_per_class, noise_seed=seed)
    data.update(sample_rate=gen.fs, fduration=gen.fduration)
    return data


def write_dataset(
    generator, out_path, n_per_class, chunk_per_class=1000,
    store_dtype="float32", shuffle=True,
):
    """
    Write ``2 * n_per_class`` whitened examples to an HDF5 file, chunk by
    chunk (memory stays bounded by ``chunk_per_class``). See
    ``README_dataset.md`` for the layout.
    """
    import h5py

    n_total = 2 * n_per_class
    D, L = generator.D, generator.L
    rng = np.random.default_rng(generator.seed)
    with h5py.File(out_path, "w") as f:
        dx = f.create_dataset(
            "x", shape=(n_total, D, L), dtype=store_dtype,
            chunks=(min(64, n_total), D, L),
        )
        dmeta = {
            k: f.create_dataset(f"metadata/{k}", shape=(n_total,),
                                dtype="int8" if k == "class" else "float64")
            for k in METADATA_FIELDS + ["gmst", "distance"]
        }
        dsnr = f.create_dataset("metadata/snr", shape=(n_total,), dtype="float32")
        dsnr_det = f.create_dataset("metadata/snr_det", shape=(n_total, D), dtype="float32")
        f.create_dataset("t", data=generator.t.numpy())
        for k, v in generator.attrs().items():
            f.attrs[k] = v

        written, chunk_idx = 0, 0
        while written < n_total:
            n = min(chunk_per_class, (n_total - written) // 2)
            data = generator.generate(n, noise_seed=generator.seed + 1000 + chunk_idx,
                                      keep_signal=False)
            order = rng.permutation(2 * n) if shuffle else np.arange(2 * n)
            sl = slice(written, written + 2 * n)
            dx[sl] = data["x"].numpy().astype(store_dtype)[order]
            for k in METADATA_FIELDS + ["gmst", "distance"]:
                dmeta[k][sl] = data["metadata"][k].numpy()[order]
            dsnr[sl] = np.concatenate(
                [np.full(n, np.nan, np.float32), data["snr"].numpy()])[order]
            dsnr_det[sl] = np.concatenate(
                [np.full((n, D), np.nan, np.float32), data["snr_det"].numpy()])[order]
            written += 2 * n
            chunk_idx += 1
            print(f"  {written}/{n_total} examples written")
    return out_path


def plot_examples(data, n_show=2, detector_names=("H1", "L1")):
    """Plot ``n_show`` negatives and ``n_show`` positives, both detectors."""
    import matplotlib.pyplot as plt

    x, y, t = data["x"], data["y"], data["t"].numpy()
    neg = torch.nonzero(y == 0).flatten()[:n_show]
    pos = torch.nonzero(y == 1).flatten()[:n_show]
    n_neg = int((y == 0).sum())
    D = x.shape[1]

    fig, axes = plt.subplots(2 * n_show, D, figsize=(6 * D, 2.2 * 2 * n_show),
                             sharex=True, squeeze=False)
    for row, idx in enumerate(list(neg) + list(pos)):
        idx = int(idx)
        for d in range(D):
            ax = axes[row, d]
            ax.plot(t, x[idx, d].float().numpy(), lw=0.5, color="0.55", label="whitened data")
            if y[idx] == 1:
                j = idx - n_neg
                if "signal_whitened" in data:
                    ax.plot(t, data["signal_whitened"][j, d].numpy(), lw=0.9, color="crimson",
                            label=f"whitened signal (SNR {data['snr_det'][j, d]:.1f})")
                ax.axvline(data["tc"][j].item(), color="k", ls="--", lw=0.8)
                title = (f"positive #{j}  network SNR {data['snr'][j]:.1f}, "
                         f"tc {data['tc'][j]:.2f} s")
            else:
                title = f"negative #{idx}"
            ax.set_title(f"{detector_names[d]} - {title}", fontsize=9)
            ax.legend(loc="upper right", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time in sample window [s] (same reference as tc)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    from sage.core.config import register_configs

    p = argparse.ArgumentParser(description="Generate whitened H1/L1 examples.")
    p.add_argument("waveform_yaml")
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--chunk-per-class", type=int, default=1000,
                   help="examples per class generated at once (bounds memory)")
    p.add_argument("--out", default=None,
                   help="HDF5 output path; if omitted, only a check + plot is run")
    p.add_argument("--store-dtype", default="float32", choices=["float32", "float64"],
                   help="dtype of x on disk; float64 also whitens in float64")
    p.add_argument("--snr-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="rescale signals to a network optimal SNR uniform in [MIN, MAX] "
                        "(distance updated accordingly); default: physical distances")
    p.add_argument("--seed", type=int, default=150914)
    p.add_argument("--fig", default="whitened_examples.png")
    args = p.parse_args()

    # Same minimal configs as testing_sage.ipynb, adapted to a 2 s sample
    # (as ggwd's seconds_before_event + seconds_after_event): 4 s padding on
    # each side (padded window 2 + 2 * 4 = 10 s) so the longest (10+10 Msun,
    # ~6 s from 20 Hz) signals fit without wrapping into the sample.
    class DummyCFG:
        export_dir = "."
        batch_size = 10
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        detectors = ["H1", "L1"]
        class_balance = 0.5
        do_point_estimate = ["tc", "mchirp", "ra", "dec"]

    class DummyDataCFG:
        sample_rate = 2048.0
        noise_low_frequency_cutoff = 15.0
        signal_low_frequency_cutoff = 20.0
        sample_length_in_s = 2.0
        padded_length_in_s = 10.0
        padding_length_in_s = 4.0
        delta_f = 1.0 / sample_length_in_s
        corrupted_length = 2.0

    register_configs(DummyCFG(), DummyDataCFG())

    gen = WhitenedExampleGenerator(
        args.waveform_yaml, snr_range=args.snr_range, seed=args.seed,
        dtype=torch.float64 if args.store_dtype == "float64" else torch.float32,
    )

    # quick check + plot on a small chunk
    data = gen.generate(min(args.n_per_class, 50), noise_seed=args.seed)
    neg_std = data["x"][data["y"] == 0].std().item()
    dt = data["t_merger"] - data["tc"][:, None]
    print(f"whitened window: [{gen.t[0]:.2f}, {gen.t[-1]:.2f}] s, "
          f"{gen.L} samples x {gen.D} detectors")
    print(f"whitened negatives std = {neg_std:.3f} (expect ~1)")
    print(f"tc {data['tc'].min():.2f}..{data['tc'].max():.2f} s; merger - tc: "
          f"{dt.min() * 1e3:.0f}..{dt.max() * 1e3:.0f} ms")
    q = torch.quantile(data["snr"], torch.tensor([0.1, 0.5, 0.9]))
    print(f"network optimal SNR: min {data['snr'].min():.1f}, 10/50/90% "
          f"{q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}, max {data['snr'].max():.1f}; "
          f"fraction >= 8: {(data['snr'] >= 8).float().mean():.2f}")
    print(f"SNR recovered from whitened signal / optimal = "
          f"{(data['snr_from_whitened'] / data['snr']).mean():.3f} (expect ~1)")
    plot_examples(data).savefig(args.fig, dpi=120)
    print(f"saved {args.fig}")

    if args.out:
        bytes_per = gen.D * gen.L * np.dtype(args.store_dtype).itemsize
        print(f"writing {2 * args.n_per_class} examples to {args.out} "
              f"(~{2 * args.n_per_class * bytes_per / 1e9:.2f} GB)")
        write_dataset(gen, args.out, args.n_per_class,
                      chunk_per_class=args.chunk_per_class, store_dtype=args.store_dtype)
        print(f"done: {args.out}")
