"""
Glitch rates, frequency ranges and SNR distributions from the public Gravity Spy O3a tables, for the
analytic glitch classes of ``sage_analytic_glitches.yaml``.

Method taken from GWForge (Koustav Chandra et al., https://github.com/koustavchandra/gwforge, MIT;
``docs/source/glitch.md`` and ``GWForge/glitch/configuration_files/blip_tomte.ini``), who motivated
their Blip and Tomte numbers on the same tables; this script reproduces them (Blip 1.30/h at H1,
0.65/h at L1; Tomte 0.22/h at H1, 6.3/h at L1) and applies the same recipe to the other classes:

- Gravity Spy machine-learning classifications of O3a Omicron triggers (Zenodo 5649212,
  doi:10.5281/zenodo.5649212, CC BY 4.0; Glanzer et al. 2023, Class. Quantum Grav. 40 065004,
  doi:10.1088/1361-6382/acb633), kept when ``ml_confidence >= 0.9``;
- rate = number of triggers / observing time, the observing time being the GWOSC O3a ``DATA``
  segments of each detector (public strain available: 3116 h at H1, 3321 h at L1);
- peak frequency and duration: 5 / 50 / 95 % quantiles of Omicron's ``peak_frequency`` and
  ``duration``;
- SNR index: maximum-likelihood power law p(snr) ~ snr**-alpha above Omicron's 7.5 threshold,
  alpha = 1 + n / sum(log(snr / 7.5)). These are Omicron SNRs; like GWForge we use the index for
  the optimal SNR on a wider range.

Usage, from the SAGE repo root (downloads ~230 MB once into --data-dir):

    python3 eucaif/gravityspy_o3a_rates.py --data-dir gravityspy_o3a [--yaml o3a_classes.yaml]

It prints the table for every Gravity Spy class and, with --yaml, writes the numbers for the classes
that have an analytic model in SAGE (Blip, Tomte, Blip_Low_Frequency, Koi_Fish with Antiglitch's model, Scattered_Light
as an ArchEnemy arch).
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

ZENODO = "https://zenodo.org/records/5649212/files/{ifo}_O3a.csv?download=1"
GWOSC = "https://gwosc.org/timeline/segments/json/O3a_4KHZ_R1/{ifo}_DATA/1238166018/15811200/"
DETECTORS = ("H1", "L1")
CONFIDENCE = 0.9
SNR_THRESHOLD = 7.5
COLUMNS = ["snr", "peak_frequency", "duration", "ml_label", "ml_confidence"]
# Gravity Spy class -> SAGE class name (in sage_analytic_glitches.yaml)
MODELLED = {"Blip": "Blip", "Tomte": "Tomte", "Blip_Low_Frequency": "BlipLowFrequency",
            "Koi_Fish": "KoiFish", "Scattered_Light": "ScatteredLight"}


def fetch(url, path):
    if not path.exists():
        print(f"downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def observing_hours(data_dir, ifo):
    seg = json.loads(fetch(GWOSC.format(ifo=ifo), data_dir / f"{ifo}_O3a_DATA_segments.json").read_text())
    return sum(b - a for a, b in seg["segments"]) / 3600.0


def snr_index(snr):
    snr = snr[snr >= SNR_THRESHOLD]
    return 1.0 + len(snr) / np.log(snr / SNR_THRESHOLD).sum() if len(snr) > 1 else float("nan")


def class_table(data_dir):
    import pandas as pd
    rows = {}
    for ifo in DETECTORS:
        hours = observing_hours(data_dir, ifo)
        df = pd.read_csv(fetch(ZENODO.format(ifo=ifo), data_dir / f"{ifo}_O3a.csv"), usecols=COLUMNS)
        df = df[df.ml_confidence >= CONFIDENCE]
        for label, g in df.groupby("ml_label"):
            r = rows.setdefault(label, {"snr": []})
            r[ifo] = dict(n=len(g), rate=len(g) / hours, hours=hours,
                          f=np.percentile(g.peak_frequency, [5, 50, 95]),
                          duration=np.percentile(g.duration, [5, 50, 95]), snr_index=snr_index(g.snr.values))
            r["snr"].append(g.snr.values)
    for r in rows.values():
        r["snr_index"] = snr_index(np.concatenate(r.pop("snr")))      # both detectors pooled
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="gravityspy_o3a")
    p.add_argument("--yaml", help="write the numbers of the modelled classes to this file")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = class_table(data_dir)

    print(f"Gravity Spy O3a, ml_confidence >= {CONFIDENCE}; rates per hour of GWOSC DATA time")
    print(f"{'class':22s} {'rate H1':>8s} {'rate L1':>8s}  {'f_peak 5-95% H1 (median)':>27s}  "
          f"{'f_peak 5-95% L1 (median)':>27s}  {'SNR index':>9s}  {'duration 5-95% [s]':>18s}")
    total = lambda r: sum(r.get(d, {}).get("rate", 0) for d in DETECTORS)
    for label, r in sorted(rows.items(), key=lambda kv: -total(kv[1])):
        cells = []
        for d in DETECTORS:
            x = r.get(d)
            cells.append(f"{x['f'][0]:6.0f}-{x['f'][2]:<6.0f} ({x['f'][1]:4.0f})" if x else "-")
        dur = np.array([r[d]["duration"] for d in DETECTORS if d in r])
        print(f"{label:22s} {r.get('H1', {}).get('rate', 0):8.3f} {r.get('L1', {}).get('rate', 0):8.3f}  "
              f"{cells[0]:>27s}  {cells[1]:>27s}  {r['snr_index']:9.2f}  "
              f"{dur[:, 0].min():6.2f}-{dur[:, 2].max():<6.2f}")

    if args.yaml:
        import yaml
        out = {}
        for gs, name in MODELLED.items():
            r = rows[gs]
            f_lo = min(r[d]["f"][0] for d in DETECTORS if d in r)
            f_hi = max(r[d]["f"][2] for d in DETECTORS if d in r)
            dur = [min(r[d]["duration"][0] for d in DETECTORS if d in r),
                   max(r[d]["duration"][2] for d in DETECTORS if d in r)]
            out[name] = dict(gravityspy_class=gs,
                             rate={d: round(float(r[d]["rate"]), 3) for d in DETECTORS if d in r},
                             peak_frequency_5_95=[round(float(f_lo), 1), round(float(f_hi), 1)],
                             duration_5_95=[round(float(v), 2) for v in dur],
                             snr_index=round(float(r["snr_index"]), 2),
                             count={d: int(r[d]["n"]) for d in DETECTORS if d in r})
        Path(args.yaml).write_text(yaml.safe_dump(out, sort_keys=False))
        print("wrote", args.yaml)


if __name__ == "__main__":
    main()
