# station-archive

Build one reconciled daily station-observation archive out of several
overlapping public climate sources — ECA&D, GHCN-Daily, NOAA GSOD and the
Copernicus C3S land collection — so that every accepted value has a single
physical-station identity, one chosen source, a recorded reason for that
choice, and a full trail back to the provider record it came from.

Concatenating these sources naively does not work. They redeliver each
other, disagree about where a station is, disagree about what a reported
zero means, and carry undocumented sentinels. This package treats all of
that as the problem to be solved rather than noise to be averaged away.

## What it does

| Stage | Module | What it decides |
|---|---|---|
| Discovery | `inventory.py`, `sources/*` | What each source offers in a domain and period, **before** anything is bulk-downloaded |
| Acquisition | `bronze.py`, `net.py`, `manifests.py` | Restartable, checksummed, idempotent downloads; provider bytes stored untouched |
| Identity | `identity.py`, `review.py` | Which source records are the same physical station |
| Standardisation | `standardize.py` | Units, and documented day-window corrections |
| Quality control | `qc.py` | Which values are not observations |
| Consolidation | `consolidate.py` | Which value wins per station/date/variable, and why |
| Homogeneity | `homogeneity.py` | Which series contain artificial breakpoints |
| Read-back | `archive_ingest.py`, `fold_manifest.py` | Serving the archive to a model, and leakage-safe spatial folds |

### Some rules worth knowing about

These are the decisions that shaped the code; several were forced by
measurements on the real archive rather than chosen up front.

- **Identity is earned, not assumed.** Two records merge only on an exact
  shared identifier, or on *all* of: within 3 km, elevation within 50 m,
  daily correlation > 0.995, and mean absolute difference ≤ 0.5 °C over
  ≥ 365 shared days. Correlation alone is never enough — stations 2.4 km
  apart were measured correlating at 0.9915 while differing 1.3 °C every
  single day.
- **Zero overlap means never merge.** Two records sharing no observation day
  are permanently unmergeable regardless of how well their metadata agrees,
  with no manual override.
- **A reported zero is not always a measured zero.** GSOD's own readme says
  a `0` "includes a trace", and flag `I` means the station reported nothing
  at all. Those are rejected as non-observations rather than read as dry
  days.
- **Gap filling is audited, not assumed.** A lower-priority source may fill
  only days the primary lacks, and only after passing bias, dispersion,
  wet-day-frequency and monthly-discontinuity checks. A failed check leaves
  the gap open. Nothing is interpolated and no two sources are ever averaged.
- **Nothing is silently homogenised.** `homogeneity.py` detects breakpoints
  and flags them; it never alters a value.

## Install

```bash
pip install -r requirements.txt
export STATION_ARCHIVE_ROOT=/path/with/several/GB/free   # not your checkout
```

`cdsapi` is only needed for the Copernicus C3S source; it reads
`~/.cdsapirc` or `CDSAPI_URL` / `CDSAPI_KEY`. No credentials belong in this
repository.

## Use

```bash
python -m observations.cli status                       # where the archive lives, and space
python -m observations.cli discover --sources ghcnd ecad gsod
python -m observations.cli report                       # read this before downloading
python -m observations.cli fetch --sources ghcnd        # restartable, checksummed
python -m observations.cli identity                     # canonical station crosswalk
python -m observations.cli review                       # workbook for unresolved pairs
python -m observations.cli consolidate                  # the gold table
python -m observations.cli homogeneity                  # breakpoint diagnosis
```

Discovery comes before acquisition on purpose: `fetch` plans from an
inventory `discover` already wrote, so there is no path that bulk-downloads
without a coverage record first.

## Breakpoint diagnosis

`homogeneity.py` applies relative homogeneity testing: build a difference
series between a station and its nearest neighbours (real weather moves
both together, an artificial break does not), then apply the Standard Normal
Homogeneity Test (Alexandersson 1986) to find the most likely breakpoint.

It is tuned conservatively, because published comparisons (MULTITEST) show
the fraction of series judged homogeneous swinging from 85 % to 22 % across
established tools on identical data — how conservative a method is matters
more than which statistic it uses. So:

- testing runs on **monthly anomalies**, not daily values, as Climatol and
  PHA do — daily difference series have far worse signal-to-noise;
- neighbours are gated on the correlation of **first differences**, which
  stays robust to the very breaks being looked for;
- a break must clear **SNHT ≥ 20** (Climatol's convention) *and* a
  permutation p-value;
- **at least two neighbours** must agree, since one neighbour cannot say
  whether the target or the neighbour moved;
- the whole report carries a **Benjamini-Hochberg** correction.

Output is a diagnostic table with a `status` column saying whether each
series was testable at all. No value is ever changed.

## Tests

```bash
python -m pytest tests/ -q
```

Tests build synthetic provider-format data; no real provider records are
included in this repository.

## Data licensing

**This repository contains code only.** It redistributes no observations,
and the archive it builds must not be committed. Provider terms apply to
anything you download — notably **ECA&D public station data is
non-commercial research and education only**. See `LICENSE` for the full
per-source list.

## Provenance

Extracted from a statistical climate-downscaling project for the eastern
Mediterranean, where it produced a 1,537-station canonical archive from
1,944 source records. Some docstrings cite numbered sections of that
project's design document; the rules they refer to are summarised above.
