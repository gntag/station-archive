# Station-archive acquisition (Step 1, download layer)

Source adapters and discovery for the Delos station-observation archive.
Implements sections 1.2, 1.5 and the bronze layer of 1.7 of
`RESEARCH_PLAN_AND_DESIGN.md`.

## Scope

This package **fetches and parses provider data**. It does not decide station
identity, quality control, source priority or gap filling — those are separate
modules per the Step 1 architecture guardrail:

> Source adapters fetch and parse provider data; they do not decide station
> identity or source priority.

`identity.py` (plan 1.3) is built: it reconciles the four sources into
canonical physical stations. `standardize.py` currently holds only the unit
primitives that identity needs.

Still to build: full silver materialization, `qc.py`, `consolidate.py`, the
fold manifest, and E-OBS as a gridded comparator.

## Where the data goes

The repository drive (G:) was profiled at ~0.1 GB free, so snapshots go
elsewhere. Set the root before running anything:

```powershell
$env:DELOS_STATION_ARCHIVE = "E:\station_archive"
```

Default if unset: `E:/station_archive`. Layout:

```
E:/station_archive/
    raw/<source>/<release>/...            immutable provider bytes
    manifests/source_releases.jsonl       source, version, time, licence
    manifests/<source>/<release>/requests.jsonl
    manifests/<source>/<release>/snapshots.jsonl
    inventory/<source>_inventory.csv
    inventory/coverage_report.md
```

## Usage

Discovery comes before download — that ordering is the point, not a formality.

```powershell
# 1. What does each source actually offer inside the domain?
python -m observations.cli discover --sources ghcnd ecad gsod

# 2. Read the report before transferring anything.
python -m observations.cli report

# 3. Fetch. Restartable and idempotent; re-running skips unchanged files.
python -m observations.cli fetch --sources ghcnd

# Inspect the plan without transferring.
python -m observations.cli fetch --sources ecad --dry-run

# Where is the archive, and is there room?
python -m observations.cli status
```

## Sources

| Source | Access | Domain footprint | Licence |
|---|---|---|---|
| `ghcnd` | per-station `by_station/<ID>.csv.gz` | ~260 kB/station | NOAA open data |
| `ecad` | ranged reads into `ECA_blend_<elem>.zip` | a few kB/station-element | **Non-commercial research and education only** |
| `gsod` | per-station-year CSV | ~80 kB/station-year | NOAA open data |
| `c3s_land` | Copernicus CDS API | one request per year | Copernicus / CC-BY-4.0 |

### ECA&D and the 5.4 GB problem

ECA&D publishes no per-station endpoint — only whole-element archives
totalling ~5.4 GB compressed for the seven elements this project needs. Of the
~3000 stations per archive, only a few dozen are in domain.

`ziprange.RemoteZip` reads each archive's central directory from its tail over
HTTP range requests, then fetches only the wanted members. The member CRC from
the central directory is verified on every read, because a mis-offset range
returns plausible bytes that would otherwise be stored as a valid snapshot.

### C3S variable coverage

The dataset offers ten variables, and **relative humidity is not among them**,
nor are daily minimum/maximum temperature. The adapter requests
`air_temperature` and `dew_point_temperature` but does not claim to supply
`hurs`, `tasmin` or `tasmax`. Deriving RH is a standardization decision
(plan 1.4 permits it "only when both inputs and the formula are recorded").

Credentials come from `~/.cdsapirc` or `CDSAPI_URL`/`CDSAPI_KEY`. Any
credential-looking payload key is redacted before it reaches a manifest.

## Provenance guarantees

- A request manifest is written **before** transfer, so a run that dies partway
  still records what it intended to fetch.
- A file is skipped only when its recorded checksum still matches the bytes on
  disk — existence alone is not enough, since an interrupted transfer leaves a
  file that looks complete.
- Transfers land in a `.part` file and are renamed only on success.
- `manifests.archive_checksum()` is order-independent, so an unchanged snapshot
  set reproduces the same value (plan 1.8).
- Raw values stay in provider units: GHCN-D tenths, ECA&D 0.1 units, GSOD
  Fahrenheit/inches/knots, C3S kelvin. Conversion belongs to `standardize.py`.
- Provider quality flags are preserved, never overwritten. ECA&D `Q=1`
  (suspect) rows are kept **with** the flag — whether to drop them is `qc.py`'s
  decision, not the downloader's.

## A caveat on the coverage report

The report groups stations by 0.02° proximity to estimate how many sites each
source adds. That is a **coverage estimate only**. It is not the canonical
crosswalk of plan 1.3, which requires WMO/WIGOS identifiers, name and elevation
compatibility, and overlapping-series correlation. Proximity grouping will
merge genuinely distinct nearby stations and separate a station that moved.

## Relationship to Station_Data.py

`Station_Data/Station_Data.py` is untouched and still works. Its output at
`Station_Data/downloaded_station_data/` is what
`downscaling/station_ingest.py` reads today, and `downscaling/config.py`
`STATION_DIR` still points there.

The `gsod` adapter re-acquires the same files into the versioned archive so
GSOD carries the same provenance as the new sources. Nothing downstream is
repointed until consolidation exists.

## Canonical identity (plan 1.3)

```powershell
python -m observations.cli identity
```

Writes to `<archive>/canonical/`: `station_aliases.csv`,
`station_metadata_history.csv`, `duplicate_evidence.csv`,
`metadata_conflicts.csv`.

Matching ladder, in descending confidence:

1. **Exact identifier.** C3S `primary_station_id` *is* a GHCN-D identifier, so
   all 118 C3S stations join here. GHCN-D and GSOD link on WMO number.
   ECA&D publishes neither, so all 1,463 of its stations must earn a link at
   tier 2/3.
2. **Coincident coordinates** (≤0.22 km) + compatible elevation + series test.
3. **Near coordinates** (≤3 km) + compatible elevation + series test.
4. **Unresolved** — stays separate, carries `duplicate_risk`.

A non-identifier merge requires *all* of: distance, elevation ≤50 m, and
daily correlation ≥0.95 over ≥365 shared days. Thresholds live in
`config.py`, because they are scientific policy.

### Two deliberate refinements, both driven by measured evidence

**The distance gate started at ~1 km and was widened to ~3 km.** At 1 km it
rejected `ecad:002346` / `ghcnd:BUM00015502`, which are identical to the last
decimal on all 3,210 days they share. Archives disagree about a station's
"official" coordinates by more than a kilometre, so distance alone cannot
separate a duplicate from a neighbour at this scale. The series test does the
discriminating, which is what tier 3 intends.

**Proven-identical series override the elevation gate.** `ecad:010963` and
`ghcnd:HRE00155411` sit at *identical* coordinates and agree exactly on all
7,670 shared days, yet their archives differ by 113 m in stated elevation.
The override is narrow — mean absolute difference ≤0.05 and ≥99% exact
agreement over ≥365 days — so merely-correlated stations never qualify. The
elevation is **not** corrected: plan 1.4 says diagnose, do not silently
homogenize. The conflict goes to `metadata_conflicts.csv` instead.

That file matters beyond identity: elevation is a model predictor
(`delta_elev_m`), so a station carrying a wrong elevation corrupts the fitted
relationship however its identity is resolved.

### A caveat on correlation

Correlation measures covariation, not agreement. Two genuinely distinct
stations a few hundred metres apart correlate above 0.98 while differing by
more than a degree every day. Every pair therefore also records
`mean_abs_diff`, `bias` and `exact_fraction`, and merges that clear the
correlation gate while disagreeing in value are flagged `agreement_review`.

### Result on the real archive

1,944 source stations → **1,644 canonical stations** (300 duplicates
reconciled); 292 duplicate-risk flags, 2 agreement reviews, 14 metadata
conflicts, 92 groups formed transitively.

## Tests

```powershell
cd G:\Dilos\Station_Data
python -m pytest tests/ -q
```

No network access. Fixtures in `tests/fixtures.py` are verbatim excerpts of
real provider responses captured 13 September 2026, so a provider format change
fails a test against the format the adapter was written for.
