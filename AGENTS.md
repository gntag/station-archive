# Notes for agents and other readers arriving cold

This file is the full detail, and it is self-contained. `README.md` is the front
door: the architecture, how to run it, and what it produces — enough to install
the package and build an archive. Everything else is here, in this order:

| If you are asking | Read |
|---|---|
| what may I never break? | **Invariants** |
| why is this code shaped like this? | **Module map**, **Things that look wrong and are not** |
| what does this output column mean? | **Reference: output schema** |
| how is each decision actually made? | **How each stage decides** |
| what did it produce on the real archive? | **Results on the real archive** |
| how do I add a source or move a threshold? | **Changing this repository** |
| what does "plan section 1.6" mean? | **Appendix: design requirements** |

Those two files are the whole documentation set. If you read only one section
before changing code, read **Invariants** — every one is load-bearing, several
were learned from a measurement, and breaking one silently is the failure mode
this repository is built to prevent.

## What this is

A pipeline that reconciles four overlapping public climate sources into one
daily station-observation archive where every value is traceable. It is a CLI
over a layered archive on disk; each command reads what the previous one wrote
and refuses to run if that is missing.

```
discover → report → fetch → identity → climatology → review → consolidate
                                                            → homogeneity
                                                            → provenance
```

Once the archive exists, the quality-control loop is two commands around one
sitting with the review workbook: `prepare` (identity → climatology → review)
and `apply` (identity → dem → consolidate → homogeneity → provenance).

`python -m station_archive.cli demo` runs the whole chain on synthetic data in
seconds, with no network and no credentials. **Start there.** It exercises the
real adapters and the real decision code, so it is the cheapest accurate picture
of what the pipeline does.

## Module map

| Module | Owns | Never does |
|---|---|---|
| `settings.py` | Archive paths, HTTP behaviour, disk limits | Anything scientific |
| `policy.py` | Domain, window, every threshold, source priority | Anything about transport or paths |
| `types.py` | The vocabulary adapters share with the pipeline | Any provider-specific logic |
| `sources/base.py` | The adapter contract and shared provenance | Decide identity, priority or units |
| `sources/registry.py` | Which sources exist and what each may do | Hold a second copy of anything |
| `sources/{ghcnd,ecad,gsod,c3s_land}.py` | Fetch and parse one provider | Convert units, rank, merge |
| `net.py` | Restartable checksummed transfer | Interpret content |
| `ziprange.py` | Reading ZIP members over HTTP ranges | Anything domain-specific |
| `manifests.py` | Provenance records, including the archive-level one | Judge data |
| `bronze.py` | Locating and reading snapshots | Convert or select |
| `standardize.py` | Unit conversion, and nothing else | Adjust or correct values |
| `qc.py` | Which values are not observations | Modify a value |
| `identity.py` | Which records are one physical station | Change any observation |
| `dem.py` | Settling elevation disagreements against AWS Terrain Tiles | Invent an elevation; write to the archive |
| `review.py` | The workbook a human adjudicates in | Decide anything itself |
| `consolidate.py` | Which value wins, and why | Interpolate or average |
| `homogeneity.py` | Which series contain artificial breakpoints | Correct one |
| `climatology.py` | Which source records sit implausibly far from an independent reference, and reading the reviewer's exclusion answers back | Remove a value; act on its own proposal |
| `inventory.py` | The coverage report that answers "is this source worth fetching" | Decide identity; its clustering is an ESTIMATE |
| `cli.py` | Stage orchestration and everything printed to a terminal | Hold a scientific rule of its own |
| `demo.py` | A runnable synthetic end-to-end | Touch the network |

Dependency direction is one-way: `sources/*` → `bronze` → `identity` →
`consolidate` → `homogeneity`, with `climatology` reading `identity` and
`consolidate` and read only by `cli` (consolidation takes the reviewer's
exclusions as plain data, never by importing the screen). `policy` and `settings` are read by everything
and depend on nothing except `types` and the registry — with two deliberate,
function-local exceptions, both deferred imports so no module-scope cycle
exists: `policy.__post_init__` reads `qc.gated_variables()` to refuse a
variable it cannot range-check, and `settings.USER_AGENT` reads
`__version__`. `inventory` reads `identity.longitude_scale`, in the allowed
direction.

**This repository builds the archive and stops there.** Reading it into a
modelling schema, cross-validation fold design and the choice of which stations
to score live in the consuming project (`ungauged-downscaling`). If a change
here starts to need a fold count, a model schema or a train/test split, it is
in the wrong repository — that was true of `fold_manifest.py`, which sat here
unused and duplicated until it moved.

## Invariants

Break any of these and the archive stops meaning what it claims. Each is tested.

1. **Nothing is invented.** No interpolation, no averaging of two sources, no
   bias correction, no coordinate averaging. A day with no accepted candidate
   stays missing. `standardize.GSOD_DAY_WINDOW_BIAS_CORRECTION` is empty on
   purpose and pinned shut by `tests/test_standardize.py` — read the comment
   there before considering an entry.
2. **Nothing is silently discarded.** Rejections are counted by reason, losing
   candidates are kept in an audit table, failed merges keep their statistics.
3. **Bronze is immutable.** A changed provider file becomes a new snapshot under
   a new release, never an overwrite. Nothing downstream writes to `raw/`.
4. **Diagnose, never homogenise.** Breakpoints and metadata conflicts are
   flagged; the values stay as the provider issued them.
5. **A format change stops loudly.** Every parser validates its structure and
   raises `SchemaError` rather than returning partial data. Do not add a
   `try/except` that turns this into a warning.
6. **Adapters do not decide policy.** They fetch and parse. The only policy an
   adapter sees is the fetch-year window and, for API sources, the domain.
7. **Determinism.** Identical snapshots produce an identical archive checksum.
   Any change to iteration or aggregation order has to preserve this.

## Things that look wrong and are not

- **`_degrees_to_km` converts a degree tolerance along a meridian, and
  `candidate_pairs` scales longitude by the cosine of the domain's *edge*
  latitude, not its middle.** Both are deliberate: the gate is a circle in
  kilometres, and scaling by the middle latitude leaves the guarantee false at
  the edge — which is where a real duplicate pair was once missed entirely.
- **`GsodSource.release()` reuses an existing release rather than minting one
  per run.** Idempotency is keyed on the release, so minting daily re-downloaded
  everything and could leave the reader choosing an empty release.
- **Precipitation homogeneity is tested in log space.** SNHT assumes constant
  variance; rainfall's variance scales with its level. A `pr` shift is therefore
  in natural-log units — consult `transform` before reading `shift`.
- **`homogeneity` reports `min_support_at_break` separately from
  `n_neighbors_used`.** The first is the number the two-neighbour rule is about;
  the second counts neighbours over the whole series and once made a break in a
  single-neighbour window look safeguarded when it was not.
- **The vectorised unit conversions duplicate the scalar formulas verbatim.**
  That is so they stay bit-identical; do not "simplify" `(v - 32) * (5/9)` into a
  scale-and-offset form, which re-associates the arithmetic.
- **Tier 1 (exact identifier) now also checks distance, but never elevation.**
  An identifier is the strongest evidence in the ladder, so it used to accept
  on that alone — until 18 GSOD stations turned out to carry a WMO number
  belonging to someone else and merged METHONI with CYCLADES ISLANDS 327 km
  apart. Distance is checked because an identifier must at least be
  geographically possible; elevation is not, because two archives disagreeing
  about a station's height is common and is settled by `dem.py`, not treated as
  evidence of two stations.
- **Coordinates come from the alias with the most observations; elevation may
  not.** `resolve_canonical_metadata` takes a station's whole metadata block
  from one alias, and `canonical_stations.csv` always honours that. But a
  consumer applying `review/elevation_overrides.csv` can get coordinates from
  one archive and a DEM-endorsed elevation from another — 15 of 25 overridden
  stations here. Deliberate: elevation is a model predictor worth settling
  against third-party evidence, a coordinate is settled by who observed most.
  The mixing happens at read time only.
- **`dem.py` picks among published elevations and never writes the DEM value
  itself.** Section 1.3 selects metadata from what a provider asserted; a DEM
  pixel is not a station's recorded height. The recorded reason carries the gap
  between the DEM and the winner, because the nearest candidate always wins —
  a large gap means the DEM endorsed nothing.
- **A reviewer's decision reaches tier 1, even though tier 1 is "never
  re-tested".** Those two statements are about different things and both hold:
  the automated gates never re-run on an identifier pair, and a human verdict
  applies to every tier. `_apply_decision` is a function precisely so the two
  loops cannot drift — it used to be inline in the candidate loop, which is the
  one place the identifier tiers never reach, so 156 accepted identifier merges
  and one too-far refusal carried a `DECISION_merge` column that silently did
  nothing. An overturned pair keeps the tier the rules gave it, so the evidence
  table never presents a human decision as machine evidence.
- **`check_compatibility` refuses an overlap in which no calendar month is well
  populated, even though every other statistic passed.** Excluding thin months
  from the discontinuity test is strictly more permissive, which opens a hole at
  the other end: 180 days spread one-per-week clears `min_overlap_days` while
  leaving the monthly gate nothing to measure. Scoring that 0.0 and passing
  would let a thinly-spread overlap skip the gate entirely, so a test that
  could not run is recorded as a failure, not a pass.
- **The permutation p-value can never be 0, and the smallest value in the
  report is therefore `1/(n+1)`.** `(1 + count) / (1 + n)` is the standard
  estimator: 500 shuffles cannot resolve a p-value below 1/501, and reporting
  0.0 claimed certainty the method does not have. It matters because p feeds
  Benjamini-Hochberg — at 0.0 a break passes BH at any rank. On this archive the
  flagged set is 481 either way; on a report where few series carry a break the
  two answers differ.
- **The climatology screen compares a record with the reference on the
  record's own days, never mean against mean.** A record's sampling can
  manufacture a departure by itself: GSOD reports a gust only on gusty days (a
  median 10% of the days a GSOD station reports wind, on which the reference
  gust runs 1.45x its all-days mean). Compared mean against mean, a Sicilian
  network reporting every day looked half as gusty as its GSOD neighbours;
  on shared days it is 0.86 of them, and against the reference a normal 0.94.
  `reporting_fraction` and `sampling_note` are on every row so this stays
  visible.
- **A blank `DECISION_exclude` keeps the record, even on a row the screen
  recommends excluding.** The screen proposes; only an explicit `yes` removes a
  record from selection. `apply` prints how many proposals were left blank, so
  "nothing was excluded" is never mistaken for "nothing was wrong".
- **The screen judges source records, not canonical stations.** That is what
  lets merges and exclusions be answered in one sitting -- an exclusion's
  subject does not change when a merge does -- and what lets an excluded
  record hand its days to the next-ranked source of the same station.
- **`check_gross_range` drops NaN for a variable it has no bounds for, but
  range-judges nothing.** The two rules are separate on purpose: the NaN guard
  is unconditional because the docstring promises it, while inventing a
  plausible range for an unbounded variable is exactly the kind of guess this
  repository does not make. `Policy.__post_init__` closes the gap from the other
  side by refusing to *select* a variable with no bounds, so anything reaching
  the gold table has been checked against limits someone wrote down.
- **`c3s_land` is in the crosswalk but in no priority list.** It redelivers
  GHCN-D in the original domain, so it is audit evidence, never a value source.
  The registry enforces this and `policy.Policy` refuses a configuration that
  contradicts it.

## The boundary, concretely

The archive owes a consumer traceable observations plus the things only it can
know. The load-bearing one is **unresolved duplicates**: `duplicate_risk` on an
alias means "this may be the same physical station as another and I could not
prove it either way". A consumer building a train/test split has to keep such a
pair together or it leaks. The archive states the risk and records the evidence
in `duplicate_evidence.csv`; it does not decide the split.

## Working on this

- `python -m pytest tests/ -q` — no network, a few seconds. Run it before and
  after any change. `dem.py` is the only module that reaches the network in
  normal use, and its tests substitute the fetcher; a test that needs a socket
  is a bug in the test.
- Adding a source? One entry in `sources/registry.py` plus the adapter. If you
  find yourself editing a second list of source names, the registry is being
  bypassed. **Adding a source** below is the full procedure.
- Changing a threshold in `policy.py` changes what a rebuilt archive says.
  `tests/test_policy.py::test_default_policy_matches_the_published_archive` will
  fail; that is the reminder to write the change up in the release notes on
  GitHub, and to put the measurement behind it in the comment beside the value.
- The docstrings cite numbered sections of a design document that is not in this
  repository. **Appendix: design requirements** below says what each cited
  section requires, so every citation resolves without it.


---

# How each stage decides

### How identity is decided

Four sources describe an overlapping set of physical stations under different
identifiers; the same airport may appear as an ECA&D STAID, a GHCN-D ID, a GSOD
USAF-WBAN pair and a C3S primary ID. Treating those as four stations would both
overstate the network and let one physical site appear in training and
validation at once.

What the archive actually supports, measured rather than assumed:

- No source publishes a WIGOS identifier.
- ECA&D publishes no WMO number at all, for any of its 1,463 domain stations —
  so tier 1 is unavailable to the largest source, and every ECA&D link must be
  earned at tier 2 or 3.
- C3S `primary_station_id` is literally a GHCN-D identifier, and all 118 C3S
  domain stations match a GHCN-D one exactly. That is a documented cross-source
  identifier, so C3S joins at tier 1.
- GHCN-D and GSOD share WMO numbers.

### The matching ladder

| Tier | Rule | Evidence required |
|---|---|---|
| 1 | Exact identifier | GHCN-D ID (shared with C3S) or WMO number; never within one source; **and the two records must be within the same ~3 km the other tiers use** |
| 2 | Coincident coordinates (≤ 0.22 km) | **plus** compatible elevation **plus** the series test |
| 3 | Near coordinates (≤ 3 km) | **plus** compatible elevation **plus** the series test |
| 4 | Unresolved | Stays separate, carries `duplicate_risk`, goes to the review workbook |

A non-identifier merge requires **all** of: distance within tolerance,
elevation within 50 m, daily correlation ≥ 0.995 over ≥ 365 shared days, and
mean absolute difference ≤ 0.5 °C. Correlation alone is never enough —
stations 2.4 km apart were measured correlating at 0.9915 while differing
1.3 °C every single day.

### Rules that came from measurements, not from theory

- **An identifier still has to be geographically possible.** Tier 1 once
  accepted on the identifier alone — no distance, no series, nothing. That is
  sound only while the identifiers are, and they were not: 18 GSOD stations
  carried a WMO number belonging to a different station, because the number was
  read off the filename without checking that the USAF id ended in the `0` that
  marks a WMO station. The result was METHONI merged with CYCLADES ISLANDS
  327 km away, KALAMATA with MIKONOS and SYROS, HERAKLION with KARPATHOS — and
  4,226 observations filed under a place up to 327 km from where they were
  measured. The derivation is fixed, and an identifier match must now also
  clear the tier-3 distance. Elevation is deliberately not tested there: two
  archives disagreeing about a height is common, and is settled separately.
- **Zero overlap means never merge.** Two records sharing no observation day
  are permanently unmergeable regardless of how well their metadata agrees.
  The rule is absolute: it outranks even a reviewer's explicit "merge" decision,
  and such pairs are listed in the workbook without a decision column because
  no answer would change anything.
- **The distance gate started at ~1 km and was widened to ~3 km.** At 1 km it
  rejected `ecad:002346` / `ghcnd:BUM00015502`, which sit 1.26 km apart and are
  identical to the last decimal on all 3,210 days they share. Archives disagree
  about a station's "official" coordinates by more than a kilometre, so distance
  alone cannot separate a duplicate from a neighbour at this scale. The series
  test does the discriminating.
- **Proven-identical series override the elevation gate.** `ecad:010963` and
  `ghcnd:HRE00155411` sit at identical coordinates and agree exactly on all
  7,670 shared days, yet their archives differ by 113 m in stated elevation.
  The override is narrow — MAD ≤ 0.05 and ≥ 99 % exact agreement over ≥ 365
  days — so merely-correlated stations never qualify. The elevation is **not**
  corrected; the conflict goes to `metadata_conflicts.csv` instead. That matters
  beyond identity: elevation is a model predictor, so a wrong elevation corrupts
  the fitted relationship however the identity is resolved.
- **Correlation is not agreement.** Every pair records `mean_abs_diff`, `bias`
  and `exact_fraction` alongside the correlation, and merges that clear the
  correlation gate while disagreeing in value are flagged `agreement_review`.
- **Metadata is selected, never averaged.** One alias — the one contributing the
  most observations **across all variables** — supplies the canonical
  coordinate, elevation and name as a block, so the result is a position some
  archive actually published. Every alias keeps its own coordinates as its own
  time-bounded segment in `station_metadata_history.csv`, and
  `canonical_stations.csv` records the full contest that decided it. Where two
  archives place one station differently, that rule is the answer, and
  `metadata_conflicts.csv` now records which alias won each conflict and on how
  many observations — in this archive the larger ECA&D record wins 11 of 14, a
  GSOD record wins 2, and one resolves to a third alias on an exact tie broken
  by station id.

  Elevation is the one field that can come from elsewhere: `station-archive dem`
  may endorse a different alias's height, and for 15 of the 25 overridden
  stations it does. That mixing happens at read time only — the archive on disk
  stays block-consistent — and it is deliberate, because a wrong elevation
  corrupts a model predictor while a coordinate is properly settled by which
  archive observed the most.
- **Transitive groups are flagged.** If A links to B and B to C but A and C were
  never compared directly, the group carries `group_formed_transitively`.

### The review workbook

Tier 4's duplicate-risk flag is a request for review, not a verdict. `review.py`
turns those flags into an adjudicable workbook: for every rejected pair within
3 km, what the two archives say about the station and what their observations
say, variable by variable, with a plain-language recommendation
(`LIKELY MERGE`, `CONSIDER MERGE`, `MIXED`, `NO EVIDENCE`, `LIKELY DISTINCT`)
and a blank `DECISION_merge` column. Feeding the filled workbook back through
`identity --decisions` applies those decisions — except against the never-merge
rule, which no decision can override.

---

### How quality control works

QC is layered, never destructive. Provider flags are preserved and never
overwritten; project rejections remove a value from the gold table and are
counted by source, variable and reason in `qc_rejections.csv`. Nothing edits
the bronze layer.

**1. Provider sentinels, at parse time.** Each adapter drops only what its
provider documents as absent: GHCN-D `-9999`, ECA&D `-9999` and `Q=9`, and
GSOD's per-column sentinels (`9999.9` for temperatures, `999.9` for wind,
`99.99` for precipitation). ECA&D `Q=1` (suspect) rows are kept **with** the
flag — whether to drop them is a QC decision, not a downloader's.

**2. Flag-based rejection — the GSOD precipitation problem.** This is the rule
that matters most in this archive, and it is quoted from NOAA's own readme
rather than inferred:

> `0` indicates no measurable precipitation (includes a trace)
>
> `I` = Station did not report any precipitation data for the day …
>
> `H` = Station reported `0` … but also reported at least one occurrence of
> precipitation … should be considered as incomplete data for the day.

Measured on the downloaded archive: 45.8 % of GSOD's zero-precipitation rows in
a 400-file sample carry flag `I`. Those are not measured zeros; they are days
the station reported nothing, and reading them as 0.0 mm inflates dry-day
frequency. Flags `I` and `H` are rejected as non-observations — **240,704
values in this archive**. Flags `A`–`G` describe which hours the total covers
and are genuine, if partial, observations, so they are kept.

GSOD cannot preserve trace precipitation separately from zero at all — its own
readme says a reported zero includes a trace, and the information is not
recoverable from the file. That is recorded as a known limitation of the source,
and is why GSOD ranks last for `pr` despite having the most stations.

**3. Sign checks.** Negative precipitation and negative wind speed are broken
records, not small amounts.

**4. Gross-range checks, in canonical units.** Deliberately wide — QC removes
the impossible, not the merely unusual, because an aggressive range filter would
silently delete the extremes a downscaling model and a bias-adjustment reference
most need (−45 °C survives; 200 °C does not).

This is not hypothetical: GHCN-D was found carrying **999.0 °C** at some Italian
stations — an undocumented sentinel that the format's own `-9999` filter misses.
One such value dragged a station's standard deviation from 6.8 to 46.1 °C and
its correlation against the same record in another archive from ~1.0 to 0.14,
causing an identical pair to be rejected on a statistic that was pure artefact.
The check therefore runs both in identity matching and before selection.

**5. Diagnostics that report rather than delete.** `tasmin ≤ tas ≤ tasmax`
violations are returned row by row with the offending values named, and runs of
an identical value (a stuck sensor) are flagged — but nothing is removed,
because which of three values to drop is a policy question that has not been
settled. Coverage and missingness are summarised by station, variable, year and
month.

**6. Units are converted, never guessed.** An unrecognised provider unit raises
`UnknownUnitError` rather than passing through as if already canonical.

**No bias correction is applied.** `standardize.GSOD_DAY_WINDOW_BIAS_CORRECTION`
is deliberately empty and pinned shut by tests. An earlier version subtracted
0.303 °C from GSOD `tas`; it was withdrawn because every step from the
measurement to the application was wrong — the constant is a *tasmax* statistic,
it was applied to *tas*; it could only ever act where GSOD meets ECA&D, a pair
whose measured bias is 0.0002 °C; and it had the wrong sign, since GSOD runs
*colder* than ECA&D here by a median −0.16 °C. Applying it pushed two Greek
stations past the fallback guardrail and discarded 4,750 previously accepted
days. The table is kept as the single named place such a correction would have
to go, so adding one is a visible, reviewable change that must state what was
measured, on which source pair, and for which variable.

---

### How consolidation chooses a value

For each `canonical_station_id × date × variable`, exactly one value becomes the
canonical observation, and the reason is recorded on the row.

**Source priority is per variable, and was set by the pilot, not by hypothesis.**

| Variable | Priority | Why |
|---|---|---|
| `tas` | ecad → gsod | GHCN-D TAVG is excluded entirely: it is reported by a subset of stations and its definition varies by contributing network |
| `tasmin`, `tasmax` | ecad → ghcnd → gsod | Coverage and semantics agree |
| `pr` | ecad → ghcnd → gsod | Coverage favours GSOD (203 stations vs GHCN-D's 155); semantics do not. Semantics win |
| `hurs` | ecad | The only source for humidity in this domain, at 425 stations |
| `sfcWind` | ecad → gsod → ghcnd | |
| `gust` | gsod → ecad → ghcnd | GSOD's 203 stations against ECA&D's 65 |

`c3s_land` appears in no priority list. Every one of its stations in this domain
carries an exact GHCN-D identifier, so it is a redelivery of GHCN-D rather than
independent evidence. It still joins the crosswalk at tier 1 and is kept as a
value-level cross-check — kept deliberately separate from the selectable set so
an audit-only source cannot silently become a fallback value provider.

**Two modes that must not be conflated.** *Same-station fallback* fills a
missing day from a lower-priority alias of the *same physical station*; this
module does that, and only after the compatibility test passes. *Network
expansion* admits a *different* physical station because it improves coverage —
that is not gap filling at all, such a station keeps its own canonical identity,
and nothing here ever writes one station's value into another's series.

**The compatibility test, run at the station where the switch is about to be
made**, over ≥ 180 shared days:

| Check | Refuses when | Why |
|---|---|---|
| Bias | \|mean difference\| > 0.5 °C / 1.0 mm / 0.5 m s⁻¹ / 5 % | A step change injected into a series later used to fit a model and define a bias-adjustment reference |
| Dispersion | σ ratio outside 0.75–1.33 | A source recording the same quantity should not be markedly smoother or noisier |
| Wet-day frequency | differs by > 10 percentage points (`pr` only) | GSOD's trace-as-zero convention moves this without necessarily moving the mean |
| Monthly discontinuity | any month departs from the overall bias by > 1.0 °C / 2.0 mm | Two records that agree on average only because a warm half cancels a cold half are not interchangeable |

No archive-wide offset stands in for this test. For `tas` with an ECA&D primary
and a GSOD fallback, the measured overlap bias runs from −1.60 to +0.58 °C
across station pairs — the spread straddles zero, which is exactly why each
switch is tested where it happens.

**What the module must never do:** interpolate (a day with no accepted candidate
stays missing), average two sources, or rewrite bronze.

**Nothing is deleted.** Every candidate value that existed but was not chosen
goes to `non_selected_candidates/` with its reason — either `outranked: date
already covered by a higher-priority source`, or the exact compatibility failure
that refused it, or `reviewer exclusion: …` for a record a reviewer excluded
after the climatology screen. In this archive that audit table holds 3,408,503
values.

Every selected row carries `selected_source`, `source_station_id`,
`source_rank`, `selection_reason`, `is_fallback` and a `source_record_id` that
points back at the provider record.

---

### Breakpoint diagnosis

A breakpoint is a date at which a station's record shifts for a non-climatic
reason — a relocation, an instrument change, a change in observation time,
growth around the sensor. Undetected, it stitches two physical regimes into one
series that a downscaling model and a downstream bias-adjustment stage both
treat as continuous, so an artificial jump can be misread as trend or mask a
real one.

`homogeneity.py` applies **relative** homogeneity testing: build a difference
series between a station and its nearest neighbours (real weather moves both
together, an artificial break does not), then apply the Standard Normal
Homogeneity Test (Alexandersson 1986) to find the most likely breakpoint.

It is tuned conservatively, because published comparisons (MULTITEST) show the
fraction of series judged homogeneous swinging from 85 % to 22 % across
established tools on identical data — how conservative a method is matters more
than which statistic it uses. So:

- testing runs on **monthly anomalies**, not daily values, as Climatol and PHA
  do — daily difference series have far worse signal-to-noise;
- neighbours are gated on the correlation of **first differences**, which stays
  robust to the very breaks being looked for (a step inflates the target's own
  variance, which would otherwise reject a good neighbour exactly when the break
  is largest);
- a break must clear **SNHT ≥ 20** (Climatol's convention) *and* a permutation
  p-value computed from the series' own data;
- **at least two neighbours** must agree, since one neighbour cannot say whether
  the target or the neighbour moved;
- candidate splits within 12 months of either end are never considered;
- the whole report carries a **Benjamini-Hochberg** correction — with ~2,000
  simultaneous tests, an uncorrected α = 0.05 would produce on the order of a
  hundred spurious breakpoints. `significant_fdr` is the column to read.
- precipitation is tested **multiplicatively** (in log space): its spread grows
  with its level, which violates SNHT's constant-variance assumption directly. A
  `pr` shift is therefore in natural-log units, and every result records which
  `transform` produced it so the number is never read in the wrong units.

Output is a diagnostic table with a `status` column saying whether each series
was testable at all (`tested`, `no_usable_neighbor`, `too_few_neighbors`,
`series_too_short`, `no_variance`), plus a flag layer that joins onto the gold
table and marks which side of a break each observation falls on.
**No value is ever changed.**

---


### The human decisions, and where they live

The pipeline stops and asks a person once, between `prepare` and `apply`, and
keeps every answer in `review/` so that no reviewed decision is buried in a
conversation, a commit message or a script:

| File | Decides | Read back by |
|---|---|---|
| `station_merge_review.xlsx` | whether two source records are one station, in `DECISION_merge` | `identity --decisions <path>` |
| `station_merge_review.xlsx`, `Series` sheet | whether a source record is excluded from selection for one variable, in `DECISION_exclude` | `consolidate --decisions <path>` |
| `elevation_overrides.csv` | which elevation is correct where two archives disagree | written by `station-archive dem`; applied by the consumer at read time |

**Why one sitting is possible.** Merges and exclusions do not depend on each
other: the climatology screen judges each source record against a reference
sampled at its own place, so its verdict does not move when a merge does, and
an exclusion never feeds back into the merge evidence -- a record carrying a
+5 degC offset still proves whether two archives hold one instrument. Elevation
asks nothing: `dem` applies the rule (coordinates from the record with the most
observations, the published elevation nearest the DEM) after the merges, to the
groups that then exist. Breakpoints are the one thing that cannot be in the same
sitting, because they can only be found on the final series; they stay flags.

**The `Series` sheet** lists records the screen put in a tier, strongest first:
`EXCLUDE` above 4 robust SDs from the median departure of that variable,
`REVIEW` between 3 and 4. Each row carries the departure, the matched-day means,
the reporting fraction, and neighbour context -- how many stations within
100 km, how they depart, how many depart the same way. A summit with no
neighbours, a station its neighbours disagree with, and a whole network offset
together read differently there. `Series_decisions` is its ledger, for the same
reason `Decisions` exists: an answer whose row is no longer offered still
governs consolidation.

It opens on `Start_here`, which lists every tab, what it shows, which column the
reviewer fills and what to write there (`review.instructions_sheet`). It is
qualitative on purpose, so it needs no analysis to rebuild and never goes
stale against the numbers, and it has no decision column, so nothing is read
from it.

Beyond `Start_here` and `Summary`, the merge decisions use five sheets.
`Review` holds pairs the rules **refused**, within
3 km, for you to overturn. `Already_merged` holds the opposite and is the newer
of the two: every pair the rules merged **without being asked**, widest and
least-agreeing first. A refusal leaves two stations that were always two; a
wrong merge fuses two sites and puts one of them in the other's series and
folds, so it is the more consequential direction. Writing `no` there splits the
pair. `Beyond_radius`, `Never_merge_rule` and `Metadata_conflicts` are context.

A further sheet, `Decisions`, is the durable ledger: every decision ever made,
with `still_offered` saying whether the pair still appears elsewhere in the
workbook. It exists because a `yes` merges its pair, so the next rebuild
presents it nowhere — not in `Review`, which holds only refusals, and not in
`Already_merged`, which excludes reviewer-decided pairs. Without the ledger a
rebuild dropped those decisions and the merges they authorised were lost on the
next run. In this archive 467 merge decisions are recorded: 146 merge a pair
the rules refused, 1 splits a pair the rules merged, and the rest confirm what
the rules concluded. Every reviewer-merged pair and every confirmed automatic
merge is in exactly that position; a confirmed refusal stays on `Review`.

Every sheet that shows a decision column is read back, and a rebuild carries
forward what was already decided — neither was always true, and a column
nothing reads is worse than no column at all.

Both are inputs, not outputs. The pipeline rebuilds them -- `review` writes
the workbook afresh, `dem` recomputes the rows it computed itself -- but never
loses an answer a person wrote: every decision is carried into the rebuilt
workbook and its ledger, and a hand-written override is kept as it stands.
Neither is applied silently: a merge decision is recorded in
`duplicate_evidence.csv` as `reviewer decision: …` alongside what the automated
gate had concluded, a series exclusion is recorded in `non_selected_candidates/`
and `qc_rejections.csv`, and an elevation override is applied at **read time**.
The consolidated archive is therefore reproducible from its source snapshots
together with the review workbook, not from the snapshots alone.

`elevation_overrides.csv` has columns `canonical_station_id`, `elevation_m` and
`reason`, one row per correction, and the reason is expected to carry the
evidence. `station-archive dem` fills it in for any station whose aliases
disagree by more than 10 m, reading AWS Terrain Tiles at the station's position
and choosing **the published elevation nearest to the DEM** — it never writes
the DEM's own reading, because a DEM pixel is not a height any provider
asserted. Rows written by hand are kept untouched, so a correction you made
by hand outranks one the command would compute. Rows the command wrote itself
(their reason begins `elevation disagreement of`) are recomputed on every run,
because each was computed for its station's group at the time, and a merge
decision applied since can change the candidates or leave nothing to disagree
about. An example of the hand-made
kind reads, in full:

> `metadata_conflicts.csv`: verified-identical series (dist=0 km, tasmax r=1.0)
> disagreed 52 m (ecad, selected by observation count) vs 165 m (ghcnd/c3s_land).
> CERRA orography at (42.56, 18.27) reports 164.03 m, confirming 165 m.

That is the shape a row here is meant to have: which table surfaced the
conflict, what the two archives claimed, and the independent evidence that
settled it. An override without that is a guess with a filename.

Note the division of labour — `identity.py` **reports** the conflict to
`metadata_conflicts.csv` and never corrects the elevation itself, because the
archive diagnoses rather than homogenises. The correction is a separate,
reviewed, versioned act, which is why it lives here and not in the crosswalk.

---

---

# Source-specific detail

### ECA&D and the 5.4 GB problem

ECA&D publishes no per-station endpoint — only whole-element archives totalling
~5.4 GB compressed for the seven elements this project needs (`rr` alone is
1.84 GB). Of the ~3,000 stations per archive, only a few dozen are in domain.

`ziprange.RemoteZip` reads each archive's central directory from its tail over
HTTP range requests, then fetches only the wanted members — turning a 5.4 GB
transfer into a few megabytes. The member CRC from the central directory is
verified on every read, because a mis-offset range request returns
plausible-looking bytes that would otherwise be stored as a valid snapshot.
ZIP64 archives are detected and rejected loudly rather than misread.

### C3S variable coverage

The dataset offers ten variables, and **relative humidity is not among them**,
nor are daily minimum/maximum temperature. The adapter requests
`air_temperature`, `accumulated_precipitation`, `wind_speed` and
`dew_point_temperature`, and does not claim to supply `hurs`, `tasmin` or
`tasmax`. Deriving RH from temperature and dew point is permitted only when both
inputs and the formula are recorded, which makes it a standardisation decision
rather than a download-time relabelling. `year` is a single-choice field on the
CDS form, which is why planning emits one request per year.

The CDS harmonised schema's column names vary between dataset versions, so each
logical role is resolved against the delivered file; an unresolvable schema
raises rather than dropping columns. Any credential-looking payload key is
redacted before it reaches a manifest.

### A caveat on the coverage report

The report groups stations by 0.02° proximity to estimate how many sites each
source adds. That is a **coverage estimate only**. It is not the canonical
crosswalk: proximity grouping will merge genuinely distinct nearby stations and
separate a station that moved. Identity resolution is `identity.py`'s job.

---

# Provenance guarantees

- A **request manifest is written before transfer**, so a run that dies partway
  still records what it intended to fetch.
- A file is **skipped only when its recorded checksum still matches the bytes on
  disk** — existence alone is not enough, since an interrupted transfer leaves a
  file that looks complete.
- Transfers land in a `.part` file and are renamed only on success.
- Snapshots are **immutable**: a changed provider file produces a new snapshot
  under a new release directory rather than overwriting the old bytes.
- `manifests.archive_checksum()` is order-independent, so an unchanged snapshot
  set reproduces the same value.
- Series are located **from the snapshot manifests, not by walking the directory
  tree** — a file no manifest vouches for is not read, which is what keeps "the
  archive" a defined set rather than whatever happens to be in the folder.
- Raw values stay in provider units: GHCN-D tenths, ECA&D 0.1 units, GSOD
  Fahrenheit/inches/knots, C3S kelvin.
- Provider quality flags are preserved, never overwritten.
- Releases come from what the provider itself publishes (`ghcnd-version.txt`,
  ECA&D's `date_timestamp.txt`, the CDS dataset version); GSOD publishes none, so
  its release is labelled `retrieved-<date>` rather than given a bare date that
  could be mistaken for a provider version.
- A provider format change **stops loudly** — every parser validates its expected
  structure and raises `SchemaError` rather than returning partial data.
- Retries are bounded with exponential back-off; 401/403 surface as `AccessError`
  rather than being retried into a timeout, and 404 is treated as a normal
  outcome when an inventory is broader than the archive.
- A fetch run refuses to start if the target drive lacks room, because a
  half-written archive whose checksums describe incomplete files is worse than a
  run that does not begin.

---

# Results on the real archive

Eastern Mediterranean domain (lat 33–43, lon 11–32, +2° buffer), admission
window 1985–2005.

**Discovery** — 6,819 station-variable rows: ECA&D 1,463 stations, GSOD 203,
GHCN-D 160, C3S 118.

**Identity** — 1,944 source stations → **1,546 canonical stations**
(398 duplicates reconciled) from 631 candidate pairs tested, 433 accepted:

| Tier | Merges |
|---|---|
| Exact identifier | 156 |
| Coincident coordinates + series | 183 |
| Near coordinates + series | 94 |

Of those 433, **287 were made by the rules alone** and 146 came from a reviewer
decision; one further pair the rules had merged (`ecad:000174` /
`gsod:16320099999`, 2.98 km) was split by a reviewer on `Already_merged`. All
288 merges the rules had made were reviewed there: 287 confirmed, that one
split. The widest separation inside any canonical station is 5.5 km; before
the identifier gate it was 327 km.

139 aliases carry `duplicate_risk`, 68 of those because the never-merge rule
forbids the only candidate link; 14 accepted merges are flagged for
value-agreement review; 14 metadata conflicts are queued. 398 aliases sit in a
group that was formed transitively rather than by all-pairs comparison, which is
flagged rather than silently trusted.

**Consolidation** — 9,503,860 accepted daily values across 1,203 stations, of
which 202,522 (2.13 %) are same-station fallbacks. Selected days by variable
and source:

| Variable | ecad | ghcnd | gsod |
|---|---|---|---|
| `tas` | 1,077,571 | — | 511,016 |
| `tasmin` | 1,155,713 | 192,328 | 404,056 |
| `tasmax` | 1,114,417 | 218,616 | 399,724 |
| `pr` | 1,771,959 | 201,031 | 237,936 |
| `hurs` | 246,749 | — | — |
| `sfcWind` | 1,551,401 | 2,123 | 299,363 |
| `gust` | 37,218 | 2,485 | 80,154 |

Before the climatology exclusions the table held 9,688,780 values across 1,206
stations (204,099 fallbacks). The difference is accounted for exactly: the
excluded records had supplied 200,834 selected values; on 15,747 of those days
another source of the same station took over (15,061 `pr` -- 10,843 from GSOD,
4,218 from GHCN-D -- 638 `tasmin`, 48 `sfcWind`), the other 185,087 days are
now missing, and promoting a lower-ranked source added 167 days its record had
that the excluded one lacked. No value from a record that was not excluded
moved or changed.

**Quality control** — 240,704 GSOD precipitation values rejected as
non-observations (230,846 on flag `I`, 9,858 on flag `H`), and 246,989 values
excluded by reviewer decision after the climatology screen (`pr` 113,487,
`sfcWind` 63,918, `tasmin` 25,984, `tas` 20,994, `tasmax` 20,295, `gust` 2,311).
The second figure is larger than the 200,834 selected values removed because it
also counts excluded copies that were already losing to a higher-ranked source.

**Climatology screen** — 4,973 source record x variable series reached, 3,002
tested against CERRA on matched days (1,778 had under 365 matched days, many of
them ECA&D records whose provider file begins on 2005-12-31; 193 had no
reference point). 108 were put in a tier, across 67 stations: 62 proposed for
exclusion and 46 for review.

| Variable | proposed for exclusion | review |
|---|---|---|
| `pr` | 42 | 4 |
| `tasmax` | 9 | 13 |
| `tasmin` | 2 | 13 |
| `tas` | 3 | 6 |
| `sfcWind` | 6 | 6 |
| `gust` | 0 | 4 |
| `hurs` | 0 | 0 |

The reviewer excluded 78 (61 of the 62 proposals, 17 of the 46 review rows),
which removed 51 station x variable series from the gold table entirely. What
the screen found:

* **ECA&D and GSOD precipitation that is exactly zero** on 400-6,600 matched days at
  Greek, Bulgarian and Turkish airports (Elefsis, Kos, Santorini, Chios,
  Anchialos, Sofia, Corlu ...). Every archive delivers the same zeros, so these
  stations now have no precipitation series rather than a false one.
* **ECA&D precipitation 4-20x too small** at Monte Cimone, Vigna di Valle,
  Monte S. Angelo, Capo Bellavista and Bracciano, with the same wet-day
  fraction as the GHCN-D and GSOD records they outranked. GHCN-D and GSOD now
  supply those stations where their own records pass.
* **A Lazio block of ECA&D stations reading +5 to +6 degC in `tasmax`** after
  the elevation adjustment (Piglio, Alatri, Licenza, Arpino, Castel di Tora,
  Agosta), each with 10-12 neighbours flagged the same way.
* **Gran Sasso (GSOD) temperatures +6 to +12.5 degC**, with a 900 m gap between
  the station's stated elevation and the reference orography.
* **Summits and capes** (Cimone, Cherni Vrah, Botev, Mourgash, Terminillo,
  Zavizan): wind 3x the reference, temperatures below it. Probably real sites
  the 5.5 km reference cannot resolve; excluded here by reviewer decision.

What it did not find: the Sicilian agrometeorological network's gusts. Its
records sat 6-10 m/s against GSOD neighbours at about 15, which looked like half;
but GSOD reports a gust on a median 10% of days, the gusty ones, on which the
reference gust runs 1.45x its all-days mean. On shared days the network is 0.86
of its GSOD neighbours, and against the reference a normal 0.94. 27 of its 29
gust records are `ok`; Patti and Palermo are in review and were kept.

**Fallbacks** — 263 source switches tested, **146 refused** and their gaps left
open: 77 for insufficient overlap, 43 on monthly discontinuity, 20 on bias, 3
because no calendar month held enough shared days for the discontinuity test to
run at all, 3 on dispersion. All 41 switches between two GSOD records of one
station are refused, almost always for want of overlap: a GSOD station that
changed identifier has none by construction. 3,408,503 non-selected candidate
values retained for audit.

The monthly-discontinuity count fell from 48 with the addition of
`FallbackPolicy.min_days_per_month` (see its comment in `policy.py`): five of those
refusals were driven by calendar months holding one or two shared days, where a
single outlier reads as a large monthly departure. One collapsed from 5.23 mm to
0.0024 mm once thin months were excluded, and 5,766 real observations that had
been left in the audit table are now in the gold table.

**Elevation** — 25 stations carry a reviewed elevation override. The DEM
confirmed a published candidate within 20 m for 15 of them; for 9 it endorsed
none, and every one of those nine sits at a coordinate rounded to a whole
arcminute (±900 m on the ground), so the gap is evidence about the coordinate
rather than the height. One override was decided by hand before the command
existed and is preserved as it stands.

**Homogeneity** — 2,100 station-variable series reached, of which 1,225 were
testable (331 had no usable neighbour, 265 too few neighbours, 260 were too
short, 13 had no variance, and 6 had neighbours but no window those neighbours
could attribute a break in). **478 flags survive the FDR correction**, across
278 stations: 123 `tasmin`, 116 `sfcWind`, 102 `tas`, 98 `tasmax`, 31 `pr`,
8 `hurs`, 0 `gust`. They are listed with station name, position, date and shift
in `consolidated/homogeneity_breakpoints.xlsx`. Before the climatology
exclusions the report reached 2,143 series and flagged 482; the excluded series
are no longer there to be tested, and removing them also changes the neighbour
references of the stations around them.

The smallest p-value in the report is 1/501, not 0: a permutation test with 500
shuffles cannot resolve anything finer, so `(1 + count) / (1 + n)` is what is
reported. On the archive before the climatology exclusions the flag count was
482 either way — the one series that changed verdict did so because a
*neighbour* gained 4,700 recovered fallback days, which moved its reference
series.

Every flag carries `min_support_at_break`, the number of neighbours standing
behind the month the break was found in, and it is at least two on all 478.
That column exists because the two-neighbour rule used to be checked against a
count over the whole series while the search ran over months whose support
could be one. The `homogeneity.py` docstrings explain how that fix and the
neighbour-selection fix pull in opposite directions.

Read a cluster of same-dated flags at neighbouring stations as one event seen
several times rather than several events: this module detects breaks but does not
attribute them across the network, so a station that moved appears in its
neighbours' difference series too, divided by the neighbour count. The `demo`
command shows this happening on data where only one station was made to move.

---

# Known gaps and provenance of the project itself

Extracted from a statistical climate-downscaling project for the eastern
Mediterranean. Some docstrings cite numbered sections of that project's design
document (`RESEARCH_PLAN_AND_DESIGN.md`), which is not part of this repository;
**Appendix: design requirements** below states what each cited section requires.

Known gaps, stated rather than implied:

- The climatology screen compares one long-term mean per record with the
  reference. A record that is right in most years and wrong in a few, or whose
  day-to-day variability or timing is wrong while its mean is right, passes it.
- The elevation override table (`review/elevation_overrides.csv`) carries 25
  corrections, 24 of them computed by `station-archive dem` and one made by hand
  against CERRA orography before that command existed. The other conflicts in
  `metadata_conflicts.csv` remain unadjudicated: they are diagnosed, queued and
  left for a reviewer, not silently corrected. That file is not committed — it
  is an input describing one domain, and a different region's archive needs its
  own.
- Breakpoint detection finds one candidate break per series, not an iterative
  multi-break decomposition, and never attributes a break across the network the
  way PHA does. So one station that moved shows up in its neighbours' flags too:
  read a cluster of same-dated flags as one event seen several times.
- Non-temperature identity thresholds (`pr`, wind) are set by analogy rather than
  measured on this archive — the match variables in practice are `tasmax` and
  `tasmin`. The same is true of the fallback bias and departure limits for those
  units.
- Cross-validation design — spatial folds, how many, and which stations are
  scored — is **not** in this repository. It belongs to the consumer, because it
  depends on the deployment question rather than on the data. See *What this
  repository does not do* in `README.md`.

---

# Reference: output schema

Every table the pipeline writes, column by column. `README.md` summarises the
shape of the output and names the files; this is the field-by-field reference,
and **How each stage decides** above explains why each value is what it is.

Two conventions hold everywhere:

- **An alias key is `source:source_station_id`** — `ecad:000247`,
  `ghcnd:GRE00105241`, `gsod:16716099999`. A `canonical_station_id` is the
  lexicographically smallest alias key in its group, so it *looks* like one
  source's key but names the physical station, not that source's record.
- **Canonical units** are `degC` (temperature), `mm` (precipitation),
  `percent` (relative humidity), `m s-1` (wind). A value is in the canonical
  unit of its variable, and `units` states it on every row.

Canonical variable names follow CMIP/CORDEX: `tas`, `tasmin`, `tasmax`, `pr`,
`hurs`, `sfcWind`, `gust`.

---

### `consolidated/observations_consolidated/` — the gold table

Parquet, written as numbered `part-NNNN.parquet` files. Read them together;
the split is a memory bound, not a grouping. One row per accepted
`canonical_station_id × date × variable`.

| Column | Type | Meaning |
|---|---|---|
| `canonical_station_id` | str | The physical station. Join key to `canonical_stations.csv`. |
| `date` | datetime64 | Observation day. No time component is meaningful. |
| `variable` | str | Canonical variable name. |
| `value` | float64 | The accepted observation, in `units`. Never interpolated, never averaged across sources. |
| `units` | str | Canonical unit for `variable`. |
| `selected_source` | str | Which provider supplied this value. |
| `source_station_id` | str | That provider's own station identifier. |
| `source_rank` | int64 | 1-based position in `policy.source_priority[variable]`. |
| `selection_reason` | str | Why this value won, in words — either "highest-ranked source with data" or the fallback's compatibility statistics. |
| `is_fallback` | bool | True when a lower-priority alias filled a day the primary lacked. |
| `source_record_id` | str | `source:station:date:variable`, traceable to the bronze snapshot. |

**A missing day has no row.** There is no NaN placeholder: absence means no
candidate passed QC, and the reason is in `non_selected_candidates` or
`qc_rejections.csv`.

### `consolidated/non_selected_candidates/`

Every value that existed for a station-day-variable and was *not* chosen.
Nothing is silently discarded (invariant 2).

| Column | Type | Meaning |
|---|---|---|
| `canonical_station_id`, `date`, `variable` | | As above. |
| `source`, `source_station_id` | str | The alias that offered this value. |
| `value`, `units` | float64 / str | The value that lost, in canonical units. |
| `source_rank` | int64 | Its priority position. |
| `rejection_reason` | str | `outranked: …` when a higher-priority source already covered the date, or `incompatible fallback: …` with the failing statistic. |

---

### `canonical/station_aliases.csv`

One row per source station. This is the crosswalk: which provider records are
one physical station.

| Column | Type | Meaning |
|---|---|---|
| `canonical_station_id` | str | The group this alias belongs to. |
| `source`, `source_station_id` | str | The provider record. |
| `name`, `lat`, `lon`, `elevation_m` | | **This alias's own** metadata, as its provider published it. Not the canonical values — see `canonical_stations.csv`. |
| `wmo_id`, `wigos_id` | str | As published. Empty for every ECA&D station; no source publishes WIGOS. |
| `n_variables`, `variables` | int / str | Variables this alias carries, `\|`-separated. |
| `is_representative` | bool | True when this alias's key *is* the canonical id. |
| `match_tier` | int | 1 exact identifier, 2 coincident coordinates, 3 near coordinates, 4 unresolved. |
| `match_tier_name` | str | The tier in words. |
| `match_evidence` | str | Why this alias joined its group, or the nearest failed link if it did not. |
| `duplicate_risk` | bool | **The load-bearing column for a consumer.** True means this may be the same physical station as another and the archive could not prove it either way. A train/test split must keep such a pair together or it leaks. |
| `agreement_review` | bool | Merged, but the two records disagree in value by more than `agreement_review` allows. The merge stands and is flagged. |
| `never_merge_no_overlap` | bool | The nearest candidate shares no observation day, so the pair is permanently unmergeable — no reviewer override. |
| `group_formed_transitively` | bool | The group was built from a chain of pairwise links, not all-pairs comparison. Flagged rather than trusted. |
| `valid_from`, `valid_to` | str | This alias's own record span. |

### `canonical/canonical_stations.csv`

One row per physical station: the metadata the archive actually uses.

| Column | Type | Meaning |
|---|---|---|
| `canonical_station_id` | str | Primary key. |
| `n_aliases` | int | Source records in this group. |
| `metadata_source` | str | The alias every metadata field below was taken from — **as a block**, so the coordinate/elevation pair is one a provider actually published. |
| `metadata_n_observations` | int | That alias's observation count, which is why it won. Blank for a single-source station. |
| `name`, `lat`, `lon`, `elevation_m`, `wmo_id` | | Canonical metadata, all from `metadata_source`. |
| `sources`, `variables` | str | `\|`-separated union across the group. |
| `duplicate_risk` | bool | True if **any** alias in the group carries the risk. |
| `selection` | str | The full contest, e.g. `ecad:000247(7300) > gsod:16716099999(5100)`. |

**One deliberate exception to the block rule.** A consumer applying
`review/elevation_overrides.csv` may end up with coordinates from one archive
and a DEM-endorsed elevation from another — 15 of 25 overridden stations here.
That mixing happens at read time only; this file always honours the block.

### `canonical/duplicate_evidence.csv`

Every pair tested, accepted or not. The audit trail for identity.

| Column | Type | Meaning |
|---|---|---|
| `left`, `right` | str | Alias keys, ordered. |
| `tier`, `tier_name` | int / str | Which rung of the ladder judged this pair. A reviewer override does **not** change the tier — it stays what the rules assigned, so a human decision is never presented as machine evidence. |
| `accepted` | bool | Whether the pair was merged. |
| `distance_km` | float | Great-circle separation. NaN when a record lacks coordinates. |
| `elevation_diff_m` | float | Absolute difference. NaN when either is missing. |
| `name_similarity` | float | Jaccard overlap of name tokens, 0–1. Supporting evidence only. |
| `reason` | str | The verdict in words. `reviewer decision: …` rows carry the automated verdict in parentheses. |
| `match_variable` | str | Which variable the series comparison used. Empty for identifier-tier pairs, which compare no series. |
| `n_overlap_days` | int | Shared days used for the statistics below. |
| `correlation` | float | Pearson r on the shared days. |
| `bias` | float | `left − right` mean difference, canonical units. |
| `mean_abs_diff` | float | Mean absolute difference. **The discriminator** — correlation above 0.98 occurs between genuinely distinct stations 1 km apart. |
| `exact_fraction` | float | Fraction of shared days agreeing to within half a provider quantum. Near 1.0 means one record redelivered. |

### `canonical/station_metadata_history.csv`

One time-bounded segment per alias. No provider publishes a relocation
history, so a moved station appears as two segments rather than one averaged
position — never coordinate averaging (invariant 1).

| Column | Meaning |
|---|---|
| `canonical_station_id` | The station. |
| `segment_source`, `segment_source_station_id` | Which alias this segment describes. |
| `name`, `lat`, `lon`, `elevation_m` | That alias's own metadata. |
| `effective_from`, `effective_to` | The segment's span. |
| `provenance` | Always "source inventory; no provider relocation history". |

### `canonical/metadata_conflicts.csv`

Pairs whose observations prove one instrument but whose metadata cannot both
be right. A review queue, not a correction — the archive diagnoses
(invariant 4).

Carries the `duplicate_evidence` statistics, plus:

| Column | Meaning |
|---|---|
| `merged` | Whether the pair was merged despite the conflict. |
| `conflict` | `elevation`, `coordinates`, or both. |
| `note` | Always "series identical; one archive's metadata is wrong". |
| `coordinates_used_from` | Which alias the archive took coordinates from. |
| `coordinates_used_side` | `left`, `right`, or `another alias` — a group can hold more than the two the conflict names, so the winner is sometimes neither. |
| `lat_used`, `lon_used`, `elevation_used` | What the archive actually uses. |
| `coordinate_rule`, `metadata_contest` | The rule, and the contest that applied it. |

---

### `consolidated/homogeneity_breakpoints.csv`

One row per station × variable **reached** — including ones no test could run
on. Diagnostic only; no value is ever adjusted.

| Column | Type | Meaning |
|---|---|---|
| `canonical_station_id`, `variable` | str | What was tested. |
| `status` | str | `tested`, or why not: `no_usable_neighbor`, `too_few_neighbors`, `no_supported_window`, `series_too_short`, `no_variance`. **Read this before anything else.** |
| `n_months_tested` | int | Months in the reference difference series. |
| `breakpoint_date` | date | First month of the shifted segment. |
| `statistic` | float | Max SNHT T(k). Must clear 20 to be significant. |
| `p_value` | float | Permutation p, as `(1 + count) / (1 + n)`. **Never 0** — the floor is `1/(n+1)`, because n shuffles cannot resolve anything smaller. |
| `significant` | bool | Uncorrected: p < alpha **and** statistic ≥ 20 **and** ≥ 2 neighbours. |
| `significant_fdr` | bool | **The column to read.** `significant` narrowed by Benjamini-Hochberg across every row in the report. |
| `transform` | str | `difference`, or `log_ratio` for `pr`. **Read this before `shift`.** |
| `shift` | float | `after − before`. In canonical units for `difference`; in **natural-log units** for `log_ratio`. |
| `shift_ratio` | float | For `log_ratio` rows, the shift as a multiplicative factor. |
| `segment_before_mean`, `segment_after_mean` | float | Segment means of the difference series. |
| `min_support_at_break` | int | Neighbours behind **the break month**. This is the number the two-neighbour rule is about. |
| `n_neighbors_used` | int | Neighbours over the **whole** series. Larger, and not the safeguard — a single-neighbour break window once looked protected because only this was reported. |
| `n_months_min_support` | int | Months with at least the minimum support. |

### `consolidated/homogeneity_flags.csv`

The subset that flagged, in the shape that joins onto the gold table. One row
per flagged station × variable, earliest breakpoint winning. Adds
`flag` and `detected_by` (`snht_relative_monthly`); other columns as above.

### `consolidated/fallback_compatibility.csv`

One row per tested source switch. Section 1.6's four tests, reported whether
or not they passed, so a refusal can be reviewed rather than merely obeyed.

| Column | Type | Meaning |
|---|---|---|
| `canonical_station_id`, `variable` | str | The series. |
| `primary`, `fallback` | str | Alias keys. |
| `fallback_rank` | int | The fallback's priority position. |
| `candidate_days` | int | Days the primary lacks that the fallback offers. 0 means nothing to fill. |
| `compatible` | bool | Whether the splice was allowed. |
| `n_overlap` | int | Finite shared days the statistics rest on. |
| `bias` | float | `fallback − primary` mean, canonical units. |
| `dispersion_ratio` | float | σ(fallback) / σ(primary). |
| `wet_day_diff` | float | Absolute wet-day frequency difference. `pr` only; NaN otherwise. |
| `max_monthly_departure` | float | Largest monthly mean departure from `bias`, over months with at least 20 shared days. NaN when no month qualifies — the test could not run, and is recorded as a failure. |
| `reason` | str | Every failing check, or the passing summary. |
| `days_filled` | int | Days actually written to the gold table. 0 on refusal — the gap stays open. |

### `consolidated/qc_rejections.csv`

| Column | Meaning |
|---|---|
| `source`, `variable` | What was rejected. |
| `reason` | Why, quoting the provider's documentation where it is a provider flag. `reviewer exclusion (climatology screen)` for values removed by a `DECISION_exclude` answer. |
| `n_values` | Count. |

### `consolidated/climatology_screen.csv`

One row per selectable source record x variable reached. Diagnostic only; the
review workbook's `Series` sheet is built from it.

| Column | Type | Meaning |
|---|---|---|
| `alias`, `canonical_station_id`, `source`, `source_station_id`, `name`, `lat`, `lon` | | The record and the station it currently belongs to. |
| `variable` | str | Canonical variable. |
| `status` | str | `tested`, or why not: `no_reference`, `too_few_matched_days` (< 365), `undefined_departure`. **Read this first.** |
| `reference_key` | str | The reference point used: the record's own key, or another member of its station's group. |
| `n_matched_days` | int | Days both the record and the reference have. |
| `station_mean`, `reference_mean` | float | Means over those days only, canonical units. |
| `elevation_adjustment` | float | Temperature only: the reference moved to the station's height at 6.5 K/km. Diagnostic; no value is changed by it. |
| `departure` | float | Log ratio for `pr`, `sfcWind`, `gust` (`-inf` when the record's mean is zero); difference in canonical units otherwise. |
| `departure_units` | str | `log ratio`, or the variable's unit. |
| `robust_z` | float | (departure - median) / (1.4826 x MAD) over every tested record of the variable. |
| `tier` | str | `propose_exclusion` (abs z > 4), `review` (3 < abs z <= 4), `ok`. |
| `reporting_fraction` | float | Days with this variable / days the record reports anything. |
| `sampling_note` | str | Set where the provider's convention biases the sample -- GSOD gust. |
| `n_neighbors`, `neighbor_median_departure`, `departure_vs_neighbors`, `n_neighbors_same_direction` | | Other stations within 100 km, counted once each: how they depart, and how many are flagged the same way. |

### `consolidated/homogeneity_breakpoints.xlsx`

`Breakpoints`: every FDR-surviving break with station name, position, date,
transform, shift and support. `All_series`: the full report. A listing for a
person; nothing reads it back.

### `consolidated/source_contribution.csv`

Accepted-day counts, never percentages, so a sparse year cannot look complete.

| Column | Meaning |
|---|---|
| `variable`, `year`, `selected_source` | The grouping. |
| `is_fallback` | Whether these days were gap-fills. |
| `days` | Count of selected days. |

---

### `inventory/` and `manifests/`

| Path | What it is |
|---|---|
| `inventory/<source>_inventory.csv` | One row per station × variable a source offers, with access status and licence. |
| `inventory/coverage_report.md` | The pre-download coverage analysis. Its station grouping is a **coverage estimate** from proximity, never the crosswalk. |
| `manifests/source_releases.jsonl` | One line per acquisition: source, provider release, retrieval time, licence. |
| `manifests/<source>/<release>/requests.jsonl` | Every planned request. Credential-looking payload keys are redacted. |
| `manifests/<source>/<release>/snapshots.jsonl` | Every file fetched, with sha256 and byte count. Bronze is located from here, not by walking directories. |
| `manifests/provenance.json` | Policy, library versions, platform and per-source archive checksums for one build, plus `climatology_reference`: the reference files the climatology screen ran on and the screen it produced, each by file name, size and sha256 (never a local path), or `not used`. |
| `manifests/climatology_reference.json` | That reference record, written by `climatology` at the moment it reads the files; `provenance` copies it. |

### `review/` — inputs, not outputs

The pipeline reads these. It rebuilds them, but never loses an answer a person
wrote in them.

| Path | Decides |
|---|---|
| `station_merge_review.xlsx` | Whether two source records are one station, in `DECISION_merge` (`yes`/`no`). Read back from every sheet that shows the column, including `Already_merged` and `Beyond_radius`. Also whether a source record is excluded from selection, in the `Series` sheet's `DECISION_exclude` (`yes`/`no`; blank keeps it), with `Series_decisions` as the ledger. |
| `elevation_overrides.csv` | Which elevation is right where two archives disagree. Columns `canonical_station_id`, `elevation_m`, `reason`. Applied by the consumer **at read time**, so the archive stays reproducible from its snapshots. |

A large DEM-to-winner gap in a `reason` is evidence about the **coordinate**,
not the elevation: every override with a gap above 20 m in this archive sits at
a whole-arcminute coordinate, which is ±900 m on the ground. Read the reason
before trusting the row.
---

# Changing this repository

The most useful contribution is a new source adapter, so that is what most of
this section is about. The module map and the invariants above are what review
will hold a change to.

### Getting set up

```bash
pip install -e ".[dev]"
export STATION_ARCHIVE_ROOT=/path/with/several/GB/free   # not your checkout
python -m pytest tests/ -q          # no network, a few seconds
python -m station_archive.cli demo  # the whole chain on synthetic data
```

`STATION_ARCHIVE_ROOT` matters more than it looks. The archive is several GB of
provider data under redistribution restrictions, and the ignore rules exist to
keep it out of git. Point it somewhere outside your checkout.

### Adding a source

The registry exists so that this is one edit rather than six. If you find
yourself adding a source name to a second place, something has gone wrong —
say so in the pull request rather than working around it.

**1. Write the adapter** in `station_archive/sources/<name>.py`, subclassing
`StationSource` and implementing the four contract methods:

```python
class MySource(StationSource):
    name = "mysource"
    licence = "..."                    # what the provider actually says
    licence_url = "..."
    redistribution = "..."             # plain language; can this be republished?

    element_to_variable = {"TX": "tasmax", "RR": "pr"}   # provider name -> canonical
    element_units = {"TX": "degC_tenths", "RR": "mm_tenths"}  # provider's OWN units

    def release(self) -> str: ...       # prefer a version the provider publishes
    def discover(self, domain, variables, period) -> StationInventory: ...
    def plan(self, inventory, policy=None) -> list[DownloadRequest]: ...
    def _transfer(self, request, destination) -> None: ...
    def parse(self, snapshot) -> Iterator[RawObservation]: ...
```

Four rules the review will check:

- **`parse` returns provider values in provider units.** Do not convert. If the
  provider ships tenths of a degree, yield tenths and declare `degC_tenths`. An
  unlisted unit raises in `standardize` rather than passing through, which is the
  behaviour you want.
- **Preserve provider flags, do not act on them** — unless the provider's own
  documentation says the value is not an observation. GSOD's precipitation flags
  `I` and `H` are the model: `qc.py` quotes NOAA's readme for both.
- **Raise `SchemaError` on anything unexpected.** A provider that changes its
  columns must stop the run, not silently drop a field.
- **Decide nothing.** No identity, no ranking, no gap filling, no units. The
  adapter's job ends at "these are the provider's values".
- **Be exact about identifiers.** A cross-source identifier is the strongest
  evidence in the matching ladder, so a wrong one merges two unrelated
  stations. Derive one only when the provider's own convention says you may:
  GSOD's `wmo_from_station_id` returns `""` unless the USAF id ends in the `0`
  that marks a WMO station, because taking the first five digits regardless
  merged METHONI with CYCLADES ISLANDS 327 km away. If in doubt, publish no
  identifier and let the series test earn the link.

**2. Register it** in `station_archive/sources/registry.py`:

```python
"mysource": SourceSpec(
    name="mysource",
    adapter=MySource,
    has_station_series=True,   # can a per-station daily series be read from bronze?
    joins_crosswalk=True,      # admitted to the canonical station crosswalk?
    selectable=True,           # may the consolidator choose its values?
    notes="",                  # required if any flag above is False: say why
),
```

Those three flags are the only places the pipeline branches on source identity.
Getting one wrong used to be silent; now `Policy` refuses a configuration that
contradicts them at construction.

**3. Decide where it sits in the priority order**, in `policy.py`. This is a
scientific claim, not a formality, so it needs evidence in the comment: station
counts for coverage, and the provider's own documentation for semantics. The
existing `pr` ordering is the worked example — GSOD has the most stations and
ranks last, because its zero "includes a trace".

**4. Write fixtures and tests.** `tests/fixtures.py` holds verbatim excerpts of
real provider responses, so a format change fails a test against the format the
adapter was written for. Do not generate fixtures; paste real ones, trimmed, and
say when you captured them. Cover at minimum: inventory parsing, missing and
sentinel values, unit metadata, quality flags, and a schema change that must
raise.

**5. Run the suite and the demo.** If the demo still passes, the adapter has not
broken the contract.

### Changing a threshold

Everything in `policy.py` is a scientific decision, and changing one changes
what a rebuilt archive says. So:

- state what you measured, on what, in the comment beside the value;
- expect `tests/test_policy.py::test_default_policy_matches_the_published_archive`
  to fail, and update it deliberately;
- write the change up in the GitHub release notes, with the reasoning.

The measurement beside the value in `policy.py` is the durable record; the
release notes are how a reader finds out that it moved.

A threshold changed without a measurement behind it is the one change most
likely to be rejected. `standardize.GSOD_DAY_WINDOW_BIAS_CORRECTION` documents a
correction that was applied and then withdrawn precisely because the number was
borrowed from a different variable and a different source pair; read that comment
before proposing an adjustment of any kind.

### Pull requests

- Say what the change does to the archive's *output*, not only to the code. "No
  change to any value" is a useful and checkable claim.
- Keep the diff to the thing you set out to do. Tidying adjacent code in the same
  commit makes the substantive change harder to review.
- If you found a defect while working, a failing test that demonstrates it is
  worth more than a description of it.
- Docstrings here carry the *why*. A comment restating what the line does will be
  asked to go; a comment naming the measurement or the provider document behind a
  decision is the point.

### Reporting a problem in the archive itself

If a value, a merge or a flag looks wrong, the audit tables are designed to
answer why:

| Question | File |
|---|---|
| Why is this value here, and from which source? | `consolidated/observations_consolidated/` |
| What else could have been chosen? | `consolidated/non_selected_candidates/` |
| Why were these two stations merged, or not? | `canonical/duplicate_evidence.csv` |
| Why was this gap left open? | `consolidated/fallback_compatibility.csv` |
| Why was this value rejected? | `consolidated/qc_rejections.csv` |
| What flagged this series, and on what support? | `consolidated/homogeneity_breakpoints.csv` |
| Why are these two stations one, when nobody approved it? | `review/station_merge_review.xlsx`, sheet `Already_merged` |
| Why does this station have this elevation? | `review/elevation_overrides.csv` |
| What built this archive at all? | `manifests/provenance.json` |

Quoting the relevant row makes a report actionable in one step.


---

# Appendix: design requirements

Docstrings throughout this package cite numbered sections of
`RESEARCH_PLAN_AND_DESIGN.md`, the design document of the climate-downscaling
project this package was extracted from. That document is not part of this
repository, and it is not needed: this appendix states what each cited section
requires, so every citation in the code resolves to something a reader can
check.

Where the implementation departs from a requirement, or goes beyond it, that is
recorded under the section it belongs to. Those departures are the interesting
part — most were forced by measurements on real provider data rather than chosen
up front.

### 1.1 — Source survey and cautions

Which sources to consider, and what is known to be wrong with each.

**Requirements.** Treat GSOD as the existing reproducible baseline. Expect
GHCN-Daily to be itself a composite of merged sources, so overlap with the others
is normal rather than exceptional. Record that some ECA&D series are
non-downloadable. Record that C3S states no assessment of potential bias has been
made, and that multiple underlying sources can repeat the same physical record.

**In the code.** Each adapter's module docstring carries its provider's cautions,
and the licence and redistribution strings are recorded on every snapshot.
ECA&D stations advertised in metadata but absent from the public archive are
inventoried with `access_status = restricted_or_absent` rather than dropped, so
the gap is visible. C3S's bias caveat is recorded on its release and acted on
nowhere.

**Beyond the requirement.** The C3S redelivery caution turned out to be exact in
this domain: every C3S station carries a GHCN-D identifier. It is therefore
excluded from value selection entirely (`registry.SourceSpec.selectable = False`)
and kept as an audit cross-check.

---

### 1.2 — Discovery before download

Four questions must be answerable before anything is bulk-transferred:

1. How many genuinely new physical stations and station-years does each source
   add?
2. Which variables become denser?
3. Where do geographic gaps remain?
4. Which candidate records cannot be redistributed, or used outside
   non-commercial research?

The admission window is 1985–2005 — the historical overlap period for the model
this archive was built for. Longer records are retained for QC and homogeneity
diagnosis but must not leak future information into a historical calibration.
The domain is the training box plus a configurable synoptic buffer.

**In the code.** `inventory.py` answers all four into
`inventory/coverage_report.md`; `policy.py` holds the window, the domain and the
buffer; `cli.py` has no path that fetches without an inventory on disk first.
Downloads are not truncated to the window, because GHCN-D and ECA&D publish
whole-series files with no subsetting option, so the longer record costs nothing.

**A caveat the code states loudly.** Question 1 is a duplicate-detection
question, and duplicate detection is section 1.3's job. The coverage report
groups stations by 0.02° proximity, which is a *coverage estimate only*: it will
merge genuinely distinct nearby stations and separate a station that moved. The
report says so in its own text, and a test asserts that it does.

---

### 1.3 — Canonical station identity

Simple concatenation would inflate the apparent station count and leak duplicates
across validation folds. A matching ladder, in descending confidence:

1. exact WIGOS/WMO or documented cross-source identifier;
2. coincident coordinates plus compatible name and elevation;
3. near coordinates within a conservative tolerance, plus overlapping
   daily-series correlation and metadata review;
4. unresolved candidates remain separate and receive a duplicate-risk flag.

Coordinate tolerance alone is insufficient, because airports and nearby
climatological stations can be distinct. A relocation should create time-bounded
metadata segments under the same canonical history, **not** silent coordinate
averaging. Duplicates must never be averaged.

**In the code.** `identity.py` implements the ladder; thresholds live in
`policy.IdentityPolicy`. Metadata is *selected* from the alias with the most
observations, as a block, so the result is a position some archive actually
published; every alias keeps its own coordinates as its own segment in
`station_metadata_history.csv`.

**What the archive actually supported**, measured rather than assumed: no source
publishes a WIGOS identifier; ECA&D publishes no WMO number for any of its 1,463
domain stations, so tier 1 is unavailable to the largest source; C3S
`primary_station_id` is a GHCN-D identifier verbatim.

**Stricter than the requirement.** Tier 2 as written would allow a merge on
coincident coordinates plus compatible name and elevation with no series evidence
at all. This project forbids that: two records sharing no observation day are
permanently unmergeable, and the rule admits no override — not even an explicit
reviewer decision.

**Beyond the requirement.** Correlation measures covariation, not agreement, and
the two come apart: pairs 2.4 km apart were measured correlating at 0.9915 while
differing 1.3 °C every day. Every pair therefore also records mean absolute
difference, bias and an exact-agreement fraction, a merge requires agreement as
well as correlation, and accepted merges that disagree in value are flagged for
review.

**Two refinements forced by evidence.** The distance gate was widened from ~1 km
to ~3 km after it rejected `ecad:002346` / `ghcnd:BUM00015502`, identical to the
last decimal on all 3,210 shared days. And proven-identical series override the
elevation gate, after `ecad:010963` / `ghcnd:HRE00155411` were found at identical
coordinates agreeing exactly on 7,670 days while their archives differed by
113 m. The elevation is *not* corrected — section 1.4 says diagnose, do not
homogenise — so the conflict goes to `metadata_conflicts.csv` instead.

---

### 1.4 — Layers, QC, and the rule against silent correction

Store provider values and flags unchanged in a raw/bronze layer. Perform
conversions and project QC in a standardized/silver layer. The modelling/gold
layer should contain only accepted values plus transparent rejection reasons.

Retain provider QC flags and *layer* project flags rather than overwriting them.
Normalise units without destroying original units or precision. Preserve trace
precipitation separately from zero. Check internal consistency
(`tasmin ≤ tas ≤ tasmax`), repeated days and impossible persistence. Create
coverage and missingness summaries by station, variable, month and year.
**Diagnose breakpoints and inhomogeneity, but do not silently homogenise raw
observations.**

**In the code.** `bronze.py` reads; `standardize.py` converts and nothing else;
`qc.py` decides what is not an observation; rejections are counted by source,
variable and reason into `qc_rejections.csv`. `homogeneity.py` flags and never
alters, and `apply_flag_layer` refuses to overwrite an existing flag column.

**Where a requirement cannot be met.** GSOD cannot preserve trace precipitation
separately from zero: its own readme states that a reported zero includes a
trace, and the information is not recoverable from the file. That is recorded as a
known limitation of the source, and is why GSOD ranks last for `pr`.

**The QC rule that matters most here** is quoted from NOAA's documentation rather
than inferred: flag `I` means the station reported no precipitation data for the
day, and flag `H` is documented as incomplete data. 45.8 % of GSOD's
zero-precipitation rows in a 400-file sample carry `I`. Reading them as 0.0 mm
inflates dry-day frequency, so both are rejected as non-observations.

**Partially implemented.** `check_internal_consistency`, `check_repeated_days`
and `coverage_summary` exist and are tested, but no command runs them; they are
library API rather than part of a pipeline run. The internal-consistency check
reports violations and removes nothing, because which of three values to drop is
a policy question this project has not settled.

---

### 1.5 — The adapter contract and restartable transfer

```python
class StationSource:
    def discover(self, domain, variables, period) -> StationInventory: ...
    def plan(self, inventory) -> list[DownloadRequest]: ...
    def fetch(self, request, raw_directory) -> SourceSnapshot: ...
    def parse(self, snapshot) -> list[RawObservation]: ...
```

Every downloader must be restartable and idempotent: create a request manifest
**before** transfer; keep provider release/version, URL or API payload, retrieval
time, response checksum and licence; skip a file only when its checksum matches;
use bounded retries and provider rate limits; never edit a raw snapshot in place.
Credentials belong in environment configuration, never in a manifest or the
repository. Test each adapter against a tiny fixed station/date fixture before
bulk acquisition. An upstream schema change must stop loudly rather than silently
dropping columns.

**The architecture guardrail this section sets:** source adapters fetch and parse
provider data; they do not decide station identity or source priority.

**In the code.** `sources/base.py` holds the contract and the shared provenance
behaviour, so an adapter cannot opt out of it. `net.py` implements the transfer
rules; `manifests.py` writes the records and redacts any credential-looking
payload key. Every parser raises `SchemaError` on an unexpected structure, and
`tests/fixtures.py` holds verbatim provider excerpts so a format change fails a
test against the format the adapter was written for.

**Two documented deviations.** `parse()` returns an iterator rather than a list,
because materialising every value in the domain as a dataclass at once would need
tens of millions of objects. And `plan()` takes an optional policy, supplying the
fetch-year window and, for API sources, the request area — the only policy an
adapter sees, and a narrower channel than reading a global.

**One exception to the manifest rule**, stated because it is real: the C3S
adapter's `discover()` retrieves a pilot year before any manifest is written,
because C3S publishes no station metadata to query and fetching one year is the
only honest way to answer section 1.2's questions.

---

### 1.6 — Source selection and gap filling

Do not encode one universal claim that a portal is always best. Store a priority
list **per variable**, because coverage and semantics differ. The plan's own
starting order is explicitly "a research hypothesis, not the final order" — to be
confirmed from the Step 1 coverage/QC pilot.

The selection procedure:

1. collect only values that pass mandatory provider and project QC;
2. sort candidates by configured source rank, then by declared tie-breakers;
3. choose the highest-ranked accepted value and record `selected_source`,
   `source_rank`, `selection_reason` and the raw-record identifier;
4. if that value is missing or rejected, move down the priority list and label
   the accepted record as a fallback;
5. retain all non-selected candidates in an audit table rather than deleting or
   averaging them;
6. leave the canonical value missing when no candidate passes QC.

Before allowing a source switch, compare overlapping records by month for bias,
dispersion, wet-day frequency and discontinuities. A failed compatibility test
leaves a gap or creates a time-bounded station segment; it does not justify blind
splicing. **Do not interpolate missing station observations in Step 1.**

Two meanings of "gap filling" must not be conflated: *same-station fallback*
fills a missing day from a lower-priority alias of the same physical station;
*network expansion* admits a different physical station for coverage, which is
not gap filling at all and fills nobody's row.

**In the code.** `consolidate.py` implements all six steps and the four
compatibility checks; `policy.FallbackPolicy` holds the tolerances;
`non_selected_candidates/` is the audit table.

**Where the pilot contradicted the hypothesis.** The hypothesis ranked `c3s_land`
first for `hurs` and `gust`, but it carries neither and redelivers GHCN-D, so it
is excluded from selection. `hurs` has exactly one source in this domain. And
`gsod` was ranked above `ghcnd` for `pr` on coverage (203 stations against 155) —
but GSOD's zero includes a trace and 46 % of its zero rows are non-observations,
so semantics win and it ranks last.

**Why no archive-wide offset stands in for the compatibility test.** For `tas`
with an ECA&D primary and a GSOD fallback, the measured overlap bias runs from
−1.60 to +0.58 °C across station pairs. The spread straddles zero, so every
switch is tested at the station it is about to be made at. A single archive-wide
correction was once applied on a median borrowed from a different variable and a
different source pair, and withdrawn; see
`standardize.GSOD_DAY_WINDOW_BIAS_CORRECTION`.

---

### 1.7 — Durable tables

The archive keeps, as durable records: `source_releases` (source, version,
retrieval time, licence, checksum and request manifest), `station_aliases`
(canonical ID, source ID, matching evidence) and `station_metadata_history`
(coordinates, elevation and name with effective dates).

**In the code.** `manifests.py` owns the first; `identity.py` writes the other
two, plus `duplicate_evidence.csv` and `metadata_conflicts.csv`.

**Beyond the requirement.** `manifests/provenance.json`, written by the
`provenance` command, records what built the archive *as a whole*: the policy in
force, each provider release with its checksum, and the library versions. The
per-request manifests say how each file arrived; nothing said what the archive
was produced by.

---

### 1.8 — Reproducibility

Re-running consolidation from unchanged snapshots must produce the identical
archive checksum.

**In the code.** `manifests.archive_checksum()` is computed from sorted
(key, checksum) pairs, so it is independent of download order. Candidate
generation, group representatives, metadata tie-breaks and fold assignment are
all deterministic, and tests assert that a second run reproduces the first
frame-for-frame. The vectorised unit conversions are written to be bit-identical
to the scalar formulas they replaced, so a rebuilt archive does not differ in the
last bits of its values.

---

### Architecture guardrails, collected

These appear across several sections and are the rules most worth not breaking:

- Source adapters fetch and parse; they do not decide identity or priority.
- Scientific policy belongs in versioned configuration, separate from download,
  modelling, gridding and adjustment code.
- The fold manifest is created once from canonical identities; model scripts may
  read it but may not invent their own split. **That manifest is built in the
  consuming project, not here** — see "What this repository does not do" in the
  README. The archive's part of the bargain is to publish the canonical
  identities and, crucially, the duplicates it could NOT resolve, so a split can
  keep an unresolved pair together.
- Diagnose, do not silently homogenise.
- Never average two sources, and never average duplicate coordinates.

The **Invariants** section at the top of this file restates these as the seven
the tests enforce.
