"""
cli.py
======
Command-line entry point for station-archive acquisition.

USAGE
-----
  # 1. Discovery first (plan section 1.2): what does each source offer?
  python -m station_archive.cli discover --sources ghcnd ecad gsod

  # 2. Read the report before transferring anything.
  python -m station_archive.cli report

  # 3. Fetch, one source at a time, restartable and idempotent.
  python -m station_archive.cli fetch --sources ghcnd

  # Show where the archive lives and whether the drive has room.
  python -m station_archive.cli status

  # Record exactly what built this archive.
  python -m station_archive.cli provenance

  # Settle elevation disagreements between archives against a DEM.
  python -m station_archive.cli dem

  # The quality-control pipeline, in two halves around one review sitting:
  #   prepare -- crosswalk, climatology screen, review workbook (merges and
  #              series exclusions in one place);
  #   apply   -- the reviewer's decisions: crosswalk, elevations, selection
  #              with exclusions, breakpoint flags, provenance.
  python -m station_archive.cli prepare --reference cerra_stations.parquet \
      --reference-static geo_stations.parquet
  python -m station_archive.cli apply

  # Run the whole chain on synthetic data, with no network and no credentials.
  python -m station_archive.cli demo

The three-command shape is the plan's discovery-before-download rule made
operational: 'fetch' plans from an inventory that 'discover' already wrote, so
there is no path that downloads in bulk without a coverage record first.

POLICY
------
Every command takes the scientific policy as arguments rather than reading a
global, so this tool is not tied to the domain it was written for. They go
AFTER the subcommand:

  station-archive discover --domain 36 44 -10 4 --buffer 2 --period 1990 2010
  station-archive homogeneity --variables tas pr --permutations 500
  station-archive consolidate --policy iberia.json

Omitted, each falls back to the policy this archive was built with. A policy
file is an override list, so changing region is three lines rather than a fork.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from station_archive import (
    bronze,
    climatology as climatology_module,
    consolidate as consolidate_module,
    dem as dem_module,
    homogeneity as homogeneity_module,
    identity as identity_module,
    manifests,
    inventory as inventory_module,
    review as review_module,
    settings,
)
from station_archive.policy import DEFAULT_POLICY, Policy
from station_archive.sources import registry
from station_archive.sources.base import StationSource
from station_archive.types import Domain, StationInventory

# Sources are declared once, in the registry. The CLI holds no second list.
DEFAULT_SOURCES = registry.series_sources()

# Layout of the CERRA-at-stations cache the downscaling project writes
# (derived/step2_cache/cerra_stations_expanded.parquet): one row per point and
# day, keyed by the canonical station id of the crosswalk it was built for.
# Its values are already in canonical units -- checked on the cache itself:
# degC, mm/day, percent and m/s -- so the screen converts nothing.
CERRA_CACHE_COLUMNS = {
    "tas": "cerra_tmean",
    "tasmin": "cerra_tmin",
    "tasmax": "cerra_tmax",
    "pr": "cerra_precip",
    "hurs": "cerra_rh",
    "sfcWind": "cerra_wind",
    "gust": "cerra_gust",
}
CERRA_CACHE_ID_COLUMN = "id"
CERRA_CACHE_DATE_COLUMN = "date"
# The reference orography, from the companion static table
# (geo_stations_expanded.parquet), for the temperature lapse adjustment.
CERRA_STATIC_ELEVATION_COLUMN = "cerra_elev_m"


def review_workbook_path() -> Path:
    """The one workbook every human decision in the pipeline lives in."""
    return settings.review_dir() / "station_merge_review.xlsx"


def build_source(name: str) -> StationSource:
    try:
        return registry.adapter_for(name)
    except registry.UnknownSourceError as exc:
        raise SystemExit(str(exc)) from exc


def inventory_path(source: str) -> Path:
    return settings.inventory_dir() / f"{source}_inventory.csv"


def policy_from_args(args: argparse.Namespace) -> Policy:
    """
    Build the run's policy: a file if given, then any explicit overrides.

    Overrides are applied on top of the file so a one-off can be expressed on
    the command line without editing the file, and the result is recorded by
    the provenance command exactly as it was used.
    """
    from dataclasses import replace

    policy = (
        Policy.from_json(Path(args.policy)) if getattr(args, "policy", None)
        else DEFAULT_POLICY
    )
    changes: dict = {}
    if getattr(args, "domain", None):
        lat_min, lat_max, lon_min, lon_max = args.domain
        buffer_deg = (
            args.buffer if getattr(args, "buffer", None) is not None
            else policy.domain.buffer_deg
        )
        changes["domain"] = Domain(lat_min, lat_max, lon_min, lon_max, buffer_deg)
    elif getattr(args, "buffer", None) is not None:
        changes["domain"] = replace(policy.domain, buffer_deg=args.buffer)
    if getattr(args, "period", None):
        changes["period"] = tuple(args.period)
    if getattr(args, "fetch_years", None):
        changes["fetch_years"] = tuple(args.fetch_years)
    if getattr(args, "variables", None):
        wanted = tuple(args.variables)
        changes["variables"] = wanted
        # Keep the priority consistent with the variable set, or construction
        # rightly refuses it.
        changes["source_priority"] = {
            v: o for v, o in policy.source_priority.items() if v in wanted
        }
    try:
        return replace(policy, **changes) if changes else policy
    except (ValueError, registry.PolicyConsistencyError) as exc:
        raise SystemExit(f"invalid policy: {exc}") from exc


def command_status(args: argparse.Namespace) -> int:
    import os

    policy = policy_from_args(args)
    root = settings.archive_root()
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    from_env = os.environ.get(settings.ARCHIVE_ROOT_ENV_VAR)
    print(f"archive root : {root}")
    print(
        "  resolved by: "
        + (
            settings.ARCHIVE_ROOT_ENV_VAR if from_env
            else f"default (set {settings.ARCHIVE_ROOT_ENV_VAR} to choose)"
        )
    )
    print(f"free space   : {usage.free / 1024**3:.2f} GB")
    print(f"domain       : {policy.domain}")
    print(f"window       : {policy.period[0]}-{policy.period[1]}")
    print(f"fetch years  : {policy.fetch_years[0]}-{policy.fetch_years[1]}")
    print(f"variables    : {', '.join(policy.variables)}")
    print("")
    for source in sorted(registry.all_sources()):
        path = inventory_path(source)
        state = f"{len(StationInventory.read_csv(path))} rows" if path.exists() else "not discovered"
        print(f"  {source:10s} {state}")
    print("")
    releases = manifests.read_releases()
    print(f"recorded releases: {len(releases)}")
    for record in releases[-5:]:
        print(f"  {record['source']:10s} {record['release']}  {record['retrieved_at']}")
    return 0


def command_discover(args: argparse.Namespace) -> int:
    policy = policy_from_args(args)
    for name in args.sources:
        source = build_source(name)
        print(f"[{name}] discovering...")
        result = source.discover(policy.domain, policy.variables, policy.period)
        path = inventory_path(name)
        result.to_csv(path)
        manifests.record_release(source.source_release())
        print(
            f"[{name}] {len(result)} station-variable rows, "
            f"{result.n_stations()} stations -> {path}"
        )
    return command_report(args)


def command_report(args: argparse.Namespace) -> int:
    policy = policy_from_args(args)
    inventories = inventory_module.load_inventories(sorted(registry.all_sources()))
    if not inventories:
        print("no inventories found; run 'discover' first")
        return 1
    frame = inventory_module.merge(inventories)
    path = inventory_module.write_report(frame, baseline=args.baseline, policy=policy)
    print(f"coverage report -> {path}")
    print("")
    print(
        inventory_module.coverage_report(
            frame, baseline=args.baseline, policy=policy
        )
    )
    return 0


def command_fetch(args: argparse.Namespace) -> int:
    policy = policy_from_args(args)
    for name in args.sources:
        path = inventory_path(name)
        if not path.exists():
            print(f"[{name}] no inventory at {path}; run 'discover' first")
            return 1

        source = build_source(name)
        stations = StationInventory.read_csv(path)
        requests = source.plan(stations, policy)

        if args.limit:
            requests = requests[: args.limit]

        print(f"[{name}] {len(requests)} requests planned")
        if args.dry_run:
            for request in requests[:20]:
                print(f"  {request.key} -> {request.relative_path}")
            if len(requests) > 20:
                print(f"  ... and {len(requests) - 20} more")
            continue

        source.fetch_all(requests)
    return 0


def command_identity(args: argparse.Namespace) -> int:
    """
    Build the canonical station crosswalk from the discovered inventories.

    Reads bronze series only for candidate pairs, so the cost scales with the
    number of near-coincident cross-source stations rather than the size of
    the archive.
    """
    policy = policy_from_args(args)
    inventories = {}
    for name in args.sources:
        path = inventory_path(name)
        if not path.exists():
            print(f"[identity] no inventory for {name}; run 'discover' first")
            return 1
        inventories[name] = StationInventory.read_csv(path)

    decisions = {}
    if args.decisions:
        decisions = identity_module.load_decisions(args.decisions)
        print(f"[identity] loaded {len(decisions)} reviewer decisions from {args.decisions}")

    loader = bronze.SeriesLoader(period=policy.period)
    crosswalk = identity_module.build_crosswalk(
        inventories, loader, progress=True, decisions=decisions, policy=policy
    )

    out = settings.canonical_dir()
    out.mkdir(parents=True, exist_ok=True)
    crosswalk.aliases.to_csv(out / "station_aliases.csv", index=False)
    crosswalk.metadata_history.to_csv(out / "station_metadata_history.csv", index=False)
    crosswalk.evidence.to_csv(out / "duplicate_evidence.csv", index=False)

    # One authoritative position per canonical station, from the alias with
    # the most observations.  Counted only for merged groups, so this reads
    # the merged stations rather than the whole archive.
    canonical = identity_module.resolve_canonical_metadata(
        crosswalk.aliases, loader, policy=policy
    )
    canonical.to_csv(out / "canonical_stations.csv", index=False)

    # Written after the canonical metadata, because each conflict records which
    # coordinate the archive went on to use. A diagnostic that states a problem
    # without stating its resolution invites someone to fix what is settled.
    conflicts = identity_module.annotate_conflicts(
        crosswalk.conflicts, crosswalk.aliases, canonical
    )
    conflicts.to_csv(out / "metadata_conflicts.csv", index=False)

    print("")
    print(f"[identity] source stations : {crosswalk.n_aliases()}")
    print(f"[identity] canonical stations: {crosswalk.n_canonical()}")
    collapsed = crosswalk.n_aliases() - crosswalk.n_canonical()
    print(f"[identity] duplicates reconciled: {collapsed}")
    risk = int(crosswalk.aliases["duplicate_risk"].sum())
    print(f"[identity] duplicate-risk flags: {risk}")
    manual = crosswalk.evidence[
        crosswalk.evidence["reason"].str.startswith("reviewer decision", na=False)
    ] if not crosswalk.evidence.empty else crosswalk.evidence
    if len(manual):
        merged_by_hand = int(manual["accepted"].sum())
        print(f"[identity] reviewer overrides applied: {len(manual)} "
              f"({merged_by_hand} merged, {len(manual) - merged_by_hand} unmerged)")
    if not crosswalk.evidence.empty:
        accepted = int(crosswalk.evidence["accepted"].sum())
        print(f"[identity] pairs tested: {len(crosswalk.evidence)} "
              f"(accepted {accepted}, rejected {len(crosswalk.evidence) - accepted})")
        print("")
        print("[identity] merges by tier:")
        merged = crosswalk.evidence[crosswalk.evidence["accepted"]]
        if not merged.empty:
            for tier_name, n in merged["tier_name"].value_counts().items():
                print(f"    {tier_name}: {n}")
    review = int(crosswalk.aliases["agreement_review"].sum())
    if review:
        print(f"[identity] merges flagged for value-agreement review: {review}")
    if not crosswalk.conflicts.empty:
        print("")
        print(f"[identity] METADATA CONFLICTS: {len(crosswalk.conflicts)} pairs whose "
              f"series are identical but whose metadata disagrees")
        for _, row in conflicts.iterrows():
            print(f"    {row['left']} / {row['right']}: {row['conflict']} "
                  f"({row['distance_km']} km, {row['elevation_diff_m']} m elevation) "
                  f"identical on {row['exact_fraction']:.0%} of {row['n_overlap_days']} days")
            print(f"        -> using {row['coordinates_used_from']} "
                  f"({row['coordinates_used_side']}, "
                  f"{row['coordinates_used_n_obs']:.0f} observations)")
        print("")
        print(f"    rule: {identity_module.COORDINATE_RULE}")
    if loader.misses:
        print(f"[identity] {len(loader.misses)} stations had no readable snapshot")
    print("")
    multi = canonical[canonical["n_aliases"] > 1]
    print(f"[identity] canonical metadata resolved for {len(canonical)} stations "
          f"({len(multi)} needed a contest)")
    if not multi.empty:
        print("[identity] metadata winner by source:")
        for source, n in multi["metadata_source"].str.split(":").str[0].value_counts().items():
            print(f"    {source}: {n}")
    print("")
    print(f"[identity] wrote -> {out}")
    return 0


class _BufferedParquetWriter:
    """
    Buffer DataFrames in memory and flush to numbered parquet parts once a
    row threshold is reached, releasing memory between flushes.

    The consolidated table (and its non-selected-candidates counterpart) run
    to tens of millions of rows with string columns; holding either fully in
    memory before writing would cost more than the values themselves.
    """

    def __init__(self, parts_dir: Path, flush_rows: int = 2_000_000) -> None:
        for stale in parts_dir.glob("part-*.parquet"):
            stale.unlink()
        parts_dir.mkdir(parents=True, exist_ok=True)
        self._dir = parts_dir
        self._flush_rows = flush_rows
        self._buffer: list[pd.DataFrame] = []
        self._buffered_rows = 0
        self._part = 0
        self.total_rows = 0

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self._buffer.append(frame)
        self._buffered_rows += len(frame)
        self.total_rows += len(frame)
        if self._buffered_rows >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        pd.concat(self._buffer, ignore_index=True).to_parquet(
            self._dir / f"part-{self._part:04d}.parquet", index=False
        )
        self._part += 1
        self._buffer = []
        self._buffered_rows = 0


def command_consolidate(args: argparse.Namespace) -> int:
    """
    Select one value per canonical station, date and variable (plan 1.6).

    Stations are processed one at a time across all variables so each
    station's bronze files are parsed once, then released -- consolidating
    variable-by-variable instead would re-read every GSOD station-year file
    seven times.
    """
    policy = policy_from_args(args)
    canonical_dir = settings.canonical_dir()
    alias_path = canonical_dir / "station_aliases.csv"
    if not alias_path.exists():
        print(f"[consolidate] no crosswalk at {alias_path}; run 'identity' first")
        return 1

    aliases = pd.read_csv(alias_path, dtype={"source_station_id": str})
    station_ids = sorted(aliases["canonical_station_id"].unique())
    if args.limit:
        station_ids = station_ids[: args.limit]

    out = settings.consolidated_dir()
    out.mkdir(parents=True, exist_ok=True)

    exclusions: dict = {}
    if getattr(args, "decisions", None):
        exclusions = climatology_module.exclusions_from_decisions(
            climatology_module.load_exclusion_decisions(Path(args.decisions))
        )
        print(f"[consolidate] {len(exclusions)} series excluded by reviewer decision "
              f"in {Path(args.decisions).name}")

    loader = bronze.SeriesLoader(period=policy.period)
    compatibility: list[dict] = []
    # Reviewer exclusions are counted with the QC rejections: a value removed
    # from the gold table by a decision is as much a rejection as one removed
    # by a flag, and the audit must say so (plan 1.4, "transparent rejection
    # reasons").
    excluded_counts: dict[tuple[str, str, str], int] = {}

    values_writer = _BufferedParquetWriter(out / "observations_consolidated")
    # Non-selected candidates (plan 1.6 point 5): every value that existed
    # for a station-day-variable but lost to a higher-priority source or an
    # incompatible-fallback refusal, kept rather than discarded.
    non_selected_writer = _BufferedParquetWriter(out / "non_selected_candidates")

    # Tracked as we go, so the summary needs no second pass over the table.
    stations_written: set[str] = set()

    for index, canonical_id in enumerate(station_ids, start=1):
        group = aliases[aliases["canonical_station_id"] == canonical_id]
        for variable in policy.source_priority:
            result = consolidate_module.consolidate_station_variable(
                canonical_id, group, variable, loader, policy=policy,
                exclusions=exclusions,
            )
            if exclusions and not result.non_selected.empty:
                reviewed = result.non_selected[
                    result.non_selected["rejection_reason"].str.startswith(
                        "reviewer exclusion", na=False
                    )
                ]
                for (src, var), n in reviewed.groupby(["source", "variable"]).size().items():
                    key = (src, var, "reviewer exclusion (climatology screen)")
                    excluded_counts[key] = excluded_counts.get(key, 0) + int(n)
            if not result.values.empty:
                stations_written.add(canonical_id)
            values_writer.append(result.values)
            non_selected_writer.append(result.non_selected)
            compatibility.extend(result.compatibility)
        loader.clear()

        if index % 100 == 0:
            print(f"[consolidate] {index}/{len(station_ids)} stations, {values_writer.total_rows} values")

    values_writer.flush()
    non_selected_writer.flush()

    compat = pd.DataFrame(compatibility)
    compat.to_csv(out / "fallback_compatibility.csv", index=False)

    # "only accepted values plus transparent rejection reasons" (plan 1.4)
    rejections = pd.DataFrame(
        [
            {"source": src, "variable": var, "reason": reason, "n_values": n}
            for (src, var, reason), n in sorted(
                {**loader.rejections, **excluded_counts}.items()
            )
        ]
    )
    rejections.to_csv(out / "qc_rejections.csv", index=False)

    # Aggregate the audit one part at a time. Reading every part back into a
    # single frame would undo the whole point of the buffered writer above:
    # ~9.6 million rows with string columns, rebuilt in memory for a group-by.
    # Each part's aggregate is a few hundred rows, and everything the summary
    # below reports is derived from those aggregates plus the writer's counters.
    written = sorted((out / "observations_consolidated").glob("part-*.parquet"))
    audit = consolidate_module.merge_source_contributions(
        consolidate_module.source_contribution(pd.read_parquet(part))
        for part in written
    )
    audit.to_csv(out / "source_contribution.csv", index=False)
    print(f"[consolidate] wrote {len(written)} parquet parts")
    print(f"[consolidate] non-selected candidates retained: {non_selected_writer.total_rows} "
          f"-> {out / 'non_selected_candidates'}")
    if not rejections.empty:
        print("")
        print(f"[consolidate] QC rejected {int(rejections['n_values'].sum())} values:")
        for _, row in rejections.iterrows():
            print(f"    {row['source']}/{row['variable']}: {row['n_values']} - {row['reason']}")

    print("")
    print(f"[consolidate] canonical values : {values_writer.total_rows}")
    print(f"[consolidate] stations         : {len(stations_written)}")
    if not audit.empty:
        print("")
        print("[consolidate] values by variable and source:")
        pivot = audit.pivot_table(
            index="variable", columns="selected_source", values="days",
            aggfunc="sum", fill_value=0,
        )
        print(pivot.to_string())
        fallback = int(audit.loc[audit["is_fallback"], "days"].sum())
        print("")
        print(f"[consolidate] fallback values   : {fallback} "
              f"({fallback / max(values_writer.total_rows, 1):.2%})")
    if not compat.empty:
        tested = compat[compat["candidate_days"] > 0]
        refused = tested[~tested["compatible"]]
        print(f"[consolidate] fallback switches tested : {len(tested)}")
        print(f"[consolidate] refused as incompatible  : {len(refused)} "
              f"(gaps left open, not spliced)")
        if not refused.empty:
            print("[consolidate] top refusal causes:")
            for cause, n in refused["reason"].str.split(";").str[0].str.split(" ").str[0].value_counts().head(5).items():
                print(f"    {cause}: {n}")
    print("")
    print(f"[consolidate] wrote -> {out}")
    return 0


def command_homogeneity(args: argparse.Namespace) -> int:
    """
    Diagnose candidate breakpoints per station/variable (plan 1.4:
    "diagnose breakpoints/inhomogeneity, but do not silently homogenize raw
    observations"). Reads the already-consolidated gold table; writes a
    diagnostic report only -- no value in the archive is changed.
    """
    policy = policy_from_args(args)
    canonical_dir = settings.canonical_dir()
    consolidated_dir = settings.consolidated_dir()
    stations_path = canonical_dir / "canonical_stations.csv"
    parts_dir = consolidated_dir / "observations_consolidated"

    if not stations_path.exists() or not parts_dir.exists():
        print(f"[homogeneity] no consolidated archive at {consolidated_dir}; run 'consolidate' first")
        return 1

    stations_all = pd.read_csv(stations_path)
    stations_meta = stations_all[["canonical_station_id", "lat", "lon"]]
    parts = sorted(parts_dir.glob("part-*.parquet"))
    # Four columns, not eleven. run_report reads only these, and the gold table
    # runs to ~9.7 million rows with string columns: measured on this archive,
    # the full read is ~1.7 GB resident against ~0.5 GB for the slice. Reading
    # it whole reinstated exactly the cost _BufferedParquetWriter and
    # merge_source_contributions were written to avoid.
    values = pd.concat(
        [
            pd.read_parquet(
                part, columns=["canonical_station_id", "date", "variable", "value"]
            )
            for part in parts
        ],
        ignore_index=True,
    )

    variables = args.variables or list(policy.source_priority)
    print(f"[homogeneity] testing {len(variables)} variable(s) across "
          f"{stations_meta['canonical_station_id'].nunique()} candidate stations …")

    report = homogeneity_module.run_report(
        values, stations_meta, variables,
        n_permutations=args.permutations, alpha=args.alpha,
    )

    out_path = consolidated_dir / "homogeneity_flags.csv"
    report_path = consolidated_dir / "homogeneity_breakpoints.csv"
    report.to_csv(report_path, index=False)

    # The full report is the diagnostic record; the flag layer is the part
    # that joins onto the gold table (plan 1.4: "layer project flags").
    layer = homogeneity_module.build_flag_layer(report)
    layer.to_csv(out_path, index=False)

    # The same flags for a person: where each break is and at which station.
    # A listing, never an input -- nothing reads it back, and the flags stay
    # diagnostic.
    listing_path = review_module.write_workbook(
        consolidated_dir / "homogeneity_breakpoints.xlsx",
        {
            "Breakpoints": homogeneity_module.flagged_breakpoint_table(
                report, stations_all
            ),
            "All_series": report,
        },
    )

    print(f"[homogeneity] {len(report)} station/variable series reached")
    if not report.empty:
        print("[homogeneity] outcome by status:")
        for status, n in report["status"].value_counts().items():
            print(f"    {status}: {n}")
        tested = report[report["status"] == "tested"]
        raw_flagged = int(report["significant"].sum())
        flagged = report[report["significant_fdr"]]
        print(f"[homogeneity] {len(tested)} actually tested; {raw_flagged} significant "
              f"uncorrected (p < {args.alpha} and SNHT >= "
              f"{homogeneity_module.MIN_SNHT_STATISTIC:g}); {len(flagged)} survive "
              f"Benjamini-Hochberg FDR correction -- read this number")
        if not flagged.empty:
            print("[homogeneity] flagged by variable (of those tested):")
            by_variable = tested.groupby("variable")["significant_fdr"].agg(["size", "sum"])
            for variable, row in by_variable.iterrows():
                print(f"    {variable}: {int(row['sum'])}/{int(row['size'])} "
                      f"({row['sum'] / max(row['size'], 1):.0%})")
            # Shifts are in each variable's own monthly units (degC, m/s), so
            # they are ranked within a variable, never across them. Variables
            # tested multiplicatively carry a log-unit shift, which is printed
            # as the factor it represents rather than as a raw number whose
            # units differ silently from the rest of the column.
            print("[homogeneity] largest FDR-surviving shift per variable:")
            for variable, group in flagged.groupby("variable"):
                top = group.reindex(group["shift"].abs().sort_values(ascending=False).index).head(3)
                for _, row in top.iterrows():
                    if row.get("transform") == "log_ratio":
                        magnitude = f"x{row['shift_ratio']:.2f}"
                    else:
                        magnitude = f"shift {row['shift']:+.2f}"
                    print(f"    {variable}: {row['canonical_station_id']} "
                          f"{magnitude} at {row['breakpoint_date']} "
                          f"(SNHT={row['statistic']:.0f}, n_neighbors={row['n_neighbors_used']})")
    print("")
    print(f"[homogeneity] report     -> {report_path}")
    print(f"[homogeneity] flag layer -> {out_path} ({len(layer)} flags)")
    print(f"[homogeneity] listing    -> {listing_path}")
    print("[homogeneity] diagnostic only -- no value in the archive was changed")
    return 0


def command_review(args: argparse.Namespace) -> int:
    """
    Build the human-review workbook for unresolved station pairs (plan 1.3).

    Tier 4 gives unresolved candidates a duplicate-risk flag, which is a
    request for review rather than a verdict.  This turns those flags into
    something a person can adjudicate.
    """
    policy = policy_from_args(args)
    canonical_dir = settings.canonical_dir()
    evidence_path = canonical_dir / "duplicate_evidence.csv"
    if not evidence_path.exists():
        print(f"[review] no crosswalk at {evidence_path}; run 'identity' first")
        return 1

    evidence = pd.read_csv(evidence_path)
    aliases = pd.read_csv(
        canonical_dir / "station_aliases.csv", dtype={"source_station_id": str}
    )
    conflicts_path = canonical_dir / "metadata_conflicts.csv"
    conflicts = pd.read_csv(conflicts_path) if conflicts_path.exists() else pd.DataFrame()
    # metadata_conflicts.csv identifies stations by ID only; give it the same
    # name/coordinate columns every other sheet carries.
    conflicts = review_module.add_station_identity(
        conflicts, review_module._station_lookup(aliases)
    )

    # Decisions already recorded in the workbook being replaced. A rebuild is
    # routine -- the crosswalk changed, so the pairs did -- and it must not cost
    # the adjudication that went into the previous one.
    workbook = review_workbook_path()
    previous: dict = {}
    if workbook.exists():
        try:
            previous = identity_module.load_decisions(workbook)
        except Exception as exc:
            print(f"[review] could not read decisions from {workbook.name}: {exc}")
        if previous:
            print(f"[review] carrying forward {len(previous)} decision(s) already made")

    loader = bronze.SeriesLoader(period=policy.period)
    sheets = review_module.build_review(
        evidence, aliases, loader, max_distance_km=args.max_distance_km, progress=True
    )
    sheets = review_module.carry_forward_decisions(sheets, previous)

    ordered = {
        "Summary": review_module.summary_sheet(sheets, conflicts),
        "Review": sheets["Review"],
        "Already_merged": sheets["Already_merged"],
        "Per_variable": sheets["Per_variable"],
        "Metadata_conflicts": conflicts,
        "Beyond_radius": sheets["Beyond_radius"],
        "Never_merge_rule": sheets["Never_merge_rule"],
    }
    ordered = review_module.carry_forward_decisions(ordered, previous)
    if "Decisions" in sheets and "Decisions" not in ordered:
        ordered["Decisions"] = sheets["Decisions"]

    # The climatology screen's proposals, so merges and series exclusions are
    # answered in one sitting. Placed after the merge sheets; the ledger last.
    # Written even without a screen: the ledger is the only place an exclusion
    # decision survives a rebuild, and dropping it would silently undo it.
    screen_path = settings.consolidated_dir() / "climatology_screen.csv"
    if screen_path.exists():
        screen = pd.read_csv(screen_path, dtype={"source_station_id": str})
    else:
        screen = pd.DataFrame()
        print(f"[review] no climatology screen at {screen_path}; "
              f"run 'climatology' to fill the Series sheet")
    previous_exclusions: dict = {}
    if workbook.exists():
        try:
            previous_exclusions = climatology_module.load_exclusion_decisions(workbook)
        except Exception as exc:
            print(f"[review] could not read series decisions from {workbook.name}: {exc}")
    series, ledger = review_module.carry_forward_exclusions(
        review_module.series_sheet(screen), previous_exclusions
    )
    rebuilt: dict = {}
    for name, frame in ordered.items():
        rebuilt[name] = frame
        if name == "Already_merged":
            rebuilt[climatology_module.SERIES_SHEET] = series
    rebuilt.setdefault(climatology_module.SERIES_SHEET, series)
    rebuilt[climatology_module.SERIES_LEDGER_SHEET] = ledger
    # The instructions come first, so the workbook opens on them.
    ordered = {review_module.INSTRUCTIONS_SHEET: review_module.instructions_sheet(), **rebuilt}
    counts = series["RECOMMENDATION"].value_counts()
    print(f"[review] series sheet: {int(counts.get('EXCLUDE', 0))} proposed for "
          f"exclusion, {int(counts.get('REVIEW', 0))} for review; "
          f"{len(previous_exclusions)} series decision(s) carried forward")
    path = review_module.write_workbook(workbook, ordered)
    kept = sum(
        int((frame["DECISION_merge"].astype(str).str.strip() != "").sum())
        for frame in ordered.values()
        if "DECISION_merge" in getattr(frame, "columns", [])
    )
    if kept:
        print(f"[review] {kept} decision(s) preserved in the rebuilt workbook")

    print("")
    print(ordered["Summary"].to_string(index=False))
    print("")
    print(f"[review] wrote -> {path}")
    return 0


def command_dem(args: argparse.Namespace) -> int:
    """
    Settle elevation disagreements against an independent DEM (plan 1.4).

    identity.py proves from the observations that two records are one station
    and notices their heights cannot both be right, but it must not choose --
    diagnose, never homogenise. This chooses, using a third source, and writes
    the result where the other reviewed decisions live. It picks among the
    elevations the archives published; it never invents one.
    """
    policy = policy_from_args(args)
    canonical_dir = settings.canonical_dir()
    alias_path = canonical_dir / "station_aliases.csv"
    station_path = canonical_dir / "canonical_stations.csv"
    if not alias_path.exists() or not station_path.exists():
        print(f"[dem] no crosswalk at {canonical_dir}; run 'identity' first")
        return 1

    aliases = pd.read_csv(alias_path, dtype={"source_station_id": str})
    canonical = pd.read_csv(station_path)
    out_path = settings.review_dir() / "elevation_overrides.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else None
    n_existing = 0 if existing is None else len(existing)

    print(f"[dem] AWS Terrain Tiles, terrarium, zoom {policy.dem.zoom}")
    print(f"[dem] settling disagreements above {policy.dem.elevation_disagreement_m:g} m")
    if n_existing:
        print(f"[dem] {n_existing} override(s) already decided; those are kept as they are")

    overrides = dem_module.build_elevation_overrides(
        aliases, canonical, policy, existing=existing, progress=True
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_csv(out_path, index=False)

    # Computed rows are recalculated on every run, so "rows added" says nothing
    # about what moved. Report what changed against the file as it was.
    before = (
        {} if existing is None
        else dict(zip(existing["canonical_station_id"], existing["elevation_m"]))
    )
    changed = overrides[[
        before.get(cid) != elev
        for cid, elev in zip(overrides["canonical_station_id"], overrides["elevation_m"])
    ]]
    removed = sorted(set(before) - set(overrides["canonical_station_id"]))
    by_hand = int((~overrides["reason"].map(dem_module.is_computed_override)).sum())

    print("")
    print(f"[dem] {len(overrides)} override(s): {by_hand} by hand (kept as written), "
          f"{len(overrides) - by_hand} computed; {len(changed)} new or changed, "
          f"{len(removed)} no longer needed")
    for cid in removed:
        print(f"    {cid:26s} removed: its aliases no longer disagree")
    if len(changed):
        for _, row in changed.iterrows():
            gap = row["reason"].split("gap ")[-1].split(" m)")[0] if "gap " in row["reason"] else "?"
            print(f"    {row['canonical_station_id']:26s} -> {row['elevation_m']:8.1f} m "
                  f"(DEM within {gap} m)")
        print("")
        print("[dem] a large gap means the DEM agreed with no candidate and the")
        print("[dem] nearest one won on relative proximity alone -- read the reason.")
    print(f"[dem] wrote -> {out_path}")
    print("[dem] applied at read time; the consolidated archive is untouched")
    return 0


def command_climatology(args: argparse.Namespace) -> int:
    """
    Screen every selectable source record against an independent reference
    (plan 1.4). Writes a diagnostic table only: it proposes exclusions for the
    review workbook and changes no value. Consolidation acts on a reviewer's
    'yes', and on nothing else.
    """
    policy = policy_from_args(args)
    alias_path = settings.canonical_dir() / "station_aliases.csv"
    if not alias_path.exists():
        print(f"[climatology] no crosswalk at {alias_path}; run 'identity' first")
        return 1
    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"[climatology] no reference table at {reference_path}")
        return 1

    aliases = pd.read_csv(alias_path, dtype={"source_station_id": str})
    columns = {
        v: c for v, c in CERRA_CACHE_COLUMNS.items() if v in policy.source_priority
    }
    print(f"[climatology] reading reference {reference_path.name} …")
    wide = pd.read_parquet(
        reference_path,
        columns=[CERRA_CACHE_ID_COLUMN, CERRA_CACHE_DATE_COLUMN, *columns.values()],
    )
    elevation: dict = {}
    if getattr(args, "reference_static", None):
        static = pd.read_parquet(
            args.reference_static,
            columns=[CERRA_CACHE_ID_COLUMN, CERRA_STATIC_ELEVATION_COLUMN],
        )
        elevation = dict(
            zip(static[CERRA_CACHE_ID_COLUMN], static[CERRA_STATIC_ELEVATION_COLUMN])
        )
    else:
        print("[climatology] no --reference-static: temperatures compared "
              "without the elevation adjustment")
    reference = climatology_module.reference_from_wide(
        wide, CERRA_CACHE_ID_COLUMN, CERRA_CACHE_DATE_COLUMN, columns, elevation
    )
    del wide

    loader = bronze.SeriesLoader(period=policy.period)
    screen = climatology_module.run_screen(
        aliases, loader, reference, policy=policy, progress=True
    )
    out = settings.consolidated_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "climatology_screen.csv"
    screen.to_csv(path, index=False)
    # Provenance: which reference this screen ran on, and the screen itself,
    # by checksum. 'provenance' carries it into manifests/provenance.json.
    manifests.record_reference({
        "reference": reference_path,
        "reference_static": (
            Path(args.reference_static) if getattr(args, "reference_static", None) else None
        ),
        "screen": path,
    })

    print("")
    print(f"[climatology] {len(screen)} record x variable series reached")
    if not screen.empty:
        print("[climatology] outcome by status:")
        for status, n in screen["status"].value_counts().items():
            print(f"    {status}: {n}")
        tested = screen[screen["status"] == "tested"]
        table = pd.crosstab(tested["variable"], tested["tier"])
        print("")
        print("[climatology] tested series by tier:")
        print(table.to_string())
        proposed = tested[tested["tier"] == climatology_module.TIER_EXCLUDE]
        if not proposed.empty:
            print("")
            print("[climatology] proposed for exclusion:")
            for _, row in proposed.sort_values(["variable", "robust_z"]).iterrows():
                print(f"    {row['variable']:8s} {row['alias']:24s} {str(row['name'])[:26]:26s} "
                      f"station {row['station_mean']:8.2f}  reference {row['reference_mean']:8.2f}  "
                      f"z {row['robust_z']:+7.1f}  neighbours {int(row['n_neighbors'])} "
                      f"({int(row['n_neighbors_same_direction'])} same way)")
    print("")
    print(f"[climatology] wrote -> {path}")
    print(f"[climatology] reference recorded -> {manifests.reference_record_path()}")
    print("[climatology] diagnostic only -- 'review' puts these in the workbook; "
          "nothing is excluded without a reviewer's 'yes'")
    return 0


def _with(args: argparse.Namespace, **values) -> argparse.Namespace:
    """A copy of the parsed arguments with some set, for chaining commands."""
    return argparse.Namespace(**{**vars(args), **values})


def command_prepare(args: argparse.Namespace) -> int:
    """
    Everything a reviewer needs, in one workbook (the first half of the QC
    pipeline): the crosswalk with any decisions already made, the climatology
    screen, then the workbook holding both kinds of question.

    Merges and series exclusions can be answered at one sitting because neither
    depends on the other: the screen judges source records, not merged
    stations, and an exclusion never changes the evidence a merge rests on.
    """
    workbook = review_workbook_path()
    decisions = str(workbook) if workbook.exists() else None
    steps = (
        ("identity", command_identity, _with(
            args, sources=list(registry.crosswalk_sources()), decisions=decisions)),
        ("climatology", command_climatology, args),
        ("review", command_review, args),
    )
    for name, handler, step_args in steps:
        print(f"\n========== prepare: {name} ==========")
        code = handler(step_args)
        if code:
            print(f"[prepare] stopped: '{name}' failed")
            return code
    print("")
    print(f"[prepare] answer the workbook, then run 'apply': {workbook}")
    return 0


def command_apply(args: argparse.Namespace) -> int:
    """
    Apply every decision in the review workbook (the second half of the QC
    pipeline): the crosswalk with merge decisions, elevations against the DEM
    for the groups that now exist, selection with series exclusions, breakpoint
    flags on the result, and the provenance record of what built it.

    Elevation needs no question: the rule keeps the coordinates of the record
    with the most observations and the published elevation nearest the DEM,
    and it runs after the merges because a merge changes the candidates.
    Breakpoints come last because they can only be found on final series.
    """
    workbook = review_workbook_path()
    if not workbook.exists():
        print(f"[apply] no review workbook at {workbook}; run 'prepare' first")
        return 1
    steps = (
        ("identity", command_identity, _with(
            args, sources=list(registry.crosswalk_sources()), decisions=str(workbook))),
        ("dem", command_dem, args),
        ("consolidate", command_consolidate, _with(args, decisions=str(workbook))),
        ("homogeneity", command_homogeneity, args),
        ("provenance", command_provenance, _with(args, policy_out=None)),
    )
    for name, handler, step_args in steps:
        print(f"\n========== apply: {name} ==========")
        code = handler(step_args)
        if code:
            print(f"[apply] stopped: '{name}' failed")
            return code

    # A blank answer keeps the series. Say how many proposals were left that
    # way, so "nothing was excluded" is never mistaken for "nothing was wrong".
    try:
        series = pd.read_excel(workbook, sheet_name=climatology_module.SERIES_SHEET, dtype=str)
    except ValueError:
        series = pd.DataFrame()
    if not series.empty and "RECOMMENDATION" in series.columns:
        answered = series["DECISION_exclude"].fillna("").str.strip().str.lower()
        open_ = series[(series["RECOMMENDATION"] == "EXCLUDE") & ~answered.isin(["yes", "no"])]
        if len(open_):
            print("")
            print(f"[apply] {len(open_)} series proposed for exclusion have no decision "
                  f"and were kept; answer them in the Series sheet and run 'apply' again")
    return 0


def command_provenance(args: argparse.Namespace) -> int:
    """
    Write one machine-readable record of what built this archive.

    The per-source manifests already describe how each file arrived. This is
    the archive-level counterpart: the policy in force, each provider release
    and its order-independent checksum, and the library versions. It is the
    artefact a paper cites and the artefact two archives are diffed through.
    """
    import json

    policy = policy_from_args(args)
    record = manifests.build_provenance(policy, registry.all_sources())
    path = manifests.provenance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[provenance] station-archive {record['station_archive_version']} "
          f"on python {record['python']}")
    print(f"[provenance] domain {policy.domain.lat_min}..{policy.domain.lat_max} lat, "
          f"{policy.domain.lon_min}..{policy.domain.lon_max} lon "
          f"(+{policy.domain.buffer_deg} deg)")
    print(f"[provenance] window {policy.period[0]}-{policy.period[1]}")
    print("")
    for source, detail in record["sources"].items():
        if "release" not in detail:
            print(f"  {source:10s} {detail['status']}")
            continue
        print(f"  {source:10s} release {detail['release']:24s} "
              f"{detail['n_snapshots']:5d} snapshots  "
              f"{detail['archive_checksum'][:16]}...")
    reference = record["climatology_reference"]
    if "reference" in reference:
        print(f"  {'reference':10s} {reference['reference'].get('file', '')} "
              f"sha256 {reference['reference'].get('sha256', '')[:16]}... "
              f"(climatology screen, recorded {reference.get('recorded_at', '?')})")
    else:
        print(f"  {'reference':10s} {reference.get('status', 'not used')}")
    print("")
    if args.policy_out:
        out = Path(args.policy_out)
        policy.to_json(out)
        print(f"[provenance] policy    -> {out}")
    print(f"[provenance] record    -> {path}")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    """
    Run the whole chain on synthetic provider data: no network, no credentials.

    Exists because the alternative first experience of this repository is
    several gigabytes of licensed downloads and a Copernicus account. The demo
    exercises the real adapters, the real crosswalk and the real consolidator
    against files written in the providers' own formats, so what a reader sees
    is the actual pipeline rather than a description of it.
    """
    from station_archive import demo as demo_module

    return demo_module.run(
        root=Path(args.root) if args.root else None,
        n_stations=args.stations,
        n_years=args.years,
        keep=args.keep,
    )


def _policy_parser() -> argparse.ArgumentParser:
    """
    The scientific-policy arguments, shared by every subcommand.

    They live on the subcommands rather than before them for a concrete reason:
    --variables and --domain take several values, and a greedy list argument
    placed before the subcommand swallows the subcommand name itself. So the
    shape is `station-archive homogeneity --variables tas pr`, not
    `station-archive --variables tas pr homogeneity`.

    Every one is optional and falls back to the policy this archive was built
    with, which is what makes the tool usable on another region without editing
    source -- the difference between a published pipeline and a published record
    of one study.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--baseline",
        default="gsod",
        help="Source treated as the baseline in the coverage report (default: gsod).",
    )
    group = shared.add_argument_group("scientific policy")
    group.add_argument(
        "--policy",
        default=None,
        metavar="FILE",
        help="JSON policy file; anything it omits is inherited from the default.",
    )
    group.add_argument(
        "--domain",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Geographic admission box.",
    )
    group.add_argument(
        "--buffer",
        type=float,
        default=None,
        metavar="DEG",
        help="Synoptic buffer widening the domain on every side.",
    )
    group.add_argument(
        "--period",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Admission window, in calendar years, for completeness statistics.",
    )
    group.add_argument(
        "--fetch-years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Years requested where a provider subsets by year.",
    )
    group.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Canonical variables to work on (default: all seven).",
    )
    return shared


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="station-archive",
        description=(
            "Build one reconciled daily station-observation archive from several "
            "overlapping public climate sources."
        ),
    )
    shared = [_policy_parser()]
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser(
        "status", parents=shared, help="Show archive location and state."
    )
    status.set_defaults(handler=command_status)

    discover = subparsers.add_parser(
        "discover", parents=shared, help="Build per-source inventories."
    )
    discover.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    discover.set_defaults(handler=command_discover)

    report = subparsers.add_parser(
        "report", parents=shared, help="Rewrite the coverage report."
    )
    report.set_defaults(handler=command_report)

    fetch = subparsers.add_parser(
        "fetch", parents=shared, help="Download planned snapshots."
    )
    fetch.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    fetch.add_argument("--limit", type=int, default=0, help="Fetch at most N requests.")
    fetch.add_argument("--dry-run", action="store_true", help="Print the plan only.")
    fetch.set_defaults(handler=command_fetch)

    crosswalk = subparsers.add_parser(
        "identity", parents=shared,
        help="Build the canonical station crosswalk (plan 1.3).",
    )
    crosswalk.add_argument(
        "--sources", nargs="+", default=list(registry.crosswalk_sources())
    )
    crosswalk.add_argument(
        "--decisions",
        default=None,
        help="Review workbook whose Review sheet carries DECISION_merge values.",
    )
    crosswalk.set_defaults(handler=command_identity)

    cons = subparsers.add_parser(
        "consolidate", parents=shared,
        help="Select canonical values per station/date/variable (plan 1.6).",
    )
    cons.add_argument("--limit", type=int, default=0, help="First N canonical stations only.")
    cons.add_argument(
        "--decisions",
        default=None,
        help="Review workbook whose Series sheet carries DECISION_exclude values.",
    )
    cons.set_defaults(handler=command_consolidate)

    # No --variables of its own: the shared policy argument means the same thing
    # here, and two spellings of one idea is how they drift apart.
    homog = subparsers.add_parser(
        "homogeneity", parents=shared,
        help="Diagnose candidate breakpoints per station/variable (plan 1.4).",
    )
    homog.add_argument("--permutations", type=int, default=500,
                       help="Permutations for the significance test (default: 500).")
    homog.add_argument("--alpha", type=float, default=0.05,
                       help="Significance threshold on the permutation p-value (default: 0.05).")
    homog.set_defaults(handler=command_homogeneity)

    rev = subparsers.add_parser(
        "review", parents=shared,
        help="Build the merge-review workbook for unresolved pairs.",
    )
    rev.add_argument("--max-distance-km", type=float, default=3.0)
    rev.set_defaults(handler=command_review)

    prov = subparsers.add_parser(
        "provenance", parents=shared,
        help="Record the policy, releases and checksums that built this archive.",
    )
    prov.add_argument(
        "--policy-out",
        default=None,
        metavar="FILE",
        help="Also write the policy in force to this JSON file.",
    )
    prov.set_defaults(handler=command_provenance)

    dem_cmd = subparsers.add_parser(
        "dem", parents=shared,
        help="Settle elevation disagreements between archives against a DEM.",
    )
    dem_cmd.set_defaults(handler=command_dem)

    def reference_arguments(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--reference", required=True, metavar="PARQUET",
            help="Reference daily values at each station (the CERRA-at-stations cache).",
        )
        sub.add_argument(
            "--reference-static", default=None, metavar="PARQUET",
            help="Reference orography per station, for the temperature lapse adjustment.",
        )

    clim = subparsers.add_parser(
        "climatology", parents=shared,
        help="Screen source records against an independent reference (plan 1.4).",
    )
    reference_arguments(clim)
    clim.set_defaults(handler=command_climatology)

    prep = subparsers.add_parser(
        "prepare", parents=shared,
        help="QC pipeline, part 1: crosswalk, climatology screen, review workbook.",
    )
    reference_arguments(prep)
    prep.add_argument("--max-distance-km", type=float, default=3.0)
    prep.set_defaults(handler=command_prepare)

    apply_cmd = subparsers.add_parser(
        "apply", parents=shared,
        help="QC pipeline, part 2: apply the workbook and rebuild the archive.",
    )
    apply_cmd.add_argument("--limit", type=int, default=0, help="First N canonical stations only.")
    apply_cmd.add_argument("--permutations", type=int, default=500)
    apply_cmd.add_argument("--alpha", type=float, default=0.05)
    apply_cmd.set_defaults(handler=command_apply)

    demo = subparsers.add_parser(
        "demo", parents=shared,
        help="Run the whole chain on synthetic data, with no network or credentials.",
    )
    demo.add_argument(
        "--root",
        default=None,
        help="Where to build the demo archive (default: a temporary directory).",
    )
    demo.add_argument("--stations", type=int, default=12, help="Synthetic stations per source.")
    demo.add_argument("--years", type=int, default=12, help="Years of daily data.")
    demo.add_argument(
        "--keep",
        action="store_true",
        help="Leave the demo archive on disk instead of deleting it.",
    )
    demo.set_defaults(handler=command_demo)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
