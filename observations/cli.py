"""
cli.py
======
Command-line entry point for station-archive acquisition.

USAGE
-----
  # 1. Discovery first (plan section 1.2): what does each source offer?
  python -m observations.cli discover --sources ghcnd ecad gsod

  # 2. Read the report before transferring anything.
  python -m observations.cli report

  # 3. Fetch, one source at a time, restartable and idempotent.
  python -m observations.cli fetch --sources ghcnd

  # Show where the archive lives and whether the drive has room.
  python -m observations.cli status

The three-command shape is the plan's discovery-before-download rule made
operational: 'fetch' plans from an inventory that 'discover' already wrote, so
there is no path that downloads in bulk without a coverage record first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from observations import (
    bronze,
    config,
    consolidate as consolidate_module,
    homogeneity as homogeneity_module,
    identity as identity_module,
    review as review_module,
    inventory as inventory_module,
    manifests,
)
from observations.sources.base import StationSource
from observations.sources.c3s_land import C3sLandSource
from observations.sources.ecad import EcadSource
from observations.sources.ghcnd import GhcndSource
from observations.sources.gsod import GsodSource
from observations.types import StationInventory

SOURCES: dict[str, type[StationSource]] = {
    "ghcnd": GhcndSource,
    "ecad": EcadSource,
    "gsod": GsodSource,
    "c3s_land": C3sLandSource,
}

DEFAULT_SOURCES = ("ghcnd", "ecad", "gsod")


def build_source(name: str) -> StationSource:
    if name not in SOURCES:
        raise SystemExit(f"unknown source {name!r}; choose from {sorted(SOURCES)}")
    return SOURCES[name]()


def inventory_path(source: str) -> Path:
    return config.inventory_dir() / f"{source}_inventory.csv"


def command_status(args: argparse.Namespace) -> int:
    root = config.archive_root()
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    print(f"archive root : {root}")
    print(f"free space   : {usage.free / 1024**3:.2f} GB")
    print(f"domain       : {config.DOMAIN}")
    print(f"window       : {config.PERIOD_START}-{config.PERIOD_END}")
    print(f"fetch years  : {config.FETCH_YEAR_START}-{config.FETCH_YEAR_END}")
    print(f"variables    : {', '.join(config.VARIABLES)}")
    print("")
    for source in sorted(SOURCES):
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
    period = (config.PERIOD_START, config.PERIOD_END)
    for name in args.sources:
        source = build_source(name)
        print(f"[{name}] discovering...")
        result = source.discover(config.DOMAIN, config.VARIABLES, period)
        path = inventory_path(name)
        result.to_csv(path)
        manifests.record_release(source.source_release())
        print(
            f"[{name}] {len(result)} station-variable rows, "
            f"{result.n_stations()} stations -> {path}"
        )
    return command_report(args)


def command_report(args: argparse.Namespace) -> int:
    inventories = inventory_module.load_inventories(sorted(SOURCES))
    if not inventories:
        print("no inventories found; run 'discover' first")
        return 1
    frame = inventory_module.merge(inventories)
    path = inventory_module.write_report(frame, baseline=args.baseline)
    print(f"coverage report -> {path}")
    print("")
    print(inventory_module.coverage_report(frame, baseline=args.baseline))
    return 0


def command_fetch(args: argparse.Namespace) -> int:
    for name in args.sources:
        path = inventory_path(name)
        if not path.exists():
            print(f"[{name}] no inventory at {path}; run 'discover' first")
            return 1

        source = build_source(name)
        stations = StationInventory.read_csv(path)
        requests = source.plan(stations)

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

    loader = bronze.SeriesLoader(period=(config.PERIOD_START, config.PERIOD_END))
    crosswalk = identity_module.build_crosswalk(
        inventories, loader, progress=True, decisions=decisions
    )

    out = config.archive_root() / "canonical"
    out.mkdir(parents=True, exist_ok=True)
    crosswalk.aliases.to_csv(out / "station_aliases.csv", index=False)
    crosswalk.metadata_history.to_csv(out / "station_metadata_history.csv", index=False)
    crosswalk.evidence.to_csv(out / "duplicate_evidence.csv", index=False)
    crosswalk.conflicts.to_csv(out / "metadata_conflicts.csv", index=False)

    # One authoritative position per canonical station, from the alias with
    # the most observations.  Counted only for merged groups, so this reads
    # the merged stations rather than the whole archive.
    canonical = identity_module.resolve_canonical_metadata(crosswalk.aliases, loader)
    canonical.to_csv(out / "canonical_stations.csv", index=False)

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
        for _, row in crosswalk.conflicts.iterrows():
            print(f"    {row['left']} / {row['right']}: {row['conflict']} "
                  f"({row['distance_km']} km, {row['elevation_diff_m']} m elevation) "
                  f"identical on {row['exact_fraction']:.0%} of {row['n_overlap_days']} days")
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
    canonical_dir = config.archive_root() / "canonical"
    alias_path = canonical_dir / "station_aliases.csv"
    if not alias_path.exists():
        print(f"[consolidate] no crosswalk at {alias_path}; run 'identity' first")
        return 1

    aliases = pd.read_csv(alias_path, dtype={"source_station_id": str})
    station_ids = sorted(aliases["canonical_station_id"].unique())
    if args.limit:
        station_ids = station_ids[: args.limit]

    out = config.archive_root() / "consolidated"
    out.mkdir(parents=True, exist_ok=True)

    loader = bronze.SeriesLoader(period=(config.PERIOD_START, config.PERIOD_END))
    compatibility: list[dict] = []

    values_writer = _BufferedParquetWriter(out / "observations_consolidated")
    # Non-selected candidates (plan 1.6 point 5): every value that existed
    # for a station-day-variable but lost to a higher-priority source or an
    # incompatible-fallback refusal, kept rather than discarded.
    non_selected_writer = _BufferedParquetWriter(out / "non_selected_candidates")

    for index, canonical_id in enumerate(station_ids, start=1):
        group = aliases[aliases["canonical_station_id"] == canonical_id]
        for variable in config.SOURCE_PRIORITY:
            result = consolidate_module.consolidate_station_variable(
                canonical_id, group, variable, loader
            )
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
            for (src, var, reason), n in sorted(loader.rejections.items())
        ]
    )
    rejections.to_csv(out / "qc_rejections.csv", index=False)

    # Re-read the parts for the audit rather than keeping them in memory.
    written = sorted((out / "observations_consolidated").glob("part-*.parquet"))
    frame = (
        pd.concat([pd.read_parquet(f) for f in written], ignore_index=True)
        if written
        else pd.DataFrame(columns=list(consolidate_module.CONSOLIDATED_COLUMNS))
    )
    audit = consolidate_module.source_contribution(frame)
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
    print(f"[consolidate] canonical values : {len(frame)}")
    print(f"[consolidate] stations         : {frame['canonical_station_id'].nunique()}")
    if not frame.empty:
        print("")
        print("[consolidate] values by variable and source:")
        pivot = frame.pivot_table(
            index="variable", columns="selected_source", values="value",
            aggfunc="size", fill_value=0,
        )
        print(pivot.to_string())
        fallback = int(frame["is_fallback"].sum())
        print("")
        print(f"[consolidate] fallback values   : {fallback} "
              f"({fallback / max(len(frame), 1):.2%})")
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
    canonical_dir = config.archive_root() / "canonical"
    consolidated_dir = config.archive_root() / "consolidated"
    stations_path = canonical_dir / "canonical_stations.csv"
    parts_dir = consolidated_dir / "observations_consolidated"

    if not stations_path.exists() or not parts_dir.exists():
        print(f"[homogeneity] no consolidated archive at {consolidated_dir}; run 'consolidate' first")
        return 1

    stations_meta = pd.read_csv(stations_path)[["canonical_station_id", "lat", "lon"]]
    parts = sorted(parts_dir.glob("part-*.parquet"))
    values = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    variables = args.variables or list(config.SOURCE_PRIORITY)
    print(f"[homogeneity] testing {len(variables)} variable(s) across "
          f"{stations_meta['canonical_station_id'].nunique()} candidate stations …")

    report = homogeneity_module.run_report(
        values, stations_meta, variables,
        n_permutations=args.permutations, alpha=args.alpha,
    )

    out_path = consolidated_dir / "homogeneity_breakpoints.csv"
    report.to_csv(out_path, index=False)

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
            # Shifts are in each variable's own monthly units (degC, mm/month,
            # m/s), so they are ranked within a variable, never across them.
            print("[homogeneity] largest FDR-surviving shift per variable:")
            for variable, group in flagged.groupby("variable"):
                top = group.reindex(group["shift"].abs().sort_values(ascending=False).index).head(3)
                for _, row in top.iterrows():
                    print(f"    {variable}: {row['canonical_station_id']} "
                          f"shift {row['shift']:+.2f} at {row['breakpoint_date']} "
                          f"(SNHT={row['statistic']:.0f}, n_neighbors={row['n_neighbors_used']})")
    print("")
    print(f"[homogeneity] wrote -> {out_path}")
    print("[homogeneity] diagnostic only -- no value in the archive was changed")
    return 0


def command_review(args: argparse.Namespace) -> int:
    """
    Build the human-review workbook for unresolved station pairs (plan 1.3).

    Tier 4 gives unresolved candidates a duplicate-risk flag, which is a
    request for review rather than a verdict.  This turns those flags into
    something a person can adjudicate.
    """
    canonical_dir = config.archive_root() / "canonical"
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

    loader = bronze.SeriesLoader(period=(config.PERIOD_START, config.PERIOD_END))
    sheets = review_module.build_review(
        evidence, aliases, loader, max_distance_km=args.max_distance_km, progress=True
    )

    ordered = {
        "Summary": review_module.summary_sheet(sheets, conflicts),
        "Review": sheets["Review"],
        "Per_variable": sheets["Per_variable"],
        "Metadata_conflicts": conflicts,
        "Beyond_radius": sheets["Beyond_radius"],
        "Never_merge_rule": sheets["Never_merge_rule"],
    }
    path = review_module.write_workbook(
        config.archive_root() / "review" / "station_merge_review.xlsx", ordered
    )

    print("")
    print(ordered["Summary"].to_string(index=False))
    print("")
    print(f"[review] wrote -> {path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="observations",
        description="Discover and download station observations for the Delos archive.",
    )
    parser.add_argument(
        "--baseline",
        default="gsod",
        help="Source treated as the baseline in the coverage report (default: gsod).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show archive location and state.")
    status.set_defaults(handler=command_status)

    discover = subparsers.add_parser("discover", help="Build per-source inventories.")
    discover.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    discover.set_defaults(handler=command_discover)

    report = subparsers.add_parser("report", help="Rewrite the coverage report.")
    report.set_defaults(handler=command_report)

    fetch = subparsers.add_parser("fetch", help="Download planned snapshots.")
    fetch.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    fetch.add_argument("--limit", type=int, default=0, help="Fetch at most N requests.")
    fetch.add_argument("--dry-run", action="store_true", help="Print the plan only.")
    fetch.set_defaults(handler=command_fetch)

    crosswalk = subparsers.add_parser(
        "identity", help="Build the canonical station crosswalk (plan 1.3)."
    )
    crosswalk.add_argument("--sources", nargs="+", default=list(config.IDENTITY_SOURCES))
    crosswalk.add_argument(
        "--decisions",
        default=None,
        help="Review workbook whose Review sheet carries DECISION_merge values.",
    )
    crosswalk.set_defaults(handler=command_identity)

    cons = subparsers.add_parser(
        "consolidate", help="Select canonical values per station/date/variable (plan 1.6)."
    )
    cons.add_argument("--limit", type=int, default=0, help="First N canonical stations only.")
    cons.set_defaults(handler=command_consolidate)

    homog = subparsers.add_parser(
        "homogeneity", help="Diagnose candidate breakpoints per station/variable (plan 1.4)."
    )
    homog.add_argument(
        "--variables", nargs="+", default=None,
        help="Variables to test (default: every variable in SOURCE_PRIORITY).",
    )
    homog.add_argument("--permutations", type=int, default=500,
                        help="Permutations for the significance test (default: 500).")
    homog.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold on the permutation p-value (default: 0.05).")
    homog.set_defaults(handler=command_homogeneity)

    rev = subparsers.add_parser(
        "review", help="Build the merge-review workbook for unresolved pairs."
    )
    rev.add_argument("--max-distance-km", type=float, default=3.0)
    rev.set_defaults(handler=command_review)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
