"""
demo.py
=======
Run the whole pipeline on synthetic provider data.

WHY THIS EXISTS
---------------
Without it, the first experience of this repository is several gigabytes of
licensed downloads and a Copernicus account, so nobody can see the pipeline
work before committing to it -- and a reader who cannot run something has to
take the README's word for what it does.

The demo writes files in the providers' OWN formats (gzipped GHCN-D by_station
CSVs in tenths, ECA&D series files with their real header block), registers
them through the same manifest machinery as a real fetch, and then runs the
real identity, consolidation and homogeneity code over them. Nothing here is a
mock: the adapters parse these files with the same parsers they use on NOAA's.

WHAT THE SYNTHETIC ARCHIVE CONTAINS
-----------------------------------
A cluster of stations sharing one regional weather signal, so neighbour
correlation behaves as it does in reality, plus three deliberate features that
exercise the decisions this archive exists to make:

  * a TRUE DUPLICATE -- one GHCN-D station reporting the same values as an
    ECA&D station from ~1 km away, which the crosswalk should merge at tier 3
    on series evidence, since ECA&D publishes no identifier to match on;
  * a GAP in the primary source that the merged alias can fill, so the
    fallback compatibility test actually runs;
  * a STEP CHANGE in one station's record, which the breakpoint diagnosis
    should find and flag without altering a single value.

Everything is deterministic: same seed, same archive, same checksums.
"""
from __future__ import annotations

import gzip
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from station_archive.types import (
    DownloadRequest,
    INVENTORY_COLUMNS,
    SourceRelease,
    SourceSnapshot,
    StationInventory,
)

SEED = 20260926

# Small enough that every station is inside the neighbour search radius, so the
# homogeneity test has something to compare against.
CLUSTER_LAT = 38.0
CLUSTER_LON = 24.0
CLUSTER_SPREAD_DEG = 0.35

ECAD_HEADER = """EUROPEAN CLIMATE ASSESSMENT & DATASET (ECA&D), file created on: 01-01-2026
THESE DATA ARE SYNTHETIC AND CARRY NO LICENCE RESTRICTION: they are generated
by station_archive.demo to exercise the pipeline without provider data.

FILE FORMAT (MISSING VALUE CODE = -9999):

01-06 STAID: Station identifier
08-13 SOUID: Source identifier
15-22 DATE : Date YYYYMMDD
24-28 {element}   : {label} in 0.1 units
30-34 Q_{element} : quality code for {element} (0='valid'; 1='suspect'; 9='missing')

STAID, SOUID,    DATE, {element}, Q_{element}
"""


def _regional_weather(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """One weather signal every station shares, which is what makes neighbours
    correlate after their own seasonal cycle is removed."""
    return rng.normal(0.0, 3.0, n_days)


def _station_series(
    n_days: int,
    weather: np.ndarray,
    rng: np.random.Generator,
    step_after: Optional[int] = None,
    step_amount: float = 0.0,
) -> np.ndarray:
    """Daily maximum temperature in degC: season + shared weather + local noise."""
    seasonal = 22.0 + 9.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    values = seasonal + weather + rng.normal(0.0, 0.6, n_days)
    if step_after is not None:
        values = values.copy()
        values[step_after:] += step_amount
    return values


def _precip_series(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Daily precipitation in mm, skewed and mostly dry, as rainfall is."""
    wet = rng.random(n_days) < 0.25
    amounts = rng.gamma(shape=1.6, scale=3.0, size=n_days)
    return np.where(wet, amounts, 0.0)


def _inventory_rows(
    source: str,
    release: str,
    stations: list[dict],
    element_map: dict[str, str],
    unit_map: dict[str, str],
    first_date: str,
    last_date: str,
) -> StationInventory:
    records = []
    for station in stations:
        for element, variable in element_map.items():
            row = {column: "" for column in INVENTORY_COLUMNS}
            row.update(
                {
                    "source": source,
                    "source_release": release,
                    "source_station_id": station["id"],
                    "wmo_id": station.get("wmo", ""),
                    "wigos_id": "",
                    "name": station["name"],
                    "lat": station["lat"],
                    "lon": station["lon"],
                    "elevation_m": station["elevation_m"],
                    "variable": variable,
                    "provider_element": element,
                    "units": unit_map[element],
                    "observing_interval": "daily",
                    "timestamp_meaning": "synthetic daily value",
                    "first_date": first_date,
                    "last_date": last_date,
                    "n_obs_period": pd.NA,
                    "missing_fraction_period": pd.NA,
                    "access_status": "public",
                    "licence": "synthetic; no restriction",
                    "request": f"demo://{source}/{station['id']}",
                    "retrieved_at": "2026-01-01T00:00:00+00:00",
                }
            )
            records.append(row)
    return StationInventory.from_records(records)


def _write_ghcnd_station(
    path: Path, station_id: str, dates: pd.DatetimeIndex, tasmax: np.ndarray, pr: np.ndarray
) -> None:
    """Headerless by_station CSV, values in tenths, exactly as NOAA ships it."""
    lines = []
    for when, tx, rr in zip(dates, tasmax, pr):
        stamp = when.strftime("%Y%m%d")
        lines.append(f"{station_id},{stamp},TMAX,{int(round(tx * 10))},,,H,")
        lines.append(f"{station_id},{stamp},PRCP,{int(round(rr * 10))},,,H,")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_ecad_series(
    path: Path, staid: str, element: str, label: str,
    dates: pd.DatetimeIndex, values: np.ndarray,
) -> None:
    """One ECA&D element series, header block and all."""
    body = [ECAD_HEADER.format(element=element, label=label).rstrip("\n")]
    for when, value in zip(dates, values):
        if np.isnan(value):
            body.append(f"{int(staid):6d},{123456:6d},{when.strftime('%Y%m%d')},{-9999:5d},{9:5d}")
        else:
            body.append(
                f"{int(staid):6d},{123456:6d},{when.strftime('%Y%m%d')},"
                f"{int(round(value * 10)):5d},{0:5d}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="latin1")


def _record(source: str, release: str, station_id: str, relative: str, path: Path) -> None:
    from station_archive import manifests, net

    request = DownloadRequest(
        source=source,
        key=f"{source}/{station_id}/{relative}",
        relative_path=relative,
        url=f"demo://{source}/{relative}",
        metadata={"source_station_id": station_id},
    )
    manifests.record_snapshot(
        SourceSnapshot(
            request=request,
            path=path,
            sha256=net.sha256_file(path),
            n_bytes=path.stat().st_size,
            retrieved_at="2026-01-01T00:00:00+00:00",
            release=release,
            status="downloaded",
        )
    )


def build(root: Path, n_stations: int = 12, n_years: int = 12) -> dict:
    """
    Write a synthetic bronze layer and its inventories under 'root'.

    Returns a description of the features planted in it, so the caller can
    report whether the pipeline found them.
    """
    from station_archive import manifests, settings

    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("1990-01-01")
    dates = pd.date_range(start, periods=int(365.25 * n_years), freq="D")
    n_days = len(dates)
    weather = _regional_weather(n_days, rng)

    ecad_release = "2026-01-01"
    ghcnd_release = "demo-1.0"

    # --- ECA&D: the primary source for both variables ----------------------
    ecad_stations: list[dict] = []
    gap_slice = slice(int(n_days * 0.55), int(n_days * 0.70))
    step_index = int(n_days * 0.60)

    for i in range(n_stations):
        staid = f"{i + 1:06d}"
        lat = CLUSTER_LAT + (i % 4) * CLUSTER_SPREAD_DEG / 4
        lon = CLUSTER_LON + (i // 4) * CLUSTER_SPREAD_DEG / 4
        # One station carries a deliberate step change; the rest are clean.
        step_after = step_index if i == 3 else None
        tasmax = _station_series(
            n_days, weather, np.random.default_rng(SEED + i),
            step_after=step_after, step_amount=3.0,
        )
        pr = _precip_series(n_days, np.random.default_rng(SEED + 100 + i))
        # And one station has a gap a lower-priority alias can fill.
        if i == 0:
            tasmax = tasmax.copy()
            tasmax[gap_slice] = np.nan
        ecad_stations.append(
            {
                "id": staid,
                "name": f"SYNTH {staid}",
                "lat": lat,
                "lon": lon,
                "elevation_m": 40.0 + 5 * i,
                "tasmax": tasmax,
                "pr": pr,
            }
        )

    raw_ecad = settings.raw_dir("ecad", ecad_release)
    for station in ecad_stations:
        for element, label, key in (("TX", "Maximum temperature", "tasmax"),
                                    ("RR", "Precipitation amount", "pr")):
            member = f"{element}_STAID{station['id']}.txt"
            path = raw_ecad / element / member
            _write_ecad_series(path, station["id"], element, label, dates, station[key])
            _record("ecad", ecad_release, station["id"], f"{element}/{member}", path)

    # --- GHCN-D: one true duplicate of ECA&D station 0, plus two of its own --
    duplicate_of = ecad_stations[0]
    ghcnd_stations = [
        {
            "id": "SYN00000001",
            "name": "SYNTH 000001 AP",
            # ~1 km away: close enough to be a candidate, far enough that the
            # coordinates alone cannot decide it. The series has to.
            "lat": duplicate_of["lat"] + 0.009,
            "lon": duplicate_of["lon"],
            "elevation_m": duplicate_of["elevation_m"] + 4.0,
            # Replaced just below with the gap-filling variant; named here so the
            # station dict is complete at construction.
            "tasmax": duplicate_of["tasmax"],
            "pr": duplicate_of["pr"],
        }
    ]
    # The duplicate reports the days ECA&D lacks, so the fallback is testable.
    filled = duplicate_of["tasmax"].copy()
    clean = _station_series(n_days, weather, np.random.default_rng(SEED))
    filled[gap_slice] = clean[gap_slice]
    ghcnd_stations[0]["tasmax"] = filled

    for j in range(2):
        ghcnd_stations.append(
            {
                "id": f"SYN0000001{j}",
                "name": f"SYNTH GH {j}",
                "lat": CLUSTER_LAT + 0.20 + j * 0.05,
                "lon": CLUSTER_LON + 0.20,
                "elevation_m": 90.0 + j,
                "tasmax": _station_series(
                    n_days, weather, np.random.default_rng(SEED + 500 + j)
                ),
                "pr": _precip_series(n_days, np.random.default_rng(SEED + 600 + j)),
            }
        )

    raw_ghcnd = settings.raw_dir("ghcnd", ghcnd_release)
    for station in ghcnd_stations:
        relative = f"by_station/{station['id']}.csv.gz"
        path = raw_ghcnd / relative
        _write_ghcnd_station(
            path, station["id"], dates,
            np.nan_to_num(station["tasmax"], nan=-999.9),
            station["pr"],
        )
        _record("ghcnd", ghcnd_release, station["id"], relative, path)

    # --- inventories and release rows --------------------------------------
    first, last = dates[0].date().isoformat(), dates[-1].date().isoformat()
    settings.inventory_dir().mkdir(parents=True, exist_ok=True)
    _inventory_rows(
        "ecad", ecad_release, ecad_stations,
        {"TX": "tasmax", "RR": "pr"},
        {"TX": "degC_tenths", "RR": "mm_tenths"},
        first, last,
    ).to_csv(settings.inventory_dir() / "ecad_inventory.csv")
    _inventory_rows(
        "ghcnd", ghcnd_release, ghcnd_stations,
        {"TMAX": "tasmax", "PRCP": "pr"},
        {"TMAX": "degC_tenths", "PRCP": "mm_tenths"},
        first, last,
    ).to_csv(settings.inventory_dir() / "ghcnd_inventory.csv")

    for source, release in (("ecad", ecad_release), ("ghcnd", ghcnd_release)):
        manifests.record_release(
            SourceRelease(
                source=source,
                release=release,
                retrieved_at="2026-01-01T00:00:00+00:00",
                licence="synthetic; no restriction",
                licence_url="",
                redistribution="unrestricted",
                notes="generated by station_archive.demo",
            )
        )

    return {
        "n_days": n_days,
        "period": (dates[0].year, dates[-1].year),
        "duplicate_pair": ("ecad:000001", "ghcnd:SYN00000001"),
        "gap_days": gap_slice.stop - gap_slice.start,
        "step_station": "ecad:000004",
        "step_date": dates[step_index].date().isoformat(),
    }


def run(
    root: Optional[Path] = None,
    n_stations: int = 12,
    n_years: int = 12,
    keep: bool = False,
) -> int:
    """Build the synthetic archive, run the pipeline over it, report findings."""
    from dataclasses import replace

    from station_archive import bronze, consolidate, homogeneity, identity, settings
    from station_archive.policy import DEFAULT_POLICY
    from station_archive.types import Domain

    temporary = root is None
    root = Path(tempfile.mkdtemp(prefix="station-archive-demo-")) if temporary else Path(root)
    # settings owns the variable's name; restating it here is how the two
    # drift, and a demo that redirected the wrong variable would build its
    # synthetic archive on top of a real one.
    previous = os.environ.get(settings.ARCHIVE_ROOT_ENV_VAR)
    os.environ[settings.ARCHIVE_ROOT_ENV_VAR] = str(root)

    try:
        print(f"[demo] building a synthetic archive in {root}")
        planted = build(root, n_stations=n_stations, n_years=n_years)
        print(f"[demo] {n_stations} ECA&D stations and 3 GHCN-D stations, "
              f"{planted['n_days']} days each, in providers' own formats")
        print("")

        policy = replace(
            DEFAULT_POLICY,
            domain=Domain(
                CLUSTER_LAT - 1, CLUSTER_LAT + 1, CLUSTER_LON - 1, CLUSTER_LON + 1, 0.5
            ),
            period=planted["period"],
            variables=("tasmax", "pr"),
            source_priority={"tasmax": ("ecad", "ghcnd"), "pr": ("ecad", "ghcnd")},
        )

        # ---- identity ------------------------------------------------------
        inventories = {
            name: StationInventory.read_csv(
                settings.inventory_dir() / f"{name}_inventory.csv"
            )
            for name in ("ecad", "ghcnd")
        }
        loader = bronze.SeriesLoader(period=policy.period)
        crosswalk = identity.build_crosswalk(inventories, loader, policy=policy)
        left, right = planted["duplicate_pair"]
        merged = (
            crosswalk.aliases.set_index(
                crosswalk.aliases["source"] + ":" + crosswalk.aliases["source_station_id"]
            )["canonical_station_id"]
        )
        found_duplicate = merged.get(left) == merged.get(right)
        print(f"[demo] identity   : {crosswalk.n_aliases()} source stations -> "
              f"{crosswalk.n_canonical()} canonical")
        print(f"[demo]              planted duplicate {left} / {right} merged: "
              f"{'YES' if found_duplicate else 'no'}")

        # ---- consolidation --------------------------------------------------
        frames, compat = [], []
        for canonical_id in sorted(crosswalk.aliases["canonical_station_id"].unique()):
            group = crosswalk.aliases[
                crosswalk.aliases["canonical_station_id"] == canonical_id
            ]
            for variable in policy.source_priority:
                result = consolidate.consolidate_station_variable(
                    canonical_id, group, variable, loader, policy=policy
                )
                frames.append(result.values)
                compat.extend(result.compatibility)
            loader.clear()
        values = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        n_fallback = int(values["is_fallback"].sum())
        tested = [c for c in compat if c["candidate_days"] > 0]
        print(f"[demo] consolidate: {len(values)} accepted values, "
              f"{n_fallback} filled from a lower-priority alias")
        print(f"[demo]              {len(tested)} source switches tested, "
              f"{sum(1 for c in tested if not c['compatible'])} refused as incompatible")

        # ---- homogeneity ----------------------------------------------------
        meta = identity.resolve_canonical_metadata(
            crosswalk.aliases, loader, policy=policy
        )[["canonical_station_id", "lat", "lon"]]
        report = homogeneity.run_report(
            values, meta, ["tasmax"], n_permutations=200, seed=0
        )
        layer = homogeneity.build_flag_layer(report)
        step_found = planted["step_station"] in set(layer["canonical_station_id"]) or any(
            planted["step_station"] in str(v) for v in layer["canonical_station_id"]
        )
        tested_rows = report[report["status"] == "tested"] if not report.empty else report
        print(f"[demo] homogeneity: {len(report)} series reached, "
              f"{len(tested_rows)} testable, {len(layer)} flagged after FDR")
        if not layer.empty:
            for _, row in layer.iterrows():
                print(f"[demo]              {row['canonical_station_id']} tasmax "
                      f"shift {row['shift']:+.2f} degC at "
                      f"{pd.Timestamp(row['breakpoint_date']).date()} "
                      f"(support {int(row['min_support_at_break'])} neighbours)")
        print(f"[demo]              planted step at {planted['step_date']} "
              f"({planted['step_station']}) detected: "
              f"{'YES' if step_found else 'no'}")
        if len(layer) > 1:
            print("[demo]")
            print("[demo]              Note the smaller shifts of the opposite sign at the")
            print("[demo]              same date on other stations. Only one station moved;")
            print("[demo]              it is a NEIGHBOUR of the others, so its step enters")
            print("[demo]              their reference series divided by the neighbour count.")
            print("[demo]              This module detects breaks but does not attribute them")
            print("[demo]              across the network the way PHA does -- see the")
            print("[demo]              homogeneity docstring. Read a cluster of same-dated")
            print("[demo]              flags as one event, not as several.")

        print("")
        print("[demo] no value was altered anywhere in that chain: consolidation "
              "selects, homogeneity flags.")
        if keep or not temporary:
            print(f"[demo] archive kept at {root}")
        return 0
    finally:
        if previous is None:
            os.environ.pop(settings.ARCHIVE_ROOT_ENV_VAR, None)
        else:
            os.environ[settings.ARCHIVE_ROOT_ENV_VAR] = previous
        if temporary and not keep:
            shutil.rmtree(root, ignore_errors=True)
