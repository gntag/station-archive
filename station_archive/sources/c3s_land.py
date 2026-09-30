"""
sources/c3s_land.py
===================
Copernicus C3S global land surface in-situ observations adapter.

DATA SOURCE
-----------
Climate Data Store dataset 'insitu-observations-surface-land'
DOI 10.24381/cds.cf5f3bac
"Global land surface atmospheric variables from 1718 to present from
comprehensive in-situ observations"

REQUEST SCHEMA
--------------
Read from the dataset's published form on 13 September 2026, not assumed:

    version           one of {2_0_0, 3_0_0}          required
    time_aggregation  one of {monthly, daily, sub_daily}  required
    variable          list from ten names            required
    year              ONE year per request           required
    month, day        lists                          required
    area              [north, west, south, east]     optional
    data_format       one of {netcdf, csv}           required

'year' being single-choice is why plan() emits one request per year rather
than one request for the whole period.

VARIABLE COVERAGE
-----------------
The ten offered variables are: accumulated_precipitation, air_pressure,
air_pressure_at_sea_level, air_temperature, dew_point_temperature,
fresh_snow, snow_depth, snow_water_equivalent, wind_from_direction,
wind_speed.

There is no relative-humidity variable and no daily minimum/maximum
temperature variable.  This adapter therefore requests air_temperature and
dew_point_temperature but does NOT claim to supply 'hurs', 'tasmin' or
'tasmax'.  Deriving RH from temperature and dew point is permitted by plan
section 1.4 only when "both inputs and the formula are recorded", which makes
it a standardization decision downstream, not a download-time one.

CREDENTIALS
-----------
cdsapi reads ~/.cdsapirc or the CDSAPI_URL / CDSAPI_KEY environment
variables.  Plan section 1.5: credentials "belong in environment
configuration and never in the manifest or repository".  manifests.py redacts
any credential-looking payload key as a second line of defence.

CAUTION FROM THE PLAN
---------------------
Section 1.1: C3S states "no assessment of potential bias has been made", and
"multiple underlying sources can repeat the same physical record".  Both are
recorded on the release; neither is acted on here.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import pandas as pd

from station_archive import net
from station_archive.sources.base import StationSource
from station_archive.types import (
    AccessError,
    Domain,
    DownloadRequest,
    RawObservation,
    SchemaError,
    SourceSnapshot,
    StationInventory,
)

if TYPE_CHECKING:            # runtime import would cycle via the registry
    from station_archive.policy import Policy

DATASET = "insitu-observations-surface-land"

DEFAULT_VERSION = "3_0_0"
DEFAULT_TIME_AGGREGATION = "daily"
DEFAULT_FORMAT = "csv"

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]

# Canonical variable -> CDS variable name.  Only mappings that are a genuine
# one-to-one match are listed; see the module docstring on hurs/tasmin/tasmax.
VARIABLE_TO_CDS: dict[str, str] = {
    "tas": "air_temperature",
    "pr": "accumulated_precipitation",
    "sfcWind": "wind_speed",
}

# Additional CDS variables always requested because standardization needs them
# even though they are not themselves canonical targets.
SUPPORTING_CDS_VARIABLES: tuple[str, ...] = ("dew_point_temperature",)

CDS_TO_CANONICAL = {cds: canonical for canonical, cds in VARIABLE_TO_CDS.items()}


class C3sLandSource(StationSource):
    name = "c3s_land"
    licence = "Copernicus / CC-BY-4.0 plus the global land observations data policy"
    licence_url = "https://cds.climate.copernicus.eu/datasets/insitu-observations-surface-land"
    redistribution = (
        "Redistribution permitted with attribution under the Copernicus licence; "
        "C3S states that no assessment of potential bias has been made."
    )
    element_to_variable = CDS_TO_CANONICAL
    # C3S declares units per row rather than per variable, so there is no
    # per-element unit table. Identity never reads a C3S series (it joins at
    # tier 1 on an exact identifier), and consolidation never selects it.
    element_units: dict[str, str] = {}

    def __init__(
        self,
        version: str = DEFAULT_VERSION,
        time_aggregation: str = DEFAULT_TIME_AGGREGATION,
        data_format: str = DEFAULT_FORMAT,
        client: Any | None = None,
    ) -> None:
        self.version = version
        self.time_aggregation = time_aggregation
        self.data_format = data_format
        self._client = client

    # ---- release ---------------------------------------------------------

    def release(self) -> str:
        """The CDS dataset version, which is the provider's own identifier."""
        return f"{self.version}-{self.time_aggregation}"

    def client(self) -> Any:
        """
        Lazily construct the cdsapi client.

        Deferred so discovery and planning can run, and be tested, on a
        machine with no CDS credentials configured.
        """
        if self._client is None:
            try:
                import cdsapi
            except ImportError as exc:
                raise AccessError(
                    "cdsapi is required for the C3S adapter: pip install cdsapi"
                ) from exc
            try:
                self._client = cdsapi.Client()
            except Exception as exc:
                raise AccessError(
                    "could not initialise the CDS client; configure ~/.cdsapirc "
                    "or CDSAPI_URL/CDSAPI_KEY"
                ) from exc
        return self._client

    # ---- discovery -------------------------------------------------------

    def cds_variables(self, variables: Sequence[str]) -> list[str]:
        """CDS variable names for the requested canonical variables."""
        selected = [VARIABLE_TO_CDS[v] for v in variables if v in VARIABLE_TO_CDS]
        return sorted(set(selected) | set(SUPPORTING_CDS_VARIABLES))

    def build_request(
        self,
        domain: Domain,
        variables: Sequence[str],
        year: int,
    ) -> dict[str, Any]:
        """
        One CDS request payload, matching the published form exactly.

        'year' is a single value because the form declares it as a
        single-choice widget; sending a list would be rejected.
        """
        return {
            "version": self.version,
            "time_aggregation": self.time_aggregation,
            "variable": self.cds_variables(variables),
            "year": str(year),
            "month": MONTHS,
            "day": DAYS,
            "area": domain.as_cds_area(),
            "data_format": self.data_format,
        }

    def discover(
        self,
        domain: Domain,
        variables: Sequence[str],
        period: tuple[int, int],
    ) -> StationInventory:
        """
        Pilot-year discovery: retrieve one year and inventory what came back.

        Unlike GHCN-D and ECA&D, C3S publishes no station-metadata file to
        query, so the only honest way to answer plan section 1.2's questions
        is to fetch a single year and read the station identities out of it.
        Plan section 1.1 asks for exactly this: the collection "should be
        queried before bulk download to measure what it adds beyond the
        existing NOAA sources".

        The pilot year is the midpoint of the admission window, which is more
        representative of the network than either endpoint.
        """
        period_start, period_end = period
        pilot_year = (period_start + period_end) // 2

        request = DownloadRequest(
            source=self.name,
            key=f"c3s_land/pilot/{pilot_year}",
            relative_path=f"pilot/{pilot_year}.{self._extension()}",
            payload=self.build_request(domain, variables, pilot_year),
            metadata={"year": pilot_year, "pilot": True},
        )

        from station_archive import settings

        destination = settings.raw_dir(self.name, self.release()) / request.relative_path
        if not destination.exists():
            self._transfer(request, destination)

        frame = self._read_table(destination)
        return self._inventory_from_frame(frame, domain, pilot_year, destination)

    def _inventory_from_frame(
        self,
        frame: pd.DataFrame,
        domain: Domain,
        pilot_year: int,
        path: Path,
    ) -> StationInventory:
        columns = _resolve_columns(frame, path.name)
        release = self.release()
        retrieved_at = net.utc_now()

        records = []
        grouped = frame.groupby([columns["station_id"], columns["variable"]], dropna=True)
        for (station_id, cds_variable), group in grouped:
            lat = pd.to_numeric(group[columns["lat"]], errors="coerce").median()
            lon = pd.to_numeric(group[columns["lon"]], errors="coerce").median()
            if not domain.contains(lat, lon):
                continue

            dates = pd.to_datetime(group[columns["date"]], errors="coerce").dropna()
            n_obs = int(len(dates))
            records.append(
                {
                    "source": self.name,
                    "source_release": release,
                    "source_station_id": str(station_id),
                    "wmo_id": _first_non_empty(group, columns.get("wmo_id")),
                    "wigos_id": _first_non_empty(group, columns.get("wigos_id")),
                    "name": _first_non_empty(group, columns.get("name")),
                    "lat": lat,
                    "lon": lon,
                    "elevation_m": pd.to_numeric(
                        group[columns["elevation"]], errors="coerce"
                    ).median()
                    if columns.get("elevation")
                    else pd.NA,
                    "variable": CDS_TO_CANONICAL.get(
                        str(cds_variable), f"c3s:{cds_variable}"
                    ),
                    "provider_element": str(cds_variable),
                    "units": _first_non_empty(group, columns.get("units")),
                    "observing_interval": self.time_aggregation,
                    "timestamp_meaning": "as reported by the contributing source",
                    "first_date": dates.min().date().isoformat() if n_obs else pd.NA,
                    "last_date": dates.max().date().isoformat() if n_obs else pd.NA,
                    # Counts are for the pilot year only, which is what the
                    # column name must not overstate: they are extrapolated by
                    # nothing and describe one year of evidence.
                    "n_obs_period": n_obs,
                    "missing_fraction_period": round(1.0 - n_obs / 365.0, 4)
                    if n_obs <= 365
                    else 0.0,
                    "access_status": "public",
                    "licence": self.licence,
                    "request": f"cds:{DATASET}?version={self.version}"
                    f"&time_aggregation={self.time_aggregation}&year={pilot_year}",
                    "retrieved_at": retrieved_at,
                }
            )
        return StationInventory.from_records(records)

    # ---- planning and transfer ------------------------------------------

    def plan(
        self,
        inventory: StationInventory,
        policy: Optional["Policy"] = None,
    ) -> list[DownloadRequest]:
        """
        One request per year over the fetch window.

        The CDS form makes 'year' single-choice, so the period cannot be
        collapsed into one request.  The variable set comes from the
        inventory's own rows, so a filtered inventory narrows the download;
        the year window and the request area come from 'policy'.
        """
        # Imported here, not at module scope: policy imports the source
        # registry, which imports this module.
        if policy is None:
            from station_archive.policy import DEFAULT_POLICY as policy

        if len(inventory) == 0:
            return []

        variables = sorted(
            {
                v
                for v in inventory.frame["variable"].dropna().unique()
                if v in VARIABLE_TO_CDS
            }
        )

        requests = []
        for year in range(policy.fetch_years[0], policy.fetch_years[1] + 1):
            requests.append(
                DownloadRequest(
                    source=self.name,
                    key=f"c3s_land/{year}",
                    relative_path=f"{year}/{DATASET}_{year}.{self._extension()}",
                    payload=self.build_request(policy.domain, variables, year),
                    metadata={"year": year},
                )
            )
        return requests

    def _extension(self) -> str:
        return "zip" if self.data_format == "csv" else "nc"

    def _transfer(self, request: DownloadRequest, destination: Path) -> None:
        """
        Submit the request to the CDS and write the result.

        cdsapi handles its own queueing and retries; a .part file is used so
        an interrupted retrieval cannot be mistaken for a complete snapshot.

        A 403 with "required licences not accepted" is a distinct failure
        mode from a transient outage: the CDS account must accept the
        dataset's licence on the website before any request succeeds, no
        matter how many times it is retried.  That is an account action only
        the credential owner can take, so it is raised as AccessError with
        the exact URL rather than left to surface as a bare HTTPError deep in
        cdsapi's retry loop.
        """
        assert request.payload is not None
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            self.client().retrieve(DATASET, request.payload, str(partial))
        except Exception as exc:
            partial.unlink(missing_ok=True)
            if "licence" in str(exc).lower() or "license" in str(exc).lower():
                raise AccessError(
                    "CDS dataset licence not accepted for "
                    f"{DATASET!r}. Visit "
                    f"https://cds.climate.copernicus.eu/datasets/{DATASET}"
                    "?tab=download#manage-licences while logged in, accept "
                    "the required licence(s), then retry."
                ) from exc
            raise
        partial.replace(destination)

    # ---- parsing ---------------------------------------------------------

    def _read_table(self, path: Path) -> pd.DataFrame:
        """
        Read a CDS delivery into a frame.

        CSV deliveries arrive as a ZIP of one or more CSVs; NetCDF deliveries
        need xarray.  Both are handled so the adapter's data_format setting is
        a real choice rather than a comment.
        """
        if path.suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(path) as archive:
                members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if not members:
                    raise SchemaError(f"{path.name}: CDS delivery contains no CSV member")
                frames = []
                for member in members:
                    with archive.open(member) as handle:
                        frames.append(pd.read_csv(handle, comment="#", low_memory=False))
            return pd.concat(frames, ignore_index=True)

        if path.suffix == ".nc":
            import xarray as xr

            with xr.open_dataset(path) as dataset:
                return dataset.to_dataframe().reset_index()

        raise SchemaError(f"{path.name}: unrecognised CDS delivery format {path.suffix!r}")

    def parse(self, snapshot: SourceSnapshot) -> Iterator[RawObservation]:
        """
        Stream one CDS delivery as untouched provider values.

        Column names are resolved against the delivered file rather than
        hard-coded, because the CDS harmonised-observation schema names vary
        between dataset versions; an unresolvable schema raises SchemaError.
        """
        frame = self._read_table(snapshot.path)
        columns = _resolve_columns(frame, snapshot.path.name)
        release = snapshot.release

        for row in frame.to_dict("records"):
            value = pd.to_numeric(row.get(columns["value"]), errors="coerce")
            if pd.isna(value):
                continue
            timestamp = pd.to_datetime(row.get(columns["date"]), errors="coerce")
            if pd.isna(timestamp):
                continue

            element = str(row.get(columns["variable"], ""))
            flags = {
                key: row.get(columns[key])
                for key in ("quality_flag", "source_id")
                if columns.get(key)
            }
            yield RawObservation(
                source=self.name,
                source_release=release,
                source_station_id=str(row.get(columns["station_id"], "")),
                obs_date=timestamp.date(),
                element=element,
                value=float(value),
                units=str(row.get(columns["units"], "")) if columns.get("units") else "",
                provider_flags=flags,
                source_record_id=f"{row.get(columns['station_id'])}:"
                f"{timestamp.date().isoformat()}:{element}",
            )


# Candidate column names, most specific first.  The CDS harmonised schema has
# used different spellings across versions, so each role is resolved by trying
# the known names rather than assuming one.
_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "station_id": ("primary_station_id", "station_id", "station_name", "primary_id"),
    "date": ("date_time", "report_timestamp", "time", "date"),
    "variable": ("observed_variable", "variable", "observation_variable"),
    "value": ("observation_value", "value", "observed_value"),
    "units": ("units", "observation_units", "value_units"),
    "lat": ("latitude", "lat", "station_latitude"),
    "lon": ("longitude", "lon", "station_longitude"),
    "elevation": ("height_of_station_above_sea_level", "elevation", "station_elevation"),
    "name": ("station_name", "name"),
    "wmo_id": ("wmo_id", "wmo_number"),
    "wigos_id": ("wigos_station_identifier", "wigos_id"),
    "quality_flag": ("quality_flag", "observation_quality_flag", "quality_control_flag"),
    "source_id": ("source_id", "data_policy_licence", "primary_station_id_scheme"),
}

# Roles without which the file cannot be interpreted at all.  Plan section 1.5
# wants an upstream schema change to stop loudly rather than drop columns.
_REQUIRED_ROLES = ("station_id", "date", "variable", "value", "lat", "lon")


def _resolve_columns(frame: pd.DataFrame, context: str) -> dict[str, str]:
    """Map each logical role onto an actual column name in the delivery."""
    available = {c.lower(): c for c in frame.columns}
    resolved: dict[str, str] = {}
    for role, candidates in _COLUMN_CANDIDATES.items():
        for candidate in candidates:
            if candidate in available:
                resolved[role] = available[candidate]
                break

    missing = [role for role in _REQUIRED_ROLES if role not in resolved]
    if missing:
        raise SchemaError(
            f"{context}: could not resolve required C3S columns {missing}; "
            f"delivered columns were {list(frame.columns)[:30]}"
        )
    return resolved


def _first_non_empty(group: pd.DataFrame, column: str | None) -> str:
    if not column or column not in group.columns:
        return ""
    values = group[column].dropna().astype(str)
    values = values[values.str.strip() != ""]
    return values.iloc[0] if len(values) else ""
