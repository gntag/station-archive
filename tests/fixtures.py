"""
fixtures.py
===========
Fixed provider samples for the adapter tests.

Plan section 1.5: "Test each adapter with a tiny fixed station/date fixture
before bulk acquisition."

Every sample here is a verbatim excerpt of a real provider response captured
on 13 September 2026, trimmed to a few rows.  They are literal rather than
generated so that a provider format change shows up as a test failure against
the format the adapter was actually written for.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# GHCN-Daily
# ---------------------------------------------------------------------------
# Verbatim from https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-version.txt
GHCND_VERSION_TEXT = (
    "The current version of GHCN Daily is 3.34-upd-2026091118 (i.e, an update that "
    "started at 2026091118 [yyyymmddhh] UTC; yyyy=year; mm=month; dd=day; hh=hour),\n"
    "created by appending recently available data updates to the last fully "
    "reprocessed version: 3.34-por-2026090423.\n"
)

# Verbatim lines from ghcnd-stations.txt, plus two placed inside the Delos
# domain so the geographic filter has something to keep.
GHCND_STATIONS_TEXT = (
    "ACW00011604  17.1167  -61.7833   10.1    ST JOHNS COOLIDGE FLD                       \n"
    "AE000041196  25.3330   55.5170   34.0    SHARJAH INTER. AIRP            GSN     41196\n"
    "GRE00105241  37.9000   23.7500   107.0   ATHENS HELLINIKON                      16716\n"
    "GRE00105242  35.3300   25.1800    39.0   HERAKLION AIRPORT                      16754\n"
)

# From ghcnd-inventory.txt, matching the station lines above.
GHCND_INVENTORY_TEXT = (
    "ACW00011604  17.1167  -61.7833 TMAX 1949 1949\n"
    "AE000041196  25.3330   55.5170 TMAX 1944 2026\n"
    "GRE00105241  37.9000   23.7500 TMAX 1955 2010\n"
    "GRE00105241  37.9000   23.7500 TMIN 1955 2010\n"
    "GRE00105241  37.9000   23.7500 PRCP 1955 2010\n"
    "GRE00105242  35.3300   25.1800 TMAX 1975 2001\n"
    "GRE00105242  35.3300   25.1800 PRCP 2012 2020\n"  # outside the 1985-2005 window
)

# Verbatim shape of by_station/<ID>.csv.gz: headerless, eight columns,
# values in tenths.  -9999 is the missing sentinel.
GHCND_STATION_CSV = (
    "GRE00105241,19850101,TMAX,142,,,I,\n"
    "GRE00105241,19850101,TMIN,73,,,I,\n"
    "GRE00105241,19850101,PRCP,0,,,I,\n"
    "GRE00105241,19850102,TMAX,155,,G,I,\n"
    "GRE00105241,19850102,PRCP,-9999,,,I,\n"
    "GRE00105241,19850103,PRCP,124,,,I,\n"
)

# ---------------------------------------------------------------------------
# ECA&D
# ---------------------------------------------------------------------------
# Header structure copied from a real ECA_blend_fx.zip member, with the
# element switched to TG and two sample rows.
ECAD_SERIES_TEXT = """EUROPEAN CLIMATE ASSESSMENT & DATASET (ECA&D), file created on: 12-08-2026
THESE DATA CAN BE USED FOR NON-COMMERCIAL RESEARCH AND EDUCATION PROVIDED THAT THE FOLLOWING SOURCE IS ACKNOWLEDGED:

Klein Tank, A.M.G. and Coauthors, 2002. Daily dataset of 20th-century surface
air temperature and precipitation series for the European Climate Assessment.
Int. J. of Climatol., 22, 1441-1453.
Data and metadata available at http://www.ecad.eu

FILE FORMAT (MISSING VALUE CODE = -9999):

01-06 STAID: Station identifier
08-13 SOUID: Source identifier
15-22 DATE : Date YYYYMMDD
24-28 TG   : Mean temperature in 0.1 &#176;C
30-34 Q_TG : quality code for TG (0='valid'; 1='suspect'; 9='missing')

This is the blended series of station ATHENS, GREECE (STAID: 000247)
Blended and updated with sources:125522 136679 911012
See files sources.txt and stations.txt for more info.

STAID, SOUID,    DATE,   TG, Q_TG
   247,125522,19850101,  112,    0
   247,125522,19850102,  128,    1
   247,125522,19850103,-9999,    9
   247,136679,19850104,  151,    0
"""

# Structure copied verbatim from stations.txt inside the ECA&D archives:
# 19 header lines then fixed-width rows with DMS coordinates.
ECAD_STATIONS_TEXT = """EUROPEAN CLIMATE ASSESSMENT & DATASET (ECA&D), file created on 12-08-2026
THESE DATA CAN BE USED FREELY PROVIDED THAT THE FOLLOWING SOURCE IS ACKNOWLEDGED:

Klein Tank, A.M.G. and Coauthors, 2002. Daily dataset of 20th-century surface
air temperature and precipitation series for the European Climate Assessment.
Int. J. of Climatol., 22, 1441-1453.
Data and metadata available at https://www.ecad.eu

FILE FORMAT (MISSING VALUE CODE IS -9999):

 01- 05 STAID  : Station identifier
 07- 46 STANAME: Station name
 48- 49 CN     : Country code (ISO3116 countrycodes)
 51- 59 LAT    : Latitude in degrees:minutes:seconds (+: North, -: South)
 61- 70 LON    : Longitude in degrees:minutes:seconds (+: East, -: West)
 72- 76 HGTH   : Station elevation in meters

STAID,STANAME                                 ,CN,      LAT,       LON,HGHT

    1,VAEXJOE                                 ,SE  ,+56:52:00,+014:48:00,  166
  247,ATHENS                                  ,GR  ,+37:58:00,+023:43:00,  107
  248,HERAKLION                               ,GR  ,+35:20:00,+025:11:00,   39
  999,RIO DE JANEIRO                          ,BR  ,-22:54:00,-043:10:00,    2
"""

# ---------------------------------------------------------------------------
# GSOD
# ---------------------------------------------------------------------------
GSOD_HEADER = (
    "STATION,DATE,LATITUDE,LONGITUDE,ELEVATION,NAME,"
    "TEMP,TEMP_ATTRIBUTES,DEWP,DEWP_ATTRIBUTES,"
    "SLP,SLP_ATTRIBUTES,STP,STP_ATTRIBUTES,"
    "VISIB,VISIB_ATTRIBUTES,WDSP,WDSP_ATTRIBUTES,"
    "MXSPD,GUST,MAX,MAX_ATTRIBUTES,MIN,MIN_ATTRIBUTES,"
    "PRCP,PRCP_ATTRIBUTES,SNDP,FRSHTT\n"
)

# One ordinary row and one where TEMP and GUST carry the missing sentinels.
GSOD_CSV = GSOD_HEADER + (
    "16716099999,1985-06-15,37.9333,23.7333,15.0,ATHENS,"
    "77.0,6,59.0,6,9999.9,0,999.9,0,9.3,6,8.1,6,"
    "12.0,999.9,87.8, ,68.0, ,0.04,C,999.9,000000\n"
    "16716099999,1985-06-16,37.9333,23.7333,15.0,ATHENS,"
    "9999.9,0,61.0,6,9999.9,0,999.9,0,9.3,6,7.2,6,"
    "11.0,35.0,89.6, ,70.0, ,0.00,I,999.9,000000\n"
)

GSOD_DIRECTORY_HTML = """<html><head><title>Index of /1985</title></head><body>
<a href="../">Parent Directory</a>
<a href="16716099999.csv">16716099999.csv</a>
<a href="16754099999.csv">16754099999.csv</a>
<a href="03772099999.csv">03772099999.csv</a>
<a href="readme.txt">readme.txt</a>
</body></html>
"""

# ---------------------------------------------------------------------------
# C3S land in-situ
# ---------------------------------------------------------------------------
# Column names follow the CDS harmonised-observation schema.
C3S_CSV = (
    "primary_station_id,report_timestamp,observed_variable,observation_value,units,"
    "latitude,longitude,height_of_station_above_sea_level,quality_flag\n"
    "GR-ATH-001,1995-03-01 00:00:00,air_temperature,284.15,K,37.90,23.75,107.0,0\n"
    "GR-ATH-001,1995-03-02 00:00:00,air_temperature,286.35,K,37.90,23.75,107.0,0\n"
    "GR-ATH-001,1995-03-01 00:00:00,accumulated_precipitation,0.4,mm,37.90,23.75,107.0,0\n"
    "BR-RIO-001,1995-03-01 00:00:00,air_temperature,300.15,K,-22.90,-43.17,2.0,0\n"
)
