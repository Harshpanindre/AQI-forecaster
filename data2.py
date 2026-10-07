"""
BENGALURU AQI / PM2.5 DATASET BUILDER
======================================

Goal:
    1. Find all nearby Bengaluru OpenAQ stations.
    2. Find stations that have a PM2.5 sensor.
    3. Evaluate their actual hourly data coverage.
    4. Select the station with the best continuity.
    5. Download PM2.5 hourly data from that station.
    6. Download hourly weather data from Open-Meteo.
    7. Clean obvious physical errors.
    8. Join AQI + weather.
    9. Audit missing timestamps and gaps.
    10. Save the final dataset.

IMPORTANT:
    We do NOT blindly interpolate multi-day gaps.

This dataset is intended for:
    - Time-series analysis
    - PM2.5 forecasting
    - LSTM / GRU
    - XGBoost / Random Forest
    - 24-hour / 48-hour forecasting

Install:
    pip install requests pandas numpy python-dateutil
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# OpenAQ API key
# ------------------------------------------------------------
#
# DO NOT put your actual API key directly in the code.
#
# Windows PowerShell:
#
#     $env:OPENAQ_API_KEY="YOUR_API_KEY"
#
# Windows CMD:
#
#     set OPENAQ_API_KEY=YOUR_API_KEY
#
# Then this script will read it automatically.
#
# ------------------------------------------------------------

OPENAQ_API_KEY = "152a83a5759624f90a3562fd2057bfe700ad2545d3076baefe76480604be996a"

if not OPENAQ_API_KEY:
    raise RuntimeError(
        "OPENAQ_API_KEY environment variable is not set.\n"
        "Set your OpenAQ API key before running the script."
    )


# ------------------------------------------------------------
# City
# ------------------------------------------------------------

CITY_NAME = "Bengaluru"

LAT = 12.9716
LON = 77.5946

# OpenAQ allows radius up to 25 km.
RADIUS_M = 25000


# ------------------------------------------------------------
# Date range
# ------------------------------------------------------------

START_DATE = "2019-01-01"
END_DATE = "2022-06-30"


# ------------------------------------------------------------
# Station selection
# ------------------------------------------------------------

# IMPORTANT:
#
# None means:
#     "Do NOT force a station."
#
# The program will inspect all nearby stations and select
# the best PM2.5 station automatically.
#
STATION_NAME = None


# ------------------------------------------------------------
# Station evaluation period
# ------------------------------------------------------------
#
# Instead of downloading 3.5 years from EVERY station,
# first test stations using this period.
#
# This saves API requests.
#
# We then download the COMPLETE period only for the
# selected station.
#
# ------------------------------------------------------------

EVALUATION_START = "2021-01-01"
EVALUATION_END = "2021-12-31"


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

OUTPUT_CSV = "bengaluru_best_station_aqi_weather.csv"

STATION_REPORT_CSV = "bengaluru_station_comparison.csv"


# ------------------------------------------------------------
# Pollutant
# ------------------------------------------------------------

TARGET_PARAMETER = "pm25"


# ------------------------------------------------------------
# OpenAQ configuration
# ------------------------------------------------------------

OPENAQ_BASE = "https://api.openaq.org/v3"

HEADERS = {
    "X-API-Key": OPENAQ_API_KEY
}


# ------------------------------------------------------------
# Open-Meteo
# ------------------------------------------------------------

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update(HEADERS)


# ============================================================
# GENERAL HTTP REQUEST FUNCTION
# ============================================================

def get_with_retries(
    url,
    params=None,
    max_retries=5,
    timeout=(10, 60)
):
    """
    Make a GET request with retries.

    Handles:
        - connection errors
        - timeouts
        - 429 rate limits
        - 500-level server errors
    """

    for attempt in range(max_retries):

        try:

            response = session.get(
                url,
                params=params,
                timeout=timeout
            )

            # Rate limit
            if response.status_code == 429:

                wait_time = 5 * (attempt + 1)

                print(
                    f"Rate limited. Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            # Temporary server errors
            if response.status_code >= 500:

                wait_time = 3 * (attempt + 1)

                print(
                    f"Server error {response.status_code}. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            response.raise_for_status()

            return response

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout
        ) as e:

            if attempt == max_retries - 1:
                raise

            wait_time = 3 * (attempt + 1)

            print(
                f"Request failed: {e}. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError("Request failed after retries.")


# ============================================================
# STEP 1
# FIND NEARBY OPENAQ LOCATIONS
# ============================================================

def get_nearby_locations():

    print("\n" + "=" * 70)
    print("STEP 1: FINDING BENGALURU AIR QUALITY STATIONS")
    print("=" * 70)

    url = f"{OPENAQ_BASE}/locations"

    params = {
        "coordinates": f"{LAT},{LON}",
        "radius": RADIUS_M,
        "limit": 100
    }

    response = get_with_retries(url, params)

    locations = response.json().get("results", [])

    print(f"\nStations found: {len(locations)}")

    return locations


# ============================================================
# STEP 2
# GET SENSORS FOR LOCATION
# ============================================================

def get_sensors_for_location(location_id):

    url = (
        f"{OPENAQ_BASE}/locations/"
        f"{location_id}/sensors"
    )

    response = get_with_retries(url)

    return response.json().get("results", [])


# ============================================================
# STEP 3
# FIND PM2.5 SENSOR
# ============================================================

def find_pm25_sensor(sensors):

    for sensor in sensors:

        parameter = sensor.get("parameter", {})

        name = parameter.get("name", "").lower()

        # OpenAQ generally uses pm25 as parameter name.
        if name == "pm25":

            return sensor

    return None


# ============================================================
# STEP 4
# DISCOVER PM2.5 STATIONS
# ============================================================

def discover_pm25_stations(locations):

    print("\n" + "=" * 70)
    print("STEP 2: FINDING STATIONS WITH PM2.5")
    print("=" * 70)

    candidates = []

    for index, location in enumerate(locations, start=1):

        location_id = location.get("id")

        station_name = location.get(
            "name",
            f"Location_{location_id}"
        )

        print(
            f"\n[{index}/{len(locations)}] "
            f"{station_name}"
        )

        # ----------------------------------------------------
        # Optional station filter
        # ----------------------------------------------------

        if STATION_NAME:

            if STATION_NAME.lower() not in station_name.lower():

                continue

        try:

            sensors = get_sensors_for_location(
                location_id
            )

        except Exception as e:

            print(
                f"Could not read sensors: {e}"
            )

            continue

        pm25_sensor = find_pm25_sensor(sensors)

        if pm25_sensor is None:

            print("    No PM2.5 sensor.")

            continue

        sensor_id = pm25_sensor.get("id")

        coordinates = location.get(
            "coordinates",
            {}
        )

        station_lat = coordinates.get(
            "latitude",
            LAT
        )

        station_lon = coordinates.get(
            "longitude",
            LON
        )

        candidates.append(
            {
                "location_id": location_id,
                "station_name": station_name,
                "sensor_id": sensor_id,
                "latitude": station_lat,
                "longitude": station_lon,
                "timezone": location.get(
                    "timezone"
                ),
            }
        )

        print(
            f"    PM2.5 sensor found: {sensor_id}"
        )

    print(
        f"\nPM2.5 candidate stations: "
        f"{len(candidates)}"
    )

    return candidates


# ============================================================
# STEP 5
# EXTRACT TIMESTAMP FROM OPENAQ RESPONSE
# ============================================================

def extract_timestamp(record):

    # --------------------------------------------
    # /hours endpoint
    # --------------------------------------------

    if "datetime" in record:

        datetime_value = record["datetime"]

        if isinstance(datetime_value, dict):

            return datetime_value.get("utc")

        return datetime_value

    # --------------------------------------------
    # /measurements endpoint
    # --------------------------------------------

    if "period" in record:

        period = record["period"]

        if isinstance(period, dict):

            datetime_from = period.get(
                "datetimeFrom"
            )

            if isinstance(
                datetime_from,
                dict
            ):

                return datetime_from.get("utc")

            return datetime_from

    return None


# ============================================================
# STEP 6
# EXTRACT VALUE
# ============================================================

def extract_value(record):

    value = record.get("value")

    try:

        return float(value)

    except (TypeError, ValueError):

        return np.nan


# ============================================================
# STEP 7
# FETCH HOURLY DATA
# ============================================================

def fetch_hourly_sensor_data(
    sensor_id,
    date_from,
    date_to
):
    """
    Fetch hourly PM2.5 data.

    Primary:
        /v3/sensors/{sensor_id}/hours

    This is preferred because OpenAQ already provides
    hourly aggregated values.

    Falls back to raw measurements if necessary.
    """

    print(
        f"\nFetching sensor {sensor_id}"
    )

    url = (
        f"{OPENAQ_BASE}/sensors/"
        f"{sensor_id}/hours"
    )

    all_records = []

    page = 1

    while True:

        params = {
            "datetime_from": (
                f"{date_from}T00:00:00Z"
            ),
            "datetime_to": (
                f"{date_to}T23:59:59Z"
            ),
            "limit": 1000,
            "page": page
        }

        try:

            response = get_with_retries(
                url,
                params
            )

        except Exception as e:

            print(
                f"Hourly endpoint failed: {e}"
            )

            print(
                "Trying raw measurements..."
            )

            return fetch_raw_measurements(
                sensor_id,
                date_from,
                date_to
            )

        records = response.json().get(
            "results",
            []
        )

        if not records:

            break

        all_records.extend(records)

        print(
            f"    Page {page}: "
            f"{len(records)} rows"
        )

        if len(records) < 1000:

            break

        page += 1

        time.sleep(0.2)

    # --------------------------------------------------------
    # Convert to dataframe
    # --------------------------------------------------------

    if not all_records:

        print(
            "No hourly records found."
        )

        return pd.DataFrame(
            columns=[
                "timestamp",
                "pm25"
            ]
        )

    data = []

    for record in all_records:

        timestamp = extract_timestamp(record)

        value = extract_value(record)

        if timestamp is None:

            continue

        data.append(
            {
                "timestamp": timestamp,
                "pm25": value
            }
        )

    df = pd.DataFrame(data)

    if df.empty:

        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="first"
    )

    # Convert UTC to timezone-naive UTC
    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_convert(None)
    )

    return df


# ============================================================
# STEP 8
# RAW MEASUREMENT FALLBACK
# ============================================================

def fetch_raw_measurements(
    sensor_id,
    date_from,
    date_to
):

    print(
        "Using raw measurement endpoint..."
    )

    url = (
        f"{OPENAQ_BASE}/sensors/"
        f"{sensor_id}/measurements"
    )

    start = pd.Timestamp(
        date_from
    )

    end = pd.Timestamp(
        date_to
    )

    all_records = []

    current = start

    # --------------------------------------------------------
    # Fetch one month at a time
    # --------------------------------------------------------

    while current <= end:

        chunk_end = min(
            current + pd.offsets.MonthEnd(0),
            end
        )

        page = 1

        print(
            f"    Fetching "
            f"{current.date()} -> "
            f"{chunk_end.date()}"
        )

        while True:

            params = {

                "datetime_from": (
                    current.strftime(
                        "%Y-%m-%dT00:00:00Z"
                    )
                ),

                "datetime_to": (
                    chunk_end.strftime(
                        "%Y-%m-%dT23:59:59Z"
                    )
                ),

                "limit": 1000,

                "page": page
            }

            try:

                response = get_with_retries(
                    url,
                    params
                )

            except Exception as e:

                print(
                    f"    Failed: {e}"
                )

                break

            records = response.json().get(
                "results",
                []
            )

            if not records:

                break

            all_records.extend(records)

            if len(records) < 1000:

                break

            page += 1

            time.sleep(0.2)

        current = (
            chunk_end
            + pd.Timedelta(days=1)
        )

    # --------------------------------------------------------
    # Convert raw records
    # --------------------------------------------------------

    if not all_records:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "pm25"
            ]
        )

    data = []

    for record in all_records:

        timestamp = extract_timestamp(record)

        value = extract_value(record)

        if timestamp is None:

            continue

        data.append(
            {
                "timestamp": timestamp,
                "pm25": value
            }
        )

    df = pd.DataFrame(data)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # Convert raw measurements to hourly averages
    # --------------------------------------------------------

    df = (
        df.set_index("timestamp")
        .sort_index()
        .resample("1h")
        ["pm25"]
        .mean()
        .reset_index()
    )

    # Convert UTC timezone to naive
    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_convert(None)
    )

    return df


# ============================================================
# STEP 9
# CALCULATE DATA COVERAGE
# ============================================================

def calculate_coverage(df):

    if df.empty:

        return {
            "records": 0,
            "start": None,
            "end": None,
            "expected_hours": 0,
            "missing_hours": 0,
            "coverage_percent": 0,
            "max_gap_hours": 0,
            "longest_continuous_hours": 0
        }

    timestamps = (
        pd.DatetimeIndex(
            df["timestamp"]
        )
        .sort_values()
        .drop_duplicates()
    )

    start = timestamps.min().floor("h")

    end = timestamps.max().floor("h")

    expected = pd.date_range(
        start=start,
        end=end,
        freq="1h"
    )

    missing = expected.difference(
        timestamps
    )

    # --------------------------------------------------------
    # Calculate gaps
    # --------------------------------------------------------

    differences = timestamps.to_series().diff()

    max_gap = differences.max()

    if pd.isna(max_gap):

        max_gap_hours = 0

    else:

        max_gap_hours = (
            max_gap.total_seconds()
            / 3600
        )

    # --------------------------------------------------------
    # Calculate longest continuous sequence
    # --------------------------------------------------------

    gap_flags = (
        differences
        != pd.Timedelta(hours=1)
    )

    segment_ids = gap_flags.cumsum()

    segment_lengths = (
        timestamps.to_series()
        .groupby(segment_ids)
        .size()
    )

    if len(segment_lengths) > 0:

        longest_segment = (
            segment_lengths.max()
        )

    else:

        longest_segment = 0

    expected_hours = len(expected)

    if expected_hours > 0:

        coverage_percent = (
            len(timestamps)
            / expected_hours
            * 100
        )

    else:

        coverage_percent = 0

    return {

        "records": len(timestamps),

        "start": start,

        "end": end,

        "expected_hours": expected_hours,

        "missing_hours": len(missing),

        "coverage_percent": (
            round(
                coverage_percent,
                2
            )
        ),

        "max_gap_hours": (
            round(
                max_gap_hours,
                2
            )
        ),

        "longest_continuous_hours": int(
            longest_segment
        )
    }


# ============================================================
# STEP 10
# EVALUATE ALL STATIONS
# ============================================================

def evaluate_stations(candidates):

    print("\n" + "=" * 70)
    print("STEP 3: EVALUATING STATION QUALITY")
    print("=" * 70)

    reports = []

    for index, station in enumerate(
        candidates,
        start=1
    ):

        print(
            f"\n[{index}/{len(candidates)}] "
            f"{station['station_name']}"
        )

        print(
            f"    Sensor ID: "
            f"{station['sensor_id']}"
        )

        try:

            df = fetch_hourly_sensor_data(
                station["sensor_id"],
                EVALUATION_START,
                EVALUATION_END
            )

            coverage = calculate_coverage(
                df
            )

            report = {
                **station,
                **coverage
            }

            reports.append(report)

            print(
                f"    Records: "
                f"{coverage['records']}"
            )

            print(
                f"    Coverage: "
                f"{coverage['coverage_percent']}%"
            )

            print(
                f"    Longest continuous: "
                f"{coverage['longest_continuous_hours']} "
                f"hours"
            )

            print(
                f"    Maximum gap: "
                f"{coverage['max_gap_hours']} hours"
            )

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

        # Avoid hammering API
        time.sleep(0.5)

    if not reports:

        raise RuntimeError(
            "No station could be evaluated."
        )

    report_df = pd.DataFrame(
        reports
    )

    # --------------------------------------------------------
    # SAVE STATION COMPARISON
    # --------------------------------------------------------

    report_df = report_df.sort_values(
        by=[
            "coverage_percent",
            "longest_continuous_hours"
        ],
        ascending=False
    )

    report_df.to_csv(
        STATION_REPORT_CSV,
        index=False
    )

    print(
        f"\nStation report saved to: "
        f"{STATION_REPORT_CSV}"
    )

    return report_df


# ============================================================
# STEP 11
# SELECT BEST STATION
# ============================================================

def select_best_station(report_df):

    print("\n" + "=" * 70)
    print("STEP 4: SELECTING BEST STATION")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Coverage is the first priority.
    # Longest continuous segment is the second priority.
    #
    # We don't simply select the station with the newest
    # measurement.
    # --------------------------------------------------------

    valid = report_df[
        report_df["records"] > 0
    ].copy()

    if valid.empty:

        raise RuntimeError(
            "No station has usable PM2.5 data."
        )

    valid = valid.sort_values(
        by=[
            "coverage_percent",
            "longest_continuous_hours"
        ],
        ascending=False
    )

    best = valid.iloc[0]

    print(
        "\nBEST STATION"
    )

    print(
        f"Name: {best['station_name']}"
    )

    print(
        f"Location ID: "
        f"{best['location_id']}"
    )

    print(
        f"Sensor ID: "
        f"{best['sensor_id']}"
    )

    print(
        f"Coverage: "
        f"{best['coverage_percent']}%"
    )

    print(
        f"Longest continuous segment: "
        f"{best['longest_continuous_hours']} hours"
    )

    print(
        f"Maximum gap: "
        f"{best['max_gap_hours']} hours"
    )

    return best


# ============================================================
# STEP 12
# DOWNLOAD FINAL PM2.5 DATA
# ============================================================

def download_final_pm25(best_station):

    print("\n" + "=" * 70)
    print("STEP 5: DOWNLOADING FINAL PM2.5 DATA")
    print("=" * 70)

    sensor_id = int(
        best_station["sensor_id"]
    )

    df = fetch_hourly_sensor_data(
        sensor_id,
        START_DATE,
        END_DATE
    )

    if df.empty:

        raise RuntimeError(
            "Final PM2.5 dataset is empty."
        )

    print(
        f"\nDownloaded PM2.5 rows: "
        f"{len(df)}"
    )

    return df


# ============================================================
# STEP 13
# DOWNLOAD OPEN-METEO WEATHER
# ============================================================

def fetch_weather():

    print("\n" + "=" * 70)
    print("STEP 6: DOWNLOADING WEATHER DATA")
    print("=" * 70)

    params = {

        "latitude": LAT,

        "longitude": LON,

        "start_date": START_DATE,

        "end_date": END_DATE,

        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation"
            ]
        ),

        # Return UTC timestamps
        "timezone": "GMT",

        # We want wind speed in m/s
        "wind_speed_unit": "ms",

        # Precipitation in mm
        "precipitation_unit": "mm"
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=(10, 60)
    )

    response.raise_for_status()

    data = response.json()

    hourly = data.get(
        "hourly",
        {}
    )

    if not hourly:

        raise RuntimeError(
            "Open-Meteo returned no hourly data."
        )

    weather = pd.DataFrame(
        {
            "timestamp": hourly["time"],

            "temperature": hourly[
                "temperature_2m"
            ],

            "humidity": hourly[
                "relative_humidity_2m"
            ],

            "wind_speed": hourly[
                "wind_speed_10m"
            ],

            "wind_direction": hourly[
                "wind_direction_10m"
            ],

            "precipitation": hourly[
                "precipitation"
            ]
        }
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        utc=True,
        errors="coerce"
    )

    weather["timestamp"] = (
        weather["timestamp"]
        .dt.tz_convert(None)
    )

    weather = weather.sort_values(
        "timestamp"
    )

    weather = weather.drop_duplicates(
        "timestamp"
    )

    print(
        f"Weather rows: {len(weather)}"
    )

    return weather


# ============================================================
# STEP 14
# CLEAN PM2.5
# ============================================================

def clean_pm25(df):

    print("\n" + "=" * 70)
    print("STEP 7: CLEANING PM2.5")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Ensure numeric
    # --------------------------------------------------------

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Count negative values
    # --------------------------------------------------------

    negative_count = (
        df["pm25"] < 0
    ).sum()

    print(
        f"Negative PM2.5 values: "
        f"{negative_count}"
    )

    # --------------------------------------------------------
    # Negative PM2.5 is physically invalid.
    # Convert to NaN.
    # --------------------------------------------------------

    df.loc[
        df["pm25"] < 0,
        "pm25"
    ] = np.nan

    # --------------------------------------------------------
    # Remove impossible extreme values.
    #
    # Do NOT use IQR to automatically remove high pollution
    # values because genuine pollution spikes are important.
    #
    # We only remove obviously impossible negative values
    # here.
    # --------------------------------------------------------

    return df


# ============================================================
# STEP 15
# CLEAN WEATHER
# ============================================================

def clean_weather(weather):

    print("\n" + "=" * 70)
    print("STEP 8: VALIDATING WEATHER")
    print("=" * 70)

    weather = weather.copy()

    # --------------------------------------------------------
    # Humidity
    # --------------------------------------------------------

    invalid_humidity = (
        (weather["humidity"] < 0)
        |
        (weather["humidity"] > 100)
    )

    print(
        "Invalid humidity values:",
        invalid_humidity.sum()
    )

    weather.loc[
        invalid_humidity,
        "humidity"
    ] = np.nan

    # --------------------------------------------------------
    # Wind speed
    # --------------------------------------------------------

    invalid_wind = (
        weather["wind_speed"] < 0
    )

    print(
        "Invalid wind speed values:",
        invalid_wind.sum()
    )

    weather.loc[
        invalid_wind,
        "wind_speed"
    ] = np.nan

    # --------------------------------------------------------
    # Wind direction
    # --------------------------------------------------------

    invalid_direction = (
        (weather["wind_direction"] < 0)
        |
        (weather["wind_direction"] > 360)
    )

    print(
        "Invalid wind direction values:",
        invalid_direction.sum()
    )

    weather.loc[
        invalid_direction,
        "wind_direction"
    ] = np.nan

    # --------------------------------------------------------
    # Precipitation
    # --------------------------------------------------------

    invalid_precipitation = (
        weather["precipitation"] < 0
    )

    print(
        "Invalid precipitation values:",
        invalid_precipitation.sum()
    )

    weather.loc[
        invalid_precipitation,
        "precipitation"
    ] = np.nan

    return weather


# ============================================================
# STEP 16
# JOIN PM2.5 + WEATHER
# ============================================================

def combine_data(pm25, weather):

    print("\n" + "=" * 70)
    print("STEP 9: COMBINING AIR QUALITY + WEATHER")
    print("=" * 70)

    pm25 = pm25.copy()

    weather = weather.copy()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use INNER JOIN.
    #
    # Only timestamps that exist in both datasets survive.
    #
    # But BEFORE joining, we already audited PM2.5
    # independently.
    # --------------------------------------------------------

    combined = pd.merge(
        pm25,
        weather,
        on="timestamp",
        how="inner"
    )

    combined = combined.sort_values(
        "timestamp"
    )

    combined = combined.drop_duplicates(
        "timestamp"
    )

    print(
        f"Combined rows: "
        f"{len(combined)}"
    )

    return combined


# ============================================================
# STEP 17
# CREATE CONTINUOUS SEGMENTS
# ============================================================

def create_segments(df):

    print("\n" + "=" * 70)
    print("STEP 10: IDENTIFYING CONTINUOUS TIME SEGMENTS")
    print("=" * 70)

    df = df.copy()

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    time_difference = (
        df["timestamp"]
        .diff()
    )

    # --------------------------------------------------------
    # New segment whenever timestamp is not exactly one hour
    # after previous timestamp.
    # --------------------------------------------------------

    new_segment = (
        time_difference
        != pd.Timedelta(hours=1)
    )

    df["segment"] = (
        new_segment
        .cumsum()
    )

    segment_sizes = (
        df.groupby("segment")
        .size()
    )

    print(
        f"Total continuous segments: "
        f"{len(segment_sizes)}"
    )

    print(
        f"Longest segment: "
        f"{segment_sizes.max()} hours"
    )

    print(
        f"Median segment length: "
        f"{segment_sizes.median()} hours"
    )

    return df


# ============================================================
# STEP 18
# DATASET AUDIT
# ============================================================

def audit_final_dataset(df):

    print("\n" + "=" * 70)
    print("STEP 11: FINAL DATASET AUDIT")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())

    print("\nLast rows:")
    print(df.tail())

    print("\nData types:")
    print(df.dtypes)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")
    print(
        df.isna().sum()
    )

    # --------------------------------------------------------
    # Duplicate timestamps
    # --------------------------------------------------------

    duplicate_timestamps = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    print(
        "\nDuplicate timestamps:",
        duplicate_timestamps
    )

    # --------------------------------------------------------
    # Timestamp range
    # --------------------------------------------------------

    print(
        "\nTimestamp range:"
    )

    print(
        "Start:",
        df["timestamp"].min()
    )

    print(
        "End:",
        df["timestamp"].max()
    )

    # --------------------------------------------------------
    # Timestamp gaps
    # --------------------------------------------------------

    differences = (
        df["timestamp"]
        .sort_values()
        .diff()
    )

    print(
        "\nTimestamp difference distribution:"
    )

    print(
        differences.value_counts()
        .head(10)
    )

    # --------------------------------------------------------
    # Large gaps
    # --------------------------------------------------------

    large_gaps = differences[
        differences > pd.Timedelta(hours=24)
    ]

    print(
        "\nGaps greater than 24 hours:",
        len(large_gaps)
    )

    if len(large_gaps) > 0:

        print(
            "\nLargest gaps:"
        )

        print(
            large_gaps
            .sort_values(
                ascending=False
            )
            .head(10)
        )

    # --------------------------------------------------------
    # PM2.5 statistics
    # --------------------------------------------------------

    print(
        "\nPM2.5 statistics:"
    )

    print(
        df["pm25"].describe()
    )


# ============================================================
# STEP 19
# SAVE DATA
# ============================================================

def save_dataset(df):

    print("\n" + "=" * 70)
    print("STEP 12: SAVING DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # We do NOT interpolate here.
    #
    # Missing PM2.5 values are intentionally preserved.
    #
    # Why?
    #
    # Large gaps should not be artificially filled.
    #
    # We will handle small missing values later during
    # preprocessing, after understanding the gap structure.
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nDataset saved successfully:"
    )

    print(
        OUTPUT_CSV
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BENGALURU PM2.5 FORECASTING DATA PIPELINE")
    print("=" * 70)

    print(
        f"\nCity: {CITY_NAME}"
    )

    print(
        f"Full date range: "
        f"{START_DATE} -> {END_DATE}"
    )

    # --------------------------------------------------------
    # 1. Find stations
    # --------------------------------------------------------

    locations = get_nearby_locations()

    # --------------------------------------------------------
    # 2. Find PM2.5 stations
    # --------------------------------------------------------

    candidates = discover_pm25_stations(
        locations
    )

    if not candidates:

        raise RuntimeError(
            "No PM2.5 stations found."
        )

    # --------------------------------------------------------
    # 3. Evaluate station quality
    # --------------------------------------------------------

    station_report = evaluate_stations(
        candidates
    )

    # --------------------------------------------------------
    # 4. Select best station
    # --------------------------------------------------------

    best_station = select_best_station(
        station_report
    )

    # --------------------------------------------------------
    # 5. Download final PM2.5 data
    # --------------------------------------------------------

    pm25 = download_final_pm25(
        best_station
    )

    # --------------------------------------------------------
    # 6. Clean PM2.5
    # --------------------------------------------------------

    pm25 = clean_pm25(
        pm25
    )

    # --------------------------------------------------------
    # 7. Download weather
    # --------------------------------------------------------

    weather = fetch_weather()

    # --------------------------------------------------------
    # 8. Validate weather
    # --------------------------------------------------------

    weather = clean_weather(
        weather
    )

    # --------------------------------------------------------
    # 9. Combine
    # --------------------------------------------------------

    combined = combine_data(
        pm25,
        weather
    )

    # --------------------------------------------------------
    # 10. Create continuous segments
    # --------------------------------------------------------

    combined = create_segments(
        combined
    )

    # --------------------------------------------------------
    # 11. Audit
    # --------------------------------------------------------

    audit_final_dataset(
        combined
    )

    # --------------------------------------------------------
    # 12. Save
    # --------------------------------------------------------

    save_dataset(
        combined
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        "\nBest station:"
    )

    print(
        best_station["station_name"]
    )

    print(
        "\nFinal dataset:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "\nStation comparison:"
    )

    print(
        STATION_REPORT_CSV
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()