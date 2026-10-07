"""
Fetch historical air quality data (OpenAQ) + weather data (Open-Meteo)
for a given city, join them on hourly timestamps, and save to CSV.

Output: a single CSV with hourly rows of:
  timestamp, pm25, pm10, no2, so2, co, o3,
  temperature, humidity, wind_speed, wind_direction, precipitation

SETUP:
  pip install requests pandas python-dateutil

  1. Get a free OpenAQ API key: https://explore.openaq.org/register
  2. Paste it into OPENAQ_API_KEY below.
  3. Adjust CITY_NAME, LAT, LON, START_DATE, END_DATE as needed.

Notes:
  - OpenAQ v3 requires an API key (free tier) and organizes data as
    location -> sensors (one sensor per pollutant) -> measurements.
  - Open-Meteo's archive API needs no key at all.
  - Pollutant coverage varies by location; not every station reports
    every pollutant. The script fills missing ones with NaN.
"""

import time
from datetime import datetime

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
OPENAQ_API_KEY = "152a83a5759624f90a3562fd2057bfe700ad2545d3076baefe76480604be996a"   # https://explore.openaq.org/register
CITY_NAME = "Bengaluru"
LAT, LON = 12.9716, 77.5946              # Bengaluru coordinates
RADIUS_M = 25000                          # search radius for stations, in meters
STATION_NAME = "Peenya"                   # substring match — only this station is used
START_DATE = "2019-01-01"  # 3.5 years (accounts for sensor downtime, yields ~8,000-10,000 rows)
END_DATE = "2022-06-30"
OUTPUT_CSV = f"{CITY_NAME.lower()}_aqi_weather.csv"

POLLUTANTS_OF_INTEREST = {"pm25"}

OPENAQ_BASE = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": OPENAQ_API_KEY}


# ---------------------------------------------------------------------------
# STEP 1 — find nearby OpenAQ monitoring stations
# ---------------------------------------------------------------------------
def get_nearby_locations(lat, lon, radius_m):
    url = f"{OPENAQ_BASE}/locations"
    params = {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": 100}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=(10, 30))
    resp.raise_for_status()
    return resp.json()["results"]


# ---------------------------------------------------------------------------
# STEP 2 — list sensors (one per pollutant) at each location
# ---------------------------------------------------------------------------
def get_sensors_for_location(location_id):
    url = f"{OPENAQ_BASE}/locations/{location_id}/sensors"
    resp = requests.get(url, headers=HEADERS, timeout=(10, 30))
    resp.raise_for_status()
    return resp.json()["results"]


# ---------------------------------------------------------------------------
# STEP 3 — pull hourly measurements for a single sensor, with pagination,
# retries, and monthly chunking (large single requests time out with a 408)
# ---------------------------------------------------------------------------
def _get_with_retries(url, params, max_retries=4):
    """GET with exponential backoff on timeouts/5xx/408/429 errors.
    Uses a (connect, read) timeout tuple: if the server doesn't even
    accept the connection within 10s, or doesn't send data within 30s,
    it fails fast instead of hanging silently."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url, headers=HEADERS, params=params, timeout=(10, 30)
            )
            if resp.status_code in (408, 429, 500, 502, 503, 504):
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} retryable error", response=resp
                )
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError) as e:
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s
            print(f"    ...request failed ({e}), retrying in {wait}s "
                  f"[attempt {attempt + 1}/{max_retries}]")
            time.sleep(wait)
    # final attempt — let it raise naturally if it still fails
    resp = requests.get(url, headers=HEADERS, params=params, timeout=(10, 30))
    resp.raise_for_status()
    return resp


def _month_chunks(date_from, date_to):
    """Split a date range into month-long (start, end) string pairs."""
    start = pd.Timestamp(date_from)
    end = pd.Timestamp(date_to)
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min((cur + pd.offsets.MonthEnd(0)), end)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def _fetch_hourly_endpoint(sensor_id, date_from, date_to):
    """Try the server-side /sensors/{id}/hours endpoint.
    Returns rows on success (~720 rows per month)."""
    url = f"{OPENAQ_BASE}/sensors/{sensor_id}/hours"
    all_rows = []

    for chunk_from, chunk_to in _month_chunks(date_from, date_to):
        print(f"    -> [hours endpoint] {chunk_from} to {chunk_to} ...")
        page = 1
        while True:
            params = {
                "datetime_from": chunk_from,
                "datetime_to":   chunk_to,
                "date_from":     chunk_from,
                "date_to":       chunk_to,
                "limit":         1000,
                "page":          page,
            }
            try:
                resp = _get_with_retries(url, params)
            except requests.exceptions.RequestException as e:
                print(f"    !! error on sensor {sensor_id} "
                      f"chunk {chunk_from}..{chunk_to} page {page}: {e}")
                break
            results = resp.json().get("results", [])
            if not results:
                break
            all_rows.extend(results)
            print(f"       page {page}: {len(results)} rows "
                  f"(running total: {len(all_rows)})")
            if len(results) < 1000:
                break
            page += 1
            time.sleep(0.3)

    return all_rows


def _fetch_raw_measurements(sensor_id, date_from, date_to):
    """Fetch raw /measurements readings.
    Will be aggregated to hourly values client-side."""
    url = f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements"
    all_rows = []

    for chunk_from, chunk_to in _month_chunks(date_from, date_to):
        print(f"    -> [raw endpoint] {chunk_from} to {chunk_to} ...")
        page = 1
        while True:
            params = {
                "datetime_from": chunk_from,
                "datetime_to":   chunk_to,
                "date_from":     chunk_from,
                "date_to":       chunk_to,
                "limit":         1000,
                "page":          page,
            }
            try:
                resp = _get_with_retries(url, params)
            except requests.exceptions.RequestException as e:
                print(f"    !! giving up on sensor {sensor_id} "
                      f"chunk {chunk_from}..{chunk_to} page {page}: {e}")
                break
            results = resp.json().get("results", [])
            if not results:
                break
            all_rows.extend(results)
            print(f"       page {page}: {len(results)} raw readings "
                  f"(running total: {len(all_rows)})")
            if len(results) < 1000:
                break
            page += 1
            time.sleep(0.3)

    return all_rows


def get_measurements(sensor_id, date_from, date_to):
    """Fetch sensor data for [date_from, date_to].

    Strategy:
      1. Try the server-side hourly aggregation endpoint (/v3/sensors/{id}/hours).
      2. If it returns 0 rows or errors (often not precomputed by OpenAQ for some stations),
         automatically fall back to the raw /measurements endpoint with client-side hourly aggregation.
    """
    # ---- attempt 1: server-side hourly endpoint ----
    try:
        probe_url = f"{OPENAQ_BASE}/sensors/{sensor_id}/hours"
        probe = requests.get(probe_url, headers=HEADERS, params={"limit": 1}, timeout=(10, 15))
        if probe.status_code == 404:
            raise requests.exceptions.HTTPError("404 — hourly endpoint not available", response=probe)
        probe.raise_for_status()
        probe_data = probe.json().get("results", [])
        if probe_data:
            print("    [✓] Server-side hourly endpoint found — testing fetch...")
            rows = _fetch_hourly_endpoint(sensor_id, date_from, date_to)
            if rows:
                print(f"    [✓] Successfully fetched {len(rows)} precomputed hourly rows.")
                return rows, True
            print("    [!] Hourly endpoint returned 0 rows for this date range.")
        else:
            print("    [!] Hourly endpoint has no records for this sensor.")
    except (requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as probe_err:
        print(f"    [!] Hourly endpoint unavailable ({probe_err})")

    # ---- attempt 2: raw measurements (reliable fallback) ----
    print("    [→] Using raw /measurements endpoint + client-side hourly aggregation...")
    rows = _fetch_raw_measurements(sensor_id, date_from, date_to)
    return rows, False


def _extract_timestamp(record):
    """Extract UTC timestamp string from either /hours or /measurements format."""
    # /v3/sensors/{id}/hours schema: record['datetime']['utc']
    if "datetime" in record and isinstance(record["datetime"], dict):
        return record["datetime"].get("utc")
    # /v3/sensors/{id}/measurements schema: record['period']['datetimeFrom']['utc']
    if "period" in record and isinstance(record["period"], dict):
        dt_from = record["period"].get("datetimeFrom", {})
        if isinstance(dt_from, dict):
            return dt_from.get("utc")
        return dt_from
    return record.get("datetime")


def fetch_openaq_data():
    print(f"Finding OpenAQ stations near {CITY_NAME}...")
    locations = get_nearby_locations(LAT, LON, RADIUS_M)
    if not locations:
        raise RuntimeError("No OpenAQ stations found nearby — widen RADIUS_M.")

    if STATION_NAME:
        matched = [loc for loc in locations
                   if STATION_NAME.lower() in loc["name"].lower()]
        if not matched:
            available = ", ".join(loc["name"] for loc in locations)
            raise RuntimeError(
                f"No station matching '{STATION_NAME}' found. "
                f"Available stations: {available}"
            )
        
        # Check which station is actually active and has recent data
        print(f"  Found {len(matched)} matching stations for '{STATION_NAME}'. Checking data recency...")
        best_loc = None
        best_last_date = ""
        for loc in matched:
            dt_last = loc.get("datetimeLast")
            last_str = dt_last.get("utc") if isinstance(dt_last, dict) else str(dt_last or "")
            print(f"    - '{loc['name']}' (ID: {loc['id']}): data available up to {last_str or 'unknown'}")
            if last_str > best_last_date:
                best_last_date = last_str
                best_loc = loc

        locations = [best_loc or matched[0]]
        print(f"  Selected most active station: '{locations[0]['name']}' (ID: {locations[0]['id']})")

    records = []  # list of dicts: {timestamp, parameter, value}
    already_hourly_flag = None

    for loc in locations:
        loc_id = loc["id"]
        sensors = get_sensors_for_location(loc_id)
        for sensor in sensors:
            param = sensor["parameter"]["name"].lower()
            if param not in POLLUTANTS_OF_INTEREST:
                continue

            # Display sensor availability metadata
            dt_first = sensor.get("datetimeFirst")
            dt_last = sensor.get("datetimeLast")
            f_str = dt_first.get("utc") if isinstance(dt_first, dict) else str(dt_first)
            l_str = dt_last.get("utc") if isinstance(dt_last, dict) else str(dt_last)
            print(f"  Sensor {sensor['id']} ({param}) | Available range: {f_str} to {l_str}")

            # Skip sensors that don't cover the requested dates
            if f_str and l_str and (START_DATE > l_str[:10] or END_DATE < f_str[:10]):
                print(f"    [i] Skipping sensor {sensor['id']} — dates {START_DATE}..{END_DATE} "
                      f"fall outside its recorded history ({f_str[:10]}..{l_str[:10]}).")
                continue

            measurements, already_hourly = get_measurements(
                sensor["id"], START_DATE, END_DATE
            )
            if already_hourly_flag is None:
                already_hourly_flag = already_hourly
            for m in measurements:
                ts = _extract_timestamp(m)
                if ts and m.get("value") is not None:
                    records.append(
                        {
                            "timestamp": ts,
                            "parameter": param,
                            "value":     m["value"],
                        }
                    )

    if not records:
        raise RuntimeError("No measurements returned — check date range / API key.")

    df = pd.DataFrame(records)
    # Normalize timestamp to UTC and remove tz-awareness so it joins cleanly with weather
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    if not already_hourly_flag:
        # Raw endpoint — floor to the hour and average within each bucket
        df["timestamp"] = df["timestamp"].dt.floor("h")

    # Average across stations/sensors in the same bucket (handles both paths)
    df = df.groupby(["timestamp", "parameter"])["value"].mean().reset_index()
    df_wide = df.pivot(index="timestamp", columns="parameter", values="value")
    return df_wide


# ---------------------------------------------------------------------------
# STEP 4 — fetch matching weather data from Open-Meteo (no API key needed)
# ---------------------------------------------------------------------------
def fetch_weather_data():
    print("Fetching weather data from Open-Meteo...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation",
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(hourly)
    # Ensure weather timestamps are tz-naive UTC
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.rename(
        columns={
            "time": "timestamp",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
        }
    )
    df = df.set_index("timestamp")
    return df


# ---------------------------------------------------------------------------
# MAIN — fetch both, join, save
# ---------------------------------------------------------------------------
def main():
    aqi_df = fetch_openaq_data()
    weather_df = fetch_weather_data()

    # Ensure index timezone compatibility
    if aqi_df.index.tz is not None:
        aqi_df.index = aqi_df.index.tz_localize(None)
    if weather_df.index.tz is not None:
        weather_df.index = weather_df.index.tz_localize(None)

    combined = aqi_df.join(weather_df, how="inner")
    combined = combined.sort_index()
    combined.to_csv(OUTPUT_CSV)

    print(f"\nDone. Saved {len(combined)} hourly rows to {OUTPUT_CSV}")
    print(combined.head())


if __name__ == "__main__":
    main()