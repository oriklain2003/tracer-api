#!/usr/bin/env python3
"""
Flights that had any point within N km of (lat,lng). For each such flight:
- Closest point (with timestamp) to the query point.
- Track chunk: points within chunk_half_km back and forward along the path from that closest point.
Chunk is fetched by querying normal_tracks in the time range that covers that distance.
"""
import sys
import time
from pathlib import Path
# Add parent directory to path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import numpy as np
import pandas as pd


from service.pg_provider import get_pool
from core.geodesy import initial_bearing_deg

EARTH_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Single-point haversine distance in km."""
    import math
    lat1, lon1, lat2, lon2 = math.radians(lat1), math.radians(lon1), math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_KM * c


def _angle_diff_deg(a: float, b: float) -> float:
    """Absolute difference in degrees (0-180), handling wrap."""
    d = (a - b + 180) % 360 - 180
    return abs(d)

def haversine_km_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_KM * c


def _query_to_df(conn, sql, params, columns):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def flights_near_point(
    lat: float, lng: float, radius: float = 5.0,
    origin_airport: str | None = None, destination_airport: str | None = None,
    alt: float | None = None, alt_band_ft: float = 1000.0,
):
    """All flights with at least one point within radius km; closest point (and its timestamp) per flight.
    If origin/destination set: only that pair. If alt set: only points in [alt - alt_band_ft, alt + alt_band_ft] ft."""
    km_per_deg = 111.32
    lat_margin_deg = radius * 1.2 / km_per_deg
    lon_margin_deg = lat_margin_deg / max(0.01, np.cos(np.radians(lat)))
    dist_sql = "research.earth_distance(research.ll_to_earth(n.lat, n.lon), research.ll_to_earth(%s, %s)) / 1000.0"
    od_filter = (
        "JOIN research.flight_metadata fm ON n.flight_id = fm.flight_id "
        "AND fm.origin_airport = %s AND fm.destination_airport = %s"
    )
    alt_filter = "AND n.alt BETWEEN %s AND %s" if alt is not None else ""
    sql = """
    WITH nearby AS (
        SELECT n.flight_id, n.timestamp, n.lat, n.lon, n.alt, (""" + dist_sql + """) AS dist_km
        FROM research.normal_tracks n
        """ + (od_filter if (origin_airport and destination_airport) else "") + """
        WHERE n.lat BETWEEN %s AND %s AND n.lon BETWEEN %s AND %s
        """ + alt_filter + """
    )
    SELECT DISTINCT ON (flight_id) flight_id, timestamp, lat, lon, alt, dist_km
    FROM nearby WHERE dist_km <= %s ORDER BY flight_id, dist_km
    """
    params = [lat, lng]
    if origin_airport and destination_airport:
        params.extend([origin_airport, destination_airport])
    params.extend([lat - lat_margin_deg, lat + lat_margin_deg, lng - lon_margin_deg, lng + lon_margin_deg])
    if alt is not None:
        params.extend([alt - alt_band_ft, alt + alt_band_ft])
    params.append(radius)
    cols = ["flight_id", "timestamp", "lat", "lon", "alt", "dist_km"]
    t0 = time.perf_counter()
    with get_pool().get_connection() as conn:
        df = _query_to_df(conn, sql, params, cols)
    return df, time.perf_counter() - t0


# Time window (sec) to cover 2*chunk_half_km at typical speed (~6 km in 30 sec)
SEC_FOR_6KM, KM_FOR_WINDOW_BASE = 30, 5

def window_sec_for_chunk(chunk_half_km: float) -> float:
    return SEC_FOR_6KM * (2 * chunk_half_km) / KM_FOR_WINDOW_BASE


TRACK_FETCH_BATCH_SIZE = 25  # flights per query to avoid statement timeout


def get_tracks_in_time_ranges(
    flight_center_ts: list, window_sec: float,
    alt_lo: float | None = None, alt_hi: float | None = None,
    batch_size: int = TRACK_FETCH_BATCH_SIZE,
) -> dict:
    """For each (flight_id, center_timestamp), fetch track points in [center_ts - window_sec, center_ts + window_sec].
    If alt_lo/alt_hi set, only points with t.alt BETWEEN alt_lo AND alt_hi (ft). Runs in batches to avoid timeout."""
    if not flight_center_ts:
        return {}
    alt_filter = " AND t.alt BETWEEN %s AND %s" if (alt_lo is not None and alt_hi is not None) else ""
    sql = """
    WITH ranges(flight_id, lo_ts, hi_ts) AS (
        SELECT u.fid, u.epoch_sec - %s, u.epoch_sec + %s
        FROM unnest(%s::text[], %s::double precision[]) AS u(fid, epoch_sec)
    )
    SELECT t.flight_id, t.timestamp, t.lat, t.lon, t.alt
    FROM research.normal_tracks t
    JOIN ranges r ON t.flight_id = r.flight_id AND t.timestamp >= r.lo_ts AND t.timestamp <= r.hi_ts
    """ + alt_filter + """
    ORDER BY t.flight_id, t.timestamp
    """
    cols = ["flight_id", "timestamp", "lat", "lon", "alt"]
    out = {}
    with get_pool().get_connection() as conn:
        for i in range(0, len(flight_center_ts), batch_size):
            batch = flight_center_ts[i : i + batch_size]
            flight_ids = [p[0] for p in batch]
            center_ts = [float(p[1]) if isinstance(p[1], (int, float)) else p[1].timestamp() for p in batch]
            params = [window_sec, window_sec, flight_ids, center_ts]
            if alt_lo is not None and alt_hi is not None:
                params.extend([alt_lo, alt_hi])
            df = _query_to_df(conn, sql, params, cols)
            if not df.empty:
                for fid, g in df.groupby("flight_id", sort=False):
                    out[fid] = list(zip(g.timestamp, g.lat, g.lon, g.alt))
    return out


def chunk_around_closest(track_points: list, query_lat: float, query_lng: float, half_km: float = 2.0) -> list:
    """Return points in track whose along-path distance from closest point is in [-half_km, +half_km] km."""
    if not track_points:
        return []
    pts = np.array(track_points, dtype=object)
    lat, lon = np.array(pts[:, 1], dtype=float), np.array(pts[:, 2], dtype=float)
    # segment lengths (km), then cumsum
    seg = haversine_km_vec(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cum = np.zeros(len(pts))
    cum[1:] = np.cumsum(seg)
    # closest point index
    best_i = np.argmin(haversine_km_vec(np.full_like(lat, query_lat), np.full_like(lon, query_lng), lat, lon))
    lo, hi = cum[best_i] - half_km, cum[best_i] + half_km
    mask = (cum >= lo) & (cum <= hi)
    return [tuple(p) for p in pts[mask]]


def chunk_to_summary(flight_id: str, chunk: list) -> dict:
    """Build { id, points (lat,lng,alt,timestamp,heading), total_heading_changes, total_speed_changes, ... }."""
    if not chunk or len(chunk) < 2:
        pts = [{"lat": p[1], "lng": p[2], "alt": p[3], "timestamp": p[0], "heading": 0.0} for p in chunk]
        return {
            "id": flight_id,
            "points": pts,
            "total_heading_changes": 0.0,
            "total_speed_changes": 0.0,
            "total_alt_changes": 0.0,
            "total_vspeed_changes": 0.0,
            "total_points": len(chunk),
            "avg_speed": 0.0,
            "avg_vspeed": 0.0,
        }
    n = len(chunk)
    # headings: bearing from i to i+1; last point = same as prev
    headings = []
    for i in range(n - 1):
        ts0, lat0, lon0, alt0 = chunk[i]
        ts1, lat1, lon1, alt1 = chunk[i + 1]
        headings.append(initial_bearing_deg(lat0, lon0, lat1, lon1))
    headings.append(headings[-1] if headings else 0.0)
    # segment speeds (km/h) and vspeeds (ft/min)
    speeds_kmh = []
    vspeeds_ftmin = []
    for i in range(n - 1):
        ts0, lat0, lon0, alt0 = chunk[i]
        ts1, lat1, lon1, alt1 = chunk[i + 1]
        dt_sec = float(ts1 - ts0) or 1e-6
        dist_km = _haversine_km(lat0, lon0, lat1, lon1)
        speeds_kmh.append(dist_km * 3600 / dt_sec)
        vspeeds_ftmin.append((alt1 - alt0) * 60 / dt_sec)
    # points with heading
    points = [
        {"lat": chunk[i][1], "lng": chunk[i][2], "alt": chunk[i][3], "timestamp": chunk[i][0], "heading": headings[i]}
        for i in range(n)
    ]
    total_heading = sum(_angle_diff_deg(headings[i + 1], headings[i]) for i in range(n - 1))
    total_speed_ch = sum(abs(speeds_kmh[i + 1] - speeds_kmh[i]) for i in range(len(speeds_kmh) - 1))
    total_alt_ch = sum(abs(chunk[i + 1][3] - chunk[i][3]) for i in range(n - 1))
    total_vspeed_ch = sum(abs(vspeeds_ftmin[i + 1] - vspeeds_ftmin[i]) for i in range(len(vspeeds_ftmin) - 1))
    return {
        "id": flight_id,
        "points": points,
        "total_heading_changes": round(total_heading, 2),
        "total_speed_changes": round(total_speed_ch, 2),
        "total_alt_changes": round(total_alt_ch, 2),
        "total_vspeed_changes": round(total_vspeed_ch, 2),
        "total_points": n,
        "avg_speed": round(sum(speeds_kmh) / len(speeds_kmh), 2),
        "avg_vspeed": round(sum(vspeeds_ftmin) / len(vspeeds_ftmin), 2),
    }


def flights_near_point_with_chunks(
    lat: float, lng: float, radius: float = 5.0, chunk_half_km: float = 2.0,
    origin_airport: str | None = None, destination_airport: str | None = None,
    alt: float | None = None, alt_band_ft: float = 1000.0,
) -> list:
    
    df, _ = flights_near_point(lat, lng, radius, origin_airport, destination_airport, alt, alt_band_ft)
    if df.empty:
        return []
    flight_center_ts = list(zip(df.flight_id, df.timestamp))
    alt_lo, alt_hi = (alt - alt_band_ft, alt + alt_band_ft) if alt is not None else (None, None)
    tracks = get_tracks_in_time_ranges(
        flight_center_ts, window_sec_for_chunk(chunk_half_km), alt_lo=alt_lo, alt_hi=alt_hi
    )
    return [
        {
            "flight_id": row.flight_id,
            "origin_airport": origin_airport,
            "destination_airport": destination_airport,
            "closest": {"timestamp": row.timestamp, "lat": row.lat, "lon": row.lon, "alt": row.alt, "dist_km": row.dist_km},
            "chunk": chunk_around_closest(tracks.get(row.flight_id, []), lat, lng, chunk_half_km),
        }
        for row in df.itertuples(index=False)
    ]

import json
if __name__ == "__main__":
    lat, lng = (29.0406, 35.505)
    radius = 10.0
    chunk_half = 8
    origin = "OMDW"
    dest = "OLBA"
    alt = 31175.0
    alt_band_ft = 10000.0
    t0 = time.perf_counter()
    results = flights_near_point_with_chunks(
        lat, lng, radius, chunk_half_km=chunk_half,
        origin_airport=origin, destination_airport=dest,
        alt=alt, alt_band_ft=alt_band_ft,
    )
    print(json.dumps(results, default=str))
    elapsed = time.perf_counter() - t0
    summaries = [chunk_to_summary(r["flight_id"], r["chunk"]) for r in results]
    print(f"Flights: {len(results)}, elapsed: {elapsed:.3f}s")
    print(json.dumps(summaries, default=str))
