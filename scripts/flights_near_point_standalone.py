#!/usr/bin/env python3
import json
import math
import os
import sys

import psycopg2

DSN = os.environ.get("POSTGRES_DSN", "postgresql://postgres:Warqi4-sywsow-zozfyc@tracer-db.cb80eku2emy0.eu-north-1.rds.amazonaws.com:5432/tracer")
R = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    a = math.radians(lat1)
    b = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))
    return R * c

def bearing_deg(lat1, lon1, lat2, lon2):
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def angle_diff(a, b):
    d = (a - b + 180) % 360 - 180
    return abs(d)

def get_conn():
    return psycopg2.connect(DSN)

def fetch_nearby(conn, lat, lng, radius, origin=None, dest=None, alt=None, alt_band_ft=1000.0):
    km_per_deg = 111.32
    lat_m = radius * 1.2 / km_per_deg
    lon_m = lat_m / max(0.01, math.cos(math.radians(lat)))
    dist_sql = "research.earth_distance(research.ll_to_earth(n.lat, n.lon), research.ll_to_earth(%s, %s)) / 1000.0"
    od = " JOIN research.flight_metadata fm ON n.flight_id = fm.flight_id AND fm.origin_airport = %s AND fm.destination_airport = %s" if (origin and dest) else ""
    alt_f = " AND n.alt BETWEEN %s AND %s" if alt is not None else ""
    sql = """
    WITH nearby AS (
      SELECT n.flight_id, n.timestamp, n.lat, n.lon, n.alt, (""" + dist_sql + """) AS dist_km
      FROM research.normal_tracks n """ + od + """
      WHERE n.lat BETWEEN %s AND %s AND n.lon BETWEEN %s AND %s
      """ + alt_f + """
    )
    SELECT DISTINCT ON (flight_id) flight_id, timestamp, lat, lon, alt, dist_km
    FROM nearby WHERE dist_km <= %s ORDER BY flight_id, dist_km
    """
    params = [lat, lng]
    if origin and dest:
        params.extend([origin, dest])
    params.extend([lat - lat_m, lat + lat_m, lng - lon_m, lng + lon_m])
    if alt is not None:
        params.extend([alt - alt_band_ft, alt + alt_band_ft])
    params.append(radius)
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()

SEC_FOR_6KM, KM_FOR_WINDOW_BASE = 30, 5

def window_sec(chunk_half_km):
    return SEC_FOR_6KM * (2 * chunk_half_km) / KM_FOR_WINDOW_BASE

def fetch_tracks(conn, flight_center_ts, window_sec, alt_lo=None, alt_hi=None, batch=25):
    if not flight_center_ts:
        return {}
    out = {}
    alt_f = " AND t.alt BETWEEN %s AND %s" if (alt_lo is not None and alt_hi is not None) else ""
    sql = """
    WITH ranges(flight_id, lo_ts, hi_ts) AS (
      SELECT u.fid, u.epoch_sec - %s, u.epoch_sec + %s
      FROM unnest(%s::text[], %s::double precision[]) AS u(fid, epoch_sec)
    )
    SELECT t.flight_id, t.timestamp, t.lat, t.lon, t.alt
    FROM research.normal_tracks t
    JOIN ranges r ON t.flight_id = r.flight_id AND t.timestamp >= r.lo_ts AND t.timestamp <= r.hi_ts
    """ + alt_f + """ ORDER BY t.flight_id, t.timestamp
    """
    cur = conn.cursor()
    for i in range(0, len(flight_center_ts), batch):
        batch_list = flight_center_ts[i : i + batch]
        fids = [p[0] for p in batch_list]
        ts = [float(p[1]) if isinstance(p[1], (int, float)) else p[1].timestamp() for p in batch_list]
        params = [window_sec, window_sec, fids, ts]
        if alt_lo is not None and alt_hi is not None:
            params += [alt_lo, alt_hi]
        cur.execute(sql, params)
        for row in cur.fetchall():
            fid, t, la, lo, al = row[0], row[1], row[2], row[3], row[4]
            out.setdefault(fid, []).append((t, la, lo, al))
    return out

def fetch_feedback_track(conn, flight_id):
    """Load track for one flight from feedback.flight_tracks. Returns list of (timestamp, lat, lon, alt)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, lat, lon, alt FROM feedback.flight_tracks WHERE flight_id = %s ORDER BY timestamp ASC",
        (flight_id,),
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        ts = float(r[0]) if isinstance(r[0], (int, float)) else r[0].timestamp()
        alt_val = float(r[3]) if r[3] is not None else 0.0
        out.append((ts, float(r[1]), float(r[2]), alt_val))
    return out

def chunk_around_closest(points, qlat, qlon, half_km):
    if not points:
        return []
    cum = [0.0]
    for i in range(1, len(points)):
        _, la, lo, _ = points[i - 1]
        _, la2, lo2, _ = points[i]
        cum.append(cum[-1] + haversine_km(la, lo, la2, lo2))
    best_i = min(range(len(points)), key=lambda i: haversine_km(qlat, qlon, points[i][1], points[i][2]))
    center = cum[best_i]
    lo_c, hi_c = center - half_km, center + half_km
    return [p for i, p in enumerate(points) if lo_c <= cum[i] <= hi_c]

def chunk_to_summary(fid, chunk):
    if not chunk or len(chunk) < 2:
        pts = [{"lat": p[1], "lng": p[2], "alt": p[3], "timestamp": p[0], "heading": 0.0} for p in chunk]
        return {"id": fid, "points": pts, "total_heading_changes": 0, "total_speed_changes": 0,
                "total_alt_changes": 0, "total_vspeed_changes": 0, "total_points": len(chunk), "avg_speed": 0.0, "avg_vspeed": 0.0}
    n = len(chunk)
    headings = [bearing_deg(chunk[i][1], chunk[i][2], chunk[i + 1][1], chunk[i + 1][2]) for i in range(n - 1)]
    headings.append(headings[-1])
    speeds = []
    vspeeds = []
    for i in range(n - 1):
        delta = chunk[i + 1][0] - chunk[i][0]
        dt = getattr(delta, "total_seconds", lambda: float(delta))() or 1e-6
        dist = haversine_km(chunk[i][1], chunk[i][2], chunk[i + 1][1], chunk[i + 1][2])
        speeds.append(dist * 3600 / dt)
        vspeeds.append((chunk[i + 1][3] - chunk[i][3]) * 60 / dt)
    points = [{"lat": chunk[i][1], "lng": chunk[i][2], "alt": chunk[i][3], "timestamp": chunk[i][0], "heading": headings[i]} for i in range(n)]
    eps = 1e-9
    count_h = sum(1 for i in range(n - 1) if angle_diff(headings[i + 1], headings[i]) > eps)
    count_s = sum(1 for i in range(len(speeds) - 1) if abs(speeds[i + 1] - speeds[i]) > eps)
    count_alt = sum(1 for i in range(n - 1) if abs(chunk[i + 1][3] - chunk[i][3]) > eps)
    count_v = sum(1 for i in range(len(vspeeds) - 1) if abs(vspeeds[i + 1] - vspeeds[i]) > eps)
    return {
        "id": fid,
        "points": points,
        "total_heading_changes": count_h,
        "total_speed_changes": count_s,
        "total_alt_changes": count_alt,
        "total_vspeed_changes": count_v,
        "total_points": n,
        "avg_speed": round(sum(speeds) / len(speeds), 2),
        "avg_vspeed": round(sum(vspeeds) / len(vspeeds), 2),
    }

def run_feedback_flight(lat, lng, flight_id, chunk_half_km=2.0):
    """Get chunk around closest point to (lat, lng) for the given flight from feedback.flight_tracks and return feature summary."""
    conn = get_conn()
    try:
        points = fetch_feedback_track(conn, flight_id)
        if not points:
            return {"id": flight_id, "points": [], "total_heading_changes": 0, "total_speed_changes": 0,
                    "total_alt_changes": 0, "total_vspeed_changes": 0, "total_points": 0, "avg_speed": 0.0, "avg_vspeed": 0.0}
        chunk = chunk_around_closest(points, lat, lng, chunk_half_km)
        return chunk_to_summary(flight_id, chunk)
    finally:
        conn.close()

def run(lat, lng, radius=5.0, chunk_half_km=2.0, origin=None, dest=None, alt=None, alt_band_ft=1000.0):
    conn = get_conn()
    try:
        rows = fetch_nearby(conn, lat, lng, radius, origin, dest, alt, alt_band_ft)
        if not rows:
            return []
        flight_center_ts = [(r[0], r[1]) for r in rows]
        ws = window_sec(chunk_half_km)
        alt_lo, alt_hi = (alt - alt_band_ft, alt + alt_band_ft) if alt is not None else (None, None)
        tracks = fetch_tracks(conn, flight_center_ts, ws, alt_lo, alt_hi)
        summaries = []
        for r in rows:
            fid, ts, la, lo, alt, dist_km = r[0], r[1], r[2], r[3], r[4], r[5]
            chunk = chunk_around_closest(tracks.get(fid, []), lat, lng, chunk_half_km)
            summaries.append(chunk_to_summary(fid, chunk))
        return summaries
    finally:
        conn.close()

if __name__ == "__main__":
    print(json.dumps(run_feedback_flight(29.0406, 35.505, "3af2006c"), default=str))
    lat, lng = (29.0406, 35.505)
    radius = 10.0
    chunk_half = 8
    origin = "OMDW" # also can be None if not enough data in that spesific pair
    dest = "OLBA"
    alt = 31175.0
    alt_band = 10000.0
    result = run(lat, lng, radius, chunk_half, origin, dest, alt, alt_band)
    print(json.dumps(result, default=str))
