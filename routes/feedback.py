"""
Feedback routes - user feedback, tagging, history, replay.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable

from fastapi import APIRouter, Depends, HTTPException

from routes.users import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Feedback"], dependencies=[Depends(get_current_user)])
public_router = APIRouter(tags=["Feedback"])

# These will be set by the main api.py module
CACHE_DB_PATH: Path = None
DB_ANOMALIES_PATH: Path = None
DB_TRACKS_PATH: Path = None
DB_RESEARCH_PATH: Path = None
PRESENT_DB_PATH: Path = None
FEEDBACK_DB_PATH: Path = None
FEEDBACK_TAGGED_DB_PATH: Path = None
PROJECT_ROOT: Path = None

# Function references from api.py
_get_pipeline = None
_get_unified_track = None
_fetch_flight_details = None
_update_flight_record = None
_save_feedback = None
_auth_middleware = None


def configure(
    cache_db_path: Path,
    db_anomalies_path: Path,
    db_tracks_path: Path,
    db_research_path: Path,
    present_db_path: Path,
    feedback_db_path: Path,
    feedback_tagged_db_path: Path,
    project_root: Path,
    get_pipeline_func,
    get_unified_track_func,
    fetch_flight_details_func,
    update_flight_record_func,
    save_feedback_func,
    auth_middleware=None,
):
    """Configure the router with paths and dependencies from api.py"""
    global CACHE_DB_PATH, DB_ANOMALIES_PATH, DB_TRACKS_PATH, DB_RESEARCH_PATH
    global PRESENT_DB_PATH, FEEDBACK_DB_PATH, FEEDBACK_TAGGED_DB_PATH, PROJECT_ROOT
    global _get_pipeline, _get_unified_track, _fetch_flight_details, _update_flight_record, _save_feedback
    global _auth_middleware
    
    CACHE_DB_PATH = cache_db_path
    DB_ANOMALIES_PATH = db_anomalies_path
    DB_TRACKS_PATH = db_tracks_path
    DB_RESEARCH_PATH = db_research_path
    PRESENT_DB_PATH = present_db_path
    FEEDBACK_DB_PATH = feedback_db_path
    FEEDBACK_TAGGED_DB_PATH = feedback_tagged_db_path
    PROJECT_ROOT = project_root
    _get_pipeline = get_pipeline_func
    _get_unified_track = get_unified_track_func
    _fetch_flight_details = fetch_flight_details_func
    _update_flight_record = update_flight_record_func
    _save_feedback = save_feedback_func
    _auth_middleware = auth_middleware
    
    # Load airports data when configured
    _load_airports_data()


def _get_feedback_permission_dependency():
    """Get permission dependency if auth is configured, otherwise return None"""
    if _auth_middleware is not None:
        return Depends(_auth_middleware.require_permission("feedback.create"))
    return None


# ============================================================================
# AIRPORTS DATA AND ORIGIN/DESTINATION CALCULATION
# ============================================================================

_AIRPORTS_DATA: List[Dict[str, Any]] = []


def _load_airports_data():
    """Load airports from docs/airports.csv for origin/destination calculation."""
    global _AIRPORTS_DATA
    if _AIRPORTS_DATA:
        return  # Already loaded
    
    import csv
    
    airports_path = Path(__file__).parent.parent.parent / "docs" / "airports.csv"
    if not airports_path.exists():
        logger.warning(f"Airports data not found at {airports_path}")
        return
    
    try:
        with open(airports_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip airports without valid coordinates
                try:
                    lat = float(row.get('latitude_deg', ''))
                    lon = float(row.get('longitude_deg', ''))
                except (ValueError, TypeError):
                    continue
                
                # Skip closed airports for origin/destination calculation
                airport_type = row.get('type', '')
                if airport_type == 'closed':
                    continue
                
                _AIRPORTS_DATA.append({
                    "ident": row.get("ident"),
                    "icao_code": row.get("icao_code") or None,
                    "iata_code": row.get("iata_code") or None,
                    "name": row.get("name"),
                    "type": airport_type,
                    "lat": lat,
                    "lon": lon,
                    "country": row.get("iso_country"),
                    "municipality": row.get("municipality"),
                })
        
        logger.info(f"Loaded {len(_AIRPORTS_DATA)} airports for origin/destination calculation")
    except Exception as e:
        logger.error(f"Failed to load airports data: {e}")


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in nautical miles between two coordinates."""
    R = 3440.065  # Earth radius in nautical miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def _find_nearest_airport(lat: float, lon: float, max_distance_nm: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Find the nearest airport within max_distance_nm nautical miles.
    Returns the airport dict or None if no airport is within range.
    """
    if not _AIRPORTS_DATA:
        _load_airports_data()
    
    nearest = None
    min_distance = float('inf')
    
    for airport in _AIRPORTS_DATA:
        distance = _haversine_nm(lat, lon, airport["lat"], airport["lon"])
        if distance < min_distance and distance <= max_distance_nm:
            min_distance = distance
            nearest = airport
    
    return nearest


def _get_airport_code(airport: Optional[Dict[str, Any]]) -> Optional[str]:
    """Get the best available airport code (ICAO, IATA, or ident)."""
    if not airport:
        return None
    return airport.get("icao_code") or airport.get("iata_code") or airport.get("ident")


def _calculate_origin_destination_from_track(flight_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Calculate origin and destination airports from flight track points.
    
    - Origin: First point of the track, find nearest airport within 5 NM
    - Destination: Last point if last 5 points are at 0 altitude (indicating landing)
    
    Returns (origin_code, destination_code)
    """
    if not FEEDBACK_TAGGED_DB_PATH or not FEEDBACK_TAGGED_DB_PATH.exists():
        return None, None
    
    try:
        conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get track points for the flight
        cursor.execute("""
            SELECT lat, lon, alt
            FROM flight_tracks
            WHERE flight_id = ?
            ORDER BY timestamp ASC
        """, (flight_id,))
        
        points = cursor.fetchall()
        conn.close()
        
        if not points or len(points) < 2:
            return None, None
        
        origin_code = None
        destination_code = None
        
        # Calculate origin from first point (takeoff)
        first_point = points[0]
        if first_point["lat"] is not None and first_point["lon"] is not None:
            origin_airport = _find_nearest_airport(first_point["lat"], first_point["lon"], max_distance_nm=5.0)
            origin_code = _get_airport_code(origin_airport)
        
        # Calculate destination from last point if it looks like a landing
        # Check if last 5 points are at 0 altitude (or very low, <100ft for tolerance)
        last_points = points[-5:] if len(points) >= 5 else points
        
        # Check if this looks like a landing (all last points at ground level)
        is_landing = True
        for pt in last_points:
            alt = pt["alt"]
            if alt is None or alt > 100:  # Allow some tolerance (100 ft)
                is_landing = False
                break
        
        if is_landing:
            last_point = points[-1]
            if last_point["lat"] is not None and last_point["lon"] is not None:
                dest_airport = _find_nearest_airport(last_point["lat"], last_point["lon"], max_distance_nm=5.0)
                destination_code = _get_airport_code(dest_airport)
        
        return origin_code, destination_code
        
    except Exception as e:
        logger.warning(f"Error calculating origin/destination for {flight_id}: {e}")
        return None, None


def _safe_join(values: Iterable[Any]) -> str:
    return ", ".join(str(v) for v in values if v is not None and str(v) != "")


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def flatten_rules(full_report: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract rule-related columns plus a per-rule row list.
    Returns (aggregate_columns, rule_rows).
    """
    if not full_report:
        return {}, []

    rules_layer = full_report.get("layer_1_rules") or {}
    rule_report = rules_layer.get("report") or {}
    matched_rules: List[Dict[str, Any]] = rule_report.get("matched_rules") or []
    evaluations: List[Dict[str, Any]] = rule_report.get("evaluations") or []

    aggregate = {
        "rules_status": rules_layer.get("status"),
        "rules_triggers": _safe_join(rules_layer.get("triggers") or []),
        "matched_rule_ids": _safe_join([r.get("id") for r in matched_rules]),
        "matched_rule_names": _safe_join([r.get("name") for r in matched_rules]),
        "matched_rule_categories": _safe_join([r.get("category") for r in matched_rules]),
    }

    rule_rows: List[Dict[str, Any]] = []
    for r in evaluations:
        rule_rows.append(
            {
                "rule_id": r.get("id"),
                "rule_name": r.get("name"),
                "category": r.get("category"),
                "severity": r.get("severity"),
                "matched": 1 if r.get("matched") else 0,
                "summary": _coerce_text(r.get("summary")),
                "details": _coerce_text(r.get("details")),
            }
        )

    return aggregate, rule_rows


@router.get("/api/feedback/track/{flight_id}")
def get_feedback_track(flight_id: str):
    """
    Return track points. Search order:
    1. feedback_tagged.db (flight_tracks, anomalies_tracks, normal_tracks)
    2. present_anomalies.db (flight_tracks)
    3. research_new.db (anomalies_tracks, normal_tracks)
    """
    points = []
    
    # 1. Try feedback_tagged.db first (most common for history/replay)
    if FEEDBACK_TAGGED_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try flight_tracks table first
            cursor.execute(
                """
                SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source
                FROM flight_tracks
                WHERE flight_id = ?
                ORDER BY timestamp ASC
                """,
                (flight_id,),
            )
            rows = cursor.fetchall()
            
            # Try anomalies_tracks if not found
            if not rows:
                try:
                    cursor.execute(
                        """
                        SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign
                        FROM anomalies_tracks
                        WHERE flight_id = ?
                        ORDER BY timestamp ASC
                        """,
                        (flight_id,),
                    )
                    rows = cursor.fetchall()
                except Exception:
                    pass  # Table might not exist
            
            # Try normal_tracks if still not found
            if not rows:
                try:
                    cursor.execute(
                        """
                        SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign
                        FROM normal_tracks
                        WHERE flight_id = ?
                        ORDER BY timestamp ASC
                        """,
                        (flight_id,),
                    )
                    rows = cursor.fetchall()
                except Exception:
                    pass  # Table might not exist
            
            conn.close()
            
            if rows:
                for r in rows:
                    points.append({
                        "flight_id": r["flight_id"],
                        "timestamp": r["timestamp"],
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "alt": r["alt"],
                        "heading": r["track"],  # Use track as heading
                        "gspeed": r["gspeed"],
                        "vspeed": r["vspeed"],
                        "track": r["track"],
                        "squawk": r["squawk"],
                        "callsign": r["callsign"],
                        "source": r["source"] if "source" in r.keys() else "feedback_tagged",
                    })
                return {"flight_id": flight_id, "points": points}
        except Exception as e:
            logger.error(f"Error fetching from feedback_tagged.db: {e}")
    
    # 2. Try present_anomalies.db
    if PRESENT_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(PRESENT_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT flight_id, timestamp, lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, source
                FROM flight_tracks
                WHERE flight_id = ?
                ORDER BY timestamp ASC
                """,
                (flight_id,),
            )
            rows = cursor.fetchall()
            conn.close()

            if rows:
                for r in rows:
                    points.append({
                        "flight_id": r["flight_id"],
                        "timestamp": r["timestamp"],
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "alt": r["alt"],
                        "heading": r["heading"],
                        "gspeed": r["gspeed"],
                        "vspeed": r["vspeed"],
                        "track": r["track"],
                        "squawk": r["squawk"],
                        "callsign": r["callsign"],
                        "source": r["source"],
                    })
                return {"flight_id": flight_id, "points": points}
        except Exception as e:
            logger.error(f"Error fetching from present_anomalies.db: {e}")
    
    # Fall back to research_new.db
    if not points and DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try anomalies_tracks (no heading column, use track instead)
            cursor.execute(
                """
                SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign
                FROM anomalies_tracks
                WHERE flight_id = ?
                ORDER BY timestamp ASC
                """,
                (flight_id,),
            )
            rows = cursor.fetchall()
            
            # Try normal_tracks if not found
            if not rows:
                cursor.execute(
                    """
                    SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign
                    FROM normal_tracks
                    WHERE flight_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (flight_id,),
                )
                rows = cursor.fetchall()
            
            conn.close()
            
            if rows:
                for r in rows:
                    points.append({
                        "flight_id": r["flight_id"],
                        "timestamp": r["timestamp"],
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "alt": r["alt"],
                        "heading": r["track"],  # Use track as heading since heading column doesn't exist
                        "gspeed": r["gspeed"],
                        "vspeed": r["vspeed"],
                        "track": r["track"],
                        "squawk": r["squawk"],
                        "callsign": r["callsign"],
                        "source": "research",
                    })
                return {"flight_id": flight_id, "points": points}
        except Exception as e:
            logger.error(f"Error fetching from research_new.db: {e}")
    
    raise HTTPException(status_code=404, detail=f"No track data for {flight_id}")


@router.get("/api/feedback/track/live/{flight_id}")
def get_live_track(flight_id: str):
    """
    Return track points from live schema in PostgreSQL.
    This is used for real-time monitoring data.
    
    Args:
        flight_id: Flight ID to fetch
    
    Returns:
        Dict with flight_id and points list
    """
    try:
        from service.pg_provider import get_flight_track
        
        # Fetch track from live schema
        points = get_flight_track(flight_id, schema='live')
        
        if not points:
            raise HTTPException(status_code=404, detail=f"No track data found in live schema for {flight_id}")
        
        # Add heading alias for compatibility (frontend expects 'heading')
        for point in points:
            if 'track' in point and 'heading' not in point:
                point['heading'] = point['track']
        
        logger.info(f"[LIVE TRACK] Returning {len(points)} points for {flight_id} from PostgreSQL live schema")
        
        return {
            "flight_id": flight_id,
            "points": points,
            "source": "postgresql.live",
            "total_points": len(points),
        }
    
    except HTTPException:
        raise
    except ImportError as ie:
        logger.error(f"PostgreSQL provider not available: {ie}")
        raise HTTPException(status_code=500, detail="PostgreSQL provider not available")
    except Exception as e:
        logger.error(f"Failed to fetch live track: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/feedback/reanalyze/{flight_id}")
def reanalyze_feedback_flight(
    flight_id: str,
    current_user=Depends(lambda: _get_feedback_permission_dependency() or (lambda: None)())
):
    """
    Re-run analysis for a flight, update feedback DB and present_anomalies DB,
    and return the new report.
    """
    pipeline = _get_pipeline()

    try:
        # 1. Fetch flight data (try feedback track first, then others)
        points = []
        if PRESENT_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(PRESENT_DB_PATH))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, timestamp, source
                    FROM flight_tracks
                    WHERE flight_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (flight_id,),
                )
                rows = cursor.fetchall()
                conn.close()
                for r in rows:
                    points.append(dict(r))
            except Exception as e:
                logger.error(f"Reanalyze: Failed to fetch from present_anomalies: {e}")

        if not points:
            # Try unified track logic
            track_data = _get_unified_track(flight_id)
            points = track_data.get("points", [])

        if not points:
            raise HTTPException(status_code=404, detail="Flight track not found")

        # Reconstruct Flight object
        from core.models import FlightTrack, TrackPoint

        track_points = []
        for p in points:
            # Handle potential key mismatch
            track_points.append(TrackPoint(
                flight_id=flight_id,
                lat=p.get("lat"),
                lon=p.get("lon"),
                alt=p.get("alt"),
                timestamp=p.get("timestamp"),
                gspeed=p.get("gspeed"),
                vspeed=p.get("vspeed"),
                track=p.get("heading") or p.get("track"),
                squawk=p.get("squawk"),
                callsign=p.get("callsign"),
                source=p.get("source")
            ))

        flight = FlightTrack(flight_id=flight_id, points=track_points)

        # 2. Run Analysis
        results = pipeline.analyze(flight)
        full_report = results

        # Serialize report
        report_json = json.dumps(full_report)

        # 3. Update Feedback DB (Source of Truth)
        if FEEDBACK_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE user_feedback SET full_report_json = ? WHERE flight_id = ?",
                    (report_json, flight_id)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Reanalyze: Failed to update feedback DB: {e}")

        # 4. Update present_anomalies.db using the new comprehensive update function
        try:
            # Fetch flight details for comprehensive update
            flight_details = None
            try:
                callsign_for_lookup = None
                if points:
                    for p in points:
                        if p.get("callsign"):
                            callsign_for_lookup = p["callsign"]
                            break
                
                flight_time = points[0].get("timestamp") if points else int(datetime.now().timestamp())
                flight_details = _fetch_flight_details(flight_id, flight_time, callsign_for_lookup)
            except Exception as e:
                logger.warning(f"Reanalyze: Failed to fetch flight details: {e}")
            
            # Use the new update_flight_record function which handles everything
            _update_flight_record(
                flight_id=flight_id,
                tagged=True,  # Keep it tagged
                flight_details=flight_details,
                full_report=full_report,
                points=points,
                research_db_path=str(DB_RESEARCH_PATH) if DB_RESEARCH_PATH.exists() else None
            )
            logger.info(f"Reanalyze: Updated present_anomalies.db for {flight_id} with new analysis and research metadata")
        except Exception as e:
            logger.error(f"Reanalyze: Failed to update present_anomalies DB: {e}")
            # Don't fail the request if just this DB update fails, return result anyway

        return full_report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Re-analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/feedback/history")
def get_feedback_history(start_ts: int = 0, end_ts: int = None, limit: int = 100, include_normal: bool = True):
    """
    Fetch tagged flights from feedback_tagged.db (the clean database).
    Returns flights with metadata, tracks count, and anomaly reports.
    Filters by flight time (first_seen_ts), not when it was tagged.
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp  
        limit: Max results
        include_normal: If True, includes flights tagged as normal (user_label=0) as well
    """
    if end_ts is None:
        end_ts = int(datetime.now().timestamp())
    
    if not FEEDBACK_TAGGED_DB_PATH.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get flights with their feedback info and metadata
        label_filter = "" if include_normal else "AND uf.user_label = 1"
        
        cursor.execute(f"""
            SELECT 
                uf.flight_id,
                uf.first_seen_ts,
                uf.last_seen_ts,
                uf.tagged_at,
                COALESCE(uf.first_seen_ts, uf.tagged_at) as timestamp,
                uf.rule_id,
                uf.rule_name,
                uf.comments,
                uf.other_details,
                uf.user_label,
                fm.callsign,
                fm.flight_number,
                fm.airline,
                fm.origin_airport,
                fm.destination_airport,
                fm.aircraft_type,
                fm.aircraft_registration,
                fm.category,
                fm.is_military,
                fm.total_points,
                fm.flight_duration_sec,
                fm.max_altitude_ft,
                fm.avg_altitude_ft,
                fm.min_altitude_ft,
                fm.avg_speed_kts,
                fm.max_speed_kts,
                fm.min_speed_kts,
                fm.total_distance_nm,
                fm.scheduled_departure,
                fm.scheduled_arrival,
                fm.squawk_codes,
                fm.emergency_squawk_detected,
                ar.full_report,
                ar.severity_cnn,
                ar.severity_dense,
                ar.matched_rule_ids,
                ar.matched_rule_names,
                ar.matched_rule_categories,
                (SELECT COUNT(*) FROM flight_tracks ft WHERE ft.flight_id = uf.flight_id) as track_count
            FROM user_feedback uf
            LEFT JOIN flight_metadata fm ON uf.flight_id = fm.flight_id
            LEFT JOIN anomaly_reports ar ON uf.flight_id = ar.flight_id
            WHERE COALESCE(uf.first_seen_ts, uf.tagged_at) BETWEEN ? AND ?
              {label_filter}
            ORDER BY COALESCE(uf.first_seen_ts, uf.tagged_at) DESC
            LIMIT ?
        """, (start_ts, end_ts, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            full_report = row['full_report']
            if isinstance(full_report, (str, bytes)):
                try:
                    full_report = json.loads(full_report)
                except:
                    pass
            
            # Get origin and destination from DB
            origin_airport = row['origin_airport']
            destination_airport = row['destination_airport']
            
            # If origin or destination is missing, calculate from track points
            if not origin_airport or not destination_airport:
                calculated_origin, calculated_dest = _calculate_origin_destination_from_track(row['flight_id'])
                if not origin_airport and calculated_origin:
                    origin_airport = calculated_origin
                if not destination_airport and calculated_dest:
                    destination_airport = calculated_dest
            
            result.append({
                'flight_id': row['flight_id'],
                'timestamp': row['timestamp'],
                'first_seen_ts': row['first_seen_ts'],
                'last_seen_ts': row['last_seen_ts'],
                'tagged_at': row['tagged_at'],
                'is_anomaly': bool(row['user_label']),
                'user_label': row['user_label'],
                'rule_id': row['rule_id'],
                'rule_name': row['rule_name'],
                'comments': row['comments'],
                'other_details': row['other_details'],
                'callsign': row['callsign'],
                'flight_number': row['flight_number'],
                'airline': row['airline'],
                'origin_airport': origin_airport,
                'destination_airport': destination_airport,
                'aircraft_type': row['aircraft_type'],
                'aircraft_registration': row['aircraft_registration'],
                'category': row['category'],
                'is_military': row['is_military'],
                'total_points': row['total_points'] or row['track_count'],
                'flight_duration_sec': row['flight_duration_sec'],
                'max_altitude_ft': row['max_altitude_ft'],
                'avg_altitude_ft': row['avg_altitude_ft'],
                'min_altitude_ft': row['min_altitude_ft'],
                'avg_speed_kts': row['avg_speed_kts'],
                'max_speed_kts': row['max_speed_kts'],
                'min_speed_kts': row['min_speed_kts'],
                'total_distance_nm': row['total_distance_nm'],
                'scheduled_departure': row['scheduled_departure'],
                'scheduled_arrival': row['scheduled_arrival'],
                'squawk_codes': row['squawk_codes'],
                'emergency_squawk_detected': row['emergency_squawk_detected'],
                'full_report': full_report,
                'severity_cnn': row['severity_cnn'],
                'severity_dense': row['severity_dense'],
                'matched_rule_ids': row['matched_rule_ids'],
                'matched_rule_names': row['matched_rule_names'],
                'matched_rule_categories': row['matched_rule_categories'],
                'track_count': row['track_count'],
                'source': 'tagged'
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching from feedback_tagged.db: {e}")
        return []


@router.get("/api/feedback/tagged/history")
def get_tagged_feedback_history(start_ts: int = 0, end_ts: int = None, limit: int = 100, include_normal: bool = False):
    """
    Fetch tagged flights from PostgreSQL feedback schema.
    Returns flights with metadata, tracks count, and anomaly reports.
    Filters by flight time (first_seen_ts), not when it was tagged.
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp
        limit: Maximum results
        include_normal: If True, includes flights tagged as normal (user_label=0) as well
    """
    try:
        # Import PostgreSQL provider
        from service.pg_provider import get_tagged_feedback_history as pg_get_history
        
        # Use PostgreSQL provider
        return pg_get_history(
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
            include_normal=include_normal
        )
        
    except ImportError as ie:
        logger.error(f"PostgreSQL provider not available, falling back to SQLite: {ie}")
        # Fallback to SQLite if PostgreSQL provider fails
        # return _get_tagged_feedback_history_sqlite(start_ts, end_ts, limit, include_normal)
    except Exception as e:
        logger.error(f"Error fetching from PostgreSQL: {e}", exc_info=True)
        # Fallback to SQLite on error
        # return _get_tagged_feedback_history_sqlite(start_ts, end_ts, limit, include_normal)


def _get_tagged_feedback_history_sqlite(start_ts: int = 0, end_ts: int = None, limit: int = 100, include_normal: bool = False):
    """
    LEGACY: SQLite fallback for tagged feedback history.
    This is kept as a backup in case PostgreSQL is unavailable.
    """
    if end_ts is None:
        end_ts = int(datetime.now().timestamp())
    
    if not FEEDBACK_TAGGED_DB_PATH.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        label_filter = "" if include_normal else "AND uf.user_label = 1"
        
        cursor.execute(f"""
            SELECT 
                uf.flight_id,
                uf.first_seen_ts,
                uf.last_seen_ts,
                uf.tagged_at,
                COALESCE(uf.first_seen_ts, uf.tagged_at) as timestamp,
                uf.rule_id,
                uf.rule_name,
                uf.comments,
                uf.other_details,
                uf.user_label,
                fm.callsign,
                fm.flight_number,
                fm.airline,
                fm.origin_airport,
                fm.destination_airport,
                fm.aircraft_type,
                fm.aircraft_registration,
                fm.category,
                fm.is_military,
                fm.total_points,
                fm.flight_duration_sec,
                fm.max_altitude_ft,
                fm.avg_altitude_ft,
                fm.min_altitude_ft,
                fm.avg_speed_kts,
                fm.max_speed_kts,
                fm.min_speed_kts,
                fm.total_distance_nm,
                fm.scheduled_departure,
                fm.scheduled_arrival,
                fm.squawk_codes,
                fm.emergency_squawk_detected,
                ar.full_report,
                ar.severity_cnn,
                ar.severity_dense,
                ar.matched_rule_ids,
                ar.matched_rule_names,
                ar.matched_rule_categories,
                (SELECT COUNT(*) FROM flight_tracks ft WHERE ft.flight_id = uf.flight_id) as track_count
            FROM user_feedback uf
            LEFT JOIN flight_metadata fm ON uf.flight_id = fm.flight_id
            LEFT JOIN anomaly_reports ar ON uf.flight_id = ar.flight_id
            WHERE COALESCE(uf.first_seen_ts, uf.tagged_at) BETWEEN ? AND ?
              {label_filter}
            ORDER BY COALESCE(uf.first_seen_ts, uf.tagged_at) DESC
            LIMIT ?
        """, (start_ts, end_ts, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            full_report = row['full_report']
            if isinstance(full_report, (str, bytes)):
                try:
                    full_report = json.loads(full_report)
                except:
                    pass
            
            # Get origin and destination from DB
            origin_airport = row['origin_airport']
            destination_airport = row['destination_airport']
            
            # If origin or destination is missing, calculate from track points
            if not origin_airport or not destination_airport:
                calculated_origin, calculated_dest = _calculate_origin_destination_from_track(row['flight_id'])
                if not origin_airport and calculated_origin:
                    origin_airport = calculated_origin
                if not destination_airport and calculated_dest:
                    destination_airport = calculated_dest
            
            result.append({
                'flight_id': row['flight_id'],
                'timestamp': row['timestamp'],
                'first_seen_ts': row['first_seen_ts'],
                'last_seen_ts': row['last_seen_ts'],
                'tagged_at': row['tagged_at'],
                'is_anomaly': bool(row['user_label']),
                'user_label': row['user_label'],
                'rule_id': row['rule_id'],
                'rule_name': row['rule_name'],
                'comments': row['comments'],
                'other_details': row['other_details'],
                'callsign': row['callsign'],
                'flight_number': row['flight_number'],
                'airline': row['airline'],
                'origin_airport': origin_airport,
                'destination_airport': destination_airport,
                'aircraft_type': row['aircraft_type'],
                'aircraft_registration': row['aircraft_registration'],
                'category': row['category'],
                'is_military': row['is_military'],
                'total_points': row['total_points'] or row['track_count'],
                'flight_duration_sec': row['flight_duration_sec'],
                'max_altitude_ft': row['max_altitude_ft'],
                'avg_altitude_ft': row['avg_altitude_ft'],
                'min_altitude_ft': row['min_altitude_ft'],
                'avg_speed_kts': row['avg_speed_kts'],
                'max_speed_kts': row['max_speed_kts'],
                'min_speed_kts': row['min_speed_kts'],
                'total_distance_nm': row['total_distance_nm'],
                'scheduled_departure': row['scheduled_departure'],
                'scheduled_arrival': row['scheduled_arrival'],
                'squawk_codes': row['squawk_codes'],
                'emergency_squawk_detected': row['emergency_squawk_detected'],
                'full_report': full_report,
                'severity_cnn': row['severity_cnn'],
                'severity_dense': row['severity_dense'],
                'matched_rule_ids': row['matched_rule_ids'],
                'matched_rule_names': row['matched_rule_names'],
                'matched_rule_categories': row['matched_rule_categories'],
                'track_count': row['track_count'],
                'source': 'tagged'
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching from feedback_tagged.db: {e}")
        return []


@router.get("/api/feedback/tagged/track/{flight_id}")
def get_tagged_feedback_track(flight_id: str):
    """
    Get track points for a tagged flight from feedback_tagged.db.
    Falls back to research_new.db if not found (useful for "other flights" in proximity alerts).
    """
    points = []
    source_db = None
    
    # 1. Try feedback_tagged.db first
    if FEEDBACK_TAGGED_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source
                FROM flight_tracks
                WHERE flight_id = ?
                ORDER BY timestamp ASC
            """, (flight_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                source_db = "feedback_tagged"
                for r in rows:
                    points.append({
                        "flight_id": r["flight_id"],
                        "timestamp": r["timestamp"],
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "alt": r["alt"],
                        "heading": r["track"],
                        "gspeed": r["gspeed"],
                        "vspeed": r["vspeed"],
                        "track": r["track"],
                        "squawk": r["squawk"],
                        "callsign": r["callsign"],
                        "source": r["source"],
                    })
        except Exception as e:
            logger.warning(f"Error fetching from feedback_tagged.db: {e}")
    
    # 2. Fallback to research_new.db if not found
    if not points and DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try anomalies_tracks first
            cursor.execute("""
                SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign
                FROM anomalies_tracks
                WHERE flight_id = ?
                ORDER BY timestamp ASC
            """, (flight_id,))
            rows = cursor.fetchall()
            
            # Fallback to normal_tracks
            if not rows:
                cursor.execute("""
                    SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign
                    FROM normal_tracks
                    WHERE flight_id = ?
                    ORDER BY timestamp ASC
                """, (flight_id,))
                rows = cursor.fetchall()
            
            conn.close()
            
            if rows:
                source_db = "research_new"
                for r in rows:
                    points.append({
                        "flight_id": r["flight_id"],
                        "timestamp": r["timestamp"],
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "alt": r["alt"],
                        "heading": r["track"],
                        "gspeed": r["gspeed"],
                        "vspeed": r["vspeed"],
                        "track": r["track"],
                        "squawk": r["squawk"],
                        "callsign": r["callsign"],
                        "source": "research",
                    })
        except Exception as e:
            logger.warning(f"Error fetching from research_new.db: {e}")
    
    if not points:
        raise HTTPException(status_code=404, detail=f"Track not found for {flight_id} in feedback_tagged or research_new")
    
    logger.info(f"[FEEDBACK TRACK] Returning {len(points)} points for {flight_id} from {source_db}")
    return {"flight_id": flight_id, "points": points}


@router.get("/api/replay/other-flight/{flight_id}")
def get_replay_other_flight(flight_id: str):
    """
    Fetch track and metadata for "other flight" in proximity alerts for replay from PostgreSQL.
    Search order:
    1. feedback schema (tagged flights)
    2. research schema (research data)
    3. live schema (live monitoring data)
    
    Also fetches flight metadata from flight_metadata table.
    """
    try:
        from service.pg_provider import get_flight_track, get_flight_metadata
        
        points = []
        metadata = None
        source_schema = None
        callsign = None
        
        # 1. Try feedback schema first (tagged flights)
        try:
            points = get_flight_track(flight_id, schema='feedback')
            if points:
                source_schema = "feedback"
                logger.debug(f"Found track in feedback schema for {flight_id}")
                
                # Get metadata from feedback schema
                metadata = get_flight_metadata(flight_id, schema='feedback')
                
                # Add heading alias for compatibility
                for point in points:
                    if 'track' in point:
                        point['heading'] = point['track']
                
                # Get callsign from points or metadata
                callsign = next((p.get('callsign') for p in points if p.get('callsign')), None)
                if not callsign and metadata and metadata.get('callsign'):
                    callsign = metadata['callsign']
        except Exception as e:
            logger.debug(f"Error fetching from feedback schema: {e}")
        
        # 2. Fallback to research schema if not found
        if not points:
            try:
                points = get_flight_track(flight_id, schema='research')
                if points:
                    source_schema = "research"
                    logger.debug(f"Found track in research schema for {flight_id}")
                    
                    # Get metadata from research schema
                    metadata = get_flight_metadata(flight_id, schema='research')
                    
                    # Add heading alias for compatibility
                    for point in points:
                        if 'track' in point:
                            point['heading'] = point['track']
                    
                    # Get callsign from points or metadata
                    callsign = next((p.get('callsign') for p in points if p.get('callsign')), None)
                    if not callsign and metadata and metadata.get('callsign'):
                        callsign = metadata['callsign']
            except Exception as e:
                logger.warning(f"Error fetching from research schema: {e}")
        
        # 3. Fallback to live schema if still not found
        if not points:
            try:
                points = get_flight_track(flight_id, schema='live')
                if points:
                    source_schema = "live"
                    logger.debug(f"Found track in live schema for {flight_id}")
                    
                    # Get metadata from live schema
                    metadata = get_flight_metadata(flight_id, schema='live')
                    
                    # Add heading alias for compatibility
                    for point in points:
                        if 'track' in point:
                            point['heading'] = point['track']
                    
                    # Get callsign from points or metadata
                    callsign = next((p.get('callsign') for p in points if p.get('callsign')), None)
                    if not callsign and metadata and metadata.get('callsign'):
                        callsign = metadata['callsign']
            except Exception as e:
                logger.warning(f"Error fetching from live schema: {e}")
        
        if not points:
            raise HTTPException(status_code=404, detail=f"Track not found for {flight_id}")
        
        # Format metadata for response (ensure consistent structure)
        formatted_metadata = None
        if metadata:
            formatted_metadata = {
                "flight_id": flight_id,
                "callsign": metadata.get("callsign"),
                "airline": metadata.get("airline"),
                "origin_airport": metadata.get("origin_airport"),
                "destination_airport": metadata.get("destination_airport"),
                "aircraft_type": metadata.get("aircraft_type"),
                "is_military": bool(metadata.get("is_military")) if metadata.get("is_military") is not None else False,
                "category": metadata.get("category"),
            }
        
        logger.info(f"[REPLAY OTHER FLIGHT] Returning {len(points)} points for {flight_id} (callsign: {callsign}) from PostgreSQL {source_schema} schema")
        
        return {
            "flight_id": flight_id,
            "callsign": callsign,
            "points": points,
            "metadata": formatted_metadata,
            "source": f"postgresql.{source_schema}",
            "total_points": len(points),
        }
    
    except HTTPException:
        raise
    except ImportError as ie:
        logger.error(f"PostgreSQL provider not available: {ie}")
        raise HTTPException(status_code=500, detail="PostgreSQL provider not available")
    except Exception as e:
        logger.error(f"Failed to fetch replay other flight: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/feedback/tagged/metadata/{flight_id}")
def get_tagged_flight_metadata(flight_id: str):
    """
        Get full flight metadata for a tagged flight from PostgreSQL feedback schema.
    Calculates origin/destination from track if not available.
    """
    try:
        from service.pg_provider import get_flight_metadata, get_connection
        import psycopg2.extras
        
        # Get flight metadata from PostgreSQL feedback schema
        metadata = get_flight_metadata(flight_id, schema='feedback')
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Metadata not found in feedback schema")
        
        # If origin or destination is missing, calculate from track points
        origin_airport = metadata.get('origin_airport')
        destination_airport = metadata.get('destination_airport')
        
        if not origin_airport or not destination_airport:
            calculated_origin, calculated_dest = _calculate_origin_destination_from_track(flight_id)
            if not origin_airport and calculated_origin:
                metadata['origin_airport'] = calculated_origin
            if not destination_airport and calculated_dest:
                metadata['destination_airport'] = calculated_dest
        
        # Also get user feedback info from PostgreSQL
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM feedback.user_feedback WHERE flight_id = %s",
                    (flight_id,)
                )
                feedback_row = cursor.fetchone()
                
                if feedback_row:
                    feedback_data = dict(feedback_row)
                    # Parse JSON arrays for rule_ids and rule_names if they're strings
                    if feedback_data.get('rule_ids') and isinstance(feedback_data['rule_ids'], str):
                        try:
                            feedback_data['rule_ids'] = json.loads(feedback_data['rule_ids'])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if feedback_data.get('rule_names') and isinstance(feedback_data['rule_names'], str):
                        try:
                            feedback_data['rule_names'] = json.loads(feedback_data['rule_names'])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    metadata['feedback'] = feedback_data
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metadata from PostgreSQL feedback schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/feedback/{feedback_id}")
def update_feedback(feedback_id: int, update_data: dict):
    """
    Update existing feedback entry (for re-tagging).
    Payload: {
        "rule_id": 1 (required, null means "Other"),
        "comments": "...",
        "other_details": "..." (optional, used when rule_id is null/Other)
    }
    """
    if not FEEDBACK_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Feedback database not found")

    rule_id = update_data.get("rule_id")
    comments = update_data.get("comments", "")
    other_details = update_data.get("other_details", "")

    # Require either rule_id or other_details
    if rule_id is None and not other_details:
        raise HTTPException(status_code=400, detail="Either rule_id or other_details is required")

    try:
        conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
        cursor = conn.cursor()

        # Update the feedback entry and set tagged = 1
        cursor.execute(
            """UPDATE user_feedback 
               SET rule_id = ?, comments = ?, other_details = ?, tagged = 1
               WHERE id = ?""",
            (rule_id, comments, other_details, feedback_id)
        )

        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="Feedback entry not found")

        conn.commit()
        conn.close()

        logger.info(f"Updated feedback ID {feedback_id}: Rule={rule_id}, Tagged=1")
        return {"status": "success", "message": "Feedback updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _ensure_feedback_partitions(cursor, ts: int):
    """
    Ensure monthly partitions exist in the feedback schema for the given timestamp.
    Creates partitions for any partitioned feedback tables if they don't already exist.
    After creating a partition, copies SERIAL/sequence defaults from the parent table
    so that auto-increment columns work correctly.
    """
    dt = datetime.utcfromtimestamp(ts)
    year = dt.year
    month = dt.month

    # Partition boundaries: start of this month -> start of next month (UTC)
    start_ts = int(datetime(year, month, 1).timestamp())
    if month == 12:
        end_ts = int(datetime(year + 1, 1, 1).timestamp())
    else:
        end_ts = int(datetime(year, month + 1, 1).timestamp())

    suffix = f"{year}_{month:02d}"

    # All feedback tables that may be range-partitioned
    partitioned_tables = [
        "feedback.flight_metadata",
        "feedback.user_feedback",
        "feedback.anomaly_reports",
        "feedback.flight_tracks",
    ]

    for table in partitioned_tables:
        partition_name = f"{table}_{suffix}"
        try:
            cursor.execute("SAVEPOINT sp_part")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF {table}
                FOR VALUES FROM (%s) TO (%s)
                """,
                (start_ts, end_ts),
            )
            # Copy sequence defaults from parent to the new partition
            # (partitions don't inherit SERIAL defaults automatically)
            schema_name = table.split(".")[0]
            table_name = table.split(".")[1]
            cursor.execute(
                """
                SELECT column_name, column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                  AND column_default IS NOT NULL
                  AND column_default LIKE 'nextval%%'
                """,
                (schema_name, table_name),
            )
            for col_name, col_default in cursor.fetchall():
                part_schema = partition_name.split(".")[0]
                part_table = partition_name.split(".", 1)[1]
                try:
                    cursor.execute(
                        f"ALTER TABLE {partition_name} ALTER COLUMN {col_name} SET DEFAULT {col_default}"
                    )
                except Exception:
                    pass  # already set

            cursor.execute("RELEASE SAVEPOINT sp_part")
        except Exception as e:
            # Partition already exists or table is not partitioned — roll back savepoint only
            cursor.execute("ROLLBACK TO SAVEPOINT sp_part")
            logger.debug(f"Partition check {partition_name}: {e}")


@public_router.post("/api/feedback")
def submit_feedback(
    feedback: dict,
):
    """
    Submit user feedback for a flight.
    Copies ALL flight data (metadata, tracks, full anomaly report) from PostgreSQL
    research schema to feedback schema to ensure nothing is lost.
    
    Payload: {
        "flight_id": "...",
        "is_anomaly": true/false,
        "comments": "...",
        "rule_id": 1 (optional, kept for backward compatibility),
        "rule_ids": [1, 2, 3] (optional, array of rule IDs for multiple selection),
        "other_details": "..." (optional, used when rule_id is null/Other)
    }
    """
    import psycopg2
    import psycopg2.extras
    from service.pg_provider import get_connection, get_flight_track, get_flight_metadata

    flight_id = feedback.get("flight_id")
    is_anomaly = feedback.get("is_anomaly")
    comments = feedback.get("comments", "")
    rule_id = feedback.get("rule_id")
    rule_ids = feedback.get("rule_ids")
    other_details = feedback.get("other_details", "")

    # Support both old (rule_id) and new (rule_ids) format
    if rule_ids is None and rule_id is not None:
        rule_ids = [rule_id]

    # Get the first rule_id for backward compatibility with legacy systems
    if rule_ids and len(rule_ids) > 0:
        rule_id = rule_ids[0]

    if not flight_id or is_anomaly is None:
        raise HTTPException(status_code=400, detail="Missing flight_id or is_anomaly")

    # If marking as anomaly, require rule selection
    if is_anomaly and (not rule_ids or len(rule_ids) == 0) and not other_details:
        raise HTTPException(status_code=400,
                            detail="When marking as anomaly, either rule_ids or other_details is required")

    logger.info(f"[FEEDBACK] Received feedback for {flight_id}: Anomaly={is_anomaly}, Rules={rule_ids}")

    # ----------------------------------------------------------------
    # 1. Read flight data from PostgreSQL research schema
    # ----------------------------------------------------------------
    points = []
    metadata = None
    full_report = None
    anomaly_report_row = None

    try:
        # Get track points from research schema (tries anomalies_tracks, then normal_tracks)
        points = get_flight_track(flight_id, schema='research')
        if points:
            logger.info(f"[FEEDBACK] Found {len(points)} track points from research schema")
        else:
            logger.warning(f"[FEEDBACK] No tracks found for {flight_id} in research schema")

        # Get flight metadata from research schema
        metadata = get_flight_metadata(flight_id, schema='research')
        if metadata:
            logger.info(f"[FEEDBACK] Found metadata with {len(metadata)} fields from research schema")
            # Log key fields to debug NULL values
            logger.debug(f"[FEEDBACK] Metadata keys: {list(metadata.keys())}")
            logger.debug(f"[FEEDBACK] origin_airport={metadata.get('origin_airport')}, "
                        f"destination_airport={metadata.get('destination_airport')}, "
                        f"airline={metadata.get('airline')}, category={metadata.get('category')}")

        # Get anomaly report from research schema
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT * FROM research.anomaly_reports
                       WHERE flight_id = %s
                       ORDER BY timestamp DESC LIMIT 1""",
                    (flight_id,)
                )
                anomaly_report_row = cursor.fetchone()
                if anomaly_report_row:
                    anomaly_report_row = dict(anomaly_report_row)
                    raw_report = anomaly_report_row.get("full_report")
                    if raw_report:
                        if isinstance(raw_report, (str, bytes)):
                            try:
                                full_report = json.loads(raw_report)
                            except json.JSONDecodeError:
                                full_report = raw_report
                        elif isinstance(raw_report, dict):
                            full_report = raw_report
                    logger.info(f"[FEEDBACK] Found anomaly report from research schema")

    except Exception as e:
        logger.error(f"[FEEDBACK] Error reading from research schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read flight data: {str(e)}")

    # Validate we have at least tracks
    if not points:
        raise HTTPException(status_code=404,
                            detail=f"Flight tracks not found for {flight_id} in research schema. Cannot save feedback.")

    # ----------------------------------------------------------------
    # 2. Resolve rule names
    # ----------------------------------------------------------------
    rule_name = None
    rule_names = []
    if rule_ids:
        try:
            from rules.rule_engine import get_rule_by_id
            for rid in rule_ids:
                rule = get_rule_by_id(rid)
                if rule:
                    rule_names.append(rule.get("name", f"Rule {rid}"))
                else:
                    rule_names.append(f"Rule {rid}")
            if rule_names:
                rule_name = rule_names[0]
        except Exception:
            pass

    # ----------------------------------------------------------------
    # 3. Copy everything to PostgreSQL feedback schema
    # ----------------------------------------------------------------
    now_ts = int(datetime.now().timestamp())
    first_seen_ts = None
    last_seen_ts = None
    callsign = None

    if metadata:
        first_seen_ts = metadata.get("first_seen_ts")
        last_seen_ts = metadata.get("last_seen_ts")
        callsign = metadata.get("callsign")

    if not callsign and points:
        callsign = next((p.get("callsign") for p in points if p.get("callsign")), None)

    if not first_seen_ts and points:
        timestamps = [p.get("timestamp", 0) for p in points if p.get("timestamp")]
        if timestamps:
            first_seen_ts = min(timestamps)
            last_seen_ts = max(timestamps)

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # --- Ensure partitions exist for feedback tables ---
                _ensure_feedback_partitions(cursor, first_seen_ts or now_ts)

                # --- Delete any existing data for this flight (safe upsert) ---
                cursor.execute("DELETE FROM feedback.flight_tracks WHERE flight_id = %s", (flight_id,))
                cursor.execute("DELETE FROM feedback.anomaly_reports WHERE flight_id = %s", (flight_id,))
                cursor.execute("DELETE FROM feedback.user_feedback WHERE flight_id = %s", (flight_id,))
                cursor.execute("DELETE FROM feedback.flight_metadata WHERE flight_id = %s", (flight_id,))

                # --- 3a. Copy flight_metadata to feedback.flight_metadata ---
                if metadata:
                    cursor.execute(
                        """INSERT INTO feedback.flight_metadata (
                                flight_id, callsign, flight_number,
                                origin_airport, destination_airport,
                                airline, aircraft_type, aircraft_model, aircraft_registration,
                                category, is_military,
                                first_seen_ts, last_seen_ts, flight_duration_sec, total_points,
                                min_altitude_ft, max_altitude_ft, avg_altitude_ft,
                                min_speed_kts, max_speed_kts, avg_speed_kts,
                                start_lat, start_lon, end_lat, end_lon,
                                total_distance_nm,
                                squawk_codes, emergency_squawk_detected,
                                scheduled_departure, scheduled_arrival
                           ) VALUES (
                                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                %s,%s,%s,%s,%s,%s,%s,%s,%s
                           )
                        """,
                        (
                            flight_id,
                            metadata.get("callsign"),
                            metadata.get("flight_number"),
                            metadata.get("origin_airport"),
                            metadata.get("destination_airport"),
                            metadata.get("airline"),
                            metadata.get("aircraft_type"),
                            metadata.get("aircraft_model"),
                            metadata.get("aircraft_registration"),
                            metadata.get("category"),
                            metadata.get("is_military", False),
                            metadata.get("first_seen_ts"),
                            metadata.get("last_seen_ts"),
                            metadata.get("flight_duration_sec"),
                            metadata.get("total_points"),
                            metadata.get("min_altitude_ft"),
                            metadata.get("max_altitude_ft"),
                            metadata.get("avg_altitude_ft"),
                            metadata.get("min_speed_kts"),
                            metadata.get("max_speed_kts"),
                            metadata.get("avg_speed_kts"),
                            metadata.get("start_lat"),
                            metadata.get("start_lon"),
                            metadata.get("end_lat"),
                            metadata.get("end_lon"),
                            metadata.get("total_distance_nm"),
                            metadata.get("squawk_codes"),
                            metadata.get("emergency_squawk_detected", False),
                            metadata.get("scheduled_departure"),
                            metadata.get("scheduled_arrival"),
                        ),
                    )
                    logger.info(f"[FEEDBACK] Copied flight_metadata to feedback schema")

                # --- 3b. Copy anomaly report to feedback.anomaly_reports ---
                if anomaly_report_row:
                    report_json = full_report if isinstance(full_report, (dict, list)) else anomaly_report_row.get("full_report")
                    if isinstance(report_json, (dict, list)):
                        report_json = json.dumps(report_json)
                    cursor.execute(
                        """INSERT INTO feedback.anomaly_reports (
                                flight_id, timestamp, full_report, is_anomaly,
                                severity_cnn, severity_dense,
                                matched_rule_ids, matched_rule_names, matched_rule_categories,
                                callsign, airline, origin_airport, destination_airport,
                                aircraft_type, flight_duration_sec, max_altitude_ft, avg_speed_kts,
                                nearest_airport, geographic_region, is_military
                           ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            flight_id,
                            anomaly_report_row.get("timestamp", now_ts),
                            report_json,
                            is_anomaly,  # Use the user's feedback
                            anomaly_report_row.get("severity_cnn"),
                            anomaly_report_row.get("severity_dense"),
                            anomaly_report_row.get("matched_rule_ids"),
                            anomaly_report_row.get("matched_rule_names"),
                            anomaly_report_row.get("matched_rule_categories"),
                            anomaly_report_row.get("callsign"),
                            anomaly_report_row.get("airline"),
                            anomaly_report_row.get("origin_airport"),
                            anomaly_report_row.get("destination_airport"),
                            anomaly_report_row.get("aircraft_type"),
                            anomaly_report_row.get("flight_duration_sec"),
                            anomaly_report_row.get("max_altitude_ft"),
                            anomaly_report_row.get("avg_speed_kts"),
                            anomaly_report_row.get("nearest_airport"),
                            anomaly_report_row.get("geographic_region"),
                            anomaly_report_row.get("is_military"),
                        ),
                    )
                    logger.info(f"[FEEDBACK] Copied anomaly_report to feedback schema")

                # --- 3c. Batch-copy track points to feedback.flight_tracks ---
                if points:
                    # Build VALUES clause with placeholders
                    values_placeholders = []
                    values_data = []
                    for pt in points:
                        values_placeholders.append("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
                        values_data.extend([
                            flight_id,
                            pt.get("timestamp"),
                            pt.get("lat"),
                            pt.get("lon"),
                            pt.get("alt"),
                            pt.get("gspeed"),
                            pt.get("vspeed"),
                            pt.get("track"),
                            pt.get("squawk"),
                            pt.get("callsign"),
                            "feedback",
                        ])
                    
                    values_clause = ",".join(values_placeholders)
                    cursor.execute(
                        """INSERT INTO feedback.flight_tracks (
                                flight_id, timestamp, lat, lon, alt,
                                gspeed, vspeed, track, squawk, callsign, source
                           ) VALUES """ + values_clause,
                        values_data,
                    )
                logger.info(f"[FEEDBACK] Batch-copied {len(points)} track points to feedback.flight_tracks")

                # --- 3d. Save user feedback record to feedback.user_feedback ---
                # Get next id value by querying max(id) and adding 1
                # This is the simplest approach that works across all partition setups
                cursor.execute("""
                    SELECT COALESCE(MAX(id), 0) + 1 FROM feedback.user_feedback
                """)
                next_id = cursor.fetchone()[0]
                
                logger.info(f"[FEEDBACK] Using id={next_id} for user_feedback")
                
                cursor.execute(
                    """INSERT INTO feedback.user_feedback (
                            id, flight_id, tagged_at, first_seen_ts, last_seen_ts,
                            user_label, comments,
                            rule_id, rule_ids, rule_name, rule_names,
                            other_details
                       ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        next_id,
                        flight_id,
                        now_ts,
                        first_seen_ts,
                        last_seen_ts,
                        1 if is_anomaly else 0,
                        comments,
                        rule_id,
                        rule_ids,
                        rule_name,
                        rule_names if rule_names else None,
                        other_details,
                    ),
                )
                logger.info(f"[FEEDBACK] Saved user_feedback record")

            conn.commit()

    except Exception as e:
        logger.error(f"[FEEDBACK] Error writing to feedback schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

    logger.info(f"[FEEDBACK] Successfully saved tagged flight {flight_id} to PostgreSQL feedback schema "
                f"(tracks={len(points)}, metadata={'YES' if metadata else 'NO'}, report={'YES' if full_report else 'NO'})")

    return {"status": "success", "message": "Feedback saved to PostgreSQL feedback schema"}

