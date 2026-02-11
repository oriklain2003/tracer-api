"""
Flight data routes - tracks, analysis, rules, paths, live/research data.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import dataclasses
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from routes.users import get_current_user
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import core models for pipeline integration
try:
    from core.models import FlightTrack, TrackPoint
except ImportError:
    # Will be imported lazily when needed
    FlightTrack = None
    TrackPoint = None

logger = logging.getLogger(__name__)

# TEMPORARILY DISABLED AUTH FOR TESTING
# router = APIRouter(tags=["Flights"], dependencies=[Depends(get_current_user)])
router = APIRouter(tags=["Flights"])
# Separate router for research_rerun routes (no authentication required)
research_rerun_router = APIRouter(tags=["Research Rerun - Public"])

# These will be set by the main api.py module
CACHE_DB_PATH: Path = None
DB_ANOMALIES_PATH: Path = None
DB_TRACKS_PATH: Path = None
DB_RESEARCH_PATH: Path = None
DB_LIVE_RESEARCH_PATH: Path = None  # New live monitor DB
FEEDBACK_TAGGED_DB_PATH: Path = None

# Function references that will be set by api.py
_get_pipeline = None
_serialize_flight = None
_deserialize_flight = None
_search_flight_path = None
_get_flight = None


def configure(
    cache_db_path: Path,
    db_anomalies_path: Path,
    db_tracks_path: Path,
    db_research_path: Path,
    feedback_tagged_db_path: Path,
    get_pipeline_func,
    serialize_flight_func,
    deserialize_flight_func,
    search_flight_path_func,
    get_flight_func,
    db_live_research_path: Path = None,
):
    """Configure the router with paths and dependencies from api.py"""
    global CACHE_DB_PATH, DB_ANOMALIES_PATH, DB_TRACKS_PATH, DB_RESEARCH_PATH, DB_LIVE_RESEARCH_PATH, FEEDBACK_TAGGED_DB_PATH
    global _get_pipeline, _serialize_flight, _deserialize_flight, _search_flight_path, _get_flight
    
    CACHE_DB_PATH = cache_db_path
    DB_ANOMALIES_PATH = db_anomalies_path
    DB_TRACKS_PATH = db_tracks_path
    DB_RESEARCH_PATH = db_research_path
    DB_LIVE_RESEARCH_PATH = db_live_research_path
    FEEDBACK_TAGGED_DB_PATH = feedback_tagged_db_path
    _get_pipeline = get_pipeline_func
    _serialize_flight = serialize_flight_func
    _deserialize_flight = deserialize_flight_func
    _search_flight_path = search_flight_path_func
    _get_flight = get_flight_func


# Rule Definitions - Matches web2 UI tagging rules (source of truth)
RULES_METADATA = [
    # Emergency & Safety (Red)
    {"id": 1, "name": "Emergency Squawks", "nameHe": "קודי חירום", "description": "Transponder emergency code (7500, 7600, 7700)", "category": "emergency", "color": "red"},
    {"id": 2, "name": "Crash", "nameHe": "התרסקות", "description": "Aircraft crash or suspected crash event", "category": "emergency", "color": "red"},
    {"id": 3, "name": "Proximity Alert", "nameHe": "התראת קרבה", "description": "Dangerous proximity between aircraft", "category": "emergency", "color": "red"},
    
    # Flight Operations (Blue)
    {"id": 4, "name": "Holding Pattern", "nameHe": "דפוס המתנה", "description": "Aircraft in holding pattern", "category": "flight_ops", "color": "blue"},
    {"id": 5, "name": "Go Around", "nameHe": "גו-אראונד", "description": "Aborted landing and go-around maneuver", "category": "flight_ops", "color": "blue"},
    {"id": 6, "name": "Return to Land", "nameHe": "חזרה לנחיתה", "description": "Aircraft returning to departure airport", "category": "flight_ops", "color": "blue"},
    {"id": 7, "name": "Unplanned Landing", "nameHe": "נחיתה לא מתוכננת", "description": "Landing at unplanned airport", "category": "flight_ops", "color": "blue"},
    
    # Technical (Purple)
    {"id": 8, "name": "Signal Loss", "nameHe": "אובדן אות", "description": "Loss of ADS-B signal", "category": "technical", "color": "purple"},
    {"id": 9, "name": "Off Course", "nameHe": "סטייה ממסלול", "description": "Significant deviation from expected flight path", "category": "technical", "color": "purple"},
    {"id": 14, "name": "GPS Jamming", "nameHe": "שיבוש GPS", "description": "GPS jamming indicators detected", "category": "technical", "color": "purple"},
    
    # Military (Green)
    {"id": 10, "name": "Military Flight", "nameHe": "טיסה צבאית", "description": "Identified military aircraft", "category": "military", "color": "green"},
    {"id": 11, "name": "Operational Military", "nameHe": "טיסה צבאית מבצעית", "description": "Military aircraft on operational mission", "category": "military", "color": "green"},
    {"id": 12, "name": "Suspicious Behavior", "nameHe": "התנהגות חשודה", "description": "Unusual or suspicious flight behavior", "category": "military", "color": "green"},
    {"id": 13, "name": "Flight Academy", "nameHe": "בית ספר לטיסה", "description": "Training flight from flight school", "category": "military", "color": "green"},
]


class SearchRequest(BaseModel):
    callsign: str
    from_date: datetime
    to_date: datetime


@router.post("/api/search")
def search_flight_endpoint(request: SearchRequest):
    """
    Search for a flight path by callsign and date range.
    """
    pipeline = _get_pipeline()

    try:
        # Search for flight
        logger.info(f"Searching for {request.callsign} between {request.from_date} and {request.to_date}")
        flight = _search_flight_path(request.callsign, request.from_date, request.to_date)

        if not flight or not flight.points:
            raise HTTPException(status_code=404, detail=f"No flight found for {request.callsign} in specified range")

        # Save to Cache
        flight_id = flight.flight_id
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO flights (flight_id, fetched_at, data) VALUES (?, ?, ?)",
            (flight_id, int(time.time()), _serialize_flight(flight))
        )
        conn.commit()
        conn.close()
        logger.info(f"Saved {flight_id} to cache.")

        # Run Pipeline
        results = pipeline.analyze(flight)

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/rules")
def get_rules():
    """
    Return list of available anomaly rules.
    """
    return RULES_METADATA


@router.get("/api/rules/{rule_id}/flights")
def get_flights_by_rule(rule_id: int):
    """
    Get all flights that triggered a specific rule from Research DB.
    """
    if not FEEDBACK_TAGGED_DB_PATH.exists():
        return []

    try:
        conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
        cursor = conn.cursor()

        # Use JSON extract to find flights where this rule was triggered
        query = """
            SELECT DISTINCT 
                t1.flight_id, 
                t1.timestamp, 
                t1.is_anomaly, 
                t1.severity_cnn, 
                t1.severity_dense, 
                t1.full_report
            FROM anomaly_reports t1, 
                 json_each(t1.full_report, '$.layer_1_rules.report.matched_rules') as rule
            WHERE json_extract(rule.value, '$.id') = ?
            ORDER BY t1.timestamp DESC
            LIMIT 200
        """

        cursor.execute(query, (rule_id,))
        rows = cursor.fetchall()

        # Fetch callsigns and flight_numbers for these flights
        flight_ids = [r[0] for r in rows]
        callsigns = {}
        flight_numbers = {}

        if flight_ids:
            placeholders = ",".join(["?"] * len(flight_ids))
            try:
                cursor.execute(
                    f"SELECT flight_id, callsign FROM anomalies_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL",
                    flight_ids)
                for fid, cs in cursor.fetchall():
                    if cs: callsigns[fid] = cs
            except:
                pass

            # Try normal_tracks as fallback
            try:
                cursor.execute(
                    f"SELECT flight_id, callsign FROM normal_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL",
                    flight_ids)
                for fid, cs in cursor.fetchall():
                    if cs and fid not in callsigns: callsigns[fid] = cs
            except:
                pass

            # Fetch flight_number from flight_metadata table
            try:
                cursor.execute(
                    f"SELECT flight_id, flight_number, callsign FROM flight_metadata WHERE flight_id IN ({placeholders})",
                    flight_ids)
                for fid, fn, cs in cursor.fetchall():
                    if fn: flight_numbers[fid] = fn
                    if cs and fid not in callsigns: callsigns[fid] = cs
            except:
                pass

        conn.close()

        results = []
        for row in rows:
            report = row[5]
            if isinstance(report, str):
                try:
                    report = json.loads(report)
                except:
                    pass

            flight_id = row[0]
            callsign = callsigns.get(flight_id)
            flight_number = flight_numbers.get(flight_id)

            if not callsign and isinstance(report, dict):
                callsign = report.get("summary", {}).get("callsign")

            results.append({
                "flight_id": flight_id,
                "timestamp": row[1],
                "is_anomaly": bool(row[2]),
                "severity_cnn": row[3],
                "severity_dense": row[4],
                "full_report": report,
                "callsign": callsign,
                "flight_number": flight_number
            })

        return results

    except Exception as e:
        logger.error(f"Failed to fetch flights by rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/track/unified/{flight_id}")
def get_unified_track(flight_id: str):
    """
    Get flight track from any available source (PostgreSQL feedback, SQLite DBs, Cache),
    or fall back to fetching and analyzing if not found.
    
    Now with PostgreSQL support for better performance!
    """
    # 1. Try PostgreSQL feedback schema first
    try:
        from service.pg_provider import get_unified_track as pg_get_unified_track
        
        result = pg_get_unified_track(flight_id)
        if result:
            logger.info(f"Unified track - Found in PostgreSQL feedback for {flight_id}")
            return result
        
        logger.debug(f"Unified track - Not found in PostgreSQL, trying SQLite")
        
    except ImportError as ie:
        logger.debug(f"PostgreSQL provider not available, using SQLite: {ie}")
    except Exception as e:
        logger.error(f"Error fetching from PostgreSQL: {e}", exc_info=True)
        logger.debug(f"Falling back to SQLite for {flight_id}")
    
    # 2. Fallback to SQLite databases
    points = []

    # 2a. Check Live DB
    if DB_TRACKS_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_TRACKS_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            conn.close()
            if rows:
                points = []
                for row in rows:
                    points.append({
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "alt": row["alt"],
                        "timestamp": row["timestamp"],
                        "gspeed": row["gspeed"],
                        "track": row["track"],
                        "flight_id": row["flight_id"]
                    })
        except Exception as e:
            logger.error(f"Unified track - Live DB error: {e}")

    # 2b. Check Research DB if not found
    if not points and DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Try anomalies_tracks
            cursor.execute("SELECT * FROM anomalies_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            if not rows:
                # Try normal_tracks
                cursor.execute("SELECT * FROM normal_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                rows = cursor.fetchall()

            conn.close()

            if rows:
                points = [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Unified track - Research DB error: {e}")

    # 2c. Check Feedback Tagged DB if not found
    if not points and FEEDBACK_TAGGED_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            conn.close()
            if rows:
                points = [dict(r) for r in rows]
                logger.info(f"Unified track - Found {len(points)} points in feedback_tagged.db for {flight_id}")
        except Exception as e:
            logger.error(f"Unified track - Feedback Tagged DB error: {e}")

    # 2d. Check Cache DB if not found
    if not points:
        try:
            conn = sqlite3.connect(str(CACHE_DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                data = json.loads(row[0])
                points = data.get("points", [])
        except Exception as e:
            logger.error(f"Unified track - Cache DB error: {e}")

    # Return if found in SQLite
    if points:
        return {
            "flight_id": flight_id,
            "points": points
        }

    # 3. Fallback: Fetch and Analyze
    logger.info(f"Unified track - Not found in any DB, analyzing {flight_id}")
    try:
        # Call analyze_flight_endpoint logic to reuse fetch/analyze code
        analysis_result = analyze_flight_endpoint(flight_id)

        if "track" in analysis_result:
            return analysis_result["track"]
        else:
            raise HTTPException(status_code=404, detail="Track not found after analysis")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified track - Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper function for the fallback
def analyze_flight_endpoint(flight_id: str):
    """
    Analyze a flight by ID using the AnomalyPipeline.
    Checks local cache first, then fetches from FR24.
    """
    pipeline = _get_pipeline()

    try:
        flight = None

        # 1. Check Cache
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            logger.info(f"Cache HIT for {flight_id}")
            flight = _deserialize_flight(row[0])
        else:
            logger.info(f"Cache MISS for {flight_id}. Fetching live...")
            # Fetch Flight Data
            t_fetch = time.time()
            flight = _get_flight(flight_id=flight_id)
            logger.info(f"  [Timer] Fetch: {time.time() - t_fetch:.4f}s")

            if flight and flight.points:
                # Save to Cache
                conn = sqlite3.connect(str(CACHE_DB_PATH))
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO flights (flight_id, fetched_at, data) VALUES (?, ?, ?)",
                    (flight_id, int(time.time()), _serialize_flight(flight))
                )
                conn.commit()
                conn.close()
                logger.info(f"Saved {flight_id} to cache.")

        if not flight or not flight.points:
            raise HTTPException(status_code=404, detail=f"Flight data not found for {flight_id}")

        # 2. Run Pipeline
        results = pipeline.analyze(flight)

        # Inject full points for UI visualization
        results["track"] = {
            "flight_id": flight.flight_id,
            "points": [dataclasses.asdict(p) for p in flight.points]
        }

        return results

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analyze/from-db/{flight_id}")
def analyze_flight_from_db_endpoint(flight_id: str):
    """
    Analyze a flight by ID using ONLY database data (no FR24 fetch).
    Checks Live DB, Research DB, Feedback Tagged DB, and Cache - but never fetches from FR24.
    """
    pipeline = _get_pipeline()

    try:
        points = []

        # 1. Check Live DB
        if DB_TRACKS_PATH and DB_TRACKS_PATH.exists():
            try:
                conn = sqlite3.connect(str(DB_TRACKS_PATH))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                rows = cursor.fetchall()
                conn.close()
                if rows:
                    points = [dict(r) for r in rows]
                    logger.info(f"Found {len(points)} points in live_tracks.db")
            except Exception as e:
                logger.error(f"Live DB error: {e}")

        # 2. Check Research DB if not found
        if not points and DB_RESEARCH_PATH and DB_RESEARCH_PATH.exists():
            try:
                conn = sqlite3.connect(str(DB_RESEARCH_PATH))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Try anomalies_tracks
                cursor.execute("SELECT * FROM anomalies_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                rows = cursor.fetchall()
                if not rows:
                    # Try normal_tracks
                    cursor.execute("SELECT * FROM normal_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                    rows = cursor.fetchall()

                conn.close()

                if rows:
                    points = [dict(r) for r in rows]
                    logger.info(f"Found {len(points)} points in research_new.db")
            except Exception as e:
                logger.error(f"Research DB error: {e}")

        # 3. Check Feedback Tagged DB if not found
        if not points and FEEDBACK_TAGGED_DB_PATH and FEEDBACK_TAGGED_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                rows = cursor.fetchall()
                conn.close()
                if rows:
                    points = [dict(r) for r in rows]
                    logger.info(f"Found {len(points)} points in feedback_tagged.db")
            except Exception as e:
                logger.error(f"Feedback Tagged DB error: {e}")

        # 4. Check Cache DB if not found
        if not points and CACHE_DB_PATH:
            try:
                conn = sqlite3.connect(str(CACHE_DB_PATH))
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
                row = cursor.fetchone()
                conn.close()

                if row:
                    data = json.loads(row[0])
                    points = data.get("points", [])
                    logger.info(f"Found {len(points)} points in cache")
            except Exception as e:
                logger.error(f"Cache DB error: {e}")

        # If no data found in any DB, return 404
        if not points:
            raise HTTPException(status_code=404, detail=f"Flight data not found in any database for {flight_id}")
        
        # Convert points dict to TrackPoint objects
        from core.models import FlightTrack, TrackPoint
        
        track_points = []
        for p in points:
            track_points.append(TrackPoint(
                lat=p.get("lat", 0.0),
                lon=p.get("lon", 0.0),
                alt=p.get("alt", 0.0),
                timestamp=p.get("timestamp", 0),
                gspeed=p.get("gspeed", 0.0),
                vspeed=p.get("vspeed", 0.0),
                track=p.get("track", 0.0),
                squawk=p.get("squawk"),
                callsign=p.get("callsign")
            ))
        
        # Create FlightTrack object
        flight = FlightTrack(
            flight_id=flight_id,
            points=track_points
        )
        
        logger.info(f"Analyzing flight {flight_id} from DB with {len(track_points)} points")
        
        # Run Pipeline
        results = pipeline.analyze(flight)

        # Inject full points for UI visualization
        results["track"] = {
            "flight_id": flight.flight_id,
            "points": [dataclasses.asdict(p) for p in flight.points]
        }
        
        results["source"] = "database"  # Indicate data source

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis from DB failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/live/anomalies")
def get_live_anomalies(start_ts: int, end_ts: int):
    """
    Fetch anomalies from PostgreSQL live schema within a time range.
    This is populated by realtime/monitor.py.
    """
    try:
        from service.pg_provider import get_connection
        from core.airport_lookup import enrich_origin_destination
        import psycopg2.extras
        
        logger.info(f"[LIVE_ANOMALIES] Fetching from PostgreSQL live schema for range {start_ts} to {end_ts}")
        
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Join with flight_metadata to get is_military, category, aircraft_type
                query = """
                    SELECT ar.flight_id, ar.timestamp, ar.is_anomaly, ar.severity_cnn, ar.severity_dense, ar.full_report,
                           ar.callsign, ar.airline, ar.origin_airport, ar.destination_airport,
                           ar.matched_rule_names, ar.matched_rule_ids, ar.matched_rule_categories,
                           fm.is_military, fm.category, fm.aircraft_type, fm.aircraft_registration
                    FROM live.anomaly_reports ar
                    LEFT JOIN live.flight_metadata fm ON ar.flight_id = fm.flight_id
                    WHERE ar.timestamp BETWEEN %s AND %s AND ar.is_anomaly = true
                    ORDER BY ar.timestamp DESC
                """

                cursor.execute(query, (start_ts, end_ts))
                rows = cursor.fetchall()
                
                logger.info(f"[LIVE_ANOMALIES] Found {len(rows)} anomalies in PostgreSQL live schema")

                results = []
                for row in rows:
                    # Parse full report if it's a string
                    report = row["full_report"]
                    if isinstance(report, str):
                        try:
                            report = json.loads(report)
                        except:
                            pass

                    # Get callsign from row directly
                    callsign = row["callsign"]
                    
                    # Fallback: Try to get callsign from the report summary
                    if not callsign and isinstance(report, dict):
                        callsign = report.get("summary", {}).get("callsign")

                    # Inject is_military and category into full_report.summary if not present
                    is_military = bool(row["is_military"]) if row["is_military"] is not None else False
                    category = row["category"]
                    aircraft_type = row["aircraft_type"]
                    
                    if isinstance(report, dict):
                        if "summary" not in report:
                            report["summary"] = {}
                        if report["summary"].get("is_military") is None:
                            report["summary"]["is_military"] = is_military
                        if report["summary"].get("category") is None and category:
                            report["summary"]["category"] = category
                        if report["summary"].get("aircraft_type") is None and aircraft_type:
                            report["summary"]["aircraft_type"] = aircraft_type

                    # Enrich airport data with place information
                    places = enrich_origin_destination(row["origin_airport"], row["destination_airport"])

                    results.append({
                        "flight_id": row["flight_id"],
                        "timestamp": row["timestamp"],
                        "is_anomaly": bool(row["is_anomaly"]),
                        "severity_cnn": row["severity_cnn"] or 0,
                        "severity_dense": row["severity_dense"] or 0,
                        "full_report": report,
                        "callsign": callsign,
                        "airline": row["airline"],
                        "origin_airport": row["origin_airport"],
                        "destination_airport": row["destination_airport"],
                        "origin_place": places["origin_place"],
                        "destination_place": places["destination_place"],
                        "matched_rule_names": row["matched_rule_names"],
                        "matched_rule_ids": row["matched_rule_ids"],
                        "is_military": is_military,
                        "category": category,
                        "aircraft_type": aircraft_type,
                    })

                return results
                
    except ImportError as ie:
        logger.error(f"PostgreSQL provider not available: {ie}")
        raise HTTPException(status_code=500, detail="PostgreSQL provider not available")
    except Exception as e:
        logger.error(f"Failed to fetch live anomalies from PostgreSQL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/live/track/research/{flight_id}")
def get_live_research_track(flight_id: str):
    """
    Get track points for a flight from PostgreSQL (live schema).
    Searches both anomalies_tracks and normal_tracks tables.
    Also includes flight metadata (airline, aircraft_type, origin, destination, etc.)
    """
    try:
        from service.pg_provider import get_flight_track, get_flight_metadata
        
        # Get track points from live schema
        points = get_flight_track(flight_id, schema='live')
        
        if not points:
            raise HTTPException(status_code=404, detail=f"No track found for flight {flight_id}")
        
        # Add heading alias for compatibility
        for point in points:
            if 'track' in point:
                point['heading'] = point['track']
        
        # Get callsign from first point
        callsign = next((p.get('callsign') for p in points if p.get('callsign')), None)
        
        # Get flight metadata
        metadata = get_flight_metadata(flight_id, schema='live') or {}
        
        # Use metadata callsign if not found in track points
        if not callsign and metadata.get('callsign'):
            callsign = metadata['callsign']
        
        return {
            "flight_id": flight_id,
            "callsign": callsign,
            "points": points,
            # Include metadata fields at top level for easy access
            "flight_number": metadata.get("flight_number"),
            "airline": metadata.get("airline"),
            "aircraft_type": metadata.get("aircraft_type"),
            "aircraft_registration": metadata.get("aircraft_registration"),
            "origin_airport": metadata.get("origin_airport"),
            "destination_airport": metadata.get("destination_airport"),
            "category": metadata.get("category"),
            "first_seen_ts": metadata.get("first_seen_ts"),
            "last_seen_ts": metadata.get("last_seen_ts"),
            "flight_duration_sec": metadata.get("flight_duration_sec"),
            "total_distance_nm": metadata.get("total_distance_nm"),
            "min_altitude_ft": metadata.get("min_altitude_ft"),
            "max_altitude_ft": metadata.get("max_altitude_ft"),
            "avg_altitude_ft": metadata.get("avg_altitude_ft"),
            "avg_speed_kts": metadata.get("avg_speed_kts"),
            "is_anomaly": metadata.get("is_anomaly"),
            "is_military": metadata.get("is_military"),
            "scheduled_departure": metadata.get("scheduled_departure"),
            "scheduled_arrival": metadata.get("scheduled_arrival"),
        }

    except HTTPException:
        raise
    except ImportError as ie:
        logger.error(f"PostgreSQL provider not available: {ie}")
        raise HTTPException(status_code=500, detail="PostgreSQL provider not available")
    except Exception as e:
        logger.error(f"Failed to fetch live research track: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/live/flights")
def get_all_live_flights():
    """
    Get all currently active flights from PostgreSQL (live schema).
    Returns all flights with their last known position and anomaly status.
    """
    try:
        from service.pg_provider import get_all_live_flights as pg_get_all_live_flights
        
        result = pg_get_all_live_flights(cutoff_minutes=15)
        logger.debug(f"Live flights - Found {result.get('total_count', 0)} flights in PostgreSQL")
        return result
        
    except ImportError as ie:
        logger.error(f"PostgreSQL provider not available: {ie}")
        raise HTTPException(status_code=500, detail="PostgreSQL provider not available")
    except Exception as e:
        logger.error(f"Error fetching live flights from PostgreSQL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch live flights: {str(e)}")


@router.get("/api/live/anomalies/since")
def get_live_anomalies_since(ts: int):
    """
    Get anomalies detected since a specific timestamp from PostgreSQL live schema.
    Used by frontend to detect new anomalies for alert sounds.
    """
    try:
        from service.pg_provider import get_connection
        from core.airport_lookup import enrich_origin_destination
        import psycopg2.extras
        
        logger.info(f"[LIVE_ANOMALIES_SINCE] Fetching from PostgreSQL live schema since {ts}")
        
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        ar.flight_id, ar.timestamp, ar.is_anomaly, ar.severity_cnn, ar.severity_dense,
                        ar.callsign, ar.airline, ar.origin_airport, ar.destination_airport,
                        ar.matched_rule_names, ar.matched_rule_ids
                    FROM live.anomaly_reports ar
                    WHERE ar.timestamp > %s AND ar.is_anomaly = true
                    ORDER BY ar.timestamp DESC
                """, (ts,))
                
                rows = cursor.fetchall()
                
                logger.info(f"[LIVE_ANOMALIES_SINCE] Found {len(rows)} new anomalies in PostgreSQL live schema")
                
                anomalies = []
                for row in rows:
                    # Enrich airport data with place information
                    places = enrich_origin_destination(row["origin_airport"], row["destination_airport"])
                    
                    anomalies.append({
                        "flight_id": row["flight_id"],
                        "timestamp": row["timestamp"],
                        "is_anomaly": bool(row["is_anomaly"]),
                        "severity_cnn": row["severity_cnn"] or 0,
                        "severity_dense": row["severity_dense"] or 0,
                        "callsign": row["callsign"],
                        "airline": row["airline"],
                        "origin_airport": row["origin_airport"],
                        "destination_airport": row["destination_airport"],
                        "origin_place": places["origin_place"],
                        "destination_place": places["destination_place"],
                        "matched_rule_names": row["matched_rule_names"],
                        "matched_rule_ids": row["matched_rule_ids"],
                    })
                
                return {
                    "anomalies": anomalies,
                    "count": len(anomalies),
                }
                
    except ImportError as ie:
        logger.error(f"PostgreSQL provider not available: {ie}")
        return {"anomalies": [], "count": 0, "error": "PostgreSQL provider not available"}
    except Exception as e:
        logger.error(f"Failed to fetch live anomalies since {ts} from PostgreSQL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/research/anomalies")
def get_research_anomalies(start_ts: int, end_ts: int):
    print(f"Getting research anomalies from {start_ts} to {end_ts}")
    """
    Fetch anomalies from the research database within a time range.
    Now uses PostgreSQL research schema instead of SQLite.
    """
    try:
        # Import PostgreSQL provider
        from service.pg_provider import get_research_anomalies as pg_get_research_anomalies
        
        # Fetch from PostgreSQL research schema
        rows = pg_get_research_anomalies(start_ts, end_ts, limit=1000)
        
        # Transform to match expected frontend format
        results = []
        for row in rows:
            # full_report is already parsed by pg_provider
            full_report = row.get('full_report')
            
            # Extract callsign and flight_number from row or report
            callsign = row.get('callsign')
            flight_number = row.get('flight_number')
            
            if not callsign and isinstance(full_report, dict):
                callsign = full_report.get("summary", {}).get("callsign")
            
            results.append({
                "flight_id": row.get('flight_id'),
                "timestamp": row.get('timestamp'),
                "is_anomaly": bool(row.get('is_anomaly', True)),
                "severity_cnn": row.get('severity_cnn'),
                "severity_dense": row.get('severity_dense'),
                "full_report": full_report,
                "callsign": callsign,
                "flight_number": flight_number,
                # Include additional metadata from PostgreSQL
                "airline": row.get('airline'),
                "origin_airport": row.get('origin_airport'),
                "destination_airport": row.get('destination_airport'),
                "aircraft_type": row.get('aircraft_type'),
                "total_points": row.get('total_points')
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to fetch research anomalies from PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/flights")
def get_dashboard_flights(start_ts: int, end_ts: int):
    """
    Get all flights for the dashboard:
    1. Normal flights from research_new.db (normal_tracks with metadata)
    2. All tagged flights from feedback_tagged.db (both anomalies and normal based on user_label)
    
    Returns a combined list with source information.
    """
    results = []
    seen_flight_ids = set()
    
    # 1. Get tagged flights from feedback_tagged.db (both anomalies AND normal)
    if FEEDBACK_TAGGED_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
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
                    fm.airline,
                    fm.origin_airport,
                    fm.destination_airport,
                    fm.aircraft_type,
                    fm.is_military,
                    fm.total_points,
                    fm.flight_duration_sec,
                    fm.max_altitude_ft,
                    fm.avg_speed_kts,
                    ar.full_report,
                    ar.severity_cnn,
                    ar.severity_dense,
                    ar.is_anomaly as report_is_anomaly,
                    ar.matched_rule_ids,
                    ar.matched_rule_names,
                    (SELECT COUNT(*) FROM flight_tracks ft WHERE ft.flight_id = uf.flight_id) as track_count
                FROM user_feedback uf
                LEFT JOIN flight_metadata fm ON uf.flight_id = fm.flight_id
                LEFT JOIN anomaly_reports ar ON uf.flight_id = ar.flight_id
                WHERE COALESCE(uf.first_seen_ts, uf.tagged_at) BETWEEN ? AND ?
                ORDER BY COALESCE(uf.first_seen_ts, uf.tagged_at) DESC
            """, (start_ts, end_ts))
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                flight_id = row['flight_id']
                seen_flight_ids.add(flight_id)
                
                full_report = row['full_report']
                if isinstance(full_report, (str, bytes)):
                    try:
                        full_report = json.loads(full_report)
                    except:
                        pass
                
                # Determine if it's an anomaly based on user_label (1 = anomaly, 0 = normal)
                is_anomaly = bool(row['user_label'])
                
                results.append({
                    'flight_id': flight_id,
                    'timestamp': row['timestamp'],
                    'first_seen_ts': row['first_seen_ts'],
                    'last_seen_ts': row['last_seen_ts'],
                    'tagged_at': row['tagged_at'],
                    'is_anomaly': is_anomaly,
                    'user_label': row['user_label'],
                    'rule_id': row['rule_id'],
                    'rule_name': row['rule_name'],
                    'comments': row['comments'],
                    'other_details': row['other_details'],
                    'callsign': row['callsign'],
                    'airline': row['airline'],
                    'origin_airport': row['origin_airport'],
                    'destination_airport': row['destination_airport'],
                    'aircraft_type': row['aircraft_type'],
                    'is_military': row['is_military'],
                    'total_points': row['total_points'] or row['track_count'],
                    'flight_duration_sec': row['flight_duration_sec'],
                    'max_altitude_ft': row['max_altitude_ft'],
                    'avg_speed_kts': row['avg_speed_kts'],
                    'full_report': full_report,
                    'severity_cnn': row['severity_cnn'],
                    'severity_dense': row['severity_dense'],
                    'matched_rule_ids': row['matched_rule_ids'],
                    'matched_rule_names': row['matched_rule_names'],
                    'source': 'feedback_tagged',
                    'tagged': True
                })
            
            logger.info(f"[DASHBOARD] Found {len(results)} tagged flights from feedback_tagged.db")
        except Exception as e:
            logger.error(f"[DASHBOARD] Error fetching from feedback_tagged.db: {e}")
    
    # 2. Get normal flights from research_new.db (that aren't already in feedback_tagged)
    if DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get unique flight_ids from normal_tracks within time range
            cursor.execute("""
                SELECT DISTINCT flight_id, MIN(timestamp) as first_ts, MAX(timestamp) as last_ts
                FROM normal_tracks
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY flight_id
            """, (start_ts, end_ts))
            
            normal_flights = cursor.fetchall()
            
            # Get metadata for these flights
            for nf in normal_flights:
                flight_id = nf['flight_id']
                
                # Skip if already in feedback_tagged
                if flight_id in seen_flight_ids:
                    continue
                
                seen_flight_ids.add(flight_id)
                
                # Get metadata
                cursor.execute("SELECT * FROM flight_metadata WHERE flight_id = ?", (flight_id,))
                meta_row = cursor.fetchone()
                
                # Get callsign from tracks if not in metadata
                callsign = None
                if meta_row:
                    callsign = meta_row['callsign']
                
                if not callsign:
                    cursor.execute(
                        "SELECT callsign FROM normal_tracks WHERE flight_id = ? AND callsign IS NOT NULL LIMIT 1",
                        (flight_id,)
                    )
                    cs_row = cursor.fetchone()
                    if cs_row:
                        callsign = cs_row['callsign']
                
                # Get track count
                cursor.execute("SELECT COUNT(*) as cnt FROM normal_tracks WHERE flight_id = ?", (flight_id,))
                track_count = cursor.fetchone()['cnt']
                
                results.append({
                    'flight_id': flight_id,
                    'timestamp': nf['first_ts'],
                    'first_seen_ts': nf['first_ts'],
                    'last_seen_ts': nf['last_ts'],
                    'tagged_at': None,
                    'is_anomaly': False,  # Normal flights
                    'user_label': 0,
                    'rule_id': None,
                    'rule_name': None,
                    'comments': None,
                    'other_details': None,
                    'callsign': callsign,
                    'airline': meta_row['airline'] if meta_row else None,
                    'origin_airport': meta_row['origin_airport'] if meta_row else None,
                    'destination_airport': meta_row['destination_airport'] if meta_row else None,
                    'aircraft_type': meta_row['aircraft_type'] if meta_row else None,
                    'is_military': meta_row['is_military'] if meta_row else False,
                    'total_points': track_count,
                    'flight_duration_sec': meta_row['flight_duration_sec'] if meta_row else None,
                    'max_altitude_ft': meta_row['max_altitude_ft'] if meta_row else None,
                    'avg_speed_kts': meta_row['avg_speed_kts'] if meta_row else None,
                    'full_report': None,
                    'severity_cnn': None,
                    'severity_dense': None,
                    'matched_rule_ids': None,
                    'matched_rule_names': None,
                    'source': 'research_normal',
                    'tagged': False
                })
            
            conn.close()
            logger.info(f"[DASHBOARD] Added {len(results) - len([r for r in results if r['source'] == 'feedback_tagged'])} normal flights from research_new.db")
        except Exception as e:
            logger.error(f"[DASHBOARD] Error fetching from research_new.db: {e}")
    
    # Sort by timestamp descending
    results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    
    logger.info(f"[DASHBOARD] Total flights returned: {len(results)}")
    return results


@router.get("/api/research/track/{flight_id}")
def get_research_track(flight_id: str):
    """
    Fetch the full track for a flight from Research schema (PostgreSQL).
    Checks both anomalies_tracks and normal_tracks.
    """
    try:
        from service.pg_provider import get_flight_track
        
        # Fetch track from PostgreSQL research schema
        points = get_flight_track(flight_id, schema='research')
        
        if not points:
            raise HTTPException(status_code=404, detail="Track not found in Research schema")
        
        return {
            "flight_id": flight_id,
            "points": points
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Research track fetch error from PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/research/callsign/{flight_id}")
def get_research_callsign(flight_id: str):
    """
    Fetch a callsign for a research flight ID.
    Tries anomalies_tracks -> normal_tracks -> anomaly_reports summary.
    """
    if not DB_RESEARCH_PATH.exists():
        raise HTTPException(status_code=404, detail="Research DB not found")

    conn = None
    try:
        conn = sqlite3.connect(str(DB_RESEARCH_PATH))
        cursor = conn.cursor()

        callsign = None

        # Try anomalies_tracks first
        try:
            cursor.execute(
                "SELECT callsign FROM anomalies_tracks WHERE flight_id = ? AND callsign IS NOT NULL AND callsign != '' LIMIT 1",
                (flight_id,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                callsign = row[0]
        except Exception:
            pass

        # Fallback to normal_tracks
        if not callsign:
            try:
                cursor.execute(
                    "SELECT callsign FROM normal_tracks WHERE flight_id = ? AND callsign IS NOT NULL AND callsign != '' LIMIT 1",
                    (flight_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    callsign = row[0]
            except Exception:
                pass

        # Final fallback: summary in anomaly_reports
        if not callsign:
            try:
                cursor.execute(
                    "SELECT full_report FROM anomaly_reports WHERE flight_id = ? LIMIT 1",
                    (flight_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    report = row[0]
                    if isinstance(report, str):
                        try:
                            report = json.loads(report)
                        except Exception:
                            report = None
                    if isinstance(report, dict):
                        callsign = report.get("summary", {}).get("callsign")
            except Exception:
                pass

        return {"callsign": callsign}
    except Exception as e:
        logger.error(f"Failed to fetch research callsign for {flight_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@router.get("/api/research/metadata/{flight_id}")
def get_research_flight_metadata(flight_id: str):
    """
    Get flight metadata for a research flight from PostgreSQL research schema.
    Extracts data from flight_metadata table, anomaly_reports, and track tables.
    """
    try:
        from service.pg_provider import get_flight_metadata, get_flight_track, get_connection
        import psycopg2.extras
        
        metadata = {"flight_id": flight_id}
        
        # 1. Get data from flight_metadata table (PostgreSQL research schema)
        fm_data = get_flight_metadata(flight_id, schema='research')
        
        if fm_data:
            # Copy all metadata fields
            metadata.update(fm_data)
            # Ensure booleans are properly converted
            if 'emergency_squawk_detected' in metadata:
                metadata['emergency_squawk_detected'] = bool(metadata['emergency_squawk_detected'])
            if 'is_military' in metadata:
                metadata['is_military'] = bool(metadata['is_military'])
        
        # 2. Get data from anomaly_reports (for anomaly-specific fields)
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT timestamp, is_anomaly, severity_cnn, severity_dense, full_report 
                    FROM research.anomaly_reports 
                    WHERE flight_id = %s 
                    LIMIT 1
                    """,
                    (flight_id,)
                )
                report_row = cursor.fetchone()
                
                full_report = None
                if report_row:
                    metadata["anomaly_timestamp"] = report_row["timestamp"]
                    metadata["is_anomaly"] = bool(report_row["is_anomaly"])
                    metadata["severity_cnn"] = report_row["severity_cnn"]
                    metadata["severity_dense"] = report_row["severity_dense"]
                    
                    # Parse full_report JSON for any missing fields
                    raw_report = report_row["full_report"]
                    if raw_report:
                        if isinstance(raw_report, str):
                            try:
                                full_report = json.loads(raw_report)
                            except:
                                full_report = None
                        else:
                            full_report = raw_report
                
                # Fill in any missing fields from full_report summary
                if full_report and isinstance(full_report, dict):
                    summary = full_report.get("summary", {})
                    # Only fill if not already set from flight_metadata
                    if not metadata.get("callsign"):
                        metadata["callsign"] = summary.get("callsign")
                    if not metadata.get("flight_number"):
                        metadata["flight_number"] = summary.get("flight_number")
                    if not metadata.get("airline"):
                        metadata["airline"] = summary.get("airline")
                    if not metadata.get("airline_code"):
                        metadata["airline_code"] = summary.get("airline_code")
                    if not metadata.get("aircraft_type"):
                        metadata["aircraft_type"] = summary.get("aircraft_type")
                    if not metadata.get("aircraft_model"):
                        metadata["aircraft_model"] = summary.get("aircraft_model")
                    if not metadata.get("aircraft_registration"):
                        metadata["aircraft_registration"] = summary.get("aircraft_registration")
                    if not metadata.get("origin_airport"):
                        metadata["origin_airport"] = summary.get("origin") or summary.get("origin_airport")
                    if not metadata.get("destination_airport"):
                        metadata["destination_airport"] = summary.get("destination") or summary.get("destination_airport")
        
        # 3. Get track data to compute/fill missing stats (only if not already from flight_metadata)
        if not fm_data:
            points = get_flight_track(flight_id, schema='research')
            
            if points:
                # Extract callsign if not already set
                if not metadata.get("callsign"):
                    for p in points:
                        if p.get("callsign"):
                            metadata["callsign"] = p["callsign"]
                            break
                
                # Compute timestamps
                if not metadata.get("first_seen_ts"):
                    metadata["first_seen_ts"] = points[0].get("timestamp")
                if not metadata.get("last_seen_ts"):
                    metadata["last_seen_ts"] = points[-1].get("timestamp")
                
                # Compute position data
                if not metadata.get("start_lat"):
                    metadata["start_lat"] = points[0].get("lat")
                if not metadata.get("start_lon"):
                    metadata["start_lon"] = points[0].get("lon")
                if not metadata.get("end_lat"):
                    metadata["end_lat"] = points[-1].get("lat")
                if not metadata.get("end_lon"):
                    metadata["end_lon"] = points[-1].get("lon")
                
                # Compute flight stats if not available
                if not metadata.get("total_points"):
                    metadata["total_points"] = len(points)
                
                if not metadata.get("flight_duration_sec") and metadata.get("first_seen_ts") and metadata.get("last_seen_ts"):
                    metadata["flight_duration_sec"] = metadata["last_seen_ts"] - metadata["first_seen_ts"]
                
                # Compute altitude stats if not available
                alts = [p.get("alt") for p in points if p.get("alt") is not None and p.get("alt") > 0]
                if alts:
                    if not metadata.get("min_altitude_ft"):
                        metadata["min_altitude_ft"] = min(alts)
                    if not metadata.get("max_altitude_ft"):
                        metadata["max_altitude_ft"] = max(alts)
                    if not metadata.get("avg_altitude_ft"):
                        metadata["avg_altitude_ft"] = sum(alts) / len(alts)
                
                # Compute speed stats if not available
                speeds = [p.get("gspeed") for p in points if p.get("gspeed") is not None and p.get("gspeed") > 0]
                if speeds:
                    if not metadata.get("min_speed_kts"):
                        metadata["min_speed_kts"] = min(speeds)
                    if not metadata.get("max_speed_kts"):
                        metadata["max_speed_kts"] = max(speeds)
                    if not metadata.get("avg_speed_kts"):
                        metadata["avg_speed_kts"] = sum(speeds) / len(speeds)
                
                # Collect squawk codes if not available
                if not metadata.get("squawk_codes"):
                    squawks = set()
                    for p in points:
                        sq = p.get("squawk")
                        if sq and sq != "0000":
                            squawks.add(str(sq))
                    if squawks:
                        metadata["squawk_codes"] = ",".join(sorted(squawks))
                        # Check for emergency squawks
                        emergency_codes = {"7500", "7600", "7700"}
                        metadata["emergency_squawk_detected"] = bool(squawks & emergency_codes)
        
        # If no data found at all, return 404
        if not fm_data and not report_row and (not 'first_seen_ts' in metadata):
            raise HTTPException(status_code=404, detail="Flight not found in Research schema")
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch research metadata from PostgreSQL for {flight_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RESEARCH RERUN ROUTES (using research_old schema)
# ============================================================================

@research_rerun_router.get("/api/research_rerun/anomalies")
def get_research_rerun_anomalies(start_ts: int, end_ts: int):
    """
    Fetch anomalies from the research_old database (rerun analysis) within a time range.
    Uses PostgreSQL research_old schema.
    """
    try:
        # Import PostgreSQL provider
        from service.pg_provider import get_research_anomalies
        
        # Fetch from PostgreSQL research_old schema
        rows = get_research_anomalies(start_ts, end_ts, limit=1000, schema='research_old')
        
        # Transform to match expected frontend format
        results = []
        for row in rows:
            # full_report is already parsed by pg_provider
            full_report = row.get('full_report')
            
            # Extract callsign and flight_number from row or report
            callsign = row.get('callsign')
            flight_number = row.get('flight_number')
            
            if not callsign and isinstance(full_report, dict):
                callsign = full_report.get("summary", {}).get("callsign")
            
            results.append({
                "flight_id": row.get('flight_id'),
                "timestamp": row.get('timestamp'),
                "is_anomaly": bool(row.get('is_anomaly', True)),
                "severity_cnn": row.get('severity_cnn'),
                "severity_dense": row.get('severity_dense'),
                "full_report": full_report,
                "callsign": callsign,
                "flight_number": flight_number,
                # Include additional metadata from PostgreSQL
                "airline": row.get('airline'),
                "origin_airport": row.get('origin_airport'),
                "destination_airport": row.get('destination_airport'),
                "aircraft_type": row.get('aircraft_type'),
                "total_points": row.get('total_points')
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to fetch research_rerun anomalies from PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@research_rerun_router.get("/api/research_rerun/track/{flight_id}")
def get_research_rerun_track(flight_id: str):
    """
    Fetch the full track for a flight from research_old schema (PostgreSQL).
    Checks both anomalies_tracks and normal_tracks in the research_old schema.
    """
    try:
        from service.pg_provider import get_flight_track
        
        # Fetch track from PostgreSQL research_old schema
        points = get_flight_track(flight_id, schema='research_old')
        
        if not points:
            raise HTTPException(status_code=404, detail="Track not found in research_old schema")
        
        return {
            "flight_id": flight_id,
            "points": points
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Research rerun track fetch error from PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@research_rerun_router.get("/api/research_rerun/metadata/{flight_id}")
def get_research_rerun_flight_metadata(flight_id: str):
    """
    Get flight metadata for a research_rerun flight from PostgreSQL research_old schema.
    Extracts data from flight_metadata table, anomaly_reports, and track tables.
    """
    try:
        from service.pg_provider import get_flight_metadata, get_flight_track, get_connection
        import psycopg2.extras
        
        metadata = {"flight_id": flight_id}
        
        # 1. Get data from flight_metadata table (PostgreSQL research_old schema)
        fm_data = get_flight_metadata(flight_id, schema='research_old')
        
        if fm_data:
            # Copy all metadata fields
            metadata.update(fm_data)
            # Ensure booleans are properly converted
            if 'emergency_squawk_detected' in metadata:
                metadata['emergency_squawk_detected'] = bool(metadata['emergency_squawk_detected'])
            if 'is_military' in metadata:
                metadata['is_military'] = bool(metadata['is_military'])
        
        # 2. Get data from anomaly_reports (for anomaly-specific fields)
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT timestamp, is_anomaly, severity_cnn, severity_dense, full_report 
                    FROM research_old.anomaly_reports 
                    WHERE flight_id = %s 
                    LIMIT 1
                    """,
                    (flight_id,)
                )
                report_row = cursor.fetchone()
                
                full_report = None
                if report_row:
                    metadata["anomaly_timestamp"] = report_row["timestamp"]
                    metadata["is_anomaly"] = bool(report_row["is_anomaly"])
                    metadata["severity_cnn"] = report_row["severity_cnn"]
                    metadata["severity_dense"] = report_row["severity_dense"]
                    
                    # Parse full_report JSON for any missing fields
                    raw_report = report_row["full_report"]
                    if raw_report:
                        if isinstance(raw_report, str):
                            try:
                                full_report = json.loads(raw_report)
                            except:
                                full_report = None
                        else:
                            full_report = raw_report
                
                # Fill in any missing fields from full_report summary
                if full_report and isinstance(full_report, dict):
                    summary = full_report.get("summary", {})
                    # Only fill if not already set from flight_metadata
                    if not metadata.get("callsign"):
                        metadata["callsign"] = summary.get("callsign")
                    if not metadata.get("flight_number"):
                        metadata["flight_number"] = summary.get("flight_number")
                    if not metadata.get("airline"):
                        metadata["airline"] = summary.get("airline")
                    if not metadata.get("airline_code"):
                        metadata["airline_code"] = summary.get("airline_code")
                    if not metadata.get("aircraft_type"):
                        metadata["aircraft_type"] = summary.get("aircraft_type")
                    if not metadata.get("aircraft_model"):
                        metadata["aircraft_model"] = summary.get("aircraft_model")
                    if not metadata.get("aircraft_registration"):
                        metadata["aircraft_registration"] = summary.get("aircraft_registration")
                    if not metadata.get("origin_airport"):
                        metadata["origin_airport"] = summary.get("origin") or summary.get("origin_airport")
                    if not metadata.get("destination_airport"):
                        metadata["destination_airport"] = summary.get("destination") or summary.get("destination_airport")
        
        # 3. Get track data to compute/fill missing stats (only if not already from flight_metadata)
        if not fm_data:
            points = get_flight_track(flight_id, schema='research_old')
            
            if points:
                # Extract callsign if not already set
                if not metadata.get("callsign"):
                    for p in points:
                        if p.get("callsign"):
                            metadata["callsign"] = p["callsign"]
                            break
                
                # Compute timestamps
                if not metadata.get("first_seen_ts"):
                    metadata["first_seen_ts"] = points[0].get("timestamp")
                if not metadata.get("last_seen_ts"):
                    metadata["last_seen_ts"] = points[-1].get("timestamp")
                
                # Compute position data
                if not metadata.get("start_lat"):
                    metadata["start_lat"] = points[0].get("lat")
                if not metadata.get("start_lon"):
                    metadata["start_lon"] = points[0].get("lon")
                if not metadata.get("end_lat"):
                    metadata["end_lat"] = points[-1].get("lat")
                if not metadata.get("end_lon"):
                    metadata["end_lon"] = points[-1].get("lon")
                
                # Compute flight stats if not available
                if not metadata.get("total_points"):
                    metadata["total_points"] = len(points)
                
                if not metadata.get("flight_duration_sec") and metadata.get("first_seen_ts") and metadata.get("last_seen_ts"):
                    metadata["flight_duration_sec"] = metadata["last_seen_ts"] - metadata["first_seen_ts"]
                
                # Compute altitude stats if not available
                alts = [p.get("alt") for p in points if p.get("alt") is not None and p.get("alt") > 0]
                if alts:
                    if not metadata.get("min_altitude_ft"):
                        metadata["min_altitude_ft"] = min(alts)
                    if not metadata.get("max_altitude_ft"):
                        metadata["max_altitude_ft"] = max(alts)
                    if not metadata.get("avg_altitude_ft"):
                        metadata["avg_altitude_ft"] = sum(alts) / len(alts)
                
                # Compute speed stats if not available
                speeds = [p.get("gspeed") for p in points if p.get("gspeed") is not None and p.get("gspeed") > 0]
                if speeds:
                    if not metadata.get("min_speed_kts"):
                        metadata["min_speed_kts"] = min(speeds)
                    if not metadata.get("max_speed_kts"):
                        metadata["max_speed_kts"] = max(speeds)
                    if not metadata.get("avg_speed_kts"):
                        metadata["avg_speed_kts"] = sum(speeds) / len(speeds)
                
                # Collect squawk codes if not available
                if not metadata.get("squawk_codes"):
                    squawks = set()
                    for p in points:
                        sq = p.get("squawk")
                        if sq and sq != "0000":
                            squawks.add(str(sq))
                    if squawks:
                        metadata["squawk_codes"] = ",".join(sorted(squawks))
                        # Check for emergency squawks
                        emergency_codes = {"7500", "7600", "7700"}
                        metadata["emergency_squawk_detected"] = bool(squawks & emergency_codes)
        
        # If no data found at all, return 404
        if not fm_data and not report_row and (not 'first_seen_ts' in metadata):
            raise HTTPException(status_code=404, detail="Flight not found in research_old schema")
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch research_rerun metadata from PostgreSQL for {flight_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@research_rerun_router.get("/api/research_rerun/callsign/{flight_id}")
def get_research_rerun_callsign(flight_id: str):
    """
    Fetch a callsign for a research_rerun flight ID from research_old schema.
    Tries anomalies_tracks -> normal_tracks -> anomaly_reports summary.
    """
    try:
        from service.pg_provider import get_connection
        import psycopg2.extras
        
        callsign = None
        
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Try anomalies_tracks first
                try:
                    cursor.execute(
                        """
                        SELECT callsign 
                        FROM research_old.anomalies_tracks 
                        WHERE flight_id = %s 
                          AND callsign IS NOT NULL 
                          AND callsign != '' 
                        LIMIT 1
                        """,
                        (flight_id,)
                    )
                    row = cursor.fetchone()
                    if row and row['callsign']:
                        callsign = row['callsign']
                except Exception:
                    pass
                
                # Fallback to normal_tracks
                if not callsign:
                    try:
                        cursor.execute(
                            """
                            SELECT callsign 
                            FROM research_old.normal_tracks 
                            WHERE flight_id = %s 
                              AND callsign IS NOT NULL 
                              AND callsign != '' 
                            LIMIT 1
                            """,
                            (flight_id,)
                        )
                        row = cursor.fetchone()
                        if row and row['callsign']:
                            callsign = row['callsign']
                    except Exception:
                        pass
                
                # Final fallback: summary in anomaly_reports
                if not callsign:
                    try:
                        cursor.execute(
                            """
                            SELECT full_report 
                            FROM research_old.anomaly_reports 
                            WHERE flight_id = %s 
                            LIMIT 1
                            """,
                            (flight_id,)
                        )
                        row = cursor.fetchone()
                        if row and row['full_report']:
                            report = row['full_report']
                            if isinstance(report, str):
                                try:
                                    report = json.loads(report)
                                except Exception:
                                    report = None
                            if isinstance(report, dict):
                                callsign = report.get("summary", {}).get("callsign")
                    except Exception:
                        pass
        
        return {"callsign": callsign}
        
    except Exception as e:
        logger.error(f"Failed to fetch research_rerun callsign for {flight_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/api/research/anomaly/{flight_id}")
def get_research_anomaly_by_id(flight_id: str):
    """
    Fetch a single flight's anomaly report from the research database (PostgreSQL) by flight_id.
    Returns the same structure as research/anomalies endpoint.
    """
    try:
        from service.pg_provider import get_connection, get_flight_metadata
        import psycopg2.extras
        
        # Fetch from PostgreSQL research schema
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Fetch the anomaly report from research schema
                cursor.execute("""
                    SELECT flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report 
                    FROM research.anomaly_reports 
                    WHERE flight_id = %s
                    LIMIT 1
                """, (flight_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    raise HTTPException(status_code=404, detail=f"Anomaly report not found for flight {flight_id}")
        
        # Get metadata (callsign, flight_number) from flight_metadata
        metadata = get_flight_metadata(flight_id, schema='research')
        callsign = metadata.get('callsign') if metadata else None
        flight_number = metadata.get('flight_number') if metadata else None
        
        # If not in metadata, try getting from track data
        if not callsign:
            with get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    # Try anomalies_tracks
                    cursor.execute(
                        "SELECT callsign FROM research.anomalies_tracks WHERE flight_id = %s AND callsign IS NOT NULL LIMIT 1",
                        (flight_id,))
                    result = cursor.fetchone()
                    if result:
                        callsign = result['callsign']
                    
                    # Try normal_tracks if not found
                    if not callsign:
                        cursor.execute(
                            "SELECT callsign FROM research.normal_tracks WHERE flight_id = %s AND callsign IS NOT NULL LIMIT 1",
                            (flight_id,))
                        result = cursor.fetchone()
                        if result:
                            callsign = result['callsign']
        
        # Parse the full_report
        report = row['full_report']
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except:
                pass
        
        # Extract callsign from report if not found in tables
        if not callsign and isinstance(report, dict):
            callsign = report.get("summary", {}).get("callsign")
        
        return {
            "flight_id": row['flight_id'],
            "timestamp": row['timestamp'],
            "is_anomaly": bool(row['is_anomaly']),
            "severity_cnn": row['severity_cnn'],
            "severity_dense": row['severity_dense'],
            "full_report": report,
            "callsign": callsign,
            "flight_number": flight_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch anomaly report for {flight_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/api/data/flights")
def get_data_flights(start_ts: int, end_ts: int):
    """
    Get a list of flights within a time range from all available databases.
    Aggregates results from Live DB and Research DB.
    """
    results = {}  # flight_id -> dict

    def process_rows(rows, source_name):
        for r in rows:
            fid = r[0]
            start_t = r[1]
            end_t = r[2]
            cs = r[3]
            count = r[4]

            if fid not in results:
                results[fid] = {
                    "flight_id": fid,
                    "callsign": cs,
                    "start_time": start_t,
                    "end_time": end_t,
                    "point_count": count,
                    "source": source_name
                }
            else:
                # Merge info
                curr = results[fid]
                curr["start_time"] = min(curr["start_time"], start_t) if curr["start_time"] else start_t
                curr["end_time"] = max(curr["end_time"], end_t) if curr["end_time"] else end_t
                curr["point_count"] += count
                if "research" in source_name:
                    curr["source"] = source_name

                if not curr["callsign"] and cs:
                    curr["callsign"] = cs

    # 1. Live Tracks
    if DB_TRACKS_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_TRACKS_PATH))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT flight_id, MIN(timestamp), MAX(timestamp), MAX(callsign), COUNT(*)
                FROM flight_tracks
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY flight_id
            """, (start_ts, end_ts))
            process_rows(cursor.fetchall(), "live")
            conn.close()
        except Exception as e:
            logger.error(f"Error querying live tracks: {e}")

    # 2. Research DB
    if DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            cursor = conn.cursor()

            # Normal Tracks
            try:
                cursor.execute("""
                    SELECT flight_id, MIN(timestamp), MAX(timestamp), MAX(callsign), COUNT(*)
                    FROM normal_tracks
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY flight_id
                """, (start_ts, end_ts))
                process_rows(cursor.fetchall(), "research_normal")
            except Exception as e:
                logger.warning(f"Error querying normal_tracks: {e}")

            # Anomalies Tracks
            try:
                cursor.execute("""
                    SELECT flight_id, MIN(timestamp), MAX(timestamp), MAX(callsign), COUNT(*)
                    FROM anomalies_tracks
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY flight_id
                """, (start_ts, end_ts))
                process_rows(cursor.fetchall(), "research_anomaly")
            except Exception as e:
                logger.warning(f"Error querying anomalies_tracks: {e}")

            conn.close()
        except Exception as e:
            logger.error(f"Error querying research tracks: {e}")

    return list(results.values())


@router.get("/api/paths")
def get_learned_paths():
    path_file = Path("rules/learned_paths.json")
    if not path_file.exists():
        return {"layers": {"strict": [], "loose": []}}
    with path_file.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/api/learned-layers")
def get_learned_layers(origin: Optional[str] = None, destination: Optional[str] = None):
    """
    Return all learned layers: paths, turns, SIDs, STARs, and tubes.
    Paths, tubes, SIDs, and STARs are loaded from PostgreSQL (10-20x faster than MongoDB!).
    Turns are still loaded from JSON files.
    
    Query parameters:
    - origin: Filter tubes and paths by origin airport code (e.g., "LLBG")
    - destination: Filter tubes and paths by destination airport code (e.g., "LLSD")
    """
    from service.learned_data_provider import get_all_learned_layers
    
    result = {"paths": [], "turns": [], "sids": [], "stars": [], "tubes": []}
    rules_dir = Path("rules")
    
    # Determine minimum member count based on HECA routes
    min_path_members = 11 if (origin == "HECA" or destination == "HECA") else 7
    min_tube_members = 11 if (origin == "HECA" or destination == "HECA") else 6
    
    # Load paths, tubes, SIDs, and STARs from PostgreSQL (FAST!)
    try:
        pg_layers = get_all_learned_layers(
            origin=origin,
            destination=destination,
            min_path_members=min_path_members,
            min_tube_members=min_tube_members,
            min_procedure_members=3
        )
        
        # Filter out Unknown origins/destinations
        valid_paths = [
            p for p in pg_layers.get("paths", [])
            if p.get("origin") != "Unknown" and p.get("destination") != "Unknown"
        ]
        
        valid_tubes = [
            t for t in pg_layers.get("tubes", [])
            if t.get("origin") != "Unknown" and t.get("destination") != "Unknown"
        ]
        
        result["paths"] = valid_paths
        result["tubes"] = valid_tubes
        result["sids"] = pg_layers.get("sids", [])
        result["stars"] = pg_layers.get("stars", [])
        
        logger.info(f"Loaded from PostgreSQL: {len(valid_paths)} paths, {len(valid_tubes)} tubes, "
                   f"{len(result['sids'])} SIDs, {len(result['stars'])} STARs")
        
    except Exception as e:
        logger.error(f"Error loading from PostgreSQL: {e}", exc_info=True)
        # Fallback to JSON files if PostgreSQL fails
        logger.warning("Falling back to JSON files...")
        
        # Fallback: Load paths from JSON
        paths_file = rules_dir / "learned_paths.json"
        if paths_file.exists():
            try:
                with paths_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    paths = data.get("paths", [])
                    
                    valid_paths = []
                    for path in paths:
                        path_origin = path.get("origin", "Unknown")
                        path_dest = path.get("destination", "Unknown")
                        
                        if path_origin == "Unknown" or path_dest == "Unknown":
                            continue
                        
                        min_members = 11 if (path_origin == "HECA" or path_dest == "HECA") else 6
                        if path.get("member_count", 0) <= min_members:
                            continue
                        
                        # Apply origin/destination filter
                        if origin and path_origin != origin:
                            continue
                        if destination and path_dest != destination:
                            continue
                        
                        valid_paths.append(path)
                    
                    result["paths"] = valid_paths
            except Exception as e2:
                logger.error(f"Error loading paths from JSON: {e2}")
        
        # Fallback: Load tubes from JSON
        tubes_file = rules_dir / "learned_tubes.json"
        if tubes_file.exists():
            try:
                with tubes_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    tubes = data.get("tubes", [])
                    
                    valid_tubes = []
                    for tube in tubes:
                        tube_origin = tube.get("origin", "Unknown")
                        tube_dest = tube.get("destination", "Unknown")
                        
                        if tube_origin == "Unknown" or tube_dest == "Unknown":
                            continue
                        
                        min_members = 11 if (tube_origin == "HECA" or tube_dest == "HECA") else 6
                        if tube.get("member_count", 0) <= min_members:
                            continue
                        
                        # Apply origin/destination filter
                        if origin and tube_origin != origin:
                            continue
                        if destination and tube_dest != destination:
                            continue
                        
                        valid_tubes.append(tube)
                    
                    result["tubes"] = valid_tubes
            except Exception as e2:
                logger.error(f"Error loading tubes from JSON: {e2}")
        
        # Fallback: Load SIDs from JSON
        sids_file = rules_dir / "learned_sid.json"
        if sids_file.exists():
            try:
                with sids_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    result["sids"] = data.get("procedures", [])
            except Exception as e2:
                logger.error(f"Error loading SIDs from JSON: {e2}")
        
        # Fallback: Load STARs from JSON
        stars_file = rules_dir / "learned_star.json"
        if stars_file.exists():
            try:
                with stars_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    result["stars"] = data.get("procedures", [])
            except Exception as e2:
                logger.error(f"Error loading STARs from JSON: {e2}")
    
    # Load turns from JSON (not in PostgreSQL yet)
    turns_file = rules_dir / "learned_turns.json"
    if turns_file.exists():
        try:
            with turns_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                result["turns"] = data.get("zones", [])
        except Exception as e:
            logger.error(f"Error loading learned_turns.json: {e}")
    
    return result


@router.get("/api/union-tubes")
def get_union_tubes(origin: Optional[str] = None, destination: Optional[str] = None):
    """
    Return unified tube geometries - one shape per origin-destination pair.
    Each OD pair's individual tubes are merged into a single polygon using convex hull.
    Now using PostgreSQL for 10-20x faster queries!
    
    Query parameters:
    - origin: Filter by origin airport code (e.g., "LLBG")
    - destination: Filter by destination airport code (e.g., "LLSD")
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from service.learned_data_provider import get_all_tubes
    
    result = {"union_tubes": []}
    
    try:
        # Load tubes from PostgreSQL (FAST!)
        min_tube_members = 11 if (origin == "HECA" or destination == "HECA") else 4
        
        pg_tubes = get_all_tubes(
            origin=origin,
            destination=destination,
            min_member_count=min_tube_members
        )
        
        # Filter out Unknown origins/destinations
        valid_tubes = [
            t for t in pg_tubes
            if t.get("origin") != "Unknown" and t.get("destination") != "Unknown"
        ]
        
        logger.info(f"Loaded {len(valid_tubes)} tubes from PostgreSQL for union-tubes endpoint")
        
    except Exception as e:
        logger.error(f"Error loading tubes from PostgreSQL: {e}", exc_info=True)
        # Fallback to JSON file
        logger.warning("Falling back to JSON file for union-tubes...")
        
        rules_dir = Path("rules")
        tubes_file = rules_dir / "learned_tubes.json"
        
        if not tubes_file.exists():
            return result
        
        try:
            with tubes_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                tubes = data.get("tubes", [])
            
            # Filter valid tubes
            valid_tubes = []
            for tube in tubes:
                tube_origin = tube.get("origin", "Unknown")
                tube_dest = tube.get("destination", "Unknown")
                
                # Skip Unknown origins/destinations
                if tube_origin == "Unknown" or tube_dest == "Unknown":
                    continue
                
                # Apply member_count threshold
                min_members = 11 if (tube_origin == "HECA" or tube_dest == "HECA") else 4
                if tube.get("member_count", 0) <= min_members:
                    continue
                
                # Apply origin/destination filter
                if origin and tube_origin != origin:
                    continue
                if destination and tube_dest != destination:
                    continue
                
                valid_tubes.append(tube)
        except Exception as e2:
            logger.error(f"Error loading tubes from JSON: {e2}")
            return result
    
    try:
        
        # Group valid tubes by OD pair
        od_groups: Dict[Tuple[str, str], List[Dict]] = {}
        for tube in valid_tubes:
            tube_origin = tube.get("origin", "")
            tube_dest = tube.get("destination", "")
            
            # Apply user filters (origin/destination query parameters)
            if origin and tube_origin != origin:
                continue
            if destination and tube_dest != destination:
                continue
            
            od_key = (tube_origin, tube_dest)
            if od_key not in od_groups:
                od_groups[od_key] = []
            od_groups[od_key].append(tube)
        
        # Create union for each OD pair
        for (od_origin, od_dest), od_tubes in od_groups.items():
            if not od_tubes:
                continue
            
            # Collect all polygons for this OD pair
            polygons = []
            min_alt = float('inf')
            max_alt = float('-inf')
            total_members = 0
            
            for tube in od_tubes:
                geometry = tube.get("geometry", [])
                if len(geometry) >= 3:  # Need at least 3 points for a polygon
                    try:
                        # Convert to (lon, lat) for Shapely
                        coords = [(pt[1], pt[0]) for pt in geometry]
                        # Close the polygon if not already closed
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        poly = Polygon(coords)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception as e:
                        logger.warning(f"Invalid tube geometry for {od_origin}->{od_dest}: {e}")
                
                # Track altitude range and member count
                if tube.get("min_alt_ft") is not None:
                    min_alt = min(min_alt, tube["min_alt_ft"])
                if tube.get("max_alt_ft") is not None:
                    max_alt = max(max_alt, tube["max_alt_ft"])
                total_members += tube.get("member_count", 0)
            
            if not polygons:
                continue
            
            # Create union of all polygons for this OD pair
            try:
                union = unary_union(polygons)
                
                # Convert back to [lat, lon] coordinates for GeoJSON
                def polygon_to_coords(poly):
                    # Extract exterior coordinates
                    coords = list(poly.exterior.coords)
                    # Convert from (lon, lat) to [lat, lon]
                    return [[lat, lon] for lon, lat in coords]
                
                # Handle MultiPolygon or Polygon result
                if isinstance(union, MultiPolygon):
                    # Take the largest polygon from the multipolygon
                    largest = max(union.geoms, key=lambda p: p.area)
                    geometry = polygon_to_coords(largest)
                elif isinstance(union, Polygon):
                    geometry = polygon_to_coords(union)
                else:
                    continue
                
                union_tube = {
                    "id": f"UNION_{od_origin}_{od_dest}",
                    "origin": od_origin,
                    "destination": od_dest,
                    "min_alt_ft": min_alt if min_alt != float('inf') else None,
                    "max_alt_ft": max_alt if max_alt != float('-inf') else None,
                    "tube_count": len(od_tubes),
                    "member_count": total_members,
                    "geometry": geometry
                }
                
                result["union_tubes"].append(union_tube)
                
            except Exception as e:
                logger.error(f"Failed to create union for {od_origin}->{od_dest}: {e}")
    
    except Exception as e:
        logger.error(f"Error creating union tubes: {e}")
    
    return result


@router.get("/api/live/track/{flight_id}")
def get_live_track(flight_id: str):
    """
    Fetch the full track for a flight.
    1. Try Live Tracks DB
    2. Fallback to Cache DB
    """
    points = []

    # 1. Try Live DB
    if DB_TRACKS_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_TRACKS_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT * FROM flight_tracks 
                WHERE flight_id = ? 
                ORDER BY timestamp ASC
            """

            cursor.execute(query, (flight_id,))
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                points.append({
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "alt": row["alt"],
                    "timestamp": row["timestamp"],
                    "gspeed": row["gspeed"],
                    "track": row["track"],
                    "flight_id": row["flight_id"]
                })
        except Exception as e:
            logger.error(f"Failed to fetch from live tracks: {e}")

    # 2. Fallback to Cache
    if not points:
        try:
            conn = sqlite3.connect(str(CACHE_DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                data = json.loads(row[0])
                points = data.get("points", [])
        except Exception as e:
            logger.error(f"Failed to fetch from cache: {e}")

    if not points:
        raise HTTPException(status_code=404, detail="Track not found in live or cache DB")

    return {
        "flight_id": flight_id,
        "points": points
    }


@router.get("/api/live/flight-status/{flight_id}")
def get_live_flight_status(flight_id: str):
    """
    Get the current flight status/phase based on the most recent track data from PostgreSQL.
    Returns flight phase, altitude, speed, heading, and route information.
    
    Search order:
    1. live schema (realtime monitor data)
    2. feedback schema (tagged flights)
    3. research schema (historical data)
    """
    try:
        from service.pg_provider import get_connection
        
        points = []
        callsign = None
        origin = None
        destination = None
        source_schema = None
        
        # Try each schema in order: live -> feedback -> research
        for schema in ['live', 'feedback', 'research']:
            try:
                with get_connection() as conn:
                    with conn.cursor() as cursor:
                        # Get metadata for origin/destination/callsign
                        cursor.execute(f"""
                            SELECT callsign, origin_airport, destination_airport 
                            FROM {schema}.flight_metadata 
                            WHERE flight_id = %s
                        """, (flight_id,))
                        meta_row = cursor.fetchone()
                        
                        if meta_row:
                            callsign = meta_row[0]
                            origin = meta_row[1]
                            destination = meta_row[2]
                        
                        # Get the most recent 10 points from tracks
                        # Try normal_tracks first (most common), then anomalies_tracks
                        for table in ['normal_tracks', 'anomalies_tracks', 'flight_tracks']:
                            try:
                                cursor.execute(f"""
                                    SELECT lat, lon, alt, timestamp, gspeed, track, callsign 
                                    FROM {schema}.{table}
                                    WHERE flight_id = %s 
                                    ORDER BY timestamp DESC
                                    LIMIT 10
                                """, (flight_id,))
                                rows = cursor.fetchall()
                                
                                if rows:
                                    source_schema = schema
                                    for row in rows:
                                        points.append({
                                            "lat": row[0],
                                            "lon": row[1],
                                            "alt": row[2],
                                            "timestamp": row[3],
                                            "gspeed": row[4],
                                            "track": row[5],
                                            "callsign": row[6]
                                        })
                                        if not callsign and row[6]:
                                            callsign = row[6]
                                    break
                            except Exception as e:
                                print(f"Error fetching from {schema} schema: {e}")
                                
                        
                        # If we found points, stop searching
                        if points:
                            logger.debug(f"Found flight status in {schema} schema for {flight_id}")
                            break
                            
            except Exception as e:
                logger.debug(f"Could not fetch from {schema} schema: {e}")
                continue
        
        if not points:
            raise HTTPException(status_code=404, detail="Flight not found")
        
        # Get the most recent point
        latest = points[0] if points else {}
        altitude_ft = latest.get("alt", 0)
        speed_kts = latest.get("gspeed", 0)
        heading = latest.get("track", 0)
        lat = latest.get("lat", 0)
        lon = latest.get("lon", 0)
        
        # Determine flight phase based on altitude and speed trends
        status = "UNKNOWN"
        if len(points) >= 2:
            alt_diff = points[0].get("alt", 0) - points[-1].get("alt", 0)
            
            if altitude_ft < 1000:
                status = "GROUND"
            elif altitude_ft < 5000:
                if alt_diff > 500:
                    status = "TAKEOFF"
                elif alt_diff < -500:
                    status = "LANDING"
                else:
                    status = "APPROACH"
            elif altitude_ft < 15000:
                if alt_diff > 1000:
                    status = "CLIMB"
                elif alt_diff < -1000:
                    status = "DESCENT"
                else:
                    status = "APPROACH"
            else:
                if alt_diff > 500:
                    status = "CLIMB"
                elif alt_diff < -500:
                    status = "DESCENT"
                else:
                    status = "CRUISE"
        elif altitude_ft > 25000:
            status = "CRUISE"
        elif altitude_ft > 10000:
            status = "CLIMB"
        elif altitude_ft < 3000:
            status = "APPROACH"
        
        logger.info(f"[FLIGHT STATUS] {flight_id} ({callsign}): {status} at {altitude_ft}ft from {source_schema} schema")
        
        return {
            "flight_id": flight_id,
            "callsign": callsign,
            "status": status,
            "altitude_ft": int(altitude_ft),
            "speed_kts": int(speed_kts),
            "heading": int(heading) if heading else 0,
            "origin": origin,
            "destination": destination,
            "lat": lat,
            "lon": lon,
            "eta_minutes": None,  # Would require route calculation
            "source": f"postgresql.{source_schema}"
        }
    
    except HTTPException:
        raise
    except ImportError as ie:
        logger.error(f"PostgreSQL provider not available: {ie}")
        raise HTTPException(status_code=500, detail="PostgreSQL provider not available")
    except Exception as e:
        logger.error(f"Failed to fetch flight status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Flight Import API - Search by callsign and import to feedback_tagged.db
# ============================================================

class FlightSearchByCallsignRequest(BaseModel):
    callsign: str
    start_ts: int  # Unix timestamp
    end_ts: int    # Unix timestamp


class FlightImportRequest(BaseModel):
    flight_id: str
    rule_ids: List[int] = []
    comments: str = ""
    is_anomaly: bool = True
    run_pipeline: bool = True


@router.post("/api/import/search")
def search_flights_by_callsign(request: FlightSearchByCallsignRequest):
    """
    Search for flights by callsign within a time range using FR24 API.
    Returns a list of matching flights with their IDs and full info.
    """
    try:
        from fr24sdk.client import Client
        FR24_TOKEN = "019aca50-8288-7260-94b5-6d82fbeb351c|dC21vuw2bsf2Y43qAlrBKb7iSM9ibqSDT50x3giN763b577b"
        
        client = Client(api_token=FR24_TOKEN)
        
        # Convert timestamps to datetime strings
        from_str = datetime.fromtimestamp(request.start_ts).strftime('%Y-%m-%dT%H:%M:%S')
        to_str = datetime.fromtimestamp(request.end_ts).strftime('%Y-%m-%dT%H:%M:%S')
        
        logger.info(f"Searching FR24 for callsign '{request.callsign}' from {from_str} to {to_str}")
        
        # Search by callsign using get_full for complete data
        summary = client.flight_summary.get_full(
            flight_datetime_from=from_str,
            flight_datetime_to=to_str,
            callsigns=[request.callsign]
        )
        
        data = summary.model_dump().get("data", [])
        
        if not data:
            return {"flights": [], "message": "No flights found"}
        
        flights = []
        for item in data:
            flight_id = item.get("fr24_id") or item.get("id")
            if not flight_id:
                continue
            
            # Extract airport coordinates if available
            origin_info = item.get("orig", {}) or {}
            dest_info = item.get("dest", {}) or {}
            
            flights.append({
                "flight_id": flight_id,
                "callsign": item.get("callsign"),
                "flight_number": item.get("flight") or item.get("flight_number"),
                "origin": item.get("orig_iata") or item.get("orig_icao") or origin_info.get("iata") or origin_info.get("icao"),
                "origin_name": origin_info.get("name"),
                "origin_lat": origin_info.get("lat"),
                "origin_lon": origin_info.get("lon"),
                "destination": item.get("dest_iata") or item.get("dest_icao") or dest_info.get("iata") or dest_info.get("icao"),
                "destination_name": dest_info.get("name"),
                "dest_lat": dest_info.get("lat"),
                "dest_lon": dest_info.get("lon"),
                "airline": item.get("airline_name") or item.get("airline"),
                "airline_code": item.get("airline_iata") or item.get("airline_icao"),
                "aircraft_type": item.get("aircraft_code") or item.get("equip") or item.get("aircraft", {}).get("model_code"),
                "aircraft_model": item.get("aircraft", {}).get("model") if isinstance(item.get("aircraft"), dict) else item.get("aircraft_model"),
                "aircraft_registration": item.get("reg") or item.get("registration") or (item.get("aircraft", {}).get("registration") if isinstance(item.get("aircraft"), dict) else None),
                "scheduled_departure": item.get("schd_dep") or item.get("scheduled_departure"),
                "scheduled_arrival": item.get("schd_arr") or item.get("scheduled_arrival"),
                "actual_departure": item.get("actual_dep") or item.get("actual_departure"),
                "actual_arrival": item.get("actual_arr") or item.get("actual_arrival"),
                "status": item.get("status") or item.get("status_text"),
                "is_military": item.get("is_military", False),
            })
        
        logger.info(f"Found {len(flights)} flights for callsign '{request.callsign}'")
        return {"flights": flights, "message": f"Found {len(flights)} flight(s)"}
        
    except ImportError:
        raise HTTPException(status_code=500, detail="FR24 SDK not available")
    except Exception as e:
        logger.error(f"Error searching flights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/import/tracks/{flight_id}")
def get_import_flight_tracks(flight_id: str):
    """
    Fetch flight tracks from FR24 API for preview before import.
    """
    try:
        from fr24sdk.client import Client
        FR24_TOKEN = "019aca50-8288-7260-94b5-6d82fbeb351c|dC21vuw2bsf2Y43qAlrBKb7iSM9ibqSDT50x3giN763b577b"
        
        client = Client(api_token=FR24_TOKEN)
        
        logger.info(f"Fetching tracks for flight {flight_id}")
        tracks_response = client.flight_tracks.get(flight_id=[flight_id])
        data_list = tracks_response.model_dump().get("data", [])
        
        if not data_list:
            raise HTTPException(status_code=404, detail="No track data found")
        
        flight_data = data_list[0]
        track_points = flight_data.get("tracks", [])
        
        points = []
        for tp in track_points:
            ts_val = tp.get("timestamp")
            if isinstance(ts_val, str):
                ts = int(datetime.fromisoformat(ts_val.replace("Z", "+00:00")).timestamp())
            elif ts_val is not None:
                ts = int(ts_val)
            else:
                continue
            
            points.append({
                "flight_id": flight_id,
                "timestamp": ts,
                "lat": float(tp["lat"]) if tp.get("lat") is not None else None,
                "lon": float(tp["lon"]) if tp.get("lon") is not None else None,
                "alt": float(tp["alt"]) if tp.get("alt") is not None else None,
                "gspeed": float(tp["gspeed"]) if tp.get("gspeed") is not None else None,
                "vspeed": float(tp["vspeed"]) if tp.get("vspeed") is not None else None,
                "track": float(tp["track"]) if tp.get("track") is not None else None,
                "squawk": str(tp["squawk"]) if tp.get("squawk") else None,
                "callsign": tp.get("callsign"),
                "source": "fr24"
            })
        
        # Sort by timestamp
        points.sort(key=lambda p: p.get("timestamp", 0))
        
        return {"flight_id": flight_id, "points": points}
        
    except ImportError:
        raise HTTPException(status_code=500, detail="FR24 SDK not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/import/save")
def import_flight_to_feedback(request: FlightImportRequest):
    """
    Import a flight from FR24 to feedback_tagged.db.
    Optionally runs the anomaly pipeline and saves the report.
    """
    import math
    
    try:
        from fr24sdk.client import Client
        FR24_TOKEN = "019aca50-8288-7260-94b5-6d82fbeb351c|dC21vuw2bsf2Y43qAlrBKb7iSM9ibqSDT50x3giN763b577b"
        
        client = Client(api_token=FR24_TOKEN)
        flight_id = request.flight_id
        
        logger.info(f"Importing flight {flight_id} to feedback_tagged.db")
        
        # 1. Fetch tracks
        tracks_response = client.flight_tracks.get(flight_id=[flight_id])
        data_list = tracks_response.model_dump().get("data", [])
        
        if not data_list:
            raise HTTPException(status_code=404, detail="No track data found")
        
        flight_data = data_list[0]
        track_points = flight_data.get("tracks", [])
        
        tracks = []
        for tp in track_points:
            ts_val = tp.get("timestamp")
            if isinstance(ts_val, str):
                ts = int(datetime.fromisoformat(ts_val.replace("Z", "+00:00")).timestamp())
            elif ts_val is not None:
                ts = int(ts_val)
            else:
                continue
            
            tracks.append({
                "flight_id": flight_id,
                "timestamp": ts,
                "lat": float(tp["lat"]) if tp.get("lat") is not None else None,
                "lon": float(tp["lon"]) if tp.get("lon") is not None else None,
                "alt": float(tp["alt"]) if tp.get("alt") is not None else None,
                "gspeed": float(tp["gspeed"]) if tp.get("gspeed") is not None else None,
                "vspeed": float(tp["vspeed"]) if tp.get("vspeed") is not None else None,
                "track": float(tp["track"]) if tp.get("track") is not None else None,
                "squawk": str(tp["squawk"]) if tp.get("squawk") else None,
                "callsign": tp.get("callsign"),
                "source": "fr24_import"
            })
        
        tracks.sort(key=lambda p: p.get("timestamp", 0))
        
        if not tracks:
            raise HTTPException(status_code=404, detail="No valid track points")
        
        # 2. Get flight time for metadata lookup
        flight_time = tracks[0].get("timestamp", int(datetime.now().timestamp()))
        
        # 3. Fetch full metadata from FR24 using get_full
        time_from = flight_time - 60 * 60 * 24
        time_to = flight_time + 60 * 60 * 24
        from_str = datetime.fromtimestamp(time_from).strftime('%Y-%m-%dT%H:%M:%S')
        to_str = datetime.fromtimestamp(time_to).strftime('%Y-%m-%dT%H:%M:%S')
        
        fr24_metadata = {}
        try:
            # Use get_full for complete flight data
            summary = client.flight_summary.get_full(
                flight_datetime_from=from_str,
                flight_datetime_to=to_str,
                flight_ids=[flight_id]
            )
            data = summary.model_dump().get("data", [])
            if data:
                item = data[0]
                
                # Extract airport info with coordinates
                origin_info = item.get("orig", {}) or {}
                dest_info = item.get("dest", {}) or {}
                aircraft_info = item.get("aircraft", {}) if isinstance(item.get("aircraft"), dict) else {}
                is_military = "military" in item.get("category", '').lower()
                fr24_metadata = {
                    # Basic identifiers
                    "callsign": item.get("callsign"),
                    "flight_number": item.get("flight") or item.get("flight_number"),
                    
                    # Airline info
                    "airline": item.get("airline_name") or item.get("airline"),
                    "airline_code": item.get("airline_iata") or item.get("airline_icao"),
                    
                    # Aircraft info
                    "aircraft_type": item.get("aircraft_code") or item.get("equip") or aircraft_info.get("model_code"),
                    "aircraft_model": aircraft_info.get("model") or item.get("aircraft_model") or (item.get("aircraft") if isinstance(item.get("aircraft"), str) else None),
                    "aircraft_registration": item.get("reg") or item.get("registration") or aircraft_info.get("registration"),
                    
                    # Origin airport with coordinates
                    "origin_airport": item.get("orig_iata") or item.get("orig_icao") or origin_info.get("iata") or origin_info.get("icao"),
                    "origin_lat": origin_info.get("lat"),
                    "origin_lon": origin_info.get("lon"),
                    
                    # Destination airport with coordinates
                    "destination_airport": item.get("dest_iata") or item.get("dest_icao") or dest_info.get("iata") or dest_info.get("icao"),
                    "dest_lat": dest_info.get("lat"),
                    "dest_lon": dest_info.get("lon"),
                    
                    # Schedule info
                    "scheduled_departure": item.get("schd_dep") or item.get("scheduled_departure"),
                    "scheduled_arrival": item.get("schd_arr") or item.get("scheduled_arrival"),
                    
                    # Military flag

                    "is_military": is_military,
                    "military_type": item.get("military_type") or (aircraft_info.get("type") if is_military else None),
                    
                    # Category from FR24
                    "category": item.get("category") or item.get("type") or item.get("flight_type"),
                }
                
                logger.info(f"FR24 metadata loaded: callsign={fr24_metadata.get('callsign')}, "
                           f"airline={fr24_metadata.get('airline')}, origin={fr24_metadata.get('origin_airport')}, "
                           f"dest={fr24_metadata.get('destination_airport')}, aircraft={fr24_metadata.get('aircraft_type')}, "
                           f"is_military={fr24_metadata.get('is_military')}, category={fr24_metadata.get('category')}")
        except Exception as e:
            logger.warning(f"Failed to fetch FR24 metadata: {e}")
        
        # 4. Compute full metadata with ALL fields from feedback_tagged.db schema
        now = int(datetime.now().timestamp())
        sorted_tracks = sorted(tracks, key=lambda p: p.get("timestamp", 0))
        
        metadata = {
            # Core identifiers
            "flight_id": flight_id,
            "callsign": fr24_metadata.get("callsign"),
            "flight_number": fr24_metadata.get("flight_number"),
            
            # Airline info
            "airline": fr24_metadata.get("airline"),
            "airline_code": fr24_metadata.get("airline_code"),
            
            # Aircraft info
            "aircraft_type": fr24_metadata.get("aircraft_type"),
            "aircraft_model": fr24_metadata.get("aircraft_model"),
            "aircraft_registration": fr24_metadata.get("aircraft_registration"),
            
            # Origin airport with coordinates
            "origin_airport": fr24_metadata.get("origin_airport"),
            "origin_lat": fr24_metadata.get("origin_lat"),
            "origin_lon": fr24_metadata.get("origin_lon"),
            
            # Destination airport with coordinates
            "destination_airport": fr24_metadata.get("destination_airport"),
            "dest_lat": fr24_metadata.get("dest_lat"),
            "dest_lon": fr24_metadata.get("dest_lon"),
            
            # Schedule
            "scheduled_departure": fr24_metadata.get("scheduled_departure"),
            "scheduled_arrival": fr24_metadata.get("scheduled_arrival"),
            
            # Military flags
            "is_military": fr24_metadata.get("is_military", False),
            "military_type": fr24_metadata.get("military_type"),
            
            # Stats (will be computed)
            "total_points": len(tracks),
            
            # System fields
            "created_at": now,
            "updated_at": now,
            "is_anomaly": request.is_anomaly,
        }
        
        # Compute from tracks
        timestamps = [p.get("timestamp", 0) for p in sorted_tracks if p.get("timestamp")]
        if timestamps:
            metadata["first_seen_ts"] = min(timestamps)
            metadata["last_seen_ts"] = max(timestamps)
            metadata["flight_duration_sec"] = metadata["last_seen_ts"] - metadata["first_seen_ts"]
        
        # Callsign from tracks if not in metadata
        if not metadata["callsign"]:
            for p in sorted_tracks:
                if p.get("callsign"):
                    metadata["callsign"] = p["callsign"]
                    break
        
        # Start/End positions
        if sorted_tracks:
            first = sorted_tracks[0]
            last = sorted_tracks[-1]
            metadata["start_lat"] = first.get("lat")
            metadata["start_lon"] = first.get("lon")
            metadata["end_lat"] = last.get("lat")
            metadata["end_lon"] = last.get("lon")
        
        # Altitude stats
        altitudes = [p.get("alt") for p in sorted_tracks if p.get("alt") is not None and p.get("alt") > 0]
        if altitudes:
            metadata["min_altitude_ft"] = min(altitudes)
            metadata["max_altitude_ft"] = max(altitudes)
            metadata["avg_altitude_ft"] = sum(altitudes) / len(altitudes)
            # Cruise altitude = most common altitude in high-altitude portion
            cruise_alts = [a for a in altitudes if a > 20000]
            if cruise_alts:
                # Use mode-like approach: round to nearest 1000ft and find most common
                rounded = [round(a / 1000) * 1000 for a in cruise_alts]
                from collections import Counter
                most_common = Counter(rounded).most_common(1)
                if most_common:
                    metadata["cruise_altitude_ft"] = most_common[0][0]
        
        # Speed stats
        speeds = [p.get("gspeed") for p in sorted_tracks if p.get("gspeed") is not None and p.get("gspeed") >= 0]
        if speeds:
            metadata["min_speed_kts"] = min(speeds)
            metadata["max_speed_kts"] = max(speeds)
            metadata["avg_speed_kts"] = sum(speeds) / len(speeds)
        
        # Total distance
        total_dist = 0.0
        for i in range(1, len(sorted_tracks)):
            p1, p2 = sorted_tracks[i-1], sorted_tracks[i]
            lat1, lon1 = p1.get("lat"), p1.get("lon")
            lat2, lon2 = p2.get("lat"), p2.get("lon")
            if all(v is not None for v in [lat1, lon1, lat2, lon2]):
                R = 3440.065  # Earth radius in NM
                lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                total_dist += R * c
        metadata["total_distance_nm"] = round(total_dist, 2)
        
        # Squawk codes
        squawks = set()
        emergency_squawks = {"7500", "7600", "7700"}
        for p in sorted_tracks:
            sq = p.get("squawk")
            if sq and str(sq) not in ("", "0", "None"):
                squawks.add(str(sq))
                if str(sq) in emergency_squawks:
                    metadata["emergency_squawk_detected"] = True
        if squawks:
            metadata["squawk_codes"] = json.dumps(list(squawks))
        
        # Signal loss detection (gaps > 60 seconds)
        signal_loss_events = 0
        for i in range(1, len(sorted_tracks)):
            ts1 = sorted_tracks[i-1].get("timestamp", 0)
            ts2 = sorted_tracks[i].get("timestamp", 0)
            if ts2 - ts1 > 60:  # Gap > 60 seconds
                signal_loss_events += 1
        metadata["signal_loss_events"] = signal_loss_events
        
        # Data quality score (0-100 based on point density and completeness)
        if metadata.get("flight_duration_sec") and metadata.get("flight_duration_sec") > 0:
            expected_points = metadata["flight_duration_sec"] / 5  # Expected 1 point per 5 seconds
            actual_points = len(sorted_tracks)
            density_score = min(100, (actual_points / expected_points) * 100) if expected_points > 0 else 0
            # Also consider completeness of data fields
            completeness = sum(
                (1 if p.get("lat") is not None else 0) +
                (1 if p.get("lon") is not None else 0) +
                (1 if p.get("alt") is not None else 0) +
                (1 if p.get("gspeed") is not None else 0)
                for p in sorted_tracks[:min(50, len(sorted_tracks))]
            ) / (min(50, len(sorted_tracks)) * 4) * 100 if sorted_tracks else 0
            metadata["data_quality_score"] = round((density_score + completeness) / 2, 1)
        
        # Flight phase summary (based on altitude profile)
        if altitudes and len(altitudes) > 10:
            phases = []
            max_alt = max(altitudes)
            min_alt = min(altitudes)
            avg_start = sum(altitudes[:5]) / 5 if len(altitudes) >= 5 else altitudes[0]
            avg_end = sum(altitudes[-5:]) / 5 if len(altitudes) >= 5 else altitudes[-1]
            
            if avg_start < 5000:
                phases.append("TAKEOFF")
            if max_alt > 25000:
                phases.append("CRUISE")
            elif max_alt > 10000:
                phases.append("ENROUTE")
            if avg_end < 5000:
                phases.append("LANDING")
            
            metadata["flight_phase_summary"] = " -> ".join(phases) if phases else "UNKNOWN"
        
        # Nearest airports (use origin/destination from FR24 as approximation)
        metadata["nearest_airport_start"] = fr24_metadata.get("origin_airport")
        metadata["nearest_airport_end"] = fr24_metadata.get("destination_airport")
        
        # Crossed borders (placeholder - would need geographic boundary data)
        # For now, set to None - could be enhanced with country boundary checks
        metadata["crossed_borders"] = None
        
        # Category from FR24
        metadata["category"] = fr24_metadata.get("category")
        
        # 5. Run anomaly pipeline if requested
        import time as time_module
        report = None
        if request.run_pipeline:
            try:
                logger.info(f"Starting pipeline analysis for {flight_id} ({len(tracks)} points)...")
                t_start = time_module.perf_counter()
                
                pipeline = _get_pipeline()
                t_pipeline = time_module.perf_counter()
                logger.info(f"  Pipeline loaded in {t_pipeline - t_start:.2f}s")
                
                track_points_obj = [
                    TrackPoint(
                        flight_id=flight_id,
                        timestamp=p.get("timestamp"),
                        lat=p.get("lat"),
                        lon=p.get("lon"),
                        alt=p.get("alt"),
                        gspeed=p.get("gspeed"),
                        vspeed=p.get("vspeed"),
                        track=p.get("track"),
                        squawk=p.get("squawk"),
                        callsign=p.get("callsign"),
                        source=p.get("source")
                    )
                    for p in tracks
                ]
                flight_track = FlightTrack(flight_id=flight_id, points=track_points_obj)
                t_prep = time_module.perf_counter()
                logger.info(f"  Track prepared in {t_prep - t_pipeline:.2f}s")
                
                report = pipeline.analyze(flight_track)
                t_analyze = time_module.perf_counter()
                logger.info(f"  Analysis complete in {t_analyze - t_prep:.2f}s (total: {t_analyze - t_start:.2f}s)")
            except Exception as e:
                logger.warning(f"Pipeline analysis failed: {e}")
        
        # 6. Get rule names
        rule_names = []
        for rid in request.rule_ids:
            rule = next((r for r in RULES_METADATA if r["id"] == rid), None)
            if rule:
                rule_names.append(rule["name"])
            else:
                rule_names.append(f"Rule {rid}")
        
        # 7. Save to feedback_tagged.db
        from ..feedback_db import save_tagged_flight
        
        success = save_tagged_flight(
            flight_id=flight_id,
            rule_id=request.rule_ids[0] if request.rule_ids else None,
            rule_name=rule_names[0] if rule_names else None,
            comments=request.comments,
            other_details="",
            metadata=metadata,
            tracks=tracks,
            report=report,
            is_anomaly=request.is_anomaly,
            rule_ids=request.rule_ids,
            rule_names=rule_names
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save flight to database")
        
        return {
            "status": "success",
            "flight_id": flight_id,
            "track_count": len(tracks),
            "rule_ids": request.rule_ids,
            "rule_names": rule_names,
            "is_anomaly": request.is_anomaly,
            "pipeline_ran": request.run_pipeline,
            "callsign": metadata.get("callsign"),
            "origin": metadata.get("origin_airport"),
            "destination": metadata.get("destination_airport"),
        }
        
    except ImportError:
        raise HTTPException(status_code=500, detail="FR24 SDK not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing flight: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# POLYGON SEARCH ENDPOINTS
# ============================================================================

class PolygonSearchRequest(BaseModel):
    """Request model for polygon search."""
    polygon: List[List[float]]  # Array of [lon, lat] coordinates
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None


class WKTSearchRequest(BaseModel):
    """Request model for WKT polygon search."""
    wkt: str  # WKT polygon string
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None


@router.post("/api/research/polygon-search")
def search_flights_by_polygon(request: PolygonSearchRequest):
    """
    Search for flights in research_new.db that pass through a polygon area.
    
    Args:
        request: PolygonSearchRequest with polygon coordinates and optional time filters
        
    Returns:
        List of flights with their metadata
    """
    try:
        from ..polygon_search import get_flights_in_polygon
        
        if not DB_RESEARCH_PATH or not DB_RESEARCH_PATH.exists():
            raise HTTPException(status_code=404, detail="Research database not found")
        
        if not request.polygon or len(request.polygon) < 3:
            raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")
        
        results = get_flights_in_polygon(
            db_path=DB_RESEARCH_PATH,
            polygon_coords=request.polygon,
            start_ts=request.start_ts,
            end_ts=request.end_ts
        )
        
        return {
            "count": len(results),
            "flights": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching flights by polygon: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/research/wkt-search")
def search_flights_by_wkt(request: WKTSearchRequest):
    """
    Search for flights in research_new.db using a WKT polygon string.
    
    Args:
        request: WKTSearchRequest with WKT string and optional time filters
        
    Returns:
        List of flights with their metadata
    """
    try:
        from ..polygon_search import get_flights_in_wkt
        
        if not DB_RESEARCH_PATH or not DB_RESEARCH_PATH.exists():
            raise HTTPException(status_code=404, detail="Research database not found")
        
        if not request.wkt or not request.wkt.strip():
            raise HTTPException(status_code=400, detail="WKT string cannot be empty")
        
        results = get_flights_in_wkt(
            db_path=DB_RESEARCH_PATH,
            wkt_polygon=request.wkt,
            start_ts=request.start_ts,
            end_ts=request.end_ts
        )
        
        return {
            "count": len(results),
            "flights": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching flights by WKT: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flight/metadata/{flight_id}")
def get_flight_metadata_by_id(flight_id: str):
    """
    Get flight metadata for ANY flight by ID from research schema.
    """
    try:
        from service.pg_provider import get_flight_metadata
        
        metadata = get_flight_metadata(flight_id, schema='research')
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Flight {flight_id} not found in research schema")
        
        return {
            "flight_id": flight_id,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching flight metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flight/track/{flight_id}")
def get_flight_track_by_id(flight_id: str):
    """
    Get flight track for ANY flight by ID from research schema.
    """
    try:
        from service.pg_provider import get_flight_track
        
        points = get_flight_track(flight_id, schema='research')
        
        if not points:
            raise HTTPException(status_code=404, detail=f"Track for flight {flight_id} not found in research schema")
        
        return {
            "flight_id": flight_id,
            "points": points,
            "count": len(points)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching flight track: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

