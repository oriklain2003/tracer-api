import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DB_PATH = ROOT_DIR / "training_ops/training_dataset.db"
FEEDBACK_DB_PATH = ROOT_DIR / "training_ops/feedback.db"

logger = logging.getLogger(__name__)

# Columns for the user_feedback table with flight details and summary
FEEDBACK_TABLE_COLUMNS = [
    # Core feedback fields
    ("flight_id", "TEXT"),
    ("timestamp", "INTEGER"),
    ("user_label", "INTEGER"),  # 0: Normal, 1: Anomaly
    ("comments", "TEXT"),
    ("model_version", "TEXT"),
    ("rule_id", "INTEGER"),
    ("other_details", "TEXT"),
    ("full_report_json", "TEXT"),
    ("tagged", "INTEGER DEFAULT 0"),
    
    # Flight identification
    ("callsign", "TEXT"),
    ("flight_number", "TEXT"),
    
    # Origin airport
    ("origin_code", "TEXT"),
    ("origin_iata", "TEXT"),
    ("origin_icao", "TEXT"),
    ("origin_name", "TEXT"),
    ("origin_city", "TEXT"),
    ("origin_country", "TEXT"),
    
    # Destination airport
    ("dest_code", "TEXT"),
    ("dest_iata", "TEXT"),
    ("dest_icao", "TEXT"),
    ("dest_name", "TEXT"),
    ("dest_city", "TEXT"),
    ("dest_country", "TEXT"),
    
    # Airline & Aircraft
    ("airline", "TEXT"),
    ("aircraft_type", "TEXT"),
    ("aircraft_model", "TEXT"),
    ("aircraft_registration", "TEXT"),
    
    # Flight status & times
    ("flight_status", "TEXT"),
    ("scheduled_departure", "TEXT"),
    ("scheduled_arrival", "TEXT"),
    ("actual_departure", "TEXT"),
    ("actual_arrival", "TEXT"),
    
    # Flight summary statistics (computed from track points)
    ("flight_start_ts", "INTEGER"),
    ("flight_end_ts", "INTEGER"),
    ("flight_duration_sec", "INTEGER"),
    ("total_points", "INTEGER"),
    ("min_altitude_ft", "REAL"),
    ("max_altitude_ft", "REAL"),
    ("avg_altitude_ft", "REAL"),
    ("min_speed_kts", "REAL"),
    ("max_speed_kts", "REAL"),
    ("avg_speed_kts", "REAL"),
    ("start_lat", "REAL"),
    ("start_lon", "REAL"),
    ("end_lat", "REAL"),
    ("end_lon", "REAL"),
    ("total_distance_nm", "REAL"),
    ("squawk_codes", "TEXT"),  # JSON array of unique squawk codes
    
    # Anomaly summary
    ("anomaly_confidence", "REAL"),
    ("triggered_rules", "TEXT"),  # JSON array of rule IDs
    ("ml_layers_flagged", "TEXT"),  # JSON array of layer names that flagged anomaly
]

TRACK_TABLE_COLUMNS = [
    ("flight_id", "TEXT"),
    ("timestamp", "INTEGER"),
    ("lat", "REAL"),
    ("lon", "REAL"),
    ("alt", "REAL"),
    ("heading", "REAL"),
    ("gspeed", "REAL"),
    ("vspeed", "REAL"),
    ("track", "REAL"),
    ("squawk", "TEXT"),
    ("callsign", "TEXT"),
    ("source", "TEXT"),
]

def ensure_table_columns(cursor: sqlite3.Cursor, table_name: str) -> None:
    """
    Make sure legacy databases contain the full schema.
    SQLite's ALTER TABLE ADD COLUMN is idempotent for new columns, so it's safe.
    """
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}

    for column_name, column_type in TRACK_TABLE_COLUMNS:
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            logger.warning(
                "Backfilled missing column '%s' on table '%s'",
                column_name,
                table_name,
            )

def ensure_feedback_columns(cursor: sqlite3.Cursor) -> None:
    """
    Ensure all feedback table columns exist for older DBs.
    """
    cursor.execute("PRAGMA table_info(user_feedback)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    for column_name, column_type in FEEDBACK_TABLE_COLUMNS:
        if column_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE user_feedback ADD COLUMN {column_name} {column_type}")
                logger.info(f"Added column '{column_name}' to user_feedback table")
            except sqlite3.OperationalError:
                pass  # Column might already exist


def init_dbs():
    """Initialize the feedback and training databases."""
    TRAINING_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Feedback DB (Meta-data about user choices with flight details)
    conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_id TEXT,
            timestamp INTEGER,
            user_label INTEGER,
            comments TEXT,
            model_version TEXT,
            rule_id INTEGER,
            other_details TEXT,
            full_report_json TEXT,
            tagged INTEGER DEFAULT 0,
            
            -- Flight identification
            callsign TEXT,
            flight_number TEXT,
            
            -- Origin airport
            origin_code TEXT,
            origin_iata TEXT,
            origin_icao TEXT,
            origin_name TEXT,
            origin_city TEXT,
            origin_country TEXT,
            
            -- Destination airport
            dest_code TEXT,
            dest_iata TEXT,
            dest_icao TEXT,
            dest_name TEXT,
            dest_city TEXT,
            dest_country TEXT,
            
            -- Airline & Aircraft
            airline TEXT,
            aircraft_type TEXT,
            aircraft_model TEXT,
            aircraft_registration TEXT,
            
            -- Flight status & times
            flight_status TEXT,
            scheduled_departure TEXT,
            scheduled_arrival TEXT,
            actual_departure TEXT,
            actual_arrival TEXT,
            
            -- Flight summary statistics
            flight_start_ts INTEGER,
            flight_end_ts INTEGER,
            flight_duration_sec INTEGER,
            total_points INTEGER,
            min_altitude_ft REAL,
            max_altitude_ft REAL,
            avg_altitude_ft REAL,
            min_speed_kts REAL,
            max_speed_kts REAL,
            avg_speed_kts REAL,
            start_lat REAL,
            start_lon REAL,
            end_lat REAL,
            end_lon REAL,
            total_distance_nm REAL,
            squawk_codes TEXT,
            
            -- Anomaly summary
            anomaly_confidence REAL,
            triggered_rules TEXT,
            ml_layers_flagged TEXT
        )
    """)
    
    # Ensure all columns exist for older DBs
    ensure_feedback_columns(cursor)
    
    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uf_flight_id ON user_feedback (flight_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uf_timestamp ON user_feedback (timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uf_rule_id ON user_feedback (rule_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uf_tagged ON user_feedback (tagged)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uf_callsign ON user_feedback (callsign)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uf_origin ON user_feedback (origin_code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uf_dest ON user_feedback (dest_code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uf_airline ON user_feedback (airline)")
    
    conn.commit()
    conn.close()

    # 2. Training DB (The actual flight data repository)
    # We merge everything here: base data + feedback data
    conn = sqlite3.connect(str(TRAINING_DB_PATH))
    cursor = conn.cursor()
    
    # Table for Normal Flights
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flight_tracks (
            flight_id TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            alt REAL,
            heading REAL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT
        )
    """)
    
    # Table for Anomalous Flights
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomalous_tracks (
            flight_id TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            alt REAL,
            heading REAL,
            gspeed REAL,
            vspeed REAL,
            track REAL,
            squawk TEXT,
            callsign TEXT,
            source TEXT
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ft_fid ON flight_tracks (flight_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_at_fid ON anomalous_tracks (flight_id)")

    # Ensure legacy DBs pick up the full schema (e.g., heading column)
    ensure_table_columns(cursor, "flight_tracks")
    ensure_table_columns(cursor, "anomalous_tracks")

    conn.commit()
    conn.close()

def compute_flight_summary(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics from flight track points.
    """
    summary = {
        "flight_start_ts": None,
        "flight_end_ts": None,
        "flight_duration_sec": None,
        "total_points": len(points),
        "min_altitude_ft": None,
        "max_altitude_ft": None,
        "avg_altitude_ft": None,
        "min_speed_kts": None,
        "max_speed_kts": None,
        "avg_speed_kts": None,
        "start_lat": None,
        "start_lon": None,
        "end_lat": None,
        "end_lon": None,
        "total_distance_nm": None,
        "squawk_codes": None,
    }
    
    if not points:
        return summary
    
    # Sort by timestamp
    sorted_points = sorted(points, key=lambda p: p.get("timestamp", 0))
    
    # Time range
    timestamps = [p.get("timestamp", 0) for p in sorted_points if p.get("timestamp")]
    if timestamps:
        summary["flight_start_ts"] = min(timestamps)
        summary["flight_end_ts"] = max(timestamps)
        summary["flight_duration_sec"] = summary["flight_end_ts"] - summary["flight_start_ts"]
    
    # Altitude stats
    altitudes = [p.get("alt", 0) for p in sorted_points if p.get("alt") is not None and p.get("alt") > 0]
    if altitudes:
        summary["min_altitude_ft"] = min(altitudes)
        summary["max_altitude_ft"] = max(altitudes)
        summary["avg_altitude_ft"] = sum(altitudes) / len(altitudes)
    
    # Speed stats (ground speed)
    speeds = [p.get("gspeed", 0) for p in sorted_points if p.get("gspeed") is not None and p.get("gspeed") >= 0]
    if speeds:
        summary["min_speed_kts"] = min(speeds)
        summary["max_speed_kts"] = max(speeds)
        summary["avg_speed_kts"] = sum(speeds) / len(speeds)
    
    # Start/End positions
    if sorted_points:
        first = sorted_points[0]
        last = sorted_points[-1]
        summary["start_lat"] = first.get("lat")
        summary["start_lon"] = first.get("lon")
        summary["end_lat"] = last.get("lat")
        summary["end_lon"] = last.get("lon")
    
    # Total distance (approximate using Haversine)
    try:
        import math
        total_dist = 0.0
        for i in range(1, len(sorted_points)):
            p1 = sorted_points[i - 1]
            p2 = sorted_points[i]
            lat1, lon1 = p1.get("lat"), p1.get("lon")
            lat2, lon2 = p2.get("lat"), p2.get("lon")
            if all(v is not None for v in [lat1, lon1, lat2, lon2]):
                # Haversine formula
                R = 3440.065  # Earth radius in nautical miles
                lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                total_dist += R * c
        summary["total_distance_nm"] = round(total_dist, 2)
    except Exception:
        pass
    
    # Unique squawk codes
    squawks = list(set(str(p.get("squawk", "")) for p in sorted_points if p.get("squawk")))
    squawks = [s for s in squawks if s and s != "None" and s != "0"]
    if squawks:
        summary["squawk_codes"] = json.dumps(squawks)
    
    return summary


def extract_anomaly_summary(full_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract anomaly summary from the full report.
    """
    summary = {
        "anomaly_confidence": None,
        "triggered_rules": None,
        "ml_layers_flagged": None,
    }
    
    if not full_report:
        return summary
    
    # Confidence score
    if "summary" in full_report:
        summary["anomaly_confidence"] = full_report["summary"].get("confidence_score")
    
    # Triggered rules
    triggered = []
    if "layer_1_rules" in full_report:
        rules_data = full_report["layer_1_rules"]
        if isinstance(rules_data, dict) and "report" in rules_data:
            matched = rules_data["report"].get("matched_rules", [])
            for r in matched:
                if isinstance(r, dict):
                    triggered.append(r.get("id"))
    if "matched_rules" in full_report:
        for r in full_report["matched_rules"]:
            if isinstance(r, dict) and r.get("id") not in triggered:
                triggered.append(r.get("id"))
    if triggered:
        summary["triggered_rules"] = json.dumps([r for r in triggered if r is not None])
    
    # ML layers that flagged anomaly
    ml_flagged = []
    ml_layers = [
        ("layer_2_xgboost", "XGBoost"),
        ("layer_3_deep_dense", "Deep Dense"),
        ("layer_4_deep_cnn", "Deep CNN"),
        ("layer_5_transformer", "Transformer"),
        ("layer_6_hybrid", "Hybrid"),
    ]
    for key, name in ml_layers:
        if key in full_report:
            layer_data = full_report[key]
            if isinstance(layer_data, dict) and layer_data.get("is_anomaly"):
                ml_flagged.append(name)
    if ml_flagged:
        summary["ml_layers_flagged"] = json.dumps(ml_flagged)
    
    return summary


def save_feedback(
    flight_id: str, 
    is_anomaly: bool, 
    points: List[Dict[str, Any]], 
    comments: str = "", 
    rule_id: Optional[int] = None, 
    other_details: str = "", 
    full_report: Optional[Dict[str, Any]] = None, 
    tagged: int = 1,
    flight_details: Optional[Dict[str, Any]] = None
):
    """
    Save user feedback and the corresponding flight data with comprehensive details.
    
    Args:
        flight_id: The flight identifier
        is_anomaly: Whether the user marked this as an anomaly
        points: List of track points for the flight
        comments: Optional user comments
        rule_id: The rule ID that caused the anomaly (required if is_anomaly=True, None means "Other")
        other_details: Details when rule_id is None (Other option selected)
        full_report: The anomaly report JSON (optional)
        tagged: Whether this feedback was explicitly tagged by user (default 1)
        flight_details: Flight details from FR24 API (origin, destination, airline, aircraft, etc.)
    """
    init_dbs()
    
    # Serialize full_report
    report_json = None
    if full_report:
        try:
            report_json = json.dumps(full_report)
        except Exception:
            pass

    # Compute flight summary from points
    flight_summary = compute_flight_summary(points)
    
    # Extract anomaly summary from report
    anomaly_summary = extract_anomaly_summary(full_report)
    
    # Extract flight details
    fd = flight_details or {}
    origin = fd.get("origin") or {}
    destination = fd.get("destination") or {}
    
    # Get callsign from flight_details or from points
    callsign = fd.get("callsign")
    if not callsign and points:
        for p in points:
            if p.get("callsign"):
                callsign = p["callsign"]
                break

    # 1. Save Metadata with all details
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO user_feedback (
                flight_id, timestamp, user_label, comments, model_version, rule_id, other_details, 
                full_report_json, tagged,
                callsign, flight_number,
                origin_code, origin_iata, origin_icao, origin_name, origin_city, origin_country,
                dest_code, dest_iata, dest_icao, dest_name, dest_city, dest_country,
                airline, aircraft_type, aircraft_model, aircraft_registration,
                flight_status, scheduled_departure, scheduled_arrival, actual_departure, actual_arrival,
                flight_start_ts, flight_end_ts, flight_duration_sec, total_points,
                min_altitude_ft, max_altitude_ft, avg_altitude_ft,
                min_speed_kts, max_speed_kts, avg_speed_kts,
                start_lat, start_lon, end_lat, end_lon, total_distance_nm, squawk_codes,
                anomaly_confidence, triggered_rules, ml_layers_flagged
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                flight_id, 
                int(datetime.now().timestamp()), 
                1 if is_anomaly else 0, 
                comments, 
                "v1", 
                rule_id, 
                other_details, 
                report_json, 
                tagged,
                # Flight identification
                callsign,
                fd.get("flight_number"),
                # Origin
                origin.get("code") if isinstance(origin, dict) else None,
                origin.get("iata") if isinstance(origin, dict) else None,
                origin.get("icao") if isinstance(origin, dict) else None,
                origin.get("name") if isinstance(origin, dict) else None,
                origin.get("city") if isinstance(origin, dict) else None,
                origin.get("country") if isinstance(origin, dict) else None,
                # Destination
                destination.get("code") if isinstance(destination, dict) else None,
                destination.get("iata") if isinstance(destination, dict) else None,
                destination.get("icao") if isinstance(destination, dict) else None,
                destination.get("name") if isinstance(destination, dict) else None,
                destination.get("city") if isinstance(destination, dict) else None,
                destination.get("country") if isinstance(destination, dict) else None,
                # Airline & Aircraft
                fd.get("airline"),
                fd.get("aircraft_type"),
                fd.get("aircraft_model"),
                fd.get("aircraft_registration"),
                # Status & Times
                fd.get("status"),
                fd.get("scheduled_departure"),
                fd.get("scheduled_arrival"),
                fd.get("actual_departure"),
                fd.get("actual_arrival"),
                # Flight summary
                flight_summary["flight_start_ts"],
                flight_summary["flight_end_ts"],
                flight_summary["flight_duration_sec"],
                flight_summary["total_points"],
                flight_summary["min_altitude_ft"],
                flight_summary["max_altitude_ft"],
                flight_summary["avg_altitude_ft"],
                flight_summary["min_speed_kts"],
                flight_summary["max_speed_kts"],
                flight_summary["avg_speed_kts"],
                flight_summary["start_lat"],
                flight_summary["start_lon"],
                flight_summary["end_lat"],
                flight_summary["end_lon"],
                flight_summary["total_distance_nm"],
                flight_summary["squawk_codes"],
                # Anomaly summary
                anomaly_summary["anomaly_confidence"],
                anomaly_summary["triggered_rules"],
                anomaly_summary["ml_layers_flagged"],
            )
        )
        conn.commit()
        conn.close()
        logger.info(f"Saved comprehensive feedback for {flight_id} (callsign: {callsign}, origin: {origin.get('code') if isinstance(origin, dict) else None}, dest: {destination.get('code') if isinstance(destination, dict) else None})")
    except Exception as e:
        logger.error(f"Failed to save feedback metadata: {e}")
        # We continue to save the data itself as that's crucial

    # 2. Save Flight Data to Training DB
    # Decide which table based on user label
    table = "anomalous_tracks" if is_anomaly else "flight_tracks"
    
    try:
        conn = sqlite3.connect(str(TRAINING_DB_PATH))
        cursor = conn.cursor()
        
        # Check if already exists to avoid duplicates (simple check)
        cursor.execute(f"SELECT count(*) FROM {table} WHERE flight_id = ?", (flight_id,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Insert points
            # Points structure expectation: dict with keys matching columns
            # map keys if necessary
            rows = []
            for p in points:
                rows.append((
                    flight_id,
                    p.get("timestamp", 0),
                    p.get("lat", 0.0),
                    p.get("lon", 0.0),
                    p.get("alt", 0.0),
                    p.get("heading", 0.0),
                    p.get("gspeed", 0.0),
                    p.get("vspeed", 0.0),
                    p.get("track", 0.0),
                    str(p.get("squawk", "")),
                    p.get("callsign", ""),
                    p.get("source", "feedback")
                ))
            
            cursor.executemany(
                f"""INSERT INTO {table} 
                   (flight_id, timestamp, lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, source) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows
            )
            conn.commit()
            logger.info(f"Saved {len(rows)} points for {flight_id} into {table}")
        else:
            logger.info(f"Flight {flight_id} already in training DB ({table})")
            
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")
        raise e

