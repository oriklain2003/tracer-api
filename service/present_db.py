"""
present_anomalies.db - The Single Source of Truth for Tagged Anomalies

This database is the master database for:
- All tagged anomaly flights
- Complete flight details (origin, destination, airline, aircraft)
- Full anomaly reports (from analysis/re-analysis)
- Flight summary statistics
- User feedback/tags

This is the source for the dashboard and AI search.
"""

import sqlite3
import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Database path
PRESENT_DB_PATH = Path(__file__).resolve().parent / "present_anomalies.db"


def init_present_db():
    """
    Initialize the present_anomalies.db with the comprehensive schema.
    This is the single source of truth for all tagged anomalies.
    """
    conn = sqlite3.connect(str(PRESENT_DB_PATH))
    cursor = conn.cursor()
    
    # ================================================================
    # MAIN TABLE: anomaly_reports
    # Contains all tagged anomaly flights with complete information
    # ================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_id TEXT UNIQUE,
            
            -- Tagging/Feedback Info
            tagged INTEGER DEFAULT 0,
            tagged_timestamp INTEGER,
            user_label INTEGER,
            user_rule_id INTEGER,
            user_comments TEXT,
            user_other_details TEXT,
            
            -- Flight Identification
            callsign TEXT,
            flight_number TEXT,
            
            -- Origin Airport
            origin_code TEXT,
            origin_iata TEXT,
            origin_icao TEXT,
            origin_name TEXT,
            origin_city TEXT,
            origin_country TEXT,
            
            -- Destination Airport
            dest_code TEXT,
            dest_iata TEXT,
            dest_icao TEXT,
            dest_name TEXT,
            dest_city TEXT,
            dest_country TEXT,
            
            -- Airline & Aircraft
            airline TEXT,
            airline_code TEXT,
            aircraft_type TEXT,
            aircraft_model TEXT,
            aircraft_registration TEXT,
            
            -- Flight Schedule
            flight_status TEXT,
            scheduled_departure TEXT,
            scheduled_arrival TEXT,
            actual_departure TEXT,
            actual_arrival TEXT,
            
            -- Flight Summary Statistics (computed from track)
            flight_start_ts INTEGER,
            flight_end_ts INTEGER,
            flight_duration_sec INTEGER,
            total_points INTEGER,
            min_altitude_ft REAL,
            max_altitude_ft REAL,
            avg_altitude_ft REAL,
            cruise_altitude_ft REAL,
            min_speed_kts REAL,
            max_speed_kts REAL,
            avg_speed_kts REAL,
            max_vspeed_fpm REAL,
            min_vspeed_fpm REAL,
            start_lat REAL,
            start_lon REAL,
            end_lat REAL,
            end_lon REAL,
            total_distance_nm REAL,
            squawk_codes TEXT,
            emergency_squawk_detected INTEGER DEFAULT 0,
            
            -- Classification & Context (from research DB)
            is_military INTEGER DEFAULT 0,
            military_type TEXT,
            flight_phase_summary TEXT,
            crossed_borders TEXT,
            signal_loss_events INTEGER DEFAULT 0,
            data_quality_score REAL,
            
            -- Anomaly Detection Results
            anomaly_timestamp INTEGER,
            pipeline_is_anomaly INTEGER,
            confidence_score REAL,
            severity_cnn REAL,
            severity_dense REAL,
            severity_transformer REAL,
            severity_hybrid REAL,
            
            -- Rules Analysis
            rules_status TEXT,
            rules_triggers TEXT,
            matched_rule_ids TEXT,
            matched_rule_names TEXT,
            matched_rule_categories TEXT,
            
            -- ML Layers Summary
            ml_layers_flagged TEXT,
            xgboost_is_anomaly INTEGER,
            deep_dense_is_anomaly INTEGER,
            deep_cnn_is_anomaly INTEGER,
            transformer_is_anomaly INTEGER,
            hybrid_is_anomaly INTEGER,
            
            -- Full Report JSON (for detailed analysis)
            full_report_json TEXT,
            
            -- Metadata
            created_at INTEGER,
            updated_at INTEGER,
            model_version TEXT
        )
    """)
    
    # ================================================================
    # TABLE: flight_tracks
    # Contains the actual track points for each flight
    # ================================================================
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
            source TEXT,
            PRIMARY KEY (flight_id, timestamp)
        )
    """)
    
    # ================================================================
    # TABLE: rule_matches
    # Detailed rule match information for each anomaly
    # ================================================================
    # Check if old schema exists (with report_id instead of flight_id)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rule_matches'")
    if cursor.fetchone():
        # Check if it has the old schema with report_id NOT NULL
        cursor.execute("PRAGMA table_info(rule_matches)")
        cols_info = cursor.fetchall()
        cols = {row[1] for row in cols_info}
        
        # Check if report_id has NOT NULL constraint (notnull is column index 3)
        report_id_notnull = False
        for row in cols_info:
            if row[1] == 'report_id' and row[3] == 1:  # notnull = 1 means NOT NULL
                report_id_notnull = True
                break
        
        if 'report_id' in cols and report_id_notnull:
            # Old schema with report_id NOT NULL - need to recreate table
            logger.info("Found old rule_matches table with report_id NOT NULL, recreating...")
            try:
                # Backup existing data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rule_matches_backup AS 
                    SELECT 
                        rm.*,
                        ar.flight_id as migrated_flight_id
                    FROM rule_matches rm
                    LEFT JOIN anomaly_reports ar ON ar.id = rm.report_id
                """)
                
                # Drop old table
                cursor.execute("DROP TABLE rule_matches")
                
                # Create new table with proper schema
                cursor.execute("""
                    CREATE TABLE rule_matches (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        flight_id TEXT,
                        rule_id INTEGER,
                        rule_name TEXT,
                        category TEXT,
                        severity REAL,
                        matched INTEGER,
                        summary TEXT,
                        details_json TEXT,
                        FOREIGN KEY (flight_id) REFERENCES anomaly_reports(flight_id)
                    )
                """)
                
                # Restore data from backup
                cursor.execute("""
                    INSERT INTO rule_matches (flight_id, rule_id, rule_name, category, severity, matched, summary)
                    SELECT migrated_flight_id, rule_id, rule_name, category, severity, matched, summary
                    FROM rule_matches_backup
                    WHERE migrated_flight_id IS NOT NULL
                """)
                
                # Drop backup
                cursor.execute("DROP TABLE rule_matches_backup")
                logger.info("Successfully migrated rule_matches table to new schema")
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not migrate rule_matches table: {e}")
        elif 'report_id' in cols and 'flight_id' not in cols:
            # Has report_id but no NOT NULL - just add flight_id
            try:
                cursor.execute("ALTER TABLE rule_matches ADD COLUMN flight_id TEXT")
                cursor.execute("""
                    UPDATE rule_matches 
                    SET flight_id = (
                        SELECT flight_id FROM anomaly_reports WHERE anomaly_reports.id = rule_matches.report_id
                    )
                """)
                logger.info("Migrated rule_matches table: added flight_id column")
            except sqlite3.OperationalError:
                pass
        
        # Refresh cols after potential migration
        cursor.execute("PRAGMA table_info(rule_matches)")
        cols = {row[1] for row in cursor.fetchall()}
        
        # Migrate: add details_json column if missing
        if 'details_json' not in cols:
            try:
                cursor.execute("ALTER TABLE rule_matches ADD COLUMN details_json TEXT")
                logger.info("Migrated rule_matches table: added details_json column")
            except sqlite3.OperationalError:
                pass
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rule_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flight_id TEXT,
                rule_id INTEGER,
                rule_name TEXT,
                category TEXT,
                severity REAL,
                matched INTEGER,
                summary TEXT,
                details_json TEXT,
                FOREIGN KEY (flight_id) REFERENCES anomaly_reports(flight_id)
            )
        """)
    
    # ================================================================
    # TABLE: ml_anomaly_points
    # Specific anomaly points detected by ML models
    # ================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_anomaly_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_id TEXT,
            model_name TEXT,
            timestamp INTEGER,
            lat REAL,
            lon REAL,
            point_score REAL,
            FOREIGN KEY (flight_id) REFERENCES anomaly_reports(flight_id)
        )
    """)
    
    # Ensure all columns exist (for migration from older schema)
    _migrate_schema(cursor)
    
    # Create indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_flight_id ON anomaly_reports (flight_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_tagged ON anomaly_reports (tagged)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_callsign ON anomaly_reports (callsign)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_origin ON anomaly_reports (origin_code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_dest ON anomaly_reports (dest_code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_airline ON anomaly_reports (airline)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_rule_id ON anomaly_reports (user_rule_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_confidence ON anomaly_reports (confidence_score)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_tagged_ts ON anomaly_reports (tagged_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_is_military ON anomaly_reports (is_military)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_emergency_squawk ON anomaly_reports (emergency_squawk_detected)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_crossed_borders ON anomaly_reports (crossed_borders)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ft_flight_id ON flight_tracks (flight_id)")
    # Only create index if flight_id column exists
    cursor.execute("PRAGMA table_info(rule_matches)")
    rm_cols = {row[1] for row in cursor.fetchall()}
    if 'flight_id' in rm_cols:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rm_flight_id ON rule_matches (flight_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mlap_flight_id ON ml_anomaly_points (flight_id)")
    
    conn.commit()
    conn.close()
    logger.info("Initialized present_anomalies.db with comprehensive schema")


def _migrate_schema(cursor: sqlite3.Cursor):
    """Migrate older schema to new schema by adding missing columns."""
    # Get existing columns
    cursor.execute("PRAGMA table_info(anomaly_reports)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    # New columns from research DB enhancement
    new_columns_from_research = [
        ("airline_code", "TEXT"),
        ("cruise_altitude_ft", "REAL"),
        ("emergency_squawk_detected", "INTEGER DEFAULT 0"),
        ("is_military", "INTEGER DEFAULT 0"),
        ("military_type", "TEXT"),
        ("flight_phase_summary", "TEXT"),
        ("crossed_borders", "TEXT"),
        ("signal_loss_events", "INTEGER DEFAULT 0"),
        ("data_quality_score", "REAL"),
    ]
    
    for col_name, col_type in new_columns_from_research:
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE anomaly_reports ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column '{col_name}' to anomaly_reports")
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not add column '{col_name}': {e}")
    
    # New columns to add
    new_columns = [
        ("tagged", "INTEGER DEFAULT 0"),
        ("tagged_timestamp", "INTEGER"),
        ("user_label", "INTEGER"),
        ("user_rule_id", "INTEGER"),
        ("user_comments", "TEXT"),
        ("user_other_details", "TEXT"),
        ("callsign", "TEXT"),
        ("flight_number", "TEXT"),
        ("origin_code", "TEXT"),
        ("origin_iata", "TEXT"),
        ("origin_icao", "TEXT"),
        ("origin_name", "TEXT"),
        ("origin_city", "TEXT"),
        ("origin_country", "TEXT"),
        ("dest_code", "TEXT"),
        ("dest_iata", "TEXT"),
        ("dest_icao", "TEXT"),
        ("dest_name", "TEXT"),
        ("dest_city", "TEXT"),
        ("dest_country", "TEXT"),
        ("airline", "TEXT"),
        ("aircraft_type", "TEXT"),
        ("aircraft_model", "TEXT"),
        ("aircraft_registration", "TEXT"),
        ("flight_status", "TEXT"),
        ("scheduled_departure", "TEXT"),
        ("scheduled_arrival", "TEXT"),
        ("actual_departure", "TEXT"),
        ("actual_arrival", "TEXT"),
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
        ("max_vspeed_fpm", "REAL"),
        ("min_vspeed_fpm", "REAL"),
        ("start_lat", "REAL"),
        ("start_lon", "REAL"),
        ("end_lat", "REAL"),
        ("end_lon", "REAL"),
        ("total_distance_nm", "REAL"),
        ("squawk_codes", "TEXT"),
        ("anomaly_timestamp", "INTEGER"),
        ("pipeline_is_anomaly", "INTEGER"),
        ("confidence_score", "REAL"),
        ("severity_cnn", "REAL"),
        ("severity_dense", "REAL"),
        ("severity_transformer", "REAL"),
        ("severity_hybrid", "REAL"),
        ("rules_status", "TEXT"),
        ("rules_triggers", "TEXT"),
        ("matched_rule_ids", "TEXT"),
        ("matched_rule_names", "TEXT"),
        ("matched_rule_categories", "TEXT"),
        ("ml_layers_flagged", "TEXT"),
        ("xgboost_is_anomaly", "INTEGER"),
        ("deep_dense_is_anomaly", "INTEGER"),
        ("deep_cnn_is_anomaly", "INTEGER"),
        ("transformer_is_anomaly", "INTEGER"),
        ("hybrid_is_anomaly", "INTEGER"),
        ("full_report_json", "TEXT"),
        ("created_at", "INTEGER"),
        ("updated_at", "INTEGER"),
        ("model_version", "TEXT"),
        ("is_military", "BOOLEAN DEFAULT 0"),
        ("geographic_region", "TEXT"),
        ("nearest_airport", "TEXT"),
    ]
    
    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE anomaly_reports ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column '{col_name}' to anomaly_reports")
            except sqlite3.OperationalError:
                pass


def compute_flight_summary(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comprehensive summary statistics from flight track points."""
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
        "max_vspeed_fpm": None,
        "min_vspeed_fpm": None,
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
    
    # Vertical speed stats
    vspeeds = [p.get("vspeed", 0) for p in sorted_points if p.get("vspeed") is not None]
    if vspeeds:
        summary["max_vspeed_fpm"] = max(vspeeds)
        summary["min_vspeed_fpm"] = min(vspeeds)
    
    # Start/End positions
    if sorted_points:
        first = sorted_points[0]
        last = sorted_points[-1]
        summary["start_lat"] = first.get("lat")
        summary["start_lon"] = first.get("lon")
        summary["end_lat"] = last.get("lat")
        summary["end_lon"] = last.get("lon")
    
    # Total distance (Haversine)
    try:
        total_dist = 0.0
        for i in range(1, len(sorted_points)):
            p1 = sorted_points[i - 1]
            p2 = sorted_points[i]
            lat1, lon1 = p1.get("lat"), p1.get("lon")
            lat2, lon2 = p2.get("lat"), p2.get("lon")
            if all(v is not None for v in [lat1, lon1, lat2, lon2]):
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


def extract_report_data(full_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract structured data from the full anomaly report."""
    data = {
        "anomaly_timestamp": None,
        "pipeline_is_anomaly": None,
        "confidence_score": None,
        "severity_cnn": None,
        "severity_dense": None,
        "severity_transformer": None,
        "severity_hybrid": None,
        "rules_status": None,
        "rules_triggers": None,
        "matched_rule_ids": None,
        "matched_rule_names": None,
        "matched_rule_categories": None,
        "ml_layers_flagged": None,
        "xgboost_is_anomaly": None,
        "deep_dense_is_anomaly": None,
        "deep_cnn_is_anomaly": None,
        "transformer_is_anomaly": None,
        "hybrid_is_anomaly": None,
    }
    
    if not full_report:
        return data
    
    # Summary
    summary = full_report.get("summary", {})
    data["pipeline_is_anomaly"] = 1 if summary.get("is_anomaly") else 0
    data["confidence_score"] = summary.get("confidence_score")
    data["anomaly_timestamp"] = full_report.get("timestamp")
    
    # Triggers
    triggers = summary.get("triggers", [])
    if triggers:
        data["rules_triggers"] = ", ".join(str(t) for t in triggers)
    
    # Rules layer
    rules_layer = full_report.get("layer_1_rules", {})
    if rules_layer:
        data["rules_status"] = rules_layer.get("status")
        matched_rules = rules_layer.get("report", {}).get("matched_rules", [])
        if matched_rules:
            rule_ids = [str(r.get("id")) for r in matched_rules if r.get("id")]
            rule_names = [r.get("name", "") for r in matched_rules if r.get("name")]
            rule_cats = [r.get("category", "") for r in matched_rules if r.get("category")]
            data["matched_rule_ids"] = ", ".join(rule_ids)
            data["matched_rule_names"] = ", ".join(rule_names)
            data["matched_rule_categories"] = ", ".join(set(rule_cats))
    
    # ML layers
    ml_flagged = []
    
    # XGBoost
    xgb = full_report.get("layer_2_xgboost", {})
    if xgb:
        data["xgboost_is_anomaly"] = 1 if xgb.get("is_anomaly") else 0
        if xgb.get("is_anomaly"):
            ml_flagged.append("XGBoost")
    
    # Deep Dense
    dense = full_report.get("layer_3_deep_dense", {})
    if dense:
        data["deep_dense_is_anomaly"] = 1 if dense.get("is_anomaly") else 0
        data["severity_dense"] = dense.get("severity")
        if dense.get("is_anomaly"):
            ml_flagged.append("Deep Dense")
    
    # Deep CNN
    cnn = full_report.get("layer_4_deep_cnn", {})
    if cnn:
        data["deep_cnn_is_anomaly"] = 1 if cnn.get("is_anomaly") else 0
        data["severity_cnn"] = cnn.get("severity")
        if cnn.get("is_anomaly"):
            ml_flagged.append("Deep CNN")
    
    # Transformer
    transformer = full_report.get("layer_5_transformer", {})
    if transformer:
        data["transformer_is_anomaly"] = 1 if transformer.get("is_anomaly") else 0
        data["severity_transformer"] = transformer.get("severity")
        if transformer.get("is_anomaly"):
            ml_flagged.append("Transformer")
    
    # Hybrid
    hybrid = full_report.get("layer_6_hybrid", {})
    if hybrid:
        data["hybrid_is_anomaly"] = 1 if hybrid.get("is_anomaly") else 0
        data["severity_hybrid"] = hybrid.get("severity")
        if hybrid.get("is_anomaly"):
            ml_flagged.append("Hybrid")
    
    if ml_flagged:
        data["ml_layers_flagged"] = json.dumps(ml_flagged)
    
    return data


def fetch_research_metadata(flight_id: str, research_db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive metadata from research DB for a flight.
    Returns dict with all enhanced fields from flight_metadata table.
    """
    if not research_db_path:
        # Try common locations
        possible_paths = [
            Path("realtime/research.db"),
            Path("realtime/research_2months.db"),
            Path("research.db"),
        ]
        for p in possible_paths:
            if p.exists():
                research_db_path = str(p)
                break
    
    if not research_db_path or not Path(research_db_path).exists():
        return None
    
    try:
        conn = sqlite3.connect(research_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                airline, airline_code, aircraft_type, aircraft_model, aircraft_registration,
                cruise_altitude_ft, emergency_squawk_detected, is_military, military_type,
                flight_phase_summary, crossed_borders, signal_loss_events, data_quality_score
            FROM flight_metadata 
            WHERE flight_id = ?
        """, (flight_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "airline": row[0],
                "airline_code": row[1],
                "aircraft_type": row[2],
                "aircraft_model": row[3],
                "aircraft_registration": row[4],
                "cruise_altitude_ft": row[5],
                "emergency_squawk_detected": row[6],
                "is_military": row[7],
                "military_type": row[8],
                "flight_phase_summary": row[9],
                "crossed_borders": row[10],
                "signal_loss_events": row[11],
                "data_quality_score": row[12],
            }
    except Exception as e:
        logger.warning(f"Could not fetch research metadata for {flight_id}: {e}")
    
    return None


def update_flight_record(
    flight_id: str,
    tagged: bool = True,
    rule_id: Optional[int] = None,
    comments: str = "",
    other_details: str = "",
    flight_details: Optional[Dict[str, Any]] = None,
    full_report: Optional[Dict[str, Any]] = None,
    points: Optional[List[Dict[str, Any]]] = None,
    research_metadata: Optional[Dict[str, Any]] = None,  # NEW: from research DB
    research_db_path: Optional[str] = None,  # Path to research DB
):
    """
    Update or insert a flight record in present_anomalies.db.
    This is the main function to call when tagging or re-analyzing a flight.
    
    Args:
        flight_id: Unique flight identifier
        tagged: Whether this is a user-tagged anomaly
        rule_id: Optional rule ID for categorization
        comments: User comments
        other_details: Additional user details
        flight_details: Flight info dict (origin, dest, airline, etc.)
        full_report: Complete anomaly analysis report
        points: Flight track points
        research_metadata: Pre-fetched research DB metadata (optional)
        research_db_path: Path to research DB if auto-fetch needed (optional)
    """
    init_present_db()
    
    # Auto-fetch research metadata if not provided
    if research_metadata is None and research_db_path:
        research_metadata = fetch_research_metadata(flight_id, research_db_path)
    
    conn = sqlite3.connect(str(PRESENT_DB_PATH))
    cursor = conn.cursor()
    
    now = int(datetime.now().timestamp())
    
    # Compute flight summary from points
    flight_summary = compute_flight_summary(points or [])
    
    # Extract report data
    report_data = extract_report_data(full_report)
    
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
    
    # Serialize full report
    report_json = None
    if full_report:
        try:
            report_json = json.dumps(full_report)
        except Exception:
            pass
    
    # Check if record exists
    cursor.execute("SELECT id FROM anomaly_reports WHERE flight_id = ?", (flight_id,))
    existing = cursor.fetchone()
    
    if existing:
        # UPDATE existing record
        cursor.execute("""
            UPDATE anomaly_reports SET
                tagged = ?,
                tagged_timestamp = ?,
                user_label = 1,
                user_rule_id = ?,
                user_comments = ?,
                user_other_details = ?,
                callsign = COALESCE(?, callsign),
                flight_number = COALESCE(?, flight_number),
                origin_code = COALESCE(?, origin_code),
                origin_iata = COALESCE(?, origin_iata),
                origin_icao = COALESCE(?, origin_icao),
                origin_name = COALESCE(?, origin_name),
                origin_city = COALESCE(?, origin_city),
                origin_country = COALESCE(?, origin_country),
                dest_code = COALESCE(?, dest_code),
                dest_iata = COALESCE(?, dest_iata),
                dest_icao = COALESCE(?, dest_icao),
                dest_name = COALESCE(?, dest_name),
                dest_city = COALESCE(?, dest_city),
                dest_country = COALESCE(?, dest_country),
                airline = COALESCE(?, airline),
                airline_code = COALESCE(?, airline_code),
                aircraft_type = COALESCE(?, aircraft_type),
                aircraft_model = COALESCE(?, aircraft_model),
                aircraft_registration = COALESCE(?, aircraft_registration),
                flight_status = COALESCE(?, flight_status),
                scheduled_departure = COALESCE(?, scheduled_departure),
                scheduled_arrival = COALESCE(?, scheduled_arrival),
                actual_departure = COALESCE(?, actual_departure),
                actual_arrival = COALESCE(?, actual_arrival),
                flight_start_ts = COALESCE(?, flight_start_ts),
                flight_end_ts = COALESCE(?, flight_end_ts),
                flight_duration_sec = COALESCE(?, flight_duration_sec),
                total_points = COALESCE(?, total_points),
                min_altitude_ft = COALESCE(?, min_altitude_ft),
                max_altitude_ft = COALESCE(?, max_altitude_ft),
                avg_altitude_ft = COALESCE(?, avg_altitude_ft),
                cruise_altitude_ft = COALESCE(?, cruise_altitude_ft),
                min_speed_kts = COALESCE(?, min_speed_kts),
                max_speed_kts = COALESCE(?, max_speed_kts),
                avg_speed_kts = COALESCE(?, avg_speed_kts),
                max_vspeed_fpm = COALESCE(?, max_vspeed_fpm),
                min_vspeed_fpm = COALESCE(?, min_vspeed_fpm),
                start_lat = COALESCE(?, start_lat),
                start_lon = COALESCE(?, start_lon),
                end_lat = COALESCE(?, end_lat),
                end_lon = COALESCE(?, end_lon),
                total_distance_nm = COALESCE(?, total_distance_nm),
                squawk_codes = COALESCE(?, squawk_codes),
                emergency_squawk_detected = COALESCE(?, emergency_squawk_detected),
                is_military = COALESCE(?, is_military),
                military_type = COALESCE(?, military_type),
                flight_phase_summary = COALESCE(?, flight_phase_summary),
                crossed_borders = COALESCE(?, crossed_borders),
                signal_loss_events = COALESCE(?, signal_loss_events),
                data_quality_score = COALESCE(?, data_quality_score),
                anomaly_timestamp = COALESCE(?, anomaly_timestamp),
                pipeline_is_anomaly = COALESCE(?, pipeline_is_anomaly),
                confidence_score = COALESCE(?, confidence_score),
                severity_cnn = COALESCE(?, severity_cnn),
                severity_dense = COALESCE(?, severity_dense),
                severity_transformer = COALESCE(?, severity_transformer),
                severity_hybrid = COALESCE(?, severity_hybrid),
                rules_status = COALESCE(?, rules_status),
                rules_triggers = COALESCE(?, rules_triggers),
                matched_rule_ids = COALESCE(?, matched_rule_ids),
                matched_rule_names = COALESCE(?, matched_rule_names),
                matched_rule_categories = COALESCE(?, matched_rule_categories),
                ml_layers_flagged = COALESCE(?, ml_layers_flagged),
                xgboost_is_anomaly = COALESCE(?, xgboost_is_anomaly),
                deep_dense_is_anomaly = COALESCE(?, deep_dense_is_anomaly),
                deep_cnn_is_anomaly = COALESCE(?, deep_cnn_is_anomaly),
                transformer_is_anomaly = COALESCE(?, transformer_is_anomaly),
                hybrid_is_anomaly = COALESCE(?, hybrid_is_anomaly),
                full_report_json = COALESCE(?, full_report_json),
                updated_at = ?,
                model_version = 'v2'
            WHERE flight_id = ?
        """, (
            1 if tagged else 0,
            now if tagged else None,
            rule_id,
            comments,
            other_details,
            callsign,
            fd.get("flight_number"),
            origin.get("code") if isinstance(origin, dict) else None,
            origin.get("iata") if isinstance(origin, dict) else None,
            origin.get("icao") if isinstance(origin, dict) else None,
            origin.get("name") if isinstance(origin, dict) else None,
            origin.get("city") if isinstance(origin, dict) else None,
            origin.get("country") if isinstance(origin, dict) else None,
            destination.get("code") if isinstance(destination, dict) else None,
            destination.get("iata") if isinstance(destination, dict) else None,
            destination.get("icao") if isinstance(destination, dict) else None,
            destination.get("name") if isinstance(destination, dict) else None,
            destination.get("city") if isinstance(destination, dict) else None,
            destination.get("country") if isinstance(destination, dict) else None,
            fd.get("airline") or (research_metadata.get("airline") if research_metadata else None),
            research_metadata.get("airline_code") if research_metadata else None,
            fd.get("aircraft_type") or (research_metadata.get("aircraft_type") if research_metadata else None),
            fd.get("aircraft_model") or (research_metadata.get("aircraft_model") if research_metadata else None),
            fd.get("aircraft_registration") or (research_metadata.get("aircraft_registration") if research_metadata else None),
            fd.get("status"),
            fd.get("scheduled_departure"),
            fd.get("scheduled_arrival"),
            fd.get("actual_departure"),
            fd.get("actual_arrival"),
            flight_summary["flight_start_ts"],
            flight_summary["flight_end_ts"],
            flight_summary["flight_duration_sec"],
            flight_summary["total_points"],
            flight_summary["min_altitude_ft"],
            flight_summary["max_altitude_ft"],
            flight_summary["avg_altitude_ft"],
            research_metadata.get("cruise_altitude_ft") if research_metadata else None,
            flight_summary["min_speed_kts"],
            flight_summary["max_speed_kts"],
            flight_summary["avg_speed_kts"],
            flight_summary["max_vspeed_fpm"],
            flight_summary["min_vspeed_fpm"],
            flight_summary["start_lat"],
            flight_summary["start_lon"],
            flight_summary["end_lat"],
            flight_summary["end_lon"],
            flight_summary["total_distance_nm"],
            flight_summary["squawk_codes"],
            research_metadata.get("emergency_squawk_detected") if research_metadata else 0,
            research_metadata.get("is_military") if research_metadata else 0,
            research_metadata.get("military_type") if research_metadata else None,
            research_metadata.get("flight_phase_summary") if research_metadata else None,
            research_metadata.get("crossed_borders") if research_metadata else None,
            research_metadata.get("signal_loss_events") if research_metadata else 0,
            research_metadata.get("data_quality_score") if research_metadata else None,
            report_data["anomaly_timestamp"],
            report_data["pipeline_is_anomaly"],
            report_data["confidence_score"],
            report_data["severity_cnn"],
            report_data["severity_dense"],
            report_data["severity_transformer"],
            report_data["severity_hybrid"],
            report_data["rules_status"],
            report_data["rules_triggers"],
            report_data["matched_rule_ids"],
            report_data["matched_rule_names"],
            report_data["matched_rule_categories"],
            report_data["ml_layers_flagged"],
            report_data["xgboost_is_anomaly"],
            report_data["deep_dense_is_anomaly"],
            report_data["deep_cnn_is_anomaly"],
            report_data["transformer_is_anomaly"],
            report_data["hybrid_is_anomaly"],
            report_json,
            now,
            flight_id
        ))
        logger.info(f"Updated flight record for {flight_id} in present_anomalies.db")
    else:
        # INSERT new record
        cursor.execute("""
            INSERT INTO anomaly_reports (
                flight_id, tagged, tagged_timestamp, user_label, user_rule_id, user_comments, user_other_details,
                callsign, flight_number,
                origin_code, origin_iata, origin_icao, origin_name, origin_city, origin_country,
                dest_code, dest_iata, dest_icao, dest_name, dest_city, dest_country,
                airline, airline_code, aircraft_type, aircraft_model, aircraft_registration,
                flight_status, scheduled_departure, scheduled_arrival, actual_departure, actual_arrival,
                flight_start_ts, flight_end_ts, flight_duration_sec, total_points,
                min_altitude_ft, max_altitude_ft, avg_altitude_ft, cruise_altitude_ft,
                min_speed_kts, max_speed_kts, avg_speed_kts,
                max_vspeed_fpm, min_vspeed_fpm,
                start_lat, start_lon, end_lat, end_lon, total_distance_nm, squawk_codes,
                emergency_squawk_detected, is_military, military_type, flight_phase_summary,
                crossed_borders, signal_loss_events, data_quality_score,
                anomaly_timestamp, pipeline_is_anomaly, confidence_score,
                severity_cnn, severity_dense, severity_transformer, severity_hybrid,
                rules_status, rules_triggers, matched_rule_ids, matched_rule_names, matched_rule_categories,
                ml_layers_flagged, xgboost_is_anomaly, deep_dense_is_anomaly, deep_cnn_is_anomaly, transformer_is_anomaly, hybrid_is_anomaly,
                full_report_json, created_at, updated_at, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            flight_id,
            1 if tagged else 0,
            now if tagged else None,
            1,  # user_label = anomaly
            rule_id,
            comments,
            other_details,
            callsign,
            fd.get("flight_number"),
            origin.get("code") if isinstance(origin, dict) else None,
            origin.get("iata") if isinstance(origin, dict) else None,
            origin.get("icao") if isinstance(origin, dict) else None,
            origin.get("name") if isinstance(origin, dict) else None,
            origin.get("city") if isinstance(origin, dict) else None,
            origin.get("country") if isinstance(origin, dict) else None,
            destination.get("code") if isinstance(destination, dict) else None,
            destination.get("iata") if isinstance(destination, dict) else None,
            destination.get("icao") if isinstance(destination, dict) else None,
            destination.get("name") if isinstance(destination, dict) else None,
            destination.get("city") if isinstance(destination, dict) else None,
            destination.get("country") if isinstance(destination, dict) else None,
            fd.get("airline") or (research_metadata.get("airline") if research_metadata else None),
            research_metadata.get("airline_code") if research_metadata else None,
            fd.get("aircraft_type") or (research_metadata.get("aircraft_type") if research_metadata else None),
            fd.get("aircraft_model") or (research_metadata.get("aircraft_model") if research_metadata else None),
            fd.get("aircraft_registration") or (research_metadata.get("aircraft_registration") if research_metadata else None),
            fd.get("status"),
            fd.get("scheduled_departure"),
            fd.get("scheduled_arrival"),
            fd.get("actual_departure"),
            fd.get("actual_arrival"),
            flight_summary["flight_start_ts"],
            flight_summary["flight_end_ts"],
            flight_summary["flight_duration_sec"],
            flight_summary["total_points"],
            flight_summary["min_altitude_ft"],
            flight_summary["max_altitude_ft"],
            flight_summary["avg_altitude_ft"],
            research_metadata.get("cruise_altitude_ft") if research_metadata else None,
            flight_summary["min_speed_kts"],
            flight_summary["max_speed_kts"],
            flight_summary["avg_speed_kts"],
            flight_summary["max_vspeed_fpm"],
            flight_summary["min_vspeed_fpm"],
            flight_summary["start_lat"],
            flight_summary["start_lon"],
            flight_summary["end_lat"],
            flight_summary["end_lon"],
            flight_summary["total_distance_nm"],
            flight_summary["squawk_codes"],
            research_metadata.get("emergency_squawk_detected") if research_metadata else 0,
            research_metadata.get("is_military") if research_metadata else 0,
            research_metadata.get("military_type") if research_metadata else None,
            research_metadata.get("flight_phase_summary") if research_metadata else None,
            research_metadata.get("crossed_borders") if research_metadata else None,
            research_metadata.get("signal_loss_events") if research_metadata else 0,
            research_metadata.get("data_quality_score") if research_metadata else None,
            report_data["anomaly_timestamp"],
            report_data["pipeline_is_anomaly"],
            report_data["confidence_score"],
            report_data["severity_cnn"],
            report_data["severity_dense"],
            report_data["severity_transformer"],
            report_data["severity_hybrid"],
            report_data["rules_status"],
            report_data["rules_triggers"],
            report_data["matched_rule_ids"],
            report_data["matched_rule_names"],
            report_data["matched_rule_categories"],
            report_data["ml_layers_flagged"],
            report_data["xgboost_is_anomaly"],
            report_data["deep_dense_is_anomaly"],
            report_data["deep_cnn_is_anomaly"],
            report_data["transformer_is_anomaly"],
            report_data["hybrid_is_anomaly"],
            report_json,
            now,
            now,
            "v2"
        ))
        logger.info(f"Inserted new flight record for {flight_id} in present_anomalies.db")
    
    # Update ML anomaly points if we have a report
    if full_report:
        _update_ml_anomaly_points(cursor, flight_id, full_report)
    
    # Update rule matches if we have a report
    if full_report:
        _update_rule_matches(cursor, flight_id, full_report)

    # Update flight tracks (CRITICAL for feedback history lookup)
    if points:
        _update_flight_tracks(cursor, flight_id, points)
    
    conn.commit()
    conn.close()
    
    return True


def _update_flight_tracks(cursor: sqlite3.Cursor, flight_id: str, points: List[Dict[str, Any]]):
    """Update flight tracks table."""
    # Delete existing tracks for this flight to avoid duplicates
    cursor.execute("DELETE FROM flight_tracks WHERE flight_id = ?", (flight_id,))
    
    if not points:
        return

    # Prepare batch insert
    data = []
    for p in points:
        data.append((
            flight_id,
            p.get("timestamp"),
            p.get("lat"),
            p.get("lon"),
            p.get("alt"),
            p.get("heading"),
            p.get("gspeed"),
            p.get("vspeed"),
            p.get("track"),
            str(p.get("squawk")) if p.get("squawk") is not None else None,
            p.get("callsign"),
            p.get("source")
        ))
        
    try:
        cursor.executemany("""
            INSERT INTO flight_tracks (
                flight_id, timestamp, lat, lon, alt, heading, gspeed, vspeed, track, squawk, callsign, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        logger.info(f"Inserted {len(data)} track points for {flight_id} into present_anomalies.db")
    except Exception as e:
        logger.error(f"Failed to insert track points for {flight_id}: {e}")


def _update_ml_anomaly_points(cursor: sqlite3.Cursor, flight_id: str, full_report: Dict[str, Any]):
    """Update ML anomaly points table from the full report."""
    # Delete existing points for this flight
    cursor.execute("DELETE FROM ml_anomaly_points WHERE flight_id = ?", (flight_id,))
    
    ml_layers = [
        ("layer_3_deep_dense", "Deep Dense"),
        ("layer_4_deep_cnn", "Deep CNN"),
        ("layer_5_transformer", "Transformer"),
        ("layer_6_hybrid", "Hybrid"),
    ]
    
    for key, name in ml_layers:
        layer_data = full_report.get(key, {})
        if layer_data and layer_data.get("anomaly_points"):
            for pt in layer_data["anomaly_points"]:
                cursor.execute("""
                    INSERT INTO ml_anomaly_points (flight_id, model_name, timestamp, lat, lon, point_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    flight_id,
                    name,
                    pt.get("timestamp"),
                    pt.get("lat"),
                    pt.get("lon"),
                    pt.get("point_score")
                ))


def _update_rule_matches(cursor: sqlite3.Cursor, flight_id: str, full_report: Dict[str, Any]):
    """Update rule matches table from the full report."""
    # Delete existing matches for this flight
    cursor.execute("DELETE FROM rule_matches WHERE flight_id = ?", (flight_id,))
    
    # Get matched rules from layer 1
    rules_layer = full_report.get("layer_1_rules", {})
    if not rules_layer:
        return
    
    matched_rules = rules_layer.get("report", {}).get("matched_rules", [])
    for rule in matched_rules:
        cursor.execute("""
            INSERT INTO rule_matches (flight_id, rule_id, rule_name, category, severity, matched, summary, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            flight_id,
            rule.get("id"),
            rule.get("name"),
            rule.get("category"),
            rule.get("severity"),
            1,
            rule.get("summary"),
            json.dumps(rule.get("details")) if rule.get("details") else None
        ))


def remove_flight_record(flight_id: str):
    """Remove a flight record from present_anomalies.db (when user un-tags it)."""
    if not PRESENT_DB_PATH.exists():
        return
    
    conn = sqlite3.connect(str(PRESENT_DB_PATH))
    cursor = conn.cursor()
    
    # Delete from all tables
    cursor.execute("DELETE FROM anomaly_reports WHERE flight_id = ?", (flight_id,))
    cursor.execute("DELETE FROM flight_tracks WHERE flight_id = ?", (flight_id,))
    cursor.execute("DELETE FROM rule_matches WHERE flight_id = ?", (flight_id,))
    cursor.execute("DELETE FROM ml_anomaly_points WHERE flight_id = ?", (flight_id,))
    
    conn.commit()
    conn.close()
    logger.info(f"Removed flight {flight_id} from present_anomalies.db")


def get_tagged_flights(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Get all tagged flights from present_anomalies.db."""
    if not PRESENT_DB_PATH.exists():
        return []
    
    conn = sqlite3.connect(str(PRESENT_DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM anomaly_reports 
        WHERE tagged = 1 
        ORDER BY tagged_timestamp DESC 
        LIMIT ? OFFSET ?
    """, (limit, offset))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

