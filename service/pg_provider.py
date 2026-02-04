"""
PostgreSQL Data Provider

Provides functions to query PostgreSQL database for flight data, anomalies, and feedback.
Replaces SQLite queries with PostgreSQL equivalents.

Uses singleton pattern for connection pool management to prevent resource exhaustion.
"""

import psycopg2
import psycopg2.extras
from psycopg2 import pool
import json
import logging
import os
import threading
import atexit
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLETON CONNECTION POOL MANAGER
# ============================================================================

class PostgreSQLConnectionPool:
    """
    Singleton connection pool manager for PostgreSQL.
    
    Ensures only one connection pool is created across the application lifecycle.
    Thread-safe implementation with automatic cleanup on exit.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    # PostgreSQL connection - loaded from environment variables
    PG_DSN = os.getenv("POSTGRES_DSN")
    if not PG_DSN:
        raise ValueError("POSTGRES_DSN environment variable is required")
    
    # Pool configuration - loaded from environment or defaults
    # LOW traffic: MIN=2, MAX=5
    # MEDIUM traffic: MIN=5, MAX=15  
    # HIGH traffic: MIN=10, MAX=30
    MIN_CONNECTIONS = int(os.getenv("PG_POOL_MIN_CONNECTIONS", "2"))
    MAX_CONNECTIONS = int(os.getenv("PG_POOL_MAX_CONNECTIONS", "10"))
    
    def __new__(cls):
        """Singleton pattern - only create one instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(PostgreSQLConnectionPool, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the connection pool (only once)."""
        # Check if already initialized (for singleton pattern)
        if self.__class__._initialized:
            return
        
        with self._lock:
            # Double-check after acquiring lock
            if self.__class__._initialized:
                return
            
            try:
                connect_timeout = int(os.getenv("PG_CONNECT_TIMEOUT", "10"))
                statement_timeout = int(os.getenv("PG_STATEMENT_TIMEOUT", "30000"))
                
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    self.MIN_CONNECTIONS,
                    self.MAX_CONNECTIONS,
                    self.PG_DSN,
                    # Connection pool settings
                    connect_timeout=connect_timeout,
                    options=f'-c statement_timeout={statement_timeout}'
                )
                self.__class__._initialized = True
                logger.info(f"PostgreSQL connection pool initialized (min={self.MIN_CONNECTIONS}, max={self.MAX_CONNECTIONS})")
                
                # Register cleanup on exit
                atexit.register(self.close_all)
                
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL pool: {e}")
                raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager to get a connection from the pool.
        
        Ensures connections are always returned to the pool, even on exceptions.
        Validates connection health before returning.
        """
        if not self.__class__._initialized:
            raise RuntimeError("Connection pool not initialized")
        
        conn = None
        try:
            # Get connection from pool
            conn = self._pool.getconn()
            
            # Validate connection is alive
            if conn.closed:
                logger.warning("Got closed connection from pool, getting new one")
                self._pool.putconn(conn, close=True)
                conn = self._pool.getconn()
            
            # Test connection with simple query
            try:
                with conn.cursor() as test_cursor:
                    test_cursor.execute("SELECT 1")
            except Exception as e:
                logger.warning(f"Connection health check failed, reconnecting: {e}")
                self._pool.putconn(conn, close=True)
                conn = self._pool.getconn()
            
            yield conn
            
        except psycopg2.OperationalError as e:
            logger.error(f"PostgreSQL operational error: {e}")
            if conn:
                self._pool.putconn(conn, close=True)
            raise
        except Exception as e:
            logger.error(f"Error with PostgreSQL connection: {e}")
            raise
        finally:
            # Always return connection to pool
            if conn is not None:
                try:
                    # Rollback any uncommitted transactions
                    if not conn.closed:
                        conn.rollback()
                    self._pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    try:
                        self._pool.putconn(conn, close=True)
                    except:
                        pass
    
    def close_all(self):
        """Close all connections in the pool."""
        if hasattr(self, '_pool') and self._pool:
            try:
                self._pool.closeall()
            except Exception as e:
                try:
                    logger.error(f"Error closing connection pool: {e}")
                except (ValueError, OSError):
                    pass
            finally:
                self.__class__._initialized = False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status for monitoring."""
        if not self.__class__._initialized or not hasattr(self, '_pool'):
            return {"status": "not_initialized"}
        
        # Note: ThreadedConnectionPool doesn't expose these directly
        # This is a basic status check
        return {
            "status": "active",
            "initialized": self.__class__._initialized,
            "dsn": self.PG_DSN.split('@')[1] if '@' in self.PG_DSN else "hidden"
        }


# ============================================================================
# GLOBAL POOL INSTANCE (Singleton)
# ============================================================================

_pool_instance = None
_pool_lock = threading.Lock()


def get_pool() -> PostgreSQLConnectionPool:
    """Get the singleton connection pool instance."""
    global _pool_instance
    
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                _pool_instance = PostgreSQLConnectionPool()
    
    return _pool_instance


@contextmanager
def get_connection():
    """
    Get a database connection from the pool.
    
    Usage:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM table")
    """
    pool = get_pool()
    with pool.get_connection() as conn:
        yield conn


def close_pool():
    """Close the connection pool (for cleanup/testing)."""
    global _pool_instance
    if _pool_instance:
        _pool_instance.close_all()
        with _pool_lock:
            _pool_instance = None


def get_pool_status() -> Dict[str, Any]:
    """Get pool status for monitoring/health checks."""
    try:
        pool = get_pool()
        return pool.get_pool_status()
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================================
# MONITOR STATUS CONTROL (replaces subprocess management)
# ============================================================================

def get_monitor_status() -> Dict[str, Any]:
    """
    Get current monitor status from database.
    
    Returns:
        Dict with keys: is_active, last_update_time, started_by, stopped_by
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT is_active, last_update_time, started_by, stopped_by
                    FROM public.monitor_status
                    WHERE id = 1
                """)
                result = cursor.fetchone()
                if result:
                    return dict(result)
                else:
                    # Table doesn't exist or no row - return default
                    return {
                        "is_active": False,
                        "last_update_time": None,
                        "started_by": None,
                        "stopped_by": None
                    }
    except Exception as e:
        logger.error(f"Failed to get monitor status: {e}")
        return {
            "is_active": False,
            "last_update_time": None,
            "error": str(e)
        }


def set_monitor_active(is_active: bool, user: str = "api") -> bool:
    """
    Set monitor active/inactive state.
    
    Args:
        is_active: True to activate, False to deactivate
        user: Who activated/deactivated the monitor
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                if is_active:
                    cursor.execute("""
                        UPDATE public.monitor_status
                        SET is_active = true,
                            last_update_time = NOW(),
                            started_by = %s
                        WHERE id = 1
                    """, (user,))
                else:
                    cursor.execute("""
                        UPDATE public.monitor_status
                        SET is_active = false,
                            last_update_time = NOW(),
                            stopped_by = %s
                        WHERE id = 1
                    """, (user,))
                
                conn.commit()
                logger.info(f"Monitor status set to {'active' if is_active else 'inactive'} by {user}")
                return True
    except Exception as e:
        logger.error(f"Failed to set monitor status: {e}")
        return False


def update_monitor_heartbeat() -> bool:
    """
    Update monitor heartbeat timestamp.
    Called by monitor.py to indicate it's alive and processing.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE public.monitor_status
                    SET last_update_time = NOW()
                    WHERE id = 1
                """)
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to update monitor heartbeat: {e}")
        return False

# ============================================================================
# FEEDBACK / TAGGED HISTORY
# ============================================================================

def get_tagged_feedback_history(
    start_ts: int = 0,
    end_ts: Optional[int] = None,
    limit: int = 100,
    include_normal: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch tagged flights from PostgreSQL feedback schema.
    
    Queries:
    - feedback.user_feedback
    - feedback.flight_metadata
    - feedback.anomaly_reports
    - feedback.normal_tracks (for count)
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp
        limit: Maximum number of results
        include_normal: If True, includes flights tagged as normal (user_label=0)
    
    Returns:
        List of flight records with metadata and anomaly reports
    """
    if end_ts is None:
        end_ts = int(datetime.now().timestamp())
    
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Build label filter
                label_filter = "" if include_normal else "AND uf.user_label = 1"
                
                # Query matching the SQLite version but for PostgreSQL
                query = f"""
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
                        uf.rule_ids,
                        uf.rule_names,
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
                        (SELECT COUNT(*) FROM feedback.flight_tracks ft WHERE ft.flight_id = uf.flight_id) as track_count
                    FROM feedback.user_feedback uf
                    LEFT JOIN feedback.flight_metadata fm ON uf.flight_id = fm.flight_id
                    LEFT JOIN feedback.anomaly_reports ar ON uf.flight_id = ar.flight_id
                    WHERE COALESCE(uf.first_seen_ts, uf.tagged_at) BETWEEN %s AND %s
                      {label_filter}
                    ORDER BY COALESCE(uf.first_seen_ts, uf.tagged_at) DESC
                    LIMIT %s
                """
                
                cursor.execute(query, (start_ts, end_ts, limit))
                rows = cursor.fetchall()
                
                result = []
                for row in rows:
                    # Parse full_report if it's JSON text
                    full_report = row['full_report']
                    if isinstance(full_report, str):
                        try:
                            full_report = json.loads(full_report)
                        except Exception as e:
                            logger.error(f"Error parsing full_report: {e}")
                            pass
                    
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
                        'rule_ids': row.get('rule_ids'),
                        'rule_names': row.get('rule_names'),
                        'comments': row['comments'],
                        'other_details': row['other_details'],
                        'callsign': row['callsign'],
                        'flight_number': row['flight_number'],
                        'airline': row['airline'],
                        'origin_airport': row['origin_airport'],
                        'destination_airport': row['destination_airport'],
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
        logger.error(f"Error fetching tagged feedback from PostgreSQL: {e}", exc_info=True)
        return []


def get_flight_track(flight_id: str, schema: str = 'feedback') -> List[Dict[str, Any]]:
    """
    Get flight track points from PostgreSQL.
    
    Args:
        flight_id: Flight ID to fetch
        schema: Schema to query (feedback, research, live)
    
    Returns:
        List of track points sorted by timestamp
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # List of tables to try, in order of preference
                tables_to_try = []
                
                if schema == 'feedback':
                    # Feedback schema uses flight_tracks as the main table
                    tables_to_try = ['flight_tracks', 'anomalies_tracks', 'normal_tracks']
                elif schema == 'live':
                    # Live schema uses normal_tracks primarily
                    tables_to_try = ['normal_tracks', 'anomalies_tracks']
                else:
                    # Research and other schemas
                    tables_to_try = ['anomalies_tracks', 'normal_tracks']
                
                # Try each table in order
                for table in tables_to_try:
                    try:
                        cursor.execute(
                            f"""
                            SELECT 
                                flight_id, timestamp, lat, lon, alt, gspeed, vspeed, 
                                track, squawk, callsign, source
                            FROM {schema}.{table}
                            WHERE flight_id = %s
                            ORDER BY timestamp
                            """,
                            (flight_id,)
                        )
                        rows = cursor.fetchall()
                        if rows:
                            logger.debug(f"Found {len(rows)} track points in {schema}.{table} for {flight_id}")
                            return [dict(row) for row in rows]
                    except psycopg2.errors.UndefinedTable:
                        # Table doesn't exist in this schema, try next one
                        logger.debug(f"Table {schema}.{table} does not exist, trying next table")
                        conn.rollback()
                        continue
                    except Exception as e:
                        # Other error, log and try next table
                        logger.debug(f"Error querying {schema}.{table}: {e}")
                        conn.rollback()
                        continue
                
                # No tracks found in any table
                return []
                
    except Exception as e:
        logger.error(f"Error fetching flight track from PostgreSQL: {e}", exc_info=True)
        return []


def get_flight_metadata(flight_id: str, schema: str = 'feedback') -> Optional[Dict[str, Any]]:
    """
    Get flight metadata from PostgreSQL.
    
    Args:
        flight_id: Flight ID to fetch
        schema: Schema to query (feedback, research, live)
    
    Returns:
        Flight metadata dict or None
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM {schema}.flight_metadata
                    WHERE flight_id = %s
                    """,
                    (flight_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
                
    except Exception as e:
        logger.error(f"Error fetching flight metadata from PostgreSQL: {e}", exc_info=True)
        return None


def get_unified_track(flight_id: str) -> Optional[Dict[str, Any]]:
    """
    Get flight track from feedback schema in PostgreSQL.
    
    Args:
        flight_id: Flight ID to fetch
    
    Returns:
        Dict with flight_id and points, or None if not found
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT 
                        flight_id, timestamp, lat, lon, alt, 
                        gspeed, vspeed, track, squawk, callsign, source
                    FROM feedback.flight_tracks
                    WHERE flight_id = %s
                    ORDER BY timestamp ASC
                    """,
                    (flight_id,)
                )
                rows = cursor.fetchall()
                
                if rows:
                    logger.info(f"Unified track - Found {len(rows)} points in feedback.flight_tracks for {flight_id}")
                    return {
                        "flight_id": flight_id,
                        "points": [dict(row) for row in rows]
                    }
                
                return None
                
    except Exception as e:
        logger.error(f"Error fetching unified track from PostgreSQL: {e}", exc_info=True)
        return None


# ============================================================================
# RESEARCH DATA
# ============================================================================

def get_research_anomalies(
    start_ts: int,
    end_ts: int,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fetch anomalies from research schema.
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp
        limit: Maximum results
    
    Returns:
        List of anomaly reports with metadata
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT 
                        ar.*,
                        fm.callsign,
                        fm.flight_number,
                        fm.airline,
                        fm.origin_airport,
                        fm.destination_airport,
                        fm.aircraft_type,
                        fm.total_points
                    FROM research.anomaly_reports ar
                    LEFT JOIN research.flight_metadata fm ON ar.flight_id = fm.flight_id
                    WHERE ar.timestamp BETWEEN %s AND %s
                      AND ar.is_anomaly = true
                    ORDER BY ar.timestamp DESC
                    LIMIT %s
                    """,
                    (start_ts, end_ts, limit)
                )
                rows = cursor.fetchall()
                
                result = []
                for row in rows:
                    full_report = row['full_report']
                    if isinstance(full_report, str):
                        try:
                            full_report = json.loads(full_report)
                        except Exception as e:
                            logger.error(f"Error parsing full_report: {e}")
                    
                    result.append(dict(row))
                return result
                
    except Exception as e:
        logger.error(f"Error fetching research anomalies from PostgreSQL: {e}", exc_info=True)
        return []


# ============================================================================
# HEALTH CHECK
# ============================================================================

def test_connection() -> bool:
    """Test PostgreSQL connection."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone()[0] == 1
    except Exception as e:
        logger.error(f"PostgreSQL connection test failed: {e}")
        return False


# ============================================================================
# LIVE RESEARCH DATA (Real-time monitoring)
# ============================================================================

def get_all_live_flights(cutoff_minutes: int = 15) -> Dict[str, Any]:
    """
    Get all currently active flights from live_research schema.
    
    Args:
        cutoff_minutes: Only return flights seen within this many minutes
    
    Returns:
        Dict with flights list, anomaly_count, and total_count
    """
    import time
    
    try:
        cutoff_ts = int(time.time()) - (cutoff_minutes * 60)
        
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Get all active flights
                cursor.execute(
                    """
                    SELECT 
                        flight_id, callsign, airline, origin_airport, destination_airport,
                        end_lat, end_lon, max_altitude_ft, avg_speed_kts,
                        is_anomaly, is_military, last_seen_ts,
                        aircraft_type, aircraft_registration, flight_number, category
                    FROM live.flight_metadata 
                    WHERE last_seen_ts > %s
                    ORDER BY last_seen_ts DESC
                    """,
                    (cutoff_ts,)
                )
                rows = cursor.fetchall()
                
                flights = []
                anomaly_count = 0
                
                for row in rows:
                    flight_id = row["flight_id"]
                    
                    # Get most recent track point for accurate position and heading
                    cursor.execute(
                        """
                        SELECT lat, lon, alt, track, gspeed
                        FROM live.normal_tracks
                        WHERE flight_id = %s
                        ORDER BY timestamp DESC LIMIT 1
                        """,
                        (flight_id,)
                    )
                    track_row = cursor.fetchone()
                    
                    # Use track data if available, otherwise metadata
                    lat = track_row["lat"] if track_row else row["end_lat"]
                    lon = track_row["lon"] if track_row else row["end_lon"]
                    alt = track_row["alt"] if track_row else row["max_altitude_ft"]
                    heading = track_row["track"] if track_row else 0
                    speed = track_row["gspeed"] if track_row else row["avg_speed_kts"]
                    
                    is_anomaly = bool(row["is_anomaly"])
                    if is_anomaly:
                        anomaly_count += 1
                    
                    # Get severity if anomaly
                    severity = 0.0
                    if is_anomaly:
                        cursor.execute(
                            """
                            SELECT severity_cnn FROM live.anomaly_reports 
                            WHERE flight_id = %s LIMIT 1
                            """,
                            (flight_id,)
                        )
                        sev_row = cursor.fetchone()
                        if sev_row and sev_row["severity_cnn"]:
                            severity = sev_row["severity_cnn"]
                    
                    flights.append({
                        "flight_id": flight_id,
                        "callsign": row["callsign"],
                        "airline": row["airline"],
                        "origin": row["origin_airport"],
                        "destination": row["destination_airport"],
                        "lat": lat,
                        "lon": lon,
                        "alt": alt or 0,
                        "heading": heading or 0,
                        "speed": speed or 0,
                        "is_anomaly": is_anomaly,
                        "is_military": bool(row["is_military"]),
                        "severity": severity,
                        "last_seen_ts": row["last_seen_ts"],
                        "aircraft_type": row["aircraft_type"],
                        "aircraft_registration": row["aircraft_registration"],
                        "flight_number": row["flight_number"],
                        "category": row["category"],
                    })
                
                return {
                    "flights": flights,
                    "anomaly_count": anomaly_count,
                    "total_count": len(flights),
                }
                
    except Exception as e:
        logger.error(f"Error fetching live flights from PostgreSQL: {e}", exc_info=True)
        return {"flights": [], "anomaly_count": 0, "total_count": 0}


def save_flight_metadata(metadata: Dict[str, Any], schema: str = 'live') -> bool:
    """
    Save or update flight metadata in PostgreSQL.
    
    Args:
        metadata: Flight metadata dict
        schema: Schema to save to (live_research, research_new, feedback)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.flight_metadata (
                        flight_id, callsign, flight_number, airline,
                        origin_airport, destination_airport,
                        aircraft_type, aircraft_registration, category,
                        first_seen_ts, last_seen_ts,
                        start_lat, start_lon, end_lat, end_lon,
                        min_altitude_ft, max_altitude_ft, avg_speed_kts,
                        total_distance_nm, flight_duration_sec, total_points,
                        is_anomaly, is_military
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (flight_id, first_seen_ts) DO UPDATE SET
                        callsign = EXCLUDED.callsign,
                        flight_number = EXCLUDED.flight_number,
                        airline = EXCLUDED.airline,
                        origin_airport = EXCLUDED.origin_airport,
                        destination_airport = EXCLUDED.destination_airport,
                        aircraft_type = EXCLUDED.aircraft_type,
                        aircraft_registration = EXCLUDED.aircraft_registration,
                        category = EXCLUDED.category,
                        last_seen_ts = EXCLUDED.last_seen_ts,
                        end_lat = EXCLUDED.end_lat,
                        end_lon = EXCLUDED.end_lon,
                        max_altitude_ft = EXCLUDED.max_altitude_ft,
                        avg_speed_kts = EXCLUDED.avg_speed_kts,
                        total_distance_nm = EXCLUDED.total_distance_nm,
                        flight_duration_sec = EXCLUDED.flight_duration_sec,
                        total_points = EXCLUDED.total_points,
                        is_anomaly = EXCLUDED.is_anomaly,
                        is_military = EXCLUDED.is_military
                    """,
                    (
                        metadata.get('flight_id'),
                        metadata.get('callsign'),
                        metadata.get('flight_number'),
                        metadata.get('airline'),
                        metadata.get('origin_airport'),
                        metadata.get('destination_airport'),
                        metadata.get('aircraft_type'),
                        metadata.get('aircraft_registration'),
                        metadata.get('category'),
                        metadata.get('first_seen_ts'),
                        metadata.get('last_seen_ts'),
                        metadata.get('start_lat'),
                        metadata.get('start_lon'),
                        metadata.get('end_lat'),
                        metadata.get('end_lon'),
                        metadata.get('min_altitude_ft'),
                        metadata.get('max_altitude_ft'),
                        metadata.get('avg_speed_kts'),
                        metadata.get('total_distance_nm'),
                        metadata.get('flight_duration_sec'),
                        metadata.get('total_points'),
                        metadata.get('is_anomaly', False),
                        metadata.get('is_military', False),
                    )
                )
                conn.commit()
                return True
                
    except Exception as e:
        logger.error(f"Error saving flight metadata to PostgreSQL: {e}", exc_info=True)
        return False


def save_flight_tracks(flight_track, is_anomaly: bool, schema: str = 'live') -> bool:
    """
    Save flight track points to PostgreSQL.
    
    Args:
        flight_track: FlightTrack object with points
        is_anomaly: Whether this is an anomaly flight
        schema: Schema to save to (live_research, research_new, feedback)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine table name - live_research uses flight_tracks, others use normal_tracks/normal_tracks
        if schema == 'live':
            table = f"{schema}.normal_tracks"
        else:
            table = f"{schema}.anomalies_tracks" if is_anomaly else f"{schema}.normal_tracks"
        
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Batch insert all points
                for point in flight_track.points:
                    cursor.execute(
                        f"""
                        INSERT INTO {table} (
                            flight_id, timestamp, lat, lon, alt,
                            gspeed, vspeed, track, squawk, callsign, source
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (flight_id, timestamp) DO NOTHING
                        """,
                        (
                            flight_track.flight_id,
                            point.timestamp,
                            point.lat,
                            point.lon,
                            point.alt,
                            point.gspeed,
                            point.vspeed,
                            point.track,
                            point.squawk,
                            point.callsign,
                            point.source,
                        )
                    )
                conn.commit()
                logger.debug(f"Saved {len(flight_track.points)} track points to {table}")
                return True
                
    except Exception as e:
        logger.error(f"Error saving flight tracks to PostgreSQL: {e}", exc_info=True)
        return False


def save_anomaly_report(report: Dict[str, Any], last_ts: int, metadata: Dict[str, Any], 
                       schema: str = 'live') -> bool:
    """
    Save or update anomaly report in PostgreSQL.
    
    Args:
        report: Anomaly report dict from pipeline
        last_ts: Last timestamp of the flight
        metadata: Flight metadata dict
        schema: Schema to save to (live_research, research_new, feedback)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Extract matched rules
                matched_rules = report.get("matched_rules", [])
                rule_ids = ",".join(r.get("id", "") for r in matched_rules) if matched_rules else None
                rule_names = ",".join(r.get("name", "") for r in matched_rules) if matched_rules else None
                rule_categories = ",".join(r.get("category", "") for r in matched_rules) if matched_rules else None
                
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.anomaly_reports (
                        flight_id, timestamp, full_report,
                        severity_cnn, severity_dense,
                        matched_rule_ids, matched_rule_names, matched_rule_categories,
                        callsign, origin_airport, destination_airport
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (flight_id, timestamp) DO UPDATE SET
                        full_report = EXCLUDED.full_report,
                        severity_cnn = EXCLUDED.severity_cnn,
                        severity_dense = EXCLUDED.severity_dense,
                        matched_rule_ids = EXCLUDED.matched_rule_ids,
                        matched_rule_names = EXCLUDED.matched_rule_names,
                        matched_rule_categories = EXCLUDED.matched_rule_categories,
                        callsign = EXCLUDED.callsign,
                        origin_airport = EXCLUDED.origin_airport,
                        destination_airport = EXCLUDED.destination_airport
                    """,
                    (
                        metadata.get('flight_id'),
                        last_ts,
                        json.dumps(report),
                        report.get("summary", {}).get("severity_cnn"),
                        report.get("summary", {}).get("severity_dense"),
                        rule_ids,
                        rule_names,
                        rule_categories,
                        metadata.get('callsign'),
                        metadata.get('origin_airport'),
                        metadata.get('destination_airport'),
                    )
                )
                conn.commit()
                logger.debug(f"Saved anomaly report for {metadata.get('flight_id')}")
                return True
                
    except Exception as e:
        logger.error(f"Error saving anomaly report to PostgreSQL: {e}", exc_info=True)
        return False
