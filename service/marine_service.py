"""
Marine Vessel Data Service

Provides access to marine vessel tracking data from the marine schema.
Queries PostgreSQL for vessel positions, metadata, tracks, and statistics.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import psycopg2.extras
import logging

from service.pg_provider import get_connection

logger = logging.getLogger(__name__)


def get_active_vessels(
    limit: int = 50,
    offset: int = 0,
    since_minutes: int = 10,
    vessel_type: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Dict:
    """
    Get list of active vessels with their latest positions.
    
    Args:
        limit: Maximum number of vessels to return
        offset: Pagination offset
        since_minutes: Consider vessels active if seen in last N minutes
        vessel_type: Filter by vessel type description (e.g., "Cargo", "Tanker")
        bbox: Bounding box filter (south, west, north, east)
    
    Returns:
        Dictionary with vessels list, total count, and pagination info
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Build query - get latest position for each vessel
                query = """
                    SELECT DISTINCT ON (vp.mmsi)
                        vp.mmsi,
                        vm.vessel_name,
                        vm.vessel_type_description,
                        vp.latitude,
                        vp.longitude,
                        vp.speed_over_ground,
                        vp.course_over_ground,
                        vp.heading,
                        vp.navigation_status,
                        vp.timestamp
                    FROM marine.vessel_positions vp
                    LEFT JOIN marine.vessel_metadata vm USING (mmsi)
                    WHERE vp.timestamp > NOW() - INTERVAL '%s minutes'
                """
                params = [since_minutes]
                
                # Add optional filters
                if vessel_type:
                    query += " AND vm.vessel_type_description ILIKE %s"
                    params.append(f"%{vessel_type}%")
                
                if bbox:
                    south, west, north, east = bbox
                    query += """ 
                        AND vp.latitude BETWEEN %s AND %s
                        AND vp.longitude BETWEEN %s AND %s
                    """
                    params.extend([south, north, west, east])
                
                query += """
                    ORDER BY vp.mmsi, vp.timestamp DESC
                    LIMIT %s OFFSET %s
                """
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                vessels = cursor.fetchall()
                
                # Get total count
                count_query = """
                    SELECT COUNT(DISTINCT mmsi) 
                    FROM marine.vessel_positions 
                    WHERE timestamp > NOW() - INTERVAL '%s minutes'
                """
                count_params = [since_minutes]
                
                if vessel_type:
                    count_query += """ 
                        AND mmsi IN (
                            SELECT mmsi FROM marine.vessel_metadata 
                            WHERE vessel_type_description ILIKE %s
                        )
                    """
                    count_params.append(f"%{vessel_type}%")
                
                if bbox:
                    south, west, north, east = bbox
                    count_query += """ 
                        AND mmsi IN (
                            SELECT DISTINCT mmsi FROM marine.vessel_positions
                            WHERE timestamp > NOW() - INTERVAL '%s minutes'
                            AND latitude BETWEEN %s AND %s
                            AND longitude BETWEEN %s AND %s
                        )
                    """
                    count_params.extend([since_minutes, south, north, west, east])
                
                cursor.execute(count_query, count_params)
                total = cursor.fetchone()['count']
                
                return {
                    "vessels": [dict(v) for v in vessels],
                    "total": total,
                    "limit": limit,
                    "offset": offset
                }
    except Exception as e:
        logger.error(f"Error fetching active vessels: {e}", exc_info=True)
        raise


def get_vessel_details(mmsi: str) -> Optional[Dict]:
    """
    Get detailed vessel information including latest position.
    
    Args:
        mmsi: Vessel MMSI identifier
    
    Returns:
        Dictionary with vessel details or None if not found
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # First try to get full details from metadata
                query = """
                    SELECT 
                        vm.mmsi,
                        vm.vessel_name,
                        vm.callsign,
                        vm.imo_number,
                        vm.vessel_type,
                        vm.vessel_type_description,
                        vm.length,
                        vm.width,
                        vm.draught,
                        vm.destination,
                        vm.eta,
                        vm.first_seen,
                        vm.last_updated,
                        vm.total_position_reports,
                        vp.latitude,
                        vp.longitude,
                        vp.speed_over_ground,
                        vp.course_over_ground,
                        vp.heading,
                        vp.navigation_status,
                        vp.timestamp as last_position_time
                    FROM marine.vessel_metadata vm
                    LEFT JOIN LATERAL (
                        SELECT *
                        FROM marine.vessel_positions
                        WHERE mmsi = vm.mmsi
                        ORDER BY timestamp DESC
                        LIMIT 1
                    ) vp ON true
                    WHERE vm.mmsi = %s
                """
                cursor.execute(query, [mmsi])
                vessel = cursor.fetchone()
                
                if vessel:
                    return dict(vessel)
                
                # If not in metadata, try to get from positions only
                position_query = """
                    SELECT 
                        mmsi,
                        NULL as vessel_name,
                        NULL as callsign,
                        NULL as imo_number,
                        NULL as vessel_type,
                        NULL as vessel_type_description,
                        NULL as length,
                        NULL as width,
                        NULL as draught,
                        NULL as destination,
                        NULL as eta,
                        MIN(timestamp) as first_seen,
                        MAX(timestamp) as last_updated,
                        COUNT(*) as total_position_reports,
                        (SELECT latitude FROM marine.vessel_positions WHERE mmsi = %s ORDER BY timestamp DESC LIMIT 1) as latitude,
                        (SELECT longitude FROM marine.vessel_positions WHERE mmsi = %s ORDER BY timestamp DESC LIMIT 1) as longitude,
                        (SELECT speed_over_ground FROM marine.vessel_positions WHERE mmsi = %s ORDER BY timestamp DESC LIMIT 1) as speed_over_ground,
                        (SELECT course_over_ground FROM marine.vessel_positions WHERE mmsi = %s ORDER BY timestamp DESC LIMIT 1) as course_over_ground,
                        (SELECT heading FROM marine.vessel_positions WHERE mmsi = %s ORDER BY timestamp DESC LIMIT 1) as heading,
                        (SELECT navigation_status FROM marine.vessel_positions WHERE mmsi = %s ORDER BY timestamp DESC LIMIT 1) as navigation_status,
                        (SELECT timestamp FROM marine.vessel_positions WHERE mmsi = %s ORDER BY timestamp DESC LIMIT 1) as last_position_time
                    FROM marine.vessel_positions
                    WHERE mmsi = %s
                    GROUP BY mmsi
                """
                cursor.execute(position_query, [mmsi] * 8)
                vessel = cursor.fetchone()
                
                return dict(vessel) if vessel else None
    except Exception as e:
        logger.error(f"Error fetching vessel details for {mmsi}: {e}", exc_info=True)
        raise


def get_vessel_track(
    mmsi: str,
    hours: int = 24,
    limit: int = 1000
) -> Dict:
    """
    Get vessel track history.
    
    Args:
        mmsi: Vessel MMSI identifier
        hours: Number of hours of history to retrieve
        limit: Maximum number of points to return
    
    Returns:
        Dictionary with track points and metadata
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Get track points
                track_query = """
                    SELECT 
                        latitude,
                        longitude,
                        speed_over_ground,
                        course_over_ground,
                        heading,
                        navigation_status,
                        timestamp
                    FROM marine.vessel_positions
                    WHERE mmsi = %s
                      AND timestamp > NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp ASC
                    LIMIT %s
                """
                cursor.execute(track_query, [mmsi, hours, limit])
                track = cursor.fetchall()
                
                # Get vessel name
                cursor.execute(
                    "SELECT vessel_name FROM marine.vessel_metadata WHERE mmsi = %s",
                    [mmsi]
                )
                vessel = cursor.fetchone()
                
                return {
                    "mmsi": mmsi,
                    "vessel_name": vessel['vessel_name'] if vessel else None,
                    "track": [dict(t) for t in track],
                    "points": len(track)
                }
    except Exception as e:
        logger.error(f"Error fetching vessel track for {mmsi}: {e}", exc_info=True)
        raise


def search_vessels(query: str, limit: int = 20) -> List[Dict]:
    """
    Search vessels by name or MMSI.
    
    Args:
        query: Search query string
        limit: Maximum number of results
    
    Returns:
        List of matching vessels
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                search_query = """
                    SELECT 
                        vm.mmsi,
                        vm.vessel_name,
                        vm.vessel_type_description,
                        vm.destination,
                        vm.last_updated
                    FROM marine.vessel_metadata vm
                    WHERE vm.mmsi LIKE %s
                       OR vm.vessel_name ILIKE %s
                    ORDER BY 
                        CASE 
                            WHEN vm.mmsi = %s THEN 1
                            WHEN vm.vessel_name ILIKE %s THEN 2
                            ELSE 3
                        END,
                        vm.last_updated DESC
                    LIMIT %s
                """
                search_pattern = f"%{query}%"
                cursor.execute(search_query, [
                    search_pattern,  # MMSI match
                    search_pattern,  # Name match
                    query,           # Exact MMSI match (for ordering)
                    query,           # Exact name match (for ordering)
                    limit
                ])
                results = cursor.fetchall()
                
                return [dict(r) for r in results]
    except Exception as e:
        logger.error(f"Error searching vessels for '{query}': {e}", exc_info=True)
        raise


def get_marine_statistics(since_hours: int = 1) -> Dict:
    """
    Get marine traffic statistics.
    
    Args:
        since_hours: Time period to analyze
    
    Returns:
        Dictionary with statistics
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Overall statistics
                stats_query = """
                    SELECT 
                        COUNT(DISTINCT mmsi) as total_vessels,
                        COUNT(*) as total_positions
                    FROM marine.vessel_positions
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                """
                cursor.execute(stats_query, [since_hours])
                overall = cursor.fetchone()
                
                # By vessel type
                type_query = """
                    SELECT 
                        vm.vessel_type_description,
                        COUNT(DISTINCT vp.mmsi) as count
                    FROM marine.vessel_positions vp
                    LEFT JOIN marine.vessel_metadata vm USING (mmsi)
                    WHERE vp.timestamp > NOW() - INTERVAL '%s hours'
                      AND vm.vessel_type_description IS NOT NULL
                    GROUP BY vm.vessel_type_description
                    ORDER BY count DESC
                """
                cursor.execute(type_query, [since_hours])
                by_type = {row['vessel_type_description']: row['count'] for row in cursor.fetchall()}
                
                # By navigation status
                status_query = """
                    SELECT 
                        navigation_status,
                        COUNT(DISTINCT mmsi) as count
                    FROM marine.vessel_positions
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                      AND navigation_status IS NOT NULL
                    GROUP BY navigation_status
                    ORDER BY count DESC
                """
                cursor.execute(status_query, [since_hours])
                by_status = {row['navigation_status']: row['count'] for row in cursor.fetchall()}
                
                now = datetime.utcnow()
                return {
                    "period": {
                        "from": (now - timedelta(hours=since_hours)).isoformat() + "Z",
                        "to": now.isoformat() + "Z"
                    },
                    "total_vessels": overall['total_vessels'],
                    "total_positions": overall['total_positions'],
                    "by_type": by_type,
                    "by_status": by_status
                }
    except Exception as e:
        logger.error(f"Error fetching marine statistics: {e}", exc_info=True)
        raise
