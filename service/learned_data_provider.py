"""
Learned Data Provider for PostgreSQL

Fast query functions for learned paths, tubes, SIDs, and STARs.
Optimized for performance with prepared statements and connection pooling.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

from service.pg_provider import get_pool

logger = logging.getLogger(__name__)


# ==============================================================================
# PATHS QUERIES
# ==============================================================================

def get_paths_by_route(
    origin: str,
    destination: str,
    min_member_count: int = 7
) -> List[Dict[str, Any]]:
    """
    Get learned paths for a specific origin-destination pair.
    
    Args:
        origin: Origin airport code (e.g., "LLBG")
        destination: Destination airport code (e.g., "LLSD")
        min_member_count: Minimum number of member flights (default: 7)
    
    Returns:
        List of path dictionaries with centerline, width, etc.
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return []
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, origin, destination, centerline, 
                        width_nm, member_count, min_alt_ft, max_alt_ft
                    FROM learned_paths
                    WHERE origin = %s 
                        AND destination = %s
                        AND member_count >= %s
                    ORDER BY member_count DESC
                """, (origin, destination, min_member_count))
                
                results = cursor.fetchall()
                
                # Parse JSONB centerline back to list
                paths = []
                for row in results:
                    path = dict(row)
                    if isinstance(path['centerline'], str):
                        path['centerline'] = json.loads(path['centerline'])
                    paths.append(path)
                
                return paths
                
    except Exception as e:
        logger.error(f"Error fetching paths for {origin}-{destination}: {e}", exc_info=True)
        return []


def get_all_paths(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    min_member_count: int = 7
) -> List[Dict[str, Any]]:
    """
    Get all learned paths, optionally filtered by origin and/or destination.
    
    Args:
        origin: Optional origin airport filter
        destination: Optional destination airport filter
        min_member_count: Minimum number of member flights
    
    Returns:
        List of path dictionaries
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return []
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build dynamic query
                conditions = ["member_count >= %s"]
                params = [min_member_count]
                
                if origin:
                    conditions.append("origin = %s")
                    params.append(origin)
                
                if destination:
                    conditions.append("destination = %s")
                    params.append(destination)
                
                query = f"""
                    SELECT 
                        id, origin, destination, centerline, 
                        width_nm, member_count, min_alt_ft, max_alt_ft
                    FROM learned_paths
                    WHERE {' AND '.join(conditions)}
                    ORDER BY member_count DESC
                """
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Parse JSONB centerline
                paths = []
                for row in results:
                    path = dict(row)
                    if isinstance(path['centerline'], str):
                        path['centerline'] = json.loads(path['centerline'])
                    paths.append(path)
                
                return paths
                
    except Exception as e:
        logger.error(f"Error fetching all paths: {e}", exc_info=True)
        return []


# ==============================================================================
# TUBES QUERIES
# ==============================================================================

def get_tubes_by_route(
    origin: str,
    destination: str,
    min_member_count: int = 6
) -> List[Dict[str, Any]]:
    """
    Get learned tubes for a specific origin-destination pair.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        min_member_count: Minimum number of member flights (default: 6)
    
    Returns:
        List of tube dictionaries with geometry, altitude range, etc.
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return []
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, origin, destination, geometry,
                        min_alt_ft, max_alt_ft, member_count,
                        buffer_nm, alpha
                    FROM learned_tubes
                    WHERE origin = %s 
                        AND destination = %s
                        AND member_count >= %s
                    ORDER BY member_count DESC
                """, (origin, destination, min_member_count))
                
                results = cursor.fetchall()
                
                # Parse JSONB geometry
                tubes = []
                for row in results:
                    tube = dict(row)
                    if isinstance(tube['geometry'], str):
                        tube['geometry'] = json.loads(tube['geometry'])
                    tubes.append(tube)
                
                return tubes
                
    except Exception as e:
        logger.error(f"Error fetching tubes for {origin}-{destination}: {e}", exc_info=True)
        return []


def get_all_tubes(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    min_member_count: int = 6
) -> List[Dict[str, Any]]:
    """
    Get all learned tubes, optionally filtered by origin and/or destination.
    
    Args:
        origin: Optional origin airport filter
        destination: Optional destination airport filter
        min_member_count: Minimum number of member flights
    
    Returns:
        List of tube dictionaries
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return []
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Learned tubes can be large; allow 90s for full-table fetch when no filter
                cursor.execute("SET LOCAL statement_timeout = '90000'")
                # Build dynamic query
                conditions = ["member_count >= %s"]
                params = [min_member_count]
                
                if origin:
                    conditions.append("origin = %s")
                    params.append(origin)
                
                if destination:
                    conditions.append("destination = %s")
                    params.append(destination)
                
                query = f"""
                    SELECT 
                        id, origin, destination, geometry,
                        min_alt_ft, max_alt_ft, member_count,
                        buffer_nm, alpha
                    FROM learned_tubes
                    WHERE {' AND '.join(conditions)}
                    ORDER BY member_count DESC
                """
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Parse JSONB geometry
                tubes = []
                for row in results:
                    tube = dict(row)
                    if isinstance(tube['geometry'], str):
                        tube['geometry'] = json.loads(tube['geometry'])
                    tubes.append(tube)
                
                return tubes
                
    except Exception as e:
        logger.error(f"Error fetching all tubes: {e}", exc_info=True)
        return []


# ==============================================================================
# SIDS QUERIES
# ==============================================================================

def get_sids_by_airport(
    airport: str,
    min_member_count: int = 3
) -> List[Dict[str, Any]]:
    """
    Get learned SIDs for a specific airport.
    
    Args:
        airport: Airport code (e.g., "LLBG")
        min_member_count: Minimum number of member flights (default: 3)
    
    Returns:
        List of SID dictionaries with centerline, width, etc.
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return []
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, airport, type, centerline,
                        width_nm, member_count, runway
                    FROM learned_sids
                    WHERE airport = %s
                        AND member_count >= %s
                    ORDER BY member_count DESC
                """, (airport, min_member_count))
                
                results = cursor.fetchall()
                
                # Parse JSONB centerline
                sids = []
                for row in results:
                    sid = dict(row)
                    if isinstance(sid['centerline'], str):
                        sid['centerline'] = json.loads(sid['centerline'])
                    sids.append(sid)
                
                return sids
                
    except Exception as e:
        logger.error(f"Error fetching SIDs for {airport}: {e}", exc_info=True)
        return []


def get_all_sids(min_member_count: int = 3) -> List[Dict[str, Any]]:
    """
    Get all learned SIDs.
    
    Args:
        min_member_count: Minimum number of member flights
    
    Returns:
        List of SID dictionaries
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return []
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, airport, type, centerline,
                        width_nm, member_count, runway
                    FROM learned_sids
                    WHERE member_count >= %s
                    ORDER BY airport, member_count DESC
                """, (min_member_count,))
                
                results = cursor.fetchall()
                
                # Parse JSONB centerline
                sids = []
                for row in results:
                    sid = dict(row)
                    if isinstance(sid['centerline'], str):
                        sid['centerline'] = json.loads(sid['centerline'])
                    sids.append(sid)
                
                return sids
                
    except Exception as e:
        logger.error(f"Error fetching all SIDs: {e}", exc_info=True)
        return []


# ==============================================================================
# STARS QUERIES
# ==============================================================================

def get_stars_by_airport(
    airport: str,
    min_member_count: int = 3
) -> List[Dict[str, Any]]:
    """
    Get learned STARs for a specific airport.
    
    Args:
        airport: Airport code (e.g., "LLBG")
        min_member_count: Minimum number of member flights (default: 3)
    
    Returns:
        List of STAR dictionaries with centerline, width, etc.
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return []
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, airport, type, centerline,
                        width_nm, member_count, runway
                    FROM learned_stars
                    WHERE airport = %s
                        AND member_count >= %s
                    ORDER BY member_count DESC
                """, (airport, min_member_count))
                
                results = cursor.fetchall()
                
                # Parse JSONB centerline
                stars = []
                for row in results:
                    star = dict(row)
                    if isinstance(star['centerline'], str):
                        star['centerline'] = json.loads(star['centerline'])
                    stars.append(star)
                
                return stars
                
    except Exception as e:
        logger.error(f"Error fetching STARs for {airport}: {e}", exc_info=True)
        return []


def get_all_stars(min_member_count: int = 3) -> List[Dict[str, Any]]:
    """
    Get all learned STARs.
    
    Args:
        min_member_count: Minimum number of member flights
    
    Returns:
        List of STAR dictionaries
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return []
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, airport, type, centerline,
                        width_nm, member_count, runway
                    FROM learned_stars
                    WHERE member_count >= %s
                    ORDER BY airport, member_count DESC
                """, (min_member_count,))
                
                results = cursor.fetchall()
                
                # Parse JSONB centerline
                stars = []
                for row in results:
                    star = dict(row)
                    if isinstance(star['centerline'], str):
                        star['centerline'] = json.loads(star['centerline'])
                    stars.append(star)
                
                return stars
                
    except Exception as e:
        logger.error(f"Error fetching all STARs: {e}", exc_info=True)
        return []


# ==============================================================================
# COMBINED QUERIES
# ==============================================================================

def get_all_learned_layers(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    min_path_members: int = 7,
    min_tube_members: int = 6,
    min_procedure_members: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all learned layers (paths, tubes, SIDs, STARs) in one optimized call.
    
    Args:
        origin: Optional origin airport filter (for paths and tubes)
        destination: Optional destination airport filter (for paths and tubes)
        min_path_members: Minimum member count for paths
        min_tube_members: Minimum member count for tubes
        min_procedure_members: Minimum member count for SIDs/STARs
    
    Returns:
        Dictionary with keys: 'paths', 'tubes', 'sids', 'stars'
    """
    # Special handling for HECA routes
    if origin == "HECA" or destination == "HECA":
        min_path_members = max(min_path_members, 11)
        min_tube_members = max(min_tube_members, 11)
    
    result = {
        "paths": get_all_paths(origin, destination, min_path_members),
        "tubes": get_all_tubes(origin, destination, min_tube_members),
        "sids": get_all_sids(min_procedure_members),
        "stars": get_all_stars(min_procedure_members)
    }
    
    # Filter SIDs and STARs by origin/destination if specified
    if origin:
        result["sids"] = [s for s in result["sids"] if s.get("airport") == origin]
    
    if destination:
        result["stars"] = [s for s in result["stars"] if s.get("airport") == destination]
    
    return result


# ==============================================================================
# STATISTICS
# ==============================================================================

def get_statistics() -> Dict[str, Any]:
    """
    Get statistics about learned data in the database.
    
    Returns:
        Dictionary with statistics for each data type
    """
    if not psycopg2:
        logger.error("psycopg2 not installed")
        return {}
    
    try:
        pool = get_pool()
        with pool.get_connection() as conn:
            with conn.cursor() as cursor:
                stats = {}
                
                # Paths statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT origin) as unique_origins,
                        COUNT(DISTINCT destination) as unique_destinations,
                        AVG(member_count) as avg_member_count
                    FROM learned_paths
                """)
                row = cursor.fetchone()
                stats["paths"] = {
                    "total": row[0],
                    "unique_origins": row[1],
                    "unique_destinations": row[2],
                    "avg_member_count": float(row[3]) if row[3] else 0.0
                }
                
                # Tubes statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT origin) as unique_origins,
                        COUNT(DISTINCT destination) as unique_destinations,
                        AVG(member_count) as avg_member_count
                    FROM learned_tubes
                """)
                row = cursor.fetchone()
                stats["tubes"] = {
                    "total": row[0],
                    "unique_origins": row[1],
                    "unique_destinations": row[2],
                    "avg_member_count": float(row[3]) if row[3] else 0.0
                }
                
                # SIDs statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT airport) as unique_airports
                    FROM learned_sids
                """)
                row = cursor.fetchone()
                stats["sids"] = {
                    "total": row[0],
                    "unique_airports": row[1]
                }
                
                # STARs statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT airport) as unique_airports
                    FROM learned_stars
                """)
                row = cursor.fetchone()
                stats["stars"] = {
                    "total": row[0],
                    "unique_airports": row[1]
                }
                
                return stats
                
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}", exc_info=True)
        return {}
