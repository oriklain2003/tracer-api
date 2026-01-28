"""
Analytics Cache Service - Provides access to pre-computed analytics data.

This module provides functions to lookup cached analytics data from analytics_cache.db.
The cache is populated by running the precompute_analytics.py script.

OPTIMIZATION: Also provides a shared in-memory cache for GPS jamming data
to avoid redundant computations across different tabs/endpoints.
"""
import json
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import threading

logger = logging.getLogger(__name__)

# ============================================================================
# SHARED IN-MEMORY CACHE FOR GPS JAMMING DATA
# This avoids redundant queries when multiple tabs request the same data
# ============================================================================

_gps_jamming_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
_gps_jamming_cache_lock = threading.Lock()


def get_shared_gps_jamming(start_ts: int, end_ts: int) -> Optional[Dict[str, Any]]:
    """
    Get GPS jamming data from shared cache.
    
    Returns:
        Dict with 'gps_jamming' (list of points) and optionally 'gps_jamming_extended' 
        if cached, None if not in cache.
    """
    cache_key = (start_ts, end_ts)
    with _gps_jamming_cache_lock:
        return _gps_jamming_cache.get(cache_key)


def set_shared_gps_jamming(start_ts: int, end_ts: int, data: Dict[str, Any]) -> None:
    """
    Store GPS jamming data in shared cache.
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp
        data: Dict with 'gps_jamming' (list of points) and optionally 'gps_jamming_extended'
    """
    cache_key = (start_ts, end_ts)
    with _gps_jamming_cache_lock:
        _gps_jamming_cache[cache_key] = data
        logger.info(f"[SHARED CACHE] Stored GPS jamming data for range {start_ts}-{end_ts}")


def clear_shared_gps_jamming(start_ts: int = None, end_ts: int = None) -> int:
    """
    Clear GPS jamming data from shared cache.
    
    Args:
        start_ts: Optional start timestamp to clear specific range
        end_ts: Optional end timestamp to clear specific range
        
    Returns:
        Number of entries cleared
    """
    with _gps_jamming_cache_lock:
        if start_ts is not None and end_ts is not None:
            cache_key = (start_ts, end_ts)
            if cache_key in _gps_jamming_cache:
                del _gps_jamming_cache[cache_key]
                return 1
            return 0
        else:
            count = len(_gps_jamming_cache)
            _gps_jamming_cache.clear()
            return count

# Cache database path (service/analytics_cache.db)
CACHE_DB_PATH = Path(__file__).parent.parent / "analytics_cache.db"

# Known cached date range (Nov 1 2025 - Dec 31 2025)
CACHED_START_TS = 1761955200  # Nov 1 2025 00:00:00 UTC
CACHED_END_TS = 1767225599    # Dec 31 2025 23:59:59 UTC


def _get_connection() -> Optional[sqlite3.Connection]:
    """Get a connection to the cache database."""
    if not CACHE_DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(CACHE_DB_PATH), timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    except Exception as e:
        logger.warning(f"Failed to connect to cache DB: {e}")
        return None


def is_range_cached(start_ts: int, end_ts: int, tolerance_seconds: int = 3600) -> bool:
    """
    Check if the requested date range matches the cached range.
    
    Args:
        start_ts: Requested start timestamp
        end_ts: Requested end timestamp
        tolerance_seconds: Allow timestamps to be off by this amount (default 1 hour)
    
    Returns:
        True if the range is cached (within tolerance)
    """
    # Check if within tolerance of cached range
    start_match = abs(start_ts - CACHED_START_TS) <= tolerance_seconds
    end_match = abs(end_ts - CACHED_END_TS) <= tolerance_seconds
    
    return start_match and end_match


def get_cached(start_ts: int, end_ts: int, tab: str, data_key: str) -> Optional[Any]:
    """
    Get cached data for a specific tab and data key.
    
    Args:
        start_ts: Request start timestamp
        end_ts: Request end timestamp
        tab: Tab name ('overview', 'safety', 'intelligence', 'traffic')
        data_key: Data key (e.g., 'emergency_codes', 'gps_jamming')
    
    Returns:
        Cached data if found, None otherwise
    """
    # First check if range matches
    if not is_range_cached(start_ts, end_ts):
        return None
    
    conn = _get_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT data_json FROM precomputed_stats
            WHERE start_ts = ? AND end_ts = ? AND tab = ? AND data_key = ?
        """, (CACHED_START_TS, CACHED_END_TS, tab, data_key))
        
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    except Exception as e:
        logger.warning(f"Error reading cache for {tab}.{data_key}: {e}")
        return None
    finally:
        conn.close()


def get_cached_batch(start_ts: int, end_ts: int, tab: str, data_keys: list) -> Dict[str, Any]:
    """
    Get multiple cached data items for a tab.
    
    Args:
        start_ts: Request start timestamp
        end_ts: Request end timestamp
        tab: Tab name
        data_keys: List of data keys to fetch
    
    Returns:
        Dict mapping data_key to cached data (only includes found items)
    """
    # First check if range matches
    if not is_range_cached(start_ts, end_ts):
        return {}
    
    conn = _get_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        
        # Build placeholders for IN clause
        placeholders = ','.join('?' * len(data_keys))
        
        cursor.execute(f"""
            SELECT data_key, data_json FROM precomputed_stats
            WHERE start_ts = ? AND end_ts = ? AND tab = ? AND data_key IN ({placeholders})
        """, (CACHED_START_TS, CACHED_END_TS, tab, *data_keys))
        
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = json.loads(row[1])
        
        return results
    except Exception as e:
        logger.warning(f"Error reading cache batch for {tab}: {e}")
        return {}
    finally:
        conn.close()


def get_cache_info() -> Dict[str, Any]:
    """Get information about the analytics cache."""
    conn = _get_connection()
    if not conn:
        return {
            "exists": False,
            "path": str(CACHE_DB_PATH),
            "cached_range": None,
            "total_entries": 0,
            "entries_by_tab": {}
        }
    
    try:
        cursor = conn.cursor()
        
        # Total entries
        cursor.execute("SELECT COUNT(*) FROM precomputed_stats")
        total = cursor.fetchone()[0]
        
        # Entries by tab
        cursor.execute("""
            SELECT tab, COUNT(*) FROM precomputed_stats
            GROUP BY tab
        """)
        by_tab = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Check if our expected range is cached
        cursor.execute("""
            SELECT COUNT(*) FROM precomputed_stats
            WHERE start_ts = ? AND end_ts = ?
        """, (CACHED_START_TS, CACHED_END_TS))
        range_cached = cursor.fetchone()[0] > 0
        
        return {
            "exists": True,
            "path": str(CACHE_DB_PATH),
            "cached_range": {
                "start_ts": CACHED_START_TS,
                "end_ts": CACHED_END_TS,
                "is_cached": range_cached
            },
            "total_entries": total,
            "entries_by_tab": by_tab
        }
    except Exception as e:
        logger.warning(f"Error getting cache info: {e}")
        return {
            "exists": True,
            "path": str(CACHE_DB_PATH),
            "error": str(e)
        }
    finally:
        conn.close()


# Mapping from include keys to cache data_keys for each batch endpoint
SAFETY_KEY_MAP = {
    'emergency_codes': 'emergency_codes',
    'near_miss': 'near_miss',
    'go_arounds': 'go_arounds',
    'hourly': 'go_arounds_hourly',
    'monthly': 'safety_monthly',
    'locations': 'near_miss_locations',
    'phase': 'safety_by_phase',
    'aftermath': 'emergency_aftermath',
    'top_airlines': 'top_airline_emergencies',
    'by_country': 'near_miss_by_country',
    'emergency_clusters': 'emergency_clusters',
    'weather_impact': 'weather_impact',
    'daily_incident_clusters': 'daily_incident_clusters',
    # NEW: Airline Safety Scorecard
    'airline_scorecard': 'airline_scorecard',
    # NEW: Near-miss polygon clusters
    'near_miss_clusters': 'near_miss_clusters',
}

INTELLIGENCE_KEY_MAP = {
    'efficiency': 'airline_efficiency',
    'holding': 'holding_patterns',
    'gps_jamming': 'gps_jamming',
    'military': 'military_patterns',
    'clusters': 'pattern_clusters',
    'routes': 'military_routes',
    'activity': 'airline_activity',
    'gps_jamming_temporal': 'gps_jamming_temporal',
    'gps_jamming_clusters': 'gps_jamming_clusters',
    'gps_jamming_zones': 'gps_jamming_zones',  # Predictive jamming zones
    'route_efficiency': 'route_efficiency',
    'signal_loss_zones': 'signal_loss_zones',
    # NEW: Advanced intelligence features
    'military_by_country': 'military_by_country',
    'bilateral_proximity': 'bilateral_proximity',
    'military_by_destination': 'military_by_destination',
    'threat_assessment': 'threat_assessment',
    'jamming_triangulation': 'jamming_triangulation',
    # Military flights with full tracks for map visualization
    'military_flights_with_tracks': 'military_flights_with_tracks',
    # NEW: Military WOW Panels (8 panels for air force buyers)
    'operational_tempo': 'operational_tempo',
    'tanker_activity': 'tanker_activity',
    'night_operations': 'night_operations',
    'isr_patterns': 'isr_patterns',
    'airspace_denial': 'airspace_denial',
    'border_crossings': 'border_crossings',
    'ew_correlation': 'ew_correlation',
    'mission_readiness': 'mission_readiness',
}

OVERVIEW_KEY_MAP = {
    'stats': 'stats',
    'flights_per_day': 'flights_per_day',
    'gps_jamming': 'gps_jamming',
    'military': 'military_patterns',
    'airspace_risk': 'airspace_risk',
    'monthly_flights': 'monthly_flights',
}

TRAFFIC_KEY_MAP = {
    'flights_per_day': 'flights_per_day',
    'airports': 'busiest_airports',
    'signal_loss': 'signal_loss',
    'signal_monthly': 'signal_loss_monthly',
    'signal_hourly': 'signal_loss_hourly',
    'signal_loss_clusters': 'signal_loss_clusters',  # Signal loss polygon clusters
    'peak_hours': 'peak_hours',
    'diversions': 'diversion_stats',
    'diversions_monthly': 'diversions_monthly',
    'alternates': 'alternate_airports',
    'rtb': 'rtb_events',
    'missing_info': 'missing_info',
    'deviations': 'deviations_by_type',
    'bottlenecks': 'bottleneck_zones',
    # 'runway_usage': 'runway_usage',  # Removed - not pre-computed, loaded separately
    # NEW: Seasonal analysis endpoints
    'seasonal_year_comparison': 'seasonal_year_comparison',
    'traffic_safety_correlation': 'traffic_safety_correlation',
    'special_events_impact': 'special_events_impact',
    'signal_loss_anomalies': 'signal_loss_anomalies',
    'diversions_seasonal': 'diversions_seasonal',
    # Holding patterns (moved from intelligence tab)
    'holding_patterns': 'holding_patterns',
}

# Result key mapping (what key to use in the API response)
SAFETY_RESULT_KEY_MAP = {
    'emergency_codes': 'emergency_codes',
    'near_miss': 'near_miss',
    'go_arounds': 'go_arounds',
    'go_arounds_hourly': 'go_arounds_hourly',
    'safety_monthly': 'safety_monthly',
    'near_miss_locations': 'near_miss_locations',
    'safety_by_phase': 'safety_by_phase',
    'emergency_aftermath': 'emergency_aftermath',
    'top_airline_emergencies': 'top_airline_emergencies',
    'near_miss_by_country': 'near_miss_by_country',
    'emergency_clusters': 'emergency_clusters',
    'weather_impact': 'weather_impact',
    'daily_incident_clusters': 'daily_incident_clusters',
    # NEW: Airline Safety Scorecard
    'airline_scorecard': 'airline_scorecard',
    # NEW: Near-miss polygon clusters
    'near_miss_clusters': 'near_miss_clusters',
}

INTELLIGENCE_RESULT_KEY_MAP = {
    'airline_efficiency': 'airline_efficiency',
    'holding_patterns': 'holding_patterns',
    'gps_jamming': 'gps_jamming',
    'military_patterns': 'military_patterns',
    'pattern_clusters': 'pattern_clusters',
    'military_routes': 'military_routes',
    'airline_activity': 'airline_activity',
    'gps_jamming_temporal': 'gps_jamming_temporal',
    'gps_jamming_clusters': 'gps_jamming_clusters',
    'gps_jamming_zones': 'gps_jamming_zones',  # Predictive jamming zones - WAS MISSING!
    'route_efficiency': 'route_efficiency',
    'signal_loss_zones': 'signal_loss_zones',
    # NEW: Advanced intelligence features
    'military_by_country': 'military_by_country',
    'bilateral_proximity': 'bilateral_proximity',
    'military_by_destination': 'military_by_destination',
    'threat_assessment': 'threat_assessment',
    'jamming_triangulation': 'jamming_triangulation',
    # Military flights with full tracks for map visualization
    'military_flights_with_tracks': 'military_flights_with_tracks',
    # NEW: Military WOW Panels (8 panels for air force buyers)
    'operational_tempo': 'operational_tempo',
    'tanker_activity': 'tanker_activity',
    'night_operations': 'night_operations',
    'isr_patterns': 'isr_patterns',
    'airspace_denial': 'airspace_denial',
    'border_crossings': 'border_crossings',
    'ew_correlation': 'ew_correlation',
    'mission_readiness': 'mission_readiness',
}

OVERVIEW_RESULT_KEY_MAP = {
    'stats': 'stats',
    'flights_per_day': 'flights_per_day',
    'gps_jamming': 'gps_jamming',
    'military_patterns': 'military_patterns',
    'airspace_risk': 'airspace_risk',
    'monthly_flights': 'monthly_flights',
}

TRAFFIC_RESULT_KEY_MAP = {
    'flights_per_day': 'flights_per_day',
    'busiest_airports': 'busiest_airports',
    'signal_loss': 'signal_loss',
    'signal_loss_monthly': 'signal_loss_monthly',
    'signal_loss_hourly': 'signal_loss_hourly',
    'signal_loss_clusters': 'signal_loss_clusters',  # Signal loss polygon clusters
    'peak_hours': 'peak_hours',
    'diversion_stats': 'diversion_stats',
    'diversions_monthly': 'diversions_monthly',
    'alternate_airports': 'alternate_airports',
    'rtb_events': 'rtb_events',
    'missing_info': 'missing_info',
    'deviations_by_type': 'deviations_by_type',
    'bottleneck_zones': 'bottleneck_zones',
    # 'runway_usage': 'runway_usage',  # Removed - not pre-computed
    # NEW: Seasonal analysis endpoints
    'seasonal_year_comparison': 'seasonal_year_comparison',
    'traffic_safety_correlation': 'traffic_safety_correlation',
    'special_events_impact': 'special_events_impact',
    'signal_loss_anomalies': 'signal_loss_anomalies',
    'diversions_seasonal': 'diversions_seasonal',
    # Holding patterns (moved from intelligence tab)
    'holding_patterns': 'holding_patterns',
}


def try_get_safety_cache(start_ts: int, end_ts: int, include_keys: list) -> Optional[Dict[str, Any]]:
    """
    Try to get safety batch data from cache.
    
    Returns:
        Dict with cached data if ALL requested keys are cached, None otherwise
    """
    if not is_range_cached(start_ts, end_ts):
        return None
    
    # Map include keys to cache data keys
    cache_keys = [SAFETY_KEY_MAP.get(k) for k in include_keys if k in SAFETY_KEY_MAP]
    if not cache_keys:
        return None
    
    cached = get_cached_batch(start_ts, end_ts, 'safety', cache_keys)
    
    # Check if we got all requested keys
    if len(cached) != len(cache_keys):
        return None
    
    # Map cache keys back to result keys
    result = {}
    for cache_key, data in cached.items():
        result_key = SAFETY_RESULT_KEY_MAP.get(cache_key, cache_key)
        result[result_key] = data
    
    logger.info(f"[CACHE HIT] Safety batch - {len(result)} items from cache")
    return result


def try_get_intelligence_cache(start_ts: int, end_ts: int, include_keys: list, partial_ok: bool = True) -> Optional[Dict[str, Any]]:
    """
    Try to get intelligence batch data from cache.
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp  
        include_keys: List of keys to fetch
        partial_ok: If True, return partial results even if some keys are missing.
                    If False, return None unless ALL keys are cached.
    
    Returns:
        Dict with cached data, or None if range not cached or no keys found.
        When partial_ok=True, also returns '_missing_keys' list of keys not in cache.
    """
    if not is_range_cached(start_ts, end_ts):
        return None
    
    cache_keys = [INTELLIGENCE_KEY_MAP.get(k) for k in include_keys if k in INTELLIGENCE_KEY_MAP]
    if not cache_keys:
        return None
    
    cached = get_cached_batch(start_ts, end_ts, 'intelligence', cache_keys)
    
    # If partial not allowed, require all keys
    if not partial_ok and len(cached) != len(cache_keys):
        return None
    
    # If nothing cached at all, return None
    if not cached:
        return None
    
    result = {}
    cached_key_set = set(cached.keys())
    missing_keys = []
    
    for include_key in include_keys:
        cache_key = INTELLIGENCE_KEY_MAP.get(include_key)
        if cache_key and cache_key in cached_key_set:
            result_key = INTELLIGENCE_RESULT_KEY_MAP.get(cache_key, cache_key)
            result[result_key] = cached[cache_key]
        elif cache_key:
            # Track which original include keys are missing
            missing_keys.append(include_key)
    
    if missing_keys:
        result['_missing_keys'] = missing_keys
        logger.info(f"[CACHE PARTIAL] Intelligence batch - {len(result)-1} items from cache, {len(missing_keys)} missing: {missing_keys}")
    else:
        logger.info(f"[CACHE HIT] Intelligence batch - {len(result)} items from cache")
    
    return result


def try_get_overview_cache(start_ts: int, end_ts: int, include_keys: list) -> Optional[Dict[str, Any]]:
    """
    Try to get overview batch data from cache.
    
    Returns:
        Dict with cached data if ALL requested keys are cached, None otherwise
    """
    if not is_range_cached(start_ts, end_ts):
        return None
    
    cache_keys = [OVERVIEW_KEY_MAP.get(k) for k in include_keys if k in OVERVIEW_KEY_MAP]
    if not cache_keys:
        return None
    
    # Overview might request gps_jamming/military from intelligence tab cache
    cached = {}
    for key in cache_keys:
        if key in ['gps_jamming', 'military_patterns']:
            data = get_cached(start_ts, end_ts, 'intelligence', key)
        else:
            data = get_cached(start_ts, end_ts, 'overview', key)
        if data is not None:
            cached[key] = data
    
    if len(cached) != len(cache_keys):
        return None
    
    result = {}
    for cache_key, data in cached.items():
        result_key = OVERVIEW_RESULT_KEY_MAP.get(cache_key, cache_key)
        result[result_key] = data
    
    logger.info(f"[CACHE HIT] Overview batch - {len(result)} items from cache")
    return result


def try_get_traffic_cache(start_ts: int, end_ts: int, include_keys: list) -> Optional[Dict[str, Any]]:
    """
    Try to get traffic batch data from cache.
    
    Returns:
        Dict with cached data if ALL requested keys are cached, None otherwise
    """
    if not is_range_cached(start_ts, end_ts):
        return None
    
    cache_keys = [TRAFFIC_KEY_MAP.get(k) for k in include_keys if k in TRAFFIC_KEY_MAP]
    if not cache_keys:
        return None
    
    cached = get_cached_batch(start_ts, end_ts, 'traffic', cache_keys)
    
    if len(cached) != len(cache_keys):
        return None
    
    result = {}
    for cache_key, data in cached.items():
        result_key = TRAFFIC_RESULT_KEY_MAP.get(cache_key, cache_key)
        result[result_key] = data
    
    logger.info(f"[CACHE HIT] Traffic batch - {len(result)} items from cache")
    return result


def try_get_single_cached(start_ts: int, end_ts: int, tab: str, data_key: str) -> Optional[Any]:
    """
    Try to get a single cached data item.
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp  
        tab: Tab name ('safety', 'intelligence', 'traffic', 'overview')
        data_key: The data key to retrieve
    
    Returns:
        Cached data if available, None otherwise
    """
    if not is_range_cached(start_ts, end_ts):
        return None
    
    data = get_cached(start_ts, end_ts, tab, data_key)
    if data is not None:
        logger.info(f"[CACHE HIT] {tab}.{data_key}")
    return data

