"""
Analytics routes - statistics, trends, intelligence, predictions, cache management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException

# Import analytics cache for pre-computed data lookup
from service.analytics.cache import (
    try_get_safety_cache,
    try_get_intelligence_cache,
    try_get_overview_cache,
    try_get_traffic_cache,
    try_get_single_cached,
    get_cache_info as get_analytics_cache_info,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analytics"])

# These will be set by the main api.py module
DB_TRACKS_PATH: Path = None
DB_ANOMALIES_PATH: Path = None
DB_RESEARCH_PATH: Path = None
PRESENT_DB_PATH: Path = None
FEEDBACK_TAGGED_DB_PATH: Path = None

# Analytics engine instances (set by configure)
stats_engine = None
trends_analyzer = None
intelligence_engine = None
predictive_analytics = None
batch_stats_engine = None
clear_stats_cache_func = None
get_cache_info_func = None


def configure(
    db_tracks_path: Path,
    db_anomalies_path: Path,
    db_research_path: Path,
    present_db_path: Path,
    feedback_tagged_db_path: Path,
    stats_engine_instance,
    trends_analyzer_instance,
    intelligence_engine_instance,
    predictive_analytics_instance,
    batch_stats_engine_instance,
    clear_cache_func,
    cache_info_func,
):
    """Configure the router with paths and engine instances from api.py"""
    global \
        DB_TRACKS_PATH, \
        DB_ANOMALIES_PATH, \
        DB_RESEARCH_PATH, \
        PRESENT_DB_PATH, \
        FEEDBACK_TAGGED_DB_PATH
    global \
        stats_engine, \
        trends_analyzer, \
        intelligence_engine, \
        predictive_analytics, \
        batch_stats_engine
    global clear_stats_cache_func, get_cache_info_func

    DB_TRACKS_PATH = db_tracks_path
    DB_ANOMALIES_PATH = db_anomalies_path
    DB_RESEARCH_PATH = db_research_path
    PRESENT_DB_PATH = present_db_path
    FEEDBACK_TAGGED_DB_PATH = feedback_tagged_db_path
    stats_engine = stats_engine_instance
    trends_analyzer = trends_analyzer_instance
    intelligence_engine = intelligence_engine_instance
    predictive_analytics = predictive_analytics_instance
    batch_stats_engine = batch_stats_engine_instance
    clear_stats_cache_func = clear_cache_func
    get_cache_info_func = cache_info_func


# ============================================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================================


@router.post("/api/cache/clear")
def clear_cache():
    """Clear all cached statistics data."""
    count = clear_stats_cache_func()
    return {"status": "ok", "cleared_entries": count}


@router.get("/api/cache/info")
def cache_info():
    """Get cache statistics."""
    return get_cache_info_func()


@router.get("/api/cache/analytics")
def analytics_cache_info():
    """Get pre-computed analytics cache information."""
    return get_analytics_cache_info()


# ============================================================================
# BATCH STATISTICS ENDPOINT
# ============================================================================

from pydantic import BaseModel
from typing import List as PyList


class BatchStatsRequest(BaseModel):
    start_ts: int
    end_ts: int
    stats: PyList[str] = ["overview", "safety", "traffic"]
    safety_stats: PyList[str] = []
    traffic_stats: PyList[str] = []


@router.post("/api/stats/batch")
def get_stats_batch(request: BatchStatsRequest):
    """
    Batch statistics endpoint - compute multiple stats in a single request.
    """
    try:
        return batch_stats_engine.compute_batch_stats(request)
    except Exception as e:
        logger.error(f"Error in batch stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SAFETY BATCH ENDPOINT - Reduces 10 API calls to 1
# ============================================================================


class SafetyBatchRequest(BaseModel):
    start_ts: int
    end_ts: int
    include: PyList[str] = [
        "emergency_codes",
        "near_miss",
        "go_arounds",
        "hourly",
        "monthly",
        "locations",
        "phase",
        "aftermath",
        "top_airlines",
        "by_country",
        # NEW: Additional precomputed safety data
        "emergency_clusters",
        "weather_impact",
        "daily_incident_clusters",
        # Airline Safety Scorecard
        "airline_scorecard",
        # Near-miss polygon clusters
        "near_miss_clusters",
    ]


@router.post("/api/stats/safety/batch")
def get_safety_batch(request: SafetyBatchRequest):
    """Batch endpoint - compute all safety stats in one request."""
    import time

    batch_start = time.perf_counter()

    # Try to get from pre-computed cache first
    cached_result = try_get_safety_cache(
        request.start_ts, request.end_ts, request.include
    )
    if cached_result is not None:
        logger.info(
            f"[SAFETY BATCH] Cache hit - returned in {time.perf_counter() - batch_start:.3f}s"
        )
        return cached_result

    try:
        result = {}
        include_set = set(request.include)
        start_ts, end_ts = request.start_ts, request.end_ts
        timings = {}

        if "emergency_codes" in include_set:
            t0 = time.perf_counter()
            try:
                result["emergency_codes"] = stats_engine.get_emergency_codes_stats(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching emergency_codes: {e}")
                result["emergency_codes"] = []
            timings["emergency_codes"] = time.perf_counter() - t0

        if "near_miss" in include_set:
            t0 = time.perf_counter()
            try:
                result["near_miss"] = stats_engine.get_near_miss_events(
                    start_ts, end_ts, None
                )
            except Exception as e:
                logger.warning(f"Error fetching near_miss: {e}")
                result["near_miss"] = []
            timings["near_miss"] = time.perf_counter() - t0

        if "go_arounds" in include_set:
            t0 = time.perf_counter()
            try:
                result["go_arounds"] = stats_engine.get_go_around_stats(
                    start_ts, end_ts, None
                )
            except Exception as e:
                logger.warning(f"Error fetching go_arounds: {e}")
                result["go_arounds"] = []
            timings["go_arounds"] = time.perf_counter() - t0

        if "hourly" in include_set:
            t0 = time.perf_counter()
            try:
                result["go_arounds_hourly"] = stats_engine.get_go_arounds_hourly(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching go_arounds_hourly: {e}")
                result["go_arounds_hourly"] = []
            timings["go_arounds_hourly"] = time.perf_counter() - t0

        if "monthly" in include_set:
            t0 = time.perf_counter()
            try:
                result["safety_monthly"] = stats_engine.get_safety_events_monthly(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching safety_monthly: {e}")
                result["safety_monthly"] = []
            timings["safety_monthly"] = time.perf_counter() - t0

        if "locations" in include_set:
            t0 = time.perf_counter()
            try:
                result["near_miss_locations"] = stats_engine.get_near_miss_locations(
                    start_ts, end_ts, 50
                )
            except Exception as e:
                logger.warning(f"Error fetching near_miss_locations: {e}")
                result["near_miss_locations"] = []
            timings["near_miss_locations"] = time.perf_counter() - t0

        if "phase" in include_set:
            t0 = time.perf_counter()
            try:
                result["safety_by_phase"] = stats_engine.get_safety_by_phase(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching safety_by_phase: {e}")
                result["safety_by_phase"] = None
            timings["safety_by_phase"] = time.perf_counter() - t0

        if "aftermath" in include_set:
            t0 = time.perf_counter()
            try:
                result["emergency_aftermath"] = stats_engine.get_emergency_aftermath(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching emergency_aftermath: {e}")
                result["emergency_aftermath"] = []
            timings["emergency_aftermath"] = time.perf_counter() - t0

        if "top_airlines" in include_set:
            t0 = time.perf_counter()
            try:
                result["top_airline_emergencies"] = (
                    stats_engine.get_top_airline_emergencies(start_ts, end_ts, 10)
                )
            except Exception as e:
                logger.warning(f"Error fetching top_airline_emergencies: {e}")
                result["top_airline_emergencies"] = []
            timings["top_airline_emergencies"] = time.perf_counter() - t0

        if "by_country" in include_set:
            t0 = time.perf_counter()
            try:
                result["near_miss_by_country"] = stats_engine.get_near_miss_by_country(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching near_miss_by_country: {e}")
                result["near_miss_by_country"] = None
            timings["near_miss_by_country"] = time.perf_counter() - t0

        # NEW: Emergency Clusters
        if "emergency_clusters" in include_set:
            t0 = time.perf_counter()
            try:
                result["emergency_clusters"] = stats_engine.get_emergency_clusters(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching emergency_clusters: {e}")
                result["emergency_clusters"] = None
            timings["emergency_clusters"] = time.perf_counter() - t0

        # NEW: Weather Impact
        if "weather_impact" in include_set:
            t0 = time.perf_counter()
            try:
                result["weather_impact"] = stats_engine.get_weather_impact_analysis(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching weather_impact: {e}")
                result["weather_impact"] = None
            timings["weather_impact"] = time.perf_counter() - t0

        # NEW: Daily Incident Clusters
        if "daily_incident_clusters" in include_set:
            t0 = time.perf_counter()
            try:
                result["daily_incident_clusters"] = (
                    stats_engine.get_daily_incident_clusters(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching daily_incident_clusters: {e}")
                result["daily_incident_clusters"] = None
            timings["daily_incident_clusters"] = time.perf_counter() - t0

        # NEW: Airline Safety Scorecard
        if "airline_scorecard" in include_set:
            t0 = time.perf_counter()
            try:
                result["airline_scorecard"] = stats_engine.get_airline_safety_scorecard(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching airline_scorecard: {e}")
                result["airline_scorecard"] = {"scorecards": [], "summary": {}}
            timings["airline_scorecard"] = time.perf_counter() - t0

        # NEW: Near-miss polygon clusters
        if "near_miss_clusters" in include_set:
            t0 = time.perf_counter()
            try:
                result["near_miss_clusters"] = stats_engine.get_near_miss_clusters(
                    start_ts,
                    end_ts,
                    cluster_threshold_nm=30,
                    min_points_for_polygon=3,
                    limit=100,
                )
            except Exception as e:
                logger.warning(f"Error fetching near_miss_clusters: {e}")
                result["near_miss_clusters"] = {
                    "clusters": [],
                    "singles": [],
                    "total_points": 0,
                    "total_clusters": 0,
                }
            timings["near_miss_clusters"] = time.perf_counter() - t0

        total_time = time.perf_counter() - batch_start
        # Log timing summary
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            f"[SAFETY BATCH] Total: {total_time:.2f}s | "
            + " | ".join([f"{k}: {v:.2f}s" for k, v in sorted_timings[:5]])
        )

        return result
    except Exception as e:
        logger.error(f"Error in safety batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# INTELLIGENCE BATCH ENDPOINT - Reduces 7 API calls to 1
# ============================================================================


class IntelligenceBatchRequest(BaseModel):
    start_ts: int
    end_ts: int
    include: PyList[str] = [
        "efficiency",
        "holding",
        "gps_jamming",
        "military",
        "clusters",
        "routes",
        "activity",
        # NEW: Additional precomputed intelligence data
        "gps_jamming_temporal",
        "gps_jamming_clusters",
        "gps_jamming_zones",
        "route_efficiency",
        "signal_loss_zones",
        # Country-specific military breakdown
        "military_by_country",
        # Bilateral proximity detection
        "bilateral_proximity",
        # Military by destination (Syria filter etc.)
        "military_by_destination",
        # Combined Threat Assessment
        "threat_assessment",
        # Jamming Source Triangulation
        "jamming_triangulation",
        # Military flights with full tracks for map visualization
        "military_flights_with_tracks",
    ]


@router.post("/api/intel/batch")
def get_intelligence_batch(request: IntelligenceBatchRequest):
    """Batch endpoint - compute all intelligence stats in one request."""
    import time

    batch_start = time.perf_counter()

    # Try to get from pre-computed cache first (with partial results support)
    cached_result = try_get_intelligence_cache(
        request.start_ts, request.end_ts, request.include, partial_ok=True
    )

    # Check if we got a full cache hit (no missing keys)
    if cached_result is not None and "_missing_keys" not in cached_result:
        logger.info(
            f"[INTEL BATCH] Full cache hit - returned in {time.perf_counter() - batch_start:.3f}s"
        )
        return cached_result

    try:
        # Start with cached results if we have partial hit
        result = {}
        if cached_result:
            missing_keys = set(cached_result.pop("_missing_keys", []))
            result = cached_result
            # Only compute the missing keys
            include_set = missing_keys
            logger.info(
                f"[INTEL BATCH] Partial cache hit - computing {len(missing_keys)} missing keys: {missing_keys}"
            )
        else:
            include_set = set(request.include)

        start_ts, end_ts = request.start_ts, request.end_ts
        timings = {}

        if "efficiency" in include_set:
            t0 = time.perf_counter()
            try:
                result["airline_efficiency"] = stats_engine.get_airline_efficiency(
                    start_ts, end_ts, 15
                )
            except Exception as e:
                logger.warning(f"Error fetching airline_efficiency: {e}")
                result["airline_efficiency"] = []
            timings["efficiency"] = time.perf_counter() - t0

        if "holding" in include_set:
            t0 = time.perf_counter()
            try:
                result["holding_patterns"] = (
                    trends_analyzer.get_holding_pattern_analysis(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching holding_patterns: {e}")
                result["holding_patterns"] = None
            timings["holding"] = time.perf_counter() - t0

        if "gps_jamming" in include_set:
            t0 = time.perf_counter()
            try:
                result["gps_jamming"] = intelligence_engine.get_gps_jamming_heatmap(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching gps_jamming: {e}")
                result["gps_jamming"] = []
            timings["gps_jamming"] = time.perf_counter() - t0

        if "military" in include_set:
            t0 = time.perf_counter()
            try:
                result["military_patterns"] = (
                    intelligence_engine.get_military_activity_patterns(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching military_patterns: {e}")
                result["military_patterns"] = []
            timings["military"] = time.perf_counter() - t0

        if "clusters" in include_set:
            t0 = time.perf_counter()
            try:
                result["pattern_clusters"] = intelligence_engine.get_pattern_clusters(
                    start_ts, end_ts, 3
                )
            except Exception as e:
                logger.warning(f"Error fetching pattern_clusters: {e}")
                result["pattern_clusters"] = []
            timings["clusters"] = time.perf_counter() - t0

        if "routes" in include_set:
            t0 = time.perf_counter()
            try:
                result["military_routes"] = intelligence_engine.get_military_routes(
                    start_ts, end_ts, 20
                )
            except Exception as e:
                logger.warning(f"Error fetching military_routes: {e}")
                result["military_routes"] = None
            timings["routes"] = time.perf_counter() - t0

        if "activity" in include_set:
            t0 = time.perf_counter()
            try:
                result["airline_activity"] = trends_analyzer.get_airline_activity(
                    start_ts, end_ts, 30
                )
            except Exception as e:
                logger.warning(f"Error fetching airline_activity: {e}")
                result["airline_activity"] = None
            timings["activity"] = time.perf_counter() - t0

        # NEW: GPS Jamming Temporal
        if "gps_jamming_temporal" in include_set:
            t0 = time.perf_counter()
            try:
                result["gps_jamming_temporal"] = stats_engine.get_gps_jamming_temporal(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching gps_jamming_temporal: {e}")
                result["gps_jamming_temporal"] = {
                    "by_hour": [],
                    "by_day_of_week": [],
                    "peak_hours": [],
                    "peak_days": [],
                    "total_events": 0,
                }
            timings["gps_jamming_temporal"] = time.perf_counter() - t0

        # NEW: GPS Jamming Clusters (polygons)
        if "gps_jamming_clusters" in include_set:
            t0 = time.perf_counter()
            try:
                result["gps_jamming_clusters"] = (
                    intelligence_engine.get_gps_jamming_clusters(
                        start_ts, end_ts, 50, 3
                    )
                )
            except Exception as e:
                logger.warning(f"Error fetching gps_jamming_clusters: {e}")
                result["gps_jamming_clusters"] = {
                    "clusters": [],
                    "singles": [],
                    "total_points": 0,
                    "total_clusters": 0,
                }
            timings["gps_jamming_clusters"] = time.perf_counter() - t0

        # NEW: GPS Jamming Zones (predictive expanded polygons)
        if "gps_jamming_zones" in include_set:
            t0 = time.perf_counter()
            try:
                result["gps_jamming_zones"] = intelligence_engine.get_gps_jamming_zones(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching gps_jamming_zones: {e}")
                result["gps_jamming_zones"] = {
                    "zones": [],
                    "total_events": 0,
                    "total_zones": 0,
                    "jamming_summary": {
                        "total_jamming_area_sq_nm": 0,
                        "avg_jamming_score": 0,
                        "primary_type": "unknown",
                        "hotspot_regions": [],
                    },
                }
            timings["gps_jamming_zones"] = time.perf_counter() - t0

        # NEW: Route Efficiency
        if "route_efficiency" in include_set:
            t0 = time.perf_counter()
            try:
                result["route_efficiency"] = (
                    stats_engine.get_route_efficiency_comparison(None, start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching route_efficiency: {e}")
                result["route_efficiency"] = {}
            timings["route_efficiency"] = time.perf_counter() - t0

        # NEW: Signal Loss Zones
        if "signal_loss_zones" in include_set:
            t0 = time.perf_counter()
            try:
                result["signal_loss_zones"] = intelligence_engine.get_signal_loss_zones(
                    start_ts, end_ts, 50
                )
            except Exception as e:
                logger.warning(f"Error fetching signal_loss_zones: {e}")
                result["signal_loss_zones"] = []
            timings["signal_loss_zones"] = time.perf_counter() - t0

        # NEW: Military by Country - detailed country-specific breakdown
        if "military_by_country" in include_set:
            t0 = time.perf_counter()
            try:
                result["military_by_country"] = (
                    intelligence_engine.get_military_by_country(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching military_by_country: {e}")
                result["military_by_country"] = {
                    "countries": {},
                    "summary": {"total_military_flights": 0, "alerts": []},
                }
            timings["military_by_country"] = time.perf_counter() - t0

        # NEW: Bilateral Proximity Detection (Russian-American, etc.)
        if "bilateral_proximity" in include_set:
            t0 = time.perf_counter()
            try:
                result["bilateral_proximity"] = (
                    intelligence_engine.get_bilateral_proximity_events(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching bilateral_proximity: {e}")
                result["bilateral_proximity"] = {
                    "events": [],
                    "by_pair": {},
                    "total_events": 0,
                    "high_risk_events": 0,
                    "alerts": [],
                }
            timings["bilateral_proximity"] = time.perf_counter() - t0

        # NEW: Military by Destination (Syria filter, etc.)
        if "military_by_destination" in include_set:
            t0 = time.perf_counter()
            try:
                result["military_by_destination"] = (
                    intelligence_engine.get_military_by_destination(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching military_by_destination: {e}")
                result["military_by_destination"] = {
                    "flights": [],
                    "by_destination": {},
                    "syria_flights": [],
                    "alerts": [],
                }
            timings["military_by_destination"] = time.perf_counter() - t0

        # NEW: Combined Threat Assessment Widget
        if "threat_assessment" in include_set:
            t0 = time.perf_counter()
            try:
                result["threat_assessment"] = (
                    intelligence_engine.get_combined_threat_assessment(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching threat_assessment: {e}")
                result["threat_assessment"] = {
                    "overall_score": 0,
                    "threat_level": "UNKNOWN",
                    "components": {},
                    "alerts": [],
                    "recommendations": [],
                }
            timings["threat_assessment"] = time.perf_counter() - t0

        # NEW: Jamming Source Triangulation
        if "jamming_triangulation" in include_set:
            t0 = time.perf_counter()
            try:
                result["jamming_triangulation"] = (
                    intelligence_engine.get_jamming_source_triangulation(
                        start_ts, end_ts
                    )
                )
            except Exception as e:
                logger.warning(f"Error fetching jamming_triangulation: {e}")
                result["jamming_triangulation"] = {
                    "estimated_sources": [],
                    "total_affected_flights": 0,
                    "triangulation_quality": "error",
                }
            timings["jamming_triangulation"] = time.perf_counter() - t0

        # NEW: Military flights with full tracks for map visualization
        if "military_flights_with_tracks" in include_set:
            t0 = time.perf_counter()
            try:
                result["military_flights_with_tracks"] = (
                    intelligence_engine.get_military_flights_with_tracks(
                        start_ts, end_ts, flights_per_country=30
                    )
                )
            except Exception as e:
                logger.warning(f"Error fetching military_flights_with_tracks: {e}")
                result["military_flights_with_tracks"] = {
                    "flights": [],
                    "by_country": {},
                    "total_flights": 0,
                    "countries": [],
                }
            timings["military_flights_with_tracks"] = time.perf_counter() - t0

        # NEW: Combined Signal Map - GPS Jamming + Signal Loss on same map
        if "combined_signal_map" in include_set:
            t0 = time.perf_counter()
            try:
                result["combined_signal_map"] = (
                    intelligence_engine.get_combined_signal_map(
                        start_ts, end_ts, limit=50
                    )
                )
            except Exception as e:
                logger.warning(f"Error fetching combined_signal_map: {e}")
                result["combined_signal_map"] = {
                    "points": [],
                    "summary": {
                        "total_jamming_events": 0,
                        "total_signal_loss_events": 0,
                        "jamming_zones": 0,
                        "signal_loss_zones": 0,
                        "total_zones": 0,
                    },
                    "legend": {
                        "jamming": {
                            "color": "#ef4444",
                            "label": "GPS Jamming",
                            "description": "Active interference",
                        },
                        "signal_loss": {
                            "color": "#f97316",
                            "label": "Signal Loss",
                            "description": "Coverage gaps",
                        },
                    },
                }
            timings["combined_signal_map"] = time.perf_counter() - t0

        total_time = time.perf_counter() - batch_start
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            f"[INTEL BATCH] Total: {total_time:.2f}s | "
            + " | ".join([f"{k}: {v:.2f}s" for k, v in sorted_timings])
        )

        return result
    except Exception as e:
        logger.error(f"Error in intelligence batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# OVERVIEW BATCH ENDPOINT - Reduces 6 API calls to 1
# ============================================================================


class OverviewBatchRequest(BaseModel):
    start_ts: int
    end_ts: int
    include: PyList[str] = [
        "stats",
        "flights_per_day",
        "gps_jamming",
        "military",
        "airspace_risk",
        "monthly_flights",
    ]


@router.post("/api/stats/overview/batch")
def get_overview_batch(request: OverviewBatchRequest):
    """Batch endpoint - compute all overview stats in one request."""
    import time

    batch_start = time.perf_counter()

    # Try to get from pre-computed cache first
    cached_result = try_get_overview_cache(
        request.start_ts, request.end_ts, request.include
    )
    if cached_result is not None:
        logger.info(
            f"[OVERVIEW BATCH] Cache hit - returned in {time.perf_counter() - batch_start:.3f}s"
        )
        return cached_result

    try:
        result = {}
        include_set = set(request.include)
        start_ts, end_ts = request.start_ts, request.end_ts
        timings = {}

        if "stats" in include_set:
            t0 = time.perf_counter()
            try:
                result["stats"] = stats_engine.get_overview_stats(start_ts, end_ts)
            except Exception as e:
                logger.warning(f"Error fetching stats: {e}")
                result["stats"] = None
            timings["stats"] = time.perf_counter() - t0

        if "flights_per_day" in include_set:
            t0 = time.perf_counter()
            try:
                result["flights_per_day"] = stats_engine.get_flights_per_day(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching flights_per_day: {e}")
                result["flights_per_day"] = []
            timings["flights_per_day"] = time.perf_counter() - t0

        if "gps_jamming" in include_set:
            t0 = time.perf_counter()
            try:
                result["gps_jamming"] = intelligence_engine.get_gps_jamming_heatmap(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching gps_jamming: {e}")
                result["gps_jamming"] = []
            timings["gps_jamming"] = time.perf_counter() - t0

        if "military" in include_set:
            t0 = time.perf_counter()
            try:
                result["military_patterns"] = (
                    intelligence_engine.get_military_activity_patterns(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching military_patterns: {e}")
                result["military_patterns"] = []
            timings["military"] = time.perf_counter() - t0

        if "airspace_risk" in include_set:
            t0 = time.perf_counter()
            try:
                result["airspace_risk"] = predictive_analytics.calculate_airspace_risk()
            except Exception as e:
                logger.warning(f"Error fetching airspace_risk: {e}")
                result["airspace_risk"] = None
            timings["airspace_risk"] = time.perf_counter() - t0

        if "monthly_flights" in include_set:
            t0 = time.perf_counter()
            try:
                result["monthly_flights"] = stats_engine.get_flights_per_month(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching monthly_flights: {e}")
                result["monthly_flights"] = []
            timings["monthly_flights"] = time.perf_counter() - t0

        total_time = time.perf_counter() - batch_start
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            f"[OVERVIEW BATCH] Total: {total_time:.2f}s | "
            + " | ".join([f"{k}: {v:.2f}s" for k, v in sorted_timings])
        )

        return result
    except Exception as e:
        logger.error(f"Error in overview batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRAFFIC BATCH ENDPOINT - Reduces 13 API calls to 1
# ============================================================================


class TrafficBatchRequest(BaseModel):
    start_ts: int
    end_ts: int
    include: PyList[str] = [
        "flights_per_day",
        "airports",
        "signal_loss",
        "signal_monthly",
        "signal_hourly",
        "signal_loss_clusters",
        "peak_hours",
        "diversions",
        "diversions_monthly",
        "alternates",
        "rtb",
        "missing_info",
        "deviations",
        "bottlenecks",
        # NEW: Seasonal analysis endpoints
        "seasonal_year_comparison",
        "traffic_safety_correlation",
        "special_events_impact",
        "signal_loss_anomalies",
        "diversions_seasonal",
        # Holding patterns (moved from intelligence tab)
        "holding_patterns",
    ]


@router.post("/api/stats/traffic/batch")
def get_traffic_batch(request: TrafficBatchRequest):
    """Batch endpoint - compute all traffic stats in one request."""
    import time

    batch_start = time.perf_counter()

    # Try to get from pre-computed cache first
    cached_result = try_get_traffic_cache(
        request.start_ts, request.end_ts, request.include
    )
    if cached_result is not None:
        logger.info(
            f"[TRAFFIC BATCH] Cache hit - returned in {time.perf_counter() - batch_start:.3f}s"
        )
        return cached_result

    try:
        result = {}
        include_set = set(request.include)
        start_ts, end_ts = request.start_ts, request.end_ts
        timings = {}

        if "flights_per_day" in include_set:
            t0 = time.perf_counter()
            try:
                result["flights_per_day"] = stats_engine.get_flights_per_day(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching flights_per_day: {e}")
                result["flights_per_day"] = []
            timings["flights_per_day"] = time.perf_counter() - t0

        if "airports" in include_set:
            t0 = time.perf_counter()
            try:
                result["busiest_airports"] = stats_engine.get_busiest_airports(
                    start_ts, end_ts, 10
                )
            except Exception as e:
                logger.warning(f"Error fetching busiest_airports: {e}")
                result["busiest_airports"] = []
            timings["airports"] = time.perf_counter() - t0

        if "signal_loss" in include_set:
            t0 = time.perf_counter()
            try:
                result["signal_loss"] = stats_engine.get_signal_loss_locations(
                    start_ts, end_ts, 50
                )
            except Exception as e:
                logger.warning(f"Error fetching signal_loss: {e}")
                result["signal_loss"] = []
            timings["signal_loss"] = time.perf_counter() - t0

        if "signal_monthly" in include_set:
            t0 = time.perf_counter()
            try:
                result["signal_loss_monthly"] = stats_engine.get_signal_loss_monthly(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching signal_loss_monthly: {e}")
                result["signal_loss_monthly"] = []
            timings["signal_monthly"] = time.perf_counter() - t0

        if "signal_hourly" in include_set:
            t0 = time.perf_counter()
            try:
                result["signal_loss_hourly"] = stats_engine.get_signal_loss_hourly(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching signal_loss_hourly: {e}")
                result["signal_loss_hourly"] = []
            timings["signal_hourly"] = time.perf_counter() - t0

        if "signal_loss_clusters" in include_set:
            t0 = time.perf_counter()
            try:
                result["signal_loss_clusters"] = stats_engine.get_signal_loss_clusters(
                    start_ts,
                    end_ts,
                    cluster_threshold_nm=15,
                    min_points_for_polygon=3,
                    limit=100,
                )
            except Exception as e:
                logger.warning(f"Error fetching signal_loss_clusters: {e}")
                result["signal_loss_clusters"] = {
                    "clusters": [],
                    "singles": [],
                    "total_points": 0,
                    "total_clusters": 0,
                }
            timings["signal_loss_clusters"] = time.perf_counter() - t0

        if "peak_hours" in include_set:
            t0 = time.perf_counter()
            try:
                result["peak_hours"] = trends_analyzer.get_peak_hours_analysis(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching peak_hours: {e}")
                result["peak_hours"] = None
            timings["peak_hours"] = time.perf_counter() - t0

        if "diversions" in include_set:
            t0 = time.perf_counter()
            try:
                result["diversion_stats"] = stats_engine.get_diversion_stats(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching diversion_stats: {e}")
                result["diversion_stats"] = None
            timings["diversions"] = time.perf_counter() - t0

        if "diversions_monthly" in include_set:
            t0 = time.perf_counter()
            try:
                result["diversions_monthly"] = stats_engine.get_diversions_monthly(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching diversions_monthly: {e}")
                result["diversions_monthly"] = []
            timings["diversions_monthly"] = time.perf_counter() - t0

        if "alternates" in include_set:
            t0 = time.perf_counter()
            try:
                result["alternate_airports"] = (
                    trends_analyzer.get_alternate_airports_by_time(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching alternate_airports: {e}")
                result["alternate_airports"] = []
            timings["alternates"] = time.perf_counter() - t0

        if "rtb" in include_set:
            t0 = time.perf_counter()
            try:
                # Use tagged RTB events from feedback_tagged.db (rule_id=6 = Return to Land)
                result["rtb_events"] = stats_engine.get_tagged_rtb_events(
                    start_ts, end_ts, 50
                )
            except Exception as e:
                logger.warning(f"Error fetching rtb_events: {e}")
                result["rtb_events"] = []
            timings["rtb"] = time.perf_counter() - t0

        if "missing_info" in include_set:
            t0 = time.perf_counter()
            try:
                result["missing_info"] = stats_engine.get_flights_missing_info(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching missing_info: {e}")
                result["missing_info"] = None
            timings["missing_info"] = time.perf_counter() - t0

        if "deviations" in include_set:
            t0 = time.perf_counter()
            try:
                result["deviations_by_type"] = (
                    stats_engine.get_deviations_by_aircraft_type(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching deviations_by_type: {e}")
                result["deviations_by_type"] = []
            timings["deviations"] = time.perf_counter() - t0

        if "bottlenecks" in include_set:
            t0 = time.perf_counter()
            try:
                result["bottleneck_zones"] = stats_engine.get_bottleneck_zones(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching bottleneck_zones: {e}")
                result["bottleneck_zones"] = []
            timings["bottlenecks"] = time.perf_counter() - t0

        # NEW: Seasonal Year Comparison
        if "seasonal_year_comparison" in include_set:
            t0 = time.perf_counter()
            try:
                result["seasonal_year_comparison"] = (
                    stats_engine.get_seasonal_year_comparison(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching seasonal_year_comparison: {e}")
                result["seasonal_year_comparison"] = {}
            timings["seasonal_year_comparison"] = time.perf_counter() - t0

        # NEW: Traffic Safety Correlation
        if "traffic_safety_correlation" in include_set:
            t0 = time.perf_counter()
            try:
                result["traffic_safety_correlation"] = (
                    stats_engine.get_traffic_safety_correlation(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching traffic_safety_correlation: {e}")
                result["traffic_safety_correlation"] = {}
            timings["traffic_safety_correlation"] = time.perf_counter() - t0

        # NEW: Special Events Impact
        if "special_events_impact" in include_set:
            t0 = time.perf_counter()
            try:
                result["special_events_impact"] = (
                    stats_engine.get_special_events_impact(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching special_events_impact: {e}")
                result["special_events_impact"] = {}
            timings["special_events_impact"] = time.perf_counter() - t0

        # NEW: Signal Loss Anomalies
        if "signal_loss_anomalies" in include_set:
            t0 = time.perf_counter()
            try:
                result["signal_loss_anomalies"] = (
                    stats_engine.get_signal_loss_anomalies(start_ts, end_ts, 30)
                )
            except Exception as e:
                logger.warning(f"Error fetching signal_loss_anomalies: {e}")
                result["signal_loss_anomalies"] = {}
            timings["signal_loss_anomalies"] = time.perf_counter() - t0

        # NEW: Diversions Seasonal
        if "diversions_seasonal" in include_set:
            t0 = time.perf_counter()
            try:
                result["diversions_seasonal"] = stats_engine.get_diversions_seasonal(
                    start_ts, end_ts
                )
            except Exception as e:
                logger.warning(f"Error fetching diversions_seasonal: {e}")
                result["diversions_seasonal"] = {}
            timings["diversions_seasonal"] = time.perf_counter() - t0

        # Holding Patterns (moved from intelligence tab)
        if "holding_patterns" in include_set:
            t0 = time.perf_counter()
            try:
                result["holding_patterns"] = (
                    trends_analyzer.get_holding_pattern_analysis(start_ts, end_ts)
                )
            except Exception as e:
                logger.warning(f"Error fetching holding_patterns: {e}")
                result["holding_patterns"] = None
            timings["holding_patterns"] = time.perf_counter() - t0

        total_time = time.perf_counter() - batch_start
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            f"[TRAFFIC BATCH] Total: {total_time:.2f}s | "
            + " | ".join([f"{k}: {v:.2f}s" for k, v in sorted_timings[:5]])
        )

        return result
    except Exception as e:
        logger.error(f"Error in traffic batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# INDIVIDUAL STATISTICS ENDPOINTS (kept for backward compatibility/debugging)
# Note: Dashboard now uses batch endpoints above for better performance
# ============================================================================


@router.get("/api/stats/overview")
def get_stats_overview(start_ts: int, end_ts: int, force_refresh: bool = False):
    """Get overview statistics for dashboard."""
    try:
        return stats_engine.get_overview_stats(
            start_ts, end_ts, use_cache=not force_refresh
        )
    except Exception as e:
        logger.error(f"Error in stats overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/live/overview")
def get_live_stats_overview():
    """Get live overview statistics from research_new.db (last 24 hours)."""
    try:
        if batch_stats_engine:
            return batch_stats_engine.get_live_overview_stats()
        else:
            # Fallback to regular stats for last 24 hours
            import time

            now = int(time.time())
            day_ago = now - 86400
            return stats_engine.get_overview_stats(day_ago, now, use_cache=False)
    except Exception as e:
        logger.error(f"Error in live stats overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/tagged/overview")
def get_tagged_stats_overview(start_ts: int, end_ts: int):
    """Get overview statistics from tagged feedback database."""
    try:
        return stats_engine.get_tagged_overview_stats(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in tagged stats overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/traffic/airport-hourly/{airport}")
def get_airport_hourly_traffic(airport: str, start_ts: int, end_ts: int):
    """Get hourly traffic distribution for a specific airport."""
    try:
        return stats_engine.get_airport_hourly_traffic(airport, start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in airport hourly traffic: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Note: Tagged DB endpoints removed - dashboard uses research_new.db via batch endpoints


# ============================================================================
# FLIGHT-SPECIFIC INTELLIGENCE ENDPOINTS (kept - used for individual analysis)
# ============================================================================


@router.get("/api/intel/anomaly-dna/{flight_id}")
def get_anomaly_dna(flight_id: str):
    """Get 'DNA fingerprint' of a specific anomaly."""
    try:
        return intelligence_engine.get_anomaly_dna(flight_id)
    except Exception as e:
        logger.error(f"Error in anomaly DNA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/flight-jamming/{flight_id}")
def get_flight_jamming_analysis(flight_id: str):
    """Get detailed GPS jamming analysis for a specific flight."""
    try:
        return intelligence_engine.get_flight_jamming_analysis(flight_id)
    except Exception as e:
        logger.error(f"Error in flight jamming analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intelligence/anomaly-dna/{flight_id}")
def get_intelligence_anomaly_dna(flight_id: str):
    """Get 'DNA fingerprint' of a specific anomaly (alias)."""
    try:
        return intelligence_engine.get_anomaly_dna(flight_id)
    except Exception as e:
        logger.error(f"Error in intelligence anomaly DNA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PREDICTIVE ANALYTICS ENDPOINTS
# ============================================================================


@router.get("/api/predict/airspace-risk")
def get_airspace_risk_prediction(
    lat: Optional[float] = None, lon: Optional[float] = None, hour: Optional[int] = None
):
    """Get overall airspace risk assessment.
    Note: lat/lon/hour parameters are accepted for API compatibility but currently
    returns overall airspace risk regardless of location.
    """
    try:
        return predictive_analytics.calculate_airspace_risk()
    except Exception as e:
        logger.error(f"Error in airspace risk prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/predict/trajectory")
def get_trajectory_prediction(flight_id: str):
    """Predict future trajectory for a flight."""
    try:
        return predictive_analytics.predict_trajectory(flight_id)
    except Exception as e:
        logger.error(f"Error in trajectory prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/predict/safety-forecast")
def get_safety_forecast(start_ts: int, end_ts: int):
    """Get safety event forecast for a time period."""
    try:
        return predictive_analytics.get_safety_forecast(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in safety forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/predict/hostile-intent/{flight_id}")
def get_hostile_intent_prediction(flight_id: str):
    """Predict hostile intent probability for a flight."""
    try:
        return predictive_analytics.predict_hostile_intent(flight_id)
    except Exception as e:
        logger.error(f"Error in hostile intent prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/predict/trajectory/{flight_id}")
def get_flight_trajectory_prediction(flight_id: str):
    """Predict future trajectory for a specific flight."""
    try:
        return predictive_analytics.predict_trajectory(flight_id)
    except Exception as e:
        logger.error(f"Error in flight trajectory prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Note: Diversion, RTB stats are now in traffic batch endpoint


# ============================================================================
# SEASONAL TRENDS ENDPOINTS
# ============================================================================


@router.get("/api/stats/seasonal/year-comparison")
def get_seasonal_year_comparison(start_ts: int, end_ts: int):
    """Get year-over-year comparison of traffic and safety statistics."""
    # Try cache first
    cached = try_get_single_cached(
        start_ts, end_ts, "traffic", "seasonal_year_comparison"
    )
    if cached is not None:
        return cached
    try:
        return stats_engine.get_seasonal_year_comparison(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in seasonal year comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/seasonal/traffic-safety-correlation")
def get_traffic_safety_correlation(start_ts: int, end_ts: int):
    """Get correlation between traffic volume and safety events by hour."""
    # Try cache first
    cached = try_get_single_cached(
        start_ts, end_ts, "traffic", "traffic_safety_correlation"
    )
    if cached is not None:
        return cached
    try:
        return stats_engine.get_traffic_safety_correlation(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in traffic-safety correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/seasonal/special-events")
def get_special_events_impact(start_ts: int, end_ts: int):
    """Detect unusual traffic patterns around holidays/special events."""
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "traffic", "special_events_impact")
    if cached is not None:
        return cached
    try:
        return stats_engine.get_special_events_impact(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in special events analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ROUTE EFFICIENCY ENDPOINTS
# ============================================================================


@router.get("/api/stats/routes/efficiency")
def get_route_efficiency(start_ts: int, end_ts: int, route: Optional[str] = None):
    """
    Compare airline efficiency on the same route.

    If route is not specified, returns top routes by traffic volume.
    Route format: ORIG-DEST (e.g., LLBG-EGLL)
    """
    # Try cache first (only for summary without specific route)
    if route is None:
        cached = try_get_single_cached(
            start_ts, end_ts, "intelligence", "route_efficiency"
        )
        if cached is not None:
            return cached
    try:
        return stats_engine.get_route_efficiency_comparison(route, start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in route efficiency: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/routes/available")
def get_available_routes(start_ts: int, end_ts: int, min_flights: int = 5):
    """Get list of routes with minimum number of flights for route efficiency analysis."""
    try:
        return stats_engine.get_available_routes(start_ts, end_ts, min_flights)
    except Exception as e:
        logger.error(f"Error getting available routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEATHER IMPACT ENDPOINTS
# ============================================================================


@router.get("/api/stats/weather/impact")
def get_weather_impact_analysis(start_ts: int, end_ts: int):
    """
    Analyze weather impact on flight operations.

    Correlates diversions, go-arounds, and deviations with potential weather events.
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "safety", "weather_impact")
    if cached is not None:
        return cached
    try:
        return stats_engine.get_weather_impact_analysis(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in weather impact analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/weather/airport/{airport}")
def get_weather_by_airport(airport: str, start_ts: int, end_ts: int):
    """Get weather-correlated anomalies for a specific airport."""
    try:
        return stats_engine.get_weather_by_airport(airport, start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in airport weather analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Note: Trends endpoints removed - data available via batch endpoints


# ============================================================================
# NEW DASHBOARD DEMANDS ENDPOINTS
# ============================================================================


@router.get("/api/stats/signal-loss/anomalies")
def get_signal_loss_anomalies(start_ts: int, end_ts: int, lookback_days: int = 30):
    """
    Detect unusual signal loss in areas that normally have good reception.

    Compares current signal loss patterns against a historical baseline.
    """
    # Try cache first (default lookback_days=30)
    if lookback_days == 30:
        cached = try_get_single_cached(
            start_ts, end_ts, "traffic", "signal_loss_anomalies"
        )
        if cached is not None:
            return cached
    try:
        return stats_engine.get_signal_loss_anomalies(start_ts, end_ts, lookback_days)
    except Exception as e:
        logger.error(f"Error in signal loss anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/gps-jamming/temporal")
def get_gps_jamming_temporal(start_ts: int, end_ts: int):
    """
    Analyze GPS jamming patterns by time of day and day of week.

    Returns hourly and daily distribution of GPS jamming events.
    """
    # Try cache first
    cached = try_get_single_cached(
        start_ts, end_ts, "intelligence", "gps_jamming_temporal"
    )
    if cached is not None:
        return cached
    try:
        return stats_engine.get_gps_jamming_temporal(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in GPS jamming temporal analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/gps-jamming/clusters")
def get_gps_jamming_clusters(
    start_ts: int, end_ts: int, cluster_threshold_nm: float = 50, min_points: int = 3
):
    """
    Get GPS jamming clusters with polygon boundaries.

    Clusters GPS jamming points that are within cluster_threshold_nm of each other.
    For clusters with min_points+ points, computes convex hull polygon coordinates.

    Returns:
        {
            'clusters': [{id, polygon, centroid, point_count, total_events, affected_flights, ...}],
            'singles': [{lat, lon, event_count, ...}],
            'total_points': int,
            'total_clusters': int
        }
    """
    # Try cache first (default params)
    if cluster_threshold_nm == 50 and min_points == 3:
        cached = try_get_single_cached(
            start_ts, end_ts, "intelligence", "gps_jamming_clusters"
        )
        if cached is not None:
            return cached
    try:
        return intelligence_engine.get_gps_jamming_clusters(
            start_ts, end_ts, cluster_threshold_nm, min_points
        )
    except Exception as e:
        logger.error(f"Error in GPS jamming clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/go-arounds/hourly")
def get_go_arounds_hourly(start_ts: int, end_ts: int):
    """
    Get hourly distribution of go-around events.

    Answers: "What time of day are there the most go-arounds?"
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "safety", "go_arounds_hourly")
    if cached is not None:
        return cached
    try:
        return stats_engine.get_go_arounds_hourly(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in go-arounds hourly analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/diversions/seasonal")
def get_diversions_seasonal(start_ts: int, end_ts: int):
    """
    Get seasonal breakdown of diversions.

    Answers: "What time of year has the most diversions?"
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "traffic", "diversions_seasonal")
    if cached is not None:
        return cached
    try:
        return stats_engine.get_diversions_seasonal(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in diversions seasonal analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats/incidents/daily-clusters")
def get_daily_incident_clusters(start_ts: int, end_ts: int):
    """
    Analyze daily clustering of incidents to detect unusual days.

    Answers: "Were there multiple incidents in one day? Were they in the same area?"
    """
    # Try cache first
    cached = try_get_single_cached(
        start_ts, end_ts, "safety", "daily_incident_clusters"
    )
    if cached is not None:
        return cached
    try:
        return stats_engine.get_daily_incident_clusters(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in daily incident clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MILITARY INTELLIGENCE ENDPOINTS - NEW WOW PANELS
# ============================================================================


@router.get("/api/intel/operational-tempo")
def get_operational_tempo(start_ts: int, end_ts: int):
    """
    Get military operational tempo over time - hourly activity levels by country.

    Shows military activity buildup patterns that can predict upcoming operations.
    """
    # Try cache first
    cached = try_get_single_cached(
        start_ts, end_ts, "intelligence", "operational_tempo"
    )
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_operational_tempo(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in operational tempo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/tanker-activity")
def get_tanker_activity(start_ts: int, end_ts: int):
    """
    Track aerial refueling tanker activity - indicates upcoming strike operations.

    Tanker positions indicate upcoming/ongoing strike operations.
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "intelligence", "tanker_activity")
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_tanker_activity(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in tanker activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/night-operations")
def get_night_operations(start_ts: int, end_ts: int):
    """
    Analyze military night operations - often indicates sensitive/covert activity.

    Night is defined as 20:00-06:00 local time.
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "intelligence", "night_operations")
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_night_operations(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in night operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/isr-patterns")
def get_isr_patterns(start_ts: int, end_ts: int):
    """
    Detect ISR (Intelligence, Surveillance, Reconnaissance) flight patterns.

    Identifies figure-8s, racetracks, and orbit patterns typical of reconnaissance.
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "intelligence", "isr_patterns")
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_isr_patterns(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in ISR patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/airspace-denial")
def get_airspace_denial_zones(start_ts: int, end_ts: int):
    """
    Detect areas where commercial aircraft are avoiding - indicates active operations.

    Compares current traffic patterns against expected patterns to find "empty" zones.
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "intelligence", "airspace_denial")
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_airspace_denial_zones(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in airspace denial zones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/border-crossings")
def get_border_crossings(start_ts: int, end_ts: int):
    """
    Track military aircraft border crossings with timeline.

    Shows when military aircraft crossed borders with temporal patterns.
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "intelligence", "border_crossings")
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_border_crossings(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in border crossings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/ew-correlation")
def get_ew_correlation(start_ts: int, end_ts: int):
    """
    Correlate GPS jamming with military flight paths - shows EW source locations.

    Visualizes jamming zones overlaid with military flight paths.
    """
    # Try cache first
    cached = try_get_single_cached(start_ts, end_ts, "intelligence", "ew_correlation")
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_ew_correlation(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in EW correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/mission-readiness")
def get_mission_readiness(start_ts: int, end_ts: int):
    """
    Generate mission readiness prediction based on multiple indicators.

    Combines tanker activity, ISR density, GPS jamming, and military buildup
    to predict likelihood of upcoming operations.
    """
    # Try cache first
    cached = try_get_single_cached(
        start_ts, end_ts, "intelligence", "mission_readiness"
    )
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_mission_readiness_indicators(start_ts, end_ts)
    except Exception as e:
        logger.error(f"Error in mission readiness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/military-flights-tracks")
def get_military_flights_with_tracks(
    start_ts: int, end_ts: int, flights_per_country: int = 30
):
    """
    Get the last N military flights per country with full track data for map visualization.

    Returns flights grouped by country with their complete flight paths.
    """
    # Try cache first
    cached = try_get_single_cached(
        start_ts, end_ts, "intelligence", "military_flights_with_tracks"
    )
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_military_flights_with_tracks(
            start_ts, end_ts, flights_per_country
        )
    except Exception as e:
        logger.error(f"Error in military flights with tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intel/combined-signal-map")
def get_combined_signal_map(start_ts: int, end_ts: int, limit: int = 50):
    """
    Get combined GPS Jamming and Signal Loss data for unified map visualization.

    Returns both data types with distinct colors for the same map:

    - GPS JAMMING (RED #ef4444): Active interference with altitude jumps, position teleports
    - SIGNAL LOSS (ORANGE #f97316): Coverage gaps where tracking was lost for 5+ minutes

    Returns:
        - points: All jamming and signal loss points with type and color
        - summary: Aggregate statistics
        - legend: Color coding and descriptions for map rendering
    """
    # Try cache first
    cached = try_get_single_cached(
        start_ts, end_ts, "intelligence", "combined_signal_map"
    )
    if cached is not None:
        return cached
    try:
        return intelligence_engine.get_combined_signal_map(
            start_ts, end_ts, limit=limit
        )
    except Exception as e:
        logger.error(f"Error in combined signal map: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MILITARY WOW PANELS BATCH ENDPOINT - Reduces 8 API calls to 1
# ============================================================================


class MilitaryBatchRequest(BaseModel):
    start_ts: int
    end_ts: int
    include: PyList[str] = [
        "operational_tempo",
        "tanker_activity",
        "night_operations",
        "isr_patterns",
        "airspace_denial",
        "border_crossings",
        "ew_correlation",
        "mission_readiness",
    ]


@router.post("/api/intel/military/batch")
def get_military_batch(request: MilitaryBatchRequest):
    """
    Batch endpoint for all 8 Military WOW panels in one request.
    Reduces 8 parallel API calls to 1, with full cache support.
    """
    import time

    batch_start = time.perf_counter()

    # Try to get ALL requested items from cache first
    from service.analytics.cache import get_cached_batch, is_range_cached

    if is_range_cached(request.start_ts, request.end_ts):
        cached = get_cached_batch(
            request.start_ts, request.end_ts, "intelligence", request.include
        )
        if len(cached) == len(request.include):
            logger.info(
                f"[MILITARY BATCH] Full cache hit - returned in {time.perf_counter() - batch_start:.3f}s"
            )
            return cached

    try:
        result = {}
        include_set = set(request.include)
        start_ts, end_ts = request.start_ts, request.end_ts
        timings = {}

        if "operational_tempo" in include_set:
            t0 = time.perf_counter()
            cached = try_get_single_cached(
                start_ts, end_ts, "intelligence", "operational_tempo"
            )
            if cached is not None:
                result["operational_tempo"] = cached
            else:
                try:
                    result["operational_tempo"] = (
                        intelligence_engine.get_operational_tempo(start_ts, end_ts)
                    )
                except Exception as e:
                    logger.warning(f"Error fetching operational_tempo: {e}")
                    result["operational_tempo"] = {
                        "hourly_data": [],
                        "daily_data": [],
                        "activity_spikes": [],
                        "total_flights": 0,
                    }
            timings["operational_tempo"] = time.perf_counter() - t0

        if "tanker_activity" in include_set:
            t0 = time.perf_counter()
            cached = try_get_single_cached(
                start_ts, end_ts, "intelligence", "tanker_activity"
            )
            if cached is not None:
                result["tanker_activity"] = cached
            else:
                try:
                    result["tanker_activity"] = intelligence_engine.get_tanker_activity(
                        start_ts, end_ts
                    )
                except Exception as e:
                    logger.warning(f"Error fetching tanker_activity: {e}")
                    result["tanker_activity"] = {
                        "active_tankers": [],
                        "tanker_count": 0,
                        "total_tanker_hours": 0,
                        "by_holding_area": {},
                        "alerts": [],
                    }
            timings["tanker_activity"] = time.perf_counter() - t0

        if "night_operations" in include_set:
            t0 = time.perf_counter()
            cached = try_get_single_cached(
                start_ts, end_ts, "intelligence", "night_operations"
            )
            if cached is not None:
                result["night_operations"] = cached
            else:
                try:
                    result["night_operations"] = (
                        intelligence_engine.get_night_operations(start_ts, end_ts)
                    )
                except Exception as e:
                    logger.warning(f"Error fetching night_operations: {e}")
                    result["night_operations"] = {
                        "night_flights": [],
                        "day_vs_night": {},
                        "by_country": {},
                        "alerts": [],
                    }
            timings["night_operations"] = time.perf_counter() - t0

        if "isr_patterns" in include_set:
            t0 = time.perf_counter()
            cached = try_get_single_cached(
                start_ts, end_ts, "intelligence", "isr_patterns"
            )
            if cached is not None:
                result["isr_patterns"] = cached
            else:
                try:
                    result["isr_patterns"] = intelligence_engine.get_isr_patterns(
                        start_ts, end_ts
                    )
                except Exception as e:
                    logger.warning(f"Error fetching isr_patterns: {e}")
                    result["isr_patterns"] = {
                        "patterns": [],
                        "total_isr_flights": 0,
                        "by_pattern_type": {},
                        "likely_collection_areas": [],
                        "alerts": [],
                    }
            timings["isr_patterns"] = time.perf_counter() - t0

        if "airspace_denial" in include_set:
            t0 = time.perf_counter()
            cached = try_get_single_cached(
                start_ts, end_ts, "intelligence", "airspace_denial"
            )
            if cached is not None:
                result["airspace_denial"] = cached
            else:
                try:
                    result["airspace_denial"] = (
                        intelligence_engine.get_airspace_denial_zones(start_ts, end_ts)
                    )
                except Exception as e:
                    logger.warning(f"Error fetching airspace_denial: {e}")
                    result["airspace_denial"] = {
                        "denial_zones": [],
                        "total_zones": 0,
                        "most_avoided_areas": [],
                        "alerts": [],
                    }
            timings["airspace_denial"] = time.perf_counter() - t0

        if "border_crossings" in include_set:
            t0 = time.perf_counter()
            cached = try_get_single_cached(
                start_ts, end_ts, "intelligence", "border_crossings"
            )
            if cached is not None:
                result["border_crossings"] = cached
            else:
                try:
                    result["border_crossings"] = (
                        intelligence_engine.get_border_crossings(start_ts, end_ts)
                    )
                except Exception as e:
                    logger.warning(f"Error fetching border_crossings: {e}")
                    result["border_crossings"] = {
                        "crossings": [],
                        "total_crossings": 0,
                        "by_country_pair": {},
                        "high_interest_crossings": [],
                        "alerts": [],
                    }
            timings["border_crossings"] = time.perf_counter() - t0

        if "ew_correlation" in include_set:
            t0 = time.perf_counter()
            cached = try_get_single_cached(
                start_ts, end_ts, "intelligence", "ew_correlation"
            )
            if cached is not None:
                result["ew_correlation"] = cached
            else:
                try:
                    result["ew_correlation"] = intelligence_engine.get_ew_correlation(
                        start_ts, end_ts
                    )
                except Exception as e:
                    logger.warning(f"Error fetching ew_correlation: {e}")
                    result["ew_correlation"] = {
                        "jamming_zones": [],
                        "military_paths": [],
                        "estimated_ew_sources": [],
                        "correlation_score": 0,
                        "total_jamming_zones": 0,
                        "zones_with_military": 0,
                        "alerts": [],
                    }
            timings["ew_correlation"] = time.perf_counter() - t0

        if "mission_readiness" in include_set:
            t0 = time.perf_counter()
            cached = try_get_single_cached(
                start_ts, end_ts, "intelligence", "mission_readiness"
            )
            if cached is not None:
                result["mission_readiness"] = cached
            else:
                try:
                    result["mission_readiness"] = (
                        intelligence_engine.get_mission_readiness_indicators(
                            start_ts, end_ts
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error fetching mission_readiness: {e}")
                    result["mission_readiness"] = {
                        "overall_readiness_score": 0,
                        "readiness_level": "UNKNOWN",
                        "indicators": {},
                        "prediction": "",
                        "confidence": "",
                        "alerts": [],
                    }
            timings["mission_readiness"] = time.perf_counter() - t0

        total_time = time.perf_counter() - batch_start
        logger.info(
            f"[MILITARY BATCH] Computed in {total_time:.3f}s - timings: {timings}"
        )
        return result
    except Exception as e:
        logger.error(f"Error in military batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))
