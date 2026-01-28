"""
Manual Route Planner - Simplified route planning with manual waypoint creation.

Features:
- Click-to-draw waypoints on map
- Per-waypoint altitude and speed configuration
- Accurate proximity detection using corridor-based path prediction
- Clean reporting of conflicts and proximities
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/route/manual", tags=["Manual Route Planning"])

# ============================================================================
# Configuration
# ============================================================================

_route_planner = None
_traffic_cache = []
_traffic_timestamp = 0

# Default proximity thresholds
DEFAULT_VERTICAL_FT = 999
DEFAULT_HORIZONTAL_NM = 5

# Boundary expansion config
AVG_AIRCRAFT_SPEED_MPH = 550  # Average speed for boundary calculations
BASE_DURATION_THRESHOLD_MIN = 20  # Minutes before we start expanding boundary
STATUTE_MILES_TO_NM = 0.868976  # Conversion factor


# ============================================================================
# Request/Response Models
# ============================================================================

class ManualWaypoint(BaseModel):
    """A manually drawn waypoint with altitude and speed."""
    lat: float
    lon: float
    alt_ft: float
    speed_kts: float
    name: Optional[str] = None


class ManualRouteRequest(BaseModel):
    """Request to plan a manual route."""
    waypoints: List[ManualWaypoint]
    aircraft_type: str = "civil"  # "civil" or "fighter"
    check_conflicts: bool = True
    proximity_thresholds: Optional[Dict[str, Any]] = None
    traffic: Optional[List[Dict[str, Any]]] = None  # Traffic data (optional, will fetch if not provided)


class ProximityAlert(BaseModel):
    """A proximity alert between planned route and traffic."""
    time_offset_min: float
    planned_lat: float
    planned_lon: float
    planned_alt_ft: float
    traffic_flight_id: str
    traffic_callsign: Optional[str]
    traffic_lat: float
    traffic_lon: float
    traffic_alt_ft: float
    horizontal_distance_nm: float
    vertical_distance_ft: float
    severity: str  # "warning" or "critical"


class ManualRouteResponse(BaseModel):
    """Response with planned route and analysis."""
    route_id: str
    waypoints: List[Dict[str, Any]]
    planned_path: List[Dict[str, Any]]
    total_distance_nm: float
    eta_minutes: float
    proximity_alerts: List[ProximityAlert]
    proximity_count: int
    proximity_score: float
    traffic_count: int
    summary: str
    boundary_polygon: List[List[float]]  # The calculated boundary
    traffic: List[Dict[str, Any]]  # The traffic flights fetched in this boundary


class ProximityPoint(BaseModel):
    """A specific point where proximity was detected."""
    time_offset_min: float
    planned_lat: float
    planned_lon: float
    planned_alt_ft: float
    traffic_flight_id: str
    traffic_callsign: Optional[str]
    traffic_lat: float
    traffic_lon: float
    traffic_alt_ft: float
    horizontal_distance_nm: float
    vertical_distance_ft: float
    danger_level: str  # "critical", "warning", or "caution"


class SimulationRequest(BaseModel):
    """Request to simulate a route with traffic."""
    planned_path: List[Dict[str, Any]]  # Path points with lat, lon, alt_ft, time_offset_min
    traffic: List[Dict[str, Any]]  # Traffic aircraft data
    proximity_thresholds: Dict[str, Any]  # Proximity threshold settings
    route_duration_min: float  # Total route duration in minutes


class SimulationResponse(BaseModel):
    """Complete simulation data with all calculations done."""
    predicted_paths: List[Dict[str, Any]]  # Predicted paths for all traffic
    proximity_report: Dict[str, Any]  # Comprehensive proximity analysis
    boundary_polygon: List[List[float]]  # Expanded boundary around route [[lon, lat], ...]
    summary: str


# ============================================================================
# Geodesy Utilities
# ============================================================================

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in nautical miles."""
    EARTH_RADIUS_KM = 6371.0
    NM_PER_KM = 0.539957
    
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c * NM_PER_KM


def calculate_route_boundary(planned_path: List[Dict[str, Any]], route_duration_min: float) -> List[List[float]]:
    """
    Calculate an expanded boundary polygon based on the training region.
    
    Uses the fixed training region bounds from config.py as the base,
    then optionally expands based on flight duration.
    
    Logic:
    - Base: Training region (Levant Region)
    - For every minute above BASE_DURATION_THRESHOLD_MIN, expand the boundary
    - Expansion = distance a plane could travel at AVG_AIRCRAFT_SPEED_MPH
    
    Returns: List of [lon, lat] coordinates forming a rectangular polygon
    """
    if not planned_path or len(planned_path) < 2:
        return []
    
    # Import training region bounds from config
    try:
        from core.config import TRAIN_NORTH, TRAIN_SOUTH, TRAIN_WEST, TRAIN_EAST
    except ImportError:
        try:
            from service.core.config import TRAIN_NORTH, TRAIN_SOUTH, TRAIN_WEST, TRAIN_EAST
        except ImportError:
            # Fallback to hardcoded values if import fails
            TRAIN_NORTH = 34.597042
            TRAIN_SOUTH = 28.536275
            TRAIN_WEST = 32.299805
            TRAIN_EAST = 37.397461
    
    # Start with training region bounds
    min_lat = TRAIN_SOUTH
    max_lat = TRAIN_NORTH
    min_lon = TRAIN_WEST
    max_lon = TRAIN_EAST
    
    # Calculate extra buffer based on duration (optional expansion)
    extra_buffer_nm = 0.0
    if route_duration_min > BASE_DURATION_THRESHOLD_MIN:
        extra_minutes = route_duration_min - BASE_DURATION_THRESHOLD_MIN
        # Distance a plane could travel in extra_minutes at AVG_AIRCRAFT_SPEED_MPH
        extra_distance_miles = (AVG_AIRCRAFT_SPEED_MPH / 60.0) * extra_minutes
        extra_buffer_nm = extra_distance_miles * STATUTE_MILES_TO_NM
    
    logger.info(f"Boundary calculation: route={route_duration_min:.1f}min, "
                f"extra_minutes={(route_duration_min - BASE_DURATION_THRESHOLD_MIN) if route_duration_min > BASE_DURATION_THRESHOLD_MIN else 0:.1f}min, "
                f"extra_buffer={extra_buffer_nm:.1f}nm")
    
    # Apply expansion if needed
    if extra_buffer_nm > 0:
        # Calculate center for buffer calculation
        center_lat = (min_lat + max_lat) / 2.0
        
        # Convert buffer from nm to degrees (approximate)
        # At equator: 1 degree latitude ≈ 60 nm
        buffer_deg_lat = extra_buffer_nm / 60.0
        buffer_deg_lon = extra_buffer_nm / (60.0 * math.cos(math.radians(center_lat)))
        
        # Expand bounding box by buffer
        min_lat = min_lat - buffer_deg_lat
        max_lat = max_lat + buffer_deg_lat
        min_lon = min_lon - buffer_deg_lon
        max_lon = max_lon + buffer_deg_lon
    
    # Create rectangular polygon (clockwise from top-left)
    polygon_points = [
        [min_lon, max_lat],  # Top-left
        [max_lon, max_lat],  # Top-right
        [max_lon, min_lat],  # Bottom-right
        [min_lon, min_lat],  # Bottom-left
        [min_lon, max_lat],  # Close polygon
    ]
    
    logger.info(f"Generated boundary from training region: "
                f"lat=[{min_lat:.2f}, {max_lat:.2f}], "
                f"lon=[{min_lon:.2f}, {max_lon:.2f}]")
    
    return polygon_points


def interpolate_waypoints(waypoints: List[ManualWaypoint], points_per_segment: int = 20) -> List[Dict[str, Any]]:
    """
    Interpolate between waypoints to create a smooth path.
    Returns list of path points with lat, lon, alt_ft, speed_kts, time_offset_min.
    """
    if len(waypoints) < 2:
        return []
    
    path = []
    cumulative_time = 0.0
    cumulative_distance = 0.0
    
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        
        # Calculate segment distance
        segment_dist = haversine_nm(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
        
        # Average speed for this segment
        avg_speed = (wp1.speed_kts + wp2.speed_kts) / 2.0
        
        # Time for this segment (hours)
        segment_time_hours = segment_dist / avg_speed if avg_speed > 0 else 0
        segment_time_min = segment_time_hours * 60
        
        # Create interpolated points
        for j in range(points_per_segment):
            fraction = j / points_per_segment
            
            lat = wp1.lat + (wp2.lat - wp1.lat) * fraction
            lon = wp1.lon + (wp2.lon - wp1.lon) * fraction
            alt = wp1.alt_ft + (wp2.alt_ft - wp1.alt_ft) * fraction
            speed = wp1.speed_kts + (wp2.speed_kts - wp1.speed_kts) * fraction
            
            point_dist = cumulative_distance + segment_dist * fraction
            point_time = cumulative_time + segment_time_min * fraction
            
            path.append({
                "lat": lat,
                "lon": lon,
                "alt_ft": alt,
                "speed_kts": speed,
                "time_offset_min": point_time,
                "distance_nm": point_dist,
            })
        
        cumulative_distance += segment_dist
        cumulative_time += segment_time_min
    
    # Add final waypoint
    final_wp = waypoints[-1]
    path.append({
        "lat": final_wp.lat,
        "lon": final_wp.lon,
        "alt_ft": final_wp.alt_ft,
        "speed_kts": final_wp.speed_kts,
        "time_offset_min": cumulative_time,
        "distance_nm": cumulative_distance,
    })
    
    return path


# ============================================================================
# Proximity Detection
# ============================================================================

def detect_proximity_for_planning(
    planned_path: List[Dict[str, Any]],
    traffic_data: List[Dict[str, Any]],
    route_planner: Any,
    proximity_thresholds: Dict[str, Any],
    route_duration: float
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Detect proximity using the SAME logic as simulate endpoint.
    Returns: (proximity_points, proximity_report)
    """
    # Import dependencies
    try:
        from service.routes.route_planning import predict_aircraft_path_unified
        from service.route_planner import TrafficAircraft
    except ImportError:
        from routes.route_planning import predict_aircraft_path_unified
        from route_planner import TrafficAircraft
    
    vertical_ft = proximity_thresholds.get("vertical_ft", DEFAULT_VERTICAL_FT)
    horizontal_nm = proximity_thresholds.get("horizontal_nm", DEFAULT_HORIZONTAL_NM)
    
    logger.info(f"Planning proximity check: {len(traffic_data)} aircraft, thresholds: {vertical_ft}ft / {horizontal_nm}nm")
    
    # Step 1: Predict paths for all traffic
    predicted_paths = []
    
    for ac_data in traffic_data:
        aircraft = TrafficAircraft(
            flight_id=ac_data.get("flight_id", ""),
            callsign=ac_data.get("callsign"),
            lat=ac_data.get("lat", 0),
            lon=ac_data.get("lon", 0),
            alt_ft=ac_data.get("alt_ft", 0),
            heading_deg=ac_data.get("heading_deg", 0),
            speed_kts=ac_data.get("speed_kts", 0),
            vspeed_fpm=ac_data.get("vspeed_fpm", 0),
            timestamp=ac_data.get("timestamp", int(time.time())),
            is_simulated=ac_data.get("is_simulated", False),
            origin=ac_data.get("origin"),
            destination=ac_data.get("destination"),
        )
        
        origin_code = ac_data.get("origin")
        dest_code = ac_data.get("destination")
        
        predicted_path_result = predict_aircraft_path_unified(
            aircraft=aircraft,
            planner=route_planner,
            duration_minutes=route_duration,
            origin_code=origin_code,
            dest_code=dest_code
        )
        
        if predicted_path_result and predicted_path_result.get("predicted_path"):
            predicted_paths.append({
                "flight_id": aircraft.flight_id,
                "callsign": aircraft.callsign,
                "predicted_path": predicted_path_result["predicted_path"]
            })
    
    # Step 2: Calculate proximity at every point (SAME as /simulate)
    proximity_points = []
    aircraft_with_proximity = set()
    
    for path_point in planned_path:
        for path_data in predicted_paths:
            closest_pred = None
            min_time_diff = float('inf')
            
            for pred in path_data["predicted_path"]:
                time_diff = abs(pred['time_offset_min'] - path_point['time_offset_min'])
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_pred = pred
            
            if closest_pred and min_time_diff <= 5:
                h_dist = haversine_nm(
                    path_point['lat'], path_point['lon'],
                    closest_pred['lat'], closest_pred['lon']
                )
                v_dist = abs(path_point['alt_ft'] - closest_pred['alt_ft'])
                
                if h_dist <= horizontal_nm and v_dist <= vertical_ft:
                    if h_dist <= horizontal_nm * 0.3 and v_dist <= vertical_ft * 0.3:
                        danger_level = "critical"
                    elif h_dist <= horizontal_nm * 0.5 and v_dist <= vertical_ft * 0.5:
                        danger_level = "warning"
                    else:
                        danger_level = "caution"
                    
                    proximity_point = {
                        "time_offset_min": path_point['time_offset_min'],
                        "planned_lat": path_point['lat'],
                        "planned_lon": path_point['lon'],
                        "planned_alt_ft": path_point['alt_ft'],
                        "traffic_flight_id": path_data["flight_id"],
                        "traffic_callsign": path_data["callsign"],
                        "traffic_lat": closest_pred['lat'],
                        "traffic_lon": closest_pred['lon'],
                        "traffic_alt_ft": closest_pred['alt_ft'],
                        "horizontal_distance_nm": h_dist,
                        "vertical_distance_ft": v_dist,
                        "danger_level": danger_level,
                    }
                    
                    proximity_points.append(proximity_point)
                    aircraft_with_proximity.add(path_data["flight_id"])
                    
                    logger.info(f"[PLAN PROXIMITY] {danger_level.upper()}: {path_data['callsign'] or path_data['flight_id']} "
                               f"at t={path_point['time_offset_min']:.1f}min - {h_dist:.1f}nm, {v_dist:.0f}ft")
    
    # Step 3: Generate report (SAME as /simulate)
    critical_count = len([p for p in proximity_points if p["danger_level"] == "critical"])
    warning_count = len([p for p in proximity_points if p["danger_level"] == "warning"])
    caution_count = len([p for p in proximity_points if p["danger_level"] == "caution"])
    
    proximity_report = {
        "total_proximity_points": len(proximity_points),
        "aircraft_with_proximity": len(aircraft_with_proximity),
        "critical_points": critical_count,
        "warning_points": warning_count,
        "caution_points": caution_count,
        "proximity_points": proximity_points,
    }
    
    return proximity_points, proximity_report


# ============================================================================
# Route Planning Endpoint
# ============================================================================

@router.post("/plan", response_model=ManualRouteResponse)
async def plan_manual_route(request: ManualRouteRequest):
    """
    Plan a route based on manually drawn waypoints.
    
    NEW WORKFLOW:
    1. Interpolate waypoints to create smooth path
    2. Calculate route metrics (distance, eta)
    3. Calculate boundary polygon based on route duration
    4. Fetch traffic WITHIN that boundary (real-time)
    5. Predict traffic positions using corridor-based prediction
    6. Detect proximity alerts
    7. Return route + boundary + traffic + analysis
    
    Each waypoint has:
    - Position (lat, lon)
    - Altitude (ft)
    - Speed (kts)
    """
    if len(request.waypoints) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")
    
    # Get route planner (for path prediction)
    global _route_planner
    if _route_planner is None:
        try:
            from service.route_planner import get_route_planner
        except ImportError:
            from route_planner import get_route_planner
        _route_planner = get_route_planner()
    
    # ============================================================
    # STEP 1: Interpolate waypoints to create smooth path
    # ============================================================
    logger.info(f"Planning manual route with {len(request.waypoints)} waypoints")
    planned_path = interpolate_waypoints(request.waypoints)
    
    if not planned_path:
        raise HTTPException(status_code=400, detail="Failed to generate path from waypoints")
    
    # Calculate route metrics
    total_distance = planned_path[-1]["distance_nm"]
    eta_minutes = planned_path[-1]["time_offset_min"]
    
    logger.info(f"Route generated: {len(planned_path)} points, {total_distance:.1f} nm, {eta_minutes:.1f} min")
    
    # ============================================================
    # STEP 2: Calculate boundary polygon based on route duration
    # ============================================================
    boundary_polygon = calculate_route_boundary(planned_path, eta_minutes)
    logger.info(f"Boundary polygon calculated: {len(boundary_polygon)} points")
    
    # ============================================================
    # STEP 3: Fetch traffic WITHIN the boundary (directly from FR24 API)
    # ============================================================
    # Extract bounding box from polygon
    min_lat = min(p[1] for p in boundary_polygon)
    max_lat = max(p[1] for p in boundary_polygon)
    min_lon = min(p[0] for p in boundary_polygon)
    max_lon = max(p[0] for p in boundary_polygon)
    
    logger.info(f"Fetching traffic from FR24 with boundary: lat=[{min_lat:.2f}, {max_lat:.2f}], lon=[{min_lon:.2f}, {max_lon:.2f}]")
    
    # Fetch traffic with custom boundary (passed directly to FR24 API)
    custom_bounds = {
        'north': max_lat,
        'south': min_lat,
        'west': min_lon,
        'east': max_lon
    }
    
    traffic_in_boundary = _route_planner.traffic_manager.fetch_live_traffic(
        force_refresh=False,  # Use cache if available for same bounds
        custom_bounds=custom_bounds
    )
    
    logger.info(f"✅ Fetched {len(traffic_in_boundary)} aircraft from FR24 within boundary")
    
    # Convert to dicts
    traffic_dicts = []
    for aircraft in traffic_in_boundary:
        traffic_dicts.append({
            "flight_id": aircraft.flight_id,
            "callsign": aircraft.callsign,
            "lat": aircraft.lat,
            "lon": aircraft.lon,
            "alt_ft": aircraft.alt_ft,
            "heading_deg": aircraft.heading_deg,
            "speed_kts": aircraft.speed_kts,
            "vspeed_fpm": aircraft.vspeed_fpm,
            "timestamp": aircraft.timestamp,
            "is_simulated": aircraft.is_simulated,
            "origin": aircraft.origin,
            "destination": aircraft.destination,
        })
    
    # ============================================================
    # STEP 4: Detect proximity alerts
    # ============================================================
    # Detect proximity alerts using SAME logic as simulate endpoint
    proximity_thresholds = request.proximity_thresholds or {
        "vertical_ft": DEFAULT_VERTICAL_FT,
        "horizontal_nm": DEFAULT_HORIZONTAL_NM,
        "name": "default"
    }
    
    proximity_points = []
    proximity_report = {}
    
    if request.check_conflicts:
        # Use the same detection logic as /simulate endpoint
        proximity_points, proximity_report = detect_proximity_for_planning(
            planned_path,
            traffic_dicts,
            _route_planner,
            proximity_thresholds,
            eta_minutes
        )
    
    # Calculate proximity score
    proximity_count = proximity_report.get("total_proximity_points", 0)
    critical_count = proximity_report.get("critical_points", 0)
    warning_count = proximity_report.get("warning_points", 0)
    caution_count = proximity_report.get("caution_points", 0)
    
    # Score: 1.0 = perfect (no alerts), lower is worse
    if proximity_count > 0:
        proximity_penalty = min(1.0, proximity_count * 0.1)
        proximity_score = max(0.0, 1.0 - proximity_penalty)
    else:
        proximity_score = 1.0
    
    # Generate summary
    summary_parts = []
    summary_parts.append(f"Route: {len(request.waypoints)} waypoints, {total_distance:.1f} NM, ETA {eta_minutes:.0f} min")
    
    if proximity_count > 0:
        summary_parts.append(f"⚠️ {proximity_count} proximity alert{'s' if proximity_count > 1 else ''}")
        if critical_count > 0:
            summary_parts.append(f"🔴 {critical_count} critical")
        if warning_count > 0:
            summary_parts.append(f"🟡 {warning_count} warnings")
        if caution_count > 0:
            summary_parts.append(f"🟢 {caution_count} cautions")
    else:
        summary_parts.append("✅ No proximity conflicts detected")
    
    summary_parts.append(f"Safety score: {proximity_score * 100:.0f}%")
    
    # Convert waypoints for response
    waypoint_dicts = [
        {
            "lat": wp.lat,
            "lon": wp.lon,
            "alt_ft": wp.alt_ft,
            "speed_kts": wp.speed_kts,
            "name": wp.name or f"WP{i+1}"
        }
        for i, wp in enumerate(request.waypoints)
    ]
    
    # Convert proximity_points to ProximityAlert objects
    proximity_alerts = []
    for p in proximity_points:
        proximity_alerts.append(ProximityAlert(
            time_offset_min=p["time_offset_min"],
            planned_lat=p["planned_lat"],
            planned_lon=p["planned_lon"],
            planned_alt_ft=p["planned_alt_ft"],
            traffic_flight_id=p["traffic_flight_id"],
            traffic_callsign=p["traffic_callsign"],
            traffic_lat=p["traffic_lat"],
            traffic_lon=p["traffic_lon"],
            traffic_alt_ft=p["traffic_alt_ft"],
            horizontal_distance_nm=p["horizontal_distance_nm"],
            vertical_distance_ft=p["vertical_distance_ft"],
            severity=p["danger_level"]
        ))
    
    return ManualRouteResponse(
        route_id=f"manual_{int(time.time())}",
        waypoints=waypoint_dicts,
        planned_path=planned_path,
        total_distance_nm=total_distance,
        eta_minutes=eta_minutes,
        proximity_alerts=proximity_alerts,
        proximity_count=proximity_count,
        proximity_score=proximity_score,
        traffic_count=len(traffic_dicts),
        summary=" | ".join(summary_parts),
        boundary_polygon=boundary_polygon,  # NEW: Include boundary
        traffic=traffic_dicts  # NEW: Include fetched traffic
    )


@router.get("/traffic")
async def get_traffic_for_planning():
    """Get current traffic data for route planning."""
    global _route_planner
    if _route_planner is None:
        try:
            from service.route_planner import get_route_planner
        except ImportError:
            from route_planner import get_route_planner
        _route_planner = get_route_planner()
    
    # Fetch traffic if cache is empty (force=False will use cache if available)
    traffic = _route_planner.traffic_manager.fetch_live_traffic(force_refresh=False)
    
    traffic_data = []
    for aircraft in traffic:
        traffic_data.append({
            "flight_id": aircraft.flight_id,
            "callsign": aircraft.callsign,
            "lat": aircraft.lat,
            "lon": aircraft.lon,
            "alt_ft": aircraft.alt_ft,
            "heading_deg": aircraft.heading_deg,
            "speed_kts": aircraft.speed_kts,
            "vspeed_fpm": aircraft.vspeed_fpm,
            "timestamp": aircraft.timestamp,
            "is_simulated": aircraft.is_simulated,
            "origin": aircraft.origin,
            "destination": aircraft.destination,
        })
    
    logger.info(f"Returning {len(traffic_data)} traffic aircraft")
    
    return {
        "traffic": traffic_data,
        "count": len(traffic_data),
        "timestamp": int(time.time())
    }


@router.post("/traffic/refresh")
async def refresh_traffic():
    """Force refresh traffic data from FR24."""
    global _route_planner
    if _route_planner is None:
        try:
            from service.route_planner import get_route_planner
        except ImportError:
            from route_planner import get_route_planner
        _route_planner = get_route_planner()
    
    traffic = _route_planner.traffic_manager.fetch_live_traffic(force_refresh=True)
    
    traffic_data = []
    for aircraft in traffic:
        traffic_data.append({
            "flight_id": aircraft.flight_id,
            "callsign": aircraft.callsign,
            "lat": aircraft.lat,
            "lon": aircraft.lon,
            "alt_ft": aircraft.alt_ft,
            "heading_deg": aircraft.heading_deg,
            "speed_kts": aircraft.speed_kts,
            "vspeed_fpm": aircraft.vspeed_fpm,
            "timestamp": aircraft.timestamp,
            "is_simulated": aircraft.is_simulated,
            "origin": aircraft.origin,
            "destination": aircraft.destination,
        })
    
    return {
        "traffic": traffic_data,
        "count": len(traffic_data),
        "timestamp": int(time.time())
    }


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_route(request: SimulationRequest):
    """
    Complete simulation analysis - ALL calculations in ONE place.
    
    Takes:
    - Planned route path
    - Traffic aircraft data
    - Proximity thresholds
    
    Returns:
    - Predicted paths for all traffic (corridor-based when possible)
    - Complete proximity report with all trigger points
    - Summary statistics
    
    This ensures all calculations happen server-side in one place.
    """
    global _route_planner
    if _route_planner is None:
        try:
            from service.route_planner import get_route_planner
        except ImportError:
            from route_planner import get_route_planner
        _route_planner = get_route_planner()
    
    # Import dependencies
    try:
        from service.routes.route_planning import predict_aircraft_path_unified
        from service.route_planner import TrafficAircraft
    except ImportError:
        from routes.route_planning import predict_aircraft_path_unified
        from route_planner import TrafficAircraft
    
    planned_path = request.planned_path
    traffic_data = request.traffic
    thresholds = request.proximity_thresholds
    route_duration = request.route_duration_min
    
    vertical_ft = thresholds.get("vertical_ft", DEFAULT_VERTICAL_FT)
    horizontal_nm = thresholds.get("horizontal_nm", DEFAULT_HORIZONTAL_NM)
    
    logger.info(f"Simulating route with {len(traffic_data)} aircraft, duration {route_duration:.1f} min, "
               f"thresholds: {vertical_ft}ft / {horizontal_nm}nm")
    
    # Debug log sample traffic data
    if traffic_data:
        sample = traffic_data[0]
        logger.info(f"Sample traffic data: flight_id={sample.get('flight_id')}, "
                   f"callsign={sample.get('callsign')}, "
                   f"lat={sample.get('lat')}, lon={sample.get('lon')}, alt_ft={sample.get('alt_ft')}, "
                   f"origin={sample.get('origin')}, dest={sample.get('destination')}")
    
    # ============================================================
    # STEP 1: Predict paths for all traffic aircraft
    # ============================================================
    predicted_paths = []
    
    for ac_data in traffic_data:
        # Create TrafficAircraft object
        aircraft = TrafficAircraft(
            flight_id=ac_data.get("flight_id", ""),
            callsign=ac_data.get("callsign"),
            lat=ac_data.get("lat", 0),
            lon=ac_data.get("lon", 0),
            alt_ft=ac_data.get("alt_ft", 0),
            heading_deg=ac_data.get("heading_deg", 0),
            speed_kts=ac_data.get("speed_kts", 0),
            vspeed_fpm=ac_data.get("vspeed_fpm", 0),
            timestamp=ac_data.get("timestamp", int(time.time())),
            is_simulated=ac_data.get("is_simulated", False),
            origin=ac_data.get("origin") or ac_data.get("origin_airport"),
            destination=ac_data.get("destination") or ac_data.get("destination_airport"),
        )
        
        # Get origin/destination
        origin_code = ac_data.get("origin") or ac_data.get("origin_airport")
        dest_code = ac_data.get("destination") or ac_data.get("destination_airport")
        
        # Use unified prediction (corridor-based when possible)
        predicted_path_result = predict_aircraft_path_unified(
            aircraft=aircraft,
            planner=_route_planner,
            duration_minutes=route_duration,
            origin_code=origin_code,
            dest_code=dest_code
        )
        
        if predicted_path_result and predicted_path_result.get("predicted_path"):
            pred_path_points = predicted_path_result["predicted_path"]
            path_data = {
                "flight_id": aircraft.flight_id,
                "callsign": aircraft.callsign,
                "origin": origin_code or "Unknown",
                "destination": dest_code or "Unknown",
                "current_position": {
                    "lat": aircraft.lat,
                    "lon": aircraft.lon,
                    "alt_ft": aircraft.alt_ft,
                    "speed_kts": aircraft.speed_kts,
                    "heading_deg": aircraft.heading_deg,
                },
                "corridor_id": predicted_path_result.get("corridor_id", "direct"),
                "corridor_name": predicted_path_result.get("corridor_name", "Direct Path"),
                "distance_to_corridor_nm": predicted_path_result.get("distance_to_corridor_nm", 0),
                "predicted_path": pred_path_points,
                "total_points": len(pred_path_points),
                "total_distance_nm": predicted_path_result.get("total_distance_nm", 0),
                "eta_minutes": predicted_path_result.get("eta_minutes", 0),
                "speed_kts": aircraft.speed_kts,
            }
            predicted_paths.append(path_data)
            
            # Debug log first few predictions
            if len(predicted_paths) <= 3:
                logger.info(f"Predicted path for {aircraft.callsign or aircraft.flight_id}: "
                           f"{len(pred_path_points)} points, corridor={predicted_path_result.get('corridor_name', 'Direct')}, "
                           f"first_point=({pred_path_points[0]['lat']:.4f},{pred_path_points[0]['lon']:.4f},{pred_path_points[0]['alt_ft']:.0f}ft)")
    
    logger.info(f"Predicted {len(predicted_paths)} paths out of {len(traffic_data)} aircraft")
    
    # ============================================================
    # STEP 2: Calculate proximity at every point
    # ============================================================
    proximity_points = []
    aircraft_with_proximity = set()  # Track which aircraft triggered proximity
    
    logger.info(f"Starting proximity check: {len(planned_path)} route points vs {len(predicted_paths)} predicted paths")
    
    # Debug route time range
    if planned_path:
        first_point = planned_path[0]
        last_point = planned_path[-1]
        logger.info(f"Route time range: {first_point.get('time_offset_min'):.1f} to {last_point.get('time_offset_min'):.1f} min")
        logger.info(f"Route altitude range: {first_point.get('alt_ft'):.0f} to {last_point.get('alt_ft'):.0f} ft")
    
    # Debug first predicted path time range
    if predicted_paths:
        first_pred = predicted_paths[0]
        pred_path = first_pred["predicted_path"]
        if pred_path:
            logger.info(f"Example predicted path ({first_pred['flight_id']}): {len(pred_path)} points, "
                       f"time range: {pred_path[0]['time_offset_min']:.1f} to {pred_path[-1]['time_offset_min']:.1f} min, "
                       f"alt range: {pred_path[0]['alt_ft']:.0f} to {pred_path[-1]['alt_ft']:.0f} ft")
    
    for path_point_idx, path_point in enumerate(planned_path):
        for path_data in predicted_paths:
            # Find closest predicted position to this time
            closest_pred = None
            min_time_diff = float('inf')
            
            for pred in path_data["predicted_path"]:
                time_diff = abs(pred['time_offset_min'] - path_point['time_offset_min'])
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_pred = pred
            
            if closest_pred and min_time_diff <= 5:  # Within 5 minutes
                # Calculate distances
                h_dist = haversine_nm(
                    path_point['lat'], path_point['lon'],
                    closest_pred['lat'], closest_pred['lon']
                )
                v_dist = abs(path_point['alt_ft'] - closest_pred['alt_ft'])
                
                # Debug logging for first few checks
                if path_point_idx < 3 and len(proximity_points) < 5:
                    logger.info(f"Proximity check at t={path_point['time_offset_min']:.1f}min: "
                               f"route=({path_point['lat']:.4f},{path_point['lon']:.4f},{path_point['alt_ft']:.0f}ft) vs "
                               f"{path_data['flight_id']}=({closest_pred['lat']:.4f},{closest_pred['lon']:.4f},{closest_pred['alt_ft']:.0f}ft) "
                               f"→ {h_dist:.1f}nm, {v_dist:.0f}ft (threshold: {horizontal_nm}nm, {vertical_ft}ft)")
                
                # Check if within threshold
                if h_dist <= horizontal_nm and v_dist <= vertical_ft:
                    # Determine danger level
                    if h_dist <= horizontal_nm * 0.3 and v_dist <= vertical_ft * 0.3:
                        danger_level = "critical"
                    elif h_dist <= horizontal_nm * 0.5 and v_dist <= vertical_ft * 0.5:
                        danger_level = "warning"
                    else:
                        danger_level = "caution"
                    
                    proximity_point = {
                        "time_offset_min": path_point['time_offset_min'],
                        "planned_lat": path_point['lat'],
                        "planned_lon": path_point['lon'],
                        "planned_alt_ft": path_point['alt_ft'],
                        "traffic_flight_id": path_data["flight_id"],
                        "traffic_callsign": path_data["callsign"],
                        "traffic_lat": closest_pred['lat'],
                        "traffic_lon": closest_pred['lon'],
                        "traffic_alt_ft": closest_pred['alt_ft'],
                        "horizontal_distance_nm": h_dist,
                        "vertical_distance_ft": v_dist,
                        "danger_level": danger_level,
                    }
                    
                    proximity_points.append(proximity_point)
                    aircraft_with_proximity.add(path_data["flight_id"])
                    
                    logger.info(f"[PROXIMITY DETECTED] {danger_level.upper()}: {path_data['callsign'] or path_data['flight_id']} at t={path_point['time_offset_min']:.1f}min - {h_dist:.1f}nm, {v_dist:.0f}ft")
    
    # ============================================================
    # STEP 3: Generate comprehensive report
    # ============================================================
    critical_count = len([p for p in proximity_points if p["danger_level"] == "critical"])
    warning_count = len([p for p in proximity_points if p["danger_level"] == "warning"])
    caution_count = len([p for p in proximity_points if p["danger_level"] == "caution"])
    
    proximity_report = {
        "total_proximity_points": len(proximity_points),
        "aircraft_with_proximity": len(aircraft_with_proximity),
        "total_aircraft": len(traffic_data),
        "critical_points": critical_count,
        "warning_points": warning_count,
        "caution_points": caution_count,
        "proximity_points": proximity_points,
        "thresholds_used": {
            "vertical_ft": vertical_ft,
            "horizontal_nm": horizontal_nm,
        },
        # Group by aircraft for detailed breakdown
        "by_aircraft": {}
    }
    
    # Group proximity points by aircraft
    for point in proximity_points:
        flight_id = point["traffic_flight_id"]
        if flight_id not in proximity_report["by_aircraft"]:
            proximity_report["by_aircraft"][flight_id] = {
                "flight_id": flight_id,
                "callsign": point["traffic_callsign"],
                "proximity_count": 0,
                "critical_count": 0,
                "warning_count": 0,
                "caution_count": 0,
                "points": []
            }
        
        aircraft_data = proximity_report["by_aircraft"][flight_id]
        aircraft_data["proximity_count"] += 1
        aircraft_data["points"].append(point)
        
        if point["danger_level"] == "critical":
            aircraft_data["critical_count"] += 1
        elif point["danger_level"] == "warning":
            aircraft_data["warning_count"] += 1
        else:
            aircraft_data["caution_count"] += 1
    
    # Convert by_aircraft dict to list
    proximity_report["by_aircraft"] = list(proximity_report["by_aircraft"].values())
    
    # ============================================================
    # STEP 4: Generate summary
    # ============================================================
    summary_parts = []
    summary_parts.append(f"Simulation: {len(predicted_paths)} aircraft paths predicted")
    
    if len(proximity_points) > 0:
        summary_parts.append(f"⚠️ {len(proximity_points)} proximity trigger points detected")
        summary_parts.append(f"🔴 {critical_count} critical")
        summary_parts.append(f"🟡 {warning_count} warnings")
        summary_parts.append(f"🟢 {caution_count} cautions")
        summary_parts.append(f"🛩️ {len(aircraft_with_proximity)} aircraft involved")
    else:
        summary_parts.append("✅ No proximity conflicts detected")
    
    summary_parts.append(f"Thresholds: {vertical_ft}ft / {horizontal_nm}nm")
    
    logger.info(f"Simulation complete: {len(proximity_points)} proximity points, "
               f"{len(aircraft_with_proximity)} aircraft involved")
    
    # ============================================================
    # STEP 5: Calculate route boundary polygon
    # ============================================================
    boundary_polygon = calculate_route_boundary(planned_path, route_duration)
    
    return SimulationResponse(
        predicted_paths=predicted_paths,
        proximity_report=proximity_report,
        boundary_polygon=boundary_polygon,
        summary=" | ".join(summary_parts)
    )


# ============================================================================
# Configuration
# ============================================================================

def configure(route_planner_instance=None):
    """Configure the manual route planner with dependencies."""
    global _route_planner
    _route_planner = route_planner_instance
