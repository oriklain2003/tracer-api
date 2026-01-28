"""
Route planning routes - route planning, traffic, conflict detection, strike planning.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests as http_requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Aviation Edge API for scheduled flights
AVIATION_EDGE_API_KEY = os.getenv("AVIATION_EDGE_API_KEY")
if not AVIATION_EDGE_API_KEY:
    logger.warning("AVIATION_EDGE_API_KEY not set in environment variables")
AVIATION_EDGE_BASE_URL = "https://aviation-edge.com/v2/public/flightsFuture"

router = APIRouter(tags=["Route Planning"])

# Function/class references from api.py
_get_route_planner = None
_get_strike_generator = None
_FIGHTER_JET_PROFILE = None
_CIVIL_AIRCRAFT_PROFILE = None
PROJECT_ROOT: Path = None
CONFIGURED_AIRPORTS: List[Dict[str, Any]] = []


def _load_airports_config(project_root: Path) -> List[Dict[str, Any]]:
    """Load airports from rule_config.json."""
    try:
        config_path = project_root / "rules" / "rule_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("airports", [])
    except Exception as e:
        logger.warning(f"Could not load airports from config: {e}")
        return []


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in nautical miles."""
    R = 3440.065  # Earth radius in nautical miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def destination_point(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> tuple[float, float]:
    """
    Calculate destination point given start, bearing, and distance.
    
    Args:
        lat: Starting latitude in degrees
        lon: Starting longitude in degrees
        bearing_deg: Bearing in degrees (0-360)
        distance_nm: Distance in nautical miles
        
    Returns:
        Tuple of (destination_lat, destination_lon) in degrees
    """
    EARTH_RADIUS_NM = 3440.065
    
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    bearing_r = math.radians(bearing_deg)
    
    angular_dist = distance_nm / EARTH_RADIUS_NM
    
    lat2 = math.asin(
        math.sin(lat_r) * math.cos(angular_dist) +
        math.cos(lat_r) * math.sin(angular_dist) * math.cos(bearing_r)
    )
    
    lon2 = lon_r + math.atan2(
        math.sin(bearing_r) * math.sin(angular_dist) * math.cos(lat_r),
        math.cos(angular_dist) - math.sin(lat_r) * math.sin(lat2)
    )
    
    return math.degrees(lat2), math.degrees(lon2)


def _get_speed_for_altitude(base_speed_kts: float, altitude_ft: float, is_descending: bool) -> float:
    """
    Calculate appropriate speed based on altitude and descent phase.
    
    Aircraft slow down during descent and approach:
    - Above 10,000 ft: cruise speed
    - 3,000-10,000 ft: ~250 kts (speed restriction below FL100)
    - 1,000-3,000 ft: ~180 kts (approach speed)
    - Below 1,000 ft: ~140 kts (final approach)
    
    Args:
        base_speed_kts: Current aircraft speed
        altitude_ft: Current altitude
        is_descending: Whether aircraft is descending (vspeed < -100 fpm)
    
    Returns:
        Adjusted speed in knots
    """
    # If not descending, use base speed
    if not is_descending:
        return base_speed_kts
    
    # Apply speed restrictions based on altitude during descent
    if altitude_ft >= 10000:
        # High altitude - use current speed (cruise)
        return base_speed_kts
    elif altitude_ft >= 3000:
        # Medium altitude - FAA speed limit below 10,000 ft is 250 kts
        return min(base_speed_kts, 250)
    elif altitude_ft >= 1000:
        # Low altitude - approach phase (~180 kts)
        return min(base_speed_kts, 180)
    else:
        # Very low altitude - final approach (~140 kts)
        return min(base_speed_kts, 140)


def _generate_straight_line_path(aircraft, duration_minutes: float = 30, interval_minutes: float = 5) -> Dict[str, Any]:
    """
    Generate straight-line path based on current heading and speed.
    Now includes dynamic speed adjustment for descending aircraft.
    
    Args:
        aircraft: TrafficAircraft object with lat, lon, heading_deg, speed_kts, alt_ft
        duration_minutes: How far ahead to predict (default 30 minutes)
        interval_minutes: Time interval between prediction points (default 5 minutes)
        
    Returns:
        Dict with predicted path in same format as corridor-based paths
    """
    predicted_path = []
    cumulative_distance_nm = 0
    
    # Check if aircraft is descending
    is_descending = aircraft.vspeed_fpm < -100  # Descending if vspeed < -100 fpm
    
    # Use aircraft speed, handle stationary aircraft (<50 kts) and low speeds
    if aircraft.speed_kts < 50:
        # Stationary or very slow aircraft - stays in place
        base_speed_kts = 0
    elif aircraft.speed_kts < 100:
        # Low speed but moving - use actual speed
        base_speed_kts = aircraft.speed_kts
    else:
        # Normal speed
        base_speed_kts = aircraft.speed_kts
    
    # Generate points at regular intervals
    num_points = int(duration_minutes / interval_minutes) + 1
    
    current_lat = aircraft.lat
    current_lon = aircraft.lon
    current_alt = aircraft.alt_ft
    
    for i in range(num_points):
        time_offset = i * interval_minutes
        
        # Calculate altitude at this time point
        alt_change = aircraft.vspeed_fpm * time_offset
        new_alt = max(0, current_alt + alt_change)
        
        # Get appropriate speed for this altitude
        speed_kts = _get_speed_for_altitude(base_speed_kts, new_alt, is_descending)
        
        # Calculate position for this time step
        if i == 0:
            # First point - use current position
            new_lat, new_lon = current_lat, current_lon
        else:
            # Calculate distance traveled in this interval
            prev_point = predicted_path[-1]
            prev_alt = prev_point['alt_ft']
            
            # Get speed at previous altitude
            prev_speed = _get_speed_for_altitude(base_speed_kts, prev_alt, is_descending)
            
            # Average speed during this interval
            avg_speed = (prev_speed + speed_kts) / 2
            
            # Distance traveled in this interval
            interval_distance_nm = avg_speed * (interval_minutes / 60.0)
            
            # Calculate new position
            if avg_speed > 0:
                new_lat, new_lon = destination_point(
                    prev_point['lat'], prev_point['lon'],
                    aircraft.heading_deg, interval_distance_nm
                )
            else:
                # Stationary - stay in place
                new_lat, new_lon = prev_point['lat'], prev_point['lon']
            
            # Update cumulative distance
            segment_distance = haversine_distance_nm(
                prev_point['lat'], prev_point['lon'],
                new_lat, new_lon
            )
            cumulative_distance_nm += segment_distance
        
        predicted_path.append({
            "lat": new_lat,
            "lon": new_lon,
            "alt_ft": new_alt,
            "speed_kts": speed_kts,  # Include speed at this point
            "time_offset_min": time_offset,
            "distance_nm": round(cumulative_distance_nm, 2),
        })
    
    return {
        "flight_id": aircraft.flight_id,
        "callsign": aircraft.callsign,
        "origin": None,
        "destination": None,
        "current_position": {
            "lat": aircraft.lat,
            "lon": aircraft.lon,
            "alt_ft": aircraft.alt_ft,
            "speed_kts": aircraft.speed_kts,
            "heading_deg": aircraft.heading_deg,
        },
        "corridor_id": "straight_line",
        "corridor_name": "Straight-line projection",
        "distance_to_corridor_nm": 0,
        "predicted_path": predicted_path,
        "total_points": len(predicted_path),
        "total_distance_nm": round(cumulative_distance_nm, 2),
        "eta_minutes": round(duration_minutes, 2),
        "speed_kts": base_speed_kts,
    }


def predict_aircraft_path_unified(
    aircraft,
    planner,
    duration_minutes: float = 30,
    origin_code: Optional[str] = None,
    dest_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Unified aircraft path prediction using corridor-based method when possible.
    This is used by BOTH route planning proximity checks AND the predict-batch endpoint
    to ensure consistency.
    
    Args:
        aircraft: Aircraft object with lat, lon, heading_deg, speed_kts, alt_ft, vspeed_fpm
        planner: Route planner instance (for corridor manager access)
        duration_minutes: How far ahead to predict
        origin_code: Optional origin airport code (enables corridor matching)
        dest_code: Optional destination airport code (enables corridor matching)
        
    Returns:
        Dict with predicted_path: List[{lat, lon, alt_ft, time_offset_min, distance_nm}]
    """
    # Try corridor-based prediction if O/D available
    if origin_code and dest_code and planner:
        corridors = planner.corridor_manager.find_direct_corridors(origin_code, dest_code)
        
        if not corridors:
            # Try nearby corridors
            from service.route_planner import Waypoint
            current_pos = Waypoint(lat=aircraft.lat, lon=aircraft.lon, alt_ft=aircraft.alt_ft)
            nearby = planner.corridor_manager.find_nearby_corridors(current_pos, max_distance_nm=30, limit=3)
            if nearby:
                corridors = [nearby[0][0]]
        
        if corridors:
            MAX_DISTANCE_CAP_NM = 3.0
            corridor_distances = []
            
            for corridor in corridors:
                centerline = corridor.get("centerline", [])
                if len(centerline) < 3:
                    continue
                
                # Find closest point on corridor
                min_dist = float('inf')
                closest_idx = 0
                for i, point in enumerate(centerline):
                    dist = haversine_distance_nm(
                        aircraft.lat, aircraft.lon,
                        point.get('lat', 0), point.get('lon', 0)
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                if min_dist <= MAX_DISTANCE_CAP_NM:
                    corridor_distances.append({
                        'corridor': corridor,
                        'distance': min_dist,
                        'closest_idx': closest_idx,
                        'member_count': corridor.get('member_count', 0)
                    })
            
            if corridor_distances:
                # Sort by distance and take top 3, then pick highest member_count
                corridor_distances.sort(key=lambda x: x['distance'])
                top_3_closest = corridor_distances[:3]
                best_candidate = max(top_3_closest, key=lambda x: x['member_count'])
                best_corridor = best_candidate['corridor']
                best_closest_idx = best_candidate['closest_idx']
                
                # Use corridor path
                centerline = best_corridor.get("centerline", [])
                remaining_path = centerline[best_closest_idx:]
                
                if len(remaining_path) >= 2:
                    # Calculate time-based positions along corridor with dynamic speed
                    base_speed = aircraft.speed_kts if aircraft.speed_kts > 100 else 450
                    
                    # Detect if descending by checking if altitude decreases along corridor
                    first_alt = remaining_path[0].get('alt', aircraft.alt_ft)
                    last_alt = remaining_path[-1].get('alt', aircraft.alt_ft)
                    is_descending = last_alt < first_alt - 1000  # Descending if drops more than 1000ft
                    
                    timed_positions = []
                    cumulative_distance_nm = 0
                    cumulative_time_min = 0
                    
                    for i, point in enumerate(remaining_path):
                        point_alt = point.get('alt', aircraft.alt_ft)
                        
                        # Get appropriate speed for this altitude
                        current_speed = _get_speed_for_altitude(base_speed, point_alt, is_descending)
                        
                        if i > 0:
                            prev_point = remaining_path[i - 1]
                            prev_alt = prev_point.get('alt', aircraft.alt_ft)
                            prev_speed = _get_speed_for_altitude(base_speed, prev_alt, is_descending)
                            
                            # Use average speed for this segment
                            avg_speed = (prev_speed + current_speed) / 2
                            
                            segment_distance = haversine_distance_nm(
                                prev_point.get('lat', 0), prev_point.get('lon', 0),
                                point.get('lat', 0), point.get('lon', 0)
                            )
                            cumulative_distance_nm += segment_distance
                            cumulative_time_min += (segment_distance / avg_speed) * 60 if avg_speed > 0 else 0
                        
                        timed_positions.append({
                            'lat': point.get('lat'),
                            'lon': point.get('lon'),
                            'alt_ft': point_alt,
                            'speed_kts': current_speed,  # Include speed at this point
                            'time_offset_min': cumulative_time_min,
                            'distance_nm': cumulative_distance_nm,
                        })
                    
                    logger.debug(f"Using corridor path for {getattr(aircraft, 'flight_id', 'UNKNOWN')} "
                               f"({len(timed_positions)} points, {cumulative_distance_nm:.1f}nm, "
                               f"descending={is_descending})")
                    return {'predicted_path': timed_positions}
    
    # Fallback to straight-line prediction
    result = _generate_straight_line_path(aircraft, duration_minutes=duration_minutes, interval_minutes=2)
    logger.debug(f"Using straight-line path for {getattr(aircraft, 'flight_id', 'UNKNOWN')}")
    return {'predicted_path': result['predicted_path']}


def configure(
    get_route_planner_func,
    get_strike_generator_func,
    fighter_jet_profile,
    civil_aircraft_profile,
    project_root: Path = None,
):
    """Configure the router with dependencies from api.py"""
    global _get_route_planner, _get_strike_generator, _FIGHTER_JET_PROFILE, _CIVIL_AIRCRAFT_PROFILE
    global PROJECT_ROOT, CONFIGURED_AIRPORTS, _traffic_snapshots_db
    
    _get_route_planner = get_route_planner_func
    _get_strike_generator = get_strike_generator_func
    _FIGHTER_JET_PROFILE = fighter_jet_profile
    _CIVIL_AIRCRAFT_PROFILE = civil_aircraft_profile
    
    if project_root:
        PROJECT_ROOT = project_root
        CONFIGURED_AIRPORTS = _load_airports_config(project_root)
        
        # Initialize traffic snapshots database
        from service.traffic_snapshots_db import TrafficSnapshotsDB
        traffic_snapshots_db_path = project_root / "service" / "traffic_snapshots.db"
        _traffic_snapshots_db = TrafficSnapshotsDB(traffic_snapshots_db_path)
        logger.info(f"Traffic snapshots database initialized at {traffic_snapshots_db_path}")


# Request/Response Models
class RoutePlanRequest(BaseModel):
    """Request model for route planning."""
    origins: List[str]  # 1-3 origin airport codes
    destination: str    # Destination airport code


class WaypointModel(BaseModel):
    """A waypoint (custom point or airport)."""
    lat: float
    lon: float
    alt_ft: Optional[float] = None
    name: Optional[str] = None
    airport_code: Optional[str] = None


class TacticalZoneModel(BaseModel):
    """Model for tactical zones (no-fly, altitude, speed zones)."""
    id: str
    type: str  # 'low-altitude', 'high-altitude', 'slow-speed', 'high-speed', 'no-fly'
    points: List[Dict[str, float]]  # [{lat, lon}, ...]
    altitude: Optional[float] = None
    speed: Optional[float] = None


class ProximityThresholds(BaseModel):
    """Proximity threshold settings for danger scoring."""
    vertical_ft: float = 999  # Default vertical separation
    horizontal_nm: float = 5  # Default horizontal separation
    name: str = "default"  # "default" or "medium"


class AdvancedRoutePlanRequest(BaseModel):
    """Request model for advanced route planning."""
    origin: WaypointModel
    destination: WaypointModel
    waypoints: Optional[List[WaypointModel]] = None
    aircraft_type: str = "civil"  # "fighter" or "civil"
    altitude_ft: Optional[float] = None
    speed_kts: Optional[float] = None
    check_conflicts: bool = True
    tactical_zones: Optional[List[TacticalZoneModel]] = None
    proximity_thresholds: Optional[ProximityThresholds] = None


class SimulatedAircraftRequest(BaseModel):
    """Request model for adding a simulated aircraft."""
    flight_id: str
    path: List[Dict[str, float]]  # [{lat, lon}, ...]
    speed_kts: float
    altitude_ft: float
    callsign: Optional[str] = None


class ConflictCheckRequest(BaseModel):
    """Request model for checking conflicts on a path."""
    path: List[Dict[str, float]]  # [{lat, lon, alt, time_offset_min}, ...]
    aircraft_type: str = "civil"


class AttackTargetModel(BaseModel):
    """Model for an attack target."""
    id: str
    lat: float
    lon: float
    priority: str = "medium"  # "high", "medium", "low"
    name: Optional[str] = None
    ammoRequired: int = 1


class MissionAircraftModel(BaseModel):
    """Model for a mission aircraft."""
    id: str
    callsign: str
    ammoCapacity: int = 4
    color: Optional[str] = "#22c55e"
    type: Optional[str] = "F-16"  # "F-16" or "F-35"


class StrikePlanRequest(BaseModel):
    """Request model for strike mission planning."""
    origin: WaypointModel
    targets: List[AttackTargetModel]
    aircraft: List[MissionAircraftModel]
    aircraft_type: str = "fighter"  # "fighter" or "civil"
    tactical_zones: Optional[List[TacticalZoneModel]] = None
    return_to_base: bool = True


# Endpoints
@router.get("/api/route/airports")
def get_route_airports():
    """
    Get list of available airports for route planning.
    
    Returns:
        List of airports with code, name, lat, lon, elevation
    """
    try:
        planner = _get_route_planner()
        airports = planner.get_airports()
        
        # Get which airports have paths in the library
        origins_with_paths = planner.get_available_origins()
        destinations_with_paths = planner.get_available_destinations()
        
        # Enrich airports with path availability info
        for airport in airports:
            code = airport["code"]
            airport["has_origin_paths"] = code in origins_with_paths
            airport["has_destination_paths"] = code in destinations_with_paths
        
        return {
            "airports": airports,
            "total": len(airports),
            "origins_with_paths": origins_with_paths,
            "destinations_with_paths": destinations_with_paths,
        }
    except Exception as e:
        logger.error(f"Error getting airports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/route/plan")
def plan_route(request: RoutePlanRequest):
    """
    Calculate optimal routes from multiple origins to a destination.
    
    Uses the learned path library to find matching routes and scores them.
    """
    try:
        if not request.origins:
            raise HTTPException(status_code=400, detail="At least one origin airport is required")
        if len(request.origins) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 origin airports allowed")
        if not request.destination:
            raise HTTPException(status_code=400, detail="Destination airport is required")
        
        planner = _get_route_planner()
        result = planner.plan_routes(request.origins, request.destination)
        
        if "error" in result and not result.get("routes"):
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error planning route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/route/path/{path_id}")
def get_route_path(path_id: str):
    """
    Get detailed path geometry for a specific path ID.
    """
    try:
        planner = _get_route_planner()
        path = planner.get_path_by_id(path_id)
        
        if not path:
            raise HTTPException(status_code=404, detail=f"Path '{path_id}' not found")
        
        # Calculate additional stats
        from service.route_planner import calculate_path_distance
        centerline = path.get("centerline", [])
        distance_nm = calculate_path_distance(centerline)
        
        return {
            "path_id": path.get("id"),
            "origin": path.get("origin"),
            "destination": path.get("destination"),
            "centerline": centerline,
            "width_nm": path.get("width_nm", 8.0),
            "distance_nm": round(distance_nm, 2),
            "waypoint_count": len(centerline),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting path {path_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/route/profiles")
def get_aircraft_profiles():
    """
    Get available aircraft profiles (Fighter Jet vs Civil Aircraft).
    """
    try:
        planner = _get_route_planner()
        return {
            "profiles": planner.get_aircraft_profiles(),
        }
    except Exception as e:
        logger.error(f"Error getting aircraft profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/route/plan-advanced")
def plan_advanced_route(request: AdvancedRoutePlanRequest):
    """
    Advanced route planning with custom waypoints and conflict detection.
    
    Features:
    - Custom waypoints (click on map, not just airports)
    - Aircraft profile selection (fighter/civil)
    - Smart corridor-aware path generation
    - Real-time traffic conflict detection
    - Configurable proximity thresholds for danger scoring
    """
    try:
        planner = _get_route_planner()
        
        zones = None
        if request.tactical_zones:
            zones = [zone.model_dump() for zone in request.tactical_zones]
        
        # Extract proximity thresholds if provided
        proximity_thresholds = None
        if request.proximity_thresholds:
            proximity_thresholds = {
                'vertical_ft': request.proximity_thresholds.vertical_ft,
                'horizontal_nm': request.proximity_thresholds.horizontal_nm,
                'name': request.proximity_thresholds.name
            }
        
        result = planner.plan_advanced_route(
            origin=request.origin.model_dump(),
            destination=request.destination.model_dump(),
            waypoints=[wp.model_dump() for wp in request.waypoints] if request.waypoints else None,
            aircraft_type=request.aircraft_type,
            altitude_ft=request.altitude_ft,
            speed_kts=request.speed_kts,
            check_conflicts=request.check_conflicts,
            tactical_zones=zones,
            proximity_thresholds=proximity_thresholds,
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in advanced route planning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/route/traffic")
def get_route_traffic():
    """
    Get current traffic in the airspace (cached data).
    """
    try:
        planner = _get_route_planner()
        traffic = planner.traffic_manager.get_traffic()
        cache_info = planner.traffic_manager.get_cache_info()
        
        return {
            "traffic": [t.to_dict() for t in traffic],
            "cache_info": cache_info,
        }
    except Exception as e:
        logger.error(f"Error getting traffic: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/route/traffic/refresh")
def refresh_route_traffic():
    """
    Refresh traffic data from FR24 API and calculate predicted paths.
    """
    try:
        planner = _get_route_planner()
        traffic = planner.traffic_manager.fetch_live_traffic(force_refresh=True)
        cache_info = planner.traffic_manager.get_cache_info()
        
        # Calculate predicted paths for each aircraft
        traffic_with_paths = []
        for aircraft in traffic:
            aircraft_dict = aircraft.to_dict()
            
            # Try to calculate predicted path based on learned corridors
            try:
                origin_code = getattr(aircraft, 'origin', None)
                dest_code = getattr(aircraft, 'destination', None)
                
                if origin_code and dest_code:
                    # Find matching corridors for this O/D pair
                    corridors = planner.corridor_manager.find_direct_corridors(
                        origin_code, dest_code
                    )
                    
                    if not corridors:
                        # Try nearby corridors if no direct match
                        from service.route_planner import Waypoint
                        current_pos = Waypoint(
                            lat=aircraft.lat,
                            lon=aircraft.lon,
                            alt_ft=aircraft.alt_ft
                        )
                        nearby = planner.corridor_manager.find_nearby_corridors(
                            current_pos, max_distance_nm=30, limit=3
                        )
                        if nearby:
                            corridors = [nearby[0][0]]
                    
                    if corridors:
                        # Find the best matching corridor:
                        # 1. Get top 3 closest corridors within 3nm cap
                        # 2. Choose the one with most flight members (highest member_count)
                        
                        MAX_DISTANCE_CAP_NM = 3.0  # 3 nautical mile cap
                        
                        # Calculate distance for each corridor
                        corridor_distances = []
                        for corridor in corridors:
                            centerline = corridor.get("centerline", [])
                            if len(centerline) < 3:
                                continue
                            
                            # Find closest point on this corridor
                            min_dist = float('inf')
                            closest_idx = 0
                            for i, point in enumerate(centerline):
                                dist = haversine_distance_nm(
                                    aircraft.lat, aircraft.lon,
                                    point.get('lat', 0), point.get('lon', 0)
                                )
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_idx = i
                            
                            # Only consider corridors within 3nm cap
                            if min_dist <= MAX_DISTANCE_CAP_NM:
                                corridor_distances.append({
                                    'corridor': corridor,
                                    'distance': min_dist,
                                    'closest_idx': closest_idx,
                                    'member_count': corridor.get('member_count', 0)
                                })
                        
                        # Sort by distance (ascending) and take top 3 closest
                        corridor_distances.sort(key=lambda x: x['distance'])
                        top_3_closest = corridor_distances[:3]
                        
                        # Among top 3 closest, choose the one with highest member_count
                        best_corridor = None
                        min_distance_to_corridor = float('inf')
                        best_closest_idx = 0
                        
                        if top_3_closest:
                            # Sort by member_count (descending) to get the most popular one
                            best_candidate = max(top_3_closest, key=lambda x: x['member_count'])
                            best_corridor = best_candidate['corridor']
                            min_distance_to_corridor = best_candidate['distance']
                            best_closest_idx = best_candidate['closest_idx']
                            
                            logger.debug(f"Selected corridor for {aircraft.flight_id}: distance={min_distance_to_corridor:.1f}nm, "
                                       f"member_count={best_candidate['member_count']} "
                                       f"(from {len(top_3_closest)} candidates within 3nm)")
                        
                        if best_corridor:
                            centerline = best_corridor.get("centerline", [])
                            remaining_path = centerline[best_closest_idx:]
                            
                            if len(remaining_path) >= 2:
                                # Calculate time-based positions along the path
                                speed_kts = aircraft.speed_kts if aircraft.speed_kts > 100 else 450
                                
                                timed_positions = []
                                cumulative_distance_nm = 0
                                cumulative_time_min = 0
                                
                                for i, point in enumerate(remaining_path):
                                    if i > 0:
                                        prev_point = remaining_path[i - 1]
                                        segment_distance = haversine_distance_nm(
                                            prev_point.get('lat', 0), prev_point.get('lon', 0),
                                            point.get('lat', 0), point.get('lon', 0)
                                        )
                                        cumulative_distance_nm += segment_distance
                                        cumulative_time_min += (segment_distance / speed_kts) * 60
                                    
                                    timed_positions.append({
                                        "lat": point.get('lat'),
                                        "lon": point.get('lon'),
                                        "alt_ft": point.get('alt', aircraft.alt_ft),
                                        "time_offset_min": round(cumulative_time_min, 2),
                                        "distance_nm": round(cumulative_distance_nm, 2),
                                    })
                                
                                # Add predicted path info to aircraft
                                aircraft_dict["predicted_path"] = {
                                    "path": timed_positions,
                                    "corridor_id": best_corridor.get("id", "unknown"),
                                    "origin": origin_code,
                                    "destination": dest_code,
                                    "total_distance_nm": round(cumulative_distance_nm, 2),
                                    "eta_minutes": round(cumulative_time_min, 2),
                                    "distance_to_corridor_nm": round(min_distance_to_corridor, 2),
                                }
            except Exception as path_error:
                logger.warning(f"Failed to calculate predicted path for {aircraft.flight_id}: {path_error}")
            
            traffic_with_paths.append(aircraft_dict)
        
        return {
            "traffic": traffic_with_paths,
            "cache_info": cache_info,
            "message": f"Refreshed {len(traffic)} aircraft",
        }
    except Exception as e:
        logger.error(f"Error refreshing traffic: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/route/traffic/simulated")
def add_simulated_aircraft(request: SimulatedAircraftRequest):
    """
    Add a simulated aircraft to the traffic for conflict analysis.
    """
    try:
        planner = _get_route_planner()
        
        aircraft = planner.traffic_manager.add_simulated_aircraft(
            flight_id=request.flight_id,
            path=request.path,
            speed_kts=request.speed_kts,
            altitude_ft=request.altitude_ft,
            callsign=request.callsign,
        )
        
        return {
            "aircraft": aircraft.to_dict(),
            "message": f"Added simulated aircraft {aircraft.flight_id}",
        }
    except Exception as e:
        logger.error(f"Error adding simulated aircraft: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/route/traffic/simulated")
def clear_simulated_aircraft():
    """
    Clear all simulated aircraft from the traffic.
    """
    try:
        planner = _get_route_planner()
        planner.traffic_manager.clear_simulated_traffic()
        
        return {
            "message": "Cleared all simulated aircraft",
        }
    except Exception as e:
        logger.error(f"Error clearing simulated aircraft: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/route/conflicts")
def check_route_conflicts(request: ConflictCheckRequest):
    """
    Check a planned path for conflicts with current traffic.
    """
    try:
        from service.route_planner import PlannedPathPoint
        
        planner = _get_route_planner()
        
        # Convert path to PlannedPathPoint objects
        planned_path = []
        for i, p in enumerate(request.path):
            planned_path.append(PlannedPathPoint(
                lat=p["lat"],
                lon=p["lon"],
                alt_ft=p.get("alt", 30000),
                time_offset_min=p.get("time_offset_min", i * 2),
                cumulative_distance_nm=p.get("distance_nm", 0),
            ))
        
        # Detect conflicts
        conflicts = planner.conflict_detector.detect_conflicts(planned_path, time_window_minutes=60)
        
        # Summarize
        critical_count = len([c for c in conflicts if c.severity.value == "critical"])
        conflict_count = len([c for c in conflicts if c.severity.value == "conflict"])
        warning_count = len([c for c in conflicts if c.severity.value == "warning"])
        
        return {
            "conflicts": [c.to_dict() for c in conflicts],
            "summary": {
                "total": len(conflicts),
                "critical": critical_count,
                "conflict": conflict_count,
                "warning": warning_count,
            },
            "is_clear": len(conflicts) == 0,
            "traffic_count": len(planner.traffic_manager.get_traffic()),
        }
    except Exception as e:
        logger.error(f"Error checking conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/route/plan-strike")
def plan_strike_route(request: StrikePlanRequest):
    """
    Plan tactical strike routes for attack missions.
    
    Features:
    - Direct routes to targets (not via airports)
    - Risk assessment (borders, population centers, sensitive areas)
    - Tactical zone compliance (no-fly, low-altitude)
    - Multi-aircraft coordination with target assignment
    """
    try:
        if request.aircraft_type == "fighter":
            profile = _FIGHTER_JET_PROFILE
        else:
            profile = _CIVIL_AIRCRAFT_PROFILE
        
        zones = None
        if request.tactical_zones:
            zones = [
                {
                    "id": zone.id,
                    "type": zone.type,
                    "polygon": zone.points,
                    "altitude": zone.altitude,
                    "speed": zone.speed,
                }
                for zone in request.tactical_zones
            ]
        
        generator = _get_strike_generator(tactical_zones=zones)
        
        origin_dict = {
            "lat": request.origin.lat,
            "lon": request.origin.lon,
            "name": request.origin.name or "Base",
            "airport_code": request.origin.airport_code,
        }
        
        targets_list = [
            {
                "id": t.id,
                "lat": t.lat,
                "lon": t.lon,
                "priority": t.priority,
                "name": t.name or f"Target {i+1}",
                "ammoRequired": t.ammoRequired,
            }
            for i, t in enumerate(request.targets)
        ]
        
        aircraft_list = [
            {
                "id": a.id,
                "callsign": a.callsign,
                "ammoCapacity": a.ammoCapacity,
                "color": a.color,
                "type": a.type,
            }
            for a in request.aircraft
        ]
        
        result = generator.plan_multi_aircraft_strike(
            origin=origin_dict,
            targets=targets_list,
            aircraft_list=aircraft_list,
            aircraft_profile=profile,
        )
        
        # Calculate mission summary
        total_distance = 0
        total_duration = 0
        total_risk = 0
        assigned_count = 0
        
        for aircraft in result:
            if aircraft.get("route"):
                total_distance += aircraft["route"].get("total_distance_nm", 0)
                total_duration = max(total_duration, aircraft["route"].get("total_duration_min", 0))
                total_risk += aircraft["route"].get("total_risk_score", 0)
                assigned_count += 1
        
        avg_risk = total_risk / assigned_count if assigned_count > 0 else 0
        
        return {
            "aircraft": result,
            "summary": {
                "total_aircraft": len(result),
                "aircraft_with_routes": assigned_count,
                "total_targets": len(targets_list),
                "targets_assigned": sum(len(a.get("assignedTargets", [])) for a in result),
                "total_distance_nm": round(total_distance, 1),
                "max_duration_min": round(total_duration, 1),
                "avg_risk_score": round(avg_risk, 3),
            },
            "origin": origin_dict,
            "targets": targets_list,
        }
    except Exception as e:
        logger.error(f"Error in strike route planning: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/route/traffic/predict/{flight_id}")
def predict_aircraft_position(flight_id: str, minutes_ahead: float = 30):
    """
    Predict future position of a specific aircraft.
    """
    try:
        from service.route_planner import PositionPredictor
        
        planner = _get_route_planner()
        traffic = planner.traffic_manager.get_traffic()
        
        aircraft = None
        for t in traffic:
            if t.flight_id == flight_id:
                aircraft = t
                break
        
        if not aircraft:
            raise HTTPException(status_code=404, detail=f"Aircraft {flight_id} not found in traffic")
        
        predictions = PositionPredictor.predict_trajectory(
            aircraft,
            duration_minutes=minutes_ahead,
            interval_minutes=5
        )
        
        return {
            "flight_id": flight_id,
            "callsign": aircraft.callsign,
            "current_position": {
                "lat": aircraft.lat,
                "lon": aircraft.lon,
                "alt_ft": aircraft.alt_ft,
                "heading_deg": aircraft.heading_deg,
                "speed_kts": aircraft.speed_kts,
            },
            "predictions": [p.to_dict() for p in predictions],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/route/traffic/predict-all")
def predict_all_traffic_paths():
    """
    Predict full paths for all traffic aircraft using learned corridors.
    
    For each aircraft:
    1. Get origin/destination from metadata
    2. Find best matching learned path (O/D pair + closest to current position)
    3. Calculate time-based positions along path
    4. Return full predicted path with timestamps
    """
    try:
        planner = _get_route_planner()
        traffic = planner.traffic_manager.get_traffic()
        
        predicted_paths = []
        
        for aircraft in traffic:
            # Get origin/destination from aircraft (extracted from FR24 live data)
            origin_code = aircraft.origin if aircraft.origin else None
            dest_code = aircraft.destination if aircraft.destination else None
            
            logger.info(f"Aircraft {aircraft.flight_id} ({aircraft.callsign}): Origin={origin_code}, Dest={dest_code}")
            
            # Fallback: If no O/D info, generate straight-line projection
            if not origin_code or not dest_code:
                logger.info(f"Using straight-line fallback for {aircraft.flight_id} (missing O/D)")
                straight_line_path = _generate_straight_line_path(aircraft, duration_minutes=30)
                predicted_paths.append(straight_line_path)
                continue
            
            # Find matching corridors for this O/D pair using CorridorManager
            corridors = planner.corridor_manager.find_direct_corridors(
                origin_code, dest_code
            )
            logger.info(f"Found {len(corridors)} corridors for {origin_code} -> {dest_code}")
            
            if not corridors:
                # Try nearby corridors if no direct match
                from service.route_planner import Waypoint
                current_pos = Waypoint(
                    lat=aircraft.lat,
                    lon=aircraft.lon,
                    alt_ft=aircraft.alt_ft
                )
                nearby = planner.corridor_manager.find_nearby_corridors(
                    current_pos, max_distance_nm=30, limit=3
                )
                if nearby:
                    # Get the corridor from nearby results
                    corridors = [nearby[0][0]]
            
            if not corridors:
                # No corridors found - use straight-line fallback
                straight_line_path = _generate_straight_line_path(aircraft, duration_minutes=30)
                predicted_paths.append(straight_line_path)
                continue
            
            # Find the best matching corridor:
            # 1. Get top 3 closest corridors within 3nm cap
            # 2. Choose the one with most flight members (highest member_count)
            
            MAX_DISTANCE_CAP_NM = 3.0  # 3 nautical mile cap
            
            # Calculate distance for each corridor
            corridor_distances = []
            for corridor in corridors:
                centerline = corridor.get("centerline", [])
                if len(centerline) < 3:
                    continue
                
                # Find closest point on this corridor
                min_dist = float('inf')
                closest_idx = 0
                for i, point in enumerate(centerline):
                    dist = haversine_distance_nm(
                        aircraft.lat, aircraft.lon,
                        point.get('lat', 0), point.get('lon', 0)
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                # Only consider corridors within 3nm cap
                if min_dist <= MAX_DISTANCE_CAP_NM:
                    corridor_distances.append({
                        'corridor': corridor,
                        'distance': min_dist,
                        'closest_idx': closest_idx,
                        'member_count': corridor.get('member_count', 0)
                    })
            
            # Sort by distance (ascending) and take top 3 closest
            corridor_distances.sort(key=lambda x: x['distance'])
            top_3_closest = corridor_distances[:3]
            
            # Among top 3 closest, choose the one with highest member_count
            best_corridor = None
            min_distance_to_corridor = float('inf')
            best_closest_idx = 0
            
            if top_3_closest:
                # Sort by member_count (descending) to get the most popular one
                best_candidate = max(top_3_closest, key=lambda x: x['member_count'])
                best_corridor = best_candidate['corridor']
                min_distance_to_corridor = best_candidate['distance']
                best_closest_idx = best_candidate['closest_idx']
                
                logger.debug(f"Selected corridor for {aircraft.flight_id}: distance={min_distance_to_corridor:.1f}nm, "
                           f"member_count={best_candidate['member_count']} "
                           f"(from {len(top_3_closest)} candidates within 3nm)")
            
            if not best_corridor:
                continue
            
            centerline = best_corridor.get("centerline", [])
            
            # Get remaining path from current position
            remaining_path = centerline[best_closest_idx:]
            
            if len(remaining_path) < 2:
                continue
            
            # Calculate time-based positions along the path
            # Use aircraft's current speed to estimate timing
            speed_kts = aircraft.speed_kts if aircraft.speed_kts > 100 else 450  # Default to 450 if unrealistic
            
            timed_positions = []
            cumulative_distance_nm = 0
            cumulative_time_min = 0
            
            for i, point in enumerate(remaining_path):
                if i > 0:
                    # Calculate distance from previous point
                    prev_point = remaining_path[i - 1]
                    segment_distance = haversine_distance_nm(
                        prev_point.get('lat', 0), prev_point.get('lon', 0),
                        point.get('lat', 0), point.get('lon', 0)
                    )
                    cumulative_distance_nm += segment_distance
                    # Time = distance / speed (convert to minutes)
                    cumulative_time_min += (segment_distance / speed_kts) * 60
                
                timed_positions.append({
                    "lat": point.get('lat'),
                    "lon": point.get('lon'),
                    "alt_ft": point.get('alt', aircraft.alt_ft),
                    "time_offset_min": round(cumulative_time_min, 2),
                    "distance_nm": round(cumulative_distance_nm, 2),
                })
            
            # Calculate ETA
            total_distance_nm = cumulative_distance_nm
            eta_minutes = cumulative_time_min
            
            predicted_paths.append({
                "flight_id": aircraft.flight_id,
                "callsign": aircraft.callsign,
                "origin": origin_code,
                "destination": dest_code,
                "current_position": {
                    "lat": aircraft.lat,
                    "lon": aircraft.lon,
                    "alt_ft": aircraft.alt_ft,
                    "speed_kts": aircraft.speed_kts,
                    "heading_deg": aircraft.heading_deg,
                },
                "corridor_id": best_corridor.get("id", "unknown"),
                "corridor_name": f"{origin_code} → {dest_code}",
                "distance_to_corridor_nm": round(min_distance_to_corridor, 2),
                "predicted_path": timed_positions,
                "total_points": len(timed_positions),
                "total_distance_nm": round(total_distance_nm, 2),
                "eta_minutes": round(eta_minutes, 2),
                "speed_kts": speed_kts,
            })
        
        return {
            "predicted_paths": predicted_paths,
            "total_aircraft": len(traffic),
            "aircraft_with_paths": len(predicted_paths),
            "message": f"Predicted paths for {len(predicted_paths)} out of {len(traffic)} aircraft",
        }
    except Exception as e:
        logger.error(f"Error predicting all traffic paths: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Route Check API - Flight Path Conflict Analysis
# ============================================================

class RouteCheckWaypoint(BaseModel):
    lat: float
    lon: float
    alt: float
    id: str


class RouteCheckRequest(BaseModel):
    waypoints: List[RouteCheckWaypoint]
    datetime: str  # ISO format: "2025-12-25T14:00:00"
    check_alternatives: Optional[bool] = True


@router.get("/api/route-check/airports")
def get_route_check_airports():
    """Get list of airports with coordinates for route planning."""
    return CONFIGURED_AIRPORTS


@router.post("/api/route-check/analyze")
async def analyze_route_conflicts(request: RouteCheckRequest):
    """
    Analyze a planned route for potential conflicts with scheduled airport traffic.
    
    This endpoint:
    1. Identifies airports near the planned route
    2. Fetches scheduled arrivals/departures from each airport
    3. Estimates flight paths for scheduled traffic
    4. Checks for conflicts (within 20 NM and 8000 ft)
    5. Suggests alternative times with fewer/no conflicts
    """
    try:
        # Parse the requested datetime
        try:
            requested_dt = datetime.fromisoformat(request.datetime.replace('Z', '+00:00'))
        except ValueError:
            requested_dt = datetime.strptime(request.datetime, "%Y-%m-%dT%H:%M:%S")
        
        date_str = requested_dt.strftime("%Y-%m-%d")
        
        # Validate date is within 7 days
        now = datetime.now()
        max_date = now + timedelta(days=7)
        if requested_dt.date() > max_date.date():
            raise HTTPException(status_code=400, detail="Date must be within 7 days from today")
        
        # Step 1: Find airports near the route (within 50 NM of any waypoint)
        AIRPORT_SEARCH_RADIUS_NM = 50
        nearby_airports = []
        
        for airport in CONFIGURED_AIRPORTS:
            for wp in request.waypoints:
                dist = haversine_distance_nm(wp.lat, wp.lon, airport["lat"], airport["lon"])
                if dist <= AIRPORT_SEARCH_RADIUS_NM:
                    if airport not in nearby_airports:
                        nearby_airports.append(airport)
                    break
        
        logger.info(f"Found {len(nearby_airports)} airports near route: {[a['code'] for a in nearby_airports]}")
        
        # Step 2: Fetch scheduled flights for each airport
        all_scheduled_flights = []
        
        for airport in nearby_airports:
            iata_code = airport["code"][:3] if len(airport["code"]) == 4 else airport["code"]
            
            # Fetch arrivals
            try:
                arrivals_response = http_requests.get(
                    AVIATION_EDGE_BASE_URL,
                    params={
                        "key": AVIATION_EDGE_API_KEY,
                        "iataCode": iata_code,
                        "type": "arrival",
                        "date": date_str
                    },
                    headers={"accept": "application/json"},
                    timeout=10
                )
                if arrivals_response.status_code == 200:
                    arrivals_data = arrivals_response.json()
                    if isinstance(arrivals_data, list):
                        for flight in arrivals_data:
                            all_scheduled_flights.append({
                                "flight_number": flight.get("flight", {}).get("iataNumber", "Unknown"),
                                "airline": flight.get("airline", {}).get("name", "Unknown"),
                                "departure_airport": flight.get("departure", {}).get("iataCode", "???"),
                                "arrival_airport": iata_code,
                                "scheduled_arrival": flight.get("arrival", {}).get("scheduledTime", ""),
                                "scheduled_departure": flight.get("departure", {}).get("scheduledTime", ""),
                                "type": "arrival",
                                "airport_data": airport,
                            })
            except Exception as e:
                logger.warning(f"Failed to fetch arrivals for {iata_code}: {e}")
            
            # Fetch departures
            try:
                departures_response = http_requests.get(
                    AVIATION_EDGE_BASE_URL,
                    params={
                        "key": AVIATION_EDGE_API_KEY,
                        "iataCode": iata_code,
                        "type": "departure",
                        "date": date_str
                    },
                    headers={"accept": "application/json"},
                    timeout=10
                )
                if departures_response.status_code == 200:
                    departures_data = departures_response.json()
                    if isinstance(departures_data, list):
                        for flight in departures_data:
                            all_scheduled_flights.append({
                                "flight_number": flight.get("flight", {}).get("iataNumber", "Unknown"),
                                "airline": flight.get("airline", {}).get("name", "Unknown"),
                                "departure_airport": iata_code,
                                "arrival_airport": flight.get("arrival", {}).get("iataCode", "???"),
                                "scheduled_arrival": flight.get("arrival", {}).get("scheduledTime", ""),
                                "scheduled_departure": flight.get("departure", {}).get("scheduledTime", ""),
                                "type": "departure",
                                "airport_data": airport,
                            })
            except Exception as e:
                logger.warning(f"Failed to fetch departures for {iata_code}: {e}")
        
        logger.info(f"Total scheduled flights near route: {len(all_scheduled_flights)}")
        
        # Step 3: Analyze time slots for conflicts
        CONFLICT_HORIZONTAL_NM = 20
        CONFLICT_VERTICAL_FT = 8000
        TIME_WINDOW_MINUTES = 30
        
        def analyze_time_slot(check_time: datetime) -> Dict[str, Any]:
            """Analyze conflicts for a specific time slot."""
            conflicts = []
            
            for flight in all_scheduled_flights:
                # Get scheduled time
                sched_time_str = flight.get("scheduled_arrival") or flight.get("scheduled_departure")
                if not sched_time_str:
                    continue
                
                try:
                    sched_dt = datetime.fromisoformat(sched_time_str.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        sched_dt = datetime.strptime(sched_time_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        continue
                
                # Check if within time window
                time_diff = abs((sched_dt - check_time).total_seconds() / 60)
                if time_diff > TIME_WINDOW_MINUTES:
                    continue
                
                # Check proximity to waypoints
                airport = flight["airport_data"]
                for i, wp in enumerate(request.waypoints):
                    dist_to_airport = haversine_distance_nm(wp.lat, wp.lon, airport["lat"], airport["lon"])
                    alt_diff = abs(wp.alt - 5000)  # Assume traffic at 5000ft near airport
                    
                    if dist_to_airport <= CONFLICT_HORIZONTAL_NM and alt_diff <= CONFLICT_VERTICAL_FT:
                        severity = "high" if dist_to_airport < 10 else "medium" if dist_to_airport < 15 else "low"
                        
                        conflicts.append({
                            "flight": {
                                "flight_number": flight["flight_number"],
                                "airline": flight["airline"],
                                "type": flight["type"],
                                "airport": airport["code"],
                                "scheduled_time": sched_time_str,
                            },
                            "waypoint": {
                                "id": wp.id,
                                "lat": wp.lat,
                                "lon": wp.lon,
                                "alt": wp.alt,
                                "waypoint_index": i,
                                "distance_nm": round(dist_to_airport, 2),
                                "altitude_diff_ft": round(alt_diff, 0),
                                "estimated_time": sched_dt.strftime("%H:%M"),
                            },
                            "severity": severity,
                        })
                        break  # Only one conflict per flight
            
            return {
                "time": check_time.strftime("%H:%M"),
                "conflict_count": len(conflicts),
                "conflicts": conflicts,
            }
        
        # Analyze original time slot
        original_analysis = analyze_time_slot(requested_dt)
        
        # Step 4: Check alternative times throughout the day
        alternative_slots = []
        if request.check_alternatives:
            # Check every 30 minutes from 06:00 to 22:00
            base_date = requested_dt.replace(hour=6, minute=0, second=0, microsecond=0)
            for hour_offset in range(0, 34):  # 6:00 to 22:00 in 30-min increments
                alt_time = base_date + timedelta(minutes=hour_offset * 30)
                if alt_time.hour >= 22:
                    break
                slot_analysis = analyze_time_slot(alt_time)
                alternative_slots.append(slot_analysis)
        
        # Step 5: Find best suggestion
        suggestion = None
        if alternative_slots:
            # Sort by conflict count
            sorted_slots = sorted(alternative_slots, key=lambda x: x["conflict_count"])
            best_slot = sorted_slots[0]
            
            if best_slot["conflict_count"] == 0:
                suggestion = {
                    "time": best_slot["time"],
                    "conflict_count": 0,
                    "message": f"No conflicts detected at {best_slot['time']}. This is the optimal time for your route."
                }
            elif best_slot["conflict_count"] < original_analysis["conflict_count"]:
                suggestion = {
                    "time": best_slot["time"],
                    "conflict_count": best_slot["conflict_count"],
                    "message": f"Fewer conflicts at {best_slot['time']} ({best_slot['conflict_count']} vs {original_analysis['conflict_count']} at your selected time)."
                }
            else:
                suggestion = {
                    "time": original_analysis["time"],
                    "conflict_count": original_analysis["conflict_count"],
                    "message": "Your selected time has the same or fewer conflicts than alternatives."
                }
        
        return {
            "original_analysis": original_analysis,
            "airports_checked": [{"code": a["code"], "name": a["name"], "lat": a["lat"], "lon": a["lon"]} for a in nearby_airports],
            "flights_analyzed": len(all_scheduled_flights),
            "suggestion": suggestion,
            "alternative_slots": sorted(alternative_slots, key=lambda x: x["conflict_count"])[:10] if alternative_slots else None,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing route conflicts: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRAFFIC SNAPSHOT ENDPOINTS
# ============================================================================

# Traffic snapshots DB instance (will be initialized in configure())
_traffic_snapshots_db = None


@router.get("/api/route/traffic/snapshots")
def list_traffic_snapshots():
    """
    Get list of saved traffic snapshots (last 3).
    Returns metadata only, not full data.
    """
    try:
        if _traffic_snapshots_db is None:
            raise HTTPException(status_code=500, detail="Traffic snapshots database not initialized")
        
        snapshots = _traffic_snapshots_db.get_snapshots()
        return {
            "snapshots": snapshots,
            "count": len(snapshots),
        }
    except Exception as e:
        logger.error(f"Error listing traffic snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/route/traffic/snapshots/{snapshot_id}")
def get_traffic_snapshot(snapshot_id: int):
    """
    Get a specific traffic snapshot with full data.
    """
    try:
        if _traffic_snapshots_db is None:
            raise HTTPException(status_code=500, detail="Traffic snapshots database not initialized")
        
        snapshot = _traffic_snapshots_db.get_snapshot(snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting traffic snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TrafficSnapshotRequest(BaseModel):
    name: str
    traffic: List[Dict[str, Any]]
    timestamp: Optional[int] = None


@router.post("/api/route/traffic/snapshots")
def save_traffic_snapshot(request: TrafficSnapshotRequest):
    """
    Save a traffic snapshot.
    Automatically maintains only the last 3 snapshots.
    """
    try:
        if _traffic_snapshots_db is None:
            raise HTTPException(status_code=500, detail="Traffic snapshots database not initialized")
        
        # Use current timestamp if not provided
        timestamp = request.timestamp or int(datetime.utcnow().timestamp())
        
        # Save snapshot
        snapshot_id = _traffic_snapshots_db.save_snapshot(
            name=request.name,
            timestamp=timestamp,
            traffic_data=request.traffic
        )
        
        return {
            "snapshot_id": snapshot_id,
            "name": request.name,
            "aircraft_count": len(request.traffic),
            "message": f"Saved traffic snapshot with {len(request.traffic)} aircraft",
        }
    except Exception as e:
        logger.error(f"Error saving traffic snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/route/traffic/snapshots/{snapshot_id}")
def delete_traffic_snapshot(snapshot_id: int):
    """
    Delete a specific traffic snapshot.
    """
    try:
        if _traffic_snapshots_db is None:
            raise HTTPException(status_code=500, detail="Traffic snapshots database not initialized")
        
        deleted = _traffic_snapshots_db.delete_snapshot(snapshot_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        
        return {"message": "Snapshot deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting traffic snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PredictBatchRequest(BaseModel):
    aircraft: List[Dict[str, Any]]


@router.post("/api/route/traffic/predict-batch")
def predict_traffic_paths_batch(request: PredictBatchRequest):
    """
    Predict flight paths for a batch of aircraft based on their current position,
    heading, speed, and learned corridor patterns.
    
    Uses the unified prediction function to ensure consistency with route planning proximity checks.
    
    Returns predicted paths for each aircraft that matches a known corridor.
    """
    try:
        planner = _get_route_planner()
        if planner is None:
            raise HTTPException(status_code=500, detail="Route planner not initialized")
        
        predicted_paths = []
        
        for aircraft_data in request.aircraft:
            try:
                flight_id = aircraft_data.get('flight_id', 'UNKNOWN')
                callsign = aircraft_data.get('callsign')
                lat = aircraft_data.get('lat')
                lon = aircraft_data.get('lon')
                alt_ft = aircraft_data.get('alt_ft', 0)
                heading_deg = aircraft_data.get('heading_deg', 0)
                speed_kts = aircraft_data.get('speed_kts', 250)
                vspeed_fpm = aircraft_data.get('vspeed_fpm', 0)
                
                if lat is None or lon is None:
                    continue
                
                # Try to get origin/destination from aircraft data
                origin_code = aircraft_data.get('origin') or aircraft_data.get('origin_airport')
                dest_code = aircraft_data.get('destination') or aircraft_data.get('destination_airport')
                
                # Create a mock aircraft object
                class MockAircraft:
                    pass
                mock_aircraft = MockAircraft()
                mock_aircraft.flight_id = flight_id
                mock_aircraft.callsign = callsign
                mock_aircraft.lat = lat
                mock_aircraft.lon = lon
                mock_aircraft.alt_ft = alt_ft
                mock_aircraft.heading_deg = heading_deg
                mock_aircraft.speed_kts = speed_kts
                mock_aircraft.vspeed_fpm = vspeed_fpm
                
                # Use the unified prediction function (same as route planning uses)
                prediction_result = predict_aircraft_path_unified(
                    aircraft=mock_aircraft,
                    planner=planner,
                    duration_minutes=30,
                    origin_code=origin_code,
                    dest_code=dest_code
                )
                
                predicted_path = prediction_result.get('predicted_path', [])
                
                if not predicted_path:
                    continue
                
                # Format result for API response
                actual_speed = speed_kts if speed_kts > 100 else 450
                total_distance = predicted_path[-1].get('distance_nm', 0) if predicted_path else 0
                total_time = predicted_path[-1].get('time_offset_min', 0) if predicted_path else 0
                
                result = {
                    'flight_id': flight_id,
                    'callsign': callsign,
                    'origin': origin_code,
                    'destination': dest_code,
                    'current_position': {
                        'lat': lat,
                        'lon': lon,
                        'alt_ft': alt_ft,
                        'speed_kts': actual_speed,
                        'heading_deg': heading_deg
                    },
                    'corridor_id': f"{origin_code}_{dest_code}" if origin_code and dest_code else "straight_line",
                    'corridor_name': f"{origin_code} → {dest_code}" if origin_code and dest_code else "Straight Line",
                    'distance_to_corridor_nm': 0,  # Calculated in unified function
                    'predicted_path': predicted_path,
                    'total_points': len(predicted_path),
                    'total_distance_nm': round(total_distance, 2),
                    'eta_minutes': round(total_time, 2),
                    'speed_kts': actual_speed
                }
                
                predicted_paths.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting path for aircraft {aircraft_data.get('flight_id')}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info(f"Generated {len(predicted_paths)} predicted paths for {len(request.aircraft)} aircraft")
        
        return {
            'predicted_paths': predicted_paths,
            'total_aircraft': len(request.aircraft),
            'aircraft_with_paths': len(predicted_paths),
            'message': f"Predicted paths for {len(predicted_paths)} out of {len(request.aircraft)} aircraft"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_traffic_paths_batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

