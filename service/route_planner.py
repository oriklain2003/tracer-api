"""
Advanced Tactical Route Planner - Calculates optimal flight paths with conflict detection.

Features:
- Aircraft profiles (Fighter Jet / Civil Aircraft)
- Smart corridor-aware path generation
- Real-time traffic integration with FR24
- Position prediction up to 1 hour ahead
- Conflict detection with separation rules
- Custom waypoint support (click on map)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Project root for resolving paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default paths
DEFAULT_PATHS_FILE = PROJECT_ROOT / "rules" / "learned_paths.json"
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "rules" / "rule_config.json"

# Bounding box for traffic (same as training)
BBOX_NORTH = 34.597042
BBOX_SOUTH = 28.536275
BBOX_WEST = 32.299805
BBOX_EAST = 37.397461

# Scoring weights - prioritize safety (conflicts) and corridor usage
CONFLICT_WEIGHT = 0.35       # Highest priority: avoid other aircraft
CORRIDOR_BONUS_WEIGHT = 0.20  # Bonus for using learned corridors (safer paths)
DISTANCE_WEIGHT = 0.15       # Prefer shorter routes
SAFETY_WEIGHT = 0.10         # Corridor width (narrower = more precise)
COVERAGE_WEIGHT = 0.05       # Number of waypoints
ZONE_PENALTY_WEIGHT = 0.15   # Penalty for violating tactical zones

# Reference values for normalization
MAX_DISTANCE_NM = 500.0
MAX_WIDTH_NM = 15.0
MAX_WAYPOINTS = 120

# Separation minimums
HORIZONTAL_SEPARATION_NM = 5.0  # Standard IFR separation
VERTICAL_SEPARATION_FT = 1000.0  # Standard vertical separation
WARNING_HORIZONTAL_NM = 10.0  # Warning threshold
WARNING_VERTICAL_FT = 2000.0  # Warning threshold

# FR24 API Token - loaded from environment
FR24_API_TOKEN = os.environ.get("FR24_API_TOKEN")
if not FR24_API_TOKEN:
    logger.warning("FR24_API_TOKEN not set in environment variables. Traffic fetching will be disabled.")


# ============================================================
# Tactical Zone Helpers
# ============================================================

def point_in_polygon(lat: float, lon: float, polygon: List[Dict[str, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    Polygon is a list of {lat, lon} points.
    """
    n = len(polygon)
    if n < 3:
        return False
    
    inside = False
    j = n - 1
    
    for i in range(n):
        yi, xi = polygon[i].get('lat', 0), polygon[i].get('lon', 0)
        yj, xj = polygon[j].get('lat', 0), polygon[j].get('lon', 0)
        
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def check_route_zone_violations(
    path: List[Dict[str, float]], 
    zones: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Check how a route interacts with tactical zones.
    
    Returns:
        {
            'no_fly_violations': int,  # Points inside no-fly zones
            'altitude_compliance': float,  # 0-1, how well altitude matches zone requirements
            'speed_compliance': float,  # 0-1, how well speed matches zone requirements
            'total_penalty': float  # 0-1, overall zone penalty (0 = good, 1 = bad)
        }
    """
    if not zones or not path:
        return {
            'no_fly_violations': 0,
            'altitude_compliance': 1.0,
            'speed_compliance': 1.0,
            'total_penalty': 0.0
        }
    
    no_fly_count = 0
    altitude_issues = 0
    speed_issues = 0
    total_points = len(path)
    
    for point in path:
        lat, lon = point.get('lat', 0), point.get('lon', 0)
        alt = point.get('alt', 35000)
        
        for zone in zones:
            zone_type = zone.get('type', '')
            zone_points = zone.get('points', [])
            
            if not zone_points or len(zone_points) < 3:
                continue
            
            if point_in_polygon(lat, lon, zone_points):
                if zone_type == 'no-fly':
                    no_fly_count += 1
                elif zone_type == 'low-altitude':
                    zone_alt = zone.get('altitude', 5000)
                    if alt > zone_alt:
                        altitude_issues += 1
                elif zone_type == 'high-altitude':
                    zone_alt = zone.get('altitude', 30000)
                    if alt < zone_alt:
                        altitude_issues += 1
                # Speed zones would need speed info per point - skip for now
    
    # Calculate compliance scores
    altitude_compliance = 1.0 - (altitude_issues / max(total_points, 1))
    speed_compliance = 1.0  # Default to good since we don't track speed per point
    
    # No-fly violations are severe
    no_fly_penalty = min(no_fly_count / max(total_points, 1) * 5, 1.0)  # 5x penalty
    
    total_penalty = (no_fly_penalty * 0.6 + (1 - altitude_compliance) * 0.3 + (1 - speed_compliance) * 0.1)
    
    return {
        'no_fly_violations': no_fly_count,
        'altitude_compliance': altitude_compliance,
        'speed_compliance': speed_compliance,
        'total_penalty': min(total_penalty, 1.0)
    }


# ============================================================
# Aircraft Profiles
# ============================================================

class AircraftType(str, Enum):
    FIGHTER = "fighter"
    CIVIL = "civil"


@dataclass
class AircraftProfile:
    """Aircraft performance profile for route planning."""
    name: str
    type: AircraftType
    min_speed_kts: float
    max_speed_kts: float
    cruise_speed_kts: float
    min_altitude_ft: float
    max_altitude_ft: float
    cruise_altitude_ft: float
    climb_rate_ft_min: float
    descent_rate_ft_min: float
    turn_rate_deg_sec: float
    radar_cross_section: float = 5.0
    sensor_range_nm: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "min_speed_kts": self.min_speed_kts,
            "max_speed_kts": self.max_speed_kts,
            "cruise_speed_kts": self.cruise_speed_kts,
            "min_altitude_ft": self.min_altitude_ft,
            "max_altitude_ft": self.max_altitude_ft,
            "cruise_altitude_ft": self.cruise_altitude_ft,
            "climb_rate_ft_min": self.climb_rate_ft_min,
            "descent_rate_ft_min": self.descent_rate_ft_min,
            "turn_rate_deg_sec": self.turn_rate_deg_sec,
            "radar_cross_section": self.radar_cross_section,
            "sensor_range_nm": self.sensor_range_nm,
        }


# Predefined aircraft profiles
F16_PROFILE = AircraftProfile(
    name="F-16 Viper",
    type=AircraftType.FIGHTER,
    min_speed_kts=150,
    max_speed_kts=1350,
    cruise_speed_kts=500,
    min_altitude_ft=500,
    max_altitude_ft=50000,
    cruise_altitude_ft=30000,
    climb_rate_ft_min=10000,
    descent_rate_ft_min=8000,
    turn_rate_deg_sec=25,
    radar_cross_section=5.0,
    sensor_range_nm=60.0,
)

F35_PROFILE = AircraftProfile(
    name="F-35 Lightning II",
    type=AircraftType.FIGHTER,
    min_speed_kts=140,
    max_speed_kts=1050,
    cruise_speed_kts=500,
    min_altitude_ft=500,
    max_altitude_ft=50000,
    cruise_altitude_ft=35000,
    climb_rate_ft_min=9000,
    descent_rate_ft_min=7000,
    turn_rate_deg_sec=20,
    radar_cross_section=0.005,
    sensor_range_nm=100.0,
)

# Backward compatibility
FIGHTER_JET_PROFILE = F16_PROFILE

CIVIL_AIRCRAFT_PROFILE = AircraftProfile(
    name="Civil Aircraft (A320)",
    type=AircraftType.CIVIL,
    min_speed_kts=130,
    max_speed_kts=480,
    cruise_speed_kts=450,
    min_altitude_ft=1000,
    max_altitude_ft=41000,
    cruise_altitude_ft=35000,
    climb_rate_ft_min=2500,
    descent_rate_ft_min=2000,
    turn_rate_deg_sec=3,
    radar_cross_section=50.0,
    sensor_range_nm=0.0,
)

AIRCRAFT_PROFILES = {
    AircraftType.FIGHTER: F16_PROFILE,  # Default fighter
    AircraftType.CIVIL: CIVIL_AIRCRAFT_PROFILE,
    "F-16": F16_PROFILE,
    "F-35": F35_PROFILE,
}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Waypoint:
    """A waypoint on a route (can be airport or custom point)."""
    lat: float
    lon: float
    alt_ft: Optional[float] = None
    name: Optional[str] = None
    is_airport: bool = False
    airport_code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt_ft": self.alt_ft,
            "name": self.name or f"{self.lat:.4f}, {self.lon:.4f}",
            "is_airport": self.is_airport,
            "airport_code": self.airport_code,
        }


@dataclass
class TrafficAircraft:
    """An aircraft in the airspace (real or simulated)."""
    flight_id: str
    callsign: Optional[str]
    lat: float
    lon: float
    alt_ft: float
    heading_deg: float
    speed_kts: float
    vspeed_fpm: float = 0.0
    timestamp: int = 0
    is_simulated: bool = False
    track_points: List[Dict[str, Any]] = field(default_factory=list)
    origin: Optional[str] = None  # Origin airport ICAO code
    destination: Optional[str] = None  # Destination airport ICAO code
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "flight_id": self.flight_id,
            "callsign": self.callsign,
            "lat": self.lat,
            "lon": self.lon,
            "alt_ft": self.alt_ft,
            "heading_deg": self.heading_deg,
            "speed_kts": self.speed_kts,
            "vspeed_fpm": self.vspeed_fpm,
            "timestamp": self.timestamp,
            "is_simulated": self.is_simulated,
            "track_points": self.track_points,
            "origin": self.origin,
            "destination": self.destination,
        }


@dataclass
class PredictedPosition:
    """A predicted future position of an aircraft."""
    flight_id: str
    lat: float
    lon: float
    alt_ft: float
    time_offset_min: float
    timestamp: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "flight_id": self.flight_id,
            "lat": self.lat,
            "lon": self.lon,
            "alt_ft": self.alt_ft,
            "time_offset_min": self.time_offset_min,
            "timestamp": self.timestamp,
        }


class ConflictSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    CONFLICT = "conflict"
    CRITICAL = "critical"


@dataclass
class Conflict:
    """A potential conflict between planned route and traffic."""
    severity: ConflictSeverity
    planned_lat: float
    planned_lon: float
    planned_alt_ft: float
    planned_time_offset_min: float
    traffic_flight_id: str
    traffic_callsign: Optional[str]
    traffic_lat: float
    traffic_lon: float
    traffic_alt_ft: float
    horizontal_distance_nm: float
    vertical_distance_ft: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "planned_lat": self.planned_lat,
            "planned_lon": self.planned_lon,
            "planned_alt_ft": self.planned_alt_ft,
            "planned_time_offset_min": self.planned_time_offset_min,
            "traffic_flight_id": self.traffic_flight_id,
            "traffic_callsign": self.traffic_callsign,
            "traffic_lat": self.traffic_lat,
            "traffic_lon": self.traffic_lon,
            "traffic_alt_ft": self.traffic_alt_ft,
            "horizontal_distance_nm": round(self.horizontal_distance_nm, 2),
            "vertical_distance_ft": round(self.vertical_distance_ft, 0),
        }


@dataclass
class PlannedPathPoint:
    """A point on the planned path with time estimates."""
    lat: float
    lon: float
    alt_ft: float
    time_offset_min: float  # Minutes from departure
    cumulative_distance_nm: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt_ft": self.alt_ft,
            "time_offset_min": round(self.time_offset_min, 1),
            "cumulative_distance_nm": round(self.cumulative_distance_nm, 1),
        }


@dataclass
class Airport:
    """Airport definition."""
    code: str
    name: str
    lat: float
    lon: float
    elevation_ft: Optional[float] = None


@dataclass
class RouteCandidate:
    """A candidate route from origin to destination."""
    path_id: str
    origin: Optional[str]
    destination: Optional[str]
    centerline: List[Dict[str, float]]
    width_nm: float
    distance_nm: float = 0.0
    score: float = 0.0
    distance_score: float = 0.0
    safety_score: float = 0.0
    coverage_score: float = 0.0
    conflict_score: float = 1.0  # 1.0 = no conflicts, 0.0 = many conflicts
    proximity_score: float = 1.0  # 1.0 = no dangerous proximities, 0.0 = many proximities
    proximity_count: int = 0  # Number of dangerous proximity encounters
    recommendation: str = ""
    conflicts: List[Conflict] = field(default_factory=list)
    planned_path: List[PlannedPathPoint] = field(default_factory=list)
    eta_minutes: float = 0.0
    corridor_ids: List[str] = field(default_factory=list)  # IDs of learned corridors used to build this route
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "origin": self.origin,
            "destination": self.destination,
            "centerline": self.centerline,
            "width_nm": round(self.width_nm, 2),
            "distance_nm": round(self.distance_nm, 2),
            "score": round(self.score, 3),
            "distance_score": round(self.distance_score, 3),
            "safety_score": round(self.safety_score, 3),
            "coverage_score": round(self.coverage_score, 3),
            "conflict_score": round(self.conflict_score, 3),
            "proximity_score": round(self.proximity_score, 3),
            "proximity_count": self.proximity_count,
            "recommendation": self.recommendation,
            "waypoint_count": len(self.centerline),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "conflict_count": len([c for c in self.conflicts if c.severity == ConflictSeverity.CONFLICT]),
            "warning_count": len([c for c in self.conflicts if c.severity == ConflictSeverity.WARNING]),
            "planned_path": [p.to_dict() for p in self.planned_path],
            "eta_minutes": round(self.eta_minutes, 1),
            "corridor_ids": self.corridor_ids,  # IDs of learned corridors used
        }


# ============================================================
# Geodesy Utilities
# ============================================================

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    EARTH_RADIUS_KM = 6371.0
    NM_PER_KM = 0.539957
    
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c * NM_PER_KM


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from point 1 to point 2."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    
    y = math.sin(dlon_r) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)
    
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def destination_point(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> Tuple[float, float]:
    """Calculate destination point given start, bearing, and distance."""
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


def calculate_path_distance(centerline: List[Dict[str, float]]) -> float:
    """Calculate total distance along a path centerline in NM."""
    if len(centerline) < 2:
        return 0.0
    
    total = 0.0
    for i in range(1, len(centerline)):
        p1 = centerline[i - 1]
        p2 = centerline[i]
        total += haversine_nm(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
    
    return total


def point_to_segment_distance(
    point_lat: float, point_lon: float,
    seg_start_lat: float, seg_start_lon: float,
    seg_end_lat: float, seg_end_lon: float
) -> float:
    """Calculate minimum distance from a point to a line segment in NM."""
    # Use projection method
    seg_len = haversine_nm(seg_start_lat, seg_start_lon, seg_end_lat, seg_end_lon)
    if seg_len < 0.01:  # Very short segment
        return haversine_nm(point_lat, point_lon, seg_start_lat, seg_start_lon)
    
    # Project point onto segment line
    # Simplified: check distances to endpoints and midpoint
    d_start = haversine_nm(point_lat, point_lon, seg_start_lat, seg_start_lon)
    d_end = haversine_nm(point_lat, point_lon, seg_end_lat, seg_end_lon)
    
    # Check midpoint too
    mid_lat = (seg_start_lat + seg_end_lat) / 2
    mid_lon = (seg_start_lon + seg_end_lon) / 2
    d_mid = haversine_nm(point_lat, point_lon, mid_lat, mid_lon)
    
    return min(d_start, d_end, d_mid)


# ============================================================
# Traffic Manager
# ============================================================

class TrafficManager:
    """Manages real-time traffic data with caching."""
    
    def __init__(self):
        self._traffic_cache: List[TrafficAircraft] = []
        self._simulated_traffic: List[TrafficAircraft] = []
        self._cache_timestamp: int = 0
        self._client = None
    
    def _get_client(self):
        """Lazy load FR24 client."""
        if self._client is None:
            try:
                from fr24sdk.client import Client
                self._client = Client(api_token=FR24_API_TOKEN)
            except ImportError:
                logger.warning("FR24 SDK not available")
                self._client = None
        return self._client
    
    def fetch_live_traffic(
        self, 
        force_refresh: bool = False,
        custom_bounds: Optional[Dict[str, float]] = None
    ) -> List[TrafficAircraft]:
        """
        Fetch live traffic from FR24 API.
        
        Args:
            force_refresh: If True, bypass cache and fetch new data
            custom_bounds: Optional custom boundary dict with keys: north, south, west, east
                          If not provided, uses default BBOX from config
        """
        # Return cache if not forcing refresh (only if no custom bounds)
        if not force_refresh and not custom_bounds and self._traffic_cache and self._cache_timestamp > 0:
            return self._traffic_cache + self._simulated_traffic
        
        client = self._get_client()
        if not client:
            logger.warning("No FR24 client available, returning cached/simulated traffic only")
            return self._traffic_cache + self._simulated_traffic
        
        try:
            from fr24sdk.models.geographic import Boundary
            
            # Use custom bounds if provided, otherwise use defaults
            if custom_bounds:
                boundary = Boundary(
                    north=custom_bounds['north'],
                    south=custom_bounds['south'],
                    west=custom_bounds['west'],
                    east=custom_bounds['east']
                )
                logger.info(f"Fetching live traffic from FR24 with custom bounds: "
                           f"lat=[{custom_bounds['south']:.2f}, {custom_bounds['north']:.2f}], "
                           f"lon=[{custom_bounds['west']:.2f}, {custom_bounds['east']:.2f}]")
            else:
                boundary = Boundary(
                    north=BBOX_NORTH,
                    south=BBOX_SOUTH,
                    west=BBOX_WEST,
                    east=BBOX_EAST
                )
                logger.info("Fetching live traffic from FR24 with default bounds...")
            
            response = client.live.flight_positions.get_full(
                bounds=boundary,
                altitude_ranges=["1000-50000"]
            )
            
            live_data = response.model_dump()["data"]
            self._traffic_cache = []
            
            for item in live_data:
                flight_id = item.get("fr24_id", "")
                if not flight_id:
                    continue
                
                # Parse timestamp
                ts = item.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
                    except:
                        ts = int(time.time())
                elif ts is None:
                    ts = int(time.time())
                
                # Extract origin and destination airports from FR24 data
                # Prefer ICAO codes, fallback to IATA (same as research_replay.py)
                origin_code = item.get("orig_icao") or item.get("orig_iata")
                dest_code = item.get("dest_icao") or item.get("dest_iata")
                
                aircraft = TrafficAircraft(
                    flight_id=flight_id,
                    callsign=item.get("callsign"),
                    lat=item.get("lat", 0),
                    lon=item.get("lon", 0),
                    alt_ft=item.get("alt", 0),
                    heading_deg=item.get("track", 0),
                    speed_kts=item.get("gspeed", 0),
                    vspeed_fpm=item.get("vspeed", 0) or 0,
                    timestamp=ts,
                    is_simulated=False,
                    origin=origin_code,
                    destination=dest_code,
                )
                self._traffic_cache.append(aircraft)
            
            self._cache_timestamp = int(time.time())
            logger.info(f"Fetched {len(self._traffic_cache)} aircraft from FR24")
            
            # NOTE: Track fetching removed - not needed for route planning
            # We only need current positions from get_full()
            # Tracks can be fetched on-demand if needed for specific aircraft
            
        except Exception as e:
            logger.error(f"Error fetching live traffic: {e}")
        
        return self._traffic_cache + self._simulated_traffic
    
    def _fetch_tracks_for_traffic(self, client, max_tracks: int = 20):
        """Fetch historical tracks for traffic aircraft."""
        count = 0
        for aircraft in self._traffic_cache:
            if count >= max_tracks:
                break
            
            try:
                time.sleep(0.5)  # Rate limiting
                tracks_resp = client.flight_tracks.get(flight_id=[aircraft.flight_id])
                if tracks_resp:
                    flight_data = tracks_resp.model_dump()["data"]
                    if flight_data:
                        track_points = flight_data[0].get("tracks", [])
                        aircraft.track_points = track_points[-50:]  # Last 50 points
                        count += 1
            except Exception as e:
                logger.debug(f"Could not fetch track for {aircraft.flight_id}: {e}")
    
    def add_simulated_aircraft(
        self,
        flight_id: str,
        path: List[Dict[str, Any]],
        speed_kts: float,
        altitude_ft: float,
        callsign: Optional[str] = None
    ) -> TrafficAircraft:
        """Add a simulated aircraft to the traffic."""
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Use first point as current position
        first_point = path[0]
        
        # Calculate heading from first two points
        heading = 0.0
        if len(path) >= 2:
            heading = initial_bearing_deg(
                path[0]["lat"], path[0]["lon"],
                path[1]["lat"], path[1]["lon"]
            )
        
        aircraft = TrafficAircraft(
            flight_id=f"SIM_{flight_id}",
            callsign=callsign or f"SIM{flight_id[:4].upper()}",
            lat=first_point["lat"],
            lon=first_point["lon"],
            alt_ft=altitude_ft,
            heading_deg=heading,
            speed_kts=speed_kts,
            vspeed_fpm=0,
            timestamp=int(time.time()),
            is_simulated=True,
            track_points=path,
        )
        
        self._simulated_traffic.append(aircraft)
        return aircraft
    
    def clear_simulated_traffic(self):
        """Clear all simulated traffic."""
        self._simulated_traffic = []
    
    def get_traffic(self) -> List[TrafficAircraft]:
        """Get all traffic (cached real + simulated)."""
        return self._traffic_cache + self._simulated_traffic
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache status information."""
        return {
            "real_aircraft_count": len(self._traffic_cache),
            "simulated_aircraft_count": len(self._simulated_traffic),
            "total_count": len(self._traffic_cache) + len(self._simulated_traffic),
            "cache_timestamp": self._cache_timestamp,
            "cache_age_seconds": int(time.time()) - self._cache_timestamp if self._cache_timestamp else None,
        }


# ============================================================
# Corridor Manager - Learned Paths
# ============================================================

class CorridorManager:
    """Manages learned flight corridors from learned_paths.json"""
    
    def __init__(self, paths_file: Path, min_member_flights: int = 5):
        self._paths_file = paths_file
        self._paths_cache: List[Dict[str, Any]] = []
        self._min_member_flights = min_member_flights
        self._load_paths()
    
    def _load_paths(self):
        """Load paths from learned_paths.json and filter by minimum member flights"""
        try:
            if not self._paths_file.exists():
                logger.warning(f"Paths file not found: {self._paths_file}")
                self._paths_cache = []
                return
            
            with open(self._paths_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract paths array from JSON
            all_paths = data.get("paths", [])
            
            # Filter paths by minimum member flights
            self._paths_cache = [
                path for path in all_paths 
                if path.get("member_count", 0) >= self._min_member_flights
            ]
            
            logger.info(f"Loaded {len(self._paths_cache)} learned paths from {self._paths_file} "
                       f"(filtered from {len(all_paths)} total, min_members={self._min_member_flights})")
        
        except Exception as e:
            logger.error(f"Error loading paths from {self._paths_file}: {e}")
            self._paths_cache = []
    
    def _normalize_airport_code(self, code: Optional[str]) -> Optional[str]:
        """Normalize airport code (None/UNK/empty -> None)"""
        if not code or code == "UNK":
            return None
        return code.upper()
    
    def find_direct_corridors(self, origin: str, dest: str) -> List[Dict[str, Any]]:
        """
        Find corridors matching origin/destination pair.
        
        Logic mirrors rule_logic.py _get_paths_for_od():
        - Exact match: origin -> dest (highest priority)
        - Partial match: origin -> None OR None -> dest (medium priority)
        - Generic match: None -> None (lowest priority)
        
        Args:
            origin: Origin airport code
            dest: Destination airport code
            
        Returns:
            List of matching corridor dicts, prioritized by match quality
        """
        # Normalize codes
        origin_norm = self._normalize_airport_code(origin)
        dest_norm = self._normalize_airport_code(dest)
        
        # If no O/D info at all, return all paths
        if origin_norm is None and dest_norm is None:
            return self._paths_cache
        
        # Find matching paths with priority levels
        matches = []
        
        for path in self._paths_cache:
            path_origin = self._normalize_airport_code(path.get("origin"))
            path_dest = self._normalize_airport_code(path.get("destination"))
            
            # Check match type
            if origin_norm and dest_norm:
                # Flight has both O/D
                # Priority 0: Exact match (X -> Y)
                if path_origin == origin_norm and path_dest == dest_norm:
                    matches.append((path, 0))
                # Priority 1: Partial match (X -> None or None -> Y)
                elif path_origin == origin_norm and path_dest is None:
                    matches.append((path, 1))
                elif path_origin is None and path_dest == dest_norm:
                    matches.append((path, 1))
                # Priority 2: Generic match (None -> None)
                elif path_origin is None and path_dest is None:
                    matches.append((path, 2))
            
            elif origin_norm and not dest_norm:
                # Flight has only origin - match same origin
                if path_origin == origin_norm:
                    priority = 0 if path_dest else 1
                    matches.append((path, priority))
            
            elif dest_norm and not origin_norm:
                # Flight has only destination - match same dest
                if path_dest == dest_norm:
                    priority = 0 if path_origin else 1
                    matches.append((path, priority))
        
        if not matches:
            # No matches found - fallback to all paths
            return self._paths_cache
        
        # Sort by priority (0 = best)
        matches.sort(key=lambda x: x[1])
        
        # Get the best priority level
        best_priority = matches[0][1]
        
        # Return only corridors with the best priority
        result = [m[0] for m in matches if m[1] == best_priority]
        
        logger.debug(f"Found {len(result)} corridors for {origin} -> {dest} (priority {best_priority})")
        return result
    
    def find_nearby_corridors(
        self, 
        pos: Waypoint, 
        max_distance_nm: float = 30.0, 
        limit: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find corridors near a position, sorted by distance.
        
        Args:
            pos: Current position (Waypoint with lat, lon)
            max_distance_nm: Maximum distance to search
            limit: Maximum number of results to return
            
        Returns:
            List of (corridor_dict, distance_nm) tuples, sorted by distance
        """
        nearby = []
        
        for path in self._paths_cache:
            centerline = path.get("centerline", [])
            if len(centerline) < 2:
                continue
            
            # Find minimum distance from position to any point on centerline
            min_dist = float('inf')
            closest_idx = 0
            
            for i, point in enumerate(centerline):
                dist = haversine_nm(
                    pos.lat, pos.lon,
                    point.get('lat', 0), point.get('lon', 0)
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Only include if within max distance
            if min_dist <= max_distance_nm:
                nearby.append((path, min_dist))
        
        # Sort by distance and limit results
        nearby.sort(key=lambda x: x[1])
        return nearby[:limit]
    
    def get_all_paths(self) -> List[Dict[str, Any]]:
        """Get all loaded paths."""
        return self._paths_cache
    
    def reload(self):
        """Reload paths from file."""
        self._load_paths()


# ============================================================
# Position Predictor
# ============================================================

class PositionPredictor:
    """Predicts future positions of aircraft."""
    
    @staticmethod
    def predict_position(
        aircraft: TrafficAircraft,
        minutes_ahead: float
    ) -> PredictedPosition:
        """Predict aircraft position at a future time."""
        # Simple linear extrapolation based on current heading and speed
        hours = minutes_ahead / 60.0
        distance_nm = aircraft.speed_kts * hours
        
        # Calculate new position
        new_lat, new_lon = destination_point(
            aircraft.lat, aircraft.lon,
            aircraft.heading_deg, distance_nm
        )
        
        # Estimate altitude change
        alt_change = aircraft.vspeed_fpm * minutes_ahead
        new_alt = aircraft.alt_ft + alt_change
        
        return PredictedPosition(
            flight_id=aircraft.flight_id,
            lat=new_lat,
            lon=new_lon,
            alt_ft=new_alt,
            time_offset_min=minutes_ahead,
            timestamp=aircraft.timestamp + int(minutes_ahead * 60),
        )
    
    @staticmethod
    def predict_trajectory(
        aircraft: TrafficAircraft,
        duration_minutes: float = 60,
        interval_minutes: float = 5
    ) -> List[PredictedPosition]:
        """Predict trajectory over a time period."""
        predictions = []
        t = 0.0
        while t <= duration_minutes:
            pred = PositionPredictor.predict_position(aircraft, t)
            predictions.append(pred)
            t += interval_minutes
        return predictions


# ============================================================
# Conflict Detector
# ============================================================

class ConflictDetector:
    """Detects conflicts between planned route and traffic."""
    
    def __init__(self, traffic_manager: TrafficManager):
        self.traffic_manager = traffic_manager
    
    def detect_conflicts(
        self,
        planned_path: List[PlannedPathPoint],
        time_window_minutes: float = 60
    ) -> List[Conflict]:
        """Detect conflicts along a planned path."""
        conflicts = []
        traffic = self.traffic_manager.get_traffic()
        
        if not traffic or not planned_path:
            return conflicts
        
        for path_point in planned_path:
            if path_point.time_offset_min > time_window_minutes:
                continue
            
            for aircraft in traffic:
                # Predict where this aircraft will be at path_point time
                predicted = PositionPredictor.predict_position(
                    aircraft, path_point.time_offset_min
                )
                
                # Calculate separation
                h_dist = haversine_nm(
                    path_point.lat, path_point.lon,
                    predicted.lat, predicted.lon
                )
                v_dist = abs(path_point.alt_ft - predicted.alt_ft)
                
                # Determine severity
                severity = ConflictSeverity.NONE
                
                if h_dist < HORIZONTAL_SEPARATION_NM and v_dist < VERTICAL_SEPARATION_FT:
                    severity = ConflictSeverity.CONFLICT
                    if h_dist < 2.5 and v_dist < 500:
                        severity = ConflictSeverity.CRITICAL
                elif h_dist < WARNING_HORIZONTAL_NM and v_dist < WARNING_VERTICAL_FT:
                    severity = ConflictSeverity.WARNING
                
                if severity != ConflictSeverity.NONE:
                    conflict = Conflict(
                        severity=severity,
                        planned_lat=path_point.lat,
                        planned_lon=path_point.lon,
                        planned_alt_ft=path_point.alt_ft,
                        planned_time_offset_min=path_point.time_offset_min,
                        traffic_flight_id=aircraft.flight_id,
                        traffic_callsign=aircraft.callsign,
                        traffic_lat=predicted.lat,
                        traffic_lon=predicted.lon,
                        traffic_alt_ft=predicted.alt_ft,
                        horizontal_distance_nm=h_dist,
                        vertical_distance_ft=v_dist,
                    )
                    conflicts.append(conflict)
        
        # Sort by severity (critical first) then by time
        severity_order = {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.CONFLICT: 1,
            ConflictSeverity.WARNING: 2,
        }
        conflicts.sort(key=lambda c: (severity_order.get(c.severity, 3), c.planned_time_offset_min))
        
        return conflicts


# ============================================================
# Smart Path Generator
# ============================================================

class SmartPathGenerator:
    """
    Generates intelligent paths that leverage learned flight corridors.
    
    Key capabilities:
    - Find and chain multiple corridor segments for O/D pairs without direct paths
    - Generate diverse route options (direct, corridor-based, via intermediate airports)
    - Score corridors by relevance to the desired route
    - Create smooth transitions between corridor segments
    """
    
    def __init__(self, paths_cache: List[Dict[str, Any]], airports_cache: List[Airport]):
        self.paths = paths_cache
        self.airports = airports_cache
        self._build_corridor_index()
    
    def _build_corridor_index(self):
        """Build indexes for fast corridor lookup."""
        # Index by origin airport
        self.corridors_by_origin: Dict[str, List[Dict[str, Any]]] = {}
        # Index by destination airport
        self.corridors_by_dest: Dict[str, List[Dict[str, Any]]] = {}
        # Index by O/D pair
        self.corridors_by_od: Dict[str, List[Dict[str, Any]]] = {}
        
        for path in self.paths:
            origin = path.get("origin")
            dest = path.get("destination")
            
            if origin:
                if origin not in self.corridors_by_origin:
                    self.corridors_by_origin[origin] = []
                self.corridors_by_origin[origin].append(path)
            
            if dest:
                if dest not in self.corridors_by_dest:
                    self.corridors_by_dest[dest] = []
                self.corridors_by_dest[dest].append(path)
            
            if origin and dest:
                key = f"{origin}->{dest}"
                if key not in self.corridors_by_od:
                    self.corridors_by_od[key] = []
                self.corridors_by_od[key].append(path)
        
        logger.info(f"Built corridor index: {len(self.corridors_by_origin)} origins, "
                   f"{len(self.corridors_by_dest)} destinations, {len(self.corridors_by_od)} O/D pairs")
    
    def find_direct_corridors(
        self,
        origin_code: Optional[str],
        dest_code: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Find corridors that directly connect origin to destination."""
        if not origin_code or not dest_code:
            return []
        
        key = f"{origin_code.upper()}->{dest_code.upper()}"
        return self.corridors_by_od.get(key, [])
    
    def find_corridors_from_origin(self, origin_code: str) -> List[Dict[str, Any]]:
        """Find all corridors departing from an origin."""
        return self.corridors_by_origin.get(origin_code.upper(), [])
    
    def find_corridors_to_dest(self, dest_code: str) -> List[Dict[str, Any]]:
        """Find all corridors arriving at a destination."""
        return self.corridors_by_dest.get(dest_code.upper(), [])
    
    def find_chained_corridors(
        self,
        origin_code: str,
        dest_code: str,
        max_hops: int = 2
    ) -> List[List[Dict[str, Any]]]:
        """
        Find corridor chains that connect origin to destination via intermediate airports.
        Returns list of corridor chains (each chain is a list of corridors).
        """
        chains = []
        origin = origin_code.upper()
        dest = dest_code.upper()
        
        # BFS to find chains
        from collections import deque
        
        # Queue items: (current_airport, chain_so_far, visited_airports)
        queue = deque([(origin, [], {origin})])
        
        while queue:
            current, chain, visited = queue.popleft()
            
            if len(chain) > max_hops:
                continue
            
            # Get corridors from current airport
            for corridor in self.corridors_by_origin.get(current, []):
                corridor_dest = corridor.get("destination")
                if not corridor_dest:
                    continue
                
                new_chain = chain + [corridor]
                
                # Found destination!
                if corridor_dest.upper() == dest:
                    chains.append(new_chain)
                    continue
                
                # Continue searching if not visited and not too many hops
                if corridor_dest not in visited and len(new_chain) < max_hops:
                    new_visited = visited | {corridor_dest}
                    queue.append((corridor_dest, new_chain, new_visited))
        
        # Sort by chain length (prefer shorter chains)
        chains.sort(key=len)
        return chains[:5]  # Return top 5 chains
    
    def find_nearest_corridor(
        self,
        lat: float,
        lon: float,
        max_distance_nm: float = 20.0,
        heading_towards: Optional[Tuple[float, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the nearest learned corridor to a point.
        Optionally filter by corridors heading in the right direction.
        """
        best_path = None
        best_score = float("inf")
        
        for path in self.paths:
            centerline = path.get("centerline", [])
            if len(centerline) < 2:
                continue
            
            # Find closest point on this corridor
            min_dist = float("inf")
            closest_idx = 0
            
            for i, p in enumerate(centerline):
                dist = haversine_nm(lat, lon, p["lat"], p["lon"])
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            if min_dist > max_distance_nm:
                continue
            
            # Score based on distance
            score = min_dist
            
            # Bonus for corridors heading in the right direction
            if heading_towards and closest_idx < len(centerline) - 1:
                corridor_bearing = initial_bearing_deg(
                    centerline[closest_idx]["lat"], centerline[closest_idx]["lon"],
                    centerline[closest_idx + 1]["lat"], centerline[closest_idx + 1]["lon"]
                )
                target_bearing = initial_bearing_deg(lat, lon, heading_towards[0], heading_towards[1])
                
                # Calculate bearing difference (0-180)
                bearing_diff = abs(corridor_bearing - target_bearing)
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff
                
                # Penalize corridors going wrong direction
                if bearing_diff < 45:
                    score *= 0.5  # Good direction bonus
                elif bearing_diff > 135:
                    score *= 3.0  # Wrong direction penalty
            
            if score < best_score:
                best_score = score
                best_path = path
        
        return best_path
    
    def find_corridors_near_line(
        self,
        start_lat: float, start_lon: float,
        end_lat: float, end_lon: float,
        max_distance_nm: float = 30.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find corridors that run roughly parallel to a line between two points.
        Returns list of (corridor, relevance_score) tuples.
        """
        results = []
        
        # Calculate the bearing of the desired route
        route_bearing = initial_bearing_deg(start_lat, start_lon, end_lat, end_lon)
        route_distance = haversine_nm(start_lat, start_lon, end_lat, end_lon)
        
        for path in self.paths:
            centerline = path.get("centerline", [])
            if len(centerline) < 5:
                continue
            
            # Calculate corridor bearing (from first to last point)
            corridor_bearing = initial_bearing_deg(
                centerline[0]["lat"], centerline[0]["lon"],
                centerline[-1]["lat"], centerline[-1]["lon"]
            )
            
            # Check bearing alignment
            bearing_diff = abs(route_bearing - corridor_bearing)
            if bearing_diff > 180:
                bearing_diff = 360 - bearing_diff
            
            # Skip corridors going perpendicular or opposite
            if bearing_diff > 60 and bearing_diff < 120:
                continue
            if bearing_diff > 120:
                # Might be usable in reverse
                bearing_diff = 180 - bearing_diff if bearing_diff < 180 else bearing_diff - 180
            
            # Check if corridor is near the route line
            min_dist_to_route = float("inf")
            coverage_count = 0
            
            # Sample points along the route
            for i in range(10):
                frac = i / 9.0
                sample_lat = start_lat + (end_lat - start_lat) * frac
                sample_lon = start_lon + (end_lon - start_lon) * frac
                
                # Find closest corridor point to this sample
                for p in centerline:
                    dist = haversine_nm(sample_lat, sample_lon, p["lat"], p["lon"])
                    min_dist_to_route = min(min_dist_to_route, dist)
                    if dist < max_distance_nm:
                        coverage_count += 1
                        break
            
            if min_dist_to_route > max_distance_nm:
                continue
            
            # Score: lower is better
            # Consider: distance, bearing alignment, coverage
            score = (
                min_dist_to_route * 0.5 +
                bearing_diff * 0.3 +
                (10 - coverage_count) * 2.0
            )
            
            results.append((path, score))
        
        # Sort by score (lower is better)
        results.sort(key=lambda x: x[1])
        return results[:10]
    
    def generate_great_circle_path(
        self,
        origin: Waypoint,
        destination: Waypoint,
        num_points: int = 50,
        cruise_alt: Optional[float] = None
    ) -> List[Dict[str, float]]:
        """Generate a great circle path between two points with altitude profile."""
        path = []
        
        origin_alt = origin.alt_ft or 0
        dest_alt = destination.alt_ft or 0
        cruise_altitude = cruise_alt or max(origin_alt, dest_alt, 30000)
        
        for i in range(num_points):
            fraction = i / (num_points - 1)
            
            # Linear interpolation (simplified great circle)
            lat = origin.lat + (destination.lat - origin.lat) * fraction
            lon = origin.lon + (destination.lon - origin.lon) * fraction
            
            # Altitude profile: climb, cruise, descend
            if fraction < 0.15:  # Climb
                alt_fraction = fraction / 0.15
                alt = origin_alt + (cruise_altitude - origin_alt) * alt_fraction
            elif fraction > 0.85:  # Descend
                alt_fraction = (1.0 - fraction) / 0.15
                alt = dest_alt + (cruise_altitude - dest_alt) * alt_fraction
            else:  # Cruise
                alt = cruise_altitude
            
            path.append({"lat": lat, "lon": lon, "alt": alt})
        
        return path
    
    def generate_corridor_chain_path(
        self,
        origin: Waypoint,
        destination: Waypoint,
        corridor_chain: List[Dict[str, Any]],
        profile: AircraftProfile
    ) -> List[Dict[str, float]]:
        """Generate a path following a chain of corridors."""
        if not corridor_chain:
            return self.generate_great_circle_path(origin, destination)
        
        full_path = []
        current_pos = origin
        
        for i, corridor in enumerate(corridor_chain):
            centerline = corridor.get("centerline", [])
            if not centerline:
                continue
            
            # Find entry point on this corridor (closest to current position)
            entry_idx = 0
            entry_dist = float("inf")
            for j, p in enumerate(centerline):
                dist = haversine_nm(current_pos.lat, current_pos.lon, p["lat"], p["lon"])
                if dist < entry_dist:
                    entry_dist = dist
                    entry_idx = j
            
            # Connect current position to corridor entry
            if entry_dist > 1.0:  # More than 1nm away
                entry_point = centerline[entry_idx]
                connector = self.generate_great_circle_path(
                    current_pos,
                    Waypoint(lat=entry_point["lat"], lon=entry_point["lon"], alt_ft=profile.cruise_altitude_ft),
                    num_points=max(5, int(entry_dist / 5))
                )
                # Skip first point if we already have path
                if full_path:
                    connector = connector[1:]
                full_path.extend(connector)
            
            # Determine which direction to traverse corridor
            # (towards the corridor's destination or towards our final destination)
            if i < len(corridor_chain) - 1:
                # Not last corridor - go to its natural end
                exit_idx = len(centerline) - 1
            else:
                # Last corridor - find exit closest to final destination
                exit_idx = len(centerline) - 1
                exit_dist = float("inf")
                for j, p in enumerate(centerline):
                    dist = haversine_nm(destination.lat, destination.lon, p["lat"], p["lon"])
                    if dist < exit_dist:
                        exit_dist = dist
                        exit_idx = j
            
            # Extract corridor segment
            if entry_idx <= exit_idx:
                segment = centerline[entry_idx:exit_idx + 1]
            else:
                segment = list(reversed(centerline[exit_idx:entry_idx + 1]))
            
            # Add segment (skip first if overlapping)
            # NOTE: We use cruise altitude instead of corridor altitude to ensure
            # realistic flight profiles - corridors may contain altitude data from
            # flights that were descending for approach to nearby airports
            for j, p in enumerate(segment):
                if j == 0 and full_path:
                    # Check for overlap
                    last = full_path[-1]
                    if haversine_nm(last["lat"], last["lon"], p["lat"], p["lon"]) < 0.5:
                        continue
                full_path.append({
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "alt": profile.cruise_altitude_ft  # Use cruise altitude, not corridor altitude
                })
            
            # Update current position
            if segment:
                last_p = segment[-1]
                current_pos = Waypoint(
                    lat=last_p["lat"],
                    lon=last_p["lon"],
                    alt_ft=profile.cruise_altitude_ft  # Use cruise altitude
                )
        
        # Connect last corridor exit to destination
        if full_path:
            last = full_path[-1]
            final_dist = haversine_nm(last["lat"], last["lon"], destination.lat, destination.lon)
            if final_dist > 1.0:
                final_connector = self.generate_great_circle_path(
                    Waypoint(lat=last["lat"], lon=last["lon"], alt_ft=last.get("alt", profile.cruise_altitude_ft)),
                    destination,
                    num_points=max(5, int(final_dist / 5))
                )
                full_path.extend(final_connector[1:])
        
        return full_path
    
    def generate_corridor_aware_path(
        self,
        origin: Waypoint,
        destination: Waypoint,
        profile: AircraftProfile
    ) -> List[Dict[str, float]]:
        """
        Generate a path that leverages nearby corridors even without exact O/D match.
        """
        # Find corridors near the route line
        nearby_corridors = self.find_corridors_near_line(
            origin.lat, origin.lon,
            destination.lat, destination.lon,
            max_distance_nm=40.0
        )
        
        if not nearby_corridors:
            return self.generate_great_circle_path(origin, destination, cruise_alt=profile.cruise_altitude_ft)
        
        # Use the best matching corridor
        best_corridor = nearby_corridors[0][0]
        centerline = best_corridor.get("centerline", [])
        
        if len(centerline) < 3:
            return self.generate_great_circle_path(origin, destination, cruise_alt=profile.cruise_altitude_ft)
        
        # Find entry and exit points on corridor
        entry_idx, entry_dist = 0, float("inf")
        exit_idx, exit_dist = 0, float("inf")
        
        for i, p in enumerate(centerline):
            d_origin = haversine_nm(origin.lat, origin.lon, p["lat"], p["lon"])
            d_dest = haversine_nm(destination.lat, destination.lon, p["lat"], p["lon"])
            
            if d_origin < entry_dist:
                entry_dist = d_origin
                entry_idx = i
            
            if d_dest < exit_dist:
                exit_dist = d_dest
                exit_idx = i
        
        # Build path: origin -> corridor entry -> corridor segment -> corridor exit -> destination
        full_path = []
        
        # 1. Origin to corridor entry
        if entry_dist > 2.0:
            entry_point = centerline[entry_idx]
            approach = self.generate_great_circle_path(
                origin,
                Waypoint(lat=entry_point["lat"], lon=entry_point["lon"], alt_ft=profile.cruise_altitude_ft),
                num_points=max(10, int(entry_dist / 3)),
                cruise_alt=profile.cruise_altitude_ft
            )
            full_path.extend(approach)
        
        # 2. Corridor segment
        if entry_idx <= exit_idx:
            segment = centerline[entry_idx:exit_idx + 1]
        else:
            segment = list(reversed(centerline[exit_idx:entry_idx + 1]))
        
        # Use cruise altitude instead of corridor altitude for realistic flight profiles
        for p in segment:
            if full_path:
                last = full_path[-1]
                if haversine_nm(last["lat"], last["lon"], p["lat"], p["lon"]) < 0.5:
                    continue
            full_path.append({
                "lat": p["lat"],
                "lon": p["lon"],
                "alt": profile.cruise_altitude_ft  # Use cruise altitude, not corridor altitude
            })
        
        # 3. Corridor exit to destination
        if exit_dist > 2.0 and full_path:
            last = full_path[-1]
            departure = self.generate_great_circle_path(
                Waypoint(lat=last["lat"], lon=last["lon"], alt_ft=profile.cruise_altitude_ft),
                destination,
                num_points=max(10, int(exit_dist / 3)),
                cruise_alt=profile.cruise_altitude_ft
            )
            full_path.extend(departure[1:])
        
        return full_path if full_path else self.generate_great_circle_path(origin, destination, cruise_alt=profile.cruise_altitude_ft)
    
    def generate_all_route_options(
        self,
        origin: Waypoint,
        destination: Waypoint,
        waypoints: List[Waypoint],
        profile: AircraftProfile
    ) -> List[Tuple[str, List[Dict[str, float]], str, List[str]]]:
        """
        Generate multiple diverse route options.
        Returns list of (route_id, path, route_type, corridor_ids) tuples.
        corridor_ids contains the IDs of learned corridors used to build this route.
        """
        routes = []
        
        origin_code = origin.airport_code
        dest_code = destination.airport_code
        
        # 1. Direct great circle route (no corridors used)
        direct_path = self.generate_great_circle_path(origin, destination, cruise_alt=profile.cruise_altitude_ft)
        if direct_path:
            routes.append(("direct", direct_path, "direct", []))
        
        # 2. Exact O/D corridor matches
        if origin_code and dest_code:
            direct_corridors = self.find_direct_corridors(origin_code, dest_code)
            for i, corridor in enumerate(direct_corridors[:3]):
                centerline = corridor.get("centerline", [])
                if centerline:
                    # Use cruise altitude instead of corridor altitude for realistic flight profiles
                    # Corridors may contain altitude data from flights descending for approach
                    path = [{"lat": p["lat"], "lon": p["lon"], "alt": profile.cruise_altitude_ft} for p in centerline]
                    corridor_id = corridor.get('id', f"learned_{i}")
                    routes.append((f"learned_{corridor_id}", path, "learned_corridor", [corridor_id]))
        
        # 3. Chained corridor routes (via intermediate airports)
        if origin_code and dest_code:
            chains = self.find_chained_corridors(origin_code, dest_code, max_hops=2)
            for i, chain in enumerate(chains[:2]):
                chain_path = self.generate_corridor_chain_path(origin, destination, chain, profile)
                if chain_path:
                    via_airports = " → ".join([c.get("destination", "?") for c in chain])
                    # Collect all corridor IDs in the chain
                    chain_corridor_ids = [c.get("id", f"chain_{j}") for j, c in enumerate(chain)]
                    routes.append((f"via_{via_airports}", chain_path, "chained_corridor", chain_corridor_ids))
        
        # 4. Corridor-aware route (uses nearby corridors even without exact match)
        corridor_aware_path, corridor_aware_ids = self._generate_corridor_aware_path_with_ids(origin, destination, profile)
        if corridor_aware_path and len(corridor_aware_path) > 5:
            # Check it's different enough from direct
            direct_dist = calculate_path_distance(direct_path) if direct_path else 0
            aware_dist = calculate_path_distance(corridor_aware_path)
            if abs(aware_dist - direct_dist) > 5:  # At least 5nm different
                routes.append(("corridor_aware", corridor_aware_path, "corridor_aware", corridor_aware_ids))
        
        # 5. If we have custom waypoints, generate path through them
        if waypoints:
            all_points = [origin] + waypoints + [destination]
            waypoint_path = []
            for i in range(len(all_points) - 1):
                segment = self.generate_great_circle_path(
                    all_points[i], all_points[i + 1],
                    num_points=20, cruise_alt=profile.cruise_altitude_ft
                )
                if waypoint_path:
                    segment = segment[1:]
                waypoint_path.extend(segment)
            if waypoint_path:
                routes.append(("custom_waypoints", waypoint_path, "custom", []))
        
        return routes
    
    def _generate_corridor_aware_path_with_ids(
        self,
        origin: Waypoint,
        destination: Waypoint,
        profile: AircraftProfile
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """
        Generate a path that leverages nearby corridors and return the corridor IDs used.
        """
        # Find corridors near the route line
        nearby_corridors = self.find_corridors_near_line(
            origin.lat, origin.lon,
            destination.lat, destination.lon,
            max_distance_nm=40.0
        )
        
        if not nearby_corridors:
            return self.generate_great_circle_path(origin, destination, cruise_alt=profile.cruise_altitude_ft), []
        
        # Use the best matching corridor
        best_corridor = nearby_corridors[0][0]
        corridor_id = best_corridor.get("id", "unknown")
        centerline = best_corridor.get("centerline", [])
        
        if len(centerline) < 3:
            return self.generate_great_circle_path(origin, destination, cruise_alt=profile.cruise_altitude_ft), []
        
        # Find entry and exit points on corridor
        entry_idx, entry_dist = 0, float("inf")
        exit_idx, exit_dist = 0, float("inf")
        
        for i, p in enumerate(centerline):
            d_origin = haversine_nm(origin.lat, origin.lon, p["lat"], p["lon"])
            d_dest = haversine_nm(destination.lat, destination.lon, p["lat"], p["lon"])
            
            if d_origin < entry_dist:
                entry_dist = d_origin
                entry_idx = i
            
            if d_dest < exit_dist:
                exit_dist = d_dest
                exit_idx = i
        
        # Build path: origin -> corridor entry -> corridor segment -> corridor exit -> destination
        full_path = []
        
        # 1. Origin to corridor entry
        if entry_dist > 2.0:
            entry_point = centerline[entry_idx]
            approach = self.generate_great_circle_path(
                origin,
                Waypoint(lat=entry_point["lat"], lon=entry_point["lon"], alt_ft=profile.cruise_altitude_ft),
                num_points=max(10, int(entry_dist / 3)),
                cruise_alt=profile.cruise_altitude_ft
            )
            full_path.extend(approach)
        
        # 2. Corridor segment
        if entry_idx <= exit_idx:
            segment = centerline[entry_idx:exit_idx + 1]
        else:
            segment = list(reversed(centerline[exit_idx:entry_idx + 1]))
        
        # Use cruise altitude instead of corridor altitude for realistic flight profiles
        for p in segment:
            if full_path:
                last = full_path[-1]
                if haversine_nm(last["lat"], last["lon"], p["lat"], p["lon"]) < 0.5:
                    continue
            full_path.append({
                "lat": p["lat"],
                "lon": p["lon"],
                "alt": profile.cruise_altitude_ft  # Use cruise altitude, not corridor altitude
            })
        
        # 3. Corridor exit to destination
        if exit_dist > 2.0 and full_path:
            last = full_path[-1]
            departure = self.generate_great_circle_path(
                Waypoint(lat=last["lat"], lon=last["lon"], alt_ft=profile.cruise_altitude_ft),
                destination,
                num_points=max(10, int(exit_dist / 3)),
                cruise_alt=profile.cruise_altitude_ft
            )
            full_path.extend(departure[1:])
        
        if full_path:
            return full_path, [corridor_id]
        return self.generate_great_circle_path(origin, destination, cruise_alt=profile.cruise_altitude_ft), []
    
    def generate_hybrid_path(
        self,
        origin: Waypoint,
        destination: Waypoint,
        waypoints: List[Waypoint] = None,
        profile: AircraftProfile = None
    ) -> List[Dict[str, float]]:
        """Generate a path that prefers learned corridors when possible."""
        if waypoints is None:
            waypoints = []
        
        if profile is None:
            profile = CIVIL_AIRCRAFT_PROFILE
        
        # Try corridor-aware path first
        return self.generate_corridor_aware_path(origin, destination, profile)
    
    def _find_corridor_entry(
        self,
        corridor: Dict[str, Any],
        point: Waypoint
    ) -> Optional[Dict[str, float]]:
        """Find the best entry point into a corridor."""
        centerline = corridor.get("centerline", [])
        if not centerline:
            return None
        
        best_point = None
        best_dist = float("inf")
        
        for p in centerline:
            dist = haversine_nm(point.lat, point.lon, p["lat"], p["lon"])
            if dist < best_dist:
                best_dist = dist
                best_point = p
        
        return best_point
    
    def _extract_corridor_segment(
        self,
        corridor: Dict[str, Any],
        start: Waypoint,
        end: Waypoint,
        profile: AircraftProfile
    ) -> List[Dict[str, float]]:
        """Extract a segment of a corridor between two points."""
        centerline = corridor.get("centerline", [])
        if len(centerline) < 2:
            return []
        
        # Find closest points on corridor to start and end
        start_idx = 0
        end_idx = len(centerline) - 1
        start_dist = float("inf")
        end_dist = float("inf")
        
        for i, p in enumerate(centerline):
            d_start = haversine_nm(start.lat, start.lon, p["lat"], p["lon"])
            d_end = haversine_nm(end.lat, end.lon, p["lat"], p["lon"])
            
            if d_start < start_dist:
                start_dist = d_start
                start_idx = i
            
            if d_end < end_dist:
                end_dist = d_end
                end_idx = i
        
        # Extract segment (handle direction)
        if start_idx <= end_idx:
            segment = centerline[start_idx:end_idx + 1]
        else:
            segment = list(reversed(centerline[end_idx:start_idx + 1]))
        
        # Add altitude if missing
        for p in segment:
            if "alt" not in p:
                p["alt"] = profile.cruise_altitude_ft
        
        return segment


# ============================================================
# Main Route Planner
# ============================================================

class RoutePlanner:
    """
    Advanced route planner with conflict detection and smart path generation.
    """
    
    def __init__(
        self,
        paths_file: Optional[Path] = None,
        config_file: Optional[Path] = None,
    ):
        self.paths_file = paths_file or DEFAULT_PATHS_FILE
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        
        self._paths_cache: Optional[List[Dict[str, Any]]] = None
        self._airports_cache: Optional[List[Airport]] = None
        
        # Initialize managers
        self.traffic_manager = TrafficManager()
        self.conflict_detector = ConflictDetector(self.traffic_manager)
        self.corridor_manager = CorridorManager(self.paths_file)
        self._path_generator: Optional[SmartPathGenerator] = None
    
    def _load_paths(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """Load paths from the learned paths JSON file."""
        if self._paths_cache is not None and not refresh:
            return self._paths_cache
        
        try:
            if self.paths_file.exists():
                with open(self.paths_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._paths_cache = data.get("paths", [])
                    logger.info(f"Loaded {len(self._paths_cache)} paths from {self.paths_file}")
            else:
                logger.warning(f"Paths file not found: {self.paths_file}")
                self._paths_cache = []
        except Exception as e:
            logger.error(f"Error loading paths: {e}")
            self._paths_cache = []
        
        return self._paths_cache
    
    def _load_airports(self, refresh: bool = False) -> List[Airport]:
        """Load airports from the rule config JSON file."""
        if self._airports_cache is not None and not refresh:
            return self._airports_cache
        
        try:
            if self.config_file.exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    airports_raw = data.get("airports", [])
                    self._airports_cache = [
                        Airport(
                            code=a["code"],
                            name=a["name"],
                            lat=a["lat"],
                            lon=a["lon"],
                            elevation_ft=a.get("elevation_ft"),
                        )
                        for a in airports_raw
                    ]
                    logger.info(f"Loaded {len(self._airports_cache)} airports")
            else:
                logger.warning(f"Config file not found: {self.config_file}")
                self._airports_cache = []
        except Exception as e:
            logger.error(f"Error loading airports: {e}")
            self._airports_cache = []
        
        return self._airports_cache
    
    def _get_path_generator(self) -> SmartPathGenerator:
        """Get or create path generator."""
        if self._path_generator is None:
            self._path_generator = SmartPathGenerator(
                self._load_paths(),
                self._load_airports()
            )
        return self._path_generator
    
    def get_airports(self) -> List[Dict[str, Any]]:
        """Get list of available airports."""
        airports = self._load_airports()
        return [
            {
                "code": a.code,
                "name": a.name,
                "lat": a.lat,
                "lon": a.lon,
                "elevation_ft": a.elevation_ft,
            }
            for a in airports
        ]
    
    def get_airport_by_code(self, code: str) -> Optional[Airport]:
        """Get airport by ICAO code."""
        airports = self._load_airports()
        for a in airports:
            if a.code.upper() == code.upper():
                return a
        return None
    
    def get_aircraft_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get available aircraft profiles."""
        return {
            "fighter": FIGHTER_JET_PROFILE.to_dict(),
            "civil": CIVIL_AIRCRAFT_PROFILE.to_dict(),
        }
    
    def find_routes(
        self,
        origins: List[str],
        destination: str,
    ) -> List[Dict[str, Any]]:
        """Find all paths matching any of the origins to the destination."""
        paths = self._load_paths()
        matching = []
        
        origins_upper = [o.upper() for o in origins]
        dest_upper = destination.upper()
        
        for path in paths:
            path_origin = path.get("origin")
            path_dest = path.get("destination")
            
            origin_match = path_origin and path_origin.upper() in origins_upper
            dest_match = path_dest and path_dest.upper() == dest_upper
            
            if origin_match and dest_match:
                matching.append({**path, "_match_type": "exact"})
            elif origin_match and not path_dest:
                matching.append({**path, "_match_type": "origin_only"})
            elif dest_match and not path_origin:
                matching.append({**path, "_match_type": "dest_only"})
        
        return matching
    
    def create_planned_path(
        self,
        centerline: List[Dict[str, float]],
        profile: AircraftProfile,
        departure_time: Optional[int] = None
    ) -> List[PlannedPathPoint]:
        """Convert centerline to planned path with time estimates."""
        if not centerline:
            return []
        
        if departure_time is None:
            departure_time = int(time.time())
        
        planned = []
        cumulative_dist = 0.0
        cumulative_time = 0.0
        
        for i, point in enumerate(centerline):
            if i > 0:
                prev = centerline[i - 1]
                segment_dist = haversine_nm(prev["lat"], prev["lon"], point["lat"], point["lon"])
                cumulative_dist += segment_dist
                
                # Time = distance / speed
                segment_time_hours = segment_dist / profile.cruise_speed_kts
                cumulative_time += segment_time_hours * 60  # Convert to minutes
            
            alt = point.get("alt", profile.cruise_altitude_ft)
            
            planned.append(PlannedPathPoint(
                lat=point["lat"],
                lon=point["lon"],
                alt_ft=alt,
                time_offset_min=cumulative_time,
                cumulative_distance_nm=cumulative_dist,
            ))
        
        return planned
    
    def score_route(
        self,
        path: Dict[str, Any],
        profile: AircraftProfile,
        check_conflicts: bool = True,
        route_type: str = "direct"
    ) -> RouteCandidate:
        """
        Score a route based on conflicts, corridor usage, distance, and safety.
        
        The best path is one that:
        1. Has the LEAST conflicts with other aircraft (highest priority)
        2. Uses learned corridors (proven safe paths)
        3. Is reasonably short
        4. Has good corridor coverage
        
        Args:
            path: Route data with centerline, width, etc.
            profile: Aircraft performance profile
            check_conflicts: Whether to check for traffic conflicts
            route_type: Type of route ('direct', 'learned_corridor', 'chained_corridor', 'corridor_aware')
        """
        centerline = path.get("centerline", [])
        width_nm = path.get("width_nm", 8.0)
        
        # Calculate distance
        distance_nm = calculate_path_distance(centerline)
        
        # Distance score (shorter is better, but not the main priority)
        distance_score = max(0.0, 1.0 - (distance_nm / MAX_DISTANCE_NM))
        
        # Safety score (narrower corridor = more precise/safer)
        safety_score = max(0.0, 1.0 - (width_nm / MAX_WIDTH_NM))
        
        # Coverage score (more waypoints = more detailed path)
        waypoint_count = len(centerline)
        coverage_score = min(1.0, waypoint_count / MAX_WAYPOINTS)
        
        # Create planned path for conflict detection
        planned_path = self.create_planned_path(centerline, profile)
        
        # ETA
        eta_minutes = planned_path[-1].time_offset_min if planned_path else 0
        
        # ============================================================
        # CONFLICT DETECTION - Highest Priority
        # The best path has the LEAST problems (other planes)
        # ============================================================
        conflicts = []
        conflict_score = 1.0
        
        if check_conflicts and planned_path:
            conflicts = self.conflict_detector.detect_conflicts(planned_path, time_window_minutes=60)
            
            # Count conflicts by severity
            critical_count = len([c for c in conflicts if c.severity == ConflictSeverity.CRITICAL])
            conflict_count = len([c for c in conflicts if c.severity == ConflictSeverity.CONFLICT])
            warning_count = len([c for c in conflicts if c.severity == ConflictSeverity.WARNING])
            
            # Heavily penalize conflicts - critical conflicts are deal-breakers
            # Each critical = -50% score, each conflict = -25%, each warning = -5%
            conflict_penalty = critical_count * 0.5 + conflict_count * 0.25 + warning_count * 0.05
            conflict_score = max(0.0, 1.0 - conflict_penalty)
            
            logger.debug(f"Route {path.get('id')}: {critical_count} critical, "
                        f"{conflict_count} conflicts, {warning_count} warnings -> score {conflict_score:.2f}")
        
        # ============================================================
        # CORRIDOR BONUS - Reward routes that use learned corridors
        # Learned corridors are proven safe paths used by real flights
        # ============================================================
        corridor_bonus = 0.0
        
        if route_type == "learned_corridor":
            # Direct learned corridor - highest bonus (exact match from database)
            corridor_bonus = 1.0
        elif route_type == "chained_corridor":
            # Chained corridors - good bonus (connected learned paths)
            corridor_bonus = 0.85
        elif route_type == "corridor_aware":
            # Corridor-aware - moderate bonus (uses nearby corridors)
            corridor_bonus = 0.7
        elif route_type == "custom":
            # Custom waypoints - small bonus if waypoints are near corridors
            corridor_bonus = 0.4
        else:  # "direct"
            # Direct great-circle - no corridor bonus (not following proven paths)
            corridor_bonus = 0.0
        
        # ============================================================
        # PROXIMITY ANALYSIS - Count dangerous proximities (time-based)
        # ============================================================
        proximity_count = 0
        proximity_score = 1.0
        
        if hasattr(self, 'proximity_thresholds') and self.proximity_thresholds and check_conflicts and planned_path:
            # Get current traffic
            traffic = self.traffic_manager.get_traffic()
            thresholds = self.proximity_thresholds
            
            logger.info(f"Checking proximity for {len(traffic)} traffic aircraft with thresholds: "
                        f"{thresholds['vertical_ft']}ft / {thresholds['horizontal_nm']}nm")
            
            # Import the unified prediction function
            try:
                from service.routes.route_planning import predict_aircraft_path_unified
            except ImportError:
                from routes.route_planning import predict_aircraft_path_unified
            
            # For each traffic aircraft, predict where it will be using the SAME method as the simulation
            for aircraft in traffic:
                route_duration = planned_path[-1].time_offset_min if planned_path else 0
                if route_duration <= 0:
                    continue
                
                # Get origin/destination if available
                origin_code = getattr(aircraft, 'origin', None) or getattr(aircraft, 'origin_airport', None)
                dest_code = getattr(aircraft, 'destination', None) or getattr(aircraft, 'destination_airport', None)
                
                # Use unified prediction (corridor-based when possible, straight-line fallback)
                # This matches EXACTLY what the frontend simulation uses
                predicted_path_result = predict_aircraft_path_unified(
                    aircraft=aircraft,
                    planner=self,
                    duration_minutes=route_duration,
                    origin_code=origin_code,
                    dest_code=dest_code
                )
                
                predicted_positions = predicted_path_result.get('predicted_path', [])
                
                if not predicted_positions:
                    continue
                
                # Check proximity at each planned route point
                for path_point in planned_path:
                    # Find the predicted position closest to this path point's time
                    closest_prediction = None
                    min_time_diff = float('inf')
                    
                    for pred in predicted_positions:
                        time_diff = abs(pred['time_offset_min'] - path_point.time_offset_min)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_prediction = pred
                    
                    if closest_prediction and min_time_diff <= 5:  # Within 5 minutes
                        # Calculate horizontal distance
                        h_dist = haversine_nm(
                            path_point.lat, path_point.lon,
                            closest_prediction['lat'], closest_prediction['lon']
                        )
                        
                        # Calculate vertical distance
                        v_dist = abs(path_point.alt_ft - closest_prediction['alt_ft'])
                        
                        # Check if within dangerous proximity
                        if h_dist <= thresholds['horizontal_nm'] and v_dist <= thresholds['vertical_ft']:
                            proximity_count += 1
                            logger.info(f"[PROXIMITY] {aircraft.flight_id or aircraft.callsign} at t={path_point.time_offset_min:.1f}min: "
                                       f"{h_dist:.1f}nm horizontal, {v_dist:.0f}ft vertical")
            
            # Calculate proximity score - penalize routes with many dangerous proximities
            # Each proximity encounter reduces score
            if proximity_count > 0:
                proximity_penalty = min(1.0, proximity_count * 0.1)  # 10% penalty per proximity
                proximity_score = max(0.0, 1.0 - proximity_penalty)
                logger.info(f"Route {path.get('id')}: {proximity_count} dangerous proximities "
                           f"({thresholds['vertical_ft']}ft/{thresholds['horizontal_nm']}nm) -> score {proximity_score:.2f}")
        
        # ============================================================
        # TACTICAL ZONE PENALTY - Penalize routes through restricted zones
        # ============================================================
        zone_penalty = 0.0
        zone_violations = None
        
        if hasattr(self, 'tactical_zones') and self.tactical_zones:
            zone_violations = check_route_zone_violations(centerline, self.tactical_zones)
            zone_penalty = zone_violations['total_penalty']
            
            if zone_violations['no_fly_violations'] > 0:
                logger.debug(f"Route {path.get('id')}: {zone_violations['no_fly_violations']} no-fly zone violations")
        
        # ============================================================
        # COMPOSITE SCORE
        # Priority: Conflicts > Proximity > Zones > Corridor Usage > Distance > Safety > Coverage
        # ============================================================
        total_score = (
            CONFLICT_WEIGHT * conflict_score +           # 35% - avoid other planes (imminent conflicts)
            0.20 * proximity_score +                     # 20% - avoid dangerous proximities
            ZONE_PENALTY_WEIGHT * (1 - zone_penalty) +   # 10% - avoid restricted zones (reduced from 15%)
            CORRIDOR_BONUS_WEIGHT * corridor_bonus +     # 15% - use proven corridors (reduced from 20%)
            DISTANCE_WEIGHT * distance_score +           # 10% - prefer shorter (reduced from 15%)
            SAFETY_WEIGHT * safety_score +               # 7%  - corridor precision (reduced from 10%)
            COVERAGE_WEIGHT * coverage_score             # 3%  - path detail (reduced from 5%)
        )
        
        logger.debug(f"Route {path.get('id')} ({route_type}): "
                    f"conflict={conflict_score:.2f}, proximity={proximity_score:.2f}, "
                    f"zone_penalty={zone_penalty:.2f}, corridor={corridor_bonus:.2f}, "
                    f"distance={distance_score:.2f} -> total={total_score:.2f}")
        
        return RouteCandidate(
            path_id=path.get("id", "custom"),
            origin=path.get("origin"),
            destination=path.get("destination"),
            centerline=centerline,
            width_nm=width_nm,
            distance_nm=distance_nm,
            score=total_score,
            distance_score=distance_score,
            safety_score=safety_score,
            coverage_score=coverage_score,
            conflict_score=conflict_score,
            proximity_score=proximity_score,
            proximity_count=proximity_count,
            conflicts=conflicts,
            planned_path=planned_path,
            eta_minutes=eta_minutes,
        )
    
    def plan_routes(
        self,
        origins: List[str],
        destination: str,
    ) -> Dict[str, Any]:
        """Plan routes from multiple origins to a destination (legacy method)."""
        if not origins or len(origins) > 3:
            return {"error": "Please provide 1-3 origin airports", "routes": [], "best_route": None}
        
        if not destination:
            return {"error": "Please provide a destination airport", "routes": [], "best_route": None}
        
        matching_paths = self.find_routes(origins, destination)
        
        if not matching_paths:
            return {
                "error": f"No routes found from {', '.join(origins)} to {destination}",
                "routes": [], "best_route": None,
                "origins": origins, "destination": destination,
            }
        
        profile = CIVIL_AIRCRAFT_PROFILE
        candidates = [self.score_route(path, profile, check_conflicts=False) for path in matching_paths]
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        if candidates:
            candidates[0].recommendation = "best"
            for i, c in enumerate(candidates[1:], 1):
                if c.score >= candidates[0].score * 0.95:
                    c.recommendation = "excellent"
                elif c.score >= candidates[0].score * 0.80:
                    c.recommendation = "good"
                else:
                    c.recommendation = "alternative"
        
        routes = [c.to_dict() for c in candidates]
        return {
            "routes": routes,
            "best_route": routes[0] if routes else None,
            "total_routes": len(routes),
            "origins": origins,
            "destination": destination,
        }
    
    def plan_advanced_route(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        waypoints: List[Dict[str, Any]] = None,
        aircraft_type: str = "civil",
        altitude_ft: Optional[float] = None,
        speed_kts: Optional[float] = None,
        check_conflicts: bool = True,
        tactical_zones: List[Dict[str, Any]] = None,
        proximity_thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advanced route planning with smart corridor utilization and conflict detection.
        
        Generates multiple route options:
        1. Direct great circle route
        2. Learned corridor routes (exact O/D matches)
        3. Chained corridor routes (via intermediate airports)
        4. Corridor-aware routes (uses nearby corridors)
        5. Custom waypoint routes
        
        Args:
            origin: {lat, lon, name?, airport_code?}
            destination: {lat, lon, name?, airport_code?}
            waypoints: List of intermediate waypoints
            aircraft_type: "fighter" or "civil"
            altitude_ft: Override cruise altitude
            speed_kts: Override cruise speed
            check_conflicts: Whether to check for traffic conflicts
            tactical_zones: List of zones for stealth planning (no-fly, altitude, speed zones)
        """
        # Store tactical zones and proximity thresholds for scoring
        self.tactical_zones = tactical_zones or []
        self.proximity_thresholds = proximity_thresholds or {
            'vertical_ft': 999,
            'horizontal_nm': 5,
            'name': 'default'
        }
        # Get profile
        profile_type = AircraftType.FIGHTER if aircraft_type == "fighter" else AircraftType.CIVIL
        profile = AIRCRAFT_PROFILES[profile_type]
        
        # Override profile values if specified
        if altitude_ft or speed_kts:
            profile = AircraftProfile(
                name=profile.name,
                type=profile.type,
                min_speed_kts=profile.min_speed_kts,
                max_speed_kts=profile.max_speed_kts,
                cruise_speed_kts=speed_kts or profile.cruise_speed_kts,
                min_altitude_ft=profile.min_altitude_ft,
                max_altitude_ft=profile.max_altitude_ft,
                cruise_altitude_ft=altitude_ft or profile.cruise_altitude_ft,
                climb_rate_ft_min=profile.climb_rate_ft_min,
                descent_rate_ft_min=profile.descent_rate_ft_min,
                turn_rate_deg_sec=profile.turn_rate_deg_sec,
            )
        
        # Convert to Waypoint objects
        origin_alt = origin.get("alt_ft")
        if not origin_alt and origin.get("airport_code"):
            airport = self.get_airport_by_code(origin.get("airport_code", ""))
            origin_alt = airport.elevation_ft if airport else 0
        
        origin_wp = Waypoint(
            lat=origin["lat"],
            lon=origin["lon"],
            alt_ft=origin_alt or 0,
            name=origin.get("name"),
            is_airport=bool(origin.get("airport_code")),
            airport_code=origin.get("airport_code"),
        )
        
        dest_alt = destination.get("alt_ft")
        if not dest_alt and destination.get("airport_code"):
            airport = self.get_airport_by_code(destination.get("airport_code", ""))
            dest_alt = airport.elevation_ft if airport else 0
        
        dest_wp = Waypoint(
            lat=destination["lat"],
            lon=destination["lon"],
            alt_ft=dest_alt or 0,
            name=destination.get("name"),
            is_airport=bool(destination.get("airport_code")),
            airport_code=destination.get("airport_code"),
        )
        
        waypoint_wps = []
        if waypoints:
            for wp in waypoints:
                waypoint_wps.append(Waypoint(
                    lat=wp["lat"],
                    lon=wp["lon"],
                    alt_ft=wp.get("alt_ft", profile.cruise_altitude_ft),
                    name=wp.get("name"),
                ))
        
        # Generate all route options using the smart path generator
        path_generator = self._get_path_generator()
        route_options = path_generator.generate_all_route_options(
            origin_wp, dest_wp, waypoint_wps, profile
        )
        
        routes = []
        seen_distances = set()  # Avoid duplicate routes
        
        for route_id, path, route_type, corridor_ids in route_options:
            if not path or len(path) < 2:
                continue
            
            # Calculate distance to check for duplicates
            distance = calculate_path_distance(path)
            distance_key = round(distance, 0)  # Round to nearest nm
            
            # Skip if we already have a route with very similar distance
            if distance_key in seen_distances:
                continue
            seen_distances.add(distance_key)
            
            # Determine corridor width based on route type
            # Learned corridors are narrower (more precise), direct routes are wider
            width_nm = 8.0 if route_type == "direct" else 4.0
            
            # Score the route - passing route_type for corridor bonus calculation
            candidate = self.score_route(
                {
                    "id": route_id,
                    "centerline": path,
                    "width_nm": width_nm,
                    "origin": origin_wp.airport_code,
                    "destination": dest_wp.airport_code
                },
                profile,
                check_conflicts,
                route_type=route_type  # Pass route type for corridor bonus
            )
            candidate.recommendation = route_type
            candidate.corridor_ids = corridor_ids  # Store the corridor IDs used
            routes.append(candidate)
        
        # Sort by score (highest first)
        routes.sort(key=lambda r: r.score, reverse=True)
        
        # Mark the best route
        if routes:
            routes[0].recommendation = "best"
        
        # Log route generation summary
        logger.info(f"Generated {len(routes)} route options from {origin_wp.airport_code or 'custom'} "
                   f"to {dest_wp.airport_code or 'custom'}")
        for r in routes:
            logger.debug(f"  - {r.path_id}: {r.distance_nm:.1f}nm, score={r.score:.3f}, "
                        f"conflicts={len(r.conflicts)}")
        
        return {
            "routes": [r.to_dict() for r in routes],
            "best_route": routes[0].to_dict() if routes else None,
            "total_routes": len(routes),
            "origin": origin_wp.to_dict(),
            "destination": dest_wp.to_dict(),
            "waypoints": [wp.to_dict() for wp in waypoint_wps],
            "aircraft_profile": profile.to_dict(),
            "traffic_count": len(self.traffic_manager.get_traffic()),
        }
    
    def get_path_by_id(self, path_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific path by its ID."""
        paths = self._load_paths()
        for path in paths:
            if path.get("id") == path_id:
                return path
        return None
    
    def get_available_destinations(self) -> List[str]:
        """Get list of all unique destinations in the path library."""
        paths = self._load_paths()
        return sorted(set(p.get("destination") for p in paths if p.get("destination")))
    
    def get_available_origins(self) -> List[str]:
        """Get list of all unique origins in the path library."""
        paths = self._load_paths()
        return sorted(set(p.get("origin") for p in paths if p.get("origin")))


# ============================================================
# Strike Route Generator - Geographic Data
# ============================================================

# Major population centers to avoid for stealth operations
POPULATION_CENTERS = [
    {"name": "Tel Aviv", "lat": 32.08, "lon": 34.78, "radius_nm": 15},
    {"name": "Jerusalem", "lat": 31.77, "lon": 35.21, "radius_nm": 10},
    {"name": "Haifa", "lat": 32.79, "lon": 34.99, "radius_nm": 8},
    {"name": "Beersheba", "lat": 31.25, "lon": 34.80, "radius_nm": 6},
    {"name": "Amman", "lat": 31.95, "lon": 35.93, "radius_nm": 12},
    {"name": "Damascus", "lat": 33.51, "lon": 36.29, "radius_nm": 15},
    {"name": "Beirut", "lat": 33.89, "lon": 35.50, "radius_nm": 12},
    {"name": "Cairo", "lat": 30.04, "lon": 31.24, "radius_nm": 20},
    {"name": "Alexandria", "lat": 31.20, "lon": 29.92, "radius_nm": 12},
    {"name": "Port Said", "lat": 31.26, "lon": 32.30, "radius_nm": 8},
    {"name": "Suez", "lat": 29.97, "lon": 32.55, "radius_nm": 8},
    {"name": "Gaza City", "lat": 31.50, "lon": 34.47, "radius_nm": 6},
    {"name": "Aqaba", "lat": 29.53, "lon": 35.01, "radius_nm": 5},
    {"name": "Eilat", "lat": 29.56, "lon": 34.95, "radius_nm": 5},
]

# International border segments for risk calculation (approximate)
# Each segment is a line from (lat1, lon1) to (lat2, lon2)
BORDER_SEGMENTS = [
    # Israel-Lebanon border
    {"name": "Israel-Lebanon", "lat1": 33.05, "lon1": 35.10, "lat2": 33.28, "lon2": 35.62, "risk_factor": 0.8},
    # Israel-Syria border (Golan)
    {"name": "Israel-Syria", "lat1": 33.28, "lon1": 35.62, "lat2": 32.70, "lon2": 35.90, "risk_factor": 0.9},
    # Israel-Jordan border (north)
    {"name": "Israel-Jordan-N", "lat1": 32.70, "lon1": 35.57, "lat2": 31.50, "lon2": 35.50, "risk_factor": 0.4},
    # Israel-Jordan border (south to Eilat)
    {"name": "Israel-Jordan-S", "lat1": 31.50, "lon1": 35.50, "lat2": 29.50, "lon2": 35.00, "risk_factor": 0.4},
    # Israel-Egypt border (Sinai)
    {"name": "Israel-Egypt", "lat1": 31.22, "lon1": 34.27, "lat2": 29.50, "lon2": 34.90, "risk_factor": 0.5},
    # Egypt-Gaza border
    {"name": "Egypt-Gaza", "lat1": 31.22, "lon1": 34.27, "lat2": 31.60, "lon2": 34.22, "risk_factor": 0.7},
    # Syria-Lebanon border
    {"name": "Syria-Lebanon", "lat1": 33.05, "lon1": 35.10, "lat2": 34.69, "lon2": 36.27, "risk_factor": 0.6},
    # Syria-Jordan border
    {"name": "Syria-Jordan", "lat1": 32.70, "lon1": 35.90, "lat2": 32.30, "lon2": 39.00, "risk_factor": 0.5},
]

# Known military/sensitive areas (simplified)
SENSITIVE_AREAS = [
    {"name": "Dimona", "lat": 31.07, "lon": 35.15, "radius_nm": 15, "risk_factor": 1.0},
    {"name": "Palmachim", "lat": 31.90, "lon": 34.69, "radius_nm": 8, "risk_factor": 0.7},
    {"name": "Nevatim AFB", "lat": 31.21, "lon": 35.01, "radius_nm": 10, "risk_factor": 0.6},
]


# ============================================================
# Strike Route Generator
# ============================================================

@dataclass
class StrikeWaypoint:
    """A waypoint in a strike route."""
    lat: float
    lon: float
    alt_ft: float
    time_offset_min: float
    waypoint_type: str  # 'origin', 'ingress', 'target', 'egress', 'return'
    name: Optional[str] = None
    risk_score: float = 0.0


@dataclass
class StrikePhase:
    """A phase of the strike mission."""
    name: str  # 'ingress', 'strike', 'egress'
    waypoints: List[StrikeWaypoint]
    distance_nm: float
    duration_min: float
    avg_risk: float


@dataclass
class StrikeRoute:
    """Complete strike mission route."""
    route_id: str
    origin: Dict[str, Any]
    targets: List[Dict[str, Any]]
    phases: List[StrikePhase]
    total_distance_nm: float
    total_duration_min: float
    total_risk_score: float
    planned_path: List[Dict[str, Any]]  # Full path with all points
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "origin": self.origin,
            "targets": self.targets,
            "phases": [
                {
                    "name": p.name,
                    "waypoints": [
                        {
                            "lat": w.lat,
                            "lon": w.lon,
                            "alt_ft": w.alt_ft,
                            "time_offset_min": w.time_offset_min,
                            "waypoint_type": w.waypoint_type,
                            "name": w.name,
                            "risk_score": w.risk_score,
                        }
                        for w in p.waypoints
                    ],
                    "distance_nm": p.distance_nm,
                    "duration_min": p.duration_min,
                    "avg_risk": p.avg_risk,
                }
                for p in self.phases
            ],
            "total_distance_nm": self.total_distance_nm,
            "total_duration_min": self.total_duration_min,
            "total_risk_score": self.total_risk_score,
            "planned_path": self.planned_path,
            "centerline": self.planned_path,  # Alias for compatibility
            "width_nm": 4.0,  # Default width for visualization
        }


class StrikeRouteGenerator:
    """
    Generates tactical strike routes that:
    - Create direct paths to targets (not via airports)
    - Consider terrain risk and threat avoidance
    - Respect tactical zones (no-fly, low-altitude, stealth)
    - Plan ingress, strike, and egress phases
    """
    
    def __init__(self, tactical_zones: List[Dict[str, Any]] = None):
        self.tactical_zones = tactical_zones or []
    
    def calculate_point_risk(self, lat: float, lon: float) -> float:
        """
        Calculate risk score for a single point (0-1 scale, higher = more risk).
        
        Considers:
        - Distance from international borders
        - Distance from population centers
        - Distance from sensitive areas
        - Tactical zone penalties
        """
        risk = 0.0
        
        # 1. Border proximity risk (within 20nm of borders = higher risk)
        min_border_dist = float('inf')
        max_border_risk = 0.0
        
        for border in BORDER_SEGMENTS:
            dist = self._distance_to_line_segment(
                lat, lon,
                border["lat1"], border["lon1"],
                border["lat2"], border["lon2"]
            )
            if dist < min_border_dist:
                min_border_dist = dist
                if dist < 20:  # Within 20nm
                    max_border_risk = max(max_border_risk, 
                        border.get("risk_factor", 0.5) * (1 - dist / 20))
        
        risk += max_border_risk * 0.3
        
        # 2. Population center avoidance (for stealth)
        for city in POPULATION_CENTERS:
            dist = haversine_nm(lat, lon, city["lat"], city["lon"])
            radius = city.get("radius_nm", 10)
            if dist < radius:
                # Inside the avoidance radius
                risk += 0.2 * (1 - dist / radius)
        
        # 3. Sensitive area penalties
        for area in SENSITIVE_AREAS:
            dist = haversine_nm(lat, lon, area["lat"], area["lon"])
            radius = area.get("radius_nm", 10)
            if dist < radius:
                risk += area.get("risk_factor", 0.5) * (1 - dist / radius)
        
        # 4. Tactical zone penalties
        for zone in self.tactical_zones:
            polygon = zone.get("polygon", [])
            if polygon and point_in_polygon(lat, lon, polygon):
                zone_type = zone.get("type", "")
                if zone_type == "no-fly":
                    risk += 1.0  # Prohibitive
                elif zone_type == "high-risk":
                    risk += 0.5
                elif zone_type == "low-altitude":
                    # Low altitude zone - not a risk, but requires altitude adjustment
                    pass
        
        return min(risk, 1.0)
    
    def _distance_to_line_segment(
        self, 
        lat: float, lon: float,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance from a point to a line segment in nautical miles."""
        # Convert to simple Cartesian approximation (good enough for nearby points)
        # This is a simplified calculation
        
        # Vector from p1 to p2
        dx = lon2 - lon1
        dy = lat2 - lat1
        
        # Vector from p1 to point
        px = lon - lon1
        py = lat - lat1
        
        # Project point onto line
        line_len_sq = dx * dx + dy * dy
        if line_len_sq < 0.0001:
            # Segment is essentially a point
            return haversine_nm(lat, lon, lat1, lon1)
        
        t = max(0, min(1, (px * dx + py * dy) / line_len_sq))
        
        # Closest point on segment
        closest_lon = lon1 + t * dx
        closest_lat = lat1 + t * dy
        
        return haversine_nm(lat, lon, closest_lat, closest_lon)
    
    def calculate_path_risk(self, path: List[Dict[str, float]]) -> float:
        """Calculate average risk score for an entire path."""
        if not path:
            return 0.0
        
        total_risk = 0.0
        for point in path:
            total_risk += self.calculate_point_risk(point["lat"], point["lon"])
        
        return total_risk / len(path)
    
    def optimize_target_order(
        self, 
        origin: Dict[str, Any], 
        targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize the order of targets using nearest-neighbor heuristic.
        Returns targets in optimized order.
        """
        if len(targets) <= 1:
            return targets
        
        # Sort by priority first (high priority first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_targets = sorted(targets, key=lambda t: priority_order.get(t.get("priority", "medium"), 1))
        
        # For targets with same priority, use nearest-neighbor
        result = []
        remaining = list(sorted_targets)
        current_lat, current_lon = origin["lat"], origin["lon"]
        
        while remaining:
            # Find nearest target
            nearest_idx = 0
            nearest_dist = float('inf')
            
            for i, target in enumerate(remaining):
                dist = haversine_nm(current_lat, current_lon, target["lat"], target["lon"])
                # Weight by priority (high priority targets are "closer")
                priority_weight = 1.0 + 0.2 * priority_order.get(target.get("priority", "medium"), 1)
                weighted_dist = dist * priority_weight
                
                if weighted_dist < nearest_dist:
                    nearest_dist = weighted_dist
                    nearest_idx = i
            
            nearest = remaining.pop(nearest_idx)
            result.append(nearest)
            current_lat, current_lon = nearest["lat"], nearest["lon"]
        
        return result
    
    def _is_point_in_no_fly_zone(self, lat: float, lon: float) -> bool:
        """Check if a point is inside any no-fly zone."""
        for zone in self.tactical_zones:
            if zone.get("type") == "no-fly":
                polygon = zone.get("polygon", [])
                if polygon and point_in_polygon(lat, lon, polygon):
                    return True
        return False
    
    def _get_no_fly_zones(self) -> List[Dict[str, Any]]:
        """Get all no-fly zones."""
        return [z for z in self.tactical_zones if z.get("type") == "no-fly"]
    
    def _get_zone_centroid(self, polygon: List[Dict[str, float]]) -> Tuple[float, float]:
        """Calculate the centroid of a polygon."""
        if not polygon:
            return 0, 0
        lat_sum = sum(p.get("lat", 0) for p in polygon)
        lon_sum = sum(p.get("lon", 0) for p in polygon)
        n = len(polygon)
        return lat_sum / n, lon_sum / n
    
    def _get_zone_bounds(self, polygon: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
        """Get bounding box of a polygon (min_lat, max_lat, min_lon, max_lon)."""
        if not polygon:
            return 0, 0, 0, 0
        lats = [p.get("lat", 0) for p in polygon]
        lons = [p.get("lon", 0) for p in polygon]
        return min(lats), max(lats), min(lons), max(lons)
    
    def _find_avoidance_waypoints(
        self,
        start_lat: float, start_lon: float,
        end_lat: float, end_lon: float
    ) -> List[Tuple[float, float]]:
        """
        Find waypoints to route around no-fly zones.
        Returns list of (lat, lon) waypoints to pass through.
        """
        no_fly_zones = self._get_no_fly_zones()
        if not no_fly_zones:
            return []
        
        # Check if direct path crosses any no-fly zone
        crosses_zone = False
        blocking_zones = []
        
        for zone in no_fly_zones:
            polygon = zone.get("polygon", [])
            if not polygon:
                continue
            
            # Sample points along the direct path to check for intersection
            for i in range(20):
                fraction = i / 19
                test_lat = start_lat + (end_lat - start_lat) * fraction
                test_lon = start_lon + (end_lon - start_lon) * fraction
                
                if point_in_polygon(test_lat, test_lon, polygon):
                    crosses_zone = True
                    blocking_zones.append(zone)
                    break
        
        if not crosses_zone:
            return []
        
        # Find waypoints to go around the blocking zones
        waypoints = []
        
        for zone in blocking_zones:
            polygon = zone.get("polygon", [])
            if not polygon:
                continue
            
            min_lat, max_lat, min_lon, max_lon = self._get_zone_bounds(polygon)
            centroid_lat, centroid_lon = self._get_zone_centroid(polygon)
            
            # Determine which side to go around based on start/end positions
            # Calculate the bearing from start to end
            bearing_to_end = math.atan2(end_lon - start_lon, end_lat - start_lat)
            bearing_to_centroid = math.atan2(centroid_lon - start_lon, centroid_lat - start_lat)
            
            # Calculate angle difference to determine which side
            angle_diff = bearing_to_centroid - bearing_to_end
            
            # Normalize to -pi to pi
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Add buffer distance (in degrees, roughly 0.2 deg ~ 12nm)
            buffer = 0.15
            
            # Determine avoidance direction based on which side is shorter
            # and which side the zone is relative to the direct path
            
            # Calculate perpendicular offset direction
            path_bearing = math.atan2(end_lon - start_lon, end_lat - start_lat)
            
            if angle_diff > 0:
                # Zone is to the left of the path, go right (clockwise)
                perp_bearing = path_bearing - math.pi / 2
            else:
                # Zone is to the right of the path, go left (counter-clockwise)
                perp_bearing = path_bearing + math.pi / 2
            
            # Find the edge of the zone closest to the path and offset from there
            # Use the zone's bounding box corners as potential waypoints
            corners = [
                (min_lat - buffer, min_lon - buffer),
                (min_lat - buffer, max_lon + buffer),
                (max_lat + buffer, min_lon - buffer),
                (max_lat + buffer, max_lon + buffer),
            ]
            
            # Find the corner that:
            # 1. Is on the correct side of the zone (based on angle_diff)
            # 2. Minimizes total path length
            
            best_corner = None
            best_total_dist = float('inf')
            
            for corner_lat, corner_lon in corners:
                # Skip if this corner is inside any no-fly zone
                if self._is_point_in_no_fly_zone(corner_lat, corner_lon):
                    continue
                
                # Calculate total distance via this corner
                dist_to_corner = haversine_nm(start_lat, start_lon, corner_lat, corner_lon)
                dist_from_corner = haversine_nm(corner_lat, corner_lon, end_lat, end_lon)
                total_dist = dist_to_corner + dist_from_corner
                
                if total_dist < best_total_dist:
                    best_total_dist = total_dist
                    best_corner = (corner_lat, corner_lon)
            
            if best_corner:
                waypoints.append(best_corner)
        
        return waypoints
    
    def generate_attack_pattern(
        self,
        target_lat: float, target_lon: float,
        ingress_heading_deg: float,
        attack_type: str = "dive",
        start_time: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate a specific attack maneuver pattern relative to the target.
        
        Args:
            target_lat, target_lon: Target location
            ingress_heading_deg: Heading TOWARDS the target
            attack_type: 'dive' or 'pop-up'
            start_time: Start time offset in minutes
            
        Returns:
            List of waypoints for the attack run
        """
        points = []
        
        # Helper to add point relative to target
        def add_offset_point(dist_nm, heading_from_target, alt_ft, relative_time_min, label=""):
            # Calculate lat/lon given distance and heading from target
            lat_deg_per_nm = 1/60.0
            # Approximate lon scaling at target latitude
            lon_deg_per_nm = 1/(60.0 * math.cos(math.radians(target_lat)))
            
            angle_rad = math.radians(heading_from_target)
            
            d_lat = dist_nm * math.cos(angle_rad) * lat_deg_per_nm
            d_lon = dist_nm * math.sin(angle_rad) * lon_deg_per_nm
            
            p_lat = target_lat + d_lat
            p_lon = target_lon + d_lon
            
            points.append({
                "lat": p_lat,
                "lon": p_lon,
                "alt_ft": alt_ft,
                "time_offset_min": start_time + relative_time_min,
                "risk_score": 0.8, # Attack runs are high risk
                "label": label
            })

        # Calculate heading FROM target (reciprocal of ingress)
        heading_from_target = (ingress_heading_deg + 180) % 360
        
        if attack_type == "pop-up":
            # Low-level ingress -> Pop-up -> Dive
            
            # 1. IP (Initial Point): 10nm out, Low
            add_offset_point(10.0, heading_from_target, 500, 0.0, "IP")
            
            # 2. Action Point: 3nm out, Low -> Start Pull-up
            add_offset_point(3.0, heading_from_target, 500, 1.2, "Pop-up Start")
            
            # 3. Apex: 1.5nm out, High (4000ft) -> Roll in
            add_offset_point(1.5, heading_from_target, 4000, 1.4, "Apex")
            
            # 4. Release: 1.0nm out, Diving (2000ft)
            add_offset_point(1.0, heading_from_target, 2000, 1.5, "Release")
            
            # 5. Recovery: Fly over target, Low (500ft)
            points.append({
                "lat": target_lat,
                "lon": target_lon,
                "alt_ft": 500,
                "time_offset_min": start_time + 1.6,
                "risk_score": 1.0,
                "label": "Target"
            })
            
        else: # High altitude dive (default)
            # 1. IP: 10nm out, High
            add_offset_point(10.0, heading_from_target, 25000, 0.0, "IP")
            
            # 2. Roll-in: 4nm out, High
            add_offset_point(4.0, heading_from_target, 25000, 1.2, "Roll-in")
            
            # 3. Release: 2nm out, Medium (15000ft)
            add_offset_point(2.0, heading_from_target, 15000, 1.4, "Release")
            
            # 4. Pull up: 0nm (Target), Medium-Low (10000ft)
            points.append({
                "lat": target_lat,
                "lon": target_lon,
                "alt_ft": 10000,
                "time_offset_min": start_time + 1.6,
                "risk_score": 1.0,
                "label": "Target"
            })

        return points

    def generate_direct_path(
        self,
        start_lat: float, start_lon: float,
        end_lat: float, end_lon: float,
        start_alt: float, end_alt: float,
        cruise_alt: float,
        num_points: int = 20,
        start_time: float = 0.0,
        speed_kts: float = 500.0
    ) -> List[Dict[str, Any]]:
        """
        Generate a path between two points, avoiding no-fly zones.
        """
        # Check if we need to route around no-fly zones
        avoidance_waypoints = self._find_avoidance_waypoints(
            start_lat, start_lon, end_lat, end_lon
        )
        
        logger.debug(f"generate_direct_path: ({start_lat:.4f},{start_lon:.4f}) -> ({end_lat:.4f},{end_lon:.4f})")
        logger.debug(f"  Avoidance waypoints: {avoidance_waypoints}")
        
        if avoidance_waypoints:
            # Generate path through waypoints
            return self._generate_path_through_waypoints(
                start_lat, start_lon,
                end_lat, end_lon,
                avoidance_waypoints,
                start_alt, end_alt, cruise_alt,
                num_points, start_time, speed_kts
            )
        
        # No avoidance needed - generate direct path
        return self._generate_simple_path(
            start_lat, start_lon,
            end_lat, end_lon,
            start_alt, end_alt, cruise_alt,
            num_points, start_time, speed_kts
        )
    
    def _generate_path_through_waypoints(
        self,
        start_lat: float, start_lon: float,
        end_lat: float, end_lon: float,
        waypoints: List[Tuple[float, float]],
        start_alt: float, end_alt: float,
        cruise_alt: float,
        num_points: int,
        start_time: float,
        speed_kts: float
    ) -> List[Dict[str, Any]]:
        """Generate a path that goes through intermediate waypoints."""
        # Build list of all points: start -> waypoints -> end
        all_points = [(start_lat, start_lon)] + waypoints + [(end_lat, end_lon)]
        
        # Calculate total distance
        total_dist = 0
        segment_dists = []
        for i in range(len(all_points) - 1):
            dist = haversine_nm(
                all_points[i][0], all_points[i][1],
                all_points[i+1][0], all_points[i+1][1]
            )
            segment_dists.append(dist)
            total_dist += dist
        
        # Distribute points proportionally across segments
        path = []
        current_time = start_time
        cumulative_dist = 0
        
        for seg_idx in range(len(all_points) - 1):
            seg_start = all_points[seg_idx]
            seg_end = all_points[seg_idx + 1]
            seg_dist = segment_dists[seg_idx]
            
            # Number of points for this segment (proportional to distance)
            seg_points = max(5, int(num_points * seg_dist / total_dist))
            
            # Use seg_points + 1 to include both endpoints, but skip first point 
            # of subsequent segments to avoid duplicates
            start_i = 0 if seg_idx == 0 else 1
            
            for i in range(start_i, seg_points + 1):
                fraction = i / seg_points if seg_points > 0 else 0
                
                lat = seg_start[0] + (seg_end[0] - seg_start[0]) * fraction
                lon = seg_start[1] + (seg_end[1] - seg_start[1]) * fraction
                
                # Calculate overall progress for altitude profile
                point_dist = cumulative_dist + seg_dist * fraction
                overall_fraction = point_dist / total_dist if total_dist > 0 else 0
                
                # Altitude profile
                if overall_fraction < 0.15:
                    alt_fraction = overall_fraction / 0.15
                    alt = start_alt + (cruise_alt - start_alt) * alt_fraction
                elif overall_fraction > 0.85:
                    alt_fraction = (1.0 - overall_fraction) / 0.15
                    alt = end_alt + (cruise_alt - end_alt) * alt_fraction
                else:
                    alt = cruise_alt
                
                # Time offset
                time_offset = current_time + (point_dist / speed_kts) * 60
                
                # Risk and low-altitude zone check
                risk = self.calculate_point_risk(lat, lon)
                in_low_alt_zone = False
                for zone in self.tactical_zones:
                    if zone.get("type") == "low-altitude":
                        polygon = zone.get("polygon", [])
                        if polygon and point_in_polygon(lat, lon, polygon):
                            in_low_alt_zone = True
                            max_alt = zone.get("max_altitude_ft", 5000)
                            alt = min(alt, max_alt)
                            break
                
                path.append({
                    "lat": lat,
                    "lon": lon,
                    "alt_ft": alt,
                    "time_offset_min": time_offset,
                    "risk_score": risk,
                    "in_low_alt_zone": in_low_alt_zone,
                })
            
            cumulative_dist += seg_dist
        
        return path
    
    def _generate_simple_path(
        self,
        start_lat: float, start_lon: float,
        end_lat: float, end_lon: float,
        start_alt: float, end_alt: float,
        cruise_alt: float,
        num_points: int = 20,
        start_time: float = 0.0,
        speed_kts: float = 500.0
    ) -> List[Dict[str, Any]]:
        """
        Generate a simple direct path between two points with proper altitude profile.
        """
        path = []
        total_dist = haversine_nm(start_lat, start_lon, end_lat, end_lon)
        
        for i in range(num_points):
            fraction = i / (num_points - 1) if num_points > 1 else 0
            
            # Linear interpolation for position
            lat = start_lat + (end_lat - start_lat) * fraction
            lon = start_lon + (end_lon - start_lon) * fraction
            
            # Altitude profile: climb, cruise, descend
            if fraction < 0.15:  # Climb phase
                alt_fraction = fraction / 0.15
                alt = start_alt + (cruise_alt - start_alt) * alt_fraction
            elif fraction > 0.85:  # Descend phase
                alt_fraction = (1.0 - fraction) / 0.15
                alt = end_alt + (cruise_alt - end_alt) * alt_fraction
            else:  # Cruise phase
                alt = cruise_alt
            
            # Calculate time offset
            dist_so_far = total_dist * fraction
            time_offset = start_time + (dist_so_far / speed_kts) * 60  # Convert to minutes
            
            # Calculate risk for this point
            risk = self.calculate_point_risk(lat, lon)
            
            # Check if in low-altitude zone
            in_low_alt_zone = False
            for zone in self.tactical_zones:
                if zone.get("type") == "low-altitude":
                    polygon = zone.get("polygon", [])
                    if polygon and point_in_polygon(lat, lon, polygon):
                        in_low_alt_zone = True
                        max_alt = zone.get("max_altitude_ft", 5000)
                        alt = min(alt, max_alt)
                        break
            
            path.append({
                "lat": lat,
                "lon": lon,
                "alt_ft": alt,
                "time_offset_min": time_offset,
                "risk_score": risk,
                "in_low_alt_zone": in_low_alt_zone,
            })
        
        return path
    
    def generate_strike_route(
        self,
        origin: Dict[str, Any],
        targets: List[Dict[str, Any]],
        aircraft_profile: AircraftProfile,
        return_to_base: bool = True
    ) -> StrikeRoute:
        """
        Generate a complete strike mission route.
        
        Args:
            origin: Base location {lat, lon, name, ...}
            targets: List of targets [{lat, lon, name, priority, ...}, ...]
            aircraft_profile: Aircraft performance characteristics
            return_to_base: Whether to include return leg
            
        Returns:
            StrikeRoute with ingress, strike, and egress phases
        """
        logger.info(f"generate_strike_route called with {len(targets)} targets")
        for i, t in enumerate(targets):
            logger.info(f"  Target {i}: {t.get('name')} at ({t.get('lat'):.4f}, {t.get('lon'):.4f})")
        
        # Optimize target order
        ordered_targets = self.optimize_target_order(origin, targets)
        logger.info(f"After optimization, target order: {[t.get('name') for t in ordered_targets]}")
        
        cruise_alt = aircraft_profile.cruise_altitude_ft
        speed_kts = aircraft_profile.cruise_speed_kts
        
        phases = []
        all_waypoints = []
        current_time = 0.0
        total_distance = 0.0
        
        # ============================================================
        # Phase 1: Ingress (Base to First Target)
        # ============================================================
        if ordered_targets:
            first_target = ordered_targets[0]
            
            ingress_path = self.generate_direct_path(
                origin["lat"], origin["lon"],
                first_target["lat"], first_target["lon"],
                start_alt=0,  # Start on ground
                end_alt=cruise_alt,  # Arrive at cruise
                cruise_alt=cruise_alt,
                num_points=30,
                start_time=current_time,
                speed_kts=speed_kts
            )
            
            ingress_dist = haversine_nm(
                origin["lat"], origin["lon"],
                first_target["lat"], first_target["lon"]
            )
            ingress_duration = (ingress_dist / speed_kts) * 60
            ingress_risk = self.calculate_path_risk(ingress_path)
            
            ingress_waypoints = [
                StrikeWaypoint(
                    lat=p["lat"],
                    lon=p["lon"],
                    alt_ft=p["alt_ft"],
                    time_offset_min=p["time_offset_min"],
                    waypoint_type="ingress" if i > 0 else "origin",
                    name=origin.get("name") if i == 0 else None,
                    risk_score=p["risk_score"]
                )
                for i, p in enumerate(ingress_path)
            ]
            
            phases.append(StrikePhase(
                name="ingress",
                waypoints=ingress_waypoints,
                distance_nm=ingress_dist,
                duration_min=ingress_duration,
                avg_risk=ingress_risk
            ))
            
            all_waypoints.extend(ingress_path)
            current_time += ingress_duration
            total_distance += ingress_dist
        
        # ============================================================
        # Phase 2: Strike (Between Targets)
        # ============================================================
        strike_waypoints = []
        strike_distance = 0.0
        strike_duration = 0.0
        strike_risks = []
        
        for i in range(len(ordered_targets)):
            target = ordered_targets[i]
            
            # Add target waypoint to phase tracking
            target_wp = StrikeWaypoint(
                lat=target["lat"],
                lon=target["lon"],
                alt_ft=cruise_alt,
                time_offset_min=current_time,
                waypoint_type="target",
                name=target.get("name", f"Target {i+1}"),
                risk_score=self.calculate_point_risk(target["lat"], target["lon"])
            )
            strike_waypoints.append(target_wp)
            
            # Path to next target (if any)
            if i < len(ordered_targets) - 1:
                next_target = ordered_targets[i + 1]
                
                segment_path = self.generate_direct_path(
                    target["lat"], target["lon"],
                    next_target["lat"], next_target["lon"],
                    start_alt=cruise_alt,
                    end_alt=cruise_alt,
                    cruise_alt=cruise_alt,
                    num_points=15,
                    start_time=current_time,
                    speed_kts=speed_kts
                )
                
                segment_dist = haversine_nm(
                    target["lat"], target["lon"],
                    next_target["lat"], next_target["lon"]
                )
                segment_duration = (segment_dist / speed_kts) * 60
                
                # Add ALL points from segment - the full path including start and end
                # Skip first point only if it duplicates the last point in all_waypoints
                if all_waypoints and len(segment_path) > 0:
                    last_wp = all_waypoints[-1]
                    first_seg = segment_path[0]
                    # Check if first segment point is very close to last waypoint (within ~0.001 deg)
                    if abs(last_wp["lat"] - first_seg["lat"]) < 0.001 and abs(last_wp["lon"] - first_seg["lon"]) < 0.001:
                        all_waypoints.extend(segment_path[1:])
                    else:
                        all_waypoints.extend(segment_path)
                else:
                    all_waypoints.extend(segment_path)
                
                strike_risks.append(self.calculate_path_risk(segment_path))
                
                current_time += segment_duration
                strike_distance += segment_dist
                strike_duration += segment_duration
                total_distance += segment_dist
        
        if strike_waypoints:
            phases.append(StrikePhase(
                name="strike",
                waypoints=strike_waypoints,
                distance_nm=strike_distance,
                duration_min=strike_duration,
                avg_risk=sum(strike_risks) / len(strike_risks) if strike_risks else 0.0
            ))
        
        # Ensure the last target is the last point in all_waypoints before egress
        if ordered_targets and len(ordered_targets) > 0:
            last_target = ordered_targets[-1]
            if all_waypoints:
                last_wp = all_waypoints[-1]
                # If last waypoint is not close to last target, add it
                if abs(last_wp["lat"] - last_target["lat"]) > 0.001 or abs(last_wp["lon"] - last_target["lon"]) > 0.001:
                    all_waypoints.append({
                        "lat": last_target["lat"],
                        "lon": last_target["lon"],
                        "alt_ft": cruise_alt,
                        "time_offset_min": current_time,
                        "risk_score": self.calculate_point_risk(last_target["lat"], last_target["lon"]),
                        "in_low_alt_zone": False,
                    })
        
        # ============================================================
        # Phase 3: Egress (Last Target to Base)
        # ============================================================
        if return_to_base and ordered_targets:
            last_target = ordered_targets[-1]
            
            egress_path = self.generate_direct_path(
                last_target["lat"], last_target["lon"],
                origin["lat"], origin["lon"],
                start_alt=cruise_alt,
                end_alt=0,  # Land at base
                cruise_alt=cruise_alt,
                num_points=30,
                start_time=current_time,
                speed_kts=speed_kts
            )
            
            egress_dist = haversine_nm(
                last_target["lat"], last_target["lon"],
                origin["lat"], origin["lon"]
            )
            egress_duration = (egress_dist / speed_kts) * 60
            egress_risk = self.calculate_path_risk(egress_path)
            
            egress_waypoints = [
                StrikeWaypoint(
                    lat=p["lat"],
                    lon=p["lon"],
                    alt_ft=p["alt_ft"],
                    time_offset_min=p["time_offset_min"],
                    waypoint_type="egress" if i < len(egress_path) - 1 else "return",
                    name=origin.get("name") if i == len(egress_path) - 1 else None,
                    risk_score=p["risk_score"]
                )
                for i, p in enumerate(egress_path)
            ]
            
            phases.append(StrikePhase(
                name="egress",
                waypoints=egress_waypoints,
                distance_nm=egress_dist,
                duration_min=egress_duration,
                avg_risk=egress_risk
            ))
            
            all_waypoints.extend(egress_path[1:])  # Skip first point
            current_time += egress_duration
            total_distance += egress_dist
        
        # ============================================================
        # Build final route
        # ============================================================
        total_risk = sum(p.avg_risk for p in phases) / len(phases) if phases else 0.0
        
        # Convert all_waypoints to planned_path format
        planned_path = [
            {
                "lat": p["lat"],
                "lon": p["lon"],
                "alt_ft": p["alt_ft"],
                "time_offset_min": p["time_offset_min"],
            }
            for p in all_waypoints
        ]
        
        return StrikeRoute(
            route_id=f"strike_{int(time.time())}",
            origin=origin,
            targets=ordered_targets,
            phases=phases,
            total_distance_nm=total_distance,
            total_duration_min=current_time,
            total_risk_score=total_risk,
            planned_path=planned_path
        )
    
    def plan_multi_aircraft_strike(
        self,
        origin: Dict[str, Any],
        targets: List[Dict[str, Any]],
        aircraft_list: List[Dict[str, Any]],
        aircraft_profile: AircraftProfile
    ) -> List[Dict[str, Any]]:
        """
        Plan strike routes for multiple aircraft with target assignments.
        
        Args:
            origin: Base location
            targets: All targets to hit
            aircraft_list: List of aircraft with ammo capacity
            aircraft_profile: Aircraft performance characteristics
            
        Returns:
            List of aircraft with assigned targets and routes
        """
        # Sort targets by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_targets = sorted(
            targets, 
            key=lambda t: priority_order.get(t.get("priority", "medium"), 1)
        )
        
        # Assign targets to aircraft (greedy algorithm)
        aircraft_assignments = []
        for aircraft in aircraft_list:
            aircraft_assignments.append({
                **aircraft,
                "assigned_targets": [],
                "remaining_ammo": aircraft.get("ammoCapacity", 4),
            })
        
        for target in sorted_targets:
            ammo_required = target.get("ammoRequired", 1)
            
            # Find aircraft with enough ammo, prefer one with fewer assignments
            available = [
                a for a in aircraft_assignments 
                if a["remaining_ammo"] >= ammo_required
            ]
            available.sort(key=lambda a: len(a["assigned_targets"]))
            
            if available:
                available[0]["assigned_targets"].append(target)
                available[0]["remaining_ammo"] -= ammo_required
        
        # Generate routes for each aircraft
        result = []
        for aircraft in aircraft_assignments:
            if aircraft["assigned_targets"]:
                # Determine profile
                current_profile = aircraft_profile
                ac_type = aircraft.get("type")
                if ac_type:
                     # Check explicit types in AIRCRAFT_PROFILES
                     current_profile = AIRCRAFT_PROFILES.get(ac_type, aircraft_profile)
                
                route = self.generate_strike_route(
                    origin=origin,
                    targets=aircraft["assigned_targets"],
                    aircraft_profile=current_profile,
                    return_to_base=True
                )
                
                result.append({
                    "id": aircraft.get("id"),
                    "callsign": aircraft.get("callsign"),
                    "ammoCapacity": aircraft.get("ammoCapacity"),
                    "color": aircraft.get("color", "#22c55e"),
                    "assignedTargets": [t.get("id") for t in aircraft["assigned_targets"]],
                    "route": route.to_dict(),
                })
            else:
                result.append({
                    "id": aircraft.get("id"),
                    "callsign": aircraft.get("callsign"),
                    "ammoCapacity": aircraft.get("ammoCapacity"),
                    "color": aircraft.get("color", "#22c55e"),
                    "assignedTargets": [],
                    "route": None,
                })
        
        return result


# ============================================================
# Singleton Instance
# ============================================================

_planner_instance: Optional[RoutePlanner] = None
_strike_generator_instance: Optional[StrikeRouteGenerator] = None


def get_route_planner() -> RoutePlanner:
    """Get or create the singleton RoutePlanner instance."""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = RoutePlanner()
    return _planner_instance


def get_strike_generator(tactical_zones: List[Dict[str, Any]] = None) -> StrikeRouteGenerator:
    """Get or create a StrikeRouteGenerator instance."""
    return StrikeRouteGenerator(tactical_zones=tactical_zones)
