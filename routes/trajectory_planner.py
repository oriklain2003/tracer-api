"""
Trajectory Planning Routes - Real-time conflict detection with interpolated flight paths.

This module provides:
1. Trajectory interpolation (position every 10 seconds based on speed)
2. FR24 airport traffic fetching (departures/arrivals)
3. Conflict detection with commercial traffic
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests as http_requests

# FlightRadar24 API for scheduled flights
try:
    from FlightRadar24 import FlightRadar24API
    FR24_API_AVAILABLE = True
except ImportError:
    FR24_API_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Trajectory Planning"])

# Initialize FR24 API client
fr24_api = None
if FR24_API_AVAILABLE:
    try:
        fr24_api = FlightRadar24API()
        logger.info("FlightRadar24 API initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize FlightRadar24 API: {e}")

# Configuration
PROJECT_ROOT: Path = None
CONFIGURED_AIRPORTS: List[Dict[str, Any]] = []

# ============================================================
# Helper Functions
# ============================================================

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


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from point 1 to point 2 in degrees."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360


def destination_point(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> Tuple[float, float]:
    """Calculate destination point given start, bearing and distance."""
    R = 3440.065  # Earth radius in nautical miles
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)
    
    d = distance_nm / R
    
    lat2 = math.asin(
        math.sin(lat_rad) * math.cos(d) +
        math.cos(lat_rad) * math.sin(d) * math.cos(bearing_rad)
    )
    
    lon2 = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(d) * math.cos(lat_rad),
        math.cos(d) - math.sin(lat_rad) * math.sin(lat2)
    )
    
    return math.degrees(lat2), math.degrees(lon2)


# Major airport coordinates for bearing calculation
# This helps estimate flight direction based on destination/origin
AIRPORT_COORDINATES: Dict[str, Tuple[float, float]] = {
    # Israel
    "TLV": (32.0114, 34.8867), "LLBG": (32.0114, 34.8867),
    "SDV": (32.1147, 34.7822), "LLSD": (32.1147, 34.7822),
    "ETH": (29.5613, 34.9601), "LLET": (29.5613, 34.9601), "LLETM": (29.5613, 34.9601),
    "VDA": (29.9403, 34.9358), "LLMG": (29.9403, 34.9358),
    "HFA": (32.8094, 35.0431), "LLHA": (32.8094, 35.0431),
    
    # Middle East
    "AMM": (31.7226, 35.9932), "OJAI": (31.7226, 35.9932),
    "AQJ": (29.6116, 35.0181), "OJAQ": (29.6116, 35.0181),
    "BEY": (33.8209, 35.4884), "OLBA": (33.8209, 35.4884),
    "DAM": (33.4114, 36.5156), "OSDI": (33.4114, 36.5156),
    "CAI": (30.1219, 31.4056), "HECA": (30.1219, 31.4056),
    "SSH": (27.9773, 34.3950), "HESH": (27.9773, 34.3950),
    "HRG": (27.1783, 33.7994), "HEGR": (27.1783, 33.7994),
    "DXB": (25.2528, 55.3644), "OMDB": (25.2528, 55.3644),
    "AUH": (24.4330, 54.6511), "OMAA": (24.4330, 54.6511),
    "DOH": (25.2731, 51.6081), "OTHH": (25.2731, 51.6081),
    "RUH": (24.9576, 46.6988), "OERK": (24.9576, 46.6988),
    "JED": (21.6796, 39.1565), "OEJN": (21.6796, 39.1565),
    
    # Europe
    "LHR": (51.4700, -0.4543), "EGLL": (51.4700, -0.4543),
    "CDG": (49.0097, 2.5479), "LFPG": (49.0097, 2.5479),
    "FRA": (50.0379, 8.5622), "EDDF": (50.0379, 8.5622),
    "AMS": (52.3086, 4.7639), "EHAM": (52.3086, 4.7639),
    "FCO": (41.8003, 12.2389), "LIRF": (41.8003, 12.2389),
    "MAD": (40.4936, -3.5668), "LEMD": (40.4936, -3.5668),
    "BCN": (41.2971, 2.0785), "LEBL": (41.2971, 2.0785),
    "IST": (41.2753, 28.7519), "LTFM": (41.2753, 28.7519),
    "ATH": (37.9364, 23.9445), "LGAV": (37.9364, 23.9445),
    "MUC": (48.3538, 11.7861), "EDDM": (48.3538, 11.7861),
    "ZRH": (47.4647, 8.5492), "LSZH": (47.4647, 8.5492),
    "VIE": (48.1103, 16.5697), "LOWW": (48.1103, 16.5697),
    "LIS": (38.7756, -9.1354), "LPPT": (38.7756, -9.1354),
    "PRG": (50.1008, 14.2600), "LKPR": (50.1008, 14.2600),
    "WAW": (52.1657, 20.9671), "EPWA": (52.1657, 20.9671),
    
    # Asia
    "BKK": (13.6900, 100.7501), "VTBS": (13.6900, 100.7501),
    "SIN": (1.3644, 103.9915), "WSSS": (1.3644, 103.9915),
    "HKG": (22.3080, 113.9185), "VHHH": (22.3080, 113.9185),
    "PVG": (31.1434, 121.8052), "ZSPD": (31.1434, 121.8052),
    "NRT": (35.7720, 140.3929), "RJAA": (35.7720, 140.3929),
    "ICN": (37.4602, 126.4407), "RKSI": (37.4602, 126.4407),
    "DEL": (28.5562, 77.1000), "VIDP": (28.5562, 77.1000),
    "BOM": (19.0896, 72.8656), "VABB": (19.0896, 72.8656),
    
    # Americas
    "JFK": (40.6413, -73.7781), "KJFK": (40.6413, -73.7781),
    "LAX": (33.9425, -118.4081), "KLAX": (33.9425, -118.4081),
    "MIA": (25.7959, -80.2870), "KMIA": (25.7959, -80.2870),
    "ORD": (41.9742, -87.9073), "KORD": (41.9742, -87.9073),
    "ATL": (33.6407, -84.4277), "KATL": (33.6407, -84.4277),
    "DFW": (32.8998, -97.0403), "KDFW": (32.8998, -97.0403),
    "EWR": (40.6895, -74.1745), "KEWR": (40.6895, -74.1745),
    "YYZ": (43.6777, -79.6248), "CYYZ": (43.6777, -79.6248),
    "GRU": (-23.4356, -46.4731), "SBGR": (-23.4356, -46.4731),
}


def get_airport_coords(code: str) -> Optional[Tuple[float, float]]:
    """Look up airport coordinates by IATA or ICAO code."""
    code_upper = code.upper().strip()
    if code_upper in AIRPORT_COORDINATES:
        return AIRPORT_COORDINATES[code_upper]
    # Try without first letter (some codes have prefix)
    if len(code_upper) > 3 and code_upper[1:] in AIRPORT_COORDINATES:
        return AIRPORT_COORDINATES[code_upper[1:]]
    return None


def interpolate_altitude(alt1: float, alt2: float, fraction: float) -> float:
    """Linear interpolation between two altitudes."""
    return alt1 + (alt2 - alt1) * fraction


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


def configure(project_root: Path = None):
    """Configure the router with dependencies."""
    global PROJECT_ROOT, CONFIGURED_AIRPORTS
    
    if project_root:
        PROJECT_ROOT = project_root
        CONFIGURED_AIRPORTS = _load_airports_config(project_root)
        logger.info(f"Loaded {len(CONFIGURED_AIRPORTS)} airports for trajectory planner")


# ============================================================
# Data Models
# ============================================================

class TrajectoryWaypoint(BaseModel):
    """A waypoint in the trajectory."""
    lat: float
    lon: float
    alt: float  # Altitude in feet
    id: str
    speed_kts: Optional[float] = 250  # Ground speed in knots


class TrajectoryPlanRequest(BaseModel):
    """Request model for trajectory planning."""
    waypoints: List[TrajectoryWaypoint]
    start_datetime: str  # ISO format: "2025-12-30T14:00:00"
    horizontal_threshold_nm: float = 5.0  # Conflict threshold in NM
    vertical_threshold_ft: float = 1000.0  # Conflict threshold in feet
    interpolation_interval_sec: int = 10  # Interpolate every N seconds


@dataclass
class InterpolatedPoint:
    """A point along the interpolated trajectory."""
    lat: float
    lon: float
    alt: float
    timestamp: datetime
    time_offset_sec: int
    segment_index: int  # Which segment (waypoint pair) this belongs to
    bearing: float
    speed_kts: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "alt": round(self.alt, 0),
            "timestamp": self.timestamp.isoformat(),
            "time_offset_sec": self.time_offset_sec,
            "segment_index": self.segment_index,
            "bearing": round(self.bearing, 1),
            "speed_kts": round(self.speed_kts, 1),
        }


@dataclass
class AirportTraffic:
    """Traffic at an airport (departures and arrivals)."""
    airport_code: str
    airport_name: str
    lat: float
    lon: float
    departures: List[Dict[str, Any]] = field(default_factory=list)
    arrivals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FlightPathPoint:
    """A point along a departure/arrival flight path."""
    lat: float
    lon: float
    alt: float  # feet
    time_offset_min: float  # minutes from scheduled time (negative = before)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "alt": round(self.alt, 0),
            "time_offset_min": round(self.time_offset_min, 1),
        }


@dataclass
class FlightPath:
    """Estimated flight path for a departure or arrival."""
    flight_number: str
    airline: str
    flight_type: str  # "departure" or "arrival"
    airport_code: str
    airport_lat: float
    airport_lon: float
    destination_or_origin: str  # destination for departures, origin for arrivals
    scheduled_time: str
    bearing: float  # direction of flight
    path_points: List[FlightPathPoint] = field(default_factory=list)
    aircraft_type: str = ""
    is_time_relevant: bool = False  # True if flight overlaps with trajectory time window
    min_distance_nm: float = -1  # Minimum distance to trajectory (-1 = not calculated)
    min_vertical_ft: float = -1  # Minimum vertical separation (-1 = not calculated)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "flight_number": self.flight_number,
            "airline": self.airline,
            "flight_type": self.flight_type,
            "airport_code": self.airport_code,
            "airport_lat": self.airport_lat,
            "airport_lon": self.airport_lon,
            "destination_or_origin": self.destination_or_origin,
            "scheduled_time": self.scheduled_time,
            "bearing": round(self.bearing, 1),
            "path_points": [p.to_dict() for p in self.path_points],
            "aircraft_type": self.aircraft_type,
            "is_time_relevant": self.is_time_relevant,
            "min_distance_nm": round(self.min_distance_nm, 1) if self.min_distance_nm >= 0 else None,
            "min_vertical_ft": round(self.min_vertical_ft, 0) if self.min_vertical_ft >= 0 else None,
        }


@dataclass
class TrafficConflict:
    """A detected conflict between our trajectory and airport traffic."""
    flight_number: str
    airline: str
    flight_type: str  # "departure" or "arrival"
    airport: str
    scheduled_time: str
    our_position: Dict[str, Any]  # lat, lon, alt, timestamp
    estimated_traffic_position: Dict[str, Any]  # lat, lon, alt
    horizontal_distance_nm: float
    vertical_distance_ft: float
    severity: str  # "critical", "warning", "info"
    time_to_conflict_sec: int
    flight_path: Optional[FlightPath] = None  # The full flight path for visualization
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "flight_number": self.flight_number,
            "airline": self.airline,
            "flight_type": self.flight_type,
            "airport": self.airport,
            "scheduled_time": self.scheduled_time,
            "our_position": self.our_position,
            "estimated_traffic_position": self.estimated_traffic_position,
            "horizontal_distance_nm": round(self.horizontal_distance_nm, 2),
            "vertical_distance_ft": round(self.vertical_distance_ft, 0),
            "severity": self.severity,
            "time_to_conflict_sec": self.time_to_conflict_sec,
            "flight_path": self.flight_path.to_dict() if self.flight_path else None,
        }


# ============================================================
# Trajectory Interpolation
# ============================================================

def interpolate_trajectory(
    waypoints: List[TrajectoryWaypoint],
    start_time: datetime,
    interval_sec: int = 10
) -> List[InterpolatedPoint]:
    """
    Interpolate a trajectory between waypoints at regular time intervals.
    
    For each segment (pair of consecutive waypoints):
    1. Calculate distance and bearing
    2. Use speed to determine travel time
    3. Generate points every interval_sec seconds
    4. Interpolate altitude linearly
    """
    if len(waypoints) < 2:
        return []
    
    points: List[InterpolatedPoint] = []
    current_time = start_time
    total_offset_sec = 0
    
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        
        # Calculate segment properties
        distance_nm = haversine_distance_nm(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
        bearing = calculate_bearing(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
        
        # Average speed for this segment
        avg_speed_kts = (wp1.speed_kts + wp2.speed_kts) / 2 if wp2.speed_kts else wp1.speed_kts
        if avg_speed_kts <= 0:
            avg_speed_kts = 250  # Default speed
        
        # Calculate travel time for this segment
        travel_time_hours = distance_nm / avg_speed_kts
        travel_time_sec = travel_time_hours * 3600
        
        # Number of interpolation points for this segment
        num_points = max(1, int(travel_time_sec / interval_sec))
        
        for j in range(num_points):
            fraction = j / num_points
            
            # Calculate distance traveled
            distance_traveled_nm = distance_nm * fraction
            
            # Calculate position
            lat, lon = destination_point(wp1.lat, wp1.lon, bearing, distance_traveled_nm)
            
            # Interpolate altitude
            alt = interpolate_altitude(wp1.alt, wp2.alt, fraction)
            
            # Calculate timestamp
            point_offset_sec = int(fraction * travel_time_sec)
            point_time = current_time + timedelta(seconds=point_offset_sec)
            
            points.append(InterpolatedPoint(
                lat=lat,
                lon=lon,
                alt=alt,
                timestamp=point_time,
                time_offset_sec=total_offset_sec + point_offset_sec,
                segment_index=i,
                bearing=bearing,
                speed_kts=avg_speed_kts,
            ))
        
        # Update time for next segment
        total_offset_sec += int(travel_time_sec)
        current_time = current_time + timedelta(seconds=int(travel_time_sec))
    
    # Add final waypoint
    final_wp = waypoints[-1]
    points.append(InterpolatedPoint(
        lat=final_wp.lat,
        lon=final_wp.lon,
        alt=final_wp.alt,
        timestamp=current_time,
        time_offset_sec=total_offset_sec,
        segment_index=len(waypoints) - 2,
        bearing=points[-1].bearing if points else 0,
        speed_kts=final_wp.speed_kts or 250,
    ))
    
    return points


# ============================================================
# Airport Traffic Fetching (using FlightRadar24 API)
# ============================================================

def fetch_airport_traffic(
    airports: List[Dict[str, Any]],
    date: datetime
) -> List[AirportTraffic]:
    """
    Fetch departures and arrivals for airports using FlightRadar24 API.
    Returns scheduled flights from the FR24 website.
    """
    global fr24_api
    
    if not FR24_API_AVAILABLE or fr24_api is None:
        logger.warning("FlightRadar24 API not available")
        return []
    
    traffic_list: List[AirportTraffic] = []
    
    for airport in airports:
        code = airport.get("code", "")
        name = airport.get("name", code)
        lat = airport.get("lat", 0)
        lon = airport.get("lon", 0)
        
        # FR24 uses ICAO codes (4 letters) or IATA codes (3 letters)
        # Try ICAO first, then IATA
        airport_code = code
        
        traffic = AirportTraffic(
            airport_code=code,
            airport_name=name,
            lat=lat,
            lon=lon,
        )
        
        try:
            # Get airport details from FR24
            logger.info(f"Fetching FR24 data for airport: {airport_code}")
            airport_details = fr24_api.get_airport_details(airport_code)
            
            if not airport_details:
                logger.warning(f"No data returned for airport {airport_code}")
                continue
            
            # Navigate to schedule data
            plugin_data = airport_details.get('airport', {}).get('pluginData', {})
            schedule = plugin_data.get('schedule', {})
            
            # Parse departures
            departures_data = schedule.get('departures', {}).get('data', [])
            logger.info(f"Found {len(departures_data)} departures for {airport_code}")
            
            for flight_data in departures_data:
                try:
                    flight_info = flight_data.get('flight', {})
                    
                    # Get flight number
                    identification = flight_info.get('identification', {})
                    flight_number = identification.get('number', {}).get('default', 'Unknown')
                    callsign = identification.get('callsign', '')
                    
                    # Get airline
                    airline_info = flight_info.get('airline', {})
                    airline_name = airline_info.get('name', 'Unknown')
                    airline_code = airline_info.get('code', {}).get('iata', '')
                    
                    # Get destination
                    destination_info = flight_info.get('airport', {}).get('destination', {})
                    destination_code = destination_info.get('code', {}).get('iata', '???')
                    destination_name = destination_info.get('name', '')
                    
                    # Get scheduled time
                    time_info = flight_info.get('time', {}).get('scheduled', {})
                    departure_timestamp = time_info.get('departure')
                    scheduled_time = ""
                    if departure_timestamp:
                        try:
                            scheduled_time = datetime.fromtimestamp(departure_timestamp).isoformat()
                        except:
                            scheduled_time = str(departure_timestamp)
                    
                    # Get aircraft info
                    aircraft_info = flight_info.get('aircraft', {})
                    aircraft_model = aircraft_info.get('model', {}).get('text', '')
                    aircraft_code = aircraft_info.get('model', {}).get('code', '')
                    
                    # Get status
                    status_info = flight_info.get('status', {})
                    status_text = status_info.get('text', '')
                    
                    traffic.departures.append({
                        "flight_number": flight_number or callsign,
                        "airline": airline_name,
                        "airline_code": airline_code,
                        "destination": destination_code,
                        "destination_name": destination_name,
                        "scheduled_time": scheduled_time,
                        "aircraft": aircraft_model or aircraft_code,
                        "status": status_text,
                    })
                except Exception as e:
                    logger.debug(f"Error parsing departure flight: {e}")
                    continue
            
            # Parse arrivals
            arrivals_data = schedule.get('arrivals', {}).get('data', [])
            logger.info(f"Found {len(arrivals_data)} arrivals for {airport_code}")
            
            for flight_data in arrivals_data:
                try:
                    flight_info = flight_data.get('flight', {})
                    
                    # Get flight number
                    identification = flight_info.get('identification', {})
                    flight_number = identification.get('number', {}).get('default', 'Unknown')
                    callsign = identification.get('callsign', '')
                    
                    # Get airline
                    airline_info = flight_info.get('airline', {})
                    airline_name = airline_info.get('name', 'Unknown')
                    airline_code = airline_info.get('code', {}).get('iata', '')
                    
                    # Get origin
                    origin_info = flight_info.get('airport', {}).get('origin', {})
                    origin_code = origin_info.get('code', {}).get('iata', '???')
                    origin_name = origin_info.get('name', '')
                    
                    # Get scheduled time
                    time_info = flight_info.get('time', {}).get('scheduled', {})
                    arrival_timestamp = time_info.get('arrival')
                    scheduled_time = ""
                    if arrival_timestamp:
                        try:
                            scheduled_time = datetime.fromtimestamp(arrival_timestamp).isoformat()
                        except:
                            scheduled_time = str(arrival_timestamp)
                    
                    # Get aircraft info
                    aircraft_info = flight_info.get('aircraft', {})
                    aircraft_model = aircraft_info.get('model', {}).get('text', '')
                    aircraft_code = aircraft_info.get('model', {}).get('code', '')
                    
                    # Get status
                    status_info = flight_info.get('status', {})
                    status_text = status_info.get('text', '')
                    
                    traffic.arrivals.append({
                        "flight_number": flight_number or callsign,
                        "airline": airline_name,
                        "airline_code": airline_code,
                        "origin": origin_code,
                        "origin_name": origin_name,
                        "scheduled_time": scheduled_time,
                        "aircraft": aircraft_model or aircraft_code,
                        "status": status_text,
                    })
                except Exception as e:
                    logger.debug(f"Error parsing arrival flight: {e}")
                    continue
            
            # Add to list if we got any data
            if traffic.arrivals or traffic.departures:
                traffic_list.append(traffic)
                logger.info(f"Airport {airport_code}: {len(traffic.departures)} departures, {len(traffic.arrivals)} arrivals")
            else:
                # Still add the airport even with no flights (for display purposes)
                traffic_list.append(traffic)
                
        except Exception as e:
            logger.warning(f"Failed to fetch FR24 data for {airport_code}: {e}")
            import traceback
            traceback.print_exc()
            # Still add the airport to the list
            traffic_list.append(traffic)
    
    return traffic_list


def estimate_traffic_position(
    airport: AirportTraffic,
    flight: Dict[str, Any],
    flight_type: str,  # "departure" or "arrival"
    reference_time: datetime
) -> Optional[Dict[str, Any]]:
    """
    Estimate the position of a scheduled flight at a given time.
    
    For departures: Aircraft climbs away from airport
    For arrivals: Aircraft descends toward airport
    
    Assumptions:
    - Average climb/descent rate: 2000 ft/min
    - Average speed: 250 kts during climb/descent
    - Typical approach starts 50 NM from airport at 10,000 ft
    - Typical departure reaches 50 NM at 10,000 ft
    """
    scheduled_time_str = flight.get("scheduled_time", "")
    if not scheduled_time_str:
        return None
    
    try:
        # Parse scheduled time
        if 'T' in scheduled_time_str:
            scheduled_time = datetime.fromisoformat(scheduled_time_str.replace('Z', '+00:00'))
        else:
            scheduled_time = datetime.strptime(scheduled_time_str, "%Y-%m-%d %H:%M")
        
        # Make both times timezone-naive for comparison
        if scheduled_time.tzinfo:
            scheduled_time = scheduled_time.replace(tzinfo=None)
        if reference_time.tzinfo:
            reference_time = reference_time.replace(tzinfo=None)
        
        # Time difference in minutes
        time_diff_min = (reference_time - scheduled_time).total_seconds() / 60
        
        # Aircraft parameters
        climb_rate_fpm = 2000  # ft/min
        descent_rate_fpm = 1500  # ft/min
        cruise_speed_kts = 250  # kts
        
        if flight_type == "departure":
            # For departures, time_diff > 0 means after departure
            if time_diff_min < -30 or time_diff_min > 60:
                return None  # Too far from scheduled time
            
            if time_diff_min < 0:
                # Before departure - aircraft at airport
                return {
                    "lat": airport.lat,
                    "lon": airport.lon,
                    "alt": 0,  # Ground level
                    "phase": "ground"
                }
            
            # After departure - climbing away
            alt = min(35000, time_diff_min * climb_rate_fpm)  # Altitude
            distance_nm = (time_diff_min / 60) * cruise_speed_kts  # Distance from airport
            
            # Get bearing based on destination airport coordinates
            bearing = 90  # Default eastward
            dest = flight.get("destination", "")
            if dest:
                dest_coords = get_airport_coords(dest)
                if dest_coords:
                    # Calculate bearing from airport to destination
                    bearing = calculate_bearing(airport.lat, airport.lon, dest_coords[0], dest_coords[1])
                else:
                    # Fallback: use hash for unknown destinations
                    bearing = (hash(dest) % 360)
            
            lat, lon = destination_point(airport.lat, airport.lon, bearing, distance_nm)
            
            return {
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "phase": "climb"
            }
        
        else:  # arrival
            # For arrivals, time_diff < 0 means before arrival
            if time_diff_min < -60 or time_diff_min > 30:
                return None  # Too far from scheduled time
            
            if time_diff_min > 0:
                # After arrival - aircraft at airport
                return {
                    "lat": airport.lat,
                    "lon": airport.lon,
                    "alt": 0,
                    "phase": "ground"
                }
            
            # Before arrival - descending toward airport
            minutes_to_arrival = abs(time_diff_min)
            
            # Estimate position based on time to arrival
            # At 30+ min out: cruise altitude, far from airport
            # At 10 min out: ~10,000 ft, ~40 NM from airport
            # At 5 min out: ~3,000 ft, ~15 NM from airport
            
            if minutes_to_arrival > 30:
                alt = 35000
                distance_nm = minutes_to_arrival * (cruise_speed_kts / 60)
            elif minutes_to_arrival > 10:
                alt = 10000 + (minutes_to_arrival - 10) * 1250  # Linear descent
                distance_nm = 40 + (minutes_to_arrival - 10) * 3
            else:
                alt = minutes_to_arrival * 1000  # Final approach
                distance_nm = minutes_to_arrival * 4
            
            # Get bearing based on origin airport coordinates (aircraft approaches FROM origin direction)
            bearing = 270  # Default westward (approaching from west)
            origin = flight.get("origin", "")
            if origin:
                origin_coords = get_airport_coords(origin)
                if origin_coords:
                    # Aircraft approaches FROM the origin direction
                    bearing = calculate_bearing(airport.lat, airport.lon, origin_coords[0], origin_coords[1])
                else:
                    # Fallback: use hash for unknown origins
                    bearing = (hash(origin) % 360 + 180) % 360
            
            lat, lon = destination_point(airport.lat, airport.lon, bearing, distance_nm)
            
            return {
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "phase": "descent" if minutes_to_arrival > 5 else "approach"
            }
    
    except Exception as e:
        logger.warning(f"Error estimating traffic position: {e}")
        return None


def generate_flight_path(
    airport: AirportTraffic,
    flight: Dict[str, Any],
    flight_type: str,  # "departure" or "arrival"
) -> Optional[FlightPath]:
    """
    Generate a flight path (short trajectory line) for a departure or arrival.
    
    For departures: Shows climb-out path from 0 to 10,000 ft (first ~5 min)
    For arrivals: Shows approach path from 10,000 ft to 0 (last ~10 min)
    
    Returns a FlightPath with multiple points for visualization.
    """
    scheduled_time_str = flight.get("scheduled_time", "")
    if not scheduled_time_str:
        return None
    
    # Get destination/origin for bearing calculation
    if flight_type == "departure":
        dest_or_origin = flight.get("destination", "")
    else:
        dest_or_origin = flight.get("origin", "")
    
    # Calculate bearing
    bearing = 90.0  # Default
    if dest_or_origin:
        coords = get_airport_coords(dest_or_origin)
        if coords:
            if flight_type == "departure":
                # Departures fly TOWARD destination
                bearing = calculate_bearing(airport.lat, airport.lon, coords[0], coords[1])
            else:
                # Arrivals approach FROM origin direction
                bearing = calculate_bearing(airport.lat, airport.lon, coords[0], coords[1])
        else:
            # Fallback
            bearing = (hash(dest_or_origin) % 360)
    
    # Create flight path
    path = FlightPath(
        flight_number=flight.get("flight_number", "Unknown"),
        airline=flight.get("airline", "Unknown"),
        flight_type=flight_type,
        airport_code=airport.airport_code,
        airport_lat=airport.lat,
        airport_lon=airport.lon,
        destination_or_origin=dest_or_origin,
        scheduled_time=scheduled_time_str,
        bearing=bearing,
        aircraft_type=flight.get("aircraft", ""),
    )
    
    # Generate path points
    climb_rate_fpm = 2000  # ft/min
    descent_rate_fpm = 1500  # ft/min
    speed_kts = 200  # Average speed during climb/descent
    
    if flight_type == "departure":
        # Generate departure path: 0 to 10,000 ft over ~5 minutes
        # Points at 0, 1, 2, 3, 4, 5 minutes after departure
        for min_offset in range(0, 11, 1):  # 0 to 10 minutes
            alt = min(10000, min_offset * climb_rate_fpm)
            distance_nm = (min_offset / 60) * speed_kts
            lat, lon = destination_point(airport.lat, airport.lon, bearing, distance_nm)
            
            path.path_points.append(FlightPathPoint(
                lat=lat,
                lon=lon,
                alt=alt,
                time_offset_min=min_offset,
            ))
    else:
        # Generate arrival path: 10,000 ft to 0 over ~10 minutes before landing
        # Points at -10, -8, -6, -4, -2, 0 minutes before arrival
        for min_offset in range(-15, 1, 1):  # -15 to 0 minutes
            minutes_to_landing = abs(min_offset)
            
            # Calculate altitude (linear descent to 0)
            if minutes_to_landing > 10:
                alt = 10000
            else:
                alt = minutes_to_landing * 1000  # 1000 ft per minute
            
            # Calculate distance from airport
            distance_nm = minutes_to_landing * 3  # ~3 NM per minute on approach
            
            lat, lon = destination_point(airport.lat, airport.lon, bearing, distance_nm)
            
            path.path_points.append(FlightPathPoint(
                lat=lat,
                lon=lon,
                alt=alt,
                time_offset_min=min_offset,
            ))
    
    return path


def generate_all_flight_paths(
    traffic_list: List[AirportTraffic],
    trajectory: Optional[List[InterpolatedPoint]] = None,
    horizontal_threshold_nm: float = 10.0,
    vertical_threshold_ft: float = 1000.0,
) -> List[FlightPath]:
    """
    Generate flight paths for all departures and arrivals.
    
    If trajectory is provided, also calculates:
    - is_time_relevant: True if flight overlaps with trajectory time window
    - min_distance_nm: Minimum horizontal distance to trajectory
    - min_vertical_ft: Minimum vertical separation
    """
    paths: List[FlightPath] = []
    
    # Get trajectory time range for relevance checking
    traj_start = trajectory[0].timestamp if trajectory else None
    traj_end = trajectory[-1].timestamp if trajectory else None
    
    # Extend window by 30 min before and 60 min after for departures/arrivals
    time_window_before = timedelta(minutes=30)
    time_window_after = timedelta(minutes=60)
    
    for airport_traffic in traffic_list:
        # Generate paths for departures
        for flight in airport_traffic.departures:
            path = generate_flight_path(airport_traffic, flight, "departure")
            if path:
                # Check time relevance
                if trajectory and traj_start and traj_end:
                    path = _calculate_path_relevance(
                        path, trajectory, traj_start, traj_end,
                        time_window_before, time_window_after,
                        horizontal_threshold_nm, vertical_threshold_ft
                    )
                paths.append(path)
        
        # Generate paths for arrivals
        for flight in airport_traffic.arrivals:
            path = generate_flight_path(airport_traffic, flight, "arrival")
            if path:
                # Check time relevance
                if trajectory and traj_start and traj_end:
                    path = _calculate_path_relevance(
                        path, trajectory, traj_start, traj_end,
                        time_window_before, time_window_after,
                        horizontal_threshold_nm, vertical_threshold_ft
                    )
                paths.append(path)
    
    return paths


def _calculate_path_relevance(
    path: FlightPath,
    trajectory: List[InterpolatedPoint],
    traj_start: datetime,
    traj_end: datetime,
    time_window_before: timedelta,
    time_window_after: timedelta,
    horizontal_threshold_nm: float,
    vertical_threshold_ft: float,
) -> FlightPath:
    """Calculate if a flight path is time-relevant and its minimum distances."""
    try:
        # Parse scheduled time
        scheduled_str = path.scheduled_time
        if 'T' in scheduled_str:
            scheduled = datetime.fromisoformat(scheduled_str.replace('Z', '+00:00'))
        else:
            scheduled = datetime.strptime(scheduled_str, "%Y-%m-%d %H:%M")
        
        if scheduled.tzinfo:
            scheduled = scheduled.replace(tzinfo=None)
        
        # Check if flight is within the time window
        # For departures: flight departs, then climbs (check scheduled to scheduled+60min)
        # For arrivals: flight descends before landing (check scheduled-60min to scheduled)
        if path.flight_type == "departure":
            flight_start = scheduled
            flight_end = scheduled + time_window_after
        else:  # arrival
            flight_start = scheduled - time_window_after
            flight_end = scheduled
        
        # Check if there's any time overlap
        is_time_relevant = not (flight_end < traj_start or flight_start > traj_end)
        path.is_time_relevant = is_time_relevant
        
        # If time-relevant, calculate minimum distances
        if is_time_relevant:
            min_h_dist = float('inf')
            min_v_dist = float('inf')
            
            # Sample trajectory points
            for point in trajectory[::6]:  # Every minute
                # Find corresponding flight path point at this time
                time_diff_min = (point.timestamp - scheduled).total_seconds() / 60
                
                # Find closest path point by time
                for fp_point in path.path_points:
                    if abs(fp_point.time_offset_min - time_diff_min) < 1.5:  # Within 1.5 minutes
                        h_dist = haversine_distance_nm(
                            point.lat, point.lon,
                            fp_point.lat, fp_point.lon
                        )
                        v_dist = abs(point.alt - fp_point.alt)
                        
                        if h_dist < min_h_dist:
                            min_h_dist = h_dist
                        if v_dist < min_v_dist:
                            min_v_dist = v_dist
            
            if min_h_dist < float('inf'):
                path.min_distance_nm = min_h_dist
            if min_v_dist < float('inf'):
                path.min_vertical_ft = min_v_dist
    
    except Exception as e:
        logger.warning(f"Error calculating path relevance for {path.flight_number}: {e}")
    
    return path


# ============================================================
# Conflict Detection
# ============================================================

def detect_conflicts(
    trajectory: List[InterpolatedPoint],
    traffic_list: List[AirportTraffic],
    horizontal_threshold_nm: float = 5.0,
    vertical_threshold_ft: float = 1000.0,
) -> List[TrafficConflict]:
    """
    Detect conflicts between our trajectory and airport traffic.
    
    For each point on our trajectory, estimate positions of all
    scheduled flights and check for proximity violations.
    """
    conflicts: List[TrafficConflict] = []
    checked_flights = set()  # Avoid duplicate conflicts for same flight
    
    # Debug: track statistics
    total_checks = 0
    positions_estimated = 0
    within_horizontal = 0
    
    # Sample every 6th point (every minute) for efficiency but log first few
    sample_points = trajectory[::6]  # Every minute instead of every 10 sec
    
    total_flights = sum(len(t.departures) + len(t.arrivals) for t in traffic_list)
    logger.info(f"Checking {len(sample_points)} trajectory points against {total_flights} flights")
    
    # Get trajectory time range
    traj_start = trajectory[0].timestamp if trajectory else None
    traj_end = trajectory[-1].timestamp if trajectory else None
    if traj_start and traj_end:
        logger.info(f"Trajectory time range: {traj_start.isoformat()} to {traj_end.isoformat()}")
    
    # Log first few flight times to debug
    sample_count = 0
    for airport_traffic in traffic_list:
        for flight in airport_traffic.departures[:3]:
            scheduled = flight.get("scheduled_time", "N/A")
            logger.info(f"Sample DEP {airport_traffic.airport_code}: {flight.get('flight_number')} scheduled at {scheduled}")
            sample_count += 1
            if sample_count >= 5:
                break
        if sample_count >= 5:
            break
    
    # Track why flights are skipped
    flights_in_time_window = 0
    flights_outside_time_window = 0
    
    for point_idx, point in enumerate(sample_points):
        for airport_traffic in traffic_list:
            # Check departures
            for flight in airport_traffic.departures:
                flight_key = f"{flight.get('flight_number')}_{flight.get('scheduled_time')}"
                if flight_key in checked_flights:
                    continue
                
                total_checks += 1
                traffic_pos = estimate_traffic_position(
                    airport_traffic, flight, "departure", point.timestamp
                )
                
                if traffic_pos is None:
                    flights_outside_time_window += 1
                elif traffic_pos.get("phase") == "ground":
                    pass  # On ground, not a conflict
                else:
                    flights_in_time_window += 1
                
                if traffic_pos and traffic_pos.get("phase") != "ground":
                    positions_estimated += 1
                    h_dist = haversine_distance_nm(
                        point.lat, point.lon,
                        traffic_pos["lat"], traffic_pos["lon"]
                    )
                    v_dist = abs(point.alt - traffic_pos["alt"])
                    
                    # Debug: log first few comparisons
                    if point_idx < 3 and positions_estimated <= 10:
                        logger.info(f"DEP {flight.get('flight_number')}: our_alt={point.alt}ft, traffic_alt={traffic_pos['alt']}ft, h_dist={h_dist:.1f}nm, v_dist={v_dist:.0f}ft")
                    
                    if h_dist < horizontal_threshold_nm * 3:  # Check wider area first
                        within_horizontal += 1
                        severity = _calculate_severity(
                            h_dist, v_dist,
                            horizontal_threshold_nm, vertical_threshold_ft
                        )
                        
                        if severity:
                            logger.info(f"CONFLICT FOUND: {flight.get('flight_number')} - {severity} - h={h_dist:.1f}nm, v={v_dist:.0f}ft")
                            checked_flights.add(flight_key)
                            conflicts.append(TrafficConflict(
                                flight_number=flight.get("flight_number", "Unknown"),
                                airline=flight.get("airline", "Unknown"),
                                flight_type="departure",
                                airport=airport_traffic.airport_code,
                                scheduled_time=flight.get("scheduled_time", ""),
                                our_position={
                                    "lat": point.lat,
                                    "lon": point.lon,
                                    "alt": point.alt,
                                    "timestamp": point.timestamp.isoformat(),
                                },
                                estimated_traffic_position=traffic_pos,
                                horizontal_distance_nm=h_dist,
                                vertical_distance_ft=v_dist,
                                severity=severity,
                                time_to_conflict_sec=point.time_offset_sec,
                            ))
            
            # Check arrivals
            for flight in airport_traffic.arrivals:
                flight_key = f"{flight.get('flight_number')}_{flight.get('scheduled_time')}"
                if flight_key in checked_flights:
                    continue
                
                total_checks += 1
                traffic_pos = estimate_traffic_position(
                    airport_traffic, flight, "arrival", point.timestamp
                )
                
                if traffic_pos is None:
                    flights_outside_time_window += 1
                elif traffic_pos.get("phase") == "ground":
                    pass  # On ground, not a conflict
                else:
                    flights_in_time_window += 1
                
                if traffic_pos and traffic_pos.get("phase") != "ground":
                    positions_estimated += 1
                    h_dist = haversine_distance_nm(
                        point.lat, point.lon,
                        traffic_pos["lat"], traffic_pos["lon"]
                    )
                    v_dist = abs(point.alt - traffic_pos["alt"])
                    
                    # Debug: log first few comparisons
                    if point_idx < 3 and positions_estimated <= 20:
                        logger.info(f"ARR {flight.get('flight_number')}: our_alt={point.alt}ft, traffic_alt={traffic_pos['alt']}ft, h_dist={h_dist:.1f}nm, v_dist={v_dist:.0f}ft")
                    
                    if h_dist < horizontal_threshold_nm * 3:
                        severity = _calculate_severity(
                            h_dist, v_dist,
                            horizontal_threshold_nm, vertical_threshold_ft
                        )
                        
                        if severity:
                            logger.info(f"CONFLICT FOUND: {flight.get('flight_number')} - {severity} - h={h_dist:.1f}nm, v={v_dist:.0f}ft")
                            checked_flights.add(flight_key)
                            conflicts.append(TrafficConflict(
                                flight_number=flight.get("flight_number", "Unknown"),
                                airline=flight.get("airline", "Unknown"),
                                flight_type="arrival",
                                airport=airport_traffic.airport_code,
                                scheduled_time=flight.get("scheduled_time", ""),
                                our_position={
                                    "lat": point.lat,
                                    "lon": point.lon,
                                    "alt": point.alt,
                                    "timestamp": point.timestamp.isoformat(),
                                },
                                estimated_traffic_position=traffic_pos,
                                horizontal_distance_nm=h_dist,
                                vertical_distance_ft=v_dist,
                                severity=severity,
                                time_to_conflict_sec=point.time_offset_sec,
                            ))
    
    # Log summary
    logger.info(f"Conflict detection summary:")
    logger.info(f"  - Total checks: {total_checks}")
    logger.info(f"  - Flights in time window: {flights_in_time_window}")
    logger.info(f"  - Flights outside time window: {flights_outside_time_window}")
    logger.info(f"  - Positions estimated: {positions_estimated}")
    logger.info(f"  - Within horizontal range: {within_horizontal}")
    logger.info(f"  - Conflicts found: {len(conflicts)}")
    
    # Sort by severity and time
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    conflicts.sort(key=lambda c: (severity_order.get(c.severity, 3), c.time_to_conflict_sec))
    
    return conflicts


def _calculate_severity(
    h_dist: float,
    v_dist: float,
    h_threshold: float,
    v_threshold: float
) -> Optional[str]:
    """Calculate conflict severity based on distances."""
    # Critical: Within separation minimums
    if h_dist < h_threshold and v_dist < v_threshold:
        return "critical"
    
    # Warning: Within 2x separation
    if h_dist < h_threshold * 2 and v_dist < v_threshold * 2:
        return "warning"
    
    # Info: Within 3x separation (awareness)
    if h_dist < h_threshold * 3 and v_dist < v_threshold * 3:
        return "info"
    
    return None


# ============================================================
# API Endpoints
# ============================================================

@router.get("/api/trajectory/airports")
def get_trajectory_airports():
    """Get list of configured airports for trajectory planning."""
    return {
        "airports": CONFIGURED_AIRPORTS,
        "total": len(CONFIGURED_AIRPORTS),
    }


@router.post("/api/trajectory/plan")
async def plan_trajectory(request: TrajectoryPlanRequest):
    """
    Plan a trajectory and detect conflicts with airport traffic.
    
    This endpoint:
    1. Interpolates the trajectory at regular intervals (default 10 sec)
    2. Fetches scheduled departures/arrivals from nearby airports
    3. Estimates traffic positions at each trajectory point
    4. Detects conflicts based on horizontal and vertical separation
    
    Returns:
        - Interpolated trajectory points
        - List of airports checked
        - List of detected conflicts
        - Summary statistics
    """
    try:
        # Parse start datetime
        try:
            start_dt = datetime.fromisoformat(request.start_datetime.replace('Z', '+00:00'))
            if start_dt.tzinfo:
                start_dt = start_dt.replace(tzinfo=None)
        except ValueError:
            start_dt = datetime.strptime(request.start_datetime, "%Y-%m-%dT%H:%M:%S")
        
        # Validate waypoints
        if len(request.waypoints) < 2:
            raise HTTPException(status_code=400, detail="At least 2 waypoints required")
        
        # Step 1: Interpolate trajectory
        trajectory = interpolate_trajectory(
            request.waypoints,
            start_dt,
            request.interpolation_interval_sec
        )
        
        if not trajectory:
            raise HTTPException(status_code=400, detail="Failed to interpolate trajectory")
        
        # Step 2: Find airports near the trajectory
        AIRPORT_SEARCH_RADIUS_NM = 60
        nearby_airports = []
        
        for airport in CONFIGURED_AIRPORTS:
            for point in trajectory[::10]:  # Check every 10th point for efficiency
                dist = haversine_distance_nm(
                    point.lat, point.lon,
                    airport.get("lat", 0), airport.get("lon", 0)
                )
                if dist <= AIRPORT_SEARCH_RADIUS_NM:
                    if airport not in nearby_airports:
                        nearby_airports.append(airport)
                    break
        
        logger.info(f"Found {len(nearby_airports)} airports near trajectory")
        
        # Step 3: Fetch airport traffic
        traffic_list = fetch_airport_traffic(nearby_airports, start_dt)
        
        total_flights = sum(
            len(t.departures) + len(t.arrivals) for t in traffic_list
        )
        logger.info(f"Fetched {total_flights} scheduled flights from {len(traffic_list)} airports")
        
        # Step 4: Generate flight paths for all traffic (for visualization)
        flight_paths = generate_all_flight_paths(
            traffic_list,
            trajectory,
            request.horizontal_threshold_nm,
            request.vertical_threshold_ft,
        )
        time_relevant_count = sum(1 for fp in flight_paths if fp.is_time_relevant)
        logger.info(f"Generated {len(flight_paths)} flight paths ({time_relevant_count} time-relevant)")
        
        # Step 5: Detect conflicts
        conflicts = detect_conflicts(
            trajectory,
            traffic_list,
            request.horizontal_threshold_nm,
            request.vertical_threshold_ft,
        )
        
        # Calculate trajectory statistics
        total_distance_nm = 0
        for i in range(len(request.waypoints) - 1):
            wp1 = request.waypoints[i]
            wp2 = request.waypoints[i + 1]
            total_distance_nm += haversine_distance_nm(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
        
        total_duration_sec = trajectory[-1].time_offset_sec if trajectory else 0
        
        # Count conflicts by severity
        critical_count = len([c for c in conflicts if c.severity == "critical"])
        warning_count = len([c for c in conflicts if c.severity == "warning"])
        info_count = len([c for c in conflicts if c.severity == "info"])
        
        return {
            "trajectory": [p.to_dict() for p in trajectory],
            "trajectory_summary": {
                "total_points": len(trajectory),
                "total_distance_nm": round(total_distance_nm, 2),
                "total_duration_sec": total_duration_sec,
                "total_duration_min": round(total_duration_sec / 60, 1),
                "start_time": start_dt.isoformat(),
                "end_time": trajectory[-1].timestamp.isoformat() if trajectory else None,
            },
            "airports_checked": [
                {
                    "code": a.get("code"),
                    "name": a.get("name"),
                    "lat": a.get("lat"),
                    "lon": a.get("lon"),
                }
                for a in nearby_airports
            ],
            "traffic_summary": {
                "airports_with_traffic": len(traffic_list),
                "total_flights_analyzed": total_flights,
                "departures": sum(len(t.departures) for t in traffic_list),
                "arrivals": sum(len(t.arrivals) for t in traffic_list),
            },
            "conflicts": [c.to_dict() for c in conflicts],
            "conflicts_summary": {
                "total": len(conflicts),
                "critical": critical_count,
                "warning": warning_count,
                "info": info_count,
            },
            "is_clear": critical_count == 0 and warning_count == 0,
            "thresholds": {
                "horizontal_nm": request.horizontal_threshold_nm,
                "vertical_ft": request.vertical_threshold_ft,
            },
            # Flight paths for visualization (departure/arrival trajectories)
            "flight_paths": [fp.to_dict() for fp in flight_paths],
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error planning trajectory: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/trajectory/traffic/{airport_code}")
async def get_airport_traffic(airport_code: str, date: Optional[str] = None):
    """
    Get departures and arrivals for a specific airport.
    
    Args:
        airport_code: ICAO or IATA airport code
        date: Date in YYYY-MM-DD format (defaults to today)
    """
    try:
        # Find airport in config
        airport = None
        for a in CONFIGURED_AIRPORTS:
            if a.get("code", "").upper() == airport_code.upper():
                airport = a
                break
        
        if not airport:
            raise HTTPException(status_code=404, detail=f"Airport {airport_code} not found")
        
        # Parse date
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            target_date = datetime.now()
        
        # Fetch traffic
        traffic_list = fetch_airport_traffic([airport], target_date)
        
        if not traffic_list:
            return {
                "airport_code": airport_code,
                "airport_name": airport.get("name", ""),
                "date": target_date.strftime("%Y-%m-%d"),
                "departures": [],
                "arrivals": [],
            }
        
        traffic = traffic_list[0]
        
        return {
            "airport_code": traffic.airport_code,
            "airport_name": traffic.airport_name,
            "date": target_date.strftime("%Y-%m-%d"),
            "departures": traffic.departures,
            "arrivals": traffic.arrivals,
            "total_departures": len(traffic.departures),
            "total_arrivals": len(traffic.arrivals),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching airport traffic: {e}")
        raise HTTPException(status_code=500, detail=str(e))



