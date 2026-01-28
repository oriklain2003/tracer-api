from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.config import load_rule_config
from core.geodesy import (
    cross_track_distance_nm,
    haversine_nm,
    initial_bearing_deg,
    is_point_in_polygon,
    create_corridor_polygon,
)
from core.path_utils import resample_track_points, point_to_polyline_distance_nm
from core.models import FlightTrack, RuleContext, RuleResult, TrackPoint
from core.military_detection import is_military

CONFIG = load_rule_config()
RULES = CONFIG.get("rules", {})
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _require_rule_config(rule_name: str) -> Dict[str, Any]:
    cfg = RULES.get(rule_name)
    if cfg is None:
        raise KeyError(f"Missing configuration section for '{rule_name}' in rule_config.json")
    return cfg


EMERGENCY_SQUAWKS = set(CONFIG.get("emergency_squawks", []))

ALTITUDE_CFG = _require_rule_config("altitude_change")
ALTITUDE_CHANGE_FT = float(ALTITUDE_CFG["delta_ft"])
ALTITUDE_WINDOW_SECONDS = int(ALTITUDE_CFG["window_seconds"])
ALTITUDE_MIN_CRUISE_FT = float(ALTITUDE_CFG["min_cruise_ft"])

TURN_CFG = _require_rule_config("abrupt_turn")
TURN_THRESHOLD_DEG = float(TURN_CFG["heading_change_deg"])
TURN_WINDOW_SECONDS = int(TURN_CFG["window_seconds"])
TURN_MIN_SPEED_KTS = float(TURN_CFG["min_speed_kts"])
TURN_ACC_DEG = float(TURN_CFG.get("accumulated_turn_deg", 270))
TURN_ACC_WINDOW = int(TURN_CFG.get("accumulation_window_seconds", 300))

PROXIMITY_CFG = _require_rule_config("proximity")
PROXIMITY_DISTANCE_NM = float(PROXIMITY_CFG["distance_nm"])
PROXIMITY_ALTITUDE_FT = float(PROXIMITY_CFG["altitude_ft"])
PROXIMITY_TIME_WINDOW = int(PROXIMITY_CFG["time_window_seconds"])
PROXIMITY_AIRPORT_EXCLUSION_NM = float(PROXIMITY_CFG.get("airport_exclusion_nm", 0))

ROUTE_DEVIATION_CFG = _require_rule_config("route_deviation")
ROUTE_DEVIATION_NM = float(ROUTE_DEVIATION_CFG["cross_track_nm"])

GO_AROUND_CFG = _require_rule_config("go_around")
GO_AROUND_RADIUS_NM = float(GO_AROUND_CFG["radius_nm"])
GO_AROUND_LOW_ALT_FT = float(GO_AROUND_CFG["min_low_alt_ft"])
GO_AROUND_RECOVERY_FT = float(GO_AROUND_CFG["recovery_ft"])

RETURN_CFG = _require_rule_config("return_to_field")
RETURN_TIME_LIMIT_SECONDS = int(RETURN_CFG["time_limit_seconds"])
RETURN_NEAR_AIRPORT_NM = float(RETURN_CFG["near_airport_nm"])
RETURN_TAKEOFF_ALT_FT = float(RETURN_CFG["takeoff_alt_ft"])
RETURN_LANDING_ALT_FT = float(RETURN_CFG["landing_alt_ft"])
RETURN_MIN_OUTBOUND_NM = float(RETURN_CFG.get("min_outbound_nm", 0))
RETURN_MIN_ELAPSED_SECONDS = int(RETURN_CFG.get("min_elapsed_seconds", 0))

DIVERSION_CFG = _require_rule_config("diversion")
DIVERSION_NEAR_AIRPORT_NM = float(DIVERSION_CFG["near_airport_nm"])

LOW_ALTITUDE_CFG = _require_rule_config("low_altitude")
LOW_ALTITUDE_THRESHOLD_FT = float(LOW_ALTITUDE_CFG["threshold_ft"])
LOW_ALTITUDE_AIRPORT_RADIUS_NM = float(LOW_ALTITUDE_CFG["airport_radius_nm"])

SIGNAL_CFG = _require_rule_config("signal_loss")
SIGNAL_GAP_SECONDS = int(SIGNAL_CFG["gap_seconds"])
SIGNAL_REPEAT_COUNT = int(SIGNAL_CFG["repeat_count"])

UNPLANNED_LANDING_CFG = _require_rule_config("unplanned_israel_landing")
UNPLANNED_LANDING_RADIUS_NM = float(UNPLANNED_LANDING_CFG["near_airport_nm"])

PATH_CFG = _require_rule_config("path_learning")
_path_candidate = Path(PATH_CFG["paths_file"])
PATH_FILE = (_path_candidate if _path_candidate.is_absolute() else (PROJECT_ROOT / _path_candidate)).resolve()
HEATMAP_FILE = Path(PATH_CFG.get("heatmap_file", "rules/flight_heatmap_v2.npy"))
PATH_NUM_SAMPLES = int(PATH_CFG.get("num_samples", 120))
PATH_PRIMARY_RADIUS_NM = float(PATH_CFG.get("primary_radius_nm", 8.0))
PATH_SECONDARY_RADIUS_NM = float(PATH_CFG.get("secondary_radius_nm", 15.0))
HEATMAP_CELL_DEG = float(PATH_CFG.get("heatmap_cell_deg", 0.05))
HEATMAP_THRESHOLD = int(PATH_CFG.get("heatmap_threshold", 5))
MIN_OFF_COURSE_POINTS = int(PATH_CFG.get("min_off_course_points", 15))
EMERGING_DISTANCE_NM = float(PATH_CFG.get("emerging_distance_nm", 12.0))
EMERGING_BUCKET_SIZE = int(PATH_CFG.get("emerging_bucket_size", 5))
EMERGING_SIMILARITY_DEG = int(PATH_CFG.get("emerging_similarity_deg", 30))
DEFAULT_PATH_WIDTH_NM = float(PATH_CFG.get("default_width_nm", 8.0))
TUBE_LATERAL_TOLERANCE_NM = float(PATH_CFG.get("tube_lateral_tolerance_nm", 6))
TUBE_ALTITUDE_TOLERANCE_FT = float(PATH_CFG.get("tube_altitude_tolerance_ft", 2000.0))


_LEARNED_POLYGONS_CACHE = None
_PATH_LIBRARY_CACHE: Optional[Dict[str, Any]] = None
_HEATMAP_CACHE: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None
_LEARNED_TURNS_CACHE: Optional[List[Dict[str, Any]]] = None
_LEARNED_SID_CACHE: Optional[List[Dict[str, Any]]] = None
_LEARNED_STAR_CACHE: Optional[List[Dict[str, Any]]] = None
_LEARNED_TUBES_CACHE: Optional[List[Dict[str, Any]]] = None

# Learned behavior configuration (optional - may not exist)
LEARNED_BEHAVIOR_CFG = RULES.get("learned_behavior", {})
_lb_turns_file = Path(LEARNED_BEHAVIOR_CFG.get("turns_file", "rules/learned_turns.json"))
LEARNED_TURNS_FILE = (_lb_turns_file if _lb_turns_file.is_absolute() else (PROJECT_ROOT / _lb_turns_file)).resolve()
_lb_sid_file = Path(LEARNED_BEHAVIOR_CFG.get("sid_file", "rules/learned_sid.json"))
LEARNED_SID_FILE = (_lb_sid_file if _lb_sid_file.is_absolute() else (PROJECT_ROOT / _lb_sid_file)).resolve()
_lb_star_file = Path(LEARNED_BEHAVIOR_CFG.get("star_file", "rules/learned_star.json"))
LEARNED_STAR_FILE = (_lb_star_file if _lb_star_file.is_absolute() else (PROJECT_ROOT / _lb_star_file)).resolve()
TURN_ZONE_TOLERANCE_NM = float(LEARNED_BEHAVIOR_CFG.get("turn_zone_tolerance_nm", 3.0))
SID_STAR_TOLERANCE_NM = float(LEARNED_BEHAVIOR_CFG.get("sid_star_tolerance_nm", 5.0))


def _load_learned_turns(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load learned turn zones from JSON file."""
    global _LEARNED_TURNS_CACHE
    if _LEARNED_TURNS_CACHE is not None and not refresh:
        return _LEARNED_TURNS_CACHE
    
    try:
        if LEARNED_TURNS_FILE.exists():
            with open(LEARNED_TURNS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _LEARNED_TURNS_CACHE = data.get("zones", [])
        else:
            _LEARNED_TURNS_CACHE = []
    except Exception:
        _LEARNED_TURNS_CACHE = []
    
    return _LEARNED_TURNS_CACHE


def _load_learned_sid(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load learned SID procedures from JSON file."""
    global _LEARNED_SID_CACHE
    if _LEARNED_SID_CACHE is not None and not refresh:
        return _LEARNED_SID_CACHE
    
    try:
        if LEARNED_SID_FILE.exists():
            with open(LEARNED_SID_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _LEARNED_SID_CACHE = data.get("procedures", [])
        else:
            _LEARNED_SID_CACHE = []
    except Exception:
        _LEARNED_SID_CACHE = []
    
    return _LEARNED_SID_CACHE


def _load_learned_star(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load learned STAR procedures from JSON file."""
    global _LEARNED_STAR_CACHE
    if _LEARNED_STAR_CACHE is not None and not refresh:
        return _LEARNED_STAR_CACHE
    
    try:
        if LEARNED_STAR_FILE.exists():
            with open(LEARNED_STAR_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _LEARNED_STAR_CACHE = data.get("procedures", [])
        else:
            _LEARNED_STAR_CACHE = []
    except Exception:
        _LEARNED_STAR_CACHE = []
    
    return _LEARNED_STAR_CACHE


# Tube configuration
_tubes_file = Path("rules/learned_tubes.json")
LEARNED_TUBES_FILE = (_tubes_file if _tubes_file.is_absolute() else (PROJECT_ROOT / _tubes_file)).resolve()


def _load_learned_tubes(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load learned tubes from JSON file."""
    global _LEARNED_TUBES_CACHE
    if _LEARNED_TUBES_CACHE is not None and not refresh:
        return _LEARNED_TUBES_CACHE
    
    try:
        if LEARNED_TUBES_FILE.exists():
            with open(LEARNED_TUBES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                tubes = data.get("tubes", [])
                # Convert geometry to tuple format for faster checking
                for tube in tubes:
                    geom = tube.get("geometry", [])
                    tube["geometry_tuples"] = [(pt[0], pt[1]) for pt in geom if len(pt) >= 2]
                _LEARNED_TUBES_CACHE = tubes
        else:
            _LEARNED_TUBES_CACHE = []
    except Exception as e:
        _LEARNED_TUBES_CACHE = []
    
    return _LEARNED_TUBES_CACHE


def _get_tubes_for_od(
    origin: Optional[str],
    destination: Optional[str],
    all_tubes: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Filter tubes to only those that the flight could belong to.
    
    Logic for flight X -> Y:
    - Prefer exact match: X -> Y
    - If none found, include: X -> None/UNK (same origin, unknown dest)
    - If none found, include: None/UNK -> Y (unknown origin, same dest)
    - If still none, fall back to generic UNK -> UNK tubes
    
    If flight has NO origin AND NO destination:
    - Fallback to all tubes
    
    Args:
        origin: Origin airport code (or None)
        destination: Destination airport code (or None)
        all_tubes: List of all available tubes
        
    Returns:
        Tuple of (filtered_tubes, used_od_filter)
        - filtered_tubes: Tubes matching the O/D (or all tubes if no match)
        - used_od_filter: True if O/D filtering was applied
    """
    # No O/D info at all - fallback to all tubes
    if origin is None and destination is None:
        return all_tubes, False
    
    # Normalize None/"UNK"/empty strings
    def normalize(val):
        if not val or val == "UNK":
            return None
        return val
    
    # Try exact match first
    exact_matches = []
    for tube in all_tubes:
        tube_origin = normalize(tube.get("origin"))
        tube_dest = normalize(tube.get("destination"))
        
        # Exact match: both O and D match (ignoring None)
        origin_match = (origin is None) or (tube_origin is None) or (origin == tube_origin)
        dest_match = (destination is None) or (tube_dest is None) or (destination == tube_dest)
        
        # Prefer tubes with known O/D that match
        if origin_match and dest_match:
            # Prioritize tubes with matching non-None values
            if (tube_origin == origin and tube_dest == destination):
                # Exact match - highest priority
                exact_matches.append((tube, 0))
            elif (tube_origin == origin) or (tube_dest == destination):
                # Partial match - medium priority
                exact_matches.append((tube, 1))
            else:
                # Generic tubes (UNK/None) - lowest priority
                exact_matches.append((tube, 2))
    
    if exact_matches:
        # Sort by priority (exact first, then partial, then generic)
        exact_matches.sort(key=lambda x: x[1])
        
        # Get the best priority level
        best_priority = exact_matches[0][1]
        
        # Use ONLY the best priority level
        # Priority 0 = exact match (LCLK -> LLBG)
        # Priority 1 = partial match (LCLK -> UNK or UNK -> LLBG)  
        # Priority 2 = generic match (UNK -> UNK)
        matched_tubes = [t[0] for t in exact_matches if t[1] == best_priority]
        
        return matched_tubes, True
    
    # No matches at all - fallback to all tubes
    return all_tubes, False


def _is_on_known_turn_zone(lat: float, lon: float) -> bool:
    """Check if a point is within a known turn zone."""
    turn_zones = _load_learned_turns()
    
    for zone in turn_zones:
        zone_lat = zone.get("lat", 0)
        zone_lon = zone.get("lon", 0)
        zone_radius = zone.get("radius_nm", 2.0)
        
        dist = haversine_nm(lat, lon, zone_lat, zone_lon)
        if dist <= zone_radius + TURN_ZONE_TOLERANCE_NM:
            return True
    
    return False


def _is_on_sid_or_star(lat: float, lon: float) -> bool:
    """Check if a point is on a learned SID or STAR centerline."""
    # Check SIDs
    sids = _load_learned_sid()
    for proc in sids:
        centerline = proc.get("centerline", [])
        width = proc.get("width_nm", 6.0)  # Increased default from 3.0 to 6.0 nm
        
        if len(centerline) >= 2:
            coords = [(p["lat"], p["lon"]) for p in centerline if "lat" in p and "lon" in p]
            if coords:
                info = point_to_polyline_distance_nm((lat, lon), coords)
                if info["distance_nm"] <= width + SID_STAR_TOLERANCE_NM:
                    return True
    
    # Check STARs
    stars = _load_learned_star()
    for proc in stars:
        centerline = proc.get("centerline", [])
        width = proc.get("width_nm", 6.0)  # Increased default from 3.0 to 6.0 nm
        
        if len(centerline) >= 2:
            coords = [(p["lat"], p["lon"]) for p in centerline if "lat" in p and "lon" in p]
            if coords:
                info = point_to_polyline_distance_nm((lat, lon), coords)
                if info["distance_nm"] <= width + SID_STAR_TOLERANCE_NM:
                    return True
    
    return False


def _is_on_known_procedure(lat: float, lon: float) -> bool:
    """
    Check if a point is on a known turn zone, SID, or STAR.
    Used to suppress false positives in the turn rule.
    """
    return _is_on_known_turn_zone(lat, lon) or _is_on_sid_or_star(lat, lon)


# New O/D-based paths file
_lb_paths_file = Path(LEARNED_BEHAVIOR_CFG.get("paths_file", "rules/learned_paths.json"))
LEARNED_PATHS_FILE = (_lb_paths_file if _lb_paths_file.is_absolute() else (PROJECT_ROOT / _lb_paths_file)).resolve()
_LEARNED_OD_PATHS_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_learned_od_paths(refresh: bool = False) -> List[Dict[str, Any]]:
    """Load O/D-based learned paths from the new format."""
    global _LEARNED_OD_PATHS_CACHE
    if _LEARNED_OD_PATHS_CACHE is not None and not refresh:
        return _LEARNED_OD_PATHS_CACHE
    
    try:
        if LEARNED_PATHS_FILE.exists():
            with open(LEARNED_PATHS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                paths_raw = data.get("paths", [])
                # Convert to the format expected by existing code
                _LEARNED_OD_PATHS_CACHE = []
                for p in paths_raw:
                    _LEARNED_OD_PATHS_CACHE.append({
                        "id": p.get("id", "unknown"),
                        "type": "od_learned",
                        "origin": p.get("origin"),
                        "destination": p.get("destination"),
                        "centerline": p.get("centerline", []),
                        "width_nm": p.get("width_nm", 4.0),
                        "num_flights": p.get("member_count", 0),
                    })
        else:
            _LEARNED_OD_PATHS_CACHE = []
    except Exception:
        _LEARNED_OD_PATHS_CACHE = []
    
    return _LEARNED_OD_PATHS_CACHE


def _load_path_library(refresh: bool = False) -> Dict[str, Any]:
    """
    Load the path library generated by build_path_library_v2.py.
    """
    global _PATH_LIBRARY_CACHE
    if _PATH_LIBRARY_CACHE is not None and not refresh:
        return _PATH_LIBRARY_CACHE

    try:
        if PATH_FILE.exists():
            with open(PATH_FILE, "r", encoding="utf-8") as f:
                library = json.load(f)
        else:
            library = {}
    except Exception:
        library = {}

    library.setdefault("paths", [])
    library.setdefault("emerging_paths", [])
    library.setdefault("emerging_buckets", [])
    library.setdefault("heatmap", {})

    _PATH_LIBRARY_CACHE = library
    return library


def _save_path_library(library: Dict[str, Any]) -> None:
    """Persist the in-memory path library."""
    global _PATH_LIBRARY_CACHE, _LEARNED_POLYGONS_CACHE
    PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH_FILE, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)
    _PATH_LIBRARY_CACHE = library
    _LEARNED_POLYGONS_CACHE = None  # invalidate polygons


def _get_paths(include_emerging: bool = True, include_od_learned: bool = True) -> List[Dict[str, Any]]:
    """
    Get all paths from various sources.
    
    Args:
        include_emerging: Include emerging/candidate paths
        include_od_learned: Include O/D-based learned paths
        
    Returns:
        List of path dictionaries
    """
    library = _load_path_library()
    paths = list(library.get("paths", []))
    if include_emerging:
        paths += library.get("emerging_paths", [])
    if include_od_learned:
        paths += _load_learned_od_paths()
    return paths


# O/D detection thresholds
OD_AIRPORT_THRESHOLD_NM = 15.0
OD_LOW_ALTITUDE_FT = 5000.0
OD_SEARCH_POINTS = 20


def _detect_flight_od(points: List[TrackPoint]) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect origin and destination airports for a flight based on its track points.
    
    Args:
        points: Sorted list of track points
        
    Returns:
        Tuple of (origin_code, destination_code), either may be None
    """
    if len(points) < 2:
        return None, None
    
    origin = None
    destination = None
    
    # Check first N points for origin (low altitude near airport)
    search_range = min(OD_SEARCH_POINTS, len(points))
    for i in range(search_range):
        p = points[i]
        if (p.alt or 0) <= OD_LOW_ALTITUDE_FT:
            nearest_ap, dist = _nearest_airport(p)
            if nearest_ap and dist <= OD_AIRPORT_THRESHOLD_NM:
                origin = nearest_ap.code
                break
    
    # Check last N points for destination (low altitude near airport)
    search_start = max(0, len(points) - OD_SEARCH_POINTS)
    for i in range(len(points) - 1, search_start - 1, -1):
        p = points[i]
        if (p.alt or 0) <= OD_LOW_ALTITUDE_FT:
            nearest_ap, dist = _nearest_airport(p)
            if nearest_ap and dist <= OD_AIRPORT_THRESHOLD_NM:
                destination = nearest_ap.code
                break
    
    return origin, destination


def _get_paths_for_od(
    origin: Optional[str],
    destination: Optional[str],
    all_paths: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Filter paths to only those that the flight could belong to.
    
    Logic for flight X -> Y:
    - Include paths: X -> Y (exact match)
    - Include paths: X -> None/UNK (same origin, unknown dest)
    - Include paths: None/UNK -> Y (unknown origin, same dest)
    
    If flight has NO origin AND NO destination:
    - Fallback to all paths
    
    Args:
        origin: Origin airport code (or None)
        destination: Destination airport code (or None)
        all_paths: List of all available paths
        
    Returns:
        Tuple of (filtered_paths, used_od_filter)
        - filtered_paths: Paths matching the O/D (or all paths if no match)
        - used_od_filter: True if O/D filtering was applied
    """
    # No O/D info at all - fallback to all paths
    if origin is None and destination is None:
        return all_paths, False
    
    # Filter paths that the flight could belong to
    matching_paths = []
    for path in all_paths:
        path_origin = path.get("origin")
        path_dest = path.get("destination")

        
        # For a flight X -> Y, include path if:
        # 1. Exact match: path is X -> Y
        # 2. Origin match with unknown dest: path is X -> None
        # 3. Dest match with unknown origin: path is None -> Y
        
        if origin and destination:
            # Flight has both O/D
            # Match: X->Y, X->None, None->Y
            exact_match = (path_origin == origin and path_dest == destination)
            origin_with_unk_dest = (path_origin == origin and path_dest is None)
            unk_origin_with_dest = (path_origin is None and path_dest == destination)
            
            if exact_match or origin_with_unk_dest or unk_origin_with_dest:
                matching_paths.append(path)
        
        elif origin and not destination:
            # Flight has only origin
            # Match: X->anything (same origin)
            if path_origin == origin:
                matching_paths.append(path)
        
        elif destination and not origin:
            # Flight has only destination
            # Match: anything->Y (same dest)
            if path_dest == destination:
                matching_paths.append(path)
    
    if matching_paths:
        return matching_paths, True
    
    # No matching paths found - fallback to all paths
    return all_paths, False


def _get_learned_polygons() -> List[Any]:
    global _LEARNED_POLYGONS_CACHE
    if _LEARNED_POLYGONS_CACHE is not None:
        return _LEARNED_POLYGONS_CACHE

    polygons: List[Any] = []
    for path in _get_paths():
        centerline = path.get("centerline") or []
        coords = [(p["lat"], p["lon"]) for p in centerline if "lat" in p and "lon" in p]
        if len(coords) < 2:
            continue
        radius_nm = float(path.get("width_nm", DEFAULT_PATH_WIDTH_NM))
        poly = create_corridor_polygon(coords, radius_nm)
        if poly:
            polygons.append(poly)

    _LEARNED_POLYGONS_CACHE = polygons
    return polygons


def _load_learned_turns() -> List[Dict[str, Any]]:
    """
    Load learned turn spots from learned_paths.json.bak.bak2 (layers.turns).
    Each turn has: centroid_lat, centroid_lon, radius_nm, avg_alt, turn_direction, etc.
    """
    global _LEARNED_TURNS_CACHE
    if _LEARNED_TURNS_CACHE is not None:
        return _LEARNED_TURNS_CACHE

    turns: List[Dict[str, Any]] = []
    learned_paths_file = PROJECT_ROOT / "rules" / "learned_paths.json"
    try:
        if learned_paths_file.exists():
            with open(learned_paths_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # turns are stored under layers.turns
            layers = data.get("layers", {})
            turns = layers.get("turns", [])
    except Exception:
        turns = []

    _LEARNED_TURNS_CACHE = turns
    return turns


def _is_point_in_learned_turn(lat: float, lon: float) -> bool:
    """
    Check if a point (lat, lon) falls within any learned turn spot.
    Returns True if the point is within any turn's radius.
    """
    # Radius overrides for specific turn clusters (cluster_id -> new radius in nm)
    RADIUS_OVERRIDES = {
        "LEFT_10": 20,  # Expanded from 10.83nm to cover Amman area turns
    }
    
    # Global buffer to add to all turn radii (nm)
    RADIUS_BUFFER_NM = 0.0
    
    turns = _load_learned_turns()
    for turn in turns:
        centroid_lat = turn.get("centroid_lat")
        centroid_lon = turn.get("centroid_lon")
        cluster_id = turn.get("cluster_id", "")
        
        # Use override if available, otherwise use stored radius + buffer
        if cluster_id in RADIUS_OVERRIDES:
            radius_nm = RADIUS_OVERRIDES[cluster_id]
        else:
            radius_nm = turn.get("radius_nm", 5.0) + RADIUS_BUFFER_NM
        
        if centroid_lat is None or centroid_lon is None:
            continue
        
        dist_nm = haversine_nm(lat, lon, centroid_lat, centroid_lon)
        if dist_nm <= radius_nm:
            return True
    
    return False


def _load_heatmap(refresh: bool = False) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Load cached heatmap grid (flightability)."""
    global _HEATMAP_CACHE
    if _HEATMAP_CACHE is not None and not refresh:
        return _HEATMAP_CACHE

    library = _load_path_library()
    meta = library.get("heatmap", {}) or {}
    heatmap_path = Path(meta.get("heatmap_file", HEATMAP_FILE))

    grid = None
    try:
        if heatmap_path.exists():
            grid = np.load(heatmap_path)
            meta.setdefault("origin", meta.get("origin", [0.0, 0.0]))
            meta.setdefault("cell_size_deg", meta.get("cell_size_deg", HEATMAP_CELL_DEG))
            meta.setdefault("threshold", meta.get("threshold", HEATMAP_THRESHOLD))
            meta.setdefault("shape", list(grid.shape))
    except Exception:
        grid = None

    _HEATMAP_CACHE = (grid, meta)
    return _HEATMAP_CACHE


def _is_in_flightable_region(point: TrackPoint) -> bool:
    grid, meta = _load_heatmap()
    if grid is None:
        return True

    origin_lat, origin_lon = meta.get("origin", [0.0, 0.0])
    cell = float(meta.get("cell_size_deg", HEATMAP_CELL_DEG))
    rows, cols = meta.get("shape", grid.shape)

    r = int((point.lat - origin_lat) / cell)
    c = int((point.lon - origin_lon) / cell)

    if r < 0 or c < 0 or r >= rows or c >= cols:
        return False

    threshold = int(meta.get("threshold", HEATMAP_THRESHOLD))
    return grid[r, c] >= threshold


def _distance_to_path(point: TrackPoint, path_entry: Dict[str, Any]) -> Tuple[float, float]:
    """Return min lateral distance (nm) and normalized position along path (0-1)."""
    centerline = path_entry.get("centerline") or []
    coords = [(p["lat"], p["lon"]) for p in centerline if "lat" in p and "lon" in p]
    if len(coords) < 2:
        return float("inf"), 0.0
    info = point_to_polyline_distance_nm((point.lat, point.lon), coords)
    return float(info["distance_nm"]), float(info["position"])


def _compress_heading_signature(
    points: Sequence[TrackPoint],
    *,
    bin_seconds: int = 10,
    bin_size_deg: int = EMERGING_SIMILARITY_DEG,
) -> Tuple[int, ...]:
    """Build a compact heading signature for emerging path detection."""
    if not points:
        return ()

    ordered = sorted(points, key=lambda p: p.timestamp)
    next_bucket_ts = ordered[0].timestamp + bin_seconds
    bucket_headings: List[float] = []
    signature: List[int] = []
    prev = ordered[0]

    for p in ordered[1:]:
        heading = p.track
        if heading is None:
            heading = initial_bearing_deg(prev.lat, prev.lon, p.lat, p.lon)

        bucket_headings.append(heading % 360.0)

        if p.timestamp >= next_bucket_ts:
            mean_heading = sum(bucket_headings) / len(bucket_headings)
            signature.append(int(mean_heading // bin_size_deg))
            bucket_headings = []
            next_bucket_ts += bin_seconds

        prev = p

    if bucket_headings:
        mean_heading = sum(bucket_headings) / len(bucket_headings)
        signature.append(int(mean_heading // bin_size_deg))

    return tuple(signature)


def _update_emerging_buckets(
    ctx: RuleContext,
    off_path_points: Sequence[TrackPoint],
) -> Optional[Dict[str, Any]]:
    """
    Append the flight to an emerging-path candidate bucket and promote when enough samples arrive.
    """
    if not off_path_points:
        return None

    signature = _compress_heading_signature(off_path_points)
    if not signature:
        return None

    library = _load_path_library()
    buckets = library.setdefault("emerging_buckets", [])

    bucket = next((b for b in buckets if tuple(b.get("signature", ())) == signature), None)
    if bucket is None:
        bucket = {"signature": signature, "count": 0, "flight_ids": []}
        buckets.append(bucket)

    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket.setdefault("flight_ids", []).append(ctx.track.flight_id)
    bucket["last_updated"] = datetime.utcnow().isoformat() + "Z"

    promoted: Optional[Dict[str, Any]] = None
    if bucket["count"] >= EMERGING_BUCKET_SIZE:
        centerline = resample_track_points(ctx.track.points, num_samples=PATH_NUM_SAMPLES)
        if centerline is not None:
            coords = [(float(lat), float(lon)) for lat, lon, _ in centerline.tolist()]
            width_nm = DEFAULT_PATH_WIDTH_NM
            try:
                dists = [
                    point_to_polyline_distance_nm((p.lat, p.lon), coords)["distance_nm"]
                    for p in ctx.track.points
                ]
                if dists:
                    width_nm = max(float(np.std(dists)), 2.0)
            except Exception:
                pass

            emerging_list = library.setdefault("emerging_paths", [])
            path_id = f"emerging_{len(emerging_list) + 1}"
            promoted = {
                "id": path_id,
                "type": "emerging",
                "width_nm": width_nm,
                "centerline": [{"lat": lat, "lon": lon} for lat, lon in coords],
                "num_flights": bucket["count"],
                "created_from_signature": signature,
            }
            emerging_list.append(promoted)
            buckets.remove(bucket)

    _save_path_library(library)
    return promoted


@dataclass(frozen=True)
class Airport:
    code: str
    name: str
    lat: float
    lon: float
    elevation_ft:Optional[Any] = None

AIRPORT_ENTRIES = CONFIG.get("airports", [])
AIRPORTS: List[Airport] = [Airport(**entry) for entry in AIRPORT_ENTRIES]
AIRPORT_BY_CODE: Dict[str, Airport] = {a.code: a for a in AIRPORTS}


def is_bad_segment(prev: TrackPoint, curr: TrackPoint) -> bool:
    dt = curr.timestamp - prev.timestamp
    if dt <= 0:
        return True

    # 1. Teleport / impossible movement
    max_nm = (curr.gspeed or prev.gspeed or 350) * dt / 3600
    dist_nm = haversine_nm(prev.lat, prev.lon, curr.lat, curr.lon)
    if dist_nm > max_nm * 3:
        return True

    # 2. Impossible heading jump
    if prev.track is not None and curr.track is not None:
        dh = abs(((curr.track - prev.track + 540) % 360) - 180)
        if dh > 80:
            return True

    # 3. FR24 ocean gap -> ignore off-course
    # Far from land + cruise altitude = probably artifact
    nearest_airport, dist_ap = _nearest_airport(curr)
    if (curr.alt or 0) > 15000 and (dist_ap or 999) > 60:
        return True

    # 4. Zero-altitude glitch at high speed
    if (curr.alt or 0) < 200 and (curr.gspeed or 0) > 200:
        return True

    return False


def is_impossible_point(
    points: List[TrackPoint],
    idx: int,
    speed_buffer: float = 1.5,
    max_turn_rate_deg_s: float = 8.0,
    max_vertical_speed_ft_s: float = 200.0
) -> bool:
    """
    Detect if a point is physically impossible based on its neighbors.
    
    This function helps identify GPS glitches, ADS-B errors, and data corruption by checking
    if an aircraft could have realistically traveled from the previous point to the current
    point to the next point given the laws of physics.
    
    Checks performed:
    1. HORIZONTAL DISTANCE: Can the aircraft travel the distance in the time available?
       - Uses reported ground speed with a buffer factor (default 1.5x for acceleration/deceleration)
       - If the plane would need to travel faster than physically possible, it's a glitch
    
    2. TURN RATE: Can the aircraft make the required turn in the time available?
       - Max turn rate for commercial aircraft: ~3 deg/sec sustained
       - Max turn rate for fighters: ~20 deg/sec sustained, 8 deg/sec for instantaneous
       - Default threshold: 8.0 deg/sec (conservative, catches most glitches)
       - If the turn would require > max_turn_rate, it's likely a heading sensor error
    
    3. VERTICAL SPEED: Can the aircraft climb/descend that fast?
       - Max vertical speed for commercial: ~2500 ft/min (~42 ft/sec)
       - Max vertical speed for fighters: ~10000 ft/min (~167 ft/sec)
       - Default threshold: 200 ft/sec (~12000 fpm, very conservative)
       - If altitude change rate exceeds this, it's an altimeter glitch
    
    Usage:
        This function should be called before flagging anomalies to ensure we're not
        reporting issues on corrupted data points. For example, a "low altitude" alert
        on a point that teleported from 5000ft to 100ft to 5000ft in 2 seconds is
        clearly a data glitch, not a real low altitude event.
    
    Args:
        points: List of TrackPoints sorted by timestamp
        idx: Index of the point to check (must have neighbors on both sides)
        speed_buffer: Multiplier for speed tolerance (default 1.5 = 50% buffer)
        max_turn_rate_deg_s: Maximum turn rate in degrees per second (default 8.0)
        max_vertical_speed_ft_s: Maximum vertical speed in feet per second (default 200.0)
    
    Returns:
        True if the point is physically impossible (should be ignored)
        False if the point appears physically plausible
    
    Example:
        # Flight track with glitch: [5000ft, 100ft, 5000ft] in 2 seconds each
        # is_impossible_point(points, 1) -> True (can't dive and climb that fast)
        
        # Flight track normal: [5000ft, 4800ft, 4600ft] in 5 seconds each  
        # is_impossible_point(points, 1) -> False (normal descent rate)
    """
    # Need both neighbors to perform the check
    if idx <= 0 or idx >= len(points) - 1:
        return False
    
    prev = points[idx - 1]
    curr = points[idx]
    next_p = points[idx + 1]
    
    # ================================================================
    # CHECK 1: HORIZONTAL DISTANCE (Can the plane travel that far?)
    # ================================================================
    
    # Check prev -> curr
    dt_prev = curr.timestamp - prev.timestamp
    if dt_prev > 0:
        dist_prev = haversine_nm(prev.lat, prev.lon, curr.lat, curr.lon)
        # Use the higher of the two speeds (to be generous)
        speed_prev = max(prev.gspeed or 0, curr.gspeed or 0, 300)  # 300 kts min assumed
        max_possible_dist_prev = (speed_prev * dt_prev / 3600.0) * speed_buffer
        
        if dist_prev > max_possible_dist_prev:
            # Plane moved too far from prev to curr
            return True
    
    # Check curr -> next
    dt_next = next_p.timestamp - curr.timestamp
    if dt_next > 0:
        dist_next = haversine_nm(curr.lat, curr.lon, next_p.lat, next_p.lon)
        speed_next = max(curr.gspeed or 0, next_p.gspeed or 0, 300)
        max_possible_dist_next = (speed_next * dt_next / 3600.0) * speed_buffer
        
        if dist_next > max_possible_dist_next:
            # Plane moved too far from curr to next
            return True
    
    # ================================================================
    # CHECK 2: TURN RATE (Can the plane turn that sharply?)
    # ================================================================
    
    # Check prev -> curr turn
    if prev.track is not None and curr.track is not None and dt_prev > 0:
        heading_change_prev = abs(((curr.track - prev.track + 540) % 360) - 180)
        turn_rate_prev = heading_change_prev / dt_prev
        
        if turn_rate_prev > max_turn_rate_deg_s:
            # Turn rate too high from prev to curr
            return True
    
    # Check curr -> next turn
    if curr.track is not None and next_p.track is not None and dt_next > 0:
        heading_change_next = abs(((next_p.track - curr.track + 540) % 360) - 180)
        turn_rate_next = heading_change_next / dt_next
        
        if turn_rate_next > max_turn_rate_deg_s:
            # Turn rate too high from curr to next
            return True
    
    # ================================================================
    # CHECK 3: VERTICAL SPEED (Can the plane climb/descend that fast?)
    # ================================================================
    
    # Check prev -> curr vertical speed
    if prev.alt is not None and curr.alt is not None and dt_prev > 0:
        alt_change_prev = abs(curr.alt - prev.alt)
        vertical_speed_prev = alt_change_prev / dt_prev
        
        if vertical_speed_prev > max_vertical_speed_ft_s:
            # Climbing/descending too fast from prev to curr
            return True
    
    # Check curr -> next vertical speed
    if curr.alt is not None and next_p.alt is not None and dt_next > 0:
        alt_change_next = abs(next_p.alt - curr.alt)
        vertical_speed_next = alt_change_next / dt_next
        
        if vertical_speed_next > max_vertical_speed_ft_s:
            # Climbing/descending too fast from curr to next
            return True
    
    # All checks passed - point appears physically plausible
    return False


def evaluate_rule(context: RuleContext, rule_id: int) -> RuleResult:
    evaluators = {
        1: _rule_emergency_squawk,
        2: _rule_extreme_altitude_change,
        3: _rule_abrupt_turn,
        4: _rule_dangerous_proximity,
        6: _rule_go_around,
        7: _rule_takeoff_return,
        # 8: _rule_diversion,
        9: _rule_low_altitude,
        # 10: _rule_signal_loss,
        12: _rule_unplanned_israel_landing,
        11: _rule_off_course,
        13: _rule_military_aircraft,
    }
    evaluator = evaluators.get(rule_id)
    if evaluator is None:
        return RuleResult(rule_id=rule_id, matched=False, summary="Rule not implemented", details={})
    return evaluator(context)


def _rule_emergency_squawk(ctx: RuleContext) -> RuleResult:
    events = [
        {"timestamp": p.timestamp, "squawk": p.squawk}
        for p in ctx.track.sorted_points()
        if p.squawk and p.squawk.strip() in EMERGENCY_SQUAWKS
    ]
    matched = bool(events)
    summary = "Emergency code transmitted" if matched else "No emergency squawk detected"
    return RuleResult(1, matched, summary, {"events": events})


def _rule_extreme_altitude_change(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    events = []

    # precompute airport distances once
    distances = [_nearest_airport(p)[1] for p in points]

    def is_noise_sequence(start_idx: int, end_idx: int) -> bool:
        """
        Check if points in [start_idx, end_idx] form a noise sequence.
        A noise sequence has suspicious altitude values (like 0) but
        the flight resumes normal altitude after the sequence.
        """
        if start_idx < 0 or end_idx >= len(points) or start_idx >= end_idx:
            return False

        # Check if we have suspicious low altitudes in the sequence
        noise_points = points[start_idx:end_idx + 1]
        if not any(p.alt < 500 for p in noise_points):
            return False

        # Check altitude before the sequence
        if start_idx == 0:
            return False
        prev_alt = points[start_idx - 1].alt

        # Check altitude after the sequence
        if end_idx + 1 >= len(points):
            return False
        next_alt = points[end_idx + 1].alt

        # If before and after are similar (within reasonable range), it's likely noise
        if prev_alt < ALTITUDE_MIN_CRUISE_FT or next_alt < ALTITUDE_MIN_CRUISE_FT:
            return False

        # Check if altitudes before and after are similar (within 5000 ft)
        if abs(next_alt - prev_alt) < 5000:
            # Check if the noise sequence is short (less than 10 points) and far from airports
            sequence_duration = points[end_idx].timestamp - points[start_idx].timestamp
            min_dist = min(distances[max(0, start_idx):min(len(distances), end_idx + 1)])
            if sequence_duration < 300 and (min_dist is None or min_dist > 5):
                return True

        return False

    i = 0
    while i < len(points) - 1:
        prev = points[i]
        curr = points[i + 1]

        # Physics check: skip impossible points (GPS glitches)
        if is_impossible_point(points, i + 1):
            i += 1
            continue

        dt = curr.timestamp - prev.timestamp
        if dt <= 0 or dt > ALTITUDE_WINDOW_SECONDS:
            i += 1
            continue

        # cruise check
        if prev.alt < ALTITUDE_MIN_CRUISE_FT:
            i += 1
            continue

        # -----------------------------
        # NOISE DETECTION:
        # Check for consecutive noise points (e.g., 4 points with 0 alt)
        # when transitioning from normal altitude
        # -----------------------------
        if curr.alt < 500 and prev.alt >= ALTITUDE_MIN_CRUISE_FT:
            # Look ahead to find the end of a potential noise sequence
            noise_start = i + 1
            noise_end = noise_start
            while noise_end + 1 < len(points) and points[noise_end + 1].alt < 500:
                noise_end += 1

            # If we found a noise sequence (multiple consecutive points), check if it's noise
            if noise_end >= noise_start:
                # Check if we have at least 2 noise points OR single point that recovers quickly
                if (noise_end > noise_start or
                    (noise_end == noise_start and noise_end + 1 < len(points) and
                     points[noise_end + 1].alt >= ALTITUDE_MIN_CRUISE_FT)):
                    if is_noise_sequence(noise_start, noise_end):
                        # Skip the noise sequence and also skip the transition back to normal
                        # by comparing the point before noise to the point after noise
                        i = noise_end + 1
                        if i < len(points) - 1 and abs(points[noise_end + 1].alt - prev.alt) < 5000:
                            # Altitudes before and after noise are similar, skip this transition too
                            i += 1
                        continue

        # Also check if we're transitioning from noise back to normal (safety net for missed cases)
        if prev.alt < 500 and curr.alt >= ALTITUDE_MIN_CRUISE_FT:
            # Look backwards to find if prev is part of a noise sequence
            noise_end = i
            noise_start = i
            while noise_start > 0 and points[noise_start - 1].alt < 500:
                noise_start -= 1

            # Check if this is a noise sequence (even single point)
            if noise_start <= noise_end and noise_start > 0:
                before_noise_alt = points[noise_start - 1].alt
                if (before_noise_alt >= ALTITUDE_MIN_CRUISE_FT and
                    abs(curr.alt - before_noise_alt) < 5000):
                    # This is noise recovering to normal, skip it
                    # Check duration and distance to confirm it's noise
                    if noise_start < noise_end:  # Multiple noise points
                        sequence_duration = points[noise_end].timestamp - points[noise_start].timestamp
                        min_dist = min(distances[max(0, noise_start):min(len(distances), noise_end + 1)])
                        if sequence_duration < 300 and (min_dist is None or min_dist > 5):
                            i += 1
                            continue
                    else:  # Single noise point
                        min_dist = distances[i] if i < len(distances) else None
                        if min_dist is None or min_dist > 5:
                            i += 1
                            continue

        # -----------------------------
        # Glitch Filter: 0 altitude
        # -----------------------------
        # If altitude drops to 0 from a significant altitude, it's a glitch.
        if (curr.alt or 0) <= 0:
             # If we were flying (> 500 ft), a sudden 0 is fake.
             if (prev.alt or 0) > 500:
                 i += 1
                 continue

        # -----------------------------
        # SOFT FILTER 1:
        # Ignore collapses to 0 ft far from airports
        # -----------------------------
        if curr.alt == 0 and distances[i + 1] and distances[i + 1] > 7:
            # noise signature: 35000 → 0 → 35000
            if i + 2 < len(points):
                next_alt = points[i + 2].alt
                if abs(next_alt - prev.alt) < 3000:
                    i += 1
                    continue

        # -----------------------------
        # SOFT FILTER 2:
        # Impossible physics: 0 ft + high speed
        # -----------------------------
        if curr.alt < 200 and curr.gspeed > 200 and distances[i + 1] > 3:
            i += 1
            continue

        delta = curr.alt - prev.alt

        if abs(delta) >= ALTITUDE_CHANGE_FT:
            rate = delta / dt
            events.append({
                "timestamp": curr.timestamp,
                "delta_ft": round(delta, 2),
                "rate_ft_per_s": round(rate, 2),
            })

        i += 1

    matched = bool(events)
    summary = "Detected rapid altitude changes" if matched else "Altitude profile nominal"
    return RuleResult(2, matched, summary, {"events": events})


def _rule_abrupt_turn(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    events = []

    if len(points) < 4:
        return RuleResult(3, False, "Not enough datapoints", {})

    # -----------------------------
    # 1. Detect abrupt single-point turns (existing logic kept)
    # -----------------------------

    def smooth_heading(i):
        if i == 0 or i == len(points) - 1:
            return points[i].track
        prev_h = points[i - 1].track
        curr_h = points[i].track
        next_h = points[i + 1].track
        if prev_h is None or curr_h is None or next_h is None:
            return curr_h
        return (prev_h + curr_h + next_h) / 3.0

    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]

        # Physics check: skip impossible points (GPS glitches)
        if is_impossible_point(points, i):
            continue

        if is_bad_segment(prev, curr):
            continue

        if prev.track is None or curr.track is None:
            continue

        dt = curr.timestamp - prev.timestamp
        if dt <= 0 or dt > TURN_WINDOW_SECONDS:
            continue

        dist_nm = haversine_nm(prev.lat, prev.lon, curr.lat, curr.lon)
        max_possible_nm = (curr.gspeed or 300) * dt / 3600.0
        if dist_nm > max_possible_nm * 3.0:
            continue

        prev_h = smooth_heading(i - 1)
        curr_h = smooth_heading(i)

        if prev_h is None or curr_h is None:
            continue

        diff = _heading_diff(curr_h, prev_h)

        # aerodynamically impossible
        if abs(diff) / dt > 5.0:
            continue

        # ignore if too slow
        if (curr.gspeed or 0.0) < TURN_MIN_SPEED_KTS:
            continue

        if abs(diff) >= TURN_THRESHOLD_DEG:
            # Suppress if within a learned turn spot
            if _is_point_in_learned_turn(curr.lat, curr.lon):
                continue
            
            # Suppress if near an airport (within 6 nm) - normal maneuvering
            nearest_ap, dist_ap = _nearest_airport(curr)
            if dist_ap is not None and dist_ap < 6.0:
                continue
            
            events.append({
                "timestamp": curr.timestamp,
                "turn_deg": round(diff, 2),
                "dt_s": dt,
                "smoothed_prev": round(prev_h, 2),
                "smoothed_curr": round(curr_h, 2),
            })

    # ==========================================================
    # ========== 2. NEW → Clean Holding Pattern Detection =======
    # ==========================================================

    TURN_MIN_DURATION = 45        # must be at least 45 seconds (allows 180 turn)
    TURN_RATE_MAX = 12.0          # deg/sec threshold (increased to 12 as per rule update)
    TURN_MIN_SPEED = 80           # kts
    TURN_MIN_ALT = 1500           # ft
    ACC_THRESHOLD_180 = 220       # half-turn (tightened to avoid known patterns)
    ACC_THRESHOLD_360 = 320       # full orbit - require near-complete 360° to avoid false positives on curved routes
    TURN_MAX_ACCEL_KTS_S = 10.0   # max acceleration to reject speed glitches

    def signed_delta(h1, h0):
        """Return signed heading delta in [-180, 180]."""
        return ((h1 - h0 + 540) % 360) - 180

    # Holding-pattern scan
    for start_idx in range(len(points)):
        start_p = points[start_idx]

        if start_p.track is None or (start_p.gspeed or 0) < TURN_MIN_SPEED or (start_p.alt or 0) < TURN_MIN_ALT:
            continue

        # START POINT SANITY CHECK
        # If the start point itself is the result of a massive instantaneous jump from the previous point,
        # it is likely a glitch start and should not be used as the anchor for a turn analysis.
        if start_idx > 0:
            p_prev = points[start_idx - 1]
            dt_prev = start_p.timestamp - p_prev.timestamp
            # Ensure previous point is recent enough to matter
            if 0 < dt_prev < TURN_ACC_WINDOW and p_prev.track is not None:
                dh_prev = signed_delta(start_p.track, p_prev.track)
                rate_prev = abs(dh_prev) / dt_prev
                accel_prev = abs((start_p.gspeed or 0) - (p_prev.gspeed or 0)) / dt_prev

                if rate_prev > TURN_RATE_MAX or accel_prev > TURN_MAX_ACCEL_KTS_S:
                    continue

        cumulative = 0.0
        direction = None

        prev_idx = start_idx
        for end_idx in range(start_idx + 1, len(points)):
            p0 = points[prev_idx]
            p1 = points[end_idx]

            # Physics check: skip impossible points (GPS glitches)
            if is_impossible_point(points, end_idx):
                continue

            if p1.timestamp - start_p.timestamp > TURN_ACC_WINDOW:
                break

            if p1.track is None:
                continue
            if (p1.gspeed or 0) < TURN_MIN_SPEED:
                continue
            if (p1.alt or 0) < TURN_MIN_ALT:
                continue
            
            # NEW: Reject impossible speeds (GPS jamming artifact)
            # Commercial aircraft max speed is ~600 kts, anything above is a glitch
            if (p1.gspeed or 0) > 600:
                continue

            dt = p1.timestamp - p0.timestamp
            if dt <= 0:
                continue
            
            # --- NEW: Ignore points descending within 5 miles of an airport ---
            # Even if above TURN_MIN_ALT, we ignore descent segments near airports
            # as they often involve maneuvering that isn't a "holding pattern" anomaly.
            nearest_ap, dist_ap = _nearest_airport(p1)
            if dist_ap is not None and dist_ap < 5.0:
                # Check if descending
                if (p1.alt or 0) < (p0.alt or 0):
                    continue

            # --- High Density Logic ---
            # If dt >= 10s, we have a gap in the data. We can't reliably compute
            # the heading change across this gap, so we reset prev_idx to this point
            # and continue looking for consecutive close points.
            if dt >= 10.0:
                # Reset to this point - it becomes the new reference for subsequent points
                prev_idx = end_idx
                continue

            # Acceleration check (reject speed glitches)
            # When we encounter a speed glitch, we skip the point but still update prev_idx
            # so subsequent points are compared against a recent reference.
            accel = abs((p1.gspeed or 0) - (p0.gspeed or 0)) / dt
            if accel > TURN_MAX_ACCEL_KTS_S:
                prev_idx = end_idx  # Update reference to avoid cascading failures
                continue
            
            # NEW: Teleportation check (GPS jamming/spoofing artifact)
            # If the aircraft moved farther than physically possible, it's a glitch
            dist_nm = haversine_nm(p0.lat, p0.lon, p1.lat, p1.lon)
            max_possible_nm = (max(p1.gspeed or 0, p0.gspeed or 0, 300)) * dt / 3600.0
            if max_possible_nm > 0 and dist_nm > max_possible_nm * 2.5:
                prev_idx = end_idx  # Update reference
                continue

            # Make sure we use the last GOOD point for heading calculation
            # p0 is the previous point (prev_idx). In this loop, prev_idx is updated
            # only at the end of the loop when a point is accepted.
            dh = signed_delta(p1.track, p0.track)
            # turn rate sanity
            if abs(dh) / dt > TURN_RATE_MAX:
                continue

            # TRACK vs BEARING Check (Detect Sensor Failure / Crabbing Glitch)
            # If the reported heading (track) is wildly different from the actual path (bearing),
            # the sensor data is likely invalid.
            # We only check this if moving at reasonable speed (not hovering) and sufficient distance.
            if (p1.gspeed or 0) > 50 and dt > 2:
                bearing = initial_bearing_deg(p0.lat, p0.lon, p1.lat, p1.lon)
                # Calculate difference between reported track and actual bearing
                diff_track_bearing = abs(((p1.track - bearing + 540) % 360) - 180)
                
                # If difference is > 90 degrees, the heading is likely garbage (e.g. sensor stuck or flipped)
                if diff_track_bearing > 90:
                    continue

            # Update previous valid index
            prev_idx = end_idx

            # establish/validate direction
            if dh == 0:
                continue

            sign = 1 if dh > 0 else -1
            if direction is None:
                direction = sign
            elif sign != direction:
                # tolerate tiny opposite blips (≤10 deg) without killing accumulation
                if abs(dh) <= 10:
                    continue
                break

            cumulative += abs(dh)

            duration = p1.timestamp - start_p.timestamp
            if duration < TURN_MIN_DURATION:
                continue

            # detect events
            if cumulative >= ACC_THRESHOLD_360 and duration >= 60:
                # For 360 degree turns (full orbit), verify this is actually an orbit
                # by checking displacement - a true orbit returns near the starting point.
                #
                # The cumulative heading can be high even on curved routes (like SIDs),
                # but a true 360° orbit has LOW displacement (aircraft loops back).
                displacement_nm = haversine_nm(start_p.lat, start_p.lon, p1.lat, p1.lon)
                
                # Calculate path length for this segment
                path_nm = 0.0
                for k in range(start_idx, end_idx):
                    path_nm += haversine_nm(points[k].lat, points[k].lon, 
                                           points[k+1].lat, points[k+1].lon)
                
                # Check for impossible average speed (GPS glitches can create impossible paths)
                avg_speed_kts = (path_nm / duration) * 3600 if duration > 0 else 999
                if avg_speed_kts > 650:
                    continue  # Impossible average speed - likely GPS glitches creating false path
                
                # A true 360° orbit should have:
                # - High path/displacement ratio (aircraft traveled far but ended up close to start)
                # - Small displacement (< 10nm for a typical holding pattern)
                #
                # If displacement is large, it's just a curved route, not an orbit
                if displacement_nm > 10.0:
                    continue  # Not an orbit - aircraft traveled away from start
                
                # If path/displacement ratio is low, it's not really an orbit
                if displacement_nm > 0 and path_nm / displacement_nm < 2.0:
                    continue  # Not looping back - just a curved path
                
                # Suppress if very close to an airport (likely a hold for landing)
                nearest_ap, dist_ap = _nearest_airport(p1)
                near_airport = dist_ap is not None and dist_ap < 6.0
                
                # NEW: Suppress go-around patterns - check if ANY point in the pattern
                # was close to an airport at low altitude (approach-related maneuver)
                is_go_around_pattern = False
                pattern_points = points[start_idx:end_idx + 1]
                for pp in pattern_points:
                    pp_ap, pp_dist = _nearest_airport(pp)
                    if pp_ap and pp_dist is not None:
                        pp_elev = pp_ap.elevation_ft or 0
                        pp_agl = (pp.alt or 0) - pp_elev
                        # If any point was within 5nm of airport at low altitude (< 2000ft AGL)
                        # this looks like an approach/go-around, not a suspicious holding pattern
                        if pp_dist < 5.0 and pp_agl < 2000:
                            is_go_around_pattern = True
                            break

                if not near_airport and not is_go_around_pattern:
                    events.append({
                        "type": "holding_pattern",
                        "timestamp": p1.timestamp,
                        "start_ts": start_p.timestamp,
                        "end_ts": p1.timestamp,
                        "duration_s": duration,
                        "cumulative_turn_deg": round(cumulative, 2),
                        "pattern": "360_turn",
                        "displacement_nm": round(displacement_nm, 2),
                        "path_nm": round(path_nm, 2),
                    })
                break

            if cumulative >= ACC_THRESHOLD_180 and duration >= 70:
                polygons = _get_learned_polygons()
                is_suppressed = False
                latlon = (p1.lat, p1.lon)
                for poly in polygons:
                    if is_point_in_polygon(latlon, poly):
                        is_suppressed = True
                        break
                nearest_ap, dist_ap = _nearest_airport(p1)
                near_airport = dist_ap is not None and dist_ap < 6.0
                
                # Check if on known turn zone, SID, or STAR
                on_known_procedure = _is_on_known_procedure(p1.lat, p1.lon)

                if not is_suppressed and not near_airport and not on_known_procedure:
                    events.append({
                        "type": "holding_pattern",
                        "timestamp": p1.timestamp,
                        "start_ts": start_p.timestamp,
                        "end_ts": p1.timestamp,
                        "duration_s": duration,
                        "cumulative_turn_deg": round(cumulative, 2),
                        "pattern": "180_turn"
                    })
                    break
                # If 180 turn is suppressed, continue looking for 360 turn
                # (don't break - the cumulative might reach 360 threshold)

    # ----------------------------------------------------------
    # Fallback: geometric loop detector (path vs displacement)
    # If the above stricter scan missed an obvious loop (e.g. sparse points),
    # use path/disp ratio and cumulative heading to decide.
    # ----------------------------------------------------------
    if not events:
        SPEED_FALLBACK = 80
        ALT_FALLBACK = 1500
        MIN_DURATION = 70          # avoid flagging short orbits in known patterns
        MAX_DURATION = 240
        MIN_HEADING_ACC = 320      # Require near-complete orbit (320°+) to flag as 360 turn
        MIN_PATH_DISP_RATIO = 1.35
        
        # GPS glitch/jamming detection thresholds
        MAX_SPEED_KTS = 600        # Max realistic speed for commercial aircraft
        MAX_TURN_RATE_DEG_S = 8.0  # Max realistic instantaneous turn rate
        MAX_TELEPORT_FACTOR = 2.5  # Max distance vs possible distance ratio

        def heading_acc_signed(points_slice):
            """Calculate signed cumulative heading change (positive = right, negative = left)."""
            acc = 0.0
            for a, b in zip(points_slice, points_slice[1:]):
                if a.track is None or b.track is None:
                    continue
                dh = ((b.track - a.track + 540) % 360) - 180
                acc += dh
            return acc
        
        def has_gps_glitches(points_slice) -> bool:
            """
            Check if a point slice contains GPS glitches/jamming artifacts.
            
            Signs of GPS jamming/spoofing:
            1. Impossible speeds (>600 kts for commercial aircraft)
            2. Teleportation (distance traveled > physically possible)
            3. Impossible turn rates (>8 deg/sec instantaneous)
            4. Heading flipping back and forth (oscillation)
            5. Track vs actual bearing mismatch
            """
            if len(points_slice) < 2:
                return False
            
            glitch_count = 0
            heading_reversals = 0
            prev_heading_change = 0
            
            for a, b in zip(points_slice, points_slice[1:]):
                dt = b.timestamp - a.timestamp
                if dt <= 0:
                    continue
                
                # 1. Check for impossible speeds
                if (b.gspeed or 0) > MAX_SPEED_KTS or (a.gspeed or 0) > MAX_SPEED_KTS:
                    glitch_count += 1
                    continue
                
                # 2. Check for teleportation
                dist_nm = haversine_nm(a.lat, a.lon, b.lat, b.lon)
                max_possible_nm = (max(b.gspeed or 0, a.gspeed or 0, 300)) * dt / 3600.0
                if max_possible_nm > 0 and dist_nm > max_possible_nm * MAX_TELEPORT_FACTOR:
                    glitch_count += 1
                    continue
                
                # 3. Check for impossible turn rates
                if a.track is not None and b.track is not None:
                    dh = abs(((b.track - a.track + 540) % 360) - 180)
                    turn_rate = dh / dt if dt > 0 else 0
                    if turn_rate > MAX_TURN_RATE_DEG_S:
                        glitch_count += 1
                    
                    # 4. Check for heading oscillation (rapid back-and-forth)
                    signed_dh = ((b.track - a.track + 540) % 360) - 180
                    if prev_heading_change != 0:
                        # If heading changed direction significantly
                        if (prev_heading_change > 20 and signed_dh < -20) or \
                           (prev_heading_change < -20 and signed_dh > 20):
                            heading_reversals += 1
                    prev_heading_change = signed_dh
                
                # 5. Check track vs actual bearing mismatch
                if b.track is not None and dist_nm > 0.1:  # Only check if moved enough
                    actual_bearing = initial_bearing_deg(a.lat, a.lon, b.lat, b.lon)
                    bearing_mismatch = abs(((b.track - actual_bearing + 540) % 360) - 180)
                    if bearing_mismatch > 120:  # Reported heading way off from actual movement
                        glitch_count += 1
            
            # If we have multiple glitches or heading reversals, it's likely GPS issues
            num_segments = len(points_slice) - 1
            if num_segments > 0:
                glitch_ratio = glitch_count / num_segments
                reversal_ratio = heading_reversals / num_segments
                
                # More than 5% glitchy segments OR more than 8% heading reversals
                # (Lowered from 10% and 15% to be more aggressive at filtering glitches)
                if glitch_ratio > 0.05 or reversal_ratio > 0.08:
                    return True
                
                # Also reject if we have any glitches AND high reversals
                # (Combined indicators are strong evidence of GPS issues)
                if glitch_count > 0 and reversal_ratio > 0.05:
                    return True
            
            return False
        
        def is_complete_orbit(points_slice):
            """
            Check if a flight segment represents a complete orbit (360° turn).
            A true orbit should:
            1. Have signed cumulative heading >= 320° (or <= -320° for left turns)
            2. End up pointing roughly the same direction as it started
            """
            if len(points_slice) < 3:
                return False, 0
            
            signed_acc = heading_acc_signed(points_slice)
            
            # Check if we completed at least ~320° in one direction
            if abs(signed_acc) < 320:
                return False, abs(signed_acc)
            
            # For a complete orbit, start and end headings should be similar
            start_hdg = points_slice[0].track
            end_hdg = points_slice[-1].track
            if start_hdg is None or end_hdg is None:
                return False, abs(signed_acc)
            
            hdg_diff = abs(((end_hdg - start_hdg + 540) % 360) - 180)
            
            # If headings are within 90° of each other, it's likely a complete orbit
            # (allowing some tolerance for wind/course corrections)
            if hdg_diff <= 90:
                return True, abs(signed_acc)
            
            # Also check for multiple orbits (720°+)
            if abs(signed_acc) >= 640:  # Nearly 2 full orbits
                return True, abs(signed_acc)
            
            return False, abs(signed_acc)

        def path_len(points_slice):
            total = 0.0
            for a, b in zip(points_slice, points_slice[1:]):
                total += haversine_nm(a.lat, a.lon, b.lat, b.lon)
            return total

        event_added = False
        for i in range(len(points)):
            start = points[i]
            if (start.gspeed or 0) < SPEED_FALLBACK or (start.alt or 0) < ALT_FALLBACK:
                continue
            for j in range(i + 2, len(points)):
                end = points[j]
                duration = end.timestamp - start.timestamp
                if duration < MIN_DURATION:
                    continue
                if duration > MAX_DURATION:
                    break
                if (end.gspeed or 0) < SPEED_FALLBACK or (end.alt or 0) < ALT_FALLBACK:
                    continue
                slice_pts = points[i : j + 1]
                
                # NEW: Check for GPS glitches/jamming in this segment
                if has_gps_glitches(slice_pts):
                    continue  # Skip segments with GPS anomalies
                
                disp = haversine_nm(start.lat, start.lon, end.lat, end.lon)
                path = path_len(slice_pts)
                if disp <= 0:
                    continue
                ratio = path / disp
                if ratio < MIN_PATH_DISP_RATIO:
                    continue
                
                # Check if this is a complete orbit using signed heading accumulation
                is_orbit, signed_acc = is_complete_orbit(slice_pts)
                
                # ADDITIONAL: Check if average speed is physically reasonable
                # If the aircraft traveled 80+ nm in 70 seconds, that's impossible
                avg_speed_kts = (path / duration) * 3600
                if avg_speed_kts > 650:  # Max realistic speed for any commercial aircraft
                    continue  # Impossible speed - GPS glitches
                
                # ADDITIONAL: Check if displacement is too large for a true holding pattern
                # Real holding patterns keep aircraft within 10 nm of a point
                # If displacement > 10 nm, it's likely a curved route or glitchy track, not a hold
                if disp > 10.0:
                    continue  # Not a true holding pattern - displacement too large
                
                if not is_orbit:
                    continue  # Not a complete orbit - skip
                
                # Sanity check: reject impossible turn rates (GPS glitches)
                # A realistic aircraft can turn at most ~6 deg/sec sustained
                avg_turn_rate = signed_acc / duration if duration > 0 else 999
                if avg_turn_rate > 6.0:
                    continue  # Impossible turn rate - likely GPS glitches
                
                # Additional sanity check: for a holding pattern of this duration,
                # the path length should be reasonable (not excessive from GPS jumping)
                # Expected max path for a real hold: ~300 kts * duration / 3600 * 1.5 (buffer)
                expected_max_path = (300.0 * duration / 3600.0) * 1.5
                if path > expected_max_path:
                    continue  # Path too long for the duration - likely GPS glitches

                # Additional check: for a true holding pattern, the aircraft should
                # have meaningful displacement (race-track pattern is ~4-6nm)
                is_true_360_turn = ratio >= 2.0 and disp >= 3.0
                
                # Suppress if near an airport (within 6 nm) - normal maneuvering
                nearest_ap, dist_ap = _nearest_airport(end)
                if dist_ap is not None and dist_ap < 6.0:
                    continue
                
                # Also suppress if START point is near airport (departure turns)
                # Departures often involve large turns that move aircraft >6nm from airport
                nearest_ap_start, dist_ap_start = _nearest_airport(start)
                if dist_ap_start is not None and dist_ap_start < 6.0:
                    continue
                
                # For non-360 patterns (just curved routes), suppress if on learned path
                if not is_true_360_turn:
                    polygons = _get_learned_polygons()
                    is_suppressed = False
                    latlon = (end.lat, end.lon)
                    for poly in polygons:
                        if is_point_in_polygon(latlon, poly):
                            is_suppressed = True
                            break
                    if is_suppressed:
                        continue
                    
                    # Check if on known turn zone, SID, or STAR
                    if _is_on_known_procedure(end.lat, end.lon):
                        continue

                events.append({
                    "type": "holding_pattern",
                    "timestamp": end.timestamp,
                    "start_ts": start.timestamp,
                    "end_ts": end.timestamp,
                    "duration_s": duration,
                    "cumulative_turn_deg": round(signed_acc, 2),
                    "pattern": "360_turn_fallback",
                    "path_nm": round(path, 2),
                    "disp_nm": round(disp, 2),
                    "path_disp_ratio": round(ratio, 2),
                })
                event_added = True
                break
            if event_added:
                break

    matched = bool(events)
    summary = "Abrupt heading change or holding pattern observed" if matched else "Heading profile nominal"
    return RuleResult(3, matched, summary, {"events": events})


def _rule_dangerous_proximity(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    if not points:
        return RuleResult(4, False, "No track data", {})

    # If we don't have a repository, we can't check for other flights
    if ctx.repository is None:
        return RuleResult(4, False, "Skipped: No flight database available", {})

    events = []

    # Pull candidate points
    start = points[0].timestamp - PROXIMITY_TIME_WINDOW
    end = points[-1].timestamp + PROXIMITY_TIME_WINDOW
    
    try:
        nearby_points = [
            p for p in ctx.repository.fetch_points_between(start, end)
            if p.flight_id != ctx.track.flight_id
        ]
    except AttributeError:
         return RuleResult(4, False, "Skipped: Repository error", {})

    for i, point in enumerate(points):
        # Physics check: skip impossible points (GPS glitches)
        if is_impossible_point(points, i):
            continue

        if point.alt < 100:
            continue

        # Skip points near airports (within ~2 miles) - normal traffic patterns
        if PROXIMITY_AIRPORT_EXCLUSION_NM > 0:
            _, airport_dist = _nearest_airport(point)
            if airport_dist <= PROXIMITY_AIRPORT_EXCLUSION_NM:
                continue

        # --- find closest timestamp for each other flight ---
        candidates = [
            other for other in nearby_points
            if abs(other.timestamp - point.timestamp) <= 5   # STRICT time sync
        ]

        for other in candidates:
            if other.alt < 5000:
                continue
            
            # Require both aircraft to be above 3000 ft to avoid ground proximity alerts
            if point.alt < 5000 or other.alt < 5000:
                continue

            dist = haversine_nm(point.lat, point.lon, other.lat, other.lon)
            alt_diff = abs(point.alt - other.alt)

            # impossible values → skip
            if dist < 0.5 and alt_diff < 200 and point.alt < 1000:
                continue

            # heading sanity (optional)
            # if point.track is not None and other.track is not None:
            #     if abs(point.track - other.track) > 130:
            #         continue

            if dist <= PROXIMITY_DISTANCE_NM and alt_diff <= PROXIMITY_ALTITUDE_FT:
                events.append({
                    "timestamp": point.timestamp,
                    "other_flight": other.flight_id,
                    "other_callsign": other.callsign,
                    "distance_nm": round(dist, 2),
                    "altitude_diff_ft": round(alt_diff, 1),
                })
                break

    matched = bool(events)
    summary = "Proximity alert triggered" if matched else "No proximity conflicts"
    return RuleResult(4, matched, summary, {"events": events})


RUNWAY_HEADINGS = {
    "LCRA": [100, 280],   # RAF Akrotiri
    "ALJAWZAH": [135, 315],
    "HEGR": [160, 340],   # El Gora Airport
    "LLBG": [76, 256],    # Ben Gurion
    "LLHA": [155, 335],   # Haifa
    "LLER": [9, 189],     # Ramon Intl
    "LLSD": [140, 320],   # Sde Dov
    "LLBS": [143, 323],   # Beersheba
    "LLET": [86, 266],    # Eilat
    "LLOV": [30, 210],    # Ovda
    "LLNV": [100, 280],   # Nevatim AFB
    "LLMG": [80, 260],    # Megiddo
    "LLHZ": [160, 340],   # Herzliya
    "OLBA": [155, 335],   # Beirut
    "OLKA": [90, 270],    # Rayak AB
    "OJAI": [75, 255],    # Queen Alia Intl
    "OJAM": [61, 241],    # Marka Intl
    "OJAQ": [20, 200],    # Aqaba
    "OJMF": [150, 330],   # Mafraq AB
    "OJJR": [113, 293],   # Jerash
    "OJMN": [142, 322],   # Ma'an
    "OSDI": [50, 230],    # Damascus Intl
    "OSKL": [75, 255],    # Al Qusayr
    "OSAP": [35, 215],    # An Nasiriya AB
}

def _heading_diff(h1, h2):
    """Smallest circular difference between two headings."""
    diff = abs(h1 - h2) % 360
    return diff if diff <= 180 else 360 - diff

def _is_runway_aligned(point, airport_code, tolerance=30):
    """Check if heading is aligned with any runway direction."""
    if airport_code not in RUNWAY_HEADINGS:
        return True  # fallback, don't block detection

    for rh in RUNWAY_HEADINGS[airport_code]:
        if _heading_diff(point.track, rh) <= tolerance:
            return True
    return False


def _rule_go_around(ctx: RuleContext) -> RuleResult:
    events = []
    all_points = ctx.track.sorted_points()

    for airport in AIRPORTS:
        segments = _points_near_airport(all_points, airport, GO_AROUND_RADIUS_NM)
        if len(segments) < 3:
            continue

        # Sort segments by timestamp for neighbor checking (should be sorted already, but ensure it)
        segments_by_time = sorted(segments, key=lambda p: p.timestamp)
        
        # Sort by altitude to find candidates for lowest
        sorted_by_alt = sorted(segments, key=lambda p: p.alt)
        
        MAX_VS_FT_SEC = 200.0  # ~12000 fpm, filter impossible vertical moves
        elevation = airport.elevation_ft or 0
        LOW_ALT_BUFFER_FT = 150.0

        # Find the go-around low point: lowest non-glitched point that has a climb after it
        for candidate in sorted_by_alt:
            # Physics check using full point list
            try:
                full_idx = all_points.index(candidate)
                if is_impossible_point(all_points, full_idx):
                    continue  # Skip this glitched point
            except ValueError:
                pass  # Candidate not in main list, continue with existing checks
            
            idx = segments_by_time.index(candidate)
            is_glitch = False
            
            # Check previous (descent into point)
            if idx > 0:
                prev = segments_by_time[idx - 1]
                dt = candidate.timestamp - prev.timestamp
                dy = abs(candidate.alt - prev.alt)
                if dt > 0 and (dy / dt) > MAX_VS_FT_SEC:
                    is_glitch = True
            
            # Check next (climb out of point)
            if not is_glitch and idx < len(segments_by_time) - 1:
                next_p = segments_by_time[idx + 1]
                dt = next_p.timestamp - candidate.timestamp
                dy = abs(next_p.alt - candidate.alt)
                if dt > 0 and (dy / dt) > MAX_VS_FT_SEC:
                    is_glitch = True
            
            if is_glitch:
                continue

            # Check AGL sanity
            agl = candidate.alt - elevation
            if agl < -200:
                continue  # Below airport - data error
            if agl > GO_AROUND_LOW_ALT_FT + LOW_ALT_BUFFER_FT:
                continue  # Too high for go-around threshold
            
            # Check runway alignment
            if not _is_runway_aligned(candidate, airport.code):
                continue
            
            # Check for descent before (aircraft came from higher altitude)
            before_low = [p for p in segments if p.timestamp < candidate.timestamp]
            if not before_low:
                continue
            descent_amount = max(p.alt for p in before_low) - candidate.alt
            if descent_amount < 300:
                continue
            
            # Check for climb after (go-around recovery)
            after_low = [p for p in segments if p.timestamp > candidate.timestamp]
            if not after_low:
                continue
            max_climb = max(p.alt for p in after_low) - candidate.alt
            if max_climb < GO_AROUND_RECOVERY_FT:
                continue
            
            # All checks passed - this is a go-around!
            events.append(
                {
                    "airport": airport.code,
                    "timestamp": candidate.timestamp,
                    "min_alt_ft": round(candidate.alt, 1),
                    "recovered_ft": round(max_climb, 1),
                    "descent_into_low_ft": round(descent_amount, 1),
                    "aligned_with_runway": True
                }
            )
            break  # Only report once per airport

    matched = bool(events)
    summary = "Go-around detected" if matched else "No go-around patterns"
    return RuleResult(6, matched, summary, {"events": events})


def _rule_takeoff_return(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    if len(points) < 4:
        return RuleResult(7, False, "Insufficient points", {})

    origin_airport, origin_dist = _nearest_airport(points[0])
    if origin_airport is None or origin_dist > RETURN_NEAR_AIRPORT_NM:
        return RuleResult(7, False, "Origin airport unknown", {})

    origin_elev = (origin_airport.elevation_ft or 0)

    takeoff_point = next((p for p in points if (p.alt or 0) >= origin_elev + RETURN_TAKEOFF_ALT_FT), None)
    if takeoff_point is None:
        return RuleResult(7, False, "Flight never departed", {})

    # Require that the aircraft actually traveled outbound before considering a return.
    max_outbound_nm = max(
        haversine_nm(p.lat, p.lon, origin_airport.lat, origin_airport.lon)
        for p in points
        if p.timestamp >= takeoff_point.timestamp
    )
    if max_outbound_nm < RETURN_MIN_OUTBOUND_NM:
        return RuleResult(7, False, "No meaningful outbound leg", {"max_outbound_nm": max_outbound_nm})

    for i, point in enumerate(points):
        # Physics check: skip impossible points (GPS glitches)
        if is_impossible_point(points, i):
            continue

        if point.timestamp <= takeoff_point.timestamp:
            continue
        distance_home = haversine_nm(point.lat, point.lon, origin_airport.lat, origin_airport.lon)
        
        # Check landing (using AGL)
        if (point.alt or 0) < origin_elev + RETURN_LANDING_ALT_FT and distance_home <= RETURN_NEAR_AIRPORT_NM:
            dt = point.timestamp - takeoff_point.timestamp
            if dt <= RETURN_TIME_LIMIT_SECONDS and dt >= RETURN_MIN_ELAPSED_SECONDS:
                info = {
                    "airport": origin_airport.code,
                    "takeoff_ts": takeoff_point.timestamp,
                    "landing_ts": point.timestamp,
                    "elapsed_s": dt,
                    "max_outbound_nm": max_outbound_nm,
                }
                return RuleResult(7, True, "Return-to-field detected", info)
    return RuleResult(7, False, "No immediate return detected", {})


def _rule_diversion(ctx: RuleContext) -> RuleResult:
    metadata = ctx.metadata
    if metadata is None or metadata.planned_destination is None:
        return RuleResult(8, False, "No planned destination provided", {})

    planned = AIRPORT_BY_CODE.get(metadata.planned_destination.upper())
    if planned is None:
        return RuleResult(8, False, "Planned destination not in airport list", {})

    last_point = ctx.track.sorted_points()[-1] if ctx.track.points else None
    if last_point is None:
        return RuleResult(8, False, "No track data", {})

    actual_airport, actual_dist = _nearest_airport(last_point)
    if actual_airport is None or actual_dist > DIVERSION_NEAR_AIRPORT_NM:
        return RuleResult(8, True, "Flight ended away from any known airport", {"distance_to_airport_nm": actual_dist})

    matched = actual_airport.code != planned.code
    summary = "Flight diverted to alternate airport" if matched else "Flight landed at planned destination"
    details = {
        "planned": planned.code,
        "actual": actual_airport.code,
        "distance_nm": round(actual_dist, 2),
    }
    return RuleResult(8, matched, summary, details)


def _rule_low_altitude(ctx: RuleContext) -> RuleResult:
    """
    Detect flights operating below minimum safe altitude outside of airport zones.
    
    This rule identifies aircraft flying dangerously low, which could indicate:
    - Terrain collision risk
    - Deliberate attempt to avoid radar detection
    - Navigation system failure
    - Pilot disorientation
    
    The rule includes extensive filtering to avoid false positives from:
    - Normal takeoff/landing operations near airports
    - Data glitches (impossible points filtered using physics checks)
    - Approach patterns (descending towards airports within 40 NM)
    - Single-point altitude "flickers" (sensor errors)
    
    Returns events where aircraft flew below LOW_ALTITUDE_THRESHOLD_FT (default 800ft)
    outside of protected airport zones.
    """
    points = ctx.track.sorted_points()
    events = []
    last_alt = None
    last_ts = None

    for i, p in enumerate(points):
        alt = p.alt or 0.0
        speed = p.gspeed or 0.0
        vs = p.vspeed or 0.0

        nearest, dist = _nearest_airport(p)

        # -----------------------
        # 0. PHYSICS CHECK: Skip impossible points (GPS glitches, ADS-B errors)
        # -----------------------
        # Before doing ANY anomaly detection, verify this point is physically plausible.
        # Impossible points (teleportation, impossible turns, impossible climb rates)
        # should never trigger anomalies as they're data corruption, not real events.
        if is_impossible_point(points, i, speed_buffer=1.5, max_turn_rate_deg_s=8.0, max_vertical_speed_ft_s=200.0):
            last_alt = alt
            last_ts = p.timestamp
            continue

        # -----------------------
        # 1. Skip ascending / climb-out
        # -----------------------

        # A. If climbing > 300 ft/min → normal ascent
        if vs > 300:
            last_alt = alt
            last_ts = p.timestamp
            continue

        # B. If was just on ground (<50 ft) in last 60 seconds → skip
        if last_alt is not None and last_alt < 50:
            dt = p.timestamp - last_ts
            if dt < 60:   # 1 minute from takeoff
                last_alt = alt
                last_ts = p.timestamp
                continue

        # C. If within 25 NM of airport and climbing → normal
        if nearest and dist < 25 and vs > 0:
            last_alt = alt
            last_ts = p.timestamp
            continue

        # -----------------------
        # HARD SANITY CHECKS
        # -----------------------

        if alt < 200 and (dist is None or dist > 15):
            last_alt = alt
            last_ts = p.timestamp
            continue

        if alt < 800 and speed > 200 and (dist is None or dist > 10):
            last_alt = alt
            last_ts = p.timestamp
            continue

        # Sudden impossible descent
        if last_alt is not None and last_ts is not None:
            dt = p.timestamp - last_ts
            if dt > 0:
                rate = (last_alt - alt) / dt
            else:
                rate = 0.0
            
            if rate > 100 and (dist is None or dist > 10):
                last_alt = alt
                last_ts = p.timestamp
                continue

        # -----------------------
        # LOW ALTITUDE CONFIRMATION
        # -----------------------

        if alt < LOW_ALTITUDE_THRESHOLD_FT:
            idx = points.index(p)
            if idx + 1 < len(points):
                if (points[idx + 1].alt or 0.0) >= LOW_ALTITUDE_THRESHOLD_FT:
                    # single flicker
                    last_alt = alt
                    last_ts = p.timestamp
                    continue

            # Check standard airport radius
            if nearest and dist <= LOW_ALTITUDE_AIRPORT_RADIUS_NM:
                last_alt = alt
                last_ts = p.timestamp
                continue

            # -----------------------
            # APPROACH PATTERN DETECTION:
            # If descending towards an airport, use larger radius (35-40 NM)
            # -----------------------
            if nearest and dist and dist <= 40 and vs < 0:  # Descending
                # Check if heading is roughly towards the airport
                if p.track is not None:
                    bearing_to_airport = initial_bearing_deg(p.lat, p.lon, nearest.lat, nearest.lon)
                    heading_diff = _heading_diff(bearing_to_airport, p.track)
                    
                    # Allow heading within 45 degrees of airport direction
                    if abs(heading_diff) <= 45:
                        # Check if in a continuous descent pattern
                        # Look back at last few points to confirm descent pattern
                        descent_confirmed = False
                        
                        # Check if altitude is decreasing (descent pattern)
                        if idx >= 1:
                            prev_alt_1 = points[idx - 1].alt or 0.0
                            if prev_alt_1 > alt:  # Altitude is decreasing
                                descent_confirmed = True
                                # Check one more point back if available
                                if idx >= 2:
                                    prev_alt_2 = points[idx - 2].alt or 0.0
                                    if prev_alt_2 > prev_alt_1:  # Continued descent
                                        descent_confirmed = True
                        
                        # Also check if distance to airport is decreasing
                        if idx > 0:
                            prev_point = points[idx - 1]
                            prev_dist = haversine_nm(prev_point.lat, prev_point.lon, nearest.lat, nearest.lon)
                            if dist < prev_dist:  # Getting closer to airport
                                descent_confirmed = True
                        
                        # Speed check: approach speeds are typically 100-180 kts
                        # For descending flights near airports, allow legitimate approach patterns
                        if 90 <= speed <= 200 and descent_confirmed:
                            # This looks like a legitimate approach pattern
                            last_alt = alt
                            last_ts = p.timestamp
                            continue

            # -----------------------
            # REAL EVENT
            # -----------------------
            events.append({
                "timestamp": p.timestamp,
                "alt_ft": round(alt, 1),
                "speed_kts": speed,
                "distance_to_airport_nm": round(dist or -1, 1),
                "vspeed_fpm": vs,
            })

        last_alt = alt
        last_ts = p.timestamp

    matched = bool(events)
    summary = "Low altitude detected outside protected zones" if matched else "Altitude remained above minima"
    return RuleResult(9, matched, summary, {"events": events})


def _rule_signal_loss(ctx: RuleContext) -> RuleResult:
    points = ctx.track.sorted_points()
    if len(points) < 2:
        return RuleResult(10, False, "Insufficient points", {})

    gaps = []
    prev = points[0]
    for curr in points[1:]:
        dt = curr.timestamp - prev.timestamp
        
        # Check if on ground (AGL < 300)
        # We need nearest airport to know ground level
        prev_airport, prev_dist = _nearest_airport(prev)
        curr_airport, curr_dist = _nearest_airport(curr)
        
        prev_elev = (prev_airport.elevation_ft if prev_airport and prev_dist < 10 else 0) or 0
        curr_elev = (curr_airport.elevation_ft if curr_airport and curr_dist < 10 else 0) or 0
        
        prev_agl = (prev.alt or 0) - prev_elev
        curr_agl = (curr.alt or 0) - curr_elev
        
        if prev_agl < 300 or curr_agl < 300:
            prev = curr
            continue
            
        if dt >= SIGNAL_GAP_SECONDS:
            gaps.append({"start_ts": prev.timestamp, "end_ts": curr.timestamp, "gap_s": dt})
        prev = curr

    matched = len(gaps) >= SIGNAL_REPEAT_COUNT
    return RuleResult(10, matched, "Signal loss" if matched else "Nominal", {"gaps": gaps})


def _rule_unplanned_israel_landing(ctx: RuleContext) -> RuleResult:
    metadata = ctx.metadata

    if metadata is None or not metadata.planned_destination:
        return RuleResult(12, False, "Missing planned destination", {})

    planned = metadata.planned_destination.upper()

    points = ctx.track.sorted_points()
    if not points:
        return RuleResult(12, False, "No track data", {})

    last_point = points[-1]

    # Find actual nearest airport (landing airport)
    actual_airport, actual_dist = _nearest_airport(last_point)

    if actual_airport is None or actual_dist > UNPLANNED_LANDING_RADIUS_NM:
        return RuleResult(12, False, "Flight did not land at a known airport", {})

    actual = actual_airport.code

    # -----------------------------
    # Core Logic:
    # landed somewhere different than the plan → anomaly
    # -----------------------------
    if planned != actual:
        return RuleResult(
            12,
            True,
            f"Flight landed at {actual} instead of planned {planned}",
            {
                "planned": planned,
                "actual": actual,
                "distance_nm": round(actual_dist, 2),
                "type": "wrong_landing_airport",
            }
        )

    # -----------------------------
    # Normal
    # -----------------------------
    return RuleResult(
        12,
        False,
        "Flight landed at planned destination",
        {
            "planned": planned,
            "actual": actual,
            "distance_nm": round(actual_dist, 2),
        }
    )


def _check_point_in_tubes(
    point: TrackPoint, 
    tubes: List[Dict[str, Any]],
    altitude_tolerance_ft: Optional[float] = None,
    lateral_tolerance_nm: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Check if a point is inside any of the provided tubes (or within tolerance).
    
    Args:
        point: TrackPoint to check
        tubes: List of tube dictionaries
        altitude_tolerance_ft: Altitude tolerance in feet (uses config if None)
        lateral_tolerance_nm: Additional lateral tolerance around tube boundary in NM (uses config if None)
        
    Returns:
        Tube dict if point is inside (or close enough), None otherwise
    """
    # Use config values if not specified
    if altitude_tolerance_ft is None:
        altitude_tolerance_ft = TUBE_ALTITUDE_TOLERANCE_FT
    if lateral_tolerance_nm is None:
        lateral_tolerance_nm = TUBE_LATERAL_TOLERANCE_NM
    
    point_alt = point.alt or 0.0
    
    for tube in tubes:
        # Check altitude range first (fast filter)
        min_alt = tube.get("min_alt_ft", 0.0)
        max_alt = tube.get("max_alt_ft", 50000.0)
        
        if not (min_alt - altitude_tolerance_ft <= point_alt <= max_alt + altitude_tolerance_ft):
            continue
        
        # Check if point is inside polygon (or within tolerance)
        geometry = tube.get("geometry_tuples", [])
        if len(geometry) < 3:
            continue
        
        # First check if point is directly inside
        if is_point_in_polygon((point.lat, point.lon), geometry):
            return tube
        
        # If not inside, check if it's within the lateral tolerance
        # Find minimum distance from point to tube boundary
        min_dist_nm = float('inf')
        for i in range(len(geometry)):
            p1 = geometry[i]
            p2 = geometry[(i + 1) % len(geometry)]
            
            # Calculate distance from point to line segment
            dist = _distance_point_to_segment(
                point.lat, point.lon,
                p1[0], p1[1], p2[0], p2[1]
            )
            min_dist_nm = min(min_dist_nm, dist)
        
        # If point is within tolerance distance from boundary, consider it "on path"
        if min_dist_nm <= lateral_tolerance_nm:
            return tube
    
    return None


def _distance_point_to_segment(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float
) -> float:
    """
    Calculate minimum distance from point (px, py) to line segment (x1,y1)-(x2,y2).
    Uses proper geodesic distance calculation.
    Returns distance in nautical miles.
    
    For geographic coordinates, we sample points along the segment and find
    the minimum distance, which is accurate enough for our purposes.
    """
    # If segment is a point
    if abs(x1 - x2) < 1e-7 and abs(y1 - y2) < 1e-7:
        return haversine_nm(px, py, x1, y1)
    
    # Check distance to endpoints
    dist_to_start = haversine_nm(px, py, x1, y1)
    dist_to_end = haversine_nm(px, py, x2, y2)
    min_dist = min(dist_to_start, dist_to_end)
    
    # Sample 10 points along the segment and find minimum distance
    # This is a simple but effective approximation for geodesic distance to segment
    for i in range(1, 10):
        t = i / 10.0
        sample_x = x1 + t * (x2 - x1)
        sample_y = y1 + t * (y2 - y1)
        dist = haversine_nm(px, py, sample_x, sample_y)
        min_dist = min(min_dist, dist)
    
    return min_dist


def _rule_off_course(ctx: RuleContext) -> RuleResult:
    """
    Path adherence using tubes (3D polygon corridors) matched by O/D pair.
    - First tries to match flight to O/D-specific tubes for polygon-based checking
    - Falls back to path-based distance checking if no tubes match
    - Falls back to all paths if no O/D match found
    - On path if inside tube polygon (with altitude check).
    - Wrong region if entering a low-activity heatmap cell.
    - Emerging detector buckets far-off trajectories.
    """

    points = ctx.track.sorted_points()
    if not points:
        return RuleResult(11, False, "No track data", {})

    # Get origin and destination from metadata - don't try to calculate it
    origin = None
    destination = None
    
    if ctx.metadata:
        origin = ctx.metadata.origin
        destination = ctx.metadata.planned_destination
    
    # If no O/D in metadata, skip deviation check
    if not origin and not destination:
        return RuleResult(11, False, "No origin/destination in metadata - cannot check deviation", {})
    
    # Try to use tubes first (more accurate than centerline-based paths)
    all_tubes = _load_learned_tubes()
    tubes, used_tube_od_filter = _get_tubes_for_od(origin, destination, all_tubes) if all_tubes else ([], False)
    
    # Fallback to paths if no tubes available
    all_paths = _get_paths()
    if not all_paths and not tubes:
        return RuleResult(11, False, "No path library or tubes loaded", {})
    
    paths, used_path_od_filter = _get_paths_for_od(origin, destination, all_paths) if all_paths else ([], False)
    
    # Track which method we're using
    using_tubes = len(tubes) > 0
    used_od_filter = used_tube_od_filter if using_tubes else used_path_od_filter

    on_path: List[Dict[str, Any]] = []
    off_path: List[Dict[str, Any]] = []
    wrong_region: List[Dict[str, Any]] = []
    far_points: List[TrackPoint] = []
    assignments: Dict[str, int] = defaultdict(int)

    for idx, p in enumerate(points):
        # Physics check: skip impossible points (GPS glitches)
        if is_impossible_point(points, idx):
            continue

        if idx > 0 and is_bad_segment(points[idx - 1], p):
            continue
        if (p.alt or 0) <= 9000:
            continue

        # Try tube-based checking first (faster and more accurate)
        if using_tubes:
            # Use configured lateral tolerance to allow for natural flight path variation
            matched_tube = _check_point_in_tubes(p, tubes)
            if matched_tube:
                tube_id = matched_tube.get("id", "unknown")
                assignments[tube_id] += 1
                on_path.append(
                    {
                        "timestamp": p.timestamp,
                        "path_id": tube_id,
                        "method": "tube",
                        "distance_nm": 0.0,  # Inside polygon or within tolerance
                        "position": 0.5,  # N/A for tubes
                        "type": "tube",
                    }
                )
                continue
        
        # Fallback to path-based checking (original method)
        best: Optional[Tuple[str, float, float, float, str]] = None  # id, dist, pos, width, type
        for path in paths:
            dist_nm, pos = _distance_to_path(p, path)
            width = float(path.get("width_nm", DEFAULT_PATH_WIDTH_NM))
            if best is None or dist_nm < best[1]:
                best = (path.get("id", "unknown"), dist_nm, pos, width, path.get("type", "primary"))

        if best is None:
            # Point not in tubes and no paths to check
            off_record = {
                "timestamp": p.timestamp,
                "distance_nm": 999.0,
                "lat": p.lat,
                "lon": p.lon,
            }
            off_path.append(off_record)
            far_points.append(p)
            if not _is_in_flightable_region(p):
                wrong_region.append(off_record)
            continue

        path_id, dist_nm, pos, width_nm, path_type = best
        if dist_nm <= width_nm:
            assignments[path_id] += 1
            on_path.append(
                {
                    "timestamp": p.timestamp,
                    "path_id": path_id,
                    "method": "path",
                    "distance_nm": round(dist_nm, 2),
                    "position": round(pos, 3),
                    "type": path_type,
                }
            )
        else:
            off_record = {
                "timestamp": p.timestamp,
                "distance_nm": round(dist_nm, 2),
                "lat": p.lat,
                "lon": p.lon,
            }
            off_path.append(off_record)
            if dist_nm >= EMERGING_DISTANCE_NM:
                far_points.append(p)
            if not _is_in_flightable_region(p):
                wrong_region.append(off_record)

    promoted = None
    if far_points:
        promoted = _update_emerging_buckets(ctx, far_points)

    matched = len(off_path) >= MIN_OFF_COURSE_POINTS or len(wrong_region) > 0
    summary = "Flight deviated from known paths" if matched else "Flight stayed within known corridors"
    if wrong_region:
        summary = "Entered low-activity region"

    details = {
        "on_path_points": len(on_path),
        "off_path_points": len(off_path),
        "wrong_region_points": len(wrong_region),
        "assignments": dict(assignments),
        "samples": {"on_path": on_path[:50], "off_path": off_path[:50], "wrong_region": wrong_region[:50]},
        "emerging_promoted": promoted["id"] if promoted else None,
        "threshold_points": MIN_OFF_COURSE_POINTS,
        "detected_origin": origin,
        "detected_destination": destination,
        "used_od_filter": used_od_filter,
        "using_tubes": using_tubes,
        "tubes_checked": len(tubes) if using_tubes else 0,
        "paths_checked": len(paths),
    }

    return RuleResult(11, matched, summary, details)


def _points_near_airport(points: Sequence[TrackPoint], airport: Airport, radius_nm: float) -> List[TrackPoint]:
    return [p for p in points if haversine_nm(p.lat, p.lon, airport.lat, airport.lon) <= radius_nm]


def _nearest_airport(point: TrackPoint) -> Tuple[Optional[Airport], float]:
    best_airport: Optional[Airport] = None
    best_distance: float = float('inf')
    for airport in AIRPORTS:
        distance = haversine_nm(point.lat, point.lon, airport.lat, airport.lon)
        if distance < best_distance:
            best_distance = distance
            best_airport = airport
    return best_airport, best_distance


def _pairwise(points: Sequence[TrackPoint]):
    iterator = iter(points)
    prev = next(iterator, None)
    for current in iterator:
        if prev is not None:
            yield prev, current
        prev = current


def has_point_above_altitude(track: FlightTrack, altitude_ft: float = 5000.0) -> bool:
    """
    Check if a flight has at least one point above the specified altitude.
    """
    c = 0
    for point in track.points:
        if point.alt > altitude_ft:
            if c > 4:
                return True
            else:
                c += 1
    return False


def _rule_military_aircraft(ctx: RuleContext) -> RuleResult:
    """
    Detect military aircraft based on callsign and registration patterns.
    
    This rule identifies military aircraft using:
    - Callsign prefixes (e.g., RCH, NAVY, GAF, BAF, etc.)
    - Aircraft registration patterns (e.g., ZZ*, 4XA*, MM*, etc.)
    
    Returns a match with organization details if military is detected.
    """
    points = ctx.track.sorted_points()
    if not points:
        return RuleResult(13, False, "No track data", {})
    
    # Extract callsign and registration from track points
    callsign = None
    aircraft_registration = None
    category = None
    
    # Try to get metadata from context if available
    if ctx.metadata:
        # Note: metadata might not have these specific fields depending on how it's populated
        # but we check just in case custom fields were added
        if hasattr(ctx.metadata, 'aircraft_registration'):
            aircraft_registration = ctx.metadata.aircraft_registration
        if hasattr(ctx.metadata, 'category'):
            category = ctx.metadata.category
            
    for p in points:
        if p.callsign and not callsign:
            callsign = p.callsign
        if callsign:
            break
    
    # Check if military using common function
    is_military_flag, military_org_info = is_military(
        callsign=callsign,
        aircraft_registration=aircraft_registration,
        category=category
    )
    
    if is_military_flag:
        summary = f"Military aircraft detected: {military_org_info}"
        details = {
            "callsign": callsign,
            "organization": military_org_info,
            "detection_method": "callsign" if callsign else "registration"
        }
        return RuleResult(13, True, summary, details)
    
    return RuleResult(13, False, "No military identification", {"callsign": callsign})
