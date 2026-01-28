from __future__ import annotations

import math
from typing import Tuple, List, Sequence
import numpy as np

EARTH_RADIUS_KM = 6371.0
NM_PER_KM = 0.539957


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c * NM_PER_KM


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.cos(math.radians(lon2 - lon1))
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def cross_track_distance_nm(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    point: Tuple[float, float],
) -> float:
    """
    Compute cross-track distance from the great-circle path connecting origin and destination.
    Uses a spherical Earth approximation, accurate enough for route-deviation heuristics.
    """
    lat1, lon1 = map(math.radians, origin)
    lat2, lon2 = map(math.radians, destination)
    lat3, lon3 = map(math.radians, point)

    dist13 = angular_distance(lat1, lon1, lat3, lon3)
    if dist13 == 0.0:
        return 0.0

    bearing13 = bearing_rad(lat1, lon1, lat3, lon3)
    bearing12 = bearing_rad(lat1, lon1, lat2, lon2)

    sin_xt = math.sin(dist13) * math.sin(bearing13 - bearing12)
    xt_distance_km = math.asin(max(-1.0, min(1.0, sin_xt))) * EARTH_RADIUS_KM
    return abs(xt_distance_km * NM_PER_KM)


def angular_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return math.atan2(y, x)


def frechet_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """
    Discrete Frechet distance between two paths (N, 2) arrays of [lat, lon].
    Uses dynamic programming. Returns distance in NM.
    """
    n = len(path_a)
    m = len(path_b)
    ca = np.ones((n, m)) * -1.0

    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        
        dist = haversine_nm(path_a[i][0], path_a[i][1], path_b[j][0], path_b[j][1])
        
        if i == 0 and j == 0:
            ca[i, j] = dist
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), dist)
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), dist)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), dist)
        else:
            ca[i, j] = float("inf")
            
        return ca[i, j]

    return _c(n - 1, m - 1)


def destination_point(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> Tuple[float, float]:
    """
    Calculate destination point given start point, bearing, and distance.
    """
    R = EARTH_RADIUS_KM
    d = distance_nm / NM_PER_KM  # convert to km
    
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    
    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
                     math.cos(lat1) * math.sin(d / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                             math.cos(d / R) - math.sin(lat1) * math.sin(lat2))
    
    return math.degrees(lat2), math.degrees(lon2)


def create_corridor_polygon(path: Sequence[Tuple[float, float]], radius_nm: float) -> List[Tuple[float, float]]:
    """
    Create a polygon buffer around a path.
    path: List of (lat, lon)
    radius_nm: Buffer radius in NM
    Returns: List of (lat, lon) forming a closed polygon
    """
    if len(path) < 2:
        return []
        
    left_boundary = []
    right_boundary = []
    
    for i in range(len(path)):
        lat, lon = path[i]
        
        # Determine bearing
        if i < len(path) - 1:
            bearing = initial_bearing_deg(lat, lon, path[i+1][0], path[i+1][1])
        else:
            # Use previous bearing for last point
            bearing = initial_bearing_deg(path[i-1][0], path[i-1][1], lat, lon)
            
        # Calculate left and right points
        left_pt = destination_point(lat, lon, bearing - 90, radius_nm)
        right_pt = destination_point(lat, lon, bearing + 90, radius_nm)
        
        left_boundary.append(left_pt)
        right_boundary.append(right_pt)
        
    # Combine to form polygon (Left side -> Right side reversed)
    return left_boundary + right_boundary[::-1] + [left_boundary[0]]


def is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm for point in polygon.
    point: (lat, lon)
    polygon: List of (lat, lon)
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def smooth_polyline(points: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth a polyline using a moving average.
    points: (N, 2) array of lat, lon
    """
    if len(points) < window:
        return points
    
    smoothed = np.zeros_like(points, dtype=float)
    N = len(points)
    half_win = window // 2
    
    for i in range(N):
        start = max(0, i - half_win)
        end = min(N, i + half_win + 1)
        smoothed[i] = np.mean(points[start:end], axis=0)
        
    return smoothed
