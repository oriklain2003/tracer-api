from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.geodesy import haversine_nm
from core.models import TrackPoint


def _sorted_points(points: Sequence[TrackPoint]) -> List[TrackPoint]:
    """Ensure points are ordered by timestamp."""
    return sorted(points, key=lambda p: p.timestamp)


def resample_track_points(
    points: Sequence[TrackPoint],
    num_samples: int = 80,
) -> Optional[np.ndarray]:
    """
    Interpolate a flight path to a fixed-length sequence of (lat, lon, alt).

    Args:
        points: Original ADS-B samples.
        num_samples: Desired number of waypoints in the normalized path.

    Returns:
        np.ndarray of shape (num_samples, 3) ordered as [lat, lon, alt],
        or None if resampling is not possible (e.g. <2 unique timestamps).
    """
    if len(points) < 2:
        return None

    ordered = _sorted_points(points)
    timestamps = np.array([p.timestamp for p in ordered], dtype=float)
    lats = np.array([p.lat for p in ordered], dtype=float)
    lons = np.array([p.lon for p in ordered], dtype=float)
    alts = np.array([p.alt or 0.0 for p in ordered], dtype=float)

    # Remove duplicate timestamps (cannot interpolate otherwise)
    uniq_ts, uniq_idx = np.unique(timestamps, return_index=True)
    if uniq_ts.size < 2:
        return None

    uniq_lats = lats[uniq_idx]
    uniq_lons = lons[uniq_idx]
    uniq_alts = alts[uniq_idx]

    # Compute cumulative distance in NM
    dists = [0.0]
    for i in range(1, len(uniq_lats)):
        d = haversine_nm(uniq_lats[i-1], uniq_lons[i-1], uniq_lats[i], uniq_lons[i])
        dists.append(d)
    
    cum_dist = np.cumsum(dists)
    total_dist = cum_dist[-1]

    if total_dist == 0.0:
        return None

    target_dist = np.linspace(0.0, total_dist, num_samples)

    interp_lat = np.interp(target_dist, cum_dist, uniq_lats)
    interp_lon = np.interp(target_dist, cum_dist, uniq_lons)
    interp_alt = np.interp(target_dist, cum_dist, uniq_alts)

    return np.stack([interp_lat, interp_lon, interp_alt], axis=1)


def flatten_resampled_path(path: np.ndarray) -> np.ndarray:
    """
    Flatten a resampled path to a 1D vector suitable for clustering.
    """
    return path.reshape(-1)


def mean_path_distance_nm(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """
    Compute the mean great-circle distance between two normalized paths.

    Args:
        path_a: np.ndarray [N, 3]
        path_b: np.ndarray [N, 3]

    Returns:
        Average nautical-mile deviation between corresponding waypoints.
    """
    if path_a.shape != path_b.shape:
        raise ValueError("Paths must have identical shapes for distance comparison")

    total = 0.0
    for (lat_a, lon_a, _), (lat_b, lon_b, _) in zip(path_a, path_b):
        total += haversine_nm(lat_a, lon_a, lat_b, lon_b)
    return total / path_a.shape[0]


def point_to_polyline_distance_nm(
    point: Tuple[float, float],
    polyline: Sequence[Tuple[float, float]],
) -> dict:
    """
    Compute the minimum distance from a point to a polyline and its normalized position.

    Returns:
        {"distance_nm": float, "position": float}
        distance_nm: closest lateral distance to any segment in NM
        position: 0-1 fraction along the path where the closest point lies
    
    OPTIMIZED: Uses vectorized numpy operations for speed.
    """
    if len(polyline) < 2:
        raise ValueError("Polyline must contain at least two points")

    n = len(polyline)
    
    # Convert to numpy arrays
    coords = np.array(polyline, dtype=np.float64)
    
    # Reference latitude for equirectangular projection
    ref_lat = coords[0, 0]
    cos_lat = np.cos(np.radians(ref_lat))
    
    # Project all points to local XY (NM)
    xy = np.empty((n, 2), dtype=np.float64)
    xy[:, 0] = coords[:, 0] * 60.0  # lat -> y in NM
    xy[:, 1] = coords[:, 1] * 60.0 * cos_lat  # lon -> x in NM
    
    # Project the query point
    px = point[0] * 60.0
    py = point[1] * 60.0 * cos_lat
    
    # Segment vectors
    seg_starts = xy[:-1]  # (n-1, 2)
    seg_ends = xy[1:]     # (n-1, 2)
    seg_vec = seg_ends - seg_starts  # (n-1, 2)
    
    # Vector from segment start to query point
    w = np.array([px, py]) - seg_starts  # (n-1, 2)
    
    # Segment lengths squared
    seg_len_sq = np.sum(seg_vec ** 2, axis=1)  # (n-1,)
    
    # Compute t parameter (clamped to [0, 1])
    # t = dot(w, seg_vec) / |seg_vec|^2
    dot_product = np.sum(w * seg_vec, axis=1)  # (n-1,)
    
    # Avoid division by zero for zero-length segments
    with np.errstate(divide='ignore', invalid='ignore'):
        t = dot_product / seg_len_sq
    t = np.nan_to_num(t, nan=0.0)
    t = np.clip(t, 0.0, 1.0)
    
    # Closest point on each segment
    closest_pts = seg_starts + t[:, np.newaxis] * seg_vec  # (n-1, 2)
    
    # Distance from query point to closest point on each segment
    dists = np.sqrt((px - closest_pts[:, 0])**2 + (py - closest_pts[:, 1])**2)  # (n-1,)
    
    # Find minimum
    min_idx = np.argmin(dists)
    min_dist = float(dists[min_idx])
    
    # Compute position along path
    # First compute segment lengths using haversine (more accurate than projected)
    seg_lengths = np.sqrt(seg_len_sq)  # Approximate segment lengths in NM
    total_length = np.sum(seg_lengths) or 1.0
    cum_length = np.sum(seg_lengths[:min_idx]) + t[min_idx] * seg_lengths[min_idx]
    best_pos = float(cum_length / total_length)

    return {"distance_nm": min_dist, "position": best_pos}

