from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from core.models import FlightTrack

EARTH_RADIUS_M = 6_371_000  # meters

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two lat/lon pairs in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def smallest_angle_diff(a: float, b: float) -> float:
    """Return the smallest signed difference between two headings (degrees)."""
    return (a - b + 180) % 360 - 180


@dataclass
class RollingStats:
    """Helper to maintain rolling window statistics."""
    window_seconds: int
    
    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.history: deque = deque()  # Stores (timestamp, value)
        self.sum_val = 0.0
        
    def update(self, timestamp: float, value: float) -> float:
        """Add new value and return current window sum."""
        self.history.append((timestamp, value))
        self.sum_val += value
        
        # Remove old values
        while self.history and (timestamp - self.history[0][0] > self.window_seconds):
            _, old_val = self.history.popleft()
            self.sum_val -= old_val
            
        return self.sum_val

    def average(self) -> float:
        if not self.history:
            return 0.0
        return self.sum_val / len(self.history)


@dataclass
class FeatureExtractor:
    """
    Build per-point engineered features from a `FlightTrack`.
    """

    def extract_flight_features(self, flight: FlightTrack) -> List[Dict[str, float]]:
        points = flight.sorted_points()
        if not points:
            return []

        # Initialize rolling windows
        roll_turn_300 = RollingStats(300)
        roll_alt_60 = RollingStats(60)
        roll_speed_300 = RollingStats(300)

        rows: List[Dict[str, float]] = []
        for idx, point in enumerate(points):
            if idx == 0:
                prev = point
                dt = 1.0
            else:
                prev = points[idx - 1]
                dt = max(1.0, point.timestamp - prev.timestamp)

            # Calculate basic deltas
            distance_m = haversine(prev.lat, prev.lon, point.lat, point.lon)
            
            # Get values with defaults
            alt_curr = point.alt or 0.0
            alt_prev = prev.alt or 0.0
            gspeed_curr = point.gspeed or 0.0
            gspeed_prev = prev.gspeed or 0.0
            vspeed_curr = point.vspeed or 0.0
            vspeed_prev = prev.vspeed or 0.0

            # Calculate rates
            climb_rate = (alt_curr - alt_prev) / dt
            speed_accel = (gspeed_curr - gspeed_prev) / dt
            vert_accel = (vspeed_curr - vspeed_prev) / dt

            # Turn calculations
            turn_deg = 0.0
            if point.track is not None and prev.track is not None:
                turn_deg = smallest_angle_diff(point.track, prev.track)
                turn_rate = turn_deg / dt
            else:
                turn_rate = 0.0

            # Update rolling windows
            cum_turn_300 = roll_turn_300.update(point.timestamp, turn_deg)
            
            # Altitude change over window
            d_alt = alt_curr - alt_prev
            cum_alt_60 = roll_alt_60.update(point.timestamp, d_alt)
            
            # Average speed over window
            roll_speed_300.update(point.timestamp, gspeed_curr) 
            avg_speed_300 = roll_speed_300.average()

            phase = self._infer_phase(alt_curr, vspeed_curr)
            phase_one_hot = [1 if phase == idx else 0 for idx in range(4)]

            rows.append(
                {
                    "flight_id": point.flight_id,
                    "timestamp": point.timestamp,
                    "alt": alt_curr,
                    "gspeed": gspeed_curr,
                    "vspeed": vspeed_curr,
                    
                    # Derived features
                    "distance_step_m": distance_m,
                    "climb_rate": climb_rate,
                    "speed_accel": speed_accel,
                    "vert_accel": vert_accel,
                    "turn_rate": turn_rate,
                    "vspeed_abs": abs(vspeed_curr),
                    
                    # Rolling Window Features (The "Memory")
                    "cum_turn_300": cum_turn_300,      # Total degrees turned in 5 mins
                    "cum_alt_60": cum_alt_60,          # Net altitude change in 1 min
                    "avg_speed_300": avg_speed_300,    # Average speed in 5 mins
                    
                    # Phase flags
                    "phase_ground": phase_one_hot[0],
                    "phase_climb": phase_one_hot[1],
                    "phase_descent": phase_one_hot[2],
                    "phase_cruise": phase_one_hot[3],
                }
            )
        return rows

    @staticmethod
    def _infer_phase(alt_ft: float, vspeed_fpm: float) -> int:
        if alt_ft < 1000:
            return 0  # ground/low
        if vspeed_fpm > 400:
            return 1  # climb
        if vspeed_fpm < -400:
            return 2  # descent
        return 3  # cruise

    @staticmethod
    def feature_columns() -> List[str]:
        return [
            "alt", "gspeed", "vspeed", "distance_step_m", "climb_rate", 
            "speed_accel", "vert_accel", "turn_rate", "vspeed_abs",
            "cum_turn_300", "cum_alt_60", "avg_speed_300",
            "phase_ground", "phase_climb", "phase_descent", "phase_cruise"
        ]

