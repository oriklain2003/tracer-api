from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional

from .models import FlightTrack, TrackPoint


@dataclass
class DbConfig:
    path: Path
    table: str = "flight_tracks"


class FlightRepository:
    def __init__(self, config: Optional[DbConfig] = None):
        if config is None:
            config = DbConfig(path=Path("last.db"))
        self._config = config

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._config.path))
        try:
            yield conn
        finally:
            conn.close()

    def fetch_flight(self, flight_id: str) -> FlightTrack:
        with self._connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source
                FROM {self._config.table}
                WHERE flight_id = ?
                ORDER BY timestamp ASC
                """,
                (flight_id,),
            )
            points = [self._row_to_point(row) for row in cursor.fetchall()]
            return FlightTrack(flight_id=flight_id, points=points)

    def iter_flights(self, limit: Optional[int] = None, min_points: int = 2) -> Iterable[FlightTrack]:
        with self._connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT flight_id, COUNT(*) as cnt
                FROM {self._config.table}
                GROUP BY flight_id
                HAVING cnt >= ?
                ORDER BY cnt DESC
                {f"LIMIT {int(limit)}" if limit else ""}
                """,
                (min_points,),
            )
            for flight_id, _ in cursor.fetchall():
                yield self.fetch_flight(flight_id)

    def fetch_points_between(self, start_ts: int, end_ts: int) -> List[TrackPoint]:
        with self._connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT flight_id, timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign, source
                FROM {self._config.table}
                WHERE timestamp BETWEEN ? AND ?
                """,
                (start_ts, end_ts),
            )
            return [self._row_to_point(row) for row in cursor.fetchall()]

    def fetch_tracks_in_box(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[FlightTrack]:
        """Fetch all flight tracks that have points within the specified bounding box."""
        flight_ids = self.fetch_flight_ids_in_box(min_lat, max_lat, min_lon, max_lon)
        return [self.fetch_flight(fid) for fid in flight_ids]

    def fetch_flight_ids_in_box(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[str]:
        """Fetch flight IDs that have at least one point within the specified bounding box."""
        with self._connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT DISTINCT flight_id
                FROM {self._config.table}
                WHERE lat BETWEEN ? AND ?
                  AND lon BETWEEN ? AND ?
                """,
                (min_lat, max_lat, min_lon, max_lon),
            )
            return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _row_to_point(row) -> TrackPoint:
        (
            flight_id,
            timestamp,
            lat,
            lon,
            alt,
            gspeed,
            vspeed,
            track,
            squawk,
            callsign,
            source,
        ) = row
        return TrackPoint(
            flight_id=flight_id or "UNKNOWN",
            timestamp=int(timestamp),
            lat=float(lat),
            lon=float(lon),
            alt=float(alt or 0.0),
            gspeed=float(gspeed) if gspeed is not None else None,
            vspeed=float(vspeed) if vspeed is not None else None,
            track=float(track) if track is not None else None,
            squawk=str(squawk) if squawk is not None else None,
            callsign=callsign,
            source=source,
        )
