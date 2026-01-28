from typing import List, Dict, Optional
from core.models import FlightTrack, TrackPoint
from core.db import FlightRepository

class InMemoryRepository:
    """
    A repository adapter that serves flight data from an in-memory dictionary
    of active flight states. Used for real-time proximity checks.
    """
    def __init__(self, active_flights: Dict[str, 'FlightState']):
        # active_flights is a reference to the RealtimeMonitor's state
        # Dict[flight_id, FlightState]
        self.active_flights = active_flights

    def fetch_flight(self, flight_id: str) -> Optional[FlightTrack]:
        if flight_id in self.active_flights:
            return self.active_flights[flight_id].to_flight_track()
        return None

    def fetch_points_between(self, min_ts: int, max_ts: int) -> List[TrackPoint]:
        """
        Fetch all points from all active flights within the time range.
        This is potentially expensive if history is long, but for realtime 
        windows (usually small), it's fast.
        """
        results = []
        for state in self.active_flights.values():
            # Optimization: Binary search could be used here if points are sorted
            # For now, linear scan is acceptable for small windows
            for p in state.points:
                if min_ts <= p.timestamp <= max_ts:
                    results.append(p)
        return results

    def fetch_recent_flights(self, duration_seconds: int = 300) -> List[FlightTrack]:
        """
        Return all flights that have had updates in the last N seconds.
        """
        # In this context, all active flights are "recent" enough to be relevant
        return [s.to_flight_track() for s in self.active_flights.values()]
