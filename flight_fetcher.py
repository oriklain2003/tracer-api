import dataclasses
import json
import os
from datetime import datetime
from dotenv import load_dotenv

from core.models import FlightTrack, TrackPoint

# Load environment variables
load_dotenv()


def get(flight_id="3bc6854c"):
    from fr24sdk.client import Client

    api_token = os.getenv("FR24_API_TOKEN")
    if not api_token:
        raise ValueError("FR24_API_TOKEN environment variable is required")
    
    client = Client(api_token=api_token)
    # emo = client.flight_summary.get_light(flight_datetime_to=datetime.now()  - timedelta(minutes=1) , flight_datetime_from=datetime.now() - timedelta(minutes=15), callsigns=["CYF461"])

    tracks = client.flight_tracks.get(flight_id=[flight_id])
    # Get first FlightTracks object
    flight_data = tracks.model_dump()["data"][0]
    flight_id = flight_data["fr24_id"]
    track_points = flight_data["tracks"]

    # Convert to TrackPoint objects
    points = []
    for tp in track_points:
        # Parse ISO timestamp to unix timestamp
        ts_str = tp["timestamp"].replace("Z", "+00:00")
        ts = int(datetime.fromisoformat(ts_str).timestamp())

        points.append(TrackPoint(
            flight_id=flight_id,
            timestamp=ts,
            lat=float(tp["lat"]),
            lon=float(tp["lon"]),
            alt=float(tp["alt"]),
            gspeed=float(tp["gspeed"]) if tp.get("gspeed") is not None else None,
            vspeed=float(tp["vspeed"]) if tp.get("vspeed") is not None else None,
            track=float(tp["track"]) if tp.get("track") is not None else None,
            squawk=str(tp["squawk"]) if tp.get("squawk") else None,
            callsign=tp.get("callsign"),
            source=tp.get("source"),
        ))

    return FlightTrack(flight_id=flight_id, points=points)


def search_flight_path(callsign: str, start_date: datetime, end_date: datetime):
    """
    Search for a flight by callsign and date range, then fetch its full path.
    Returns a FlightTrack object or None if not found.
    """
    from fr24sdk.client import Client
    
    api_token = os.getenv("FR24_API_TOKEN")
    if not api_token:
        raise ValueError("FR24_API_TOKEN environment variable is required")
    
    client = Client(api_token=api_token)

    # Search for flight summary
    try:
        # Format dates to 'YYYY-MM-DDTHH:MM:SS' as required by the API
        # The API seems to dislike spaces and timezones in the query param
        s_str = start_date.strftime('%Y-%m-%dT%H:%M:%S')
        e_str = end_date.strftime('%Y-%m-%dT%H:%M:%S')

        summary_response = client.flight_summary.get_light(
            flight_datetime_to=e_str,
            flight_datetime_from=s_str,
            callsigns=[callsign]
        )

        data = summary_response.model_dump().get("data", [])
        if not data:
            return None

        # Pick the first one
        first_flight = data[0]
        # Attempt to get ID
        flight_id = first_flight.get("fr24_id") or first_flight.get("id")

        if not flight_id:
            return None

        # Use existing get() to fetch full track
        return get(flight_id)

    except Exception as e:
        print(f"Error searching flight: {e}")
        return None


# Helper: Serialize FlightTrack to JSON
def serialize_flight(flight: FlightTrack) -> str:
    data = {
        "flight_id": flight.flight_id,
        "points": [dataclasses.asdict(p) for p in flight.points]
    }
    return json.dumps(data)


# Helper: Deserialize JSON to FlightTrack
def deserialize_flight(json_str: str) -> FlightTrack:
    data = json.loads(json_str)
    flight_id = data["flight_id"]
    points = []
    for p_dict in data["points"]:
        points.append(TrackPoint(**p_dict))
    return FlightTrack(flight_id=flight_id, points=points)


if __name__ == '__main__':
    # Example usage
    # track = get("3bc6854c")
    pass
