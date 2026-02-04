# Research Rerun Routes Documentation

## Overview

New API routes have been added to access the rerun analysis data stored in the `research_old` PostgreSQL schema. These routes mirror the existing `/api/research/*` endpoints but query data from the `research_old` schema instead of the `research` schema.

## New Routes

### 1. Get Rerun Anomalies

**Endpoint:** `GET /api/research_rerun/anomalies`

**Query Parameters:**
- `start_ts` (int, required): Start timestamp for the time range
- `end_ts` (int, required): End timestamp for the time range

**Description:** Fetches anomalies from the research_old database (rerun analysis) within the specified time range.

**Example:**
```
GET /api/research_rerun/anomalies?start_ts=1704067200&end_ts=1706659200
```

**Response:** Array of anomaly objects containing:
- `flight_id`: Unique flight identifier
- `timestamp`: Anomaly detection timestamp
- `is_anomaly`: Boolean indicating if this is an anomaly
- `severity_cnn`: CNN model severity score
- `severity_dense`: Dense model severity score
- `full_report`: Complete anomaly report JSON
- `callsign`: Flight callsign
- `flight_number`: Flight number
- `airline`: Airline name
- `origin_airport`: Origin airport code
- `destination_airport`: Destination airport code
- `aircraft_type`: Aircraft type
- `total_points`: Number of track points

---

### 2. Get Rerun Flight Track

**Endpoint:** `GET /api/research_rerun/track/{flight_id}`

**Path Parameters:**
- `flight_id` (string, required): The flight ID to fetch the track for

**Description:** Fetches the full track for a flight from the research_old schema. Checks both anomalies_tracks and normal_tracks tables.

**Example:**
```
GET /api/research_rerun/track/abc123xyz
```

**Response:**
```json
{
  "flight_id": "abc123xyz",
  "points": [
    {
      "timestamp": 1704067200,
      "lat": 40.7128,
      "lon": -74.0060,
      "alt": 35000,
      "gspeed": 450,
      "vspeed": 0,
      "track": 90,
      "squawk": "1200",
      "callsign": "UAL123",
      "source": "adsb"
    },
    ...
  ]
}
```

---

### 3. Get Rerun Flight Metadata

**Endpoint:** `GET /api/research_rerun/metadata/{flight_id}`

**Path Parameters:**
- `flight_id` (string, required): The flight ID to fetch metadata for

**Description:** Gets comprehensive flight metadata from the research_old schema, including data from flight_metadata table, anomaly_reports, and track tables.

**Example:**
```
GET /api/research_rerun/metadata/abc123xyz
```

**Response:** Object containing all available flight metadata:
- Flight identification (callsign, flight_number, etc.)
- Timestamps (first_seen_ts, last_seen_ts, etc.)
- Location data (start/end coordinates, airports)
- Flight statistics (duration, altitude stats, speed stats)
- Aircraft information (type, model, registration)
- Anomaly data (severity scores, is_anomaly flag)
- Squawk codes and emergency indicators

---

### 4. Get Rerun Flight Callsign

**Endpoint:** `GET /api/research_rerun/callsign/{flight_id}`

**Path Parameters:**
- `flight_id` (string, required): The flight ID to fetch the callsign for

**Description:** Fetches the callsign for a flight from the research_old schema. Tries multiple sources: anomalies_tracks, normal_tracks, and anomaly_reports.

**Example:**
```
GET /api/research_rerun/callsign/abc123xyz
```

**Response:**
```json
{
  "callsign": "UAL123"
}
```

---

## Backend Changes

### Modified Files

1. **`service/pg_provider.py`**
   - Updated `get_research_anomalies()` function to accept a `schema` parameter (defaults to 'research')
   - Now supports querying both `research` and `research_old` schemas

2. **`routes/flights.py`**
   - Added 4 new research_rerun route handlers:
     - `get_research_rerun_anomalies()`
     - `get_research_rerun_track()`
     - `get_research_rerun_flight_metadata()`
     - `get_research_rerun_callsign()`

3. **`api/api.py`**
   - Updated API documentation to include research_rerun routes
   - Added `research_rerun` to the endpoints list in the root endpoint response

## Database Schema

These routes query the `research_old` PostgreSQL schema, which should have the same structure as the `research` schema:

### Tables Used:
- `research_old.anomaly_reports` - Anomaly detection results
- `research_old.flight_metadata` - Flight metadata
- `research_old.anomalies_tracks` - Track points for anomalous flights
- `research_old.normal_tracks` - Track points for normal flights

## Integration with UI

The frontend can now access rerun analysis data by changing the endpoint prefix from `/api/research/` to `/api/research_rerun/`. For example:

**Original (current analysis):**
```javascript
fetch('/api/research/anomalies?start_ts=1704067200&end_ts=1706659200')
```

**Rerun (historical analysis):**
```javascript
fetch('/api/research_rerun/anomalies?start_ts=1704067200&end_ts=1706659200')
```

This allows the UI to show both current and rerun analysis data side-by-side or switch between them.

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200 OK` - Successful request
- `404 Not Found` - Flight/data not found in research_old schema
- `500 Internal Server Error` - Database or processing error

Error responses include a detail message explaining the issue.

## Notes

- The research_rerun routes use the exact same response format as the research routes for easy frontend integration
- All queries are optimized with appropriate indexes (assuming the research_old schema mirrors research schema structure)
- The routes support the same filtering and pagination as the original research routes
