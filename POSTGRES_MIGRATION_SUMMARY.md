# PostgreSQL Migration Summary

## Endpoints Migrated to PostgreSQL

### 1. `/api/live/flights` (GET)
**File:** `routes/flights.py` (line ~797)
**Module:** Flights

**Changes:**
- Removed SQLite fallback code
- Now exclusively uses `get_all_live_flights()` from `service.pg_provider`
- Queries the `live` schema in PostgreSQL
- Returns active flights with position, anomaly status, and metadata
- Cutoff: 15 minutes (configurable)

**PostgreSQL Tables Used:**
- `live.flight_metadata` - Flight information and status
- `live.normal_tracks` - Track points for position data
- `live.anomaly_reports` - Severity scores for anomalies

**Error Handling:**
- Raises HTTP 500 if PostgreSQL provider is unavailable
- Returns empty results if no flights found (valid state)

---

### 2. `/api/live/track/research/{flight_id}` (GET)
**File:** `routes/flights.py` (line ~666)
**Module:** Flights

**Changes:**
- Completely replaced SQLite implementation
- Now uses `get_flight_track()` and `get_flight_metadata()` from `service.pg_provider`
- Queries the `live` schema in PostgreSQL
- Returns track points + flight metadata for a specific flight

**PostgreSQL Tables Used:**
- `live.normal_tracks` - Track points (lat, lon, alt, speed, heading, etc.)
- `live.anomalies_tracks` - Anomaly track points (fallback)
- `live.flight_metadata` - Flight details (airline, aircraft, airports, etc.)

**Response Structure:**
```json
{
  "flight_id": "string",
  "callsign": "string",
  "points": [
    {
      "flight_id": "string",
      "timestamp": 123456789,
      "lat": 31.5,
      "lon": 34.8,
      "alt": 35000,
      "gspeed": 450,
      "vspeed": 0,
      "track": 90,
      "heading": 90,
      "squawk": "1234",
      "callsign": "ABC123",
      "source": "adsb"
    }
  ],
  "flight_number": "ABC123",
  "airline": "Example Airlines",
  "aircraft_type": "B738",
  "origin_airport": "TLV",
  "destination_airport": "JFK",
  "is_anomaly": false,
  "is_military": false
}
```

**Error Handling:**
- Returns HTTP 404 if flight not found
- Returns HTTP 500 if PostgreSQL provider unavailable or other errors

---

### 3. `/api/live/flight-status/{flight_id}` (GET)
**File:** `routes/flights.py` (line ~1802)
**Module:** Flights

**Changes:**
- Removed 260+ lines of SQLite code (5 different database fallbacks)
- Now uses PostgreSQL connection from `service.pg_provider`
- Searches multiple schemas in priority order: live → feedback → research
- Maintains flight phase detection logic (GROUND, TAKEOFF, CLIMB, CRUISE, DESCENT, LANDING, APPROACH)
- Returns status with most recent position and metadata

**PostgreSQL Tables Used:**
- `live.flight_metadata` - Current flight metadata
- `live.normal_tracks` - Recent track points for live flights
- `live.anomalies_tracks` - Anomaly track points (fallback)
- `feedback.flight_metadata` - Tagged flight metadata (second priority)
- `feedback.normal_tracks` - Tagged flight tracks
- `feedback.anomalies_tracks` - Tagged anomaly tracks
- `research.flight_metadata` - Historical metadata (third priority)
- `research.normal_tracks` - Historical tracks
- `research.anomalies_tracks` - Historical anomaly tracks

**Search Order:**
1. Try `live` schema (realtime monitor data)
2. Try `feedback` schema (user-tagged flights)
3. Try `research` schema (historical research data)

**Flight Phase Detection:**
- Analyzes last 10 track points
- Calculates altitude trends to determine phase
- Returns: GROUND, TAKEOFF, CLIMB, CRUISE, DESCENT, LANDING, APPROACH, or UNKNOWN

**Response Structure:**
```json
{
  "flight_id": "string",
  "callsign": "ABC123",
  "status": "CRUISE",
  "altitude_ft": 35000,
  "speed_kts": 450,
  "heading": 90,
  "origin": "TLV",
  "destination": "JFK",
  "lat": 31.5,
  "lon": 34.8,
  "eta_minutes": null,
  "source": "postgresql.live"
}
```

**Error Handling:**
- Returns HTTP 404 if flight not found in any schema
- Returns HTTP 500 if PostgreSQL provider unavailable or other errors

---

### 4. `/api/replay/other-flight/{flight_id}` (GET)
**File:** `routes/feedback.py` (line ~1021)
**Module:** Feedback

**Changes:**
- Completely replaced SQLite implementation (190+ lines)
- Now uses `get_flight_track()` and `get_flight_metadata()` from `service.pg_provider`
- Searches feedback schema first, then falls back to research schema
- Returns track points + metadata for proximity alert replay

**PostgreSQL Tables Used:**
- `feedback.flight_tracks` - Track points from tagged flights
- `feedback.anomalies_tracks` - Anomaly track points (fallback)
- `feedback.normal_tracks` - Normal track points (fallback)
- `feedback.flight_metadata` - Flight metadata from feedback
- `research.anomalies_tracks` - Research anomaly tracks (second fallback)
- `research.normal_tracks` - Research normal tracks (second fallback)
- `research.flight_metadata` - Research flight metadata (second fallback)

**Search Order:**
1. Try `feedback` schema (tagged flights from user feedback)
2. Fallback to `research` schema (historical research data)

**Response Structure:**
```json
{
  "flight_id": "string",
  "callsign": "string",
  "points": [
    {
      "flight_id": "string",
      "timestamp": 123456789,
      "lat": 31.5,
      "lon": 34.8,
      "alt": 35000,
      "gspeed": 450,
      "vspeed": 0,
      "track": 90,
      "heading": 90,
      "squawk": "1234",
      "callsign": "ABC123",
      "source": "adsb"
    }
  ],
  "metadata": {
    "flight_id": "string",
    "callsign": "string",
    "airline": "Example Airlines",
    "origin_airport": "TLV",
    "destination_airport": "JFK",
    "aircraft_type": "B738",
    "is_military": false,
    "category": "commercial"
  },
  "source": "postgresql.feedback",
  "total_points": 150
}
```

**Error Handling:**
- Returns HTTP 404 if flight not found in any schema
- Returns HTTP 500 if PostgreSQL provider unavailable or other errors

---

## Complete Migration Status

All four endpoints now exclusively use PostgreSQL:

| Endpoint | Status | Schema(s) Used | Lines Removed |
|----------|--------|----------------|---------------|
| `/api/live/flights` | ✅ Migrated | `live` | 130+ lines |
| `/api/live/track/research/{flight_id}` | ✅ Migrated | `live` | 130+ lines |
| `/api/live/flight-status/{flight_id}` | ✅ Migrated | `live`, `feedback`, `research` | 260+ lines |
| `/api/replay/other-flight/{flight_id}` | ✅ Migrated | `feedback`, `research` | 190+ lines |

**Total SQLite code removed:** ~710 lines

---

## Benefits of PostgreSQL Migration

1. **Scalability**: PostgreSQL handles concurrent connections and large datasets better than SQLite
2. **Performance**: Indexed queries and connection pooling improve response times
3. **Reliability**: ACID compliance and better concurrent write handling
4. **Centralization**: All data (live, research, feedback) in one database with proper schemas
5. **Consistency**: No file-locking issues that SQLite can have with concurrent access
6. **Multi-schema Support**: Clean separation of concerns (live, research, feedback schemas)
7. **Reduced Code**: Eliminated 300+ lines of duplicate SQLite query code

---

## PostgreSQL Provider Functions Used

### `get_all_live_flights(cutoff_minutes: int = 15)`
- Returns all active flights within cutoff window
- Includes position, heading, speed, anomaly status
- Schema: `live`

### `get_flight_track(flight_id: str, schema: str = 'live')`
- Returns list of track points for a flight
- Searches normal_tracks and anomalies_tracks
- Schema: `live` (or configurable)

### `get_flight_metadata(flight_id: str, schema: str = 'live')`
- Returns flight metadata dict
- Includes airline, aircraft, airports, category, etc.
- Schema: `live` (or configurable)

---

## Related Endpoints (Not Yet Migrated)

The following live endpoints still use SQLite and may need migration:

1. **`/api/live/anomalies`** - Fetch anomalies within time range
2. **`/api/live/anomalies/since`** - Fetch anomalies since timestamp

These can be migrated using similar patterns with PostgreSQL provider functions.

---

## Testing Recommendations

1. **Test `/api/live/flights`:**
   ```bash
   curl http://localhost:8000/api/live/flights
   ```
   Expected: JSON with flights array, anomaly_count, total_count

2. **Test `/api/live/track/research/{flight_id}`:**
   ```bash
   curl http://localhost:8000/api/live/track/research/{flight_id}
   ```
   Expected: JSON with flight_id, callsign, points array, and metadata

3. **Test `/api/live/flight-status/{flight_id}`:**
   ```bash
   curl http://localhost:8000/api/live/flight-status/{flight_id}
   ```
   Expected: JSON with status (CRUISE, CLIMB, etc.), altitude, speed, heading, origin, destination

4. **Test `/api/replay/other-flight/{flight_id}`:**
   ```bash
   curl http://localhost:8000/api/replay/other-flight/{flight_id}
   ```
   Expected: JSON with flight_id, callsign, points array, metadata, source, and total_points

5. **Verify PostgreSQL Connection:**
   ```bash
   curl http://localhost:8000/api/health
   ```
   Should show PostgreSQL status as "active"

---

## Environment Variables Required

Ensure these are set in `.env`:

```env
POSTGRES_DSN=postgresql://user:password@host:port/database

# Optional pool configuration
PG_POOL_MIN_CONNECTIONS=2
PG_POOL_MAX_CONNECTIONS=10
PG_CONNECT_TIMEOUT=10
PG_STATEMENT_TIMEOUT=30000
```

---

## Migration Date
January 28, 2026
