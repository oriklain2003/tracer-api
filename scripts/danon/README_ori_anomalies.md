# Flight Anomaly Detection System

## Overview

This script detects anomalous flight behavior by comparing a suspect flight track against historical "normal" flights from the same origin-destination pair. For each point in the suspect track, it finds where other flights were when they passed nearby and checks if the suspect's speed profile is statistically normal.

## Core Algorithm

### 1. Setup Phase
- Load all "normal" flight tracks from PostgreSQL that share the same origin and destination airports
- Insert them into a local SQLite database with spatial indexing for fast geographic queries
- Optionally exclude the suspect flight itself from the normal dataset

### 2. Point-by-Point Analysis

For each point in the suspect track:

1. **Find Nearby Historical Tracks**
   - Query SQLite to find the closest point from each historical flight
   - Uses spatial bounding box (±5km × 1.2) for initial filtering
   - For each normal flight, keeps only its closest point to the suspect position
   - Filters to tracks within 5km distance threshold

2. **Statistical Speed Analysis**
   - Collect ground speeds (gspeed) and vertical speeds (vspeed) from nearby tracks
   - Calculate if suspect's speeds are statistically anomalous:
     - Compute mean and std dev of normal flights' speeds at this location
     - Calculate z-score: how many standard deviations away is the suspect?
     - Convert to probability using 2-tailed normal distribution test
     - Flag as WARNING if probability < 0.01 (99% confidence) AND std > 1.0

3. **Warning Queue Smoothing**
   - Maintains sliding window of last 15 point states
   - Only triggers WARNING if ≥80% of recent points are anomalous
   - Prevents false positives from temporary speed variations

4. **Edge Case Handling**
   - **Insufficient Data**: If < 3 nearby tracks found, returns STANDBY (can't make data-driven decision)
   - **Track Deviation**: If zero nearby tracks AND flight has been airborne >5 min AND started with ≥10 nearby tracks → WARNING (flight completely off normal path)
   - **Takeoff Grace Period**: Ignores speed warnings in first 5 minutes to avoid false positives during departure

### 3. Output States

- **NORMAL**: All tests passed, flight behaving normally
- **WARNING**: Statistical anomaly detected OR flight deviated from all normal tracks
- **STANDBY**: Insufficient data or too close to takeoff for reliable decision

## Key Components

### Data Structures

```python
FlightDataPoint(flight_id, timestamp, latitude, longitude, altitude, gspeed, vspeed)
```

### Classes

**`WarningQueue`**
- Sliding window buffer (default 15 points)
- Accumulates WARNING states
- Returns WARNING only when ≥80% of window is anomalous
- Smooths out noise and temporary deviations

**`FlightTracker`**
- Main analysis engine
- Manages database queries
- Parameters:
  - `max_track_proximity_threshold`: 5km (how close tracks must be to compare)
  - `minimal_close_tracks_for_decision`: 3 (minimum tracks needed for statistical analysis)

**`monitor_flight()` Generator**
- Main entry point
- Yields (state, reason) for each point in suspect track
- Parameters:
  - `minimum_flight_duration_for_warning`: 5.0 min (grace period for takeoff)
  - `minimum_initial_tracks_for_immediate_warning`: 10 (tracks needed at start to detect path deviation)

## Database Schema

### PostgreSQL Sources
- `research.normal_tracks`: Historical flight position data
- `research.flight_metadata`: Origin/destination mapping
- `feedback.flight_tracks`: Alternative track source

### SQLite Working Database
- `flight_tracks`: Temporary spatial-indexed copy of normal tracks for fast querying
- Indexed on (latitude, longitude) for bounding box queries
- Uses SpatiaLite extension for geodesic distance calculations

## Statistical Method

### Probability Calculation

```
z_score = |x - μ| / σ
p_value = 2 × P(Z > z_score)  # 2-tailed test
```

- **Anomaly threshold**: p < 0.01 (suspect value is in outer 1% of distribution)
- **Minimum variance**: σ > 1.0 (prevents false positives when all flights have nearly identical speeds)

### Why 2-Tailed Test?
Both unusually high AND unusually low speeds are suspicious (e.g., flying too fast OR too slow for location).

## Spatial Query Strategy

1. **Bounding Box Pre-filter**
   - Converts 5km threshold to lat/lon degrees
   - Adjusts longitude margin for latitude-dependent distortion
   - Fast index-based filtering

2. **Per-Flight Ranking**
   - Uses `ROW_NUMBER() OVER (PARTITION BY flight_id ORDER BY distance)`
   - Ensures each normal flight contributes exactly 1 point (its closest to suspect)
   - Prevents overweighting flights with dense sampling

3. **Exact Distance**
   - SpatiaLite's ST_Distance with geodesic mode (ellipsoid-accurate)
   - WGS84 coordinate system (standard GPS)

## Example Usage

```python
origin = "OMDW"
destination = "OLBA"
flight_id = "3d735e94"

# Load suspect track
flight_iterable = [FlightDataPoint(*row) for row in fetch_normal_track(postgres_conn, flight_id)]

# Analyze each point
states, reasons = [], []
for state, reason in monitor_flight(
    postgres_conn,
    sqlite_conn,
    origin,
    destination,
    flight_iterable,
    id_to_delete=None,  # or flight_id to exclude it from normals
    to_tqdm=True,
    minimum_flight_duration_for_warning=5.0,
    minimum_initial_tracks_for_immediate_warning=10
):
    states.append(state)
    reasons.append(reason)
```

## Decision Logic Flow

```
For each point:
  ├─ Find closest tracks (within 5km)
  │
  ├─ < 3 nearby tracks?
  │  └─ STANDBY (insufficient data)
  │
  ├─ 0 nearby tracks AND flight >5min AND started with ≥10 tracks?
  │  └─ WARNING (deviated from all normal paths)
  │
  ├─ Calculate speed probabilities
  │  ├─ gspeed p-value < 0.01? → queue WARNING
  │  └─ vspeed p-value < 0.01? → queue WARNING
  │
  ├─ Queue state = WARNING AND flight >5min?
  │  └─ WARNING (statistically anomalous speeds)
  │
  └─ Otherwise → NORMAL or STANDBY
```

## Performance Optimizations

1. **SQLite with WAL mode**: Better concurrent read performance
2. **Spatial indexing**: Fast bounding box queries
3. **Single closest point per flight**: Reduces data volume
4. **In-memory computation**: Statistical tests on small arrays

## Limitations & Considerations

- **Route Coverage**: Requires sufficient historical flights on same route
- **Weather/Traffic**: Normal variations (holding patterns, weather routing) may trigger false positives
- **Takeoff/Landing**: First/last 5 minutes have reduced sensitivity
- **Sampling Rate**: Works best with consistent track sampling (30-second intervals assumed)
- **Coordinate System**: Assumes WGS84, geodesic distances via SpatiaLite

## Key Thresholds (Tunable)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Distance threshold | 5 km | Max distance to consider tracks "nearby" |
| Minimum tracks | 3 | Minimum historical tracks needed for decision |
| P-value threshold | 0.01 | Statistical significance (99% confidence) |
| Warning queue size | 15 | Sliding window for smoothing |
| Warning percentage | 80% | % of window that must be anomalous |
| Takeoff grace period | 5 min | Ignore warnings near departure |
| Initial tracks threshold | 10 | Tracks needed to detect path deviation |
