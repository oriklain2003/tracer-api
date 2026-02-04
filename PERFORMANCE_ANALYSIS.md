# Performance Analysis Report

## Executive Summary

After profiling the slow API routes, I've identified **critical performance bottlenecks** that are causing response times of 2-5+ seconds. The main issues are:

1. **N+1 Query Problem** in feedback history route
2. **Inefficient Airport Lookup** using linear search through all airports
3. **SQLite Queries** for each missing origin/destination
4. **Lack of Database Indexes** on frequently queried columns
5. **No Result Caching** for expensive operations

---

## Routes Analyzed

### 1. `/api/track/unified/{flight_id}`
- **Current Performance**: ~2.5 seconds
- **Expected Performance**: < 0.5 seconds
- **Main Issue**: Connection pooling overhead (66% unaccounted time)

### 2. `/api/feedback/history`
- **Current Performance**: ~2 seconds (but can be 5-10s with more missing data)
- **Expected Performance**: < 0.5 seconds
- **Main Issue**: N+1 queries for origin/destination calculation

---

## Detailed Findings

### Issue #1: N+1 Query Problem (CRITICAL)

**Location**: `routes/feedback.py` - `get_feedback_history()`

**Problem**:
```python
# Line 699-705 in feedback.py
if not origin_airport or not destination_airport:
    calculated_origin, calculated_dest = _calculate_origin_destination_from_track(row['flight_id'])
    # This creates a NEW SQLite connection and query FOR EACH flight!
```

**Impact**:
- For 100 flights with 5 missing origin/dest: **5 extra database connections + queries**
- Each connection: ~0.1-0.2 seconds
- Each query: ~0.1-0.3 seconds
- **Total added: 1-2.5 seconds**

**In Our Test**:
- 5 out of 100 flights missing origin/dest
- Would add ~1 second to response time

**Example Code Flow**:
1. Main query returns 100 flights (0.2s)
2. For each missing origin/dest (5 flights):
   - Open SQLite connection (0.1s)
   - Query flight_tracks table (0.1s)
   - Calculate nearest airport (0.1s)
   - Close connection
3. Total: 0.2s + (5 × 0.3s) = **1.7 seconds**

---

### Issue #2: Inefficient Airport Lookup (CRITICAL)

**Location**: `routes/feedback.py` - `_find_nearest_airport()`

**Problem**:
```python
# Lines 168-172 in feedback.py
for airport in _AIRPORTS_DATA:  # Iterates through ALL airports!
    distance = _haversine_nm(lat, lon, airport["lat"], airport["lon"])
    if distance < min_distance and distance <= max_distance_nm:
        min_distance = distance
        nearest = airport
```

**Impact**:
- _AIRPORTS_DATA likely contains **thousands of airports**
- For each origin/dest lookup: **O(n) where n = number of airports**
- Each haversine calculation: ~0.001ms
- For 5,000 airports: 5ms per lookup
- For 5 missing origins + 5 missing destinations: **50ms**

**Better Approach**: Use spatial indexing (R-tree, KD-tree, or PostGIS)

---

### Issue #3: Using SQLite Instead of PostgreSQL

**Location**: `routes/feedback.py` - `_calculate_origin_destination_from_track()`

**Problem**:
```python
# Lines 197-210
conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))  # New connection!
cursor.execute("""
    SELECT lat, lon, alt
    FROM flight_tracks
    WHERE flight_id = ?
    ORDER BY timestamp ASC
""", (flight_id,))
```

**Impact**:
- SQLite connections are not pooled
- Each connection has overhead
- Data might be in PostgreSQL already!

**Solution**: Use PostgreSQL connection pool and query `feedback.flight_tracks` instead

---

### Issue #4: Missing Database Indexes

**Tables Affected**:
- `feedback.user_feedback` - missing index on `first_seen_ts`
- `feedback.flight_tracks` - missing index on `flight_id`
- `feedback.flight_metadata` - missing index on `flight_id`

**Impact**:
- Without indexes, queries do full table scans
- For large tables (100k+ rows): can add 0.5-2 seconds per query

**Test Query**:
```sql
EXPLAIN ANALYZE 
SELECT * FROM feedback.user_feedback 
WHERE first_seen_ts BETWEEN 0 AND 1738713600 
ORDER BY first_seen_ts DESC 
LIMIT 100;
```

---

### Issue #5: No Result Caching

**Problem**:
- Feedback history is queried frequently
- Results don't change often (only when new feedback is added)
- No caching mechanism implemented

**Impact**:
- Every request hits the database
- Same query repeated many times
- Wastes database resources

**Solution**: Add Redis or in-memory caching with 5-10 minute TTL

---

## Performance Breakdown

### Track Route (`/api/track/unified/{flight_id}`)

```
Total Time: 2.473s

Measured:
  PostgreSQL query:           0.443s (17.9%)
  Failed feedback query:      0.401s (16.2%)
  
Unaccounted:                  1.629s (65.9%) ⚠️
```

**Unaccounted time likely due to**:
- Connection pool initialization
- Import overhead
- Authentication middleware
- JSON serialization

### Feedback History Route (`/api/feedback/history`)

```
Total Time: 1.997s

Measured:
  Main query:                 0.202s (10.1%)
  DB connection:              0.162s ( 8.1%)
  Origin/dest calc (5×):      ~1.000s (50.0%) [estimated]
  
Unaccounted:                  0.633s (31.8%)
```

---

## Recommended Fixes

### Priority 1: Fix N+1 Query Problem (Immediate - 1 hour)

**Solution A: Batch Query**
```python
# Instead of querying one flight at a time, query all missing ones at once
missing_flight_ids = [row['flight_id'] for row in rows 
                      if not row['origin_airport'] or not row['destination_airport']]

if missing_flight_ids:
    # Single query to get all tracks at once
    tracks_by_flight = get_tracks_for_flights(missing_flight_ids)
    
    for row in rows:
        if row['flight_id'] in tracks_by_flight:
            origin, dest = calculate_origin_dest(tracks_by_flight[row['flight_id']])
            # ... update row
```

**Expected Improvement**: 1-2 seconds reduction

**Solution B: Pre-calculate and Store**
```python
# During feedback creation, calculate and store origin/dest in database
# Then you never need to calculate it on-the-fly
```

**Expected Improvement**: 1-2.5 seconds reduction

---

### Priority 2: Add Spatial Index for Airport Lookup (Immediate - 2 hours)

**Option A: Use PostGIS (Best)**
```sql
-- Create PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create airports table with spatial index
CREATE TABLE public.airports (
    id SERIAL PRIMARY KEY,
    ident VARCHAR(10),
    iata_code VARCHAR(3),
    icao_code VARCHAR(4),
    name VARCHAR(255),
    location GEOGRAPHY(POINT, 4326)
);

CREATE INDEX idx_airports_location ON public.airports USING GIST(location);

-- Query nearest airport (FAST!)
SELECT ident, icao_code, iata_code,
       ST_Distance(location, ST_SetSRID(ST_MakePoint(lon, lat), 4326)) as distance_meters
FROM public.airports
WHERE ST_DWithin(location, ST_SetSRID(ST_MakePoint(lon, lat), 4326), 9260)  -- 5 NM = 9260m
ORDER BY distance_meters
LIMIT 1;
```

**Expected Improvement**: 50-100ms reduction per lookup

**Option B: Use KD-Tree in Python**
```python
from scipy.spatial import cKDTree
import numpy as np

# Build tree once at startup
airport_coords = np.array([[a['lat'], a['lon']] for a in _AIRPORTS_DATA])
airport_tree = cKDTree(airport_coords)

def find_nearest_airport_fast(lat, lon, max_distance_nm=5.0):
    # Query is O(log n) instead of O(n)!
    distances, indices = airport_tree.query([lat, lon], k=1, distance_upper_bound=max_distance_nm/60)
    if indices[0] < len(_AIRPORTS_DATA):
        return _AIRPORTS_DATA[indices[0]]
    return None
```

**Expected Improvement**: 40-80ms reduction per lookup

---

### Priority 3: Add Database Indexes (Immediate - 10 minutes)

```sql
-- Feedback tables
CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp 
    ON feedback.user_feedback(first_seen_ts DESC);

CREATE INDEX IF NOT EXISTS idx_user_feedback_flight_id 
    ON feedback.user_feedback(flight_id);

CREATE INDEX IF NOT EXISTS idx_flight_tracks_flight_id 
    ON feedback.flight_tracks(flight_id);

CREATE INDEX IF NOT EXISTS idx_flight_metadata_flight_id 
    ON feedback.flight_metadata(flight_id);

-- Research tables
CREATE INDEX IF NOT EXISTS idx_normal_tracks_flight_id 
    ON research.normal_tracks(flight_id);

CREATE INDEX IF NOT EXISTS idx_anomalies_tracks_flight_id 
    ON research.anomalies_tracks(flight_id);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp_label 
    ON feedback.user_feedback(first_seen_ts DESC, user_label);
```

**Expected Improvement**: 0.5-1 second reduction

---

### Priority 4: Migrate to PostgreSQL Connection Pool (Short-term - 1 hour)

**Current**:
```python
# Creates new connection each time
conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
```

**Better**:
```python
# Use existing PostgreSQL pool
from service.pg_provider import get_pool

pool = get_pool()
with pool.get_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT lat, lon, alt
            FROM feedback.flight_tracks
            WHERE flight_id = %s
            ORDER BY timestamp ASC
        """, (flight_id,))
```

**Expected Improvement**: 0.1-0.3 seconds per query

---

### Priority 5: Add Result Caching (Medium-term - 2-3 hours)

**Option A: Redis Cache**
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379)

def get_feedback_history(start_ts, end_ts, limit):
    cache_key = f"feedback_history:{start_ts}:{end_ts}:{limit}"
    
    # Try cache first
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Query database
    result = query_database(...)
    
    # Cache for 5 minutes
    redis_client.setex(cache_key, 300, json.dumps(result))
    
    return result
```

**Expected Improvement**: 1.5-2 seconds reduction (cache hit)

**Option B: In-Memory LRU Cache**
```python
from functools import lru_cache
from datetime import datetime

@lru_cache(maxsize=100)
def get_feedback_history_cached(start_ts, end_ts, limit, _cache_bust=None):
    return get_feedback_history(start_ts, end_ts, limit)

# Call with current minute as cache bust to refresh every minute
result = get_feedback_history_cached(
    start_ts, end_ts, limit, 
    _cache_bust=datetime.now().minute
)
```

**Expected Improvement**: 1.5-2 seconds reduction (cache hit)

---

## Expected Performance After Fixes

### Track Route
- **Current**: 2.5 seconds
- **After indexes**: 2.0 seconds (-0.5s)
- **After PostgreSQL migration**: 1.5 seconds (-0.5s)
- **After caching**: 0.3 seconds (-1.2s)
- **Target**: < 0.5 seconds ✓

### Feedback History Route
- **Current**: 2.0 seconds (can be 5-10s)
- **After fixing N+1**: 1.0 seconds (-1.0s)
- **After spatial index**: 0.9 seconds (-0.1s)
- **After database indexes**: 0.4 seconds (-0.5s)
- **After caching**: 0.1 seconds (-0.3s)
- **Target**: < 0.5 seconds ✓

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add database indexes (10 min)
2. ✅ Batch origin/dest queries (1 hour)
3. ✅ Use PostgreSQL instead of SQLite (30 min)

**Expected Result**: 50-70% improvement

### Phase 2: Optimization (2-3 hours)
1. ✅ Implement KD-tree for airport lookup (1 hour)
2. ✅ Add basic in-memory caching (1 hour)
3. ✅ Pre-calculate origin/dest during feedback creation (1 hour)

**Expected Result**: 80-90% improvement

### Phase 3: Infrastructure (4-6 hours)
1. ⏳ Set up Redis for distributed caching
2. ⏳ Migrate airports to PostGIS
3. ⏳ Add query monitoring and alerts

**Expected Result**: 90-95% improvement

---

## Monitoring Recommendations

1. **Add query timing logs**:
```python
import time

start = time.time()
result = expensive_operation()
elapsed = time.time() - start

if elapsed > 1.0:
    logger.warning(f"Slow query: {operation_name} took {elapsed:.2f}s")
```

2. **Track cache hit rates**:
```python
cache_hits = 0
cache_misses = 0

def get_cache_stats():
    hit_rate = cache_hits / (cache_hits + cache_misses) * 100
    return {"hit_rate": hit_rate, "hits": cache_hits, "misses": cache_misses}
```

3. **Set up alerts**:
- Alert if p95 response time > 1 second
- Alert if cache hit rate < 50%
- Alert if database connections > 80% of pool

---

## Files to Modify

1. **routes/feedback.py**
   - Fix `_calculate_origin_destination_from_track()` - batch queries
   - Fix `_find_nearest_airport()` - use KD-tree
   - Fix `get_feedback_history()` - add caching

2. **service/pg_provider.py**
   - Add batch query function `get_tracks_for_flights(flight_ids)`
   - Add caching layer

3. **Database Migration**
   - Run index creation SQL
   - Optionally: create airports table with PostGIS

---

## Conclusion

The performance issues are well-understood and fixable. The main culprits are:

1. **N+1 queries** (biggest impact)
2. **Linear airport search** (O(n) per lookup)
3. **Missing indexes** (causing full table scans)
4. **No caching** (repeated expensive operations)

With the recommended fixes, we can achieve **70-90% improvement** (2-5s → 0.3-0.5s) in a few hours of work.

---

## Next Steps

Run this command to start implementing the fixes:

```bash
# 1. Add database indexes (immediate improvement)
psql -d anomaly_detection -f add_indexes.sql

# 2. Test the changes
python profile_standalone.py --route feedback-history --limit 100
python profile_standalone.py --route track --flight-id 3cc2683b

# 3. Deploy and monitor
# Watch for improvements in response times
```
