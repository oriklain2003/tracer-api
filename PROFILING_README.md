# API Route Profiling Guide

This guide explains how to profile the slow API routes to identify performance bottlenecks.

## Overview

Two profiling scripts are available:

1. **`profile_routes.py`** - Uses `cProfile` for function-level profiling
2. **`profile_detailed.py`** - Provides detailed timing breakdown for each step

## Installation

No additional packages needed for basic profiling. Both scripts use Python's built-in `cProfile`.

## Routes to Profile

### 1. Unified Track Route (`/api/track/unified/{flight_id}`)

This route retrieves flight track data from multiple sources:
- PostgreSQL feedback schema
- SQLite live_tracks.db
- SQLite research_new.db
- SQLite feedback_tagged.db
- SQLite flight_cache.db
- FR24 API (fallback)

### 2. Feedback History Route (`/api/feedback/history`)

This route retrieves feedback history with:
- Complex SQL query with multiple JOINs
- Origin/destination calculation (potentially slow)
- JSON parsing
- Result formatting

## Usage

### Basic Profiling (cProfile)

```bash
# Profile unified track route
python profile_routes.py --route track --flight-id <FLIGHT_ID>

# Profile feedback history route
python profile_routes.py --route feedback-history

# With custom parameters
python profile_routes.py --route feedback-history --start-ts 1704067200 --limit 50

# Save output to file
python profile_routes.py --route track --flight-id <FLIGHT_ID> --output my_profile.prof
```

### Detailed Timing Analysis

```bash
# Detailed analysis of unified track
python profile_detailed.py --route track --flight-id <FLIGHT_ID>

# Detailed analysis of feedback history
python profile_detailed.py --route feedback-history

# With parameters
python profile_detailed.py --route feedback-history --limit 50 --exclude-normal
```

## Finding Sample Flight IDs

Use the helper script to find flight IDs from your databases:

```bash
python get_sample_flight_ids.py
```

Or manually query the database:

```bash
# SQLite
sqlite3 realtime/live_tracks.db "SELECT DISTINCT flight_id FROM flight_tracks LIMIT 10;"
sqlite3 api/feedback_tagged.db "SELECT DISTINCT flight_id FROM user_feedback LIMIT 10;"

# PostgreSQL
psql -d anomaly_detection -c "SELECT DISTINCT flight_id FROM feedback.user_feedback LIMIT 10;"
```

## Understanding the Output

### cProfile Output (profile_routes.py)

The output shows:
- **ncalls**: Number of times the function was called
- **tottime**: Total time spent in the function (excluding sub-functions)
- **cumtime**: Cumulative time (including sub-functions)
- **percall**: Average time per call

Look for:
- ⚠️ Functions with high `cumtime` (bottlenecks)
- ⚠️ Functions called many times with high `tottime`
- ⚠️ Database queries taking > 1 second
- ⚠️ Network calls (FR24 API)

### Detailed Output (profile_detailed.py)

Shows step-by-step timing:
- Database connection time
- Query execution time
- Data fetching time
- Processing time
- Percentage breakdown

Look for:
- ⚠️ Steps taking > 30% of total time
- ⚠️ Origin/destination calculations (can be very slow)
- ⚠️ Multiple database queries in sequence
- ⚠️ JSON parsing overhead

## Common Performance Issues

### Unified Track Route

1. **Sequential database checks** - Checks databases one by one instead of in parallel
2. **Missing indexes** - Database queries without proper indexes
3. **Large track data** - Returning thousands of points
4. **FR24 API fallback** - Network call when data not in cache

### Feedback History Route

1. **Complex SQL query** - Multiple JOINs and subqueries
2. **Origin/destination calculation** - Queries flight_tracks for EACH flight missing origin/dest
3. **No result caching** - Same query repeated multiple times
4. **Large result set** - Returning many flights with full metadata

## Optimization Recommendations

### Quick Wins

1. **Add database indexes**:
   ```sql
   CREATE INDEX IF NOT EXISTS idx_flight_tracks_flight_id ON flight_tracks(flight_id);
   CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp ON user_feedback(first_seen_ts);
   CREATE INDEX IF NOT EXISTS idx_flight_metadata_flight_id ON flight_metadata(flight_id);
   ```

2. **Cache origin/destination** - Calculate once and store in database

3. **Reduce data transfer** - Don't return all track points, only needed fields

4. **Add result caching** - Cache feedback history results for 5-10 minutes

### Medium-Term Improvements

1. **Batch origin/destination calculations** - Calculate for all flights at once
2. **Use connection pooling** - Reuse database connections
3. **Optimize SQL query** - Remove unnecessary JOINs, use CTEs
4. **Add pagination** - Don't load all results at once

### Long-Term Improvements

1. **Move to PostgreSQL** - Migrate all SQLite databases to PostgreSQL
2. **Add Redis caching** - Cache frequently accessed data
3. **Background workers** - Pre-calculate expensive operations
4. **Database sharding** - Split data across multiple databases

## Example Output

### Good Performance (< 1 second)

```
Total execution time: 0.45 seconds
Result: 1234 track points returned

TIME BREAKDOWN:
  postgresql_query              :   0.120s ( 26.7%)
  main_query                    :   0.180s ( 40.0%)
  fetchall                      :   0.080s ( 17.8%)
  process_results               :   0.070s ( 15.6%)
```

### Poor Performance (> 5 seconds)

```
Total execution time: 12.35 seconds
Result: 100 flights returned

TIME BREAKDOWN:
  main_query                    :   2.450s ( 19.8%)
  origin_dest_calculation       :   8.920s ( 72.2%)  ⚠️ BOTTLENECK!
    (87 calculations, avg 0.103s each)
  process_results               :   0.980s (  7.9%)
```

## Analyzing Profile Data

To analyze saved profile data in detail:

```bash
# Interactive analysis
python -m pstats profile_output.prof

# In pstats prompt:
>>> sort cumtime
>>> stats 20
>>> sort tottime
>>> stats 20
>>> callers
```

## Getting Help

If you find a bottleneck but aren't sure how to fix it:

1. Share the profiling output
2. Include sample data (flight IDs, parameters used)
3. Note your database sizes (number of records)
4. Include any error messages

## Next Steps

1. Run both profilers on your slow routes
2. Identify the top 2-3 bottlenecks
3. Check if database indexes exist
4. Consider caching strategies
5. Optimize the slowest operations first
