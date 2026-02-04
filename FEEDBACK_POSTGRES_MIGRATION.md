# Feedback System PostgreSQL Migration

## Overview

The feedback system has been migrated from SQLite to PostgreSQL for better scalability, reliability, and integration with the rest of the application.

## Changes Made

### 1. Database Migration

**Before (SQLite):**
- `training_ops/feedback.db` - User feedback metadata
- `training_ops/training_dataset.db` - Flight tracks (flight_tracks and anomalous_tracks tables)

**After (PostgreSQL):**
- `feedback` schema with 4 tables:
  - `feedback.user_feedback` - User feedback metadata
  - `feedback.flight_metadata` - Comprehensive flight details
  - `feedback.flight_tracks` - Flight track points
  - `feedback.anomaly_reports` - Anomaly report data

### 2. Code Changes

#### api.py
- Removed import from `training_ops.db_utils`
- Added import from `service.pg_provider`
- Created wrapper function `save_feedback()` that calls `save_feedback_to_postgres()`
- Commented out `init_dbs()` call (no longer needed)

#### service/pg_provider.py
- Added `save_feedback_to_postgres()` function
- Added helper functions:
  - `_compute_flight_summary()` - Calculate flight statistics
  - `_extract_anomaly_summary()` - Extract anomaly data from report

### 3. Schema Structure

#### feedback.user_feedback
Stores user feedback choices and metadata:
- `flight_id` (unique) - Flight identifier
- `tagged_at` - When feedback was submitted
- `first_seen_ts` / `last_seen_ts` - Flight time range
- `user_label` - 0=Normal, 1=Anomaly
- `comments` - User comments
- `rule_id` / `rule_ids` - Single or multiple rule IDs
- `rule_name` / `rule_names` - Rule names
- `other_details` - Custom details for "Other" option
- `callsign` - Aircraft callsign

#### feedback.flight_metadata
Comprehensive flight information:
- Flight identification (callsign, flight_number)
- Origin/Destination airports
- Airline and aircraft details
- Flight statistics (duration, points, distance)
- Altitude/Speed statistics (min/max/avg)
- Position data (start/end lat/lon)
- Schedule information
- Military flag

#### feedback.flight_tracks
Individual track points for each flight:
- `flight_id` - Flight identifier
- `timestamp` - Point timestamp
- `lat`, `lon`, `alt` - Position and altitude
- `gspeed`, `vspeed`, `track` - Speed and heading
- `squawk`, `callsign` - Aircraft identification
- `source` - Data source (e.g., "feedback")

#### feedback.anomaly_reports
Anomaly detection results:
- `flight_id` (unique) - Flight identifier
- `full_report` (JSONB) - Complete anomaly report
- `severity_cnn`, `severity_dense` - Severity scores
- `matched_rule_ids`, `matched_rule_names`, `matched_rule_categories` - Rule matches
- Quick reference fields (callsign, origin, destination)

## Migration Steps

### 1. Run the Migration Script

```bash
cd api
python scripts/migrate_feedback_to_postgres.py
```

This will:
- Create the `feedback` schema
- Create all 4 tables
- Create all necessary indexes
- Verify the setup

### 2. Environment Variables

Ensure your `.env` file has the PostgreSQL connection string:

```env
POSTGRES_DSN=postgresql://user:password@host:port/database
```

### 3. Deploy Updated Code

Deploy the updated `api.py` and `service/pg_provider.py` files.

### 4. Optional: Migrate Existing SQLite Data

If you have existing feedback data in SQLite that needs to be migrated, create a migration script to:
1. Read from `training_ops/feedback.db`
2. Read from `training_ops/training_dataset.db`
3. Insert into PostgreSQL tables using `save_feedback_to_postgres()`

## API Compatibility

The `save_feedback()` function signature remains the same for backward compatibility:

```python
save_feedback(
    flight_id: str,
    is_anomaly: bool,
    points: List[Dict[str, Any]],
    comments: str = "",
    rule_id: Optional[int] = None,
    other_details: str = "",
    full_report: Optional[Dict[str, Any]] = None,
    tagged: int = 1,
    flight_details: Optional[Dict[str, Any]] = None
)
```

All existing code calling `save_feedback()` will work without changes.

## Benefits

1. **Scalability**: PostgreSQL handles large datasets better than SQLite
2. **Reliability**: ACID compliance and better crash recovery
3. **Concurrency**: Multiple users can submit feedback simultaneously
4. **Integration**: Single database for all application data
5. **Advanced Features**: JSONB support, full-text search, advanced indexing
6. **Backup**: Enterprise-grade backup and replication tools
7. **Performance**: Optimized indexes for common queries

## Monitoring

Check feedback system health:

```python
from service.pg_provider import get_pool_status

status = get_pool_status()
print(status)
```

Query feedback data:

```python
from service.pg_provider import get_tagged_feedback_history

# Get recent feedback
feedback = get_tagged_feedback_history(
    start_ts=0,
    end_ts=int(time.time()),
    limit=100,
    include_normal=False
)
```

## Rollback Plan

If issues occur:

1. Restore old `api.py` from git:
   ```bash
   git checkout HEAD~1 api.py
   ```

2. Re-import `training_ops.db_utils`:
   ```python
   from training_ops.db_utils import save_feedback, init_dbs
   ```

3. Restart the API

## Testing

Test the feedback system:

```python
from service.pg_provider import save_feedback_to_postgres

# Test save
result = save_feedback_to_postgres(
    flight_id="TEST123",
    is_anomaly=True,
    points=[
        {"timestamp": 1234567890, "lat": 51.5, "lon": -0.1, "alt": 10000, "gspeed": 450}
    ],
    comments="Test feedback",
    rule_id=1,
    flight_details={
        "callsign": "TEST123",
        "airline": "Test Airways"
    }
)

print(f"Save result: {result}")
```

## Support

For issues or questions:
1. Check PostgreSQL logs for connection/query errors
2. Verify schema exists: `SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'feedback';`
3. Check table structure: `\d feedback.user_feedback` (in psql)
4. Monitor connection pool: `get_pool_status()`

## Future Enhancements

Potential improvements:
1. Add feedback analytics dashboard
2. Implement feedback export to CSV/JSON
3. Add feedback approval workflow
4. Create feedback statistics API endpoints
5. Add feedback search and filtering
6. Implement feedback versioning/audit trail
