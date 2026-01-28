# Learned Data Upload to PostgreSQL

This directory contains the script for uploading learned paths, tubes, SIDs, and STARs from local JSON files to PostgreSQL with optimized table structures for fast querying.

## 📋 Overview

The upload script reads the following JSON files from the `rules/` directory:
- `learned_paths.json` - Learned flight paths between origin-destination pairs
- `learned_tubes.json` - 3D corridor tubes for flight paths
- `learned_sid.json` - Standard Instrument Departures
- `learned_star.json` - Standard Terminal Arrival Routes

And uploads them to PostgreSQL with:
- ✅ Optimized table schemas
- ✅ Proper indexes for fast queries (origin, destination, airport)
- ✅ JSONB columns for geometry with GiST indexes
- ✅ Composite indexes for common query patterns
- ✅ Batch inserts for performance

## 🚀 Usage

### Prerequisites

1. **Install psycopg2**:
   ```bash
   pip install psycopg2-binary
   ```

2. **Set PostgreSQL connection**:
   Make sure your `.env` file has the `POSTGRES_DSN` variable:
   ```env
   POSTGRES_DSN=postgresql://username:password@host:5432/database
   ```

### Run the Upload Script

**First time / Update data:**
```bash
python scripts/upload_learned_data_to_postgres.py
```

**Drop existing tables and re-upload (WARNING: deletes all data):**
```bash
python scripts/upload_learned_data_to_postgres.py --drop-tables
```

### Expected Output

```
============================================================
🚀 LEARNED DATA UPLOAD TO POSTGRESQL
============================================================
Started at: 2026-01-28 10:30:00

🔌 Connecting to PostgreSQL...
✓ Connected successfully

Creating tables...
✓ Tables and indexes created successfully

📂 Reading paths from rules/learned_paths.json...
Found 3170 paths
✓ Uploaded 3170 paths

📂 Reading tubes from rules/learned_tubes.json...
Found 3145 tubes
✓ Uploaded 3145 tubes

📂 Reading SIDs from rules/learned_sid.json...
Found 189 SIDs
✓ Uploaded 189 SIDs

📂 Reading STARs from rules/learned_star.json...
Found 88 STARs
✓ Uploaded 88 STARs

============================================================
📊 DATABASE STATISTICS
============================================================

📍 PATHS:
  Total: 3170
  Unique Origins: 45
  Unique Destinations: 48
  Avg Member Count: 12.3
  Max Member Count: 156

🔵 TUBES:
  Total: 3145
  Unique Origins: 45
  Unique Destinations: 48
  Avg Member Count: 11.8

🛫 SIDS:
  Total: 189
  Unique Airports: 12
  Avg Member Count: 23.5

🛬 STARS:
  Total: 88
  Unique Airports: 10
  Avg Member Count: 18.2

============================================================

✅ SUCCESS! Uploaded 6592 total records
Completed at: 2026-01-28 10:30:15
```

## 🗄️ Database Schema

### `learned_paths` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR(100) PRIMARY KEY | Unique identifier |
| `origin` | VARCHAR(10) | Origin airport code |
| `destination` | VARCHAR(10) | Destination airport code |
| `centerline` | JSONB | Path centerline coordinates |
| `width_nm` | FLOAT | Path width in nautical miles |
| `member_count` | INTEGER | Number of flights used to learn this path |
| `min_alt_ft` | FLOAT | Minimum altitude in feet |
| `max_alt_ft` | FLOAT | Maximum altitude in feet |
| `created_at` | TIMESTAMP | Creation timestamp |

**Indexes:**
- `idx_paths_origin` - Fast lookup by origin
- `idx_paths_destination` - Fast lookup by destination
- `idx_paths_od_pair` - Fast lookup by origin-destination pair (most common query)
- `idx_paths_member_count` - Fast sorting by popularity
- `idx_paths_centerline_gin` - Fast JSONB queries

### `learned_tubes` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR(100) PRIMARY KEY | Unique identifier |
| `origin` | VARCHAR(10) | Origin airport code |
| `destination` | VARCHAR(10) | Destination airport code |
| `geometry` | JSONB | Tube polygon coordinates |
| `min_alt_ft` | FLOAT | Minimum altitude in feet |
| `max_alt_ft` | FLOAT | Maximum altitude in feet |
| `member_count` | INTEGER | Number of flights in this tube |
| `buffer_nm` | FLOAT | Buffer distance in nautical miles |
| `alpha` | FLOAT | Alpha shape parameter |
| `created_at` | TIMESTAMP | Creation timestamp |

**Indexes:**
- `idx_tubes_origin` - Fast lookup by origin
- `idx_tubes_destination` - Fast lookup by destination
- `idx_tubes_od_pair` - Fast lookup by origin-destination pair
- `idx_tubes_member_count` - Fast sorting by popularity
- `idx_tubes_geometry_gin` - Fast JSONB queries

### `learned_sids` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR(100) PRIMARY KEY | Unique identifier |
| `airport` | VARCHAR(10) | Airport code |
| `type` | VARCHAR(10) | Always "SID" |
| `centerline` | JSONB | Procedure centerline coordinates |
| `width_nm` | FLOAT | Procedure width in nautical miles |
| `member_count` | INTEGER | Number of flights used |
| `runway` | VARCHAR(10) | Associated runway (if applicable) |
| `created_at` | TIMESTAMP | Creation timestamp |

**Indexes:**
- `idx_sids_airport` - Fast lookup by airport
- `idx_sids_member_count` - Fast sorting
- `idx_sids_centerline_gin` - Fast JSONB queries

### `learned_stars` Table

Similar structure to `learned_sids` but for arrival procedures.

## 🔍 Querying the Data

### Using the Python Provider

The `service/learned_data_provider.py` module provides fast, optimized query functions:

```python
from service.learned_data_provider import (
    get_paths_by_route,
    get_tubes_by_route,
    get_sids_by_airport,
    get_stars_by_airport,
    get_all_learned_layers,
    get_statistics
)

# Get paths for a specific route
paths = get_paths_by_route("LLBG", "LLSD", min_member_count=7)

# Get tubes for a route
tubes = get_tubes_by_route("LLBG", "LLSD", min_member_count=6)

# Get SIDs for an airport
sids = get_sids_by_airport("LLBG", min_member_count=3)

# Get STARs for an airport
stars = get_stars_by_airport("LLSD", min_member_count=3)

# Get all layers at once (optimized)
all_layers = get_all_learned_layers(
    origin="LLBG",
    destination="LLSD",
    min_path_members=7,
    min_tube_members=6
)

# Get database statistics
stats = get_statistics()
print(f"Total paths: {stats['paths']['total']}")
```

### Direct SQL Queries

```sql
-- Get paths between two airports
SELECT * FROM learned_paths
WHERE origin = 'LLBG' AND destination = 'LLSD'
    AND member_count >= 7
ORDER BY member_count DESC;

-- Get all tubes for an origin
SELECT * FROM learned_tubes
WHERE origin = 'LLBG'
    AND member_count >= 6
ORDER BY member_count DESC;

-- Get SIDs for an airport
SELECT * FROM learned_sids
WHERE airport = 'LLBG'
ORDER BY member_count DESC;

-- Find most popular routes
SELECT origin, destination, COUNT(*) as path_count, AVG(member_count) as avg_members
FROM learned_paths
GROUP BY origin, destination
ORDER BY avg_members DESC
LIMIT 10;

-- Get airports with most procedures
SELECT airport, COUNT(*) as sid_count
FROM learned_sids
GROUP BY airport
ORDER BY sid_count DESC;
```

## ⚡ Performance Notes

- **Batch Inserts**: Uses `execute_batch` with 1000 records per batch for optimal performance
- **Composite Indexes**: The `idx_paths_od_pair` and `idx_tubes_od_pair` indexes make origin-destination queries extremely fast
- **JSONB with GiST**: GIN indexes on JSONB columns allow fast queries on geometry data
- **Connection Pooling**: Uses the existing PostgreSQL connection pool from `service/pg_provider.py`

### Benchmark (approximate)

- Upload 6000+ records: ~10-15 seconds
- Query single route paths: < 5ms
- Query all paths for an origin: < 10ms
- Get all learned layers (filtered): < 50ms

## 🔄 Updating Data

To update the data after regenerating the JSON files:

1. **Update without dropping** (recommended - uses UPSERT):
   ```bash
   python scripts/upload_learned_data_to_postgres.py
   ```
   
   This will update existing records and insert new ones.

2. **Clean slate** (if structure changed):
   ```bash
   python scripts/upload_learned_data_to_postgres.py --drop-tables
   ```

## 🐛 Troubleshooting

### Connection Error

```
ERROR: POSTGRES_DSN environment variable is required
```

**Solution**: Make sure your `.env` file has the `POSTGRES_DSN` variable set correctly.

### Import Error

```
ERROR: psycopg2 is required
```

**Solution**: Install psycopg2:
```bash
pip install psycopg2-binary
```

### File Not Found

```
⚠️  Paths file not found: rules/learned_paths.json
```

**Solution**: Make sure you're running the script from the project root directory and that the JSON files exist in the `rules/` directory.

### Slow Queries

If queries are slow, check:
1. Indexes are created: `\d learned_paths` in psql
2. PostgreSQL is analyzing tables: `ANALYZE learned_paths;`
3. Connection pool is properly configured in `.env`

## 📝 Notes

- The script uses **UPSERT** (INSERT ... ON CONFLICT DO UPDATE) so it's safe to run multiple times
- All timestamps are in UTC
- JSONB columns store geometry as JSON for flexibility
- For very large datasets (100k+ records), consider increasing the batch size in the script
