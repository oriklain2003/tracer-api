# Quick Reference: Learned Data PostgreSQL

## 🚀 Commands

```bash
# Upload data
python scripts/upload_learned_data_to_postgres.py

# Upload with fresh start (WARNING: deletes existing data)
python scripts/upload_learned_data_to_postgres.py --drop-tables

# Test everything
python scripts/test_upload.py

# Test connection only
python -c "from service.pg_provider import test_connection; test_connection()"
```

## 🔍 Quick Queries

```python
from service.learned_data_provider import *

# Get everything for a route (FASTEST)
layers = get_all_learned_layers(origin="LLBG", destination="LLSD")

# Get paths for specific route
paths = get_paths_by_route("LLBG", "LLSD", min_member_count=7)

# Get tubes for specific route
tubes = get_tubes_by_route("LLBG", "LLSD", min_member_count=6)

# Get all paths from an origin
paths = get_all_paths(origin="LLBG", min_member_count=7)

# Get SIDs for airport
sids = get_sids_by_airport("LLBG", min_member_count=3)

# Get STARs for airport
stars = get_stars_by_airport("LLSD", min_member_count=3)

# Get statistics
stats = get_statistics()
print(f"Total paths: {stats['paths']['total']}")
```

## 📊 Database Tables

| Table | Primary Use | Main Index |
|-------|-------------|------------|
| `learned_paths` | Routes between airports | `(origin, destination)` |
| `learned_tubes` | Flight corridors | `(origin, destination)` |
| `learned_sids` | Departures | `(airport)` |
| `learned_stars` | Arrivals | `(airport)` |

## 🔧 Common SQL Queries

```sql
-- Get paths for route
SELECT * FROM learned_paths 
WHERE origin = 'LLBG' AND destination = 'LLSD'
ORDER BY member_count DESC;

-- Get tubes for origin
SELECT * FROM learned_tubes 
WHERE origin = 'LLBG'
ORDER BY member_count DESC;

-- Get SIDs for airport
SELECT * FROM learned_sids 
WHERE airport = 'LLBG'
ORDER BY member_count DESC;

-- Popular routes
SELECT origin, destination, COUNT(*) as paths, AVG(member_count) as avg_flights
FROM learned_paths
GROUP BY origin, destination
ORDER BY avg_flights DESC
LIMIT 10;

-- Total counts
SELECT 
    (SELECT COUNT(*) FROM learned_paths) as paths,
    (SELECT COUNT(*) FROM learned_tubes) as tubes,
    (SELECT COUNT(*) FROM learned_sids) as sids,
    (SELECT COUNT(*) FROM learned_stars) as stars;
```

## ⚡ Performance Tips

1. **Use combined query** for best performance:
   ```python
   # ONE fast query instead of multiple
   layers = get_all_learned_layers(origin="LLBG", destination="LLSD")
   ```

2. **Filter at database level**, not in Python:
   ```python
   # Good - filters in DB
   paths = get_all_paths(origin="LLBG", min_member_count=10)
   
   # Bad - fetches all then filters
   paths = [p for p in get_all_paths() if p['origin'] == 'LLBG']
   ```

3. **Reuse connections** - the provider uses connection pooling automatically

4. **Run ANALYZE** periodically:
   ```sql
   ANALYZE learned_paths;
   ANALYZE learned_tubes;
   ```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "POSTGRES_DSN not found" | Add to `.env`: `POSTGRES_DSN=postgresql://user:pass@host/db` |
| "psycopg2 not found" | `pip install psycopg2-binary` |
| Slow queries | Run `ANALYZE` on tables |
| Empty results | Check `min_member_count` isn't too high |
| Connection errors | Test with `test_connection()` |

## 📁 File Structure

```
scripts/
├── upload_learned_data_to_postgres.py  ← Main upload script
├── test_upload.py                       ← Test script
├── QUICKSTART.md                        ← 2-min guide
├── README_LEARNED_DATA_UPLOAD.md       ← Full docs
└── MIGRATION_GUIDE.md                   ← MongoDB migration

service/
└── learned_data_provider.py             ← Query functions
```

## 🎯 Common Use Cases

### API Endpoint
```python
@router.get("/api/learned-layers")
def get_learned_layers(origin: str = None, destination: str = None):
    from service.learned_data_provider import get_all_learned_layers
    return get_all_learned_layers(origin=origin, destination=destination)
```

### Rule Checking
```python
from service.learned_data_provider import get_paths_by_route

paths = get_paths_by_route(flight.origin, flight.destination)
for path in paths:
    if is_flight_on_path(flight, path):
        return True
return False
```

### Analytics
```python
from service.learned_data_provider import get_statistics

stats = get_statistics()
print(f"Coverage: {stats['paths']['unique_origins']} origins, "
      f"{stats['paths']['unique_destinations']} destinations")
```

## 📝 Response Format

All functions return dicts with these fields:

**Paths:**
```python
{
    'id': 'LLBG_LLSD_3_abc123',
    'origin': 'LLBG',
    'destination': 'LLSD',
    'centerline': [{'lat': 32.0, 'lon': 35.0, 'alt': 1000}, ...],
    'width_nm': 4.0,
    'member_count': 25,
    'min_alt_ft': 1000.0,
    'max_alt_ft': 35000.0
}
```

**Tubes:**
```python
{
    'id': 'LLBG_LLSD_3_abc123',
    'origin': 'LLBG',
    'destination': 'LLSD',
    'geometry': [[32.0, 35.0], [32.1, 35.1], ...],
    'min_alt_ft': 1000.0,
    'max_alt_ft': 35000.0,
    'member_count': 25,
    'buffer_nm': 1.0,
    'alpha': 0.5
}
```

**SIDs/STARs:**
```python
{
    'id': 'LLBG_SID_16_abc123',
    'airport': 'LLBG',
    'type': 'SID',  # or 'STAR'
    'centerline': [{'lat': 32.0, 'lon': 35.0, 'alt': 500}, ...],
    'width_nm': 6.0,
    'member_count': 15,
    'runway': '16'  # optional
}
```

---

**Need more details?** See `README_LEARNED_DATA_UPLOAD.md`
