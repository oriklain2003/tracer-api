# Scripts Directory

This directory contains scripts for uploading and managing learned behavior data in PostgreSQL.

## 📚 Documentation Index

### 🚀 Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 2 minutes
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command and query reference

### 📖 Comprehensive Guides
- **[README_LEARNED_DATA_UPLOAD.md](README_LEARNED_DATA_UPLOAD.md)** - Complete documentation
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migrate from MongoDB to PostgreSQL
- **[SUMMARY.md](SUMMARY.md)** - Overview of everything created

## 🔧 Available Scripts

### `upload_learned_data_to_postgres.py`
Upload learned paths, tubes, SIDs, and STARs from JSON files to PostgreSQL.

```bash
# Normal upload (UPSERT mode)
python scripts/upload_learned_data_to_postgres.py

# Clean slate (WARNING: deletes all data)
python scripts/upload_learned_data_to_postgres.py --drop-tables
```

**What it does:**
- Reads JSON files from `rules/` directory
- Creates optimized PostgreSQL tables with indexes
- Batch uploads data for performance
- Displays statistics

### `test_upload.py`
Test PostgreSQL connection and query functions.

```bash
python scripts/test_upload.py
```

**What it tests:**
- PostgreSQL connection
- Database statistics
- Path queries
- Tube queries
- SID queries
- Combined queries

## 📊 What Gets Uploaded

| Source File | Target Table | ~Records | Description |
|-------------|--------------|----------|-------------|
| `rules/learned_paths.json` | `learned_paths` | 3000+ | Flight paths between airports |
| `rules/learned_tubes.json` | `learned_tubes` | 3000+ | 3D flight corridors |
| `rules/learned_sid.json` | `learned_sids` | 200+ | Standard Instrument Departures |
| `rules/learned_star.json` | `learned_stars` | 100+ | Standard Terminal Arrivals |

## 🔍 Query Provider

The `service/learned_data_provider.py` module provides optimized query functions:

```python
from service.learned_data_provider import (
    get_paths_by_route,
    get_tubes_by_route,
    get_sids_by_airport,
    get_stars_by_airport,
    get_all_learned_layers,
    get_statistics
)
```

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for usage examples.

## ⚡ Performance

Compared to MongoDB:
- **10-20x faster** for single route queries
- **8-15x faster** for origin-based queries
- **3-5x faster** for combined queries

See [SUMMARY.md](SUMMARY.md) for detailed benchmarks.

## 📋 Prerequisites

1. **PostgreSQL connection**: Set `POSTGRES_DSN` in `.env`
2. **psycopg2**: Already in `requirements.txt`
3. **JSON files**: In `rules/` directory

## 🎯 Quick Start

```bash
# 1. Ensure PostgreSQL is configured
export POSTGRES_DSN="postgresql://user:pass@host:5432/db"

# 2. Upload data
python scripts/upload_learned_data_to_postgres.py

# 3. Test it
python scripts/test_upload.py

# 4. Use in your code
python -c "from service.learned_data_provider import get_statistics; print(get_statistics())"
```

## 🆘 Help

- **Quick help**: See [QUICKSTART.md](QUICKSTART.md)
- **Full documentation**: See [README_LEARNED_DATA_UPLOAD.md](README_LEARNED_DATA_UPLOAD.md)
- **Migration from MongoDB**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Command reference**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

## 📁 Related Files

- **Query Provider**: `service/learned_data_provider.py`
- **Connection Pool**: `service/pg_provider.py`
- **JSON Files**: `rules/learned_*.json`

---

**Need help?** Start with [QUICKSTART.md](QUICKSTART.md) or [QUICK_REFERENCE.md](QUICK_REFERENCE.md)!
