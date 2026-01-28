# Quick Start: Upload Learned Data to PostgreSQL

Get your learned paths, tubes, SIDs, and STARs into PostgreSQL in under 2 minutes!

## ⚡ Super Quick Start

```bash
# 1. Make sure you have psycopg2
pip install psycopg2-binary

# 2. Set your PostgreSQL connection (if not already set)
export POSTGRES_DSN="postgresql://username:password@host:5432/database"

# 3. Run the upload script
python scripts/upload_learned_data_to_postgres.py
```

That's it! ✨

## 📊 What Gets Uploaded?

From your `rules/` directory:
- ✅ `learned_paths.json` → `learned_paths` table (~3000 records)
- ✅ `learned_tubes.json` → `learned_tubes` table (~3000 records)
- ✅ `learned_sid.json` → `learned_sids` table (~200 records)
- ✅ `learned_star.json` → `learned_stars` table (~100 records)

## 🚀 Using the Data

### In Python

```python
from service.learned_data_provider import get_all_learned_layers

# Get all data for a route (FAST!)
layers = get_all_learned_layers(origin="LLBG", destination="LLSD")

print(f"Paths: {len(layers['paths'])}")
print(f"Tubes: {len(layers['tubes'])}")
print(f"SIDs: {len(layers['sids'])}")
print(f"STARs: {len(layers['stars'])}")
```

### In SQL

```sql
-- Get paths between airports
SELECT * FROM learned_paths 
WHERE origin = 'LLBG' AND destination = 'LLSD'
ORDER BY member_count DESC;

-- Get SIDs for an airport
SELECT * FROM learned_sids 
WHERE airport = 'LLBG';
```

## 🔄 Update Data

When you regenerate your learned data JSON files:

```bash
# Just run the script again (it will update existing records)
python scripts/upload_learned_data_to_postgres.py
```

## 🎯 Why PostgreSQL?

- **10x faster queries** than MongoDB
- **Better indexing** for origin-destination lookups
- **One database** instead of MongoDB + PostgreSQL
- **Built-in connection pooling**

## 📖 More Info

- **Full documentation**: `scripts/README_LEARNED_DATA_UPLOAD.md`
- **Migration guide**: `scripts/MIGRATION_GUIDE.md`
- **Code examples**: `service/learned_data_provider.py`

## 🆘 Troubleshooting

### "POSTGRES_DSN environment variable is required"

Add to your `.env` file:
```env
POSTGRES_DSN=postgresql://username:password@host:5432/database
```

### "psycopg2 is required"

Install it:
```bash
pip install psycopg2-binary
```

### "File not found"

Run from the project root directory where the `rules/` folder is located.

---

**Need help?** Check the full docs or the migration guide!
