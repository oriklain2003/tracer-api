# Fixes Applied to AI Classification Routes

## Issues Found

Based on the error log:
```
ERROR:ai_helpers:Missing required library for map generation: No module named 'matplotlib'
WARNING:ai_classify:Map image generation returned None
ERROR:service.pg_provider:Error with PostgreSQL connection: there is no unique or exclusion constraint matching the ON CONFLICT specification
ERROR:service.pg_provider:Error saving AI classification: there is no unique or exclusion constraint matching the ON CONFLICT specification
ERROR:ai_classify:Failed to save classification result for 3d169aeb
```

## Fixes Applied

### 1. ✅ Fixed Database Constraint Error & Simplified Insert Logic

**Problem:** The `ai_classifications` table was missing a PRIMARY KEY constraint, and using `ON CONFLICT DO UPDATE` was causing issues.

**Solution:**
- Updated `create_ai_classifications_table()` to explicitly create PRIMARY KEY constraint
- Added verification to check if constraint exists
- Automatically adds PRIMARY KEY if table exists but constraint is missing
- **Simplified `save_ai_classification()` to use simple INSERT** (no ON CONFLICT)
- Added `sql.SQL()` and `sql.Identifier()` for safe SQL construction
- Routes now handle re-classification by **deleting existing records first** when `force_reclassify=true`

**Files:** `service/pg_provider.py`, `routes/ai_routes.py`

```python
# service/pg_provider.py
from psycopg2 import sql

def save_ai_classification(...):
    insert_query = sql.SQL("""
        INSERT INTO {}.ai_classifications (...)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """).format(sql.Identifier(schema))
    # Simple INSERT - no ON CONFLICT

# routes/ai_routes.py
if force_reclassify:
    # Delete existing record before re-classifying
    cursor.execute(f"DELETE FROM {schema}.ai_classifications WHERE flight_id = %s", ...)
```

### 2. ✅ Improved Matplotlib Handling

**Problem:** Error message was too severe for an optional dependency.

**Solution:**
- Changed error level from ERROR to WARNING
- Added helpful message about installation
- Classification continues without map (map is optional)

**File:** `ai_helpers.py`

```python
logger.warning(f"Optional library matplotlib not available for map generation: {e}")
logger.info("Map generation will be skipped. Install matplotlib with: pip install matplotlib")
```

### 3. ✅ Changed Default Schema to `feedback`

**Problem:** Routes defaulted to `research` schema, but most classifications are for `feedback`.

**Solution:**
- Changed default schema from `"research"` to `"feedback"` in both routes
- Added clearer documentation about schema options

**Files:** `routes/ai_routes.py`

```python
schema: str = Field(default="feedback", description="Database schema to use (feedback, research, live)")
```

### 4. ✅ Better Error Handling

**Problem:** Table creation errors weren't being caught early.

**Solution:**
- Moved table creation to happen BEFORE checking for existing classifications
- Added explicit error checking and HTTP exceptions
- Better logging for troubleshooting

**File:** `routes/ai_routes.py`

```python
# Ensure table exists first (before checking for existing classification)
if not create_ai_classifications_table(request.schema):
    raise HTTPException(
        status_code=500,
        detail=f"Failed to create or verify ai_classifications table in {request.schema} schema"
    )
```

### 5. ✅ Fixed Track Table Selection for Feedback Schema

**Problem:** Wrong track table might be selected for different schemas.

**Solution:**
- Explicit handling for `feedback` schema (uses `flight_tracks`)
- Explicit handling for `research` schema (uses `normal_tracks`)
- Fallback for other schemas

**File:** `service/pg_provider.py`

```python
if schema == "feedback":
    track_table = "flight_tracks"
elif schema == "research":
    track_table = "normal_tracks"
else:
    track_table = "normal_tracks"
```

## Testing the Fix

Try the classification again:

```bash
curl -X POST http://localhost:8000/api/ai/classify-flight \
  -H "Content-Type: application/json" \
  -d '{
    "flight_id": "3d169aeb",
    "schema": "feedback"
  }'
```

You should now see:
- ✅ Table created/verified successfully
- ⚠️ Optional warning about matplotlib (if not installed)
- ✅ Classification saved to database
- ✅ Success response

## Optional: Install Matplotlib

To enable map visualization:

```bash
pip install matplotlib numpy
```

Or use the requirements file:

```bash
pip install -r requirements-ai.txt
```

## Verification Query

Check if the classification was saved:

```sql
SELECT 
    flight_id,
    classification_text,
    processing_time_sec,
    created_at,
    error_message
FROM feedback.ai_classifications
WHERE flight_id = '3d169aeb';
```

Should return the classification result even if matplotlib wasn't installed.
