# AI Classification Routes Implementation

## Overview
This document describes the implementation of two new API routes for AI-powered flight anomaly classification, based on the existing scripts `ai_classify.py` and `batch_classify_flights.py`.

## New Routes

### 1. Single Flight Classification - `/api/ai/classify-flight`
**Method:** POST

**Purpose:** Classify a single flight by flight ID

**Request Body:**
```json
{
  "flight_id": "3d7211ef",
  "schema": "research",
  "force_reclassify": false
}
```

**Response:**
```json
{
  "success": true,
  "flight_id": "3d7211ef",
  "classification_text": "Emergency weather diversion",
  "processing_time_sec": 3.45,
  "gemini_model": "gemini-3-flash-preview",
  "message": "Flight classified successfully"
}
```

**Features:**
- Fetches flight data, metadata, and anomaly report from database
- Sends to Gemini AI for classification
- Saves result to database (simple INSERT, no upsert)
- Skips already classified flights (unless `force_reclassify=true`)
- When `force_reclassify=true`: deletes existing classification before re-classifying
- Returns concise 3-6 word root cause summary

---

### 2. Batch Classification by Date Range - `/api/ai/classify-flights-by-date`
**Method:** POST

**Purpose:** Classify multiple flights within a date range

**Request Body:**
```json
{
  "start_date": "2025-01-01",
  "end_date": "2025-01-31",
  "schema": "research",
  "limit": 100,
  "force_reclassify": false
}
```

**Response:**
```json
{
  "success": true,
  "total": 45,
  "classified": 40,
  "skipped": 3,
  "failed": 2,
  "processing_time_sec": 156.7,
  "errors": [
    {
      "flight_id": "3abc123",
      "error": "Flight data not found"
    }
  ],
  "message": "Classified 40 flights, skipped 3, failed 2"
}
```

**Features:**
- Fetches unclassified anomaly flights in the specified date range
- Processes flights in parallel (2 workers by default)
- Tracks success, skip, and failure counts
- Returns first 10 errors for troubleshooting
- Automatically skips already classified flights

---

## Database Functions Added to `service/pg_provider.py`

### 1. `create_ai_classifications_table(schema: str)`
Creates the `ai_classifications` table if it doesn't exist.

**Table Schema:**
- `flight_id` (TEXT, PRIMARY KEY)
- `classification_text` (TEXT) - The 3-6 word classification
- `confidence_score` (FLOAT)
- `processing_time_sec` (FLOAT)
- `error_message` (TEXT)
- `gemini_model` (TEXT)
- `full_response` (TEXT)
- `created_at` (TIMESTAMP)

---

### 2. `save_ai_classification(flight_id, classification, schema)`
Saves or updates AI classification result in the database.

Uses `ON CONFLICT DO UPDATE` to handle re-classifications.

---

### 3. `fetch_flight_data_for_classification(flight_id, schema)`
Fetches all necessary data for classification:
- Flight metadata (from `flight_metadata` table)
- Track points (from `normal_tracks` or `flight_tracks` table)
- Anomaly report (from `anomaly_reports` table)

Returns a dictionary with `flight_data`, `metadata`, and `anomaly_report`.

---

### 4. `fetch_flights_in_date_range(start_date, end_date, schema, limit)`
Fetches flight IDs of unclassified anomalies within a date range.

Uses a LEFT JOIN to find flights in `anomaly_reports` that don't have entries in `ai_classifications`.

---

## Helper Module - `ai_helpers.py`

### 1. `build_anomaly_context(anomaly_report, metadata, flight_data)`
Builds formatted context text for AI classification, including:
- Flight identification (ID, callsign, flight number)
- Aircraft details (type, model, registration)
- Route information (origin, destination)
- Flight statistics (duration, distance, altitude, speed)
- Special indicators (military, emergency squawks)
- Anomaly analysis (confidence, triggers, matched rules)
- Track data summary

---

### 2. `generate_flight_map(flight_data, width, height)`
Generates a PNG map visualization of the flight path:
- Plots flight path colored by altitude
- Marks start (green) and end (red) points
- Includes grid and legend
- Returns PNG bytes for inclusion in AI request

Uses matplotlib for visualization.

---

## How It Works

### Single Flight Classification Flow:
1. Check if flight is already classified (unless `force_reclassify=true`)
2. Ensure `ai_classifications` table exists
3. Fetch flight data from database using `fetch_flight_data_for_classification()`
4. Initialize AIClassifier with Gemini API key
5. Call `classifier._classify_sync()` which:
   - Builds context using `build_anomaly_context()`
   - Generates map image using `generate_flight_map()`
   - Calls Gemini API with context + image
   - Saves result to database using `save_ai_classification()`
6. Return classification result

### Batch Classification Flow:
1. Ensure `ai_classifications` table exists
2. Fetch unclassified flight IDs in date range using `fetch_flights_in_date_range()`
3. Initialize AIClassifier
4. Process flights in parallel using ThreadPoolExecutor:
   - For each flight: fetch data → classify → save to DB
   - Track success/skip/failure counts
5. Return summary statistics

---

## Usage Examples

### Example 1: Classify a single flight
```bash
curl -X POST http://localhost:8000/api/ai/classify-flight \
  -H "Content-Type: application/json" \
  -d '{
    "flight_id": "3d7211ef",
    "schema": "research"
  }'
```

### Example 2: Classify flights from January 2025
```bash
curl -X POST http://localhost:8000/api/ai/classify-flights-by-date \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-01-01",
    "end_date": "2025-01-31",
    "schema": "research",
    "limit": 50
  }'
```

### Example 3: Force re-classify an already classified flight
```bash
curl -X POST http://localhost:8000/api/ai/classify-flight \
  -H "Content-Type: application/json" \
  -d '{
    "flight_id": "3d7211ef",
    "schema": "research",
    "force_reclassify": true
  }'
```

---

## Environment Variables Required

- `GEMINI_API_KEY` - Google Gemini API key for AI classification
- `POSTGRES_DSN` - PostgreSQL connection string (already configured)

The code falls back to a hardcoded API key if `GEMINI_API_KEY` is not set (same as in the original scripts).

---

## Optional Dependencies

For map visualization in AI classifications, install matplotlib:

```bash
pip install -r requirements-ai.txt
```

Or manually:
```bash
pip install matplotlib numpy
```

**Note:** Map generation is optional. If matplotlib is not available, the AI will still classify flights but without the map image in the context.

---

## Files Modified/Created

### Created:
- `ai_helpers.py` - Helper functions for context building and map generation

### Modified:
- `routes/ai_routes.py` - Added 2 new routes and request/response models
- `service/pg_provider.py` - Added 4 new database functions
- `ai_classify.py` - Updated imports to use `service.pg_provider`
- `batch_classify_flights.py` - Updated imports to use `service.pg_provider`

---

## Notes

1. **Parallel Processing:** The batch route uses 2 parallel workers by default to speed up classification while respecting API rate limits.

2. **Error Handling:** Both routes include comprehensive error handling and logging. Failed classifications are saved to the database with error messages.

3. **Idempotency:** Both routes check for existing classifications and skip them by default, making them safe to run multiple times.

4. **Schemas:** Both routes support different database schemas (`live`, `research`, `feedback`) to work with different datasets. **Default is now `feedback`** which is the most commonly used schema.

5. **AI Model:** Uses `gemini-3-flash-preview` model with the same system instruction as in the original scripts (3-6 word root cause summaries).

6. **Database Integration:** All results are persisted to PostgreSQL for tracking and analysis.

---

## Troubleshooting

### Database Constraint Error

**Error:** `there is no unique or exclusion constraint matching the ON CONFLICT specification`

**Solution:** This error has been fixed! The code now:
1. Uses simple INSERT instead of ON CONFLICT
2. Automatically creates PRIMARY KEY constraint
3. Deletes existing records before re-classifying when `force_reclassify=true`

If you created the table manually without a primary key, the code will try to add it automatically. If that fails, manually drop and recreate:

```sql
-- Be careful - this deletes all classifications!
DROP TABLE IF EXISTS feedback.ai_classifications;

-- The code will recreate it properly on next run
```

### Missing Matplotlib

**Error:** `Missing required library for map generation: No module named 'matplotlib'`

**Solution:** This is optional. Install with:

```bash
pip install matplotlib numpy
```

Or skip map visualization - the AI will still work without it.

### Wrong Schema

**Error:** `Flight data not found` or `No track points found`

**Solution:** Make sure you're using the correct schema:
- `feedback` - For tagged/feedback flights (most common)
- `research` - For research flights
- `live` - For live monitoring flights

Check which schema your flight is in:

```sql
SELECT 'feedback' as schema, COUNT(*) 
FROM feedback.flight_metadata 
WHERE flight_id = 'YOUR_FLIGHT_ID'
UNION ALL
SELECT 'research' as schema, COUNT(*) 
FROM research.flight_metadata 
WHERE flight_id = 'YOUR_FLIGHT_ID';
```

### API Key Issues

**Error:** `Failed to initialize Gemini client`

**Solution:** Make sure `GEMINI_API_KEY` environment variable is set, or the hardcoded fallback key is valid.
