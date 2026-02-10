# API Query Guide for AI Classifications

## Simple SQL Queries

### 1. Get AI Classification for a Flight

```sql
SELECT 
    classification_text,
    processing_time_sec,
    created_at
FROM live.ai_classifications
WHERE flight_id = '3d7211ef';
```

**Result:**
```
classification_text          | processing_time_sec | created_at
-----------------------------+--------------------+--------------------------
Weather Avoidance Maneuver   | 4.23               | 2026-02-09 14:30:15
```

### 2. Get Latest Classifications

```sql
SELECT 
    flight_id,
    classification_text,
    created_at
FROM live.ai_classifications
ORDER BY created_at DESC
LIMIT 20;
```

### 3. Join with Flight Metadata

```sql
SELECT 
    c.flight_id,
    c.classification_text,
    m.callsign,
    m.origin_airport,
    m.destination_airport,
    c.created_at
FROM live.ai_classifications c
JOIN live.flight_metadata m ON c.flight_id = m.flight_id
ORDER BY c.created_at DESC
LIMIT 20;
```

## API Endpoint Examples

### Option 1: Add to Existing API Route (FastAPI)

Add these endpoints to your `api.py` or create a new route file:

```python
from fastapi import APIRouter, HTTPException
from typing import List, Optional
import psycopg2.extras

router = APIRouter()

@router.get("/api/flights/{flight_id}/ai-classification")
def get_flight_ai_classification(flight_id: str, schema: str = "live"):
    """Get AI classification for a specific flight."""
    try:
        from pg_provider import get_connection
        
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT 
                        flight_id,
                        classification_text,
                        confidence_score,
                        processing_time_sec,
                        created_at,
                        error_message,
                        gemini_model
                    FROM {schema}.ai_classifications
                    WHERE flight_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (flight_id,))
                
                result = cursor.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="AI classification not found")
                
                return {
                    "flight_id": result["flight_id"],
                    "classification": result["classification_text"],
                    "confidence_score": result["confidence_score"],
                    "processing_time_sec": result["processing_time_sec"],
                    "created_at": result["created_at"].isoformat() if result["created_at"] else None,
                    "error": result["error_message"],
                    "model": result["gemini_model"]
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ai-classifications/recent")
def get_recent_classifications(
    schema: str = "live",
    limit: int = 20,
    offset: int = 0
):
    """Get recent AI classifications."""
    try:
        from pg_provider import get_connection
        
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT 
                        c.flight_id,
                        c.classification_text,
                        c.created_at,
                        c.processing_time_sec,
                        m.callsign,
                        m.origin_airport,
                        m.destination_airport
                    FROM {schema}.ai_classifications c
                    LEFT JOIN {schema}.flight_metadata m ON c.flight_id = m.flight_id
                    WHERE c.error_message IS NULL
                    ORDER BY c.created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                results = cursor.fetchall()
                
                return {
                    "total": len(results),
                    "limit": limit,
                    "offset": offset,
                    "classifications": [
                        {
                            "flight_id": r["flight_id"],
                            "classification": r["classification_text"],
                            "callsign": r["callsign"],
                            "origin": r["origin_airport"],
                            "destination": r["destination_airport"],
                            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                            "processing_time": r["processing_time_sec"]
                        }
                        for r in results
                    ]
                }
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ai-classifications/stats")
def get_classification_stats(schema: str = "live"):
    """Get AI classification statistics."""
    try:
        from pg_provider import get_connection
        
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE error_message IS NULL) as success,
                        COUNT(*) FILTER (WHERE error_message IS NOT NULL) as failed,
                        AVG(processing_time_sec) FILTER (WHERE error_message IS NULL) as avg_time,
                        MIN(created_at) as first_classification,
                        MAX(created_at) as last_classification
                    FROM {schema}.ai_classifications
                """)
                
                stats = cursor.fetchone()
                
                # Get top classifications
                cursor.execute(f"""
                    SELECT 
                        classification_text,
                        COUNT(*) as count
                    FROM {schema}.ai_classifications
                    WHERE error_message IS NULL
                    GROUP BY classification_text
                    ORDER BY count DESC
                    LIMIT 10
                """)
                
                top_classifications = cursor.fetchall()
                
                return {
                    "total_classifications": stats["total"],
                    "successful": stats["success"],
                    "failed": stats["failed"],
                    "success_rate": round(stats["success"] / stats["total"] * 100, 2) if stats["total"] > 0 else 0,
                    "avg_processing_time_sec": round(stats["avg_time"], 2) if stats["avg_time"] else None,
                    "first_classification": stats["first_classification"].isoformat() if stats["first_classification"] else None,
                    "last_classification": stats["last_classification"].isoformat() if stats["last_classification"] else None,
                    "top_classifications": [
                        {
                            "text": r["classification_text"],
                            "count": r["count"]
                        }
                        for r in top_classifications
                    ]
                }
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Option 2: Simple Python Function

If you just need to query from Python code:

```python
from pg_provider import get_connection
import psycopg2.extras

def get_ai_classification(flight_id: str, schema: str = "live"):
    """Get AI classification for a flight."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT * FROM {schema}.ai_classifications
                WHERE flight_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (flight_id,))
            return cursor.fetchone()

# Usage
classification = get_ai_classification("3d7211ef")
if classification:
    print(f"Classification: {classification['classification_text']}")
```

## API Response Examples

### 1. Get Single Flight Classification

**Request:**
```
GET /api/flights/3d7211ef/ai-classification
```

**Response:**
```json
{
  "flight_id": "3d7211ef",
  "classification": "Weather Avoidance Maneuver",
  "confidence_score": null,
  "processing_time_sec": 4.23,
  "created_at": "2026-02-09T14:30:15",
  "error": null,
  "model": "gemini-3-flash-preview"
}
```

### 2. Get Recent Classifications

**Request:**
```
GET /api/ai-classifications/recent?limit=5
```

**Response:**
```json
{
  "total": 5,
  "limit": 5,
  "offset": 0,
  "classifications": [
    {
      "flight_id": "3d7211ef",
      "classification": "Weather Avoidance Maneuver",
      "callsign": "ELY123",
      "origin": "LLBG",
      "destination": "LCLK",
      "created_at": "2026-02-09T14:30:15",
      "processing_time": 4.23
    },
    {
      "flight_id": "3cf959dd",
      "classification": "Technical Emergency Return",
      "callsign": "ISR456",
      "origin": "LLBG",
      "destination": "LLBG",
      "created_at": "2026-02-09T14:25:42",
      "processing_time": 3.87
    }
  ]
}
```

### 3. Get Statistics

**Request:**
```
GET /api/ai-classifications/stats
```

**Response:**
```json
{
  "total_classifications": 150,
  "successful": 147,
  "failed": 3,
  "success_rate": 98.0,
  "avg_processing_time_sec": 4.12,
  "first_classification": "2026-02-09T10:00:00",
  "last_classification": "2026-02-09T14:30:15",
  "top_classifications": [
    {
      "text": "Weather Avoidance Maneuver",
      "count": 25
    },
    {
      "text": "Diplomatic Route Adjustment",
      "count": 18
    },
    {
      "text": "Technical Emergency Return",
      "count": 15
    }
  ]
}
```

## Frontend Integration

### JavaScript/TypeScript Example

```typescript
// Fetch classification for a flight
async function getFlightClassification(flightId: string) {
  const response = await fetch(`/api/flights/${flightId}/ai-classification`);
  if (!response.ok) {
    console.log("No AI classification available");
    return null;
  }
  const data = await response.json();
  return data.classification;
}

// Usage in React component
function FlightDetails({ flightId }) {
  const [classification, setClassification] = useState(null);
  
  useEffect(() => {
    getFlightClassification(flightId).then(setClassification);
  }, [flightId]);
  
  return (
    <div>
      {classification && (
        <div className="ai-classification">
          <h4>AI Analysis</h4>
          <p>{classification}</p>
        </div>
      )}
    </div>
  );
}
```

## Database Helper Function

Add this to your `pg_provider.py`:

```python
def get_ai_classification(flight_id: str, schema: str = 'live') -> Optional[Dict]:
    """
    Get AI classification for a flight.
    
    Args:
        flight_id: Flight identifier
        schema: Database schema
    
    Returns:
        Dict with classification data or None if not found
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT 
                        flight_id,
                        classification_text,
                        confidence_score,
                        processing_time_sec,
                        created_at,
                        error_message,
                        gemini_model
                    FROM {schema}.ai_classifications
                    WHERE flight_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (flight_id,))
                
                result = cursor.fetchone()
                return dict(result) if result else None
                
    except Exception as e:
        logger.error(f"Error fetching AI classification for {flight_id}: {e}")
        return None
```

## Quick Reference

### Table Schema
```sql
ai_classifications (
    id SERIAL PRIMARY KEY,
    flight_id TEXT NOT NULL,
    classification_text TEXT NOT NULL,    -- The 3-6 word summary
    confidence_score FLOAT,
    full_response TEXT,
    processing_time_sec FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    error_message TEXT,
    gemini_model TEXT
)
```

### Common Queries

```sql
-- Get classification for one flight
SELECT classification_text 
FROM live.ai_classifications 
WHERE flight_id = '3d7211ef';

-- Get latest 10
SELECT flight_id, classification_text, created_at
FROM live.ai_classifications
ORDER BY created_at DESC
LIMIT 10;

-- Count by classification type
SELECT classification_text, COUNT(*)
FROM live.ai_classifications
GROUP BY classification_text
ORDER BY COUNT(*) DESC;

-- Join with anomaly report
SELECT 
    c.classification_text,
    ar.matched_rule_names,
    m.callsign
FROM live.ai_classifications c
JOIN live.anomaly_reports ar ON c.flight_id = ar.flight_id
JOIN live.flight_metadata m ON c.flight_id = m.flight_id
WHERE c.flight_id = '3d7211ef';
```

## Tips

1. **Always check for NULL** - Classification might not exist yet
2. **Use LEFT JOIN** when joining with other tables
3. **Cache results** - Classifications don't change once created
4. **Handle errors gracefully** - Return 404 if not found, not 500

That's it! Simple queries to get AI classifications from your database into your API. 🚀
