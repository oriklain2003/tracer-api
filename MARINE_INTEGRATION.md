# Marine Data Integration Guide for tracer-api

This guide shows how to add marine vessel tracking endpoints to the tracer-api service.

## Quick Start

The marine data is already being collected in the `marine` schema. You just need to add API endpoints to expose it.

## Database Connection

The marine data uses the same PostgreSQL database and connection pool as the flight data.

```python
# tracer-api/service/pg_provider.py already has the connection pool
# Just use the existing get_connection() context manager
```

## Recommended Endpoints to Add

### 1. Add Marine Service Module

Create `tracer-api/service/marine_service.py`:

```python
"""
Marine Vessel Data Service

Provides access to marine vessel tracking data from the marine schema.
"""

from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime, timedelta
import psycopg2.extras

from .pg_provider import get_connection


def get_active_vessels(
    limit: int = 50,
    offset: int = 0,
    since_minutes: int = 10,
    vessel_type: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Dict:
    """
    Get list of active vessels with their latest positions.
    
    Args:
        limit: Maximum number of vessels to return
        offset: Pagination offset
        since_minutes: Consider vessels active if seen in last N minutes
        vessel_type: Filter by vessel type description (e.g., "Cargo", "Tanker")
        bbox: Bounding box filter (south, west, north, east)
    
    Returns:
        Dictionary with vessels list, total count, and pagination info
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Build query
            query = """
                SELECT DISTINCT ON (vp.mmsi)
                    vp.mmsi,
                    vm.vessel_name,
                    vm.vessel_type_description,
                    vp.latitude,
                    vp.longitude,
                    vp.speed_over_ground,
                    vp.course_over_ground,
                    vp.heading,
                    vp.navigation_status,
                    vp.timestamp
                FROM marine.vessel_positions vp
                LEFT JOIN marine.vessel_metadata vm USING (mmsi)
                WHERE vp.timestamp > NOW() - INTERVAL '%s minutes'
            """
            params = [since_minutes]
            
            # Add optional filters
            if vessel_type:
                query += " AND vm.vessel_type_description ILIKE %s"
                params.append(f"%{vessel_type}%")
            
            if bbox:
                south, west, north, east = bbox
                query += """ 
                    AND vp.latitude BETWEEN %s AND %s
                    AND vp.longitude BETWEEN %s AND %s
                """
                params.extend([south, north, west, east])
            
            query += """
                ORDER BY vp.mmsi, vp.timestamp DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            vessels = cursor.fetchall()
            
            # Get total count
            count_query = "SELECT COUNT(DISTINCT mmsi) FROM marine.vessel_positions WHERE timestamp > NOW() - INTERVAL '%s minutes'"
            cursor.execute(count_query, [since_minutes])
            total = cursor.fetchone()['count']
            
            return {
                "vessels": [dict(v) for v in vessels],
                "total": total,
                "limit": limit,
                "offset": offset
            }


def get_vessel_details(mmsi: str) -> Optional[Dict]:
    """
    Get detailed vessel information including latest position.
    
    Args:
        mmsi: Vessel MMSI identifier
    
    Returns:
        Dictionary with vessel details or None if not found
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            query = """
                SELECT 
                    vm.mmsi,
                    vm.vessel_name,
                    vm.callsign,
                    vm.imo_number,
                    vm.vessel_type,
                    vm.vessel_type_description,
                    vm.length,
                    vm.width,
                    vm.draught,
                    vm.destination,
                    vm.eta,
                    vm.first_seen,
                    vm.last_updated,
                    vm.total_position_reports,
                    vp.latitude,
                    vp.longitude,
                    vp.speed_over_ground,
                    vp.course_over_ground,
                    vp.heading,
                    vp.navigation_status,
                    vp.timestamp as last_position_time
                FROM marine.vessel_metadata vm
                LEFT JOIN LATERAL (
                    SELECT *
                    FROM marine.vessel_positions
                    WHERE mmsi = vm.mmsi
                    ORDER BY timestamp DESC
                    LIMIT 1
                ) vp ON true
                WHERE vm.mmsi = %s
            """
            cursor.execute(query, [mmsi])
            vessel = cursor.fetchone()
            
            return dict(vessel) if vessel else None


def get_vessel_track(
    mmsi: str,
    hours: int = 24,
    limit: int = 1000
) -> Dict:
    """
    Get vessel track history.
    
    Args:
        mmsi: Vessel MMSI identifier
        hours: Number of hours of history to retrieve
        limit: Maximum number of points to return
    
    Returns:
        Dictionary with track points and metadata
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Get track points
            track_query = """
                SELECT 
                    latitude,
                    longitude,
                    speed_over_ground,
                    course_over_ground,
                    heading,
                    navigation_status,
                    timestamp
                FROM marine.vessel_positions
                WHERE mmsi = %s
                  AND timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp ASC
                LIMIT %s
            """
            cursor.execute(track_query, [mmsi, hours, limit])
            track = cursor.fetchall()
            
            # Get vessel name
            cursor.execute(
                "SELECT vessel_name FROM marine.vessel_metadata WHERE mmsi = %s",
                [mmsi]
            )
            vessel = cursor.fetchone()
            
            return {
                "mmsi": mmsi,
                "vessel_name": vessel['vessel_name'] if vessel else None,
                "track": [dict(t) for t in track],
                "points": len(track)
            }


def search_vessels(query: str, limit: int = 20) -> List[Dict]:
    """
    Search vessels by name or MMSI.
    
    Args:
        query: Search query string
        limit: Maximum number of results
    
    Returns:
        List of matching vessels
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            search_query = """
                SELECT 
                    vm.mmsi,
                    vm.vessel_name,
                    vm.vessel_type_description,
                    vm.destination,
                    vm.last_updated
                FROM marine.vessel_metadata vm
                WHERE vm.mmsi LIKE %s
                   OR vm.vessel_name ILIKE %s
                ORDER BY 
                    CASE 
                        WHEN vm.mmsi = %s THEN 1
                        WHEN vm.vessel_name ILIKE %s THEN 2
                        ELSE 3
                    END,
                    vm.last_updated DESC
                LIMIT %s
            """
            search_pattern = f"%{query}%"
            cursor.execute(search_query, [
                search_pattern,  # MMSI match
                search_pattern,  # Name match
                query,           # Exact MMSI match (for ordering)
                query,           # Exact name match (for ordering)
                limit
            ])
            results = cursor.fetchall()
            
            return [dict(r) for r in results]


def get_marine_statistics(since_hours: int = 1) -> Dict:
    """
    Get marine traffic statistics.
    
    Args:
        since_hours: Time period to analyze
    
    Returns:
        Dictionary with statistics
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Overall statistics
            stats_query = """
                SELECT 
                    COUNT(DISTINCT mmsi) as total_vessels,
                    COUNT(*) as total_positions
                FROM marine.vessel_positions
                WHERE timestamp > NOW() - INTERVAL '%s hours'
            """
            cursor.execute(stats_query, [since_hours])
            overall = cursor.fetchone()
            
            # By vessel type
            type_query = """
                SELECT 
                    vm.vessel_type_description,
                    COUNT(DISTINCT vp.mmsi) as count
                FROM marine.vessel_positions vp
                LEFT JOIN marine.vessel_metadata vm USING (mmsi)
                WHERE vp.timestamp > NOW() - INTERVAL '%s hours'
                  AND vm.vessel_type_description IS NOT NULL
                GROUP BY vm.vessel_type_description
                ORDER BY count DESC
            """
            cursor.execute(type_query, [since_hours])
            by_type = {row['vessel_type_description']: row['count'] for row in cursor.fetchall()}
            
            # By navigation status
            status_query = """
                SELECT 
                    navigation_status,
                    COUNT(DISTINCT mmsi) as count
                FROM marine.vessel_positions
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                  AND navigation_status IS NOT NULL
                GROUP BY navigation_status
                ORDER BY count DESC
            """
            cursor.execute(status_query, [since_hours])
            by_status = {row['navigation_status']: row['count'] for row in cursor.fetchall()}
            
            now = datetime.utcnow()
            return {
                "period": {
                    "from": (now - timedelta(hours=since_hours)).isoformat() + "Z",
                    "to": now.isoformat() + "Z"
                },
                "total_vessels": overall['total_vessels'],
                "total_positions": overall['total_positions'],
                "by_type": by_type,
                "by_status": by_status
            }
```

### 2. Add API Routes

Create `tracer-api/routes/marine_routes.py`:

```python
"""
Marine vessel tracking API routes.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from ..service import marine_service

router = APIRouter(prefix="/marine", tags=["marine"])


@router.get("/vessels")
async def get_vessels(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of vessels to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    since: int = Query(10, ge=1, le=1440, description="Consider vessels active if seen in last N minutes"),
    type: Optional[str] = Query(None, description="Filter by vessel type (e.g., Cargo, Tanker)"),
    bbox: Optional[str] = Query(None, description="Bounding box: south,west,north,east")
):
    """
    Get list of active vessels with their latest positions.
    
    Example: /api/marine/vessels?limit=100&since=30&type=Cargo
    """
    # Parse bounding box
    bbox_tuple = None
    if bbox:
        try:
            bbox_tuple = tuple(map(float, bbox.split(',')))
            if len(bbox_tuple) != 4:
                raise ValueError()
        except:
            raise HTTPException(status_code=400, detail="Invalid bbox format. Use: south,west,north,east")
    
    return marine_service.get_active_vessels(
        limit=limit,
        offset=offset,
        since_minutes=since,
        vessel_type=type,
        bbox=bbox_tuple
    )


@router.get("/vessels/{mmsi}")
async def get_vessel(mmsi: str):
    """
    Get detailed vessel information including latest position.
    
    Example: /api/marine/vessels/368207620
    """
    vessel = marine_service.get_vessel_details(mmsi)
    if not vessel:
        raise HTTPException(status_code=404, detail="Vessel not found")
    return vessel


@router.get("/vessels/{mmsi}/track")
async def get_track(
    mmsi: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of track history"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum track points")
):
    """
    Get vessel track history.
    
    Example: /api/marine/vessels/368207620/track?hours=48
    """
    return marine_service.get_vessel_track(mmsi, hours, limit)


@router.get("/search")
async def search(
    q: str = Query(..., min_length=2, description="Search query (vessel name or MMSI)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """
    Search vessels by name or MMSI.
    
    Example: /api/marine/search?q=EVER+GIVEN
    """
    results = marine_service.search_vessels(q, limit)
    return {
        "query": q,
        "results": results,
        "total": len(results)
    }


@router.get("/statistics")
async def get_statistics(
    since: int = Query(1, ge=1, le=168, description="Statistics for last N hours")
):
    """
    Get marine traffic statistics.
    
    Example: /api/marine/statistics?since=24
    """
    return marine_service.get_marine_statistics(since_hours=since)
```

### 3. Register Routes in Main App

In `tracer-api/main.py` or wherever you register routes:

```python
from routes import marine_routes

app = FastAPI()

# Register marine routes
app.include_router(marine_routes.router, prefix="/api")
```

## TypeScript Types for Frontend

Create `anomaly-prod/src/types/marine.ts`:

```typescript
export interface VesselPosition {
  mmsi: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  speed_over_ground: number | null;
  course_over_ground: number | null;
  heading: number | null;
  navigation_status: string | null;
}

export interface VesselMetadata {
  mmsi: string;
  vessel_name: string | null;
  callsign: string | null;
  imo_number: string | null;
  vessel_type: number | null;
  vessel_type_description: string | null;
  length: number | null;
  width: number | null;
  draught: number | null;
  destination: string | null;
  eta: string | null;
  first_seen: string;
  last_updated: string;
  total_position_reports: number;
}

export interface Vessel extends VesselMetadata {
  latitude: number;
  longitude: number;
  speed_over_ground: number | null;
  course_over_ground: number | null;
  heading: number | null;
  navigation_status: string | null;
  last_position_time: string;
}

export interface VesselTrack {
  mmsi: string;
  vessel_name: string | null;
  track: VesselPosition[];
  points: number;
}

export interface MarineStatistics {
  period: {
    from: string;
    to: string;
  };
  total_vessels: number;
  total_positions: number;
  by_type: Record<string, number>;
  by_status: Record<string, number>;
}
```

## API Client for Frontend

Create `anomaly-prod/src/api/marine.ts`:

```typescript
import { Vessel, VesselTrack, MarineStatistics } from '../types/marine';

const API_BASE = '/api/marine';

export const marineApi = {
  /**
   * Get list of active vessels
   */
  async getVessels(params?: {
    limit?: number;
    offset?: number;
    since?: number;
    type?: string;
    bbox?: string;
  }): Promise<{
    vessels: Vessel[];
    total: number;
    limit: number;
    offset: number;
  }> {
    const query = new URLSearchParams();
    if (params?.limit) query.append('limit', params.limit.toString());
    if (params?.offset) query.append('offset', params.offset.toString());
    if (params?.since) query.append('since', params.since.toString());
    if (params?.type) query.append('type', params.type);
    if (params?.bbox) query.append('bbox', params.bbox);
    
    const response = await fetch(`${API_BASE}/vessels?${query}`);
    if (!response.ok) throw new Error('Failed to fetch vessels');
    return response.json();
  },

  /**
   * Get vessel details
   */
  async getVessel(mmsi: string): Promise<Vessel> {
    const response = await fetch(`${API_BASE}/vessels/${mmsi}`);
    if (!response.ok) {
      if (response.status === 404) throw new Error('Vessel not found');
      throw new Error('Failed to fetch vessel');
    }
    return response.json();
  },

  /**
   * Get vessel track history
   */
  async getVesselTrack(
    mmsi: string,
    hours: number = 24
  ): Promise<VesselTrack> {
    const response = await fetch(
      `${API_BASE}/vessels/${mmsi}/track?hours=${hours}`
    );
    if (!response.ok) throw new Error('Failed to fetch vessel track');
    return response.json();
  },

  /**
   * Search vessels
   */
  async search(query: string, limit: number = 20): Promise<{
    query: string;
    results: Vessel[];
    total: number;
  }> {
    const response = await fetch(
      `${API_BASE}/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
    if (!response.ok) throw new Error('Search failed');
    return response.json();
  },

  /**
   * Get marine traffic statistics
   */
  async getStatistics(since: number = 1): Promise<MarineStatistics> {
    const response = await fetch(`${API_BASE}/statistics?since=${since}`);
    if (!response.ok) throw new Error('Failed to fetch statistics');
    return response.json();
  },
};
```

## React Component Example

Create `anomaly-prod/src/components/MarineTracker.tsx`:

```tsx
import React, { useEffect, useState } from 'react';
import { marineApi } from '../api/marine';
import { Vessel } from '../types/marine';

export const MarineTracker: React.FC = () => {
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchVessels = async () => {
      try {
        setLoading(true);
        const data = await marineApi.getVessels({ limit: 100, since: 10 });
        setVessels(data.vessels);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch vessels');
      } finally {
        setLoading(false);
      }
    };

    fetchVessels();
    const interval = setInterval(fetchVessels, 30000); // Refresh every 30s

    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading vessels...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h2>Active Vessels ({vessels.length})</h2>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>MMSI</th>
            <th>Type</th>
            <th>Position</th>
            <th>Speed</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {vessels.map((vessel) => (
            <tr key={vessel.mmsi}>
              <td>{vessel.vessel_name || 'Unknown'}</td>
              <td>{vessel.mmsi}</td>
              <td>{vessel.vessel_type_description || 'N/A'}</td>
              <td>
                {vessel.latitude?.toFixed(4)}, {vessel.longitude?.toFixed(4)}
              </td>
              <td>{vessel.speed_over_ground?.toFixed(1) || 'N/A'} kts</td>
              <td>{vessel.navigation_status || 'N/A'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
```

## Testing the Integration

### 1. Test Database Access

```bash
# Connect to database
psql -h tracer-db.cb80eku2emy0.eu-north-1.rds.amazonaws.com \
     -U postgres -d tracer

# Check data exists
SELECT COUNT(*) FROM marine.vessel_positions;
SELECT COUNT(*) FROM marine.vessel_metadata;
```

### 2. Test API Endpoints

```bash
# Get active vessels
curl http://localhost:8000/api/marine/vessels?limit=10

# Get vessel details
curl http://localhost:8000/api/marine/vessels/368207620

# Get vessel track
curl http://localhost:8000/api/marine/vessels/368207620/track?hours=24

# Search vessels
curl http://localhost:8000/api/marine/search?q=EVER

# Get statistics
curl http://localhost:8000/api/marine/statistics?since=1
```

## Performance Tips

1. **Add caching** for vessel metadata (changes infrequently):
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_vessel_metadata_cached(mmsi: str):
       return get_vessel_metadata(mmsi)
   ```

2. **Use pagination** for large result sets

3. **Add indexes** if queries are slow (already done in schema)

4. **Consider Redis** for caching active vessel lists (30-60 second TTL)

## Next Steps

1. Add the service and route files to tracer-api
2. Test endpoints with Postman or curl
3. Add TypeScript types to anomaly-prod
4. Create marine visualization component
5. Add to UI navigation

## Additional Features to Consider

- **Real-time updates**: WebSocket endpoint for live position updates
- **Vessel alerts**: Notify when vessel enters/exits region
- **Traffic heatmaps**: Position density visualization
- **Port analytics**: Track vessels by destination
- **Historical playback**: Replay vessel movements

## Documentation

For complete schema and query documentation, see:
- `tracer-worker/MARINE_DATA_API.md` - Complete API documentation
- `tracer-worker/MARINE_SETUP.md` - Setup guide
- `tracer-worker/create_marine_schema.sql` - Database schema

---

**Need Help?**
- Check existing flight tracking endpoints for patterns
- Review `tracer-api/service/pg_provider.py` for database access
- See `tracer-worker/MARINE_DATA_API.md` for more examples
