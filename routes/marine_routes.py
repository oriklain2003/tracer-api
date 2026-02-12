"""
Marine vessel tracking API routes.

Provides REST API endpoints for accessing marine vessel data:
- List active vessels
- Get vessel details
- Get vessel track history
- Search vessels
- Get traffic statistics
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging

from service import marine_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/marine", tags=["marine"])


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
    
    This endpoint returns vessels that have reported positions within the specified time window.
    Results include position, speed, heading, and navigation status - perfect for map visualization.
    
    Example: /api/marine/vessels?limit=100&since=30&type=Cargo
    """
    try:
        # Parse bounding box if provided
        bbox_tuple = None
        if bbox:
            try:
                bbox_tuple = tuple(map(float, bbox.split(',')))
                if len(bbox_tuple) != 4:
                    raise ValueError("Bounding box must have 4 values")
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid bbox format. Use: south,west,north,east. Error: {str(e)}"
                )
        
        result = marine_service.get_active_vessels(
            limit=limit,
            offset=offset,
            since_minutes=since,
            vessel_type=type,
            bbox=bbox_tuple
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_vessels endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch vessels: {str(e)}")


@router.get("/vessels/{mmsi}")
async def get_vessel(mmsi: str):
    """
    Get detailed vessel information including latest position.
    
    Returns complete vessel metadata (name, type, dimensions, destination) along with
    the most recent position report.
    
    Example: /api/marine/vessels/368207620
    """
    try:
        vessel = marine_service.get_vessel_details(mmsi)
        if not vessel:
            raise HTTPException(status_code=404, detail=f"Vessel with MMSI {mmsi} not found")
        return vessel
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_vessel endpoint for {mmsi}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch vessel details: {str(e)}")


@router.get("/vessels/{mmsi}/track")
async def get_track(
    mmsi: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of track history to retrieve"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of track points")
):
    """
    Get vessel track history for visualization.
    
    Returns a time-ordered list of positions for drawing the vessel's path on a map.
    Useful for understanding vessel movement patterns and routes.
    
    Example: /api/marine/vessels/368207620/track?hours=48
    """
    try:
        result = marine_service.get_vessel_track(mmsi, hours, limit)
        
        if result["points"] == 0:
            # Check if vessel exists at all
            vessel = marine_service.get_vessel_details(mmsi)
            if not vessel:
                raise HTTPException(status_code=404, detail=f"Vessel with MMSI {mmsi} not found")
            # Vessel exists but no track data in time window
            return result
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_track endpoint for {mmsi}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch vessel track: {str(e)}")


@router.get("/search")
async def search(
    q: str = Query(..., min_length=2, description="Search query (vessel name or MMSI)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results")
):
    """
    Search vessels by name or MMSI.
    
    Searches both vessel names (case-insensitive) and MMSI numbers. Results are ordered
    by relevance with exact matches first.
    
    Example: /api/marine/search?q=EVER+GIVEN
    """
    try:
        results = marine_service.search_vessels(q, limit)
        return {
            "query": q,
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in search endpoint for query '{q}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/statistics")
async def get_statistics(
    since: int = Query(1, ge=1, le=168, description="Statistics for last N hours")
):
    """
    Get marine traffic statistics.
    
    Returns aggregated statistics including total vessel count, position reports,
    and breakdowns by vessel type and navigation status.
    
    Example: /api/marine/statistics?since=24
    """
    try:
        result = marine_service.get_marine_statistics(since_hours=since)
        return result
        
    except Exception as e:
        logger.error(f"Error in get_statistics endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch statistics: {str(e)}")
