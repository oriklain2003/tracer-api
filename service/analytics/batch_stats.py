"""
Batch statistics endpoint for efficient dashboard data fetching.

This module provides a single endpoint that can compute multiple statistics
in one API call, sharing intermediate DataFrames to minimize database queries
and redundant computations.

Performance Benefits:
- Single database connection
- Shared DataFrame loading
- Reduced network round-trips
- Parallel computation where possible

Example:
    POST /api/stats/batch
    {
        "start_ts": 1703001600,
        "end_ts": 1703606400,
        "stats": ["overview", "safety", "traffic"]
    }
    
    Response:
    {
        "overview": {...},
        "safety": {
            "emergency_codes": [...],
            "near_miss": [...],
            "go_arounds": [...]
        },
        "traffic": {
            "flights_per_day": [...],
            "busiest_airports": [...]
        },
        "computed_in_ms": 450
    }
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from pydantic import BaseModel


class BatchStatsRequest(BaseModel):
    """Request model for batch statistics."""
    start_ts: int
    end_ts: int
    stats: List[str]  # e.g., ["overview", "safety", "traffic", "all"]
    cache: Optional[bool] = True
    
    # Fine-grained control for each category
    safety_stats: Optional[List[str]] = None  # ["emergency_codes", "near_miss", "go_arounds"]
    traffic_stats: Optional[List[str]] = None  # ["flights_per_day", "busiest_airports", "signal_loss"]


class BatchStatsResponse(BaseModel):
    """Response model for batch statistics."""
    overview: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    traffic: Optional[Dict[str, Any]] = None
    computed_in_ms: float
    cached: bool = False
    timestamp: int


class BatchStatisticsEngine:
    """Engine for computing multiple statistics in a single pass."""
    
    def __init__(self, stats_engine):
        """
        Initialize batch engine.
        
        Args:
            stats_engine: StatisticsEngine instance (optimized version)
        """
        self.stats_engine = stats_engine
        self._df_cache: Dict[str, pd.DataFrame] = {}
    
    def _load_tracks_dataframe(self, start_ts: int, end_ts: int) -> pd.DataFrame:
        """
        Load all tracks data once for reuse across calculations.
        
        This is the expensive operation we want to do only once.
        """
        cache_key = f"tracks_{start_ts}_{end_ts}"
        if cache_key in self._df_cache:
            return self._df_cache[cache_key]
        
        conn = self.stats_engine._get_connection('research')
        if not conn:
            return pd.DataFrame()
        
        # Single query to fetch all track data
        query = """
        SELECT 
            flight_id, 
            timestamp, 
            lat, 
            lon, 
            alt, 
            callsign,
            'anomaly' as track_type
        FROM anomalies_tracks 
        WHERE timestamp BETWEEN ? AND ?
        UNION ALL
        SELECT 
            flight_id, 
            timestamp, 
            lat, 
            lon, 
            alt, 
            callsign,
            'normal' as track_type
        FROM normal_tracks 
        WHERE timestamp BETWEEN ? AND ?
        """
        
        df = pd.read_sql_query(query, conn, params=(start_ts, end_ts, start_ts, end_ts))
        self._df_cache[cache_key] = df
        return df
    
    def _load_reports_dataframe(self, start_ts: int, end_ts: int) -> pd.DataFrame:
        """
        Load all anomaly reports once for reuse across calculations.
        """
        cache_key = f"reports_{start_ts}_{end_ts}"
        if cache_key in self._df_cache:
            return self._df_cache[cache_key]
        
        conn = self.stats_engine._get_connection('research')
        if not conn:
            return pd.DataFrame()
        
        # Fetch all reports with denormalized columns
        query = """
        SELECT 
            flight_id,
            timestamp,
            full_report,
            matched_rule_ids,
            has_emergency_squawk,
            has_proximity_event,
            has_go_around,
            has_signal_loss
        FROM anomaly_reports
        WHERE timestamp BETWEEN ? AND ?
        """
        
        df = pd.read_sql_query(query, conn, params=(start_ts, end_ts))
        self._df_cache[cache_key] = df
        return df
    
    def compute_overview_from_dataframes(self, df_tracks: pd.DataFrame, df_reports: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute overview stats from pre-loaded DataFrames.
        
        Much faster than multiple queries since data is already in memory.
        """
        stats = {
            'total_flights': int(df_tracks['flight_id'].nunique()) if not df_tracks.empty else 0,
            'total_anomalies': int(df_tracks[df_tracks['track_type'] == 'anomaly']['flight_id'].nunique()) if not df_tracks.empty else 0,
            'safety_events': 0,
            'go_arounds': 0,
            'emergency_codes': 0,
            'near_miss': 0
        }
        
        if not df_reports.empty:
            # Use denormalized columns for fast aggregation
            stats['emergency_codes'] = int(df_reports[df_reports['has_emergency_squawk'] == 1]['flight_id'].nunique())
            stats['near_miss'] = int(df_reports[df_reports['has_proximity_event'] == 1]['flight_id'].nunique())
            stats['go_arounds'] = int(df_reports[df_reports['has_go_around'] == 1]['flight_id'].nunique())
            
            # Safety events: any of the above
            safety_mask = (
                (df_reports['has_emergency_squawk'] == 1) |
                (df_reports['has_proximity_event'] == 1) |
                (df_reports['has_go_around'] == 1)
            )
            stats['safety_events'] = int(df_reports[safety_mask]['flight_id'].nunique())
        
        return stats
    
    def compute_safety_stats_from_dataframes(self, df_reports: pd.DataFrame, 
                                            requested_stats: Set[str]) -> Dict[str, Any]:
        """
        Compute all safety statistics from pre-loaded reports DataFrame.
        
        Args:
            df_reports: Pre-loaded anomaly reports
            requested_stats: Set of requested stat names
        
        Returns:
            Dictionary with requested safety statistics
        """
        result = {}
        
        # Emergency codes
        if 'emergency_codes' in requested_stats:
            if self.stats_engine.use_optimized and self.stats_engine._optimized_engine:
                # Use optimized engine but pass filtered DataFrame
                df_emergency = df_reports[df_reports['has_emergency_squawk'] == 1]
                # For now, fall back to optimized method (could optimize further)
                result['emergency_codes'] = self.stats_engine.get_emergency_codes_stats(
                    int(df_reports['timestamp'].min()) if not df_reports.empty else 0,
                    int(df_reports['timestamp'].max()) if not df_reports.empty else 0
                )
        
        # Near-miss events
        if 'near_miss' in requested_stats or 'near_miss_events' in requested_stats:
            if self.stats_engine.use_optimized and self.stats_engine._optimized_engine:
                result['near_miss'] = self.stats_engine.get_near_miss_events(
                    int(df_reports['timestamp'].min()) if not df_reports.empty else 0,
                    int(df_reports['timestamp'].max()) if not df_reports.empty else 0
                )
        
        # Go-arounds
        if 'go_arounds' in requested_stats:
            if self.stats_engine.use_optimized and self.stats_engine._optimized_engine:
                result['go_arounds'] = self.stats_engine.get_go_around_stats(
                    int(df_reports['timestamp'].min()) if not df_reports.empty else 0,
                    int(df_reports['timestamp'].max()) if not df_reports.empty else 0
                )
        
        return result
    
    def compute_traffic_stats_from_dataframes(self, df_tracks: pd.DataFrame, 
                                             df_reports: pd.DataFrame,
                                             requested_stats: Set[str],
                                             start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Compute all traffic statistics from pre-loaded DataFrames.
        """
        result = {}
        
        # Flights per day
        if 'flights_per_day' in requested_stats:
            if not df_tracks.empty:
                # Compute directly from DataFrame (already in memory)
                df_tracks['date'] = pd.to_datetime(df_tracks['timestamp'], unit='s').dt.date
                
                # Group by date and count flights
                daily = df_tracks.groupby('date')['flight_id'].nunique().reset_index()
                daily.columns = ['date', 'count']
                
                # Convert to list of dicts
                result['flights_per_day'] = [
                    {'date': str(row['date']), 'count': int(row['count'])}
                    for _, row in daily.iterrows()
                ]
        
        # Busiest airports
        if 'busiest_airports' in requested_stats:
            if self.stats_engine.use_optimized and self.stats_engine._optimized_engine:
                # Use optimized method (KDTree is still fastest)
                result['busiest_airports'] = self.stats_engine.get_busiest_airports(
                    start_ts, end_ts, limit=10
                )
        
        # Signal loss
        if 'signal_loss' in requested_stats:
            if not df_reports.empty:
                signal_loss_df = df_reports[df_reports['has_signal_loss'] == 1]
                result['signal_loss'] = {
                    'count': int(signal_loss_df['flight_id'].nunique()),
                    'events': len(signal_loss_df)
                }
        
        return result
    
    def compute_batch_stats(self, request: BatchStatsRequest) -> BatchStatsResponse:
        """
        Compute multiple statistics in a single pass.
        
        This is the main entry point for batch computation.
        """
        start_time = time.perf_counter()
        
        # Determine what data to load
        needs_tracks = 'overview' in request.stats or 'traffic' in request.stats or 'all' in request.stats
        needs_reports = 'overview' in request.stats or 'safety' in request.stats or 'all' in request.stats
        
        # Load data once
        df_tracks = pd.DataFrame()
        df_reports = pd.DataFrame()
        
        if needs_tracks:
            df_tracks = self._load_tracks_dataframe(request.start_ts, request.end_ts)
        
        if needs_reports:
            df_reports = self._load_reports_dataframe(request.start_ts, request.end_ts)
        
        # Compute requested statistics
        response = BatchStatsResponse(
            computed_in_ms=0,
            timestamp=int(time.time())
        )
        
        # Overview stats
        if 'overview' in request.stats or 'all' in request.stats:
            response.overview = self.compute_overview_from_dataframes(df_tracks, df_reports)
        
        # Safety stats
        if 'safety' in request.stats or 'all' in request.stats:
            safety_stats_to_compute = set(request.safety_stats or ['emergency_codes', 'near_miss', 'go_arounds'])
            response.safety = self.compute_safety_stats_from_dataframes(df_reports, safety_stats_to_compute)
        
        # Traffic stats
        if 'traffic' in request.stats or 'all' in request.stats:
            traffic_stats_to_compute = set(request.traffic_stats or ['flights_per_day', 'busiest_airports'])
            response.traffic = self.compute_traffic_stats_from_dataframes(
                df_tracks, df_reports, traffic_stats_to_compute, request.start_ts, request.end_ts
            )
        
        # Clear DataFrame cache to free memory
        self._df_cache.clear()
        
        end_time = time.perf_counter()
        response.computed_in_ms = round((end_time - start_time) * 1000, 2)
        
        return response
    
    def clear_cache(self):
        """Clear the DataFrame cache."""
        self._df_cache.clear()
