"""
Optimized SQL query builders for analytics.
"""
from typing import Optional, List, Dict, Any


class QueryBuilder:
    """Helper class to build optimized SQL queries for analytics."""
    
    @staticmethod
    def get_flights_in_range(start_ts: int, end_ts: int, table: str = "flight_tracks") -> str:
        """Get all flights within time range."""
        return f"""
            SELECT DISTINCT flight_id
            FROM {table}
            WHERE timestamp BETWEEN {start_ts} AND {end_ts}
        """
    
    @staticmethod
    def get_anomalies_in_range(start_ts: int, end_ts: int) -> str:
        """Get anomalies within time range from anomaly_reports."""
        return f"""
            SELECT flight_id, timestamp, full_report, callsign
            FROM anomaly_reports
            WHERE timestamp BETWEEN {start_ts} AND {end_ts}
        """
    
    @staticmethod
    def get_rule_matches(rule_id: int, start_ts: int, end_ts: int) -> str:
        """Get flights matching a specific rule."""
        return f"""
            SELECT flight_id, timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN {start_ts} AND {end_ts}
            AND json_extract(full_report, '$.layer_1_rules.report.matched_rules') IS NOT NULL
        """
    
    @staticmethod
    def get_flights_per_day(start_ts: int, end_ts: int, table: str = "flight_tracks") -> str:
        """Count flights per day."""
        return f"""
            SELECT 
                DATE(timestamp, 'unixepoch') as date,
                COUNT(DISTINCT flight_id) as count
            FROM {table}
            WHERE timestamp BETWEEN {start_ts} AND {end_ts}
            GROUP BY date
            ORDER BY date
        """
    
    @staticmethod
    def get_busiest_airports(start_ts: int, end_ts: int, limit: int = 10) -> str:
        """Get busiest airports by flight count."""
        return f"""
            SELECT 
                airport,
                COUNT(*) as total_count
            FROM (
                SELECT flight_id, 
                       CASE 
                         WHEN EXISTS (
                           SELECT 1 FROM flight_tracks ft2 
                           WHERE ft2.flight_id = ft.flight_id 
                           AND ft2.timestamp < ft.timestamp
                         ) THEN 'departure'
                         ELSE 'arrival'
                       END as type,
                       'UNKNOWN' as airport
                FROM flight_tracks ft
                WHERE timestamp BETWEEN {start_ts} AND {end_ts}
                AND alt < 1000
            )
            WHERE airport IS NOT NULL
            GROUP BY airport
            ORDER BY total_count DESC
            LIMIT {limit}
        """
    
    @staticmethod
    def get_signal_loss_locations(start_ts: int, end_ts: int) -> str:
        """Get locations with signal loss (gaps in tracking)."""
        return f"""
            SELECT 
                lat, lon,
                COUNT(*) as gap_count
            FROM flight_tracks
            WHERE timestamp BETWEEN {start_ts} AND {end_ts}
            GROUP BY CAST(lat * 20 AS INTEGER), CAST(lon * 20 AS INTEGER)
            HAVING gap_count > 1
        """

