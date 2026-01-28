"""
Statistics aggregation engine for Level 1 analytics.

Provides methods to compute statistics from flight data including:
- Safety events (emergency codes, near-miss, go-arounds)
- Traffic statistics (flights/day, busiest routes, military tracking)
- Signal loss analysis

Includes caching layer for expensive queries with 1-hour expiry.

OPTIMIZATIONS:
- Optimized methods available using pandas/numpy (10-50x faster)
- Feature flag USE_OPTIMIZED_METHODS to enable/disable
- Original methods kept as _legacy fallback
"""
from __future__ import annotations

import json
import sqlite3
import time
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

# Import core military detection
try:
    from core.military_detection import is_military
except ImportError:
    # Fallback if import fails (e.g. running standalone without path setup)
    is_military = None

from .queries import QueryBuilder

# Feature flag for optimized methods
USE_OPTIMIZED_METHODS = os.environ.get('USE_OPTIMIZED_STATS', 'true').lower() == 'true'


# In-memory cache for expensive queries
_stats_cache: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRY_SECONDS = 3600  # 1 hour


def _get_cache_key(method_name: str, *args) -> str:
    """Generate a unique cache key for a method call."""
    key_data = f"{method_name}:{':'.join(str(a) for a in args)}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_cached(cache_key: str) -> Optional[Any]:
    """Get cached result if not expired."""
    if cache_key in _stats_cache:
        entry = _stats_cache[cache_key]
        if time.time() - entry['timestamp'] < CACHE_EXPIRY_SECONDS:
            return entry['data']
        else:
            # Expired, remove it
            del _stats_cache[cache_key]
    return None


def _set_cached(cache_key: str, data: Any) -> None:
    """Store result in cache."""
    _stats_cache[cache_key] = {
        'data': data,
        'timestamp': time.time()
    }


def clear_stats_cache() -> int:
    """Clear all cached statistics. Returns number of entries cleared."""
    count = len(_stats_cache)
    _stats_cache.clear()
    return count


def get_cache_info() -> Dict[str, Any]:
    """Get cache statistics."""
    now = time.time()
    valid_entries = sum(1 for e in _stats_cache.values() if now - e['timestamp'] < CACHE_EXPIRY_SECONDS)
    return {
        'total_entries': len(_stats_cache),
        'valid_entries': valid_entries,
        'expiry_seconds': CACHE_EXPIRY_SECONDS
    }


class StatisticsEngine:
    """Engine for computing flight statistics with caching."""
    
    def __init__(self, db_paths: Dict[str, Path], use_optimized: bool = None):
        """
        Initialize statistics engine.
        
        Args:
            db_paths: Dictionary mapping db names to paths
                     e.g., {'live': Path('realtime/live_tracks.db'),
                            'research': Path('research.db'),
                            'anomalies': Path('realtime/live_anomalies.db'),
                            'tagged': Path('service/feedback_tagged.db')}
            use_optimized: Override USE_OPTIMIZED_METHODS flag (default None uses env var)
        """
        self.db_paths = db_paths
        self.query_builder = QueryBuilder()
        # Check if tagged DB is available for optimized queries
        self._tagged_available = self._get_connection('tagged') is not None
        
        # Optimization flag
        self.use_optimized = use_optimized if use_optimized is not None else USE_OPTIMIZED_METHODS
        
        # Initialize optimized engine if enabled
        self._optimized_engine = None
        if self.use_optimized:
            try:
                from .statistics_optimized import OptimizedStatisticsEngine
                self._optimized_engine = OptimizedStatisticsEngine(db_paths)
            except ImportError:
                print("Warning: Optimized statistics engine not available, falling back to legacy")
                self.use_optimized = False
    
    def _get_connection(self, db_name: str) -> Optional[sqlite3.Connection]:
        """Get database connection."""
        path = self.db_paths.get(db_name)
        if not path or not path.exists():
            return None
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn
    
    def _execute_query(self, db_name: str, query: str, params: tuple = ()) -> List[tuple]:
        """Execute query and return results."""
        conn = self._get_connection(db_name)
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except sqlite3.OperationalError as e:
            # Handle missing tables gracefully
            if "no such table" in str(e):
                return []
            raise
        finally:
            conn.close()
    
    def _is_military_callsign(self, callsign: str, category: str = None) -> bool:
        """
        Detect if aircraft is military.
        
        Checks both callsign patterns AND the category field from FR24.
        Uses core.military_detection for consistent logic.
        """
        if is_military:
            is_mil, _ = is_military(callsign=callsign, category=category)
            return is_mil

        # Fallback implementation if core module not available
        # Check category field first (from FR24 data)
        if category and category.lower() in ('military_and_government', 'military'):
            return True
        
        if not callsign:
            return False
        
        # Common military callsign patterns
        military_prefixes = [
            'RCH', 'REACH',  # US Air Force
            'RRR',  # Russian Air Force
            'RFF', 'RAF',  # Royal Air Force
            'IAF',  # Israeli Air Force
            'CNV', 'CONVOY',
            'EVAC',
            'DUKE', 'KING', 'VIPER', 'HAWK', 'EAGLE',
            'N00',  # US Navy
            'SHAHD',  # Royal Jordanian Air Force
            'SHAHED',  # Iranian Military Drones
        ]
        
        callsign_upper = callsign.upper()
        for prefix in military_prefixes:
            if callsign_upper.startswith(prefix):
                return True
        
        # Callsigns without typical 3-letter airline prefix
        if len(callsign) <= 4 and not any(c.isdigit() for c in callsign[:3]):
            return False  # Likely civilian
        
        # Numeric-heavy callsigns often military
        if len(callsign) > 5 and sum(c.isdigit() for c in callsign) > 3:
            return True
            
        return False
    
    def get_overview_stats(self, start_ts: int, end_ts: int, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get overview statistics for the dashboard.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            use_cache: Whether to use cached results (default True)
        
        Returns:
            {
                'total_flights': int,
                'total_anomalies': int,
                'safety_events': int,
                'go_arounds': int,
                'emergency_codes': int,
                'near_miss': int
            }
        """
        # Use optimized version if available
        if self.use_optimized and self._optimized_engine:
            # Check cache first
            cache_key = _get_cache_key('get_overview_stats', start_ts, end_ts)
            if use_cache:
                cached = _get_cached(cache_key)
                if cached is not None:
                    return cached
            
            result = self._optimized_engine.get_overview_stats_optimized(start_ts, end_ts)
            if use_cache:
                _set_cached(cache_key, result)
            return result
        
        # Legacy implementation
        # Check cache first
        cache_key = _get_cache_key('get_overview_stats', start_ts, end_ts)
        if use_cache:
            cached = _get_cached(cache_key)
            if cached is not None:
                return cached
        
        stats = {
            'total_flights': 0,
            'total_anomalies': 0,
            'safety_events': 0,
            'go_arounds': 0,
            'emergency_codes': 0,
            'near_miss': 0
        }
        
        # Count total flights - optimized query using timestamp index
        # Simply count distinct flight_ids that have ANY point in the time range
        # This is much faster than computing min/max per flight
        
        # Query research.db tables (anomalies_tracks and normal_tracks)
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT COUNT(DISTINCT flight_id)
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            if results:
                stats['total_flights'] += results[0][0] or 0
        
        # Also check live_tracks.db for real-time data
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM flight_tracks
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('live', query, (start_ts, end_ts))
        if results:
            stats['total_flights'] += results[0][0] or 0
        
        # Count unique flights with anomalies from anomalies_tracks table
        # This ensures we only count flights that have actual track data
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM anomalies_tracks
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        stats['total_anomalies'] = results[0][0] if results else 0
        
        # Count specific event types using denormalized matched_rule_ids field (OPTIMIZED)
        # This is 50-100x faster than parsing JSON
        
        # Rule 1: Emergency squawk
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND (matched_rule_ids LIKE '%1,%' OR matched_rule_ids LIKE '%1' OR matched_rule_ids = '1')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        stats['emergency_codes'] = results[0][0] if results else 0
        
        # Rule 4: Dangerous proximity (near-miss)
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND (matched_rule_ids LIKE '%4,%' OR matched_rule_ids LIKE '%4' OR matched_rule_ids = '4' OR matched_rule_ids LIKE '%, 4%')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        stats['near_miss'] = results[0][0] if results else 0
        
        # Rule 6: Go-around
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND (matched_rule_ids LIKE '%6,%' OR matched_rule_ids LIKE '%6' OR matched_rule_ids = '6' OR matched_rule_ids LIKE '%, 6%')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        stats['go_arounds'] = results[0][0] if results else 0
        
        # Total safety events (any of rules 1, 4, 6)
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND matched_rule_ids IS NOT NULL
            AND (matched_rule_ids LIKE '%1%' OR matched_rule_ids LIKE '%4%' OR matched_rule_ids LIKE '%6%')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        stats['safety_events'] = results[0][0] if results else 0
        
        # Cache the results
        _set_cached(cache_key, stats)
        return stats
    
    def get_emergency_codes_stats(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get emergency code statistics broken down by code and airline.
        
        Returns:
            [{code, count, airline, flights: [...]}]
        """
        # Use optimized version if available
        if self.use_optimized and self._optimized_engine:
            return self._optimized_engine.get_emergency_codes_stats_optimized(start_ts, end_ts)
        
        # Legacy implementation
        query = """
            SELECT flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Group by code, track distinct flights per code
        code_stats = defaultdict(lambda: {'flights': set(), 'airlines': Counter()})
        
        for row in results:
            flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                # Get callsign from the JSON report
                callsign = report.get('summary', {}).get('callsign', '')
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 1:  # Emergency squawk rule
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            code = event.get('squawk', 'UNKNOWN')
                            code_stats[code]['flights'].add(flight_id)
                            
                            # Extract airline from callsign (first 3 chars typically)
                            airline = callsign[:3] if callsign else 'UNKNOWN'
                            code_stats[code]['airlines'][airline] += 1
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Format output with distinct flight counts
        result = []
        for code, data in code_stats.items():
            result.append({
                'code': code,
                'count': len(data['flights']),  # Distinct flights
                'airlines': dict(data['airlines']),
                'flights': list(data['flights'])[:10]  # Limit to 10 examples
            })
        
        return sorted(result, key=lambda x: x['count'], reverse=True)
    
    def get_airline_safety_scorecard(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Generate an airline safety scorecard with letter grades.
        
        Combines multiple safety metrics per airline:
        - Emergency codes (7700, 7600, 7500)
        - Go-arounds
        - Near-miss events
        - Diversions
        
        Each airline gets a letter grade (A-F) based on incident rate per flight.
        
        Returns:
            {
                'scorecards': [{
                    airline, 
                    grade, 
                    score,
                    total_flights,
                    incidents: {emergencies, go_arounds, near_miss, diversions},
                    incident_rate_per_100
                }],
                'summary': {avg_grade, safest_airline, most_incidents_airline}
            }
        """
        # Query all relevant data
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Track incidents per airline
        airline_data = defaultdict(lambda: {
            'total_flights': set(),
            'emergencies': 0,
            'go_arounds': 0,
            'near_miss': 0,
            'diversions': 0
        })
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                callsign = report.get('summary', {}).get('callsign', '')
                flight_id = report.get('flight_id') or report.get('summary', {}).get('flight_id', '')
                
                if not callsign:
                    continue
                
                # Extract airline code (first 3 chars typically)
                airline = callsign[:3].upper()
                
                # Track unique flights
                if flight_id:
                    airline_data[airline]['total_flights'].add(flight_id)
                
                # Check matched rules
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    if rule_id == 1:  # Emergency squawk
                        airline_data[airline]['emergencies'] += 1
                    elif rule_id == 4:  # Proximity/Near-miss
                        airline_data[airline]['near_miss'] += 1
                    elif rule_id == 6:  # Go-around
                        airline_data[airline]['go_arounds'] += 1
                    elif rule_id == 8:  # Diversion
                        airline_data[airline]['diversions'] += 1
                        
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Calculate scores for each airline
        scorecards = []
        
        for airline, data in airline_data.items():
            total_flights = len(data['total_flights'])
            if total_flights < 5:  # Require minimum 5 flights for scoring
                continue
            
            total_incidents = (
                data['emergencies'] * 3 +  # Emergencies weighted 3x
                data['go_arounds'] * 1 +   # Go-arounds 1x
                data['near_miss'] * 2 +    # Near-miss 2x
                data['diversions'] * 1      # Diversions 1x
            )
            
            # Calculate incident rate per 100 flights
            incident_rate = (total_incidents / total_flights) * 100
            
            # Convert to score (0-100, inverse of incident rate)
            # Perfect airline = 100, many incidents = lower score
            raw_score = max(0, 100 - (incident_rate * 5))  # Each 20% incident rate = -100 points
            score = min(100, max(0, raw_score))
            
            # Assign letter grade
            if score >= 90:
                grade = 'A'
            elif score >= 80:
                grade = 'B'
            elif score >= 70:
                grade = 'C'
            elif score >= 60:
                grade = 'D'
            else:
                grade = 'F'
            
            scorecards.append({
                'airline': airline,
                'grade': grade,
                'score': round(score, 1),
                'total_flights': total_flights,
                'incidents': {
                    'emergencies': data['emergencies'],
                    'go_arounds': data['go_arounds'],
                    'near_miss': data['near_miss'],
                    'diversions': data['diversions']
                },
                'total_incidents': total_incidents,
                'incident_rate_per_100': round(incident_rate, 2)
            })
        
        # Sort by score (highest first)
        scorecards.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate summary
        if scorecards:
            avg_score = sum(s['score'] for s in scorecards) / len(scorecards)
            safest = scorecards[0]['airline'] if scorecards else 'N/A'
            most_incidents = max(scorecards, key=lambda x: x['total_incidents'])['airline'] if scorecards else 'N/A'
            
            # Convert avg score to grade
            if avg_score >= 90:
                avg_grade = 'A'
            elif avg_score >= 80:
                avg_grade = 'B'
            elif avg_score >= 70:
                avg_grade = 'C'
            elif avg_score >= 60:
                avg_grade = 'D'
            else:
                avg_grade = 'F'
        else:
            avg_grade = 'N/A'
            avg_score = 0
            safest = 'N/A'
            most_incidents = 'N/A'
        
        return {
            'scorecards': scorecards[:30],  # Top 30 airlines
            'summary': {
                'avg_grade': avg_grade,
                'avg_score': round(avg_score, 1),
                'safest_airline': safest,
                'most_incidents_airline': most_incidents,
                'total_airlines': len(scorecards)
            }
        }
    
    def get_near_miss_events(self, start_ts: int, end_ts: int, 
                            severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get flights that triggered proximity events (Rule 4).
        
        Simple approach: one entry per flight that had a proximity event.
        Returns list of flights with their closest proximity event details.
        
        Args:
            severity: 'high' for < 1nm, 'medium' for >= 1nm, None for all
        
        Returns:
            [{timestamp, flight_id, callsign, other_flight_id, other_callsign, 
              distance_nm, altitude_diff_ft, severity, is_military}]
        """
        # Simple query: get flights with Rule 4, one event per flight (the closest)
        query = """
            SELECT timestamp, flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND (matched_rule_ids LIKE '%4,%' OR matched_rule_ids LIKE '%4' 
                 OR matched_rule_ids = '4' OR matched_rule_ids LIKE '%, 4%')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # One entry per flight - keep the closest proximity event
        flights = {}
        
        for row in results:
            timestamp, flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                # Get callsign from report summary
                summary = report.get('summary', {})
                callsign = summary.get('callsign', '') or ''
                category = summary.get('category', '')
                
                # Check if military using the helper method
                flight_is_military = self._is_military_callsign(callsign, category)
                
                for rule in matched_rules:
                    if rule.get('id') == 4:
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            distance_nm = event.get('distance_nm', 999)
                            alt_diff = event.get('altitude_diff_ft', 9999)
                            
                            # Skip safe vertical separation
                            if alt_diff >= 1000:
                                continue
                            
                            event_severity = 'high' if distance_nm < 1 else 'medium'
                            
                            if severity and event_severity != severity:
                                continue
                            
                            # Get other flight callsign from event if available
                            other_flight_id = event.get('other_flight', event.get('other_aircraft', 'UNKNOWN'))
                            other_callsign = event.get('other_callsign', '') or other_flight_id
                            
                            # Check if other flight is military
                            other_is_military = self._is_military_callsign(other_callsign, '')
                            
                            # Keep closest event per flight
                            if flight_id not in flights or distance_nm < flights[flight_id]['distance_nm']:
                                flights[flight_id] = {
                                    'timestamp': timestamp,
                                    'flight_id': flight_id,
                                    'callsign': callsign or flight_id,
                                    'other_flight_id': other_flight_id,
                                    'other_callsign': other_callsign,
                                    'distance_nm': round(distance_nm, 2),
                                    'altitude_diff_ft': int(alt_diff),
                                    'severity': event_severity,
                                    'is_military': flight_is_military,
                                    'other_is_military': other_is_military
                                }
                        break  # Only process first Rule 4 match
            except (json.JSONDecodeError, KeyError):
                continue
        
        return sorted(flights.values(), key=lambda x: x['timestamp'], reverse=True)
    
    def get_go_around_stats(self, start_ts: int, end_ts: int, 
                           airport: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get go-around statistics.
        
        Returns:
            [{airport, count, avg_per_day, by_hour: {...}}]
        """
        # Query both anomalies db and research db
        all_results = []
        for db_name in ['anomalies', 'research']:
            query = """
                SELECT timestamp, full_report
                FROM anomaly_reports
                WHERE timestamp BETWEEN ? AND ?
            """
            results = self._execute_query(db_name, query, (start_ts, end_ts))
            all_results.extend(results)
        
        # Group by airport
        airport_stats = defaultdict(lambda: {'count': 0, 'by_hour': Counter()})
        results = all_results  # Use combined results
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 6:  # Go-around rule
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            apt = event.get('airport', 'UNKNOWN')
                            if airport and apt != airport:
                                continue
                            
                            airport_stats[apt]['count'] += 1
                            
                            # Extract hour from timestamp
                            hour = datetime.fromtimestamp(timestamp).hour
                            airport_stats[apt]['by_hour'][hour] += 1
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Calculate avg per day
        days = (end_ts - start_ts) / 86400
        result = []
        for apt, data in airport_stats.items():
            result.append({
                'airport': apt,
                'count': data['count'],
                'avg_per_day': round(data['count'] / max(days, 1), 2),
                'by_hour': dict(data['by_hour'])
            })
        
        return sorted(result, key=lambda x: x['count'], reverse=True)
    
    def get_flights_per_day(self, start_ts: int, end_ts: int, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get flight counts per day.
        
        Uses flight_metadata table with is_military field and category='Military_and_government'.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            use_cache: Whether to use cached results (default True)
        
        Returns:
            [{date, count, military_count, civilian_count}]
        """
        # Use optimized version if available
        if self.use_optimized and self._optimized_engine:
            cache_key = _get_cache_key('get_flights_per_day', start_ts, end_ts)
            if use_cache:
                cached = _get_cached(cache_key)
                if cached is not None:
                    return cached
            
            result = self._optimized_engine.get_flights_per_day_optimized(start_ts, end_ts)
            if use_cache:
                _set_cached(cache_key, result)
            return result
        
        # Check cache first
        cache_key = _get_cache_key('get_flights_per_day', start_ts, end_ts)
        if use_cache:
            cached = _get_cached(cache_key)
            if cached is not None:
                return cached
        
        # Use flight_metadata table - much faster and includes category field
        query = """
            SELECT 
                DATE(first_seen_ts, 'unixepoch') as date,
                COUNT(*) as total,
                SUM(CASE WHEN is_military = 1 OR category = 'Military_and_government' THEN 1 ELSE 0 END) as military
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            GROUP BY date
            ORDER BY date
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        result = []
        for row in results:
            date, total, military = row
            if date:
                result.append({
                    'date': date,
                    'count': total or 0,
                    'military_count': military or 0,
                    'civilian_count': (total or 0) - (military or 0)
                })
        
        # Cache the results
        _set_cached(cache_key, result)
        return result
    
    def get_busiest_airports(self, start_ts: int, end_ts: int, 
                            limit: int = 10, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get busiest airports using flight_metadata origin/destination fields.
        
        Uses actual flight data instead of track-based proximity detection
        for accurate airport statistics.
        
        Returns:
            [{airport, arrivals, departures, total}]
        """
        # Check cache first
        cache_key = _get_cache_key('get_busiest_airports', start_ts, end_ts, limit)
        if use_cache:
            cached = _get_cached(cache_key)
            if cached is not None:
                return cached
        
        # Airport name mapping for display
        airport_names = {
            # Israel
            'LLBG': 'Ben Gurion',
            'LLER': 'Ramon',
            'LLHA': 'Haifa',
            'LLOV': 'Ovda',
            'LLET': 'Eilat',
            'LLRD': 'Rosh Pina',
            # Cyprus
            'LCLK': 'Larnaca',
            'LCPH': 'Paphos',
            # Lebanon
            'OLBA': 'Beirut',
            # Jordan
            'OJAI': 'Amman',
            'OJAQ': 'Aqaba',
            # Egypt
            'HECA': 'Cairo',
            'HEAR': 'Hurghada',
            'HESH': 'Sharm El Sheikh',
            'HEGN': 'Hurghada Intl',
            # Syria
            'OSDI': 'Damascus',
            # Gulf States
            'OMDB': 'Dubai',
            'OERK': 'Riyadh',
            'OEJN': 'Jeddah',
            'OEMA': 'Medina',
            'OTHH': 'Doha',
            'OMAA': 'Abu Dhabi',
            'OMSJ': 'Sharjah',
            'OKKK': 'Kuwait',
            'OKBK': 'Kuwait Intl',
            'ORBI': 'Baghdad',
            # Turkey/Greece
            'LGAV': 'Athens',
            'LTFM': 'Istanbul',
            'LTBA': 'Istanbul Ataturk',
        }
        
        airport_stats = defaultdict(lambda: {'arrivals': 0, 'departures': 0})
        
        # Query flight_metadata for departures (origin_airport)
        departures_query = """
            SELECT origin_airport, COUNT(*) as cnt
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
              AND origin_airport IS NOT NULL
              AND origin_airport != ''
            GROUP BY origin_airport
        """
        
        # Query flight_metadata for arrivals (destination_airport)
        arrivals_query = """
            SELECT destination_airport, COUNT(*) as cnt
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
              AND destination_airport IS NOT NULL
              AND destination_airport != ''
            GROUP BY destination_airport
        """
        
        params = (start_ts, end_ts)
        
        # Get departures
        departures_results = self._execute_query('research', departures_query, params)
        for row in departures_results:
            airport, count = row
            if airport:
                airport_stats[airport]['departures'] = count
        
        # Get arrivals
        arrivals_results = self._execute_query('research', arrivals_query, params)
        for row in arrivals_results:
            airport, count = row
            if airport:
                airport_stats[airport]['arrivals'] = count
        
        # Format output
        result = []
        for icao, stats in airport_stats.items():
            total = stats['arrivals'] + stats['departures']
            if total > 0:
                result.append({
                    'airport': icao,
                    'name': airport_names.get(icao, icao),  # Use code as name if unknown
                    'arrivals': stats['arrivals'],
                    'departures': stats['departures'],
                    'total': total
                })
        
        sorted_result = sorted(result, key=lambda x: x['total'], reverse=True)[:limit]
        
        # Cache the results
        _set_cached(cache_key, sorted_result)
        return sorted_result
    
    def get_runway_usage(self, airport: str, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get runway usage statistics by inferring runway from approach/departure heading.
        
        Detects both landings (descending to low altitude) and takeoffs (climbing from low altitude).
        
        Returns:
            [{runway, landings, takeoffs, total}]
        """
        # Runway definitions for Israeli airports
        # Format: runway_name: (min_heading, max_heading)
        runway_configs = {
            'LLBG': {  # Ben Gurion International
                '03': (20, 40),   # Heading 030 +/- 10
                '21': (200, 220),
                '08': (70, 90),   # Heading 080 +/- 10
                '26': (250, 270),
                '12': (110, 130), # Heading 120 +/- 10
                '30': (290, 310)
            },
            'LLER': {  # Ramon International (Eilat)
                '03': (20, 40),
                '21': (200, 220)
            },
            'LLHA': {  # Haifa Airport
                '16': (150, 170),
                '34': (330, 350)
            },
            'LLOV': {  # Ovda (military/civilian)
                '02': (10, 30),
                '20': (190, 210)
            },
            'LLSD': {  # Sde Dov (now closed, but historical data)
                '08': (70, 90),
                '26': (250, 270)
            },
            'LLET': {  # Eilat (old airport)
                '01': (0, 20),
                '19': (180, 200)
            },
            'LLRD': {  # Rosh Pina
                '15': (140, 160),
                '33': (320, 340)
            },
            'LLMZ': {  # Mitzpe Ramon (Airstrip)
                '03': (20, 40),
                '21': (200, 220)
            }
        }
        
        if airport not in runway_configs:
            return []
        
        runways = runway_configs[airport]
        runway_stats = {rwy: {'landings': 0, 'takeoffs': 0} for rwy in runways}
        
        # Airport coordinates for all Israeli airports
        airport_coords = {
            'LLBG': {'lat': 32.0114, 'lon': 34.8867},  # Ben Gurion
            'LLER': {'lat': 29.7255, 'lon': 35.0119},  # Ramon
            'LLHA': {'lat': 32.8094, 'lon': 35.0431},  # Haifa
            'LLOV': {'lat': 29.9403, 'lon': 34.9358},  # Ovda
            'LLSD': {'lat': 32.1147, 'lon': 34.7822},  # Sde Dov
            'LLET': {'lat': 29.5613, 'lon': 34.9601},  # Eilat old
            'LLRD': {'lat': 32.9810, 'lon': 35.5718},  # Rosh Pina
            'LLMZ': {'lat': 30.7761, 'lon': 34.8067}   # Mitzpe Ramon
        }
        
        if airport not in airport_coords:
            return []
        
        coords = airport_coords[airport]
        proximity_nm = 10
        lat_range = proximity_nm / 60
        lon_range = proximity_nm / 60
        
        # Query low altitude flights with heading and altitude
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT flight_id, track, alt, timestamp, gspeed
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                  AND lat BETWEEN ? AND ?
                  AND lon BETWEEN ? AND ?
                  AND alt BETWEEN 0 AND 5000
                ORDER BY flight_id, timestamp
            """
            params = (
                start_ts, end_ts,
                coords['lat'] - lat_range, coords['lat'] + lat_range,
                coords['lon'] - lon_range, coords['lon'] + lon_range
            )
            results = self._execute_query('research', query, params)
            
            # Group by flight
            flight_data = defaultdict(list)
            for row in results:
                flight_id, heading, alt, ts, gspeed = row
                if heading is not None and alt is not None:
                    flight_data[flight_id].append({
                        'ts': ts, 
                        'heading': heading, 
                        'alt': alt,
                        'gspeed': gspeed or 0
                    })
            
            for flight_id, points in flight_data.items():
                if len(points) < 3:
                    continue
                
                # Sort by timestamp
                points.sort(key=lambda x: x['ts'])
                
                # Determine if landing or takeoff by altitude trend
                first_alt = points[0]['alt']
                last_alt = points[-1]['alt']
                min_alt = min(p['alt'] for p in points)
                
                # Find the point at minimum altitude
                min_alt_point = min(points, key=lambda x: x['alt'])
                
                is_landing = False
                is_takeoff = False
                
                # Landing: descending trend, ends at low altitude
                if first_alt > last_alt and min_alt < 1500:
                    is_landing = True
                    heading_at_runway = min_alt_point['heading']
                
                # Takeoff: ascending trend, starts at low altitude  
                elif first_alt < last_alt and min_alt < 1500:
                    is_takeoff = True
                    heading_at_runway = min_alt_point['heading']
                
                # Match to runway
                if is_landing or is_takeoff:
                    for runway, (min_hdg, max_hdg) in runways.items():
                        if min_hdg <= heading_at_runway <= max_hdg:
                            if is_landing:
                                runway_stats[runway]['landings'] += 1
                            else:
                                runway_stats[runway]['takeoffs'] += 1
                            break
        
        # Format output
        result = []
        for runway, stats in runway_stats.items():
            total = stats['landings'] + stats['takeoffs']
            result.append({
                'runway': runway,
                'airport': airport,
                'landings': stats['landings'],
                'takeoffs': stats['takeoffs'],
                'total': total
            })
        
        return sorted(result, key=lambda x: x['total'], reverse=True)
    
    def get_signal_loss_locations(self, start_ts: int, end_ts: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get geographic distribution of signal loss events by detecting gaps in track data.
        
        Improved Algorithm:
        1. Use SQL window functions to detect gaps efficiently
        2. Filter out low altitude gaps (< 5000ft) - normal near airports
        3. Exclude gaps near major airports (within 5nm)
        4. Aggregate by geographic grid (0.25 degrees ~28km for precision)
        5. Classify gaps by duration (brief/medium/extended)
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            limit: Maximum number of zones to return (default 50, prevents browser overload)
        
        Returns:
            [{lat, lon, count, avgDuration, intensity, affected_flights, gap_type, ...}]
        """
        import math
        
        # Configuration
        grid_size = 0.25  # ~28km cells for better precision
        gap_threshold_seconds = 300  # Signal loss if gap > 5 minutes
        min_altitude_ft = 5000  # Exclude low altitude gaps (normal near airports)
        airport_exclusion_nm = 5  # Exclude gaps within 5nm of airports
        
        # Major airports to exclude (signal loss near airports is normal)
        airports = {
            'LLBG': (32.0114, 34.8867),  # Ben Gurion
            'LLER': (29.9403, 35.0004),  # Ramon
            'LLHA': (32.8094, 35.0431),  # Haifa
            'LLOV': (31.2875, 34.7228),  # Ovda
            'OJAI': (31.7226, 35.9932),  # Amman
            'OJAM': (31.9726, 35.9916),  # Marka
            'OLBA': (33.8209, 35.4884),  # Beirut
            'LCRA': (34.5904, 32.9879),  # Akrotiri
            'LCLK': (34.8751, 33.6249),  # Larnaca
            'HECA': (30.1219, 31.4056),  # Cairo
            'HEGN': (27.1783, 33.7994),  # Hurghada
            'HESH': (27.9773, 34.3950),  # Sharm El Sheikh
        }
        
        def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate distance in nautical miles."""
            R = 3440.065
            lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(delta_lon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        def is_near_airport(lat: float, lon: float) -> bool:
            """Check if position is within exclusion radius of any airport."""
            for icao, (apt_lat, apt_lon) in airports.items():
                if haversine_nm(lat, lon, apt_lat, apt_lon) <= airport_exclusion_nm:
                    return True
            return False
        
        def classify_gap(gap_seconds: int) -> str:
            """Classify gap duration."""
            if gap_seconds < 900:  # < 15 min
                return 'brief'
            elif gap_seconds < 3600:  # < 1 hour
                return 'medium'
            else:
                return 'extended'
        
        location_stats = defaultdict(lambda: {
            'count': 0, 
            'total_duration': 0,
            'flights': set(),
            'brief_count': 0,
            'medium_count': 0,
            'extended_count': 0,
            'lat_sum': 0.0,
            'lon_sum': 0.0,
            'timestamps': []
        })
        
        # Use optimized SQL with window functions for gap detection
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            # SQLite supports window functions since 3.25 (2018)
            query = f"""
                WITH ordered_tracks AS (
                    SELECT 
                        flight_id, timestamp, lat, lon, alt,
                        LAG(timestamp) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_ts,
                        LAG(lat) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lat,
                        LAG(lon) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lon,
                        LAG(alt) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_alt
                    FROM {table_name}
                    WHERE timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL 
                      AND lon IS NOT NULL
                )
                SELECT 
                    flight_id, 
                    prev_ts as gap_start,
                    timestamp as gap_end,
                    prev_lat as lat, 
                    prev_lon as lon,
                    prev_alt as alt,
                    (timestamp - prev_ts) as gap_seconds
                FROM ordered_tracks
                WHERE prev_ts IS NOT NULL
                  AND (timestamp - prev_ts) > ?
                  AND (prev_alt IS NULL OR prev_alt > ?)
            """
            results = self._execute_query('research', query, (start_ts, end_ts, gap_threshold_seconds, min_altitude_ft))
            
            for row in results:
                flight_id, gap_start, gap_end, lat, lon, alt, gap_seconds = row
                
                # Skip if near airport
                if is_near_airport(lat, lon):
                    continue
                
                # Grid the location
                grid_lat = round(lat / grid_size) * grid_size
                grid_lon = round(lon / grid_size) * grid_size
                key = (grid_lat, grid_lon)
                
                # Update stats
                cell = location_stats[key]
                cell['count'] += 1
                cell['total_duration'] += gap_seconds
                cell['flights'].add(flight_id)
                cell['lat_sum'] += lat
                cell['lon_sum'] += lon
                cell['timestamps'].append(gap_start)
                
                # Classify gap
                gap_type = classify_gap(gap_seconds)
                cell[f'{gap_type}_count'] += 1
        
        # Format output with centroid calculation
        result = []
        max_count = max((d['count'] for d in location_stats.values()), default=1)
        
        for (grid_lat, grid_lon), data in location_stats.items():
            if data['count'] == 0:
                continue
                
            avg_duration = data['total_duration'] / data['count']
            intensity = min(100, int((data['count'] / max_count) * 100))
            
            # Use centroid of actual positions for better accuracy
            centroid_lat = data['lat_sum'] / data['count']
            centroid_lon = data['lon_sum'] / data['count']
            
            # Determine predominant gap type
            gap_types = {
                'brief': data['brief_count'],
                'medium': data['medium_count'],
                'extended': data['extended_count']
            }
            predominant_type = max(gap_types, key=gap_types.get)
            
            result.append({
                'lat': round(centroid_lat, 4),
                'lon': round(centroid_lon, 4),
                'count': data['count'],
                'avgDuration': int(avg_duration),
                'intensity': intensity,
                'affected_flights': len(data['flights']),
                'gap_type': predominant_type,
                'brief_count': data['brief_count'],
                'medium_count': data['medium_count'],
                'extended_count': data['extended_count'],
                'first_seen': min(data['timestamps']) if data['timestamps'] else None,
                'last_seen': max(data['timestamps']) if data['timestamps'] else None
            })
        
        # Sort by count and limit
        sorted_result = sorted(result, key=lambda x: x['count'], reverse=True)
        return sorted_result[:limit]
    
    def get_signal_loss_clusters(self, start_ts: int, end_ts: int,
                                  cluster_threshold_nm: float = 15,
                                  min_points_for_polygon: int = 3,
                                  limit: int = 100) -> Dict[str, Any]:
        """
        Compute signal loss clusters with polygon (convex hull) boundaries.
        
        Clusters signal loss points that are within cluster_threshold_nm of each other.
        For clusters with 3+ points, computes convex hull polygon.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            cluster_threshold_nm: Distance threshold for clustering in nautical miles (default 15nm)
            min_points_for_polygon: Minimum points required to form a polygon (default 3)
            limit: Maximum number of signal loss points to retrieve (default 100)
        
        Returns:
            {
                'clusters': [{
                    'id': int,
                    'polygon': [[lon, lat], ...] or None,  # GeoJSON coordinate format
                    'centroid': [lon, lat],
                    'point_count': int,
                    'total_events': int,
                    'affected_flights': int,
                    'avg_duration': float,
                    'points': [{lat, lon, count, avgDuration}, ...]
                }],
                'singles': [{lat, lon, count, avgDuration, ...}],  # Points not in clusters
                'total_points': int,
                'total_clusters': int
            }
        """
        import math
        import time as time_module
        
        start_time = time_module.perf_counter()
        print(f"[SIGNAL_LOSS_CLUSTERS] Computing clusters with threshold={cluster_threshold_nm}nm")
        
        # First get the raw signal loss points
        signal_loss_points = self.get_signal_loss_locations(start_ts, end_ts, limit)
        
        if not signal_loss_points:
            return {
                'clusters': [],
                'singles': [],
                'total_points': 0,
                'total_clusters': 0
            }
        
        print(f"[SIGNAL_LOSS_CLUSTERS] Got {len(signal_loss_points)} signal loss points to cluster")
        
        # Haversine distance function
        def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate haversine distance in nautical miles."""
            R = 3440.065  # Earth radius in nm
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
            return R * 2 * math.asin(math.sqrt(a))
        
        # Build clusters using single-linkage clustering
        n_points = len(signal_loss_points)
        used = set()
        clusters = []
        singles = []
        
        for i in range(n_points):
            if i in used:
                continue
            
            # Start a new cluster with this point
            cluster_indices = [i]
            used.add(i)
            
            # Find all points within threshold (transitive closure)
            j = 0
            while j < len(cluster_indices):
                current_idx = cluster_indices[j]
                current_point = signal_loss_points[current_idx]
                current_lat = current_point['lat']
                current_lon = current_point['lon']
                
                for k in range(n_points):
                    if k in used:
                        continue
                    
                    other_point = signal_loss_points[k]
                    dist = haversine_nm(current_lat, current_lon, other_point['lat'], other_point['lon'])
                    if dist <= cluster_threshold_nm:
                        cluster_indices.append(k)
                        used.add(k)
                
                j += 1
            
            # Collect cluster points
            cluster_points = [signal_loss_points[idx] for idx in cluster_indices]
            
            if len(cluster_points) >= min_points_for_polygon:
                # Compute convex hull for polygon
                polygon = None
                if len(cluster_points) >= 3:
                    try:
                        from scipy.spatial import ConvexHull
                        import numpy as np
                        
                        cluster_coords = np.array([[p['lon'], p['lat']] for p in cluster_points])
                        hull = ConvexHull(cluster_coords)
                        hull_points = cluster_coords[hull.vertices].tolist()
                        # Close the polygon
                        hull_points.append(hull_points[0])
                        polygon = hull_points
                    except Exception as e:
                        print(f"[SIGNAL_LOSS_CLUSTERS] ConvexHull failed for cluster: {e}")
                        # Fallback: generate a circle buffer around centroid
                        centroid_lon = sum(p['lon'] for p in cluster_points) / len(cluster_points)
                        centroid_lat = sum(p['lat'] for p in cluster_points) / len(cluster_points)
                        polygon = self._generate_circle_polygon(centroid_lon, centroid_lat, 10)
                
                # Calculate cluster centroid and stats
                centroid_lon = sum(p['lon'] for p in cluster_points) / len(cluster_points)
                centroid_lat = sum(p['lat'] for p in cluster_points) / len(cluster_points)
                total_events = sum(p.get('count', 1) for p in cluster_points)
                affected_flights = sum(p.get('affected_flights', 1) for p in cluster_points)
                avg_duration = sum(p.get('avgDuration', 300) for p in cluster_points) / len(cluster_points)
                
                clusters.append({
                    'id': len(clusters),
                    'polygon': polygon,
                    'centroid': [centroid_lon, centroid_lat],
                    'point_count': len(cluster_points),
                    'total_events': total_events,
                    'affected_flights': affected_flights,
                    'avg_duration': float(avg_duration),
                    'points': [{
                        'lat': p['lat'],
                        'lon': p['lon'],
                        'count': p.get('count', 1),
                        'avgDuration': p.get('avgDuration', 300),
                        'event_count': p.get('count', 1)  # Alias for consistency with GPS jamming
                    } for p in cluster_points]
                })
            else:
                # Single points (not enough for a cluster)
                singles.extend(cluster_points)
        
        total_time = time_module.perf_counter() - start_time
        print(f"[SIGNAL_LOSS_CLUSTERS] Found {len(clusters)} clusters and {len(singles)} singles in {total_time:.2f}s")
        
        return {
            'clusters': clusters,
            'singles': singles,
            'total_points': len(signal_loss_points),
            'total_clusters': len(clusters)
        }
    
    def _generate_circle_polygon(self, center_lon: float, center_lat: float, radius_nm: float, num_points: int = 32) -> List[List[float]]:
        """Generate a circular polygon around a center point."""
        import math
        
        # Convert radius from nm to degrees (approximate)
        # 1 degree latitude ≈ 60 nm
        radius_deg = radius_nm / 60.0
        
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            lat = center_lat + radius_deg * math.cos(angle)
            # Adjust longitude for latitude
            lon = center_lon + (radius_deg / math.cos(math.radians(center_lat))) * math.sin(angle)
            points.append([lon, lat])
        
        # Close the polygon
        points.append(points[0])
        return points
    
    def get_coverage_gap_zones(self, start_ts: int, end_ts: int,
                                buffer_radius_nm: float = 20,
                                min_events_for_zone: int = 2,
                                merge_threshold_nm: float = 30,
                                limit: int = 50) -> Dict[str, Any]:
        """
        Compute COVERAGE GAP ZONES - estimated areas where flights disappear.
        
        OPTIMIZED: Uses scipy cKDTree for O(n log n) spatial clustering instead of O(n²).
        
        Algorithm:
        1. Get all signal loss events (gap > 5min at altitude > 5000ft) via SQL
        2. Use KDTree for efficient neighbor lookup
        3. Cluster with Union-Find for fast merging
        4. Create expanded polygon zones with statistics
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            buffer_radius_nm: Base buffer radius around loss points (default 20nm)
            min_events_for_zone: Minimum events to form a zone (default 2)
            merge_threshold_nm: Distance to merge nearby zones (default 30nm)
            limit: Maximum zones to return (default 50)
        
        Returns:
            Dict with 'zones', 'total_events', 'total_zones', 'coverage_summary'
        """
        import time as time_module
        import numpy as np
        from scipy.spatial import cKDTree
        
        start_time = time_module.perf_counter()
        print(f"[COVERAGE_GAP_ZONES] Computing coverage gap zones with buffer={buffer_radius_nm}nm")
        
        # Configuration
        gap_threshold_seconds = 300  # Signal loss if gap > 5 minutes
        min_altitude_ft = 5000
        
        # Pre-compute airport positions as numpy array for vectorized distance check
        airports_arr = np.array([
            [32.0114, 34.8867], [29.9403, 35.0004], [32.8094, 35.0431], [31.2875, 34.7228],
            [31.7226, 35.9932], [31.9726, 35.9916], [33.8209, 35.4884], [34.5904, 32.9879],
            [34.8751, 33.6249], [32.1147, 34.7822], [30.1219, 31.4056], [27.1783, 33.7994],
            [33.4114, 36.5156], [36.1807, 37.2244]
        ])
        airport_exclusion_deg = 8 / 60.0  # ~8nm in degrees (approximate)
        
        # Step 1: Single optimized SQL query for both tables
        loss_events = []
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                WITH ordered_tracks AS (
                    SELECT 
                        flight_id, timestamp, lat, lon, alt,
                        LAG(timestamp) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_ts,
                        LAG(lat) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lat,
                        LAG(lon) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lon,
                        LAG(alt) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_alt
                    FROM {table_name}
                    WHERE timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL AND lon IS NOT NULL
                )
                SELECT flight_id, prev_ts, timestamp, prev_lat, prev_lon, prev_alt, (timestamp - prev_ts)
                FROM ordered_tracks
                WHERE prev_ts IS NOT NULL
                  AND (timestamp - prev_ts) > ?
                  AND (prev_alt IS NULL OR prev_alt > ?)
            """
            results = self._execute_query('research', query, (start_ts, end_ts, gap_threshold_seconds, min_altitude_ft))
            
            for row in results:
                flight_id, gap_start, gap_end, lat, lon, alt, gap_seconds = row
                if lat is None or lon is None:
                    continue
                loss_events.append([lat, lon, flight_id, gap_start, gap_seconds])
        
        if not loss_events:
            return {'zones': [], 'total_events': 0, 'total_zones': 0, 
                    'coverage_summary': {'total_gap_area_sq_nm': 0, 'avg_zone_risk': 0, 'hotspot_regions': []}}
        
        # Convert to numpy for vectorized operations
        events_arr = np.array([[e[0], e[1]] for e in loss_events])  # lat, lon
        
        # Vectorized airport exclusion using broadcasting
        if len(airports_arr) > 0:
            dists_to_airports = np.min(np.sqrt(
                (events_arr[:, 0:1] - airports_arr[:, 0])**2 + 
                (events_arr[:, 1:2] - airports_arr[:, 1])**2
            ), axis=1)
            valid_mask = dists_to_airports > airport_exclusion_deg
            events_arr = events_arr[valid_mask]
            loss_events = [loss_events[i] for i in range(len(loss_events)) if valid_mask[i]]
        
        if len(loss_events) == 0:
            return {'zones': [], 'total_events': 0, 'total_zones': 0,
                    'coverage_summary': {'total_gap_area_sq_nm': 0, 'avg_zone_risk': 0, 'hotspot_regions': []}}
        
        n_events = len(loss_events)
        print(f"[COVERAGE_GAP_ZONES] {n_events} events after airport filtering")
        
        # Step 2: EFFICIENT CLUSTERING using KDTree + Union-Find
        # Convert merge threshold from nm to degrees (approximate: 1 deg ≈ 60nm)
        merge_threshold_deg = (buffer_radius_nm * 2 + merge_threshold_nm) / 60.0
        
        # Build KDTree for O(log n) neighbor queries
        tree = cKDTree(events_arr)
        
        # Union-Find for efficient cluster merging
        parent = list(range(n_events))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Find all pairs within threshold using KDTree (O(n log n) average)
        pairs = tree.query_pairs(r=merge_threshold_deg, output_type='ndarray')
        for i, j in pairs:
            union(i, j)
        
        # Group events by cluster
        from collections import defaultdict
        clusters = defaultdict(list)
        for i in range(n_events):
            clusters[find(i)].append(i)
        
        # Filter clusters by minimum size
        raw_zones = [indices for indices in clusters.values() if len(indices) >= min_events_for_zone]
        print(f"[COVERAGE_GAP_ZONES] Created {len(raw_zones)} zones via KDTree clustering")
        
        # Helper functions
        def calculate_buffer_radius(gap_duration_sec: float) -> float:
            if gap_duration_sec < 900: return buffer_radius_nm
            elif gap_duration_sec < 1800: return buffer_radius_nm * 1.5
            elif gap_duration_sec < 3600: return buffer_radius_nm * 2.0
            else: return buffer_radius_nm * 2.5
        
        def classify_gap(gap_seconds: int) -> str:
            if gap_seconds < 900: return 'brief'
            elif gap_seconds < 3600: return 'medium'
            else: return 'extended'
        
        # Step 3: Create zone polygons and calculate statistics (vectorized where possible)
        from scipy.spatial import ConvexHull
        zones = []
        total_area = 0
        
        for zone_idx, indices in enumerate(raw_zones):
            # Extract zone data using indices
            zone_lats = events_arr[indices, 0]
            zone_lons = events_arr[indices, 1]
            zone_data = [loss_events[i] for i in indices]
            
            # Vectorized centroid calculation
            centroid_lat = np.mean(zone_lats)
            centroid_lon = np.mean(zone_lons)
            
            # Calculate zone statistics
            flight_ids = set(e[2] for e in zone_data)  # flight_id at index 2
            gap_durations = [e[4] for e in zone_data]  # gap_duration at index 4
            avg_gap = np.mean(gap_durations)
            max_gap = max(gap_durations)
            timestamps = [e[3] for e in zone_data]  # gap_start at index 3
            
            gap_type = classify_gap(int(avg_gap))
            
            # Calculate risk score
            event_factor = min(50, len(indices) * 5)
            duration_factor = min(30, (avg_gap / 1800) * 30)
            flight_factor = min(20, len(flight_ids) * 2)
            risk_score = min(100, event_factor + duration_factor + flight_factor)
            
            # Calculate max buffer for polygon expansion
            max_buffer = max(calculate_buffer_radius(d) for d in gap_durations)
            
            # Create polygon
            polygon = None
            area_sq_nm = 0
            
            try:
                if len(indices) >= 3:
                    coords_array = np.column_stack([zone_lons, zone_lats])
                    hull = ConvexHull(coords_array)
                    hull_points = coords_array[hull.vertices].tolist()
                    
                    # Expand hull outward
                    expanded_points = self._expand_polygon(hull_points, centroid_lon, centroid_lat, max_buffer)
                    expanded_points.append(expanded_points[0])
                    polygon = expanded_points
                    
                    # Quick area estimate
                    lons = [p[0] for p in expanded_points]
                    lats = [p[1] for p in expanded_points]
                    width_deg = max(lons) - min(lons)
                    height_deg = max(lats) - min(lats)
                    area_sq_nm = (width_deg * 60) * (height_deg * 60) * 0.7
                else:
                    avg_buffer = np.mean([calculate_buffer_radius(d) for d in gap_durations])
                    polygon = self._generate_circle_polygon(centroid_lon, centroid_lat, avg_buffer, 16)
                    area_sq_nm = 3.14159 * avg_buffer * avg_buffer
            except Exception as e:
                avg_buffer = buffer_radius_nm * 1.5
                polygon = self._generate_circle_polygon(centroid_lon, centroid_lat, avg_buffer, 16)
                area_sq_nm = 3.14159 * avg_buffer * avg_buffer
            
            total_area += area_sq_nm
            
            zones.append({
                'id': zone_idx,
                'polygon': polygon,
                'centroid': [float(centroid_lon), float(centroid_lat)],
                'area_sq_nm': round(area_sq_nm, 1),
                'event_count': len(indices),
                'affected_flights': len(flight_ids),
                'avg_gap_duration_sec': int(avg_gap),
                'max_gap_duration_sec': int(max_gap),
                'risk_score': round(risk_score, 1),
                'gap_type': gap_type,
                'first_seen': min(timestamps) if timestamps else None,
                'last_seen': max(timestamps) if timestamps else None,
                'points': [{
                    'lat': zone_data[i][0], 'lon': zone_data[i][1],
                    'gap_duration': zone_data[i][4], 'timestamp': zone_data[i][3],
                    'flight_id': zone_data[i][2]
                } for i in range(min(15, len(zone_data)))]
            })
        
        # Sort by risk score (highest first)
        zones.sort(key=lambda z: z['risk_score'], reverse=True)
        zones = zones[:limit]
        
        # Identify hotspot regions
        hotspot_regions = self._identify_hotspot_regions(zones)
        
        avg_risk = sum(z['risk_score'] for z in zones) / len(zones) if zones else 0
        
        total_time = time_module.perf_counter() - start_time
        print(f"[COVERAGE_GAP_ZONES] Created {len(zones)} coverage gap zones in {total_time:.2f}s")
        
        return {
            'zones': zones,
            'total_events': n_events,
            'total_zones': len(zones),
            'coverage_summary': {
                'total_gap_area_sq_nm': round(total_area, 1),
                'avg_zone_risk': round(avg_risk, 1),
                'hotspot_regions': hotspot_regions
            }
        }
    
    def _expand_polygon(self, hull_points: List[List[float]], centroid_lon: float, centroid_lat: float, 
                        buffer_nm: float) -> List[List[float]]:
        """Expand a polygon outward from its centroid by buffer distance."""
        import math
        
        expanded = []
        for point in hull_points:
            lon, lat = point
            
            # Calculate direction from centroid to point
            delta_lon = lon - centroid_lon
            delta_lat = lat - centroid_lat
            
            # Distance from centroid
            dist = math.sqrt(delta_lon**2 + delta_lat**2)
            if dist == 0:
                expanded.append([lon, lat])
                continue
            
            # Convert buffer_nm to degrees (approximate)
            buffer_deg = buffer_nm / 60.0
            
            # Expand outward
            scale = 1 + (buffer_deg / dist) if dist > 0 else 1
            new_lon = centroid_lon + delta_lon * scale
            new_lat = centroid_lat + delta_lat * scale
            
            expanded.append([new_lon, new_lat])
        
        return expanded
    
    def _identify_hotspot_regions(self, zones: List[Dict[str, Any]]) -> List[str]:
        """Identify named regions where coverage gaps are concentrated."""
        regions = []
        
        # Define known regions with their approximate bounds
        region_bounds = {
            'Syria Border': {'lat_min': 32.5, 'lat_max': 37.0, 'lon_min': 35.5, 'lon_max': 42.0},
            'Lebanon': {'lat_min': 33.0, 'lat_max': 34.5, 'lon_min': 35.0, 'lon_max': 36.5},
            'Mediterranean Sea': {'lat_min': 31.0, 'lat_max': 36.0, 'lon_min': 28.0, 'lon_max': 35.0},
            'Sinai Peninsula': {'lat_min': 28.0, 'lat_max': 31.5, 'lon_min': 32.5, 'lon_max': 35.0},
            'Jordan': {'lat_min': 29.0, 'lat_max': 33.5, 'lon_min': 35.0, 'lon_max': 39.0},
            'Cyprus': {'lat_min': 34.5, 'lat_max': 35.7, 'lon_min': 32.0, 'lon_max': 34.5},
            'Northern Israel': {'lat_min': 32.5, 'lat_max': 33.5, 'lon_min': 34.5, 'lon_max': 36.0},
            'Gaza Border': {'lat_min': 31.0, 'lat_max': 31.8, 'lon_min': 34.0, 'lon_max': 34.8},
        }
        
        region_counts = defaultdict(int)
        
        for zone in zones:
            centroid = zone['centroid']
            lon, lat = centroid[0], centroid[1]
            
            for region_name, bounds in region_bounds.items():
                if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                    bounds['lon_min'] <= lon <= bounds['lon_max']):
                    region_counts[region_name] += zone['event_count']
        
        # Return top 3 regions by event count
        sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_regions[:3]]
    
    def get_near_miss_clusters(self, start_ts: int, end_ts: int,
                               cluster_threshold_nm: float = 30,
                               min_points_for_polygon: int = 3,
                               limit: int = 100) -> Dict[str, Any]:
        """
        Compute near-miss event clusters with polygon (convex hull) boundaries.
        
        Clusters near-miss points that are within cluster_threshold_nm of each other.
        For clusters with 3+ points, computes convex hull polygon.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            cluster_threshold_nm: Distance threshold for clustering in nautical miles (default 30nm)
            min_points_for_polygon: Minimum points required to form a polygon (default 3)
            limit: Maximum number of near-miss points to retrieve (default 100)
        
        Returns:
            {
                'clusters': [{
                    'id': int,
                    'polygon': [[lon, lat], ...] or None,  # GeoJSON coordinate format
                    'centroid': [lon, lat],
                    'point_count': int,
                    'total_events': int,
                    'severity_high': int,
                    'severity_medium': int,
                    'points': [{lat, lon, count, severity_high, severity_medium}, ...]
                }],
                'singles': [{lat, lon, count, severity_high, severity_medium, ...}],
                'total_points': int,
                'total_clusters': int
            }
        """
        import math
        import time as time_module
        
        start_time = time_module.perf_counter()
        print(f"[NEAR_MISS_CLUSTERS] Computing clusters with threshold={cluster_threshold_nm}nm")
        
        # First get the raw near-miss location points
        near_miss_points = self.get_near_miss_locations(start_ts, end_ts, limit)
        
        if not near_miss_points:
            return {
                'clusters': [],
                'singles': [],
                'total_points': 0,
                'total_clusters': 0
            }
        
        print(f"[NEAR_MISS_CLUSTERS] Got {len(near_miss_points)} near-miss points to cluster")
        
        # Haversine distance function
        def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate haversine distance in nautical miles."""
            R = 3440.065  # Earth radius in nm
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
            return R * 2 * math.asin(math.sqrt(a))
        
        # Build clusters using single-linkage clustering
        n_points = len(near_miss_points)
        used = set()
        clusters = []
        singles = []
        
        for i in range(n_points):
            if i in used:
                continue
            
            # Start a new cluster with this point
            cluster_indices = [i]
            used.add(i)
            
            # Find all points within threshold (transitive closure)
            j = 0
            while j < len(cluster_indices):
                current_idx = cluster_indices[j]
                current_point = near_miss_points[current_idx]
                current_lat = current_point['lat']
                current_lon = current_point['lon']
                
                for k in range(n_points):
                    if k in used:
                        continue
                    
                    other_point = near_miss_points[k]
                    dist = haversine_nm(current_lat, current_lon, other_point['lat'], other_point['lon'])
                    if dist <= cluster_threshold_nm:
                        cluster_indices.append(k)
                        used.add(k)
                
                j += 1
            
            # Collect cluster points
            cluster_points = [near_miss_points[idx] for idx in cluster_indices]
            
            if len(cluster_points) >= min_points_for_polygon:
                # Compute convex hull for polygon
                polygon = None
                if len(cluster_points) >= 3:
                    try:
                        from scipy.spatial import ConvexHull
                        import numpy as np
                        
                        cluster_coords = np.array([[p['lon'], p['lat']] for p in cluster_points])
                        hull = ConvexHull(cluster_coords)
                        hull_points = cluster_coords[hull.vertices].tolist()
                        # Close the polygon
                        hull_points.append(hull_points[0])
                        polygon = hull_points
                    except Exception as e:
                        print(f"[NEAR_MISS_CLUSTERS] ConvexHull failed for cluster: {e}")
                        # Fallback: generate a circle buffer around centroid
                        centroid_lon = sum(p['lon'] for p in cluster_points) / len(cluster_points)
                        centroid_lat = sum(p['lat'] for p in cluster_points) / len(cluster_points)
                        polygon = self._generate_circle_polygon(centroid_lon, centroid_lat, 15)
                
                # Calculate cluster centroid and stats
                centroid_lon = sum(p['lon'] for p in cluster_points) / len(cluster_points)
                centroid_lat = sum(p['lat'] for p in cluster_points) / len(cluster_points)
                total_events = sum(p.get('count', 1) for p in cluster_points)
                severity_high = sum(p.get('severity_high', 0) for p in cluster_points)
                severity_medium = sum(p.get('severity_medium', 0) for p in cluster_points)
                
                # Collect sample flight IDs from all points in cluster
                all_flight_ids = []
                for p in cluster_points:
                    sample_ids = p.get('sample_flight_ids', [])
                    for fid in sample_ids:
                        if fid and fid not in all_flight_ids:
                            all_flight_ids.append(fid)
                            if len(all_flight_ids) >= 10:
                                break
                    if len(all_flight_ids) >= 10:
                        break
                
                clusters.append({
                    'id': len(clusters),
                    'polygon': polygon,
                    'centroid': [centroid_lon, centroid_lat],
                    'point_count': len(cluster_points),
                    'total_events': total_events,
                    'severity_high': severity_high,
                    'severity_medium': severity_medium,
                    'sample_flight_ids': all_flight_ids[:5],  # Top 5 sample flight IDs
                    'points': [{
                        'lat': p['lat'],
                        'lon': p['lon'],
                        'count': p.get('count', 1),
                        'severity_high': p.get('severity_high', 0),
                        'severity_medium': p.get('severity_medium', 0),
                        'event_count': p.get('count', 1)  # Alias for consistency
                    } for p in cluster_points]
                })
            else:
                # Single points (not enough for a cluster)
                singles.extend(cluster_points)
        
        total_time = time_module.perf_counter() - start_time
        print(f"[NEAR_MISS_CLUSTERS] Found {len(clusters)} clusters and {len(singles)} singles in {total_time:.2f}s")
        
        return {
            'clusters': clusters,
            'singles': singles,
            'total_points': len(near_miss_points),
            'total_clusters': len(clusters)
        }
    
    def get_signal_loss_monthly(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get monthly breakdown of signal loss events.
        
        Answers the demand: "Was there a specific month with more signal losses?"
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
        
        Returns:
            [{month: 'YYYY-MM', count: int, affected_flights: int, avg_duration: float}]
        """
        import math
        from datetime import datetime
        
        gap_threshold_seconds = 300
        min_altitude_ft = 5000
        airport_exclusion_nm = 5
        
        airports = {
            'LLBG': (32.0114, 34.8867), 'LLER': (29.9403, 35.0004),
            'LLHA': (32.8094, 35.0431), 'OJAI': (31.7226, 35.9932),
            'OLBA': (33.8209, 35.4884), 'LCLK': (34.8751, 33.6249),
            'HECA': (30.1219, 31.4056),
        }
        
        def haversine_nm(lat1, lon1, lat2, lon2):
            R = 3440.065
            lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(delta_lon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        def is_near_airport(lat, lon):
            for apt_lat, apt_lon in airports.values():
                if haversine_nm(lat, lon, apt_lat, apt_lon) <= airport_exclusion_nm:
                    return True
            return False
        
        monthly_stats = defaultdict(lambda: {'count': 0, 'flights': set(), 'total_duration': 0})
        
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                WITH ordered_tracks AS (
                    SELECT 
                        flight_id, timestamp, lat, lon, alt,
                        LAG(timestamp) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_ts,
                        LAG(lat) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lat,
                        LAG(lon) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lon,
                        LAG(alt) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_alt
                    FROM {table_name}
                    WHERE timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL AND lon IS NOT NULL
                )
                SELECT 
                    flight_id, prev_ts, (timestamp - prev_ts) as gap_seconds,
                    prev_lat, prev_lon
                FROM ordered_tracks
                WHERE prev_ts IS NOT NULL
                  AND (timestamp - prev_ts) > ?
                  AND (prev_alt IS NULL OR prev_alt > ?)
            """
            results = self._execute_query('research', query, (start_ts, end_ts, gap_threshold_seconds, min_altitude_ft))
            
            for row in results:
                flight_id, gap_start, gap_seconds, lat, lon = row
                if is_near_airport(lat, lon):
                    continue
                
                month_key = datetime.fromtimestamp(gap_start).strftime('%Y-%m')
                monthly_stats[month_key]['count'] += 1
                monthly_stats[month_key]['flights'].add(flight_id)
                monthly_stats[month_key]['total_duration'] += gap_seconds
        
        result = []
        for month, data in sorted(monthly_stats.items()):
            result.append({
                'month': month,
                'count': data['count'],
                'affected_flights': len(data['flights']),
                'avg_duration': int(data['total_duration'] / max(data['count'], 1))
            })
        
        return result
    
    def get_signal_loss_hourly(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get hourly distribution of signal loss events.
        
        Answers the demand: "What time of day has the most interference?"
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
        
        Returns:
            [{hour: 0-23, count: int, affected_flights: int}]
        """
        import math
        from datetime import datetime
        
        gap_threshold_seconds = 300
        min_altitude_ft = 5000
        airport_exclusion_nm = 5
        
        airports = {
            'LLBG': (32.0114, 34.8867), 'LLER': (29.9403, 35.0004),
            'LLHA': (32.8094, 35.0431), 'OJAI': (31.7226, 35.9932),
            'OLBA': (33.8209, 35.4884), 'LCLK': (34.8751, 33.6249),
        }
        
        def haversine_nm(lat1, lon1, lat2, lon2):
            R = 3440.065
            lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(delta_lon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        def is_near_airport(lat, lon):
            for apt_lat, apt_lon in airports.values():
                if haversine_nm(lat, lon, apt_lat, apt_lon) <= airport_exclusion_nm:
                    return True
            return False
        
        hourly_stats = {h: {'count': 0, 'flights': set()} for h in range(24)}
        
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                WITH ordered_tracks AS (
                    SELECT 
                        flight_id, timestamp, lat, lon, alt,
                        LAG(timestamp) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_ts,
                        LAG(lat) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lat,
                        LAG(lon) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lon,
                        LAG(alt) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_alt
                    FROM {table_name}
                    WHERE timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL AND lon IS NOT NULL
                )
                SELECT 
                    flight_id, prev_ts, prev_lat, prev_lon
                FROM ordered_tracks
                WHERE prev_ts IS NOT NULL
                  AND (timestamp - prev_ts) > ?
                  AND (prev_alt IS NULL OR prev_alt > ?)
            """
            results = self._execute_query('research', query, (start_ts, end_ts, gap_threshold_seconds, min_altitude_ft))
            
            for row in results:
                flight_id, gap_start, lat, lon = row
                if is_near_airport(lat, lon):
                    continue
                
                hour = datetime.fromtimestamp(gap_start).hour
                hourly_stats[hour]['count'] += 1
                hourly_stats[hour]['flights'].add(flight_id)
        
        result = []
        for hour in range(24):
            result.append({
                'hour': hour,
                'count': hourly_stats[hour]['count'],
                'affected_flights': len(hourly_stats[hour]['flights'])
            })
        
        return result
    
    def get_diversion_stats(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get diversion statistics - flights that changed destination or deviated from route.
        
        Analyzes anomaly reports for:
        - Rule 8: Diversions (planned vs actual destination mismatch)
        - Rule 3: Holding patterns (360-degree turns)
        - Rule 11: Off-course deviations
        
        Returns:
            {
                'total_diversions': int,
                'total_large_deviations': int,
                'total_holding_360s': int,
                'by_airport': {airport_code: count},
                'by_airline': {airline_code: count}
            }
        """
        total_diversions = 0
        total_large_deviations = 0
        total_holding_360s = 0
        by_airport = defaultdict(int)
        by_airline = defaultdict(int)
        
        # Query anomaly reports
        query = """
            SELECT full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            report_json = row[0]
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                callsign = report.get('summary', {}).get('callsign', '')
                
                # Extract airline code from callsign
                airline = ''.join(c for c in (callsign or '')[:3] if c.isalpha()).upper() if callsign else None
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    details = rule.get('details', {})
                    
                    # Rule 8: Diversion
                    if rule_id == 8:
                        total_diversions += 1
                        actual_airport = details.get('actual')
                        if actual_airport:
                            by_airport[actual_airport] += 1
                        if airline:
                            by_airline[airline] += 1
                    
                    # Rule 3: Holding patterns (360-degree turns)
                    elif rule_id == 3:
                        events = details.get('events', [])
                        for event in events:
                            # Count 360-degree turns
                            turn_deg = abs(event.get('accumulated_deg', 0))
                            if turn_deg >= 300:  # Near full circle
                                total_holding_360s += 1
                    
                    # Rule 11: Off-course
                    elif rule_id == 11:
                        # Count significant deviations
                        off_path_events = details.get('off_path', [])
                        if len(off_path_events) >= 3:  # Multiple off-path points
                            total_large_deviations += 1
                            
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        return {
            'total_diversions': total_diversions,
            'total_large_deviations': total_large_deviations,
            'total_holding_360s': total_holding_360s,
            'by_airport': dict(by_airport),
            'by_airline': dict(by_airline)
        }
    
    def get_diversions_monthly(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get monthly breakdown of diversions.
        
        Answers the demand: "Diversions by period/season"
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
        
        Returns:
            [{month: 'YYYY-MM', diversions: int, holding_patterns: int, off_course: int}]
        """
        from datetime import datetime
        
        monthly_stats = defaultdict(lambda: {
            'diversions': 0,
            'holding_patterns': 0,
            'off_course': 0,
            'flights': set()
        })
        
        # Query anomaly reports with timestamp
        query = """
            SELECT timestamp, flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            ts, flight_id, report_json = row
            month_key = datetime.fromtimestamp(ts).strftime('%Y-%m')
            
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    
                    if rule_id == 8:  # Diversion
                        monthly_stats[month_key]['diversions'] += 1
                        monthly_stats[month_key]['flights'].add(flight_id)
                    elif rule_id == 3:  # Holding pattern
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            if abs(event.get('accumulated_deg', 0)) >= 300:
                                monthly_stats[month_key]['holding_patterns'] += 1
                    elif rule_id == 11:  # Off-course
                        off_path = rule.get('details', {}).get('off_path', [])
                        if len(off_path) >= 3:
                            monthly_stats[month_key]['off_course'] += 1
                            
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        result = []
        for month in sorted(monthly_stats.keys()):
            data = monthly_stats[month]
            result.append({
                'month': month,
                'diversions': data['diversions'],
                'holding_patterns': data['holding_patterns'],
                'off_course': data['off_course'],
                'total_events': data['diversions'] + data['holding_patterns'] + data['off_course'],
                'affected_flights': len(data['flights'])
            })
        
        return result
    
    def get_rtb_events(self, start_ts: int, end_ts: int, max_duration_min: int = 30) -> List[Dict[str, Any]]:
        """
        Get Return-To-Base events - flights where _rule_takeoff_return (rule id 7) was triggered.
        
        This extracts RTB events from anomaly reports where rule 7 matched,
        which detects flights that took off and returned to the same airport.
        
        The _rule_takeoff_return rule validates:
        - Flight reached takeoff altitude
        - Aircraft traveled meaningful outbound distance
        - Returned to origin airport within time limit
        
        Returns:
            [{flight_id, callsign, departure_time, landing_time, duration_min, airport, max_outbound_nm}]
        """
        from datetime import datetime

        rtb_events = []
        seen_flights = set()
        
        # Query anomaly reports for RTB events (Rule 7 = _rule_takeoff_return)
        query = """
            SELECT flight_id, callsign, timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            if not row or len(row) < 4:
                continue
            
            flight_id, callsign, timestamp, report_json = row
            
            if flight_id in seen_flights:
                continue
            
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 7:  # _rule_takeoff_return
                        details = rule.get('details', {})
                        
                        # Extract RTB details from the rule
                        airport = details.get('airport', 'UNKNOWN')
                        takeoff_ts = details.get('takeoff_ts', timestamp)
                        landing_ts = details.get('landing_ts', timestamp)
                        elapsed_s = details.get('elapsed_s', 0)
                        max_outbound_nm = details.get('max_outbound_nm', 0)
                        
                        # Calculate duration
                        if elapsed_s:
                            duration_min = elapsed_s / 60
                        elif takeoff_ts and landing_ts:
                            duration_min = (landing_ts - takeoff_ts) / 60
                        else:
                            duration_min = 0
                        
                        # Apply max_duration filter if specified
                        if max_duration_min and duration_min > max_duration_min:
                            continue
                        
                        rtb_events.append({
                            'flight_id': flight_id,
                            'callsign': callsign or 'UNKNOWN',
                            'departure_time': takeoff_ts,
                            'landing_time': landing_ts,
                            'duration_min': round(duration_min, 1),
                            'airport': airport,
                            'max_outbound_nm': round(max_outbound_nm, 1) if max_outbound_nm else 0
                        })
                        seen_flights.add(flight_id)
                        break  # Only count one RTB event per flight
                        
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        return sorted(rtb_events, key=lambda x: x['departure_time'], reverse=True)[:50]
    
    def _find_nearest_airport(self, lat: float, lon: float) -> str:
        """Find nearest airport code to given coordinates."""
        import math
        
        # Major airports in the region
        airports = [
            {'code': 'LLBG', 'lat': 32.0094, 'lon': 34.8869},  # Ben Gurion
            {'code': 'LLER', 'lat': 29.7256, 'lon': 35.0114},  # Ramon
            {'code': 'LLHA', 'lat': 32.8111, 'lon': 35.0431},  # Haifa
            {'code': 'LLOV', 'lat': 29.9403, 'lon': 34.9358},  # Ovda
            {'code': 'OLBA', 'lat': 33.8208, 'lon': 35.4883},  # Beirut
            {'code': 'OJAM', 'lat': 31.7226, 'lon': 35.9932},  # Amman
            {'code': 'LCLK', 'lat': 34.8750, 'lon': 33.6250},  # Larnaca
            {'code': 'LCPH', 'lat': 34.7180, 'lon': 32.4856},  # Paphos
        ]
        
        best_code = 'UNKNOWN'
        best_dist = float('inf')
        
        for apt in airports:
            dlat = math.radians(apt['lat'] - lat)
            dlon = math.radians(apt['lon'] - lon)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(apt['lat'])) * math.sin(dlon/2)**2
            dist = 3440.065 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            if dist < best_dist:
                best_dist = dist
                best_code = apt['code']
        
        return best_code if best_dist < 50 else 'UNKNOWN'  # Within 50nm
    
    def get_safety_events_monthly(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get monthly breakdown of safety events.
        
        Answers: "Which month was the most dangerous?"
        
        Returns:
            [{month: 'YYYY-MM', emergency_codes: int, near_miss: int, go_arounds: int, 
              total_events: int, affected_flights: int}]
        """
        monthly_stats = defaultdict(lambda: {
            'emergency_codes': 0,
            'near_miss': 0,
            'go_arounds': 0,
            'flights': set()
        })
        
        # Query anomaly reports
        query = """
            SELECT timestamp, flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            timestamp, flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                month = datetime.fromtimestamp(timestamp).strftime('%Y-%m')
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    if rule_id == 1:  # Emergency squawk
                        monthly_stats[month]['emergency_codes'] += 1
                        monthly_stats[month]['flights'].add(flight_id)
                    elif rule_id == 4:  # Near-miss
                        monthly_stats[month]['near_miss'] += 1
                        monthly_stats[month]['flights'].add(flight_id)
                    elif rule_id == 6:  # Go-around
                        monthly_stats[month]['go_arounds'] += 1
                        monthly_stats[month]['flights'].add(flight_id)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        result = []
        for month in sorted(monthly_stats.keys()):
            data = monthly_stats[month]
            result.append({
                'month': month,
                'emergency_codes': data['emergency_codes'],
                'near_miss': data['near_miss'],
                'go_arounds': data['go_arounds'],
                'total_events': data['emergency_codes'] + data['near_miss'] + data['go_arounds'],
                'affected_flights': len(data['flights'])
            })
        
        return result
    
    def get_go_arounds_hourly(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get hourly distribution of go-around events.
        
        Answers: "At which hours do go-arounds peak?"
        
        Returns:
            [{hour: 0-23, count: int, airports: {airport: count}}]
        """
        hourly_stats = defaultdict(lambda: {'count': 0, 'airports': Counter()})
        
        # Query anomaly reports
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 6:  # Go-around rule
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            hour = datetime.fromtimestamp(timestamp).hour
                            airport = event.get('airport', 'UNKNOWN')
                            hourly_stats[hour]['count'] += 1
                            hourly_stats[hour]['airports'][airport] += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Build result for all 24 hours
        result = []
        for hour in range(24):
            data = hourly_stats.get(hour, {'count': 0, 'airports': Counter()})
            result.append({
                'hour': hour,
                'count': data['count'],
                'airports': dict(data['airports'])
            })
        
        return result
    
    def get_flights_missing_info(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get count of flights missing callsign or destination.
        
        Answers: "How many aircraft fly with no callsign/destination?"
        
        Uses flight_metadata table for accurate counts:
        - no_callsign: flights where callsign is NULL
        - no_destination: flights where destination_airport is NULL
        - total_flights: total distinct flights in the period
        
        Returns:
            {no_callsign: int, no_destination: int, total_flights: int}
        """
        no_callsign = 0
        no_destination = 0
        total_flights = 0
        
        # Query flight_metadata table for accurate counts
        # Total flights
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        if results and results[0][0]:
            total_flights = results[0][0]
        
        # Flights without callsign
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (callsign IS NULL OR callsign = '')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        if results and results[0][0]:
            no_callsign = results[0][0]
        
        # Flights without destination
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (destination_airport IS NULL OR destination_airport = '')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        if results and results[0][0]:
            no_destination = results[0][0]
        
        return {
            'no_callsign': no_callsign,
            'no_destination': no_destination,
            'total_flights': total_flights
        }
    
    def get_near_miss_locations(self, start_ts: int, end_ts: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get geographic locations of near-miss events for heatmap.
        
        Each near-miss between 2 flights is counted as 1 event (deduplicated).
        A proximity alert in an anomaly report adds 1 to the counter, not 2
        (even if both flights have the same event in their reports).
        
        Answers: "Where do most safety events occur?"
        
        Strategy:
        1. Try to get lat/lon from rule details (proximity events) - rarely available
        2. Look up coordinates from flight tracks at event timestamp (PRIMARY)
        3. Use country centroids as fallback for events with only country data
        
        Returns:
            [{lat, lon, count, severity_high, severity_medium}]
        """
        import random
        
        # Grid size for clustering (0.5 degrees for better visibility)
        grid_size = 0.5
        
        # Country centroids as fallback
        COUNTRY_CENTROIDS = {
            'IL': (31.5, 34.8),   # Israel
            'JO': (31.0, 36.0),   # Jordan
            'EG': (27.0, 30.0),   # Egypt
            'CY': (35.0, 33.0),   # Cyprus
            'LB': (33.9, 35.5),   # Lebanon
            'SY': (35.0, 38.0),   # Syria
            'SA': (24.0, 45.0),   # Saudi Arabia
            'IQ': (33.0, 44.0),   # Iraq
            'TR': (39.0, 35.0),   # Turkey
            'GR': (39.0, 22.0),   # Greece
        }
        
        # Collect all near-miss events with their flight_ids and timestamps
        near_miss_events = []  # [(flight_id, timestamp, distance_nm, alt_diff, other_flight)]
        
        # Strategy 1: Get from research DB with rule details (matched_rules)
        query = """
            SELECT timestamp, flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND (matched_rule_ids LIKE '%4,%' OR matched_rule_ids LIKE '%4' 
                 OR matched_rule_ids = '4' OR matched_rule_ids LIKE '%, 4%')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Track unique events to avoid double-counting
        seen_pairs = set()  # (sorted flight pair, time bucket)
        
        for row in results:
            timestamp, flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 4:  # Proximity rule
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            distance_nm = event.get('distance_nm', 999)
                            alt_diff = event.get('altitude_diff_ft', 9999)
                            
                            # Skip safe vertical separation (same filter as get_near_miss_events)
                            if alt_diff >= 1000:
                                continue
                            
                            other_flight = event.get('other_flight', event.get('other_aircraft', 'UNKNOWN'))
                            
                            # Deduplicate by flight pair and time bucket
                            flight_pair = tuple(sorted([flight_id, other_flight]))
                            time_bucket = timestamp // 300  # 5-minute windows
                            dedup_key = (flight_pair, time_bucket)
                            
                            if dedup_key not in seen_pairs:
                                seen_pairs.add(dedup_key)
                                # Check if event has lat/lon directly
                                lat = event.get('lat')
                                lon = event.get('lon')
                                near_miss_events.append({
                                    'flight_id': flight_id,
                                    'timestamp': timestamp,
                                    'distance_nm': distance_nm,
                                    'alt_diff': alt_diff,
                                    'other_flight': other_flight,
                                    'lat': lat,
                                    'lon': lon
                                })
                        break  # Only process first Rule 4 match
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Strategy 2: Look up coordinates from flight tracks for events without lat/lon
        events_needing_coords = [e for e in near_miss_events if e['lat'] is None]
        
        if events_needing_coords:
            # Get unique flight_ids
            flight_ids = list(set(e['flight_id'] for e in events_needing_coords))
            
            # Query tracks for these flights
            flight_positions = {}  # flight_id -> [(timestamp, lat, lon)]
            
            for table_name in ['anomalies_tracks', 'normal_tracks']:
                # Query in batches
                for i in range(0, len(flight_ids), 50):
                    batch = flight_ids[i:i+50]
                    placeholders = ','.join(['?' for _ in batch])
                    track_query = f"""
                        SELECT flight_id, timestamp, lat, lon
                        FROM {table_name}
                        WHERE flight_id IN ({placeholders})
                          AND lat IS NOT NULL AND lon IS NOT NULL
                        ORDER BY flight_id, timestamp
                    """
                    track_results = self._execute_query('research', track_query, tuple(batch))
                    
                    for row in track_results:
                        fid, ts, lat, lon = row
                        if fid not in flight_positions:
                            flight_positions[fid] = []
                        flight_positions[fid].append((ts, lat, lon))
            
            # Assign coordinates to events based on closest timestamp
            for event in events_needing_coords:
                fid = event['flight_id']
                event_ts = event['timestamp']
                
                if fid in flight_positions and flight_positions[fid]:
                    # Find position closest to event timestamp
                    closest = min(flight_positions[fid], key=lambda p: abs(p[0] - event_ts))
                    _, lat, lon = closest
                    event['lat'] = lat
                    event['lon'] = lon
        
        # Aggregate events by grid cell
        location_stats = defaultdict(lambda: {'count': 0, 'high': 0, 'medium': 0, 'flight_ids': []})
        events_with_coords = 0
        
        for event in near_miss_events:
            lat = event.get('lat')
            lon = event.get('lon')
            
            if lat is not None and lon is not None:
                events_with_coords += 1
                grid_lat = round(lat / grid_size) * grid_size
                grid_lon = round(lon / grid_size) * grid_size
                grid_key = (grid_lat, grid_lon)
                
                distance_nm = event['distance_nm']
                alt_diff = event['alt_diff']
                severity = 'high' if (distance_nm < 1 and alt_diff < 1000) else 'medium'
                
                location_stats[grid_key]['count'] += 1
                if severity == 'high':
                    location_stats[grid_key]['high'] += 1
                else:
                    location_stats[grid_key]['medium'] += 1
                
                # Track flight IDs (keep up to 10 unique per grid cell)
                fid = event.get('flight_id')
                other_fid = event.get('other_flight')
                if fid and fid not in location_stats[grid_key]['flight_ids'] and len(location_stats[grid_key]['flight_ids']) < 10:
                    location_stats[grid_key]['flight_ids'].append(fid)
                if other_fid and other_fid != 'UNKNOWN' and other_fid not in location_stats[grid_key]['flight_ids'] and len(location_stats[grid_key]['flight_ids']) < 10:
                    location_stats[grid_key]['flight_ids'].append(other_fid)
        
        # Strategy 3: Fallback to country centroids for events still without coords
        events_without_coords = [e for e in near_miss_events if e.get('lat') is None]
        if events_without_coords:
            # Try to get country from tagged DB
            flight_ids = list(set(e['flight_id'] for e in events_without_coords))
            
            if flight_ids:
                placeholders = ','.join(['?' for _ in flight_ids])
                country_query = f"""
                    SELECT flight_id, crossed_borders
                    FROM flight_metadata
                    WHERE flight_id IN ({placeholders})
                """
                country_results = self._execute_query('tagged', country_query, tuple(flight_ids))
                
                flight_countries = {}
                for row in country_results:
                    fid, borders = row
                    if borders:
                        flight_countries[fid] = borders.split(',')[0].strip()
                
                for event in events_without_coords:
                    fid = event['flight_id']
                    country = flight_countries.get(fid)
                    
                    if country and country in COUNTRY_CENTROIDS:
                        base_lat, base_lon = COUNTRY_CENTROIDS[country]
                        # Add jitter for visual spread
                        lat = base_lat + random.uniform(-0.3, 0.3)
                        lon = base_lon + random.uniform(-0.3, 0.3)
                        
                        grid_lat = round(lat / grid_size) * grid_size
                        grid_lon = round(lon / grid_size) * grid_size
                        grid_key = (grid_lat, grid_lon)
                        
                        distance_nm = event['distance_nm']
                        alt_diff = event['alt_diff']
                        severity = 'high' if (distance_nm < 1 and alt_diff < 1000) else 'medium'
                        
                        location_stats[grid_key]['count'] += 1
                        if severity == 'high':
                            location_stats[grid_key]['high'] += 1
                        else:
                            location_stats[grid_key]['medium'] += 1
                        
                        # Track flight IDs for fallback events too
                        fid = event.get('flight_id')
                        other_fid = event.get('other_flight')
                        if fid and fid not in location_stats[grid_key]['flight_ids'] and len(location_stats[grid_key]['flight_ids']) < 10:
                            location_stats[grid_key]['flight_ids'].append(fid)
                        if other_fid and other_fid != 'UNKNOWN' and other_fid not in location_stats[grid_key]['flight_ids'] and len(location_stats[grid_key]['flight_ids']) < 10:
                            location_stats[grid_key]['flight_ids'].append(other_fid)
        
        # Convert to list
        result = []
        for (lat, lon), data in location_stats.items():
            result.append({
                'lat': lat,
                'lon': lon,
                'count': data['count'],
                'severity_high': data['high'],
                'severity_medium': data['medium'],
                'sample_flight_ids': data['flight_ids'][:5]  # Return up to 5 sample flight IDs
            })
        
        # Sort by count and limit
        result = sorted(result, key=lambda x: x['count'], reverse=True)[:limit]
        return result
    
    def get_safety_by_phase(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get safety events breakdown by flight phase (cruise vs approach).
        
        Answers: "How many safety events occur at cruise vs approach?"
        
        Flight phases determined by:
        1. Altitude if available (> 25,000 ft = cruise, 10,000-25,000 = descent/climb, < 10,000 = approach)
        2. Event type inference:
           - Go-arounds are ALWAYS approach phase (by definition)
           - Near-miss events: infer from context or default to descent/climb
           - Emergency: use altitude or default based on speed/context
        
        IMPORTANT: Counts UNIQUE FLIGHTS per rule type per phase to match other statistics.
        Near-miss events apply the same altitude_diff < 1000ft filter as get_near_miss_events.
        
        Returns:
            {cruise: {count, events}, descent: {...}, approach: {...}}
        """
        # Track unique flights per phase per rule type to match other statistics
        phase_flights = {
            'cruise': {'emergency': set(), 'near_miss': set(), 'go_around': set()},
            'descent_climb': {'emergency': set(), 'near_miss': set(), 'go_around': set()},
            'approach': {'emergency': set(), 'near_miss': set(), 'go_around': set()}
        }
        
        # Safety-related rule IDs: 1=Emergency, 4=Near-miss/Proximity, 6=Go-around
        SAFETY_RULES = {1, 4, 6}
        
        query = """
            SELECT timestamp, flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            timestamp, flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                # Get flight info from summary
                summary = report.get('summary', {})
                summary_altitude = summary.get('altitude')
                ground_speed = summary.get('ground_speed') or summary.get('gspeed')
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    
                    # Only count safety-related rules
                    if rule_id not in SAFETY_RULES:
                        continue
                    
                    events = rule.get('details', {}).get('events', [])
                    
                    # If no events, create dummy event for processing
                    if not events:
                        events = [{}]
                    
                    # For each flight, find the best phase based on available altitude data
                    # Only count the flight once per rule type (not every event)
                    best_phase = None
                    best_altitude = None
                    valid_near_miss = False  # Track if near-miss passes altitude diff filter
                    
                    for event in events:
                        # For near-miss (Rule 4), apply the same filter as get_near_miss_events
                        if rule_id == 4:
                            alt_diff = event.get('altitude_diff_ft', 9999)
                            if alt_diff >= 1000:
                                continue  # Skip events with safe vertical separation
                            valid_near_miss = True
                        
                        # Get altitude - try multiple sources
                        altitude = event.get('altitude') or event.get('alt') or event.get('altitude_ft')
                        if altitude is None:
                            altitude = summary_altitude
                        
                        # Track the highest altitude (most likely to find cruise events)
                        if altitude is not None and (best_altitude is None or altitude > best_altitude):
                            best_altitude = altitude
                        
                        # Determine phase based on rule type and altitude
                        if rule_id == 6:  # Go-around
                            # Go-arounds by definition happen during approach phase
                            best_phase = 'approach'
                            break  # No need to check other events
                        elif altitude is not None:
                            # Use altitude to determine phase
                            if altitude > 25000:
                                phase = 'cruise'
                            elif altitude > 10000:
                                phase = 'descent_climb'
                            else:
                                phase = 'approach'
                            # Keep the highest phase (cruise > descent_climb > approach)
                            if best_phase is None:
                                best_phase = phase
                            elif phase == 'cruise':
                                best_phase = 'cruise'
                            elif phase == 'descent_climb' and best_phase == 'approach':
                                best_phase = 'descent_climb'
                    
                    # If no altitude data found, use speed-based heuristics
                    if best_phase is None:
                        event_speed = ground_speed
                        if events and events[0]:
                            event_speed = events[0].get('ground_speed') or events[0].get('gspeed') or ground_speed
                        
                        if rule_id == 4:  # Near-miss
                            if event_speed and event_speed > 400:
                                best_phase = 'cruise'
                            elif event_speed and event_speed < 200:
                                best_phase = 'approach'
                            else:
                                # Check if near airport
                                airport = events[0].get('airport') or events[0].get('near_airport') if events else None
                                if airport:
                                    best_phase = 'approach'
                                else:
                                    best_phase = 'descent_climb'
                        elif rule_id == 1:  # Emergency
                            if event_speed and event_speed > 400:
                                best_phase = 'cruise'
                            elif event_speed and event_speed < 200:
                                best_phase = 'approach'
                            else:
                                best_phase = 'descent_climb'
                        else:
                            best_phase = 'descent_climb'
                    
                    # For near-miss, only count if at least one event passed the altitude diff filter
                    if rule_id == 4 and not valid_near_miss:
                        continue
                    
                    # Add flight to the appropriate phase set (counts unique flights)
                    if rule_id == 1:
                        phase_flights[best_phase]['emergency'].add(flight_id)
                    elif rule_id == 4:
                        phase_flights[best_phase]['near_miss'].add(flight_id)
                    elif rule_id == 6:
                        phase_flights[best_phase]['go_around'].add(flight_id)
                        
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                continue
        
        # Convert sets to counts
        phase_stats = {}
        for phase, rule_sets in phase_flights.items():
            emergency_count = len(rule_sets['emergency'])
            near_miss_count = len(rule_sets['near_miss'])
            go_around_count = len(rule_sets['go_around'])
            phase_stats[phase] = {
                'count': emergency_count + near_miss_count + go_around_count,
                'emergency': emergency_count,
                'near_miss': near_miss_count,
                'go_around': go_around_count
            }
        
        # Calculate totals
        total = sum(p['count'] for p in phase_stats.values())
        
        return {
            'phases': phase_stats,
            'total_events': total,
            'percentages': {
                phase: round((data['count'] / total * 100) if total > 0 else 0, 1)
                for phase, data in phase_stats.items()
            }
        }
    
    def get_deviations_by_aircraft_type(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get route deviations breakdown by aircraft type.
        
        Answers: "Which aircraft types deviate most from defined routes?"
        
        Returns:
            [{aircraft_type, deviation_count, avg_deviation_nm, flights: []}]
        """
        type_stats = defaultdict(lambda: {
            'count': 0, 
            'total_deviation': 0, 
            'flights': set(),
            'large_deviations': 0  # >20nm
        })
        
        query = """
            SELECT flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                
                # Get aircraft type from report
                aircraft_type = report.get('summary', {}).get('aircraft_type', 'UNKNOWN')
                if not aircraft_type or aircraft_type == '':
                    aircraft_type = 'UNKNOWN'
                
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    # Rule 3 is typically off-course/deviation rule
                    if rule.get('id') in [3, 7]:  # Off-course or deviation rules
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            deviation_nm = event.get('deviation_nm', event.get('distance_nm', 0))
                            type_stats[aircraft_type]['count'] += 1
                            type_stats[aircraft_type]['total_deviation'] += deviation_nm
                            type_stats[aircraft_type]['flights'].add(flight_id)
                            if deviation_nm > 20:
                                type_stats[aircraft_type]['large_deviations'] += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        result = []
        for aircraft_type, data in type_stats.items():
            if data['count'] > 0:
                result.append({
                    'aircraft_type': aircraft_type,
                    'deviation_count': data['count'],
                    'avg_deviation_nm': round(data['total_deviation'] / data['count'], 2),
                    'large_deviations': data['large_deviations'],
                    'unique_flights': len(data['flights']),
                    'flights': list(data['flights'])[:5]  # Sample flights
                })
        
        return sorted(result, key=lambda x: x['deviation_count'], reverse=True)
    
    def get_emergency_aftermath(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get emergency code events with aftermath analysis.
        
        Answers: "What happened after aircraft switched to emergency code?"
        
        Analyzes flight behavior after emergency squawk:
        - Landed normally
        - Diverted to alternate
        - Returned to base
        - Continued flight
        
        Returns:
            [{flight_id, callsign, emergency_code, timestamp, outcome, 
              landing_airport, flight_duration_after, details}]
        """
        emergency_events = []
        
        query = """
            SELECT timestamp, flight_id, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            timestamp, flight_id, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                callsign = report.get('summary', {}).get('callsign', '')
                dest = report.get('summary', {}).get('destination', '')
                origin = report.get('summary', {}).get('origin', '')
                
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 1:  # Emergency squawk rule
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            squawk = event.get('squawk', 'UNKNOWN')
                            event_ts = event.get('timestamp', timestamp)
                            
                            # Analyze aftermath based on other rules and flight data
                            outcome = 'unknown'
                            landing_airport = None
                            
                            # Check for go-around (indicates attempted landing)
                            has_go_around = any(r.get('id') == 6 for r in matched_rules)
                            
                            # Check for diversion
                            has_diversion = any(r.get('id') in [5, 7] for r in matched_rules)  # Diversion rules
                            
                            # Check for RTB (origin == landing)
                            has_rtb = any(r.get('id') == 8 for r in matched_rules)  # RTB rule
                            
                            # Determine outcome
                            if has_rtb:
                                outcome = 'returned_to_base'
                                landing_airport = origin
                            elif has_diversion:
                                outcome = 'diverted'
                                # Try to find where they diverted to
                                for r in matched_rules:
                                    if r.get('id') in [5, 7]:
                                        div_events = r.get('details', {}).get('events', [])
                                        if div_events:
                                            landing_airport = div_events[0].get('alternate_airport', 'UNKNOWN')
                                            break
                            elif has_go_around:
                                outcome = 'go_around_then_landed'
                                landing_airport = dest
                            elif dest:
                                outcome = 'landed_at_destination'
                                landing_airport = dest
                            else:
                                outcome = 'continued_flight'
                            
                            # Describe the emergency code
                            code_description = {
                                '7700': 'General Emergency',
                                '7600': 'Radio Failure',
                                '7500': 'Hijacking'
                            }.get(str(squawk), 'Emergency')
                            
                            emergency_events.append({
                                'flight_id': flight_id,
                                'callsign': callsign or 'UNKNOWN',
                                'emergency_code': squawk,
                                'code_description': code_description,
                                'timestamp': event_ts,
                                'outcome': outcome,
                                'landing_airport': landing_airport,
                                'origin': origin,
                                'destination': dest,
                                'had_go_around': has_go_around
                            })
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Group by outcome for summary
        return sorted(emergency_events, key=lambda x: x['timestamp'], reverse=True)
    
    def get_bottleneck_zones(self, start_ts: int, end_ts: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Identify airspace bottleneck zones with high traffic density.
        
        Answers: "Where are the bottleneck areas in the airspace?"
        
        Detects zones with:
        - High flight density
        - Multiple holding patterns
        - Frequent altitude changes
        
        Returns polygon data for each zone:
        [{lat, lon, density_score, flight_count, holding_count, avg_altitude, 
          congestion_level, polygon: [[lon, lat], ...]}]
        """
        import math

        # Grid size for zone clustering (0.25 degrees ~ 15nm)
        grid_size = 0.25
        zone_stats = defaultdict(lambda: {
            'flight_count': 0,
            'flights': set(),
            'holding_count': 0,
            'altitudes': [],
            'timestamps': [],
            'all_points': []  # Store all track points for polygon calculation
        })
        
        # Query track data for density analysis
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT flight_id, lat, lon, alt, timestamp
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                  AND lat IS NOT NULL AND lon IS NOT NULL
                  AND lat BETWEEN 28 AND 38
                  AND lon BETWEEN 30 AND 40
                LIMIT 100000
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            for row in results:
                flight_id, lat, lon, alt, ts = row
                if lat is None or lon is None:
                    continue
                
                # Grid cell
                grid_lat = round(lat / grid_size) * grid_size
                grid_lon = round(lon / grid_size) * grid_size
                key = (grid_lat, grid_lon)
                
                zone_stats[key]['flight_count'] += 1
                zone_stats[key]['flights'].add(flight_id)
                zone_stats[key]['timestamps'].append(ts)
                zone_stats[key]['all_points'].append((lon, lat))
                if alt:
                    zone_stats[key]['altitudes'].append(alt)
        
        # Query for holding patterns to identify congestion
        query = """
            SELECT full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            try:
                report = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') in [5, 6]:  # Holding/Go-around rules
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            lat = event.get('lat')
                            lon = event.get('lon')
                            if lat and lon:
                                grid_lat = round(lat / grid_size) * grid_size
                                grid_lon = round(lon / grid_size) * grid_size
                                key = (grid_lat, grid_lon)
                                zone_stats[key]['holding_count'] += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Helper function to compute convex hull
        def convex_hull(points):
            """Compute convex hull using Graham scan algorithm."""
            if len(points) < 3:
                return points
            
            # Remove duplicates
            points = list(set(points))
            if len(points) < 3:
                return points
            
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            # Sort points lexicographically
            points = sorted(points)
            
            # Build lower hull
            lower = []
            for p in points:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            
            # Build upper hull
            upper = []
            for p in reversed(points):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            
            # Concatenate (remove last point of each half because it's repeated)
            return lower[:-1] + upper[:-1]
        
        # Calculate density scores and format results
        result = []
        for (lat, lon), data in zone_stats.items():
            unique_flights = len(data['flights'])
            if unique_flights < 3:  # Minimum threshold
                continue
            
            # Calculate density score
            # Higher score = more congestion
            density_score = (
                unique_flights * 2 +  # Base flight count
                data['holding_count'] * 5 +  # Holding patterns indicate congestion
                min(data['flight_count'] / 100, 10)  # Track point density
            )
            
            # Calculate time spread to detect sustained congestion
            if data['timestamps']:
                time_spread = max(data['timestamps']) - min(data['timestamps'])
                hours_active = max(time_spread / 3600, 1)
                flights_per_hour = unique_flights / hours_active
            else:
                flights_per_hour = 0
            
            # Determine congestion level
            if density_score > 50 or data['holding_count'] > 5:
                congestion_level = 'critical'
            elif density_score > 30 or data['holding_count'] > 2:
                congestion_level = 'high'
            elif density_score > 15:
                congestion_level = 'moderate'
            else:
                congestion_level = 'low'
            
            # Average altitude
            avg_altitude = sum(data['altitudes']) / len(data['altitudes']) if data['altitudes'] else 0
            
            # Calculate polygon from all track points
            all_pts = data['all_points']
            if len(all_pts) >= 3:
                # Sample points if too many (for performance)
                if len(all_pts) > 500:
                    import random
                    all_pts = random.sample(all_pts, 500)
                
                hull_pts = convex_hull(all_pts)
                # Ensure polygon is closed (first point = last point)
                if hull_pts and len(hull_pts) >= 3:
                    polygon = [[p[0], p[1]] for p in hull_pts]
                    polygon.append(polygon[0])  # Close the polygon
                else:
                    # Fallback to grid cell boundary
                    half = grid_size / 2
                    polygon = [
                        [lon - half, lat - half],
                        [lon + half, lat - half],
                        [lon + half, lat + half],
                        [lon - half, lat + half],
                        [lon - half, lat - half]  # Close
                    ]
            else:
                # Fallback to grid cell boundary for zones with few points
                half = grid_size / 2
                polygon = [
                    [lon - half, lat - half],
                    [lon + half, lat - half],
                    [lon + half, lat + half],
                    [lon - half, lat + half],
                    [lon - half, lat - half]  # Close
                ]
            
            result.append({
                'lat': lat,
                'lon': lon,
                'density_score': round(density_score, 1),
                'flight_count': unique_flights,
                'holding_count': data['holding_count'],
                'avg_altitude': round(avg_altitude),
                'flights_per_hour': round(flights_per_hour, 1),
                'congestion_level': congestion_level,
                'polygon': polygon
            })
        
        # Sort by density score and limit
        result = sorted(result, key=lambda x: x['density_score'], reverse=True)[:limit]
        return result
    
    def get_military_stats(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """Get military aircraft statistics using is_military field and category='Military_and_government'."""
        query = """
            SELECT 
                callsign, military_type, airline_code,
                origin_airport, destination_airport,
                flight_duration_sec, total_distance_nm, crossed_borders, category
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
            ORDER BY first_seen_ts DESC
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        by_type = defaultdict(int)
        by_country = defaultdict(int)
        flights = []
        
        for row in results:
            callsign, mil_type, airline_code, origin, dest, duration, distance, borders, category = row
            if mil_type:
                by_type[mil_type] += 1
            elif category == 'Military_and_government':
                by_type['Government/Military'] += 1
            
            country = 'Unknown'
            if callsign:
                cs = callsign.upper()
                if cs.startswith(('RCH', 'REACH', 'CNV', 'EVAC')):
                    country = 'USA'
                elif cs.startswith(('RAF', 'RRR', 'ASCOT')):
                    country = 'UK'
                elif cs.startswith(('RFF', 'RSD')):
                    country = 'Russia'
                elif cs.startswith(('IAF', '4XA', '4XB', '4XC')):
                    country = 'Israel'
                elif cs.startswith(('SHAHD', 'RJAF')):
                    country = 'Jordan'
            
            by_country[country] += 1
            flights.append({
                'callsign': callsign,
                'type': mil_type or category or 'Unknown',
                'country': country,
                'route': f"{origin} → {dest}" if origin and dest else 'Unknown'
            })
        
        return {
            'total_military': len(flights),
            'by_type': dict(by_type),
            'by_country': dict(by_country),
            'flights': flights[:50]
        }
    
    def get_airline_efficiency(self, start_ts: int = 0, end_ts: int = 0, limit: int = 20, route: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compare airline efficiency using pre-calculated fields."""
        if route and '-' in route:
            origin, dest = route.split('-', 1)
            if start_ts and end_ts:
                query = """
                    SELECT airline, AVG(flight_duration_sec), AVG(total_distance_nm), 
                           AVG(avg_speed_kts), COUNT(*)
                    FROM flight_metadata
                    WHERE origin_airport = ? AND destination_airport = ?
                    AND first_seen_ts BETWEEN ? AND ?
                    AND airline IS NOT NULL AND flight_duration_sec > 0
                    GROUP BY airline HAVING COUNT(*) >= 2 ORDER BY AVG(flight_duration_sec)
                """
                results = self._execute_query('tagged', query, (origin.strip(), dest.strip(), start_ts, end_ts))
            else:
                query = """
                    SELECT airline, AVG(flight_duration_sec), AVG(total_distance_nm), 
                           AVG(avg_speed_kts), COUNT(*)
                    FROM flight_metadata
                    WHERE origin_airport = ? AND destination_airport = ?
                    AND airline IS NOT NULL AND flight_duration_sec > 0
                    GROUP BY airline HAVING COUNT(*) >= 2 ORDER BY AVG(flight_duration_sec)
                """
                results = self._execute_query('research', query, (origin.strip(), dest.strip()))
        else:
            if start_ts and end_ts:
                query = """
                    SELECT airline, COUNT(*), AVG(flight_duration_sec), 
                           AVG(total_distance_nm), AVG(avg_speed_kts)
                    FROM flight_metadata
                    WHERE first_seen_ts BETWEEN ? AND ?
                    AND airline IS NOT NULL AND flight_duration_sec > 0
                    GROUP BY airline HAVING COUNT(*) >= 5 ORDER BY COUNT(*) DESC LIMIT ?
                """
                results = self._execute_query('tagged', query, (start_ts, end_ts, limit))
            else:
                query = """
                    SELECT airline, COUNT(*), AVG(flight_duration_sec), 
                           AVG(total_distance_nm), AVG(avg_speed_kts)
                    FROM flight_metadata
                    WHERE airline IS NOT NULL AND flight_duration_sec > 0
                    GROUP BY airline HAVING COUNT(*) >= 5 ORDER BY COUNT(*) DESC LIMIT ?
                """
                results = self._execute_query('research', query, (limit,))
        
        data = []
        for row in results:
            if route:
                airline, avg_dur, avg_dist, avg_speed, count = row
            else:
                airline, count, avg_dur, avg_dist, avg_speed = row
            data.append({
                'airline': airline,
                'avg_duration_hours': round(avg_dur / 3600, 2) if avg_dur else 0,
                'avg_distance_nm': round(avg_dist, 1) if avg_dist else 0,
                'avg_speed_kts': round(avg_speed, 1) if avg_speed else 0,
                'flight_count': count
            })
        return data
    
    def get_busiest_routes(self, start_ts: int, end_ts: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get busiest origin-destination pairs."""
        query = """
            SELECT origin_airport, destination_airport, COUNT(*), 
                   AVG(flight_duration_sec), AVG(total_distance_nm)
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND origin_airport IS NOT NULL AND destination_airport IS NOT NULL
            GROUP BY origin_airport, destination_airport
            ORDER BY COUNT(*) DESC LIMIT ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts, limit))
        
        routes = []
        for row in results:
            origin, dest, count, avg_dur, avg_dist = row
            routes.append({
                'route': f"{origin} → {dest}",
                'origin': origin,
                'destination': dest,
                'flight_count': count,
                'avg_duration_hours': round(avg_dur / 3600, 2) if avg_dur else 0
            })
        return routes

    # =========================================================================
    # OPTIMIZED METHODS FOR FEEDBACK_TAGGED.DB
    # These methods use pre-computed fields for better performance
    # =========================================================================
    
    def get_tagged_overview_stats(self, start_ts: int, end_ts: int, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get overview statistics from feedback_tagged.db using pre-computed fields.
        
        This is optimized to use indexes and denormalized fields for fast queries.
        Uses:
        - flight_metadata.is_military (pre-computed)
        - flight_metadata.emergency_squawk_detected (pre-computed)
        - anomaly_reports.matched_rule_ids (denormalized)
        - user_feedback.first_seen_ts (indexed)
        
        Returns:
            {
                'total_flights': int,
                'total_anomalies': int,
                'safety_events': int,
                'go_arounds': int,
                'emergency_codes': int,
                'near_miss': int,
                'holding_patterns': int,
                'military_flights': int,
                'return_to_field': int,
                'unplanned_landing': int,
                'avg_severity': float
            }
        """
        cache_key = _get_cache_key('get_tagged_overview_stats', start_ts, end_ts)
        if use_cache:
            cached = _get_cached(cache_key)
            if cached is not None:
                return cached
        
        stats = {
            'total_flights': 0,
            'total_anomalies': 0,
            'safety_events': 0,
            'go_arounds': 0,
            'emergency_codes': 0,
            'near_miss': 0,
            'holding_patterns': 0,
            'military_flights': 0,
            'return_to_field': 0,
            'unplanned_landing': 0,
            'avg_severity': 0.0
        }
        
        # Total tagged flights in time range (uses idx_flight_metadata_timestamps)
        query = """
            SELECT COUNT(*) FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        stats['total_flights'] = results[0][0] if results else 0
        
        # Anomalies count - use user_feedback.user_label = 1 (user confirmed anomaly)
        # NOT flight_metadata.is_anomaly which is always 1 for all tagged flights
        query = """
            SELECT COUNT(*) FROM user_feedback
            WHERE first_seen_ts BETWEEN ? AND ?
            AND user_label = 1
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        stats['total_anomalies'] = results[0][0] if results else 0
        
        # Military flights (uses idx_metadata_military + category check)
        query = """
            SELECT COUNT(*) FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        stats['military_flights'] = results[0][0] if results else 0
        
        # Emergency codes using pre-computed field (uses idx_metadata_emergency)
        query = """
            SELECT COUNT(*) FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND emergency_squawk_detected = 1
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        stats['emergency_codes'] = results[0][0] if results else 0
        
        # Rule-based counts - ONLY for flights confirmed as anomalies (user_label=1)
        # Rule IDs from migrate_rule_ids.py:
        #   3: Proximity Alert (near_miss), 4: Holding Pattern, 5: Go Around
        #   6: Return to Land, 7: Unplanned Landing
        # Check BOTH ar.matched_rule_ids AND uf.rule_id to match what the UI displays
        # The UI uses uf.rule_id to show the reason, so we must count it too
        # NOTE: Use uf.first_seen_ts (not ar.timestamp) to match the flight list filtering
        query = """
            SELECT 
                SUM(CASE WHEN uf.rule_id = 4
                         OR (',' || REPLACE(ar.matched_rule_ids, ' ', '') || ',') LIKE '%,4,%' THEN 1 ELSE 0 END) as holding_patterns,
                SUM(CASE WHEN uf.rule_id = 3
                         OR (',' || REPLACE(ar.matched_rule_ids, ' ', '') || ',') LIKE '%,3,%' THEN 1 ELSE 0 END) as near_miss,
                SUM(CASE WHEN uf.rule_id = 5
                         OR (',' || REPLACE(ar.matched_rule_ids, ' ', '') || ',') LIKE '%,5,%' THEN 1 ELSE 0 END) as go_arounds,
                SUM(CASE WHEN uf.rule_id = 6
                         OR (',' || REPLACE(ar.matched_rule_ids, ' ', '') || ',') LIKE '%,6,%' THEN 1 ELSE 0 END) as return_to_field,
                SUM(CASE WHEN uf.rule_id = 7
                         OR (',' || REPLACE(ar.matched_rule_ids, ' ', '') || ',') LIKE '%,7,%' THEN 1 ELSE 0 END) as unplanned_landing,
                AVG(COALESCE(ar.severity_cnn, 0) + COALESCE(ar.severity_dense, 0)) / 2 as avg_severity
            FROM user_feedback uf
            LEFT JOIN anomaly_reports ar ON ar.flight_id = uf.flight_id
            WHERE uf.first_seen_ts BETWEEN ? AND ?
            AND uf.user_label = 1
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        if results and results[0]:
            stats['holding_patterns'] = results[0][0] or 0
            stats['near_miss'] = results[0][1] or 0
            stats['go_arounds'] = results[0][2] or 0
            stats['return_to_field'] = results[0][3] or 0
            stats['unplanned_landing'] = results[0][4] or 0
            stats['avg_severity'] = round(results[0][5] or 0, 2)
        
        # Safety events = emergency + near_miss + go_arounds
        stats['safety_events'] = stats['emergency_codes'] + stats['near_miss'] + stats['go_arounds']
        
        _set_cached(cache_key, stats)
        return stats
    
    def get_tagged_flights_per_day(self, start_ts: int, end_ts: int, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get flight counts per day from feedback_tagged.db using pre-computed is_military field.
        
        Uses:
        - flight_metadata.first_seen_ts (indexed)
        - flight_metadata.is_military (pre-computed, indexed)
        
        Returns:
            [{date, count, military_count, civilian_count, anomaly_count}]
        """
        cache_key = _get_cache_key('get_tagged_flights_per_day', start_ts, end_ts)
        if use_cache:
            cached = _get_cached(cache_key)
            if cached is not None:
                return cached
        
        query = """
            SELECT 
                DATE(first_seen_ts, 'unixepoch') as date,
                COUNT(*) as total,
                SUM(CASE WHEN is_military = 1 OR category = 'Military_and_government' THEN 1 ELSE 0 END) as military,
                SUM(CASE WHEN (is_military = 0 OR is_military IS NULL) AND (category IS NULL OR category != 'Military_and_government') THEN 1 ELSE 0 END) as civilian,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            GROUP BY DATE(first_seen_ts, 'unixepoch')
            ORDER BY date
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        data = []
        for row in results:
            if row[0]:  # date is not None
                data.append({
                    'date': row[0],
                    'count': row[1] or 0,
                    'military_count': row[2] or 0,
                    'civilian_count': row[3] or 0,
                    'anomaly_count': row[4] or 0
                })
        
        _set_cached(cache_key, data)
        return data
    
    def get_tagged_busiest_airports(self, start_ts: int, end_ts: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get busiest airports from feedback_tagged.db using pre-computed fields.
        
        Uses:
        - flight_metadata.origin_airport, destination_airport (indexed)
        - flight_metadata.first_seen_ts (indexed)
        
        Returns:
            [{airport, arrivals, departures, total}]
        """
        # Count departures
        query = """
            SELECT origin_airport as airport, COUNT(*) as count
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND origin_airport IS NOT NULL AND origin_airport != ''
            GROUP BY origin_airport
        """
        departures = {}
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        for row in results:
            departures[row[0]] = row[1]
        
        # Count arrivals
        query = """
            SELECT destination_airport as airport, COUNT(*) as count
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND destination_airport IS NOT NULL AND destination_airport != ''
            GROUP BY destination_airport
        """
        arrivals = {}
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        for row in results:
            arrivals[row[0]] = row[1]
        
        # Combine
        all_airports = set(departures.keys()) | set(arrivals.keys())
        data = []
        for airport in all_airports:
            dep = departures.get(airport, 0)
            arr = arrivals.get(airport, 0)
            data.append({
                'airport': airport,
                'departures': dep,
                'arrivals': arr,
                'total': dep + arr
            })
        
        return sorted(data, key=lambda x: x['total'], reverse=True)[:limit]
    
    def get_tagged_safety_by_rule(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get safety events breakdown by rule from feedback_tagged.db.
        
        Uses denormalized matched_rule_ids and matched_rule_names for fast queries.
        
        Returns:
            {
                'by_rule': [{rule_id, rule_name, count}],
                'by_category': {category: count},
                'total_events': int
            }
        """
        query = """
            SELECT 
                matched_rule_ids,
                matched_rule_names,
                matched_rule_categories,
                COUNT(*) as count
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND matched_rule_ids IS NOT NULL
            GROUP BY matched_rule_ids
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        rule_counts = Counter()
        rule_names = {}
        category_counts = Counter()
        
        for row in results:
            rule_ids_str, rule_names_str, categories_str, count = row
            
            if rule_ids_str:
                # Parse comma-separated rule IDs
                for rule_id in rule_ids_str.split(','):
                    rule_id = rule_id.strip()
                    if rule_id:
                        rule_counts[rule_id] += count
            
            if rule_names_str:
                # Map rule IDs to names
                names = [n.strip() for n in rule_names_str.split(',')]
                ids = [i.strip() for i in (rule_ids_str or '').split(',')]
                for i, name in enumerate(names):
                    if i < len(ids) and ids[i]:
                        rule_names[ids[i]] = name
            
            if categories_str:
                for cat in categories_str.split(','):
                    cat = cat.strip()
                    if cat:
                        category_counts[cat] += count
        
        by_rule = []
        for rule_id, count in rule_counts.most_common():
            by_rule.append({
                'rule_id': int(rule_id) if rule_id.isdigit() else rule_id,
                'rule_name': rule_names.get(rule_id, f'Rule {rule_id}'),
                'count': count
            })
        
        return {
            'by_rule': by_rule,
            'by_category': dict(category_counts),
            'total_events': sum(rule_counts.values())
        }
    
    def get_tagged_emergency_stats(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get emergency code statistics from feedback_tagged.db using pre-computed fields.
        
        Uses:
        - flight_metadata.emergency_squawk_detected (pre-computed, indexed)
        - flight_metadata.squawk_codes (contains all squawks seen)
        
        Returns:
            [{code, count, flights: [...]}]
        """
        query = """
            SELECT flight_id, squawk_codes, callsign, airline
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND emergency_squawk_detected = 1
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        code_stats = defaultdict(lambda: {'flights': [], 'airlines': Counter()})
        
        for row in results:
            flight_id, squawk_codes, callsign, airline = row
            
            # Parse squawk codes (comma-separated or JSON list)
            if squawk_codes:
                codes = []
                if squawk_codes.startswith('['):
                    try:
                        codes = json.loads(squawk_codes)
                    except:
                        codes = [squawk_codes]
                else:
                    codes = [c.strip() for c in squawk_codes.split(',')]
                
                for code in codes:
                    if code in ('7500', '7600', '7700'):
                        code_stats[code]['flights'].append(flight_id)
                        if airline:
                            code_stats[code]['airlines'][airline] += 1
        
        result = []
        for code, data in code_stats.items():
            result.append({
                'code': code,
                'count': len(data['flights']),
                'airlines': dict(data['airlines']),
                'flights': data['flights'][:10]  # Limit to 10 examples
            })
        
        return sorted(result, key=lambda x: x['count'], reverse=True)
    
    def get_tagged_military_stats(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get military aircraft statistics from feedback_tagged.db using pre-computed fields.
        
        Uses:
        - flight_metadata.is_military (pre-computed, indexed)
        - flight_metadata.military_type (pre-computed)
        
        Returns:
            {total_military, by_type: {}, by_country: {}, flights: [...]}
        """
        query = """
            SELECT flight_id, callsign, military_type, airline_code, 
                   origin_airport, destination_airport, category
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        by_type = Counter()
        by_country = Counter()
        flights = []
        
        for row in results:
            flight_id, callsign, mil_type, airline_code, origin, dest, category = row
            
            if mil_type:
                by_type[mil_type] += 1
            elif category == 'Military_and_government':
                by_type['Government/Military'] += 1
            
            # Infer country from callsign
            country = 'Unknown'
            if callsign:
                cs = callsign.upper()
                if cs.startswith(('RCH', 'REACH', 'CNV', 'EVAC', 'N00')):
                    country = 'USA'
                elif cs.startswith(('RAF', 'RRR', 'ASCOT')):
                    country = 'UK'
                elif cs.startswith(('RFF', 'RSD')):
                    country = 'Russia'
                elif cs.startswith(('IAF', '4XA', '4XB', '4XC')):
                    country = 'Israel'
                elif cs.startswith(('SHAHD', 'RJAF')):
                    country = 'Jordan'
            
            by_country[country] += 1
            flights.append({
                'flight_id': flight_id,
                'callsign': callsign,
                'type': mil_type or category or 'Unknown',
                'country': country,
                'route': f"{origin or '?'} → {dest or '?'}"
            })
        
        return {
            'total_military': len(flights),
            'by_type': dict(by_type),
            'by_country': dict(by_country),
            'flights': flights[:50]  # Limit to 50
        }
    
    def get_tagged_signal_loss_stats(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get signal loss statistics from feedback_tagged.db using pre-computed field.
        
        Uses:
        - flight_metadata.signal_loss_events (pre-computed count)
        
        Returns:
            {total_events, affected_flights, avg_events_per_flight, flights_with_loss: [...]}
        """
        query = """
            SELECT flight_id, callsign, signal_loss_events, 
                   origin_airport, destination_airport
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND signal_loss_events > 0
            ORDER BY signal_loss_events DESC
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        total_events = 0
        flights_with_loss = []
        
        for row in results:
            flight_id, callsign, events, origin, dest = row
            total_events += events or 0
            flights_with_loss.append({
                'flight_id': flight_id,
                'callsign': callsign,
                'signal_loss_count': events,
                'route': f"{origin or '?'} → {dest or '?'}"
            })
        
        affected = len(flights_with_loss)
        return {
            'total_events': total_events,
            'affected_flights': affected,
            'avg_events_per_flight': round(total_events / max(affected, 1), 2),
            'flights_with_loss': flights_with_loss[:20]  # Limit to 20
        }
    
    def get_tagged_severity_distribution(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get severity distribution from feedback_tagged.db using pre-computed fields.
        
        Uses:
        - anomaly_reports.severity_cnn (pre-computed)
        - anomaly_reports.severity_dense (pre-computed)
        
        Returns:
            {
                'distribution': [{severity_range, count}],
                'avg_cnn': float,
                'avg_dense': float,
                'max_cnn': float,
                'max_dense': float
            }
        """
        query = """
            SELECT 
                severity_cnn, severity_dense,
                AVG(severity_cnn) as avg_cnn,
                AVG(severity_dense) as avg_dense,
                MAX(severity_cnn) as max_cnn,
                MAX(severity_dense) as max_dense,
                COUNT(*) as total
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY 
                CASE 
                    WHEN COALESCE(severity_cnn, 0) >= 0.8 THEN 'critical'
                    WHEN COALESCE(severity_cnn, 0) >= 0.6 THEN 'high'
                    WHEN COALESCE(severity_cnn, 0) >= 0.4 THEN 'medium'
                    WHEN COALESCE(severity_cnn, 0) >= 0.2 THEN 'low'
                    ELSE 'minimal'
                END
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        # Get aggregates separately
        agg_query = """
            SELECT 
                AVG(severity_cnn), AVG(severity_dense),
                MAX(severity_cnn), MAX(severity_dense)
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        agg_results = self._execute_query('tagged', agg_query, (start_ts, end_ts))
        
        distribution = []
        severity_ranges = ['minimal', 'low', 'medium', 'high', 'critical']
        range_counts = {r: 0 for r in severity_ranges}
        
        for row in results:
            severity_cnn = row[0] or 0
            count = row[6] if len(row) > 6 else 1
            
            if severity_cnn >= 0.8:
                range_counts['critical'] += count
            elif severity_cnn >= 0.6:
                range_counts['high'] += count
            elif severity_cnn >= 0.4:
                range_counts['medium'] += count
            elif severity_cnn >= 0.2:
                range_counts['low'] += count
            else:
                range_counts['minimal'] += count
        
        for r in severity_ranges:
            distribution.append({
                'severity_range': r,
                'count': range_counts[r]
            })
        
        avg_cnn, avg_dense, max_cnn, max_dense = (0, 0, 0, 0)
        if agg_results and agg_results[0]:
            avg_cnn = agg_results[0][0] or 0
            avg_dense = agg_results[0][1] or 0
            max_cnn = agg_results[0][2] or 0
            max_dense = agg_results[0][3] or 0
        
        return {
            'distribution': distribution,
            'avg_cnn': round(avg_cnn, 3),
            'avg_dense': round(avg_dense, 3),
            'max_cnn': round(max_cnn, 3),
            'max_dense': round(max_dense, 3)
        }
    
    def get_tagged_airline_stats(self, start_ts: int, end_ts: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get airline statistics from feedback_tagged.db using pre-computed fields.
        
        Uses:
        - flight_metadata.airline (indexed)
        - flight_metadata.flight_duration_sec, total_distance_nm, avg_speed_kts (pre-computed)
        
        Returns:
            [{airline, flight_count, avg_duration_hours, avg_distance_nm, anomaly_rate}]
        """
        query = """
            SELECT 
                airline,
                COUNT(*) as flight_count,
                AVG(flight_duration_sec) as avg_duration,
                AVG(total_distance_nm) as avg_distance,
                AVG(avg_speed_kts) as avg_speed,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND airline IS NOT NULL AND airline != ''
            GROUP BY airline
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC
            LIMIT ?
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, limit))
        
        data = []
        for row in results:
            airline, count, avg_dur, avg_dist, avg_speed, anomalies = row
            data.append({
                'airline': airline,
                'flight_count': count,
                'avg_duration_hours': round((avg_dur or 0) / 3600, 2),
                'avg_distance_nm': round(avg_dist or 0, 1),
                'avg_speed_kts': round(avg_speed or 0, 1),
                'anomaly_count': anomalies or 0,
                'anomaly_rate': round((anomalies or 0) / max(count, 1) * 100, 1)
            })
        
        return data
    
    def get_tagged_routes_stats(self, start_ts: int, end_ts: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get route statistics from feedback_tagged.db using pre-computed fields.
        
        Uses:
        - flight_metadata.origin_airport, destination_airport (indexed)
        - flight_metadata.flight_duration_sec, total_distance_nm (pre-computed)
        
        Returns:
            [{route, origin, destination, flight_count, avg_duration_hours, anomaly_rate}]
        """
        query = """
            SELECT 
                origin_airport,
                destination_airport,
                COUNT(*) as flight_count,
                AVG(flight_duration_sec) as avg_duration,
                AVG(total_distance_nm) as avg_distance,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND origin_airport IS NOT NULL AND destination_airport IS NOT NULL
            AND origin_airport != '' AND destination_airport != ''
            GROUP BY origin_airport, destination_airport
            ORDER BY COUNT(*) DESC
            LIMIT ?
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, limit))
        
        data = []
        for row in results:
            origin, dest, count, avg_dur, avg_dist, anomalies = row
            data.append({
                'route': f"{origin} → {dest}",
                'origin': origin,
                'destination': dest,
                'flight_count': count,
                'avg_duration_hours': round((avg_dur or 0) / 3600, 2),
                'avg_distance_nm': round(avg_dist or 0, 1),
                'anomaly_count': anomalies or 0,
                'anomaly_rate': round((anomalies or 0) / max(count, 1) * 100, 1)
            })
        
        return data
    
    def get_tagged_rtb_events(self, start_ts: int, end_ts: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get Return-To-Base events from feedback_tagged.db where rule_id = 6.
        
        This is a simple SQL query that fetches RTB events directly from the 
        user_feedback table joined with flight_metadata for additional details.
        
        Rule mapping (from reason.ts):
        - Rule 6: 'Return to Land' - Aircraft returning to departure airport
        
        Returns:
            [{flight_id, callsign, departure_time, duration_min, airport, max_outbound_nm}]
        """
        # Simple SQL query to get RTB events (rule_id = 6) from feedback_tagged.db
        query = """
            SELECT 
                uf.flight_id,
                fm.callsign,
                fm.first_seen_ts as departure_time,
                fm.last_seen_ts as landing_time,
                fm.flight_duration_sec,
                fm.origin_airport as airport,
                fm.total_distance_nm as max_outbound_nm,
                uf.comments,
                uf.other_details
            FROM user_feedback uf
            LEFT JOIN flight_metadata fm ON uf.flight_id = fm.flight_id
            WHERE uf.rule_id = 6
            AND uf.first_seen_ts BETWEEN ? AND ?
            AND uf.user_label = 1
            ORDER BY uf.first_seen_ts DESC
            LIMIT ?
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, limit))
        
        rtb_events = []
        for row in results:
            if not row:
                continue
            
            flight_id, callsign, departure_time, landing_time, duration_sec, airport, max_outbound_nm, comments, other_details = row
            
            # Calculate duration in minutes
            duration_min = 0
            if duration_sec:
                duration_min = round(duration_sec / 60, 1)
            elif departure_time and landing_time:
                duration_min = round((landing_time - departure_time) / 60, 1)
            
            rtb_events.append({
                'flight_id': flight_id,
                'callsign': callsign or 'UNKNOWN',
                'departure_time': departure_time or 0,
                'landing_time': landing_time or 0,
                'duration_min': duration_min,
                'airport': airport or 'UNKNOWN',
                'max_outbound_nm': round(max_outbound_nm, 1) if max_outbound_nm else 0,
                'comments': comments or '',
                'other_details': other_details or ''
            })
        
        return rtb_events
    
    # ==========================================
    # New optimized methods for dashboard
    # ==========================================
    
    def get_emergency_aftermath(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get emergency aftermath statistics - what happened after emergency codes.
        
        Analyzes flights that declared emergencies (7700, 7600, 7500) and determines
        their outcome: landed safely, diverted, returned to base, or unknown.
        
        Returns:
            {
                'total_emergencies': int,
                'outcomes': {landed_safely, diverted, rtb, unknown},
                'by_code': {7700: count, 7600: count, 7500: count},
                'by_airline': [{airline, count, outcome_breakdown}],
                'recent_events': [{flight_id, callsign, code, outcome, airport}]
            }
        """
        outcomes = {
            'landed_safely': 0,
            'diverted': 0,
            'rtb': 0,  # Return to base
            'unknown': 0
        }
        by_code = {'7700': 0, '7600': 0, '7500': 0}
        by_airline = defaultdict(lambda: {'count': 0, 'landed': 0, 'diverted': 0, 'rtb': 0})
        recent_events = []
        
        # Query anomaly reports for emergency squawks (Rule 1)
        query = """
            SELECT ar.flight_id, ar.timestamp, ar.full_report,
                   fm.callsign, fm.airline, fm.origin_airport, fm.destination_airport,
                   fm.squawk_codes
            FROM anomaly_reports ar
            LEFT JOIN flight_metadata fm ON ar.flight_id = fm.flight_id
            WHERE ar.timestamp BETWEEN ? AND ?
            AND ar.matched_rule_ids LIKE '%1%'
            ORDER BY ar.timestamp DESC
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        for row in results:
            flight_id, ts, report_json, callsign, airline, origin, dest, squawks = row
            
            # Determine which emergency code
            code = None
            if squawks:
                for c in ['7700', '7600', '7500']:
                    if c in squawks:
                        code = c
                        by_code[c] += 1
                        break
            
            # Determine outcome based on destination vs actual landing
            outcome = 'unknown'
            actual_airport = dest  # Default to planned destination
            
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                # Check for diversion (Rule 8)
                for rule in matched_rules:
                    if rule.get('id') == 8:
                        outcome = 'diverted'
                        actual_airport = rule.get('details', {}).get('actual', dest)
                        break
                
                # Check for RTB (same origin and destination)
                if outcome == 'unknown' and origin and dest and origin == dest:
                    outcome = 'rtb'
                elif outcome == 'unknown' and dest:
                    outcome = 'landed_safely'
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            
            outcomes[outcome] += 1
            
            # Track by airline
            if airline:
                by_airline[airline]['count'] += 1
                if outcome == 'landed_safely':
                    by_airline[airline]['landed'] += 1
                elif outcome == 'diverted':
                    by_airline[airline]['diverted'] += 1
                elif outcome == 'rtb':
                    by_airline[airline]['rtb'] += 1
            
            # Add to recent events (limit to 20)
            if len(recent_events) < 20:
                recent_events.append({
                    'flight_id': flight_id,
                    'callsign': callsign or 'UNKNOWN',
                    'code': code or 'UNKNOWN',
                    'outcome': outcome,
                    'airport': actual_airport or 'UNKNOWN',
                    'timestamp': ts
                })
        
        # Format airline stats
        airline_stats = []
        for airline, stats in sorted(by_airline.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            airline_stats.append({
                'airline': airline,
                'count': stats['count'],
                'landed_safely': stats['landed'],
                'diverted': stats['diverted'],
                'rtb': stats['rtb']
            })
        
        return {
            'total_emergencies': sum(outcomes.values()),
            'outcomes': outcomes,
            'by_code': by_code,
            'by_airline': airline_stats,
            'recent_events': recent_events
        }
    
    def get_top_airline_emergencies(self, start_ts: int, end_ts: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get airlines with most emergency declarations.
        
        Uses pre-computed emergency_squawk_detected field from flight_metadata.
        
        Returns:
            [{airline, emergency_count, total_flights, emergency_rate}]
        """
        query = """
            SELECT 
                airline,
                SUM(CASE WHEN emergency_squawk_detected = 1 THEN 1 ELSE 0 END) as emergencies,
                COUNT(*) as total_flights
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND airline IS NOT NULL AND airline != ''
            GROUP BY airline
            HAVING SUM(CASE WHEN emergency_squawk_detected = 1 THEN 1 ELSE 0 END) > 0
            ORDER BY emergencies DESC
            LIMIT ?
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, limit))
        
        data = []
        for row in results:
            airline, emergencies, total = row
            data.append({
                'airline': airline,
                'emergency_count': emergencies,
                'total_flights': total,
                'emergency_rate': round(emergencies / max(total, 1) * 100, 2)
            })
        
        return data
    
    def get_emergency_clusters(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Detect days with multiple emergency incidents and identify geographic clusters.
        
        Answers: "Were there multiple incidents in one day? Were they in the same area?"
        
        Returns:
            {
                'multi_incident_days': [{date, count, events, cluster_detected}],
                'geographic_clusters': [{area_name, lat, lon, count, dates}],
                'total_cluster_days': int,
                'insights': [str]
            }
        """
        from datetime import datetime
        from collections import defaultdict
        
        # Query emergency events with location
        query = """
            SELECT 
                ar.flight_id, 
                ar.timestamp, 
                ar.full_report,
                fm.callsign,
                fm.origin_airport,
                fm.destination_airport,
                fm.squawk_codes
            FROM anomaly_reports ar
            LEFT JOIN flight_metadata fm ON ar.flight_id = fm.flight_id
            WHERE ar.timestamp BETWEEN ? AND ?
            AND ar.matched_rule_ids LIKE '%1%'
            ORDER BY ar.timestamp
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        # Group events by date
        events_by_date = defaultdict(list)
        all_events = []
        
        for row in results:
            flight_id, ts, report_json, callsign, origin, dest, squawks = row
            
            date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            
            # Extract location from report
            lat, lon = None, None
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                summary = report.get('summary', {})
                # Try to get coordinates from the report
                if 'lat' in summary and 'lon' in summary:
                    lat = summary['lat']
                    lon = summary['lon']
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            
            # Determine emergency code
            code = 'UNKNOWN'
            if squawks:
                for c in ['7700', '7600', '7500']:
                    if c in squawks:
                        code = c
                        break
            
            event = {
                'flight_id': flight_id,
                'timestamp': ts,
                'date': date_str,
                'callsign': callsign or 'UNKNOWN',
                'code': code,
                'origin': origin,
                'destination': dest,
                'lat': lat,
                'lon': lon
            }
            
            events_by_date[date_str].append(event)
            all_events.append(event)
        
        # Identify multi-incident days (2+ emergencies in same day)
        multi_incident_days = []
        for date_str, events in sorted(events_by_date.items(), key=lambda x: len(x[1]), reverse=True):
            if len(events) >= 2:
                # Check if events are in same area (within ~50nm of each other)
                cluster_detected = False
                events_with_coords = [e for e in events if e['lat'] and e['lon']]
                
                if len(events_with_coords) >= 2:
                    # Simple proximity check
                    for i, e1 in enumerate(events_with_coords):
                        for e2 in events_with_coords[i+1:]:
                            lat_diff = abs(e1['lat'] - e2['lat'])
                            lon_diff = abs(e1['lon'] - e2['lon'])
                            # Roughly 1 degree = 60nm, so 0.8 degree = ~50nm
                            if lat_diff < 0.8 and lon_diff < 0.8:
                                cluster_detected = True
                                break
                        if cluster_detected:
                            break
                
                multi_incident_days.append({
                    'date': date_str,
                    'count': len(events),
                    'events': [{
                        'callsign': e['callsign'],
                        'code': e['code'],
                        'time': datetime.fromtimestamp(e['timestamp']).strftime('%H:%M')
                    } for e in events],
                    'cluster_detected': cluster_detected
                })
        
        # Find geographic clusters across all dates
        geographic_clusters = []
        grid_size = 1.0  # 1 degree grid cells (~60nm)
        location_counts = defaultdict(lambda: {'count': 0, 'dates': set(), 'events': []})
        
        for event in all_events:
            if event['lat'] and event['lon']:
                grid_key = (round(event['lat'] / grid_size) * grid_size, 
                           round(event['lon'] / grid_size) * grid_size)
                location_counts[grid_key]['count'] += 1
                location_counts[grid_key]['dates'].add(event['date'])
                location_counts[grid_key]['events'].append(event)
        
        for (lat, lon), data in sorted(location_counts.items(), key=lambda x: x[1]['count'], reverse=True):
            if data['count'] >= 2:  # At least 2 events in this area
                # Generate area name based on coordinates
                area_name = f"Area {lat:.1f}°N, {lon:.1f}°E"
                
                geographic_clusters.append({
                    'area_name': area_name,
                    'lat': lat,
                    'lon': lon,
                    'count': data['count'],
                    'dates': sorted(list(data['dates']))[:5],  # Top 5 dates
                    'unique_days': len(data['dates'])
                })
        
        # Generate insights
        insights = []
        cluster_days = len([d for d in multi_incident_days if d['cluster_detected']])
        
        if multi_incident_days:
            insights.append(f"Found {len(multi_incident_days)} days with multiple emergency incidents")
        
        if cluster_days > 0:
            insights.append(f"{cluster_days} days had emergencies in the same geographic area")
        
        if geographic_clusters:
            top_area = geographic_clusters[0]
            insights.append(f"Most affected area: {top_area['area_name']} with {top_area['count']} emergencies")
        
        max_day = multi_incident_days[0] if multi_incident_days else None
        if max_day and max_day['count'] >= 3:
            insights.append(f"Highest concentration: {max_day['count']} emergencies on {max_day['date']}")
        
        return {
            'multi_incident_days': multi_incident_days[:20],  # Limit to top 20 days
            'geographic_clusters': geographic_clusters[:10],  # Top 10 areas
            'total_cluster_days': cluster_days,
            'total_multi_incident_days': len(multi_incident_days),
            'insights': insights
        }
    
    def get_airport_hourly_traffic(self, airport: str, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get hourly traffic distribution for a specific airport.
        
        Answers: "What is the busiest hour at LLBG?"
        
        Uses pre-computed origin_airport and destination_airport fields.
        
        Returns:
            [{hour: 0-23, departures: int, arrivals: int, total: int}]
        """
        # Initialize hourly stats
        hourly = {h: {'departures': 0, 'arrivals': 0} for h in range(24)}
        
        # Count departures by hour
        query = """
            SELECT 
                CAST(strftime('%H', first_seen_ts, 'unixepoch') AS INTEGER) as hour,
                COUNT(*) as count
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND origin_airport = ?
            GROUP BY hour
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, airport))
        for row in results:
            hour, count = row
            if hour is not None:
                hourly[hour]['departures'] = count
        
        # Count arrivals by hour (using last_seen_ts for arrival time)
        query = """
            SELECT 
                CAST(strftime('%H', last_seen_ts, 'unixepoch') AS INTEGER) as hour,
                COUNT(*) as count
            FROM flight_metadata
            WHERE last_seen_ts BETWEEN ? AND ?
            AND destination_airport = ?
            GROUP BY hour
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, airport))
        for row in results:
            hour, count = row
            if hour is not None:
                hourly[hour]['arrivals'] = count
        
        # Format output
        data = []
        for hour in range(24):
            stats = hourly[hour]
            data.append({
                'hour': hour,
                'departures': stats['departures'],
                'arrivals': stats['arrivals'],
                'total': stats['departures'] + stats['arrivals']
            })
        
        return data
    
    def get_flights_per_month(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get monthly flight aggregation.
        
        Answers: "Which month has the most flights?"
        
        Uses pre-computed fields from flight_metadata in research_new.db.
        Military detection checks both is_military field AND category='Military_and_government'.
        
        Returns:
            [{month: 'YYYY-MM', total_flights, military_count, anomaly_count, avg_duration_hours}]
        """
        query = """
            SELECT 
                strftime('%Y-%m', first_seen_ts, 'unixepoch') as month,
                COUNT(*) as total,
                SUM(CASE WHEN is_military = 1 OR category = 'Military_and_government' THEN 1 ELSE 0 END) as military,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies,
                AVG(flight_duration_sec) as avg_duration
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            GROUP BY month
            ORDER BY month
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        data = []
        for row in results:
            month, total, military, anomalies, avg_dur = row
            if month:
                data.append({
                    'month': month,
                    'total_flights': total or 0,
                    'military_count': military or 0,
                    'anomaly_count': anomalies or 0,
                    'avg_duration_hours': round((avg_dur or 0) / 3600, 2)
                })
        
        return data
    
    def get_near_miss_by_country(self, start_ts: int, end_ts: int, country: str = None) -> Dict[str, Any]:
        """
        Get near-miss events filtered by country/region.
        
        Uses pre-computed crossed_borders field from flight_metadata.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            country: Optional 2-letter country code (IL, JO, etc.) to filter by
        
        Returns:
            {
                'total_near_miss': int,
                'by_country': {country_code: count},
                'events': [{flight_id, callsign, countries, severity}]
            }
        """
        # Query near-miss events (Rule 4) with country data
        if country:
            query = """
                SELECT ar.flight_id, ar.timestamp, ar.severity_cnn,
                       fm.callsign, fm.crossed_borders
                FROM anomaly_reports ar
                LEFT JOIN flight_metadata fm ON ar.flight_id = fm.flight_id
                WHERE ar.timestamp BETWEEN ? AND ?
                AND ar.matched_rule_ids LIKE '%4%'
                AND fm.crossed_borders LIKE ?
                ORDER BY ar.timestamp DESC
            """
            results = self._execute_query('tagged', query, (start_ts, end_ts, f'%{country}%'))
        else:
            query = """
                SELECT ar.flight_id, ar.timestamp, ar.severity_cnn,
                       fm.callsign, fm.crossed_borders
                FROM anomaly_reports ar
                LEFT JOIN flight_metadata fm ON ar.flight_id = fm.flight_id
                WHERE ar.timestamp BETWEEN ? AND ?
                AND ar.matched_rule_ids LIKE '%4%'
                ORDER BY ar.timestamp DESC
            """
            results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        by_country = Counter()
        events = []
        
        for row in results:
            flight_id, ts, severity, callsign, borders = row
            
            # Parse crossed borders
            countries_list = []
            if borders:
                countries_list = [c.strip() for c in borders.split(',') if c.strip()]
                for c in countries_list:
                    by_country[c] += 1
            
            if len(events) < 50:
                events.append({
                    'flight_id': flight_id,
                    'callsign': callsign or 'UNKNOWN',
                    'countries': countries_list,
                    'severity': round(severity or 0, 2),
                    'timestamp': ts
                })
        
        return {
            'total_near_miss': len(results) if results else 0,
            'by_country': dict(by_country.most_common()),
            'events': events
        }
    
    def get_bottleneck_zones(self, start_ts: int, end_ts: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get airspace bottleneck zones based on traffic density and holding patterns.
        
        Aggregates flight track points into geographic grid cells and calculates
        congestion metrics.
        
        Returns:
            [{lat, lon, flight_count, holding_count, density_score, congestion_level, avg_altitude, flights_per_hour}]
        """
        import math

        grid_size = 0.5  # ~55km cells
        
        # Grid cell stats
        cells = defaultdict(lambda: {
            'flights': set(),
            'holding_count': 0,
            'lat_sum': 0.0,
            'lon_sum': 0.0,
            'point_count': 0,
            'altitudes': [],
            'timestamps': []
        })
        
        # Query track points from research.db (including altitude and timestamp)
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT flight_id, lat, lon, alt, timestamp
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                AND lat IS NOT NULL AND lon IS NOT NULL
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            for row in results:
                flight_id, lat, lon, alt, ts = row
                grid_lat = round(lat / grid_size) * grid_size
                grid_lon = round(lon / grid_size) * grid_size
                key = (grid_lat, grid_lon)
                
                cells[key]['flights'].add(flight_id)
                cells[key]['lat_sum'] += lat
                cells[key]['lon_sum'] += lon
                cells[key]['point_count'] += 1
                if alt:
                    cells[key]['altitudes'].append(alt)
                if ts:
                    cells[key]['timestamps'].append(ts)
        
        # Count holding patterns per cell from anomaly reports
        query = """
            SELECT full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND matched_rule_ids LIKE '%3%'
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            try:
                report = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 3:
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            lat = event.get('lat')
                            lon = event.get('lon')
                            if lat and lon:
                                grid_lat = round(lat / grid_size) * grid_size
                                grid_lon = round(lon / grid_size) * grid_size
                                cells[(grid_lat, grid_lon)]['holding_count'] += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Calculate density scores and format output
        result = []
        max_flights = max((len(c['flights']) for c in cells.values()), default=1)
        
        for (grid_lat, grid_lon), data in cells.items():
            flight_count = len(data['flights'])
            if flight_count < 2:  # Skip low traffic cells
                continue
            
            # Calculate centroid
            centroid_lat = data['lat_sum'] / max(data['point_count'], 1)
            centroid_lon = data['lon_sum'] / max(data['point_count'], 1)
            
            # Density score based on flight count and holding patterns
            density_score = (flight_count / max_flights) * 50 + (data['holding_count'] * 5)
            density_score = min(100, density_score)
            
            # Classify congestion level
            if density_score >= 50:
                level = 'critical'
            elif density_score >= 30:
                level = 'high'
            elif density_score >= 15:
                level = 'moderate'
            else:
                level = 'low'
            
            # Calculate average altitude
            avg_altitude = sum(data['altitudes']) / len(data['altitudes']) if data['altitudes'] else 0
            
            # Calculate flights per hour
            if data['timestamps']:
                time_spread = max(data['timestamps']) - min(data['timestamps'])
                hours_active = max(time_spread / 3600, 1)
                flights_per_hour = flight_count / hours_active
            else:
                flights_per_hour = 0
            
            result.append({
                'lat': round(centroid_lat, 4),
                'lon': round(centroid_lon, 4),
                'flight_count': flight_count,
                'holding_count': data['holding_count'],
                'density_score': round(density_score, 1),
                'congestion_level': level,
                'avg_altitude': round(avg_altitude),
                'flights_per_hour': round(flights_per_hour, 1)
            })
        
        # Sort by density score and limit
        return sorted(result, key=lambda x: x['density_score'], reverse=True)[:limit]
    
    def get_deviations_by_aircraft_type(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get route deviations grouped by aircraft type.
        
        Answers: "Which aircraft types deviate most from defined routes?"
        
        Returns:
            [{aircraft_type, deviation_count, avg_deviation_nm, large_deviations}]
        """
        # Query anomaly reports with aircraft type from metadata
        query = """
            SELECT fm.aircraft_type, ar.full_report
            FROM anomaly_reports ar
            LEFT JOIN flight_metadata fm ON ar.flight_id = fm.flight_id
            WHERE ar.timestamp BETWEEN ? AND ?
            AND (ar.matched_rule_ids LIKE '%5%' OR ar.matched_rule_ids LIKE '%11%')
            AND fm.aircraft_type IS NOT NULL AND fm.aircraft_type != ''
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        type_stats = defaultdict(lambda: {
            'count': 0,
            'total_deviation_nm': 0,
            'large_count': 0  # > 20nm
        })
        
        for row in results:
            aircraft_type, report_json = row
            
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    details = rule.get('details', {})
                    
                    if rule_id in [5, 11]:  # Route deviation rules
                        type_stats[aircraft_type]['count'] += 1
                        
                        # Try to get deviation distance
                        deviation_nm = details.get('max_deviation_nm', 0)
                        if not deviation_nm:
                            off_path = details.get('off_path', [])
                            if off_path:
                                deviation_nm = max(p.get('distance_nm', 0) for p in off_path)
                        
                        type_stats[aircraft_type]['total_deviation_nm'] += deviation_nm
                        if deviation_nm > 20:
                            type_stats[aircraft_type]['large_count'] += 1
                            
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Format output
        result = []
        for aircraft_type, stats in sorted(type_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            if stats['count'] > 0:
                result.append({
                    'aircraft_type': aircraft_type,
                    'deviation_count': stats['count'],
                    'avg_deviation_nm': round(stats['total_deviation_nm'] / stats['count'], 1),
                    'large_deviations': stats['large_count']
                })
        
        return result[:20]  # Limit to top 20
    
    # ============================================================================
    # SEASONAL TRENDS METHODS
    # ============================================================================
    
    def get_seasonal_year_comparison(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Compare flight and safety statistics across years for the same period.
        
        Answers: "How does this year compare to last year?" and "Yom Kippur effect"
        
        Returns:
            {
                'years': [{year, total_flights, anomalies, safety_events, military_flights}],
                'month_comparison': [{month, current_year, previous_year, change_percent}],
                'insights': [...]
            }
        """
        # Get the time range span
        days_span = (end_ts - start_ts) / 86400
        
        # Query by year from tagged db
        query = """
            SELECT 
                strftime('%Y', first_seen_ts, 'unixepoch') as year,
                strftime('%m', first_seen_ts, 'unixepoch') as month,
                COUNT(*) as total_flights,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies,
                SUM(CASE WHEN is_military = 1 OR category = 'Military_and_government' THEN 1 ELSE 0 END) as military_flights
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            GROUP BY year, month
            ORDER BY year, month
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        # Organize by year
        year_data = defaultdict(lambda: {
            'total_flights': 0, 'anomalies': 0, 'military_flights': 0,
            'months': defaultdict(lambda: {'flights': 0, 'anomalies': 0})
        })
        
        for row in results:
            year, month, total, anomalies, military = row
            if year:
                year_data[year]['total_flights'] += total or 0
                year_data[year]['anomalies'] += anomalies or 0
                year_data[year]['military_flights'] += military or 0
                if month:
                    year_data[year]['months'][month]['flights'] = total or 0
                    year_data[year]['months'][month]['anomalies'] = anomalies or 0
        
        # Get safety events per year
        query = """
            SELECT 
                strftime('%Y', timestamp, 'unixepoch') as year,
                COUNT(DISTINCT flight_id) as safety_events
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND matched_rule_ids IS NOT NULL
            AND (matched_rule_ids LIKE '%1%' OR matched_rule_ids LIKE '%4%' OR matched_rule_ids LIKE '%6%')
            GROUP BY year
        """
        safety_results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in safety_results:
            year, safety = row
            if year and year in year_data:
                year_data[year]['safety_events'] = safety or 0
        
        # Format years output
        years_sorted = sorted(year_data.keys())
        years_output = []
        for year in years_sorted:
            data = year_data[year]
            years_output.append({
                'year': int(year),
                'total_flights': data['total_flights'],
                'anomalies': data['anomalies'],
                'safety_events': data.get('safety_events', 0),
                'military_flights': data['military_flights']
            })
        
        # Month-over-month comparison (if we have 2+ years)
        month_comparison = []
        if len(years_sorted) >= 2:
            current_year = years_sorted[-1]
            prev_year = years_sorted[-2]
            
            for month in sorted(year_data[current_year]['months'].keys()):
                curr_flights = year_data[current_year]['months'][month]['flights']
                prev_flights = year_data[prev_year]['months'].get(month, {}).get('flights', 0)
                
                change_pct = 0
                if prev_flights > 0:
                    change_pct = round((curr_flights - prev_flights) / prev_flights * 100, 1)
                
                month_comparison.append({
                    'month': month,
                    'month_name': datetime(2000, int(month), 1).strftime('%B'),
                    'current_year': curr_flights,
                    'previous_year': prev_flights,
                    'change_percent': change_pct
                })
        
        # Generate insights
        insights = []
        if len(years_output) >= 2:
            curr = years_output[-1]
            prev = years_output[-2]
            
            if curr['total_flights'] > prev['total_flights']:
                pct = round((curr['total_flights'] - prev['total_flights']) / max(prev['total_flights'], 1) * 100, 1)
                insights.append(f"Traffic increased by {pct}% compared to {prev['year']}")
            elif curr['total_flights'] < prev['total_flights']:
                pct = round((prev['total_flights'] - curr['total_flights']) / max(prev['total_flights'], 1) * 100, 1)
                insights.append(f"Traffic decreased by {pct}% compared to {prev['year']}")
            
            if curr.get('safety_events', 0) > prev.get('safety_events', 0) * 1.2:
                insights.append(f"Safety events increased significantly in {curr['year']}")
        
        return {
            'years': years_output,
            'month_comparison': month_comparison,
            'insights': insights
        }
    
    def get_traffic_safety_correlation(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze correlation between traffic volume and safety events.
        
        Answers: "Rush hours correlate with 40% more safety events"
        
        Returns:
            {
                'hourly_correlation': [{hour, traffic_count, safety_count, ratio}],
                'correlation_score': float (0-1),
                'peak_risk_hours': [int],
                'insights': [str]
            }
        """
        # Get hourly traffic from tracks
        query = """
            SELECT 
                CAST(strftime('%H', timestamp, 'unixepoch') AS INTEGER) as hour,
                COUNT(DISTINCT flight_id) as flight_count
            FROM (
                SELECT flight_id, timestamp FROM anomalies_tracks WHERE timestamp BETWEEN ? AND ?
                UNION ALL
                SELECT flight_id, timestamp FROM normal_tracks WHERE timestamp BETWEEN ? AND ?
            )
            GROUP BY hour
            ORDER BY hour
        """
        traffic_results = self._execute_query('research', query, (start_ts, end_ts, start_ts, end_ts))
        
        hourly_traffic = {int(row[0]): row[1] for row in traffic_results if row[0] is not None}
        
        # Get hourly safety events
        query = """
            SELECT 
                CAST(strftime('%H', timestamp, 'unixepoch') AS INTEGER) as hour,
                COUNT(DISTINCT flight_id) as safety_count
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND matched_rule_ids IS NOT NULL
            AND (matched_rule_ids LIKE '%1%' OR matched_rule_ids LIKE '%4%' OR matched_rule_ids LIKE '%6%')
            GROUP BY hour
            ORDER BY hour
        """
        safety_results = self._execute_query('research', query, (start_ts, end_ts))
        
        hourly_safety = {int(row[0]): row[1] for row in safety_results if row[0] is not None}
        
        # Calculate correlation
        hourly_data = []
        traffic_values = []
        safety_values = []
        
        for hour in range(24):
            traffic = hourly_traffic.get(hour, 0)
            safety = hourly_safety.get(hour, 0)
            
            ratio = 0
            if traffic > 0:
                ratio = round(safety / traffic * 1000, 2)  # Safety events per 1000 flights
            
            hourly_data.append({
                'hour': hour,
                'traffic_count': traffic,
                'safety_count': safety,
                'safety_per_1000': ratio
            })
            
            traffic_values.append(traffic)
            safety_values.append(safety)
        
        # Calculate Pearson correlation coefficient
        correlation_score = 0.0
        if len(traffic_values) >= 2:
            n = len(traffic_values)
            sum_x = sum(traffic_values)
            sum_y = sum(safety_values)
            sum_xy = sum(x * y for x, y in zip(traffic_values, safety_values))
            sum_x2 = sum(x ** 2 for x in traffic_values)
            sum_y2 = sum(y ** 2 for y in safety_values)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
            
            if denominator > 0:
                correlation_score = round(numerator / denominator, 3)
        
        # Find peak risk hours (highest safety_per_1000)
        sorted_by_risk = sorted(hourly_data, key=lambda x: x['safety_per_1000'], reverse=True)
        peak_risk_hours = [h['hour'] for h in sorted_by_risk[:3] if h['safety_per_1000'] > 0]
        
        # Generate insights
        insights = []
        
        if correlation_score > 0.7:
            insights.append(f"Strong correlation ({correlation_score:.0%}) between traffic and safety events")
        elif correlation_score > 0.4:
            insights.append(f"Moderate correlation ({correlation_score:.0%}) between traffic and safety events")
        else:
            insights.append("Safety events appear independent of traffic volume")
        
        if peak_risk_hours:
            peak_str = ', '.join(f"{h}:00" for h in peak_risk_hours)
            insights.append(f"Highest risk hours: {peak_str}")
            
            # Compare peak to off-peak
            peak_avg = sum(hourly_data[h]['safety_per_1000'] for h in peak_risk_hours) / len(peak_risk_hours)
            off_peak = [h['safety_per_1000'] for h in hourly_data if h['hour'] not in peak_risk_hours and h['safety_per_1000'] > 0]
            if off_peak:
                off_peak_avg = sum(off_peak) / len(off_peak)
                if off_peak_avg > 0 and peak_avg > off_peak_avg:
                    increase = round((peak_avg - off_peak_avg) / off_peak_avg * 100)
                    insights.append(f"Peak hours have {increase}% more safety events per flight")
        
        return {
            'hourly_correlation': hourly_data,
            'correlation_score': correlation_score,
            'peak_risk_hours': peak_risk_hours,
            'insights': insights
        }
    
    def get_special_events_impact(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Detect traffic patterns around special events/holidays.
        
        Answers: "Yom Kippur effect: reduced volume and altered routing"
        
        Returns:
            {
                'detected_events': [{date, event_name, traffic_change_percent, anomaly_change_percent}],
                'weekly_pattern': [{day_of_week, avg_traffic, avg_anomalies}],
                'insights': [str]
            }
        """
        # Known special events (dates that typically have unusual traffic)
        # Format: (month, day, event_name, expected_change)
        known_events = [
            (9, None, "Yom Kippur", -80),  # Day varies by year
            (10, None, "Yom Kippur", -80),
            (12, 25, "Christmas", -20),
            (1, 1, "New Year", -15),
            (3, None, "Purim", -10),
            (4, None, "Passover", -30),
        ]
        
        # Get daily flight counts
        query = """
            SELECT 
                DATE(first_seen_ts, 'unixepoch') as date,
                strftime('%w', first_seen_ts, 'unixepoch') as day_of_week,
                COUNT(*) as total_flights,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            GROUP BY date
            ORDER BY date
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        daily_data = []
        weekly_totals = defaultdict(lambda: {'flights': [], 'anomalies': []})
        
        for row in results:
            date_str, dow, flights, anomalies = row
            if date_str:
                daily_data.append({
                    'date': date_str,
                    'day_of_week': int(dow) if dow else 0,
                    'flights': flights or 0,
                    'anomalies': anomalies or 0
                })
                if dow is not None:
                    weekly_totals[int(dow)]['flights'].append(flights or 0)
                    weekly_totals[int(dow)]['anomalies'].append(anomalies or 0)
        
        # Calculate weekly pattern
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        weekly_pattern = []
        for dow in range(7):
            data = weekly_totals[dow]
            avg_flights = sum(data['flights']) / len(data['flights']) if data['flights'] else 0
            avg_anomalies = sum(data['anomalies']) / len(data['anomalies']) if data['anomalies'] else 0
            weekly_pattern.append({
                'day_of_week': dow,
                'day_name': day_names[dow],
                'avg_traffic': round(avg_flights, 1),
                'avg_anomalies': round(avg_anomalies, 1)
            })
        
        # Detect unusual days (significant drops from weekly average)
        detected_events = []
        if daily_data:
            overall_avg = sum(d['flights'] for d in daily_data) / len(daily_data)
            
            for day in daily_data:
                dow_avg = weekly_pattern[day['day_of_week']]['avg_traffic']
                if dow_avg > 0 and day['flights'] < dow_avg * 0.5:  # More than 50% drop
                    change_pct = round((day['flights'] - dow_avg) / dow_avg * 100, 1)
                    
                    # Try to identify the event
                    event_name = "Unusual low traffic"
                    try:
                        date_obj = datetime.strptime(day['date'], '%Y-%m-%d')
                        for month, day_num, name, _ in known_events:
                            if date_obj.month == month:
                                if day_num is None or date_obj.day == day_num:
                                    event_name = f"Possible {name}"
                                    break
                    except:
                        pass
                    
                    detected_events.append({
                        'date': day['date'],
                        'event_name': event_name,
                        'traffic_change_percent': change_pct,
                        'flights': day['flights'],
                        'expected_flights': round(dow_avg)
                    })
        
        # Generate insights
        insights = []
        
        if weekly_pattern:
            # Find busiest and quietest days
            busiest = max(weekly_pattern, key=lambda x: x['avg_traffic'])
            quietest = min(weekly_pattern, key=lambda x: x['avg_traffic'])
            insights.append(f"Busiest day: {busiest['day_name']} ({busiest['avg_traffic']:.0f} avg flights)")
            insights.append(f"Quietest day: {quietest['day_name']} ({quietest['avg_traffic']:.0f} avg flights)")
        
        if detected_events:
            insights.append(f"Detected {len(detected_events)} unusual traffic days")
        
        return {
            'detected_events': detected_events[:10],  # Limit to top 10
            'weekly_pattern': weekly_pattern,
            'insights': insights
        }
    
    # ============================================================================
    # ROUTE EFFICIENCY METHODS
    # ============================================================================
    
    def get_route_efficiency_comparison(self, route: str = None, start_ts: int = 0, end_ts: int = 0) -> Dict[str, Any]:
        """
        Compare airline efficiency on the same route.
        
        Answers: "Why does Airline A fly 15 minutes longer than B?"
        
        Args:
            route: Route in format "ORIG-DEST" (e.g., "LLBG-EGLL")
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            {
                'route': str,
                'airlines': [{airline, flights, avg_duration_min, avg_deviation_nm, efficiency_score}],
                'best_performer': str,
                'worst_performer': str,
                'time_difference_min': float,
                'insights': [str]
            }
        """
        # If no route specified, get top routes first
        if not route:
            return self._get_top_routes_efficiency(start_ts, end_ts)
        
        # Parse route
        parts = route.upper().split('-')
        if len(parts) != 2:
            return {'error': 'Route must be in format ORIG-DEST (e.g., LLBG-EGLL)'}
        
        origin, dest = parts
        
        # Query flights on this route with airline/callsign info
        # NOTE: Use subquery to get ONE anomaly_report per flight (flights can have multiple reports)
        # This prevents double-counting flights in aggregations
        query = """
            SELECT 
                fm.callsign,
                SUBSTR(fm.callsign, 1, 3) as airline,
                fm.flight_duration_sec,
                fm.is_anomaly,
                ar.full_report
            FROM flight_metadata fm
            LEFT JOIN (
                SELECT flight_id, full_report
                FROM anomaly_reports
                WHERE flight_id IN (
                    SELECT flight_id FROM flight_metadata 
                    WHERE origin_airport = ? AND destination_airport = ?
                    AND first_seen_ts BETWEEN ? AND ?
                )
                GROUP BY flight_id
            ) ar ON fm.flight_id = ar.flight_id
            WHERE fm.origin_airport = ? AND fm.destination_airport = ?
            AND fm.first_seen_ts BETWEEN ? AND ?
            AND fm.callsign IS NOT NULL AND fm.callsign != ''
        """
        # Parameters: (origin, dest, start_ts, end_ts) for subquery + (origin, dest, start_ts, end_ts) for main query
        results = self._execute_query('tagged', query, (origin, dest, start_ts, end_ts, origin, dest, start_ts, end_ts))
        
        if not results:
            return {
                'route': route,
                'airlines': [],
                'best_performer': None,
                'worst_performer': None,
                'insights': [f'No flights found for route {route}']
            }
        
        # Group by airline
        airline_stats = defaultdict(lambda: {
            'durations': [],
            'deviations': [],
            'anomaly_count': 0,
            'flights': 0
        })
        
        for row in results:
            callsign, airline, duration_sec, is_anomaly, report_json = row
            if not airline or len(airline) < 2:
                continue
            
            airline_stats[airline]['flights'] += 1
            if duration_sec:
                airline_stats[airline]['durations'].append(duration_sec)
            if is_anomaly:
                airline_stats[airline]['anomaly_count'] += 1
            
            # Try to extract deviation info from report
            if report_json:
                try:
                    report = json.loads(report_json) if isinstance(report_json, str) else report_json
                    matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                    for rule in matched_rules:
                        if rule.get('id') in [5, 11]:  # Route deviation rules
                            deviation_nm = rule.get('details', {}).get('max_deviation_nm', 0)
                            if deviation_nm:
                                airline_stats[airline]['deviations'].append(deviation_nm)
                except:
                    pass
        
        # Calculate metrics for each airline
        airlines = []
        for airline, stats in airline_stats.items():
            if stats['flights'] < 1:
                continue
            
            avg_duration = sum(stats['durations']) / len(stats['durations']) if stats['durations'] else 0
            avg_deviation = sum(stats['deviations']) / len(stats['deviations']) if stats['deviations'] else 0
            
            # Efficiency score (lower is better): based on duration, deviations, and anomaly rate
            anomaly_rate = stats['anomaly_count'] / stats['flights']
            efficiency_score = 100 - (avg_deviation * 0.5) - (anomaly_rate * 20)
            efficiency_score = max(0, min(100, efficiency_score))
            
            airlines.append({
                'airline': airline,
                'flights': stats['flights'],
                'avg_duration_min': round(avg_duration / 60, 1),
                'avg_deviation_nm': round(avg_deviation, 1),
                'anomaly_rate': round(anomaly_rate * 100, 1),
                'efficiency_score': round(efficiency_score, 1)
            })
        
        # Sort by efficiency score (descending)
        airlines.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        # Determine best/worst
        best = airlines[0] if airlines else None
        worst = airlines[-1] if len(airlines) > 1 else None
        
        # Calculate time difference
        time_diff = 0
        if best and worst and best['avg_duration_min'] and worst['avg_duration_min']:
            time_diff = worst['avg_duration_min'] - best['avg_duration_min']
        
        # Generate insights
        insights = []
        if best:
            insights.append(f"{best['airline']} has the best efficiency on this route (score: {best['efficiency_score']})")
        if time_diff > 5:
            insights.append(f"Time difference between best and worst: {time_diff:.0f} minutes")
        if worst and worst['avg_deviation_nm'] > 10:
            insights.append(f"{worst['airline']} averages {worst['avg_deviation_nm']}nm deviation from optimal route")
        
        return {
            'route': route,
            'airlines': airlines,
            'best_performer': best['airline'] if best else None,
            'worst_performer': worst['airline'] if worst else None,
            'time_difference_min': round(time_diff, 1),
            'insights': insights
        }
    
    def _get_top_routes_efficiency(self, start_ts: int, end_ts: int, limit: int = 10) -> Dict[str, Any]:
        """Get efficiency summary for top routes by traffic volume."""
        
        # Get routes with most traffic
        query = """
            SELECT 
                origin_airport || '-' || destination_airport as route,
                COUNT(*) as flight_count,
                AVG(flight_duration_sec) as avg_duration,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies,
                COUNT(DISTINCT SUBSTR(callsign, 1, 3)) as airline_count
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND origin_airport IS NOT NULL AND destination_airport IS NOT NULL
            AND origin_airport != '' AND destination_airport != ''
            GROUP BY route
            HAVING flight_count >= 5
            ORDER BY flight_count DESC
            LIMIT ?
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, limit))
        
        routes = []
        for row in results:
            route, count, avg_dur, anomalies, airlines = row
            anomaly_rate = (anomalies / count * 100) if count > 0 else 0
            routes.append({
                'route': route,
                'flight_count': count,
                'avg_duration_min': round((avg_dur or 0) / 60, 1),
                'anomaly_rate': round(anomaly_rate, 1),
                'airline_count': airlines
            })
        
        return {
            'summary': 'Top routes by traffic volume',
            'routes': routes,
            'note': 'Select a specific route for detailed airline comparison'
        }
    
    def get_available_routes(self, start_ts: int, end_ts: int, min_flights: int = 5) -> List[str]:
        """Get list of routes with minimum number of flights."""
        query = """
            SELECT origin_airport || '-' || destination_airport as route
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND origin_airport IS NOT NULL AND destination_airport IS NOT NULL
            AND origin_airport != '' AND destination_airport != ''
            GROUP BY route
            HAVING COUNT(*) >= ?
            ORDER BY COUNT(*) DESC
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, min_flights))
        return [row[0] for row in results]
    
    # ============================================================================
    # WEATHER IMPACT ANALYSIS
    # ============================================================================
    
    def get_weather_impact_analysis(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze weather impact on flight operations.
        
        Answers: "Count diversions due to weather" and "Count deviations caused by storms"
        
        Note: This is a proxy analysis using anomaly patterns that correlate with
        weather events. For full weather correlation, integrate with METAR/weather API.
        
        Returns:
            {
                'weather_correlated_anomalies': int,
                'diversions_likely_weather': [{airport, count, dates}],
                'go_arounds_weather_pattern': [{airport, count, peak_hour}],
                'monthly_weather_impact': [{month, diversion_count, go_around_count, deviation_count}],
                'insights': [str]
            }
        """
        # Weather-correlated anomalies are identified by:
        # 1. Diversions (Rule 10)
        # 2. Go-arounds (Rule 6)
        # 3. Large route deviations (Rule 5, 11)
        # 4. Holding patterns (Rule 3)
        
        # Query weather-related diversions (flights that diverted - Rule 10)
        # NOTE: Use COUNT(DISTINCT fm.flight_id) to avoid double-counting flights with multiple anomaly_reports
        query = """
            SELECT 
                fm.destination_airport as intended_dest,
                fm.origin_airport as origin,
                DATE(fm.first_seen_ts, 'unixepoch') as date,
                strftime('%Y-%m', fm.first_seen_ts, 'unixepoch') as month,
                COUNT(DISTINCT fm.flight_id) as count
            FROM flight_metadata fm
            JOIN anomaly_reports ar ON fm.flight_id = ar.flight_id
            WHERE fm.first_seen_ts BETWEEN ? AND ?
            AND ar.matched_rule_ids LIKE '%10%'
            GROUP BY intended_dest, date
            ORDER BY count DESC
        """
        diversion_results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        # Query go-arounds (weather-related at airports)
        # NOTE: Use COUNT(DISTINCT fm.flight_id) to avoid double-counting flights with multiple anomaly_reports
        query = """
            SELECT 
                fm.destination_airport as airport,
                strftime('%H', fm.first_seen_ts, 'unixepoch') as hour,
                strftime('%Y-%m', fm.first_seen_ts, 'unixepoch') as month,
                COUNT(DISTINCT fm.flight_id) as count
            FROM flight_metadata fm
            JOIN anomaly_reports ar ON fm.flight_id = ar.flight_id
            WHERE fm.first_seen_ts BETWEEN ? AND ?
            AND ar.matched_rule_ids LIKE '%6%'
            GROUP BY airport, hour
        """
        go_around_results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        # Query route deviations (potential weather avoidance)
        query = """
            SELECT 
                strftime('%Y-%m', timestamp, 'unixepoch') as month,
                COUNT(DISTINCT flight_id) as deviation_count
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND (matched_rule_ids LIKE '%5%' OR matched_rule_ids LIKE '%11%')
            GROUP BY month
        """
        deviation_results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Process diversions by airport
        diversions_by_airport = defaultdict(lambda: {'count': 0, 'dates': set()})
        for row in diversion_results:
            dest, origin, date, month, count = row
            if dest:
                diversions_by_airport[dest]['count'] += count
                diversions_by_airport[dest]['dates'].add(date)
        
        diversions_likely_weather = [
            {
                'airport': airport,
                'count': data['count'],
                'dates': list(data['dates'])[:5]
            }
            for airport, data in sorted(diversions_by_airport.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        ]
        
        # Process go-arounds by airport
        go_arounds_by_airport = defaultdict(lambda: {'count': 0, 'hours': defaultdict(int)})
        for row in go_around_results:
            airport, hour, month, count = row
            if airport:
                go_arounds_by_airport[airport]['count'] += count
                go_arounds_by_airport[airport]['hours'][int(hour) if hour else 0] += count
        
        go_arounds_weather = []
        for airport, data in sorted(go_arounds_by_airport.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            peak_hour = max(data['hours'].items(), key=lambda x: x[1])[0] if data['hours'] else 0
            go_arounds_weather.append({
                'airport': airport,
                'count': data['count'],
                'peak_hour': peak_hour
            })
        
        # Monthly weather impact summary
        monthly_deviations = {row[0]: row[1] for row in deviation_results if row[0]}
        
        # Query monthly diversions and go-arounds from anomaly_reports
        # Rule 10 = diversion, Rule 6 = go-around
        query = """
            SELECT 
                strftime('%Y-%m', ar.timestamp, 'unixepoch') as month,
                SUM(CASE WHEN ar.matched_rule_ids LIKE '%10%' THEN 1 ELSE 0 END) as diversions,
                SUM(CASE WHEN ar.matched_rule_ids LIKE '%6%' THEN 1 ELSE 0 END) as go_arounds
            FROM anomaly_reports ar
            WHERE ar.timestamp BETWEEN ? AND ?
            GROUP BY month
        """
        monthly_results = self._execute_query('tagged', query, (start_ts, end_ts))
        
        monthly_weather_impact = []
        for row in monthly_results:
            month, diversions, go_arounds = row
            if month:
                monthly_weather_impact.append({
                    'month': month,
                    'diversion_count': diversions or 0,
                    'go_around_count': go_arounds or 0,
                    'deviation_count': monthly_deviations.get(month, 0)
                })
        
        # Sort by month
        monthly_weather_impact.sort(key=lambda x: x['month'])
        
        # Calculate totals
        total_diversions = sum(d['count'] for d in diversions_likely_weather)
        total_go_arounds = sum(g['count'] for g in go_arounds_weather)
        total_deviations = sum(d['deviation_count'] for d in monthly_weather_impact)
        
        # Generate insights
        insights = []
        
        if diversions_likely_weather:
            top_airport = diversions_likely_weather[0]['airport']
            insights.append(f"{top_airport} had the most weather-related diversions ({diversions_likely_weather[0]['count']})")
        
        if go_arounds_weather:
            top_airport = go_arounds_weather[0]['airport']
            peak_hour = go_arounds_weather[0]['peak_hour']
            insights.append(f"Go-arounds peak at {peak_hour}:00 at {top_airport}")
        
        if monthly_weather_impact:
            worst_month = max(monthly_weather_impact, key=lambda x: x['diversion_count'] + x['go_around_count'])
            if worst_month['diversion_count'] + worst_month['go_around_count'] > 0:
                insights.append(f"Worst weather impact month: {worst_month['month']}")
        
        insights.append(f"Total weather-correlated events: {total_diversions + total_go_arounds + total_deviations}")
        
        return {
            'weather_correlated_anomalies': total_diversions + total_go_arounds + total_deviations,
            'diversions_likely_weather': diversions_likely_weather,
            'go_arounds_weather_pattern': go_arounds_weather,
            'monthly_weather_impact': monthly_weather_impact,
            'total_diversions': total_diversions,
            'total_go_arounds': total_go_arounds,
            'total_deviations': total_deviations,
            'insights': insights
        }
    
    def get_weather_by_airport(self, airport: str, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get weather-correlated anomalies for a specific airport.
        
        Returns:
            {
                'airport': str,
                'diversions_to': int,
                'diversions_from': int,
                'go_arounds': int,
                'monthly_breakdown': [{month, diversions, go_arounds}],
                'hourly_distribution': [{hour, count}]
            }
        """
        # Diversions TO this airport
        query = """
            SELECT COUNT(*)
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND destination_airport = ?
            AND is_diversion = 1
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, airport))
        diversions_to = results[0][0] if results else 0
        
        # Diversions FROM this airport
        query = """
            SELECT COUNT(*)
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND origin_airport = ?
            AND is_diversion = 1
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, airport))
        diversions_from = results[0][0] if results else 0
        
        # Go-arounds at this airport
        query = """
            SELECT COUNT(DISTINCT fm.flight_id)
            FROM flight_metadata fm
            JOIN anomaly_reports ar ON fm.flight_id = ar.flight_id
            WHERE fm.first_seen_ts BETWEEN ? AND ?
            AND fm.destination_airport = ?
            AND ar.matched_rule_ids LIKE '%6%'
        """
        results = self._execute_query('tagged', query, (start_ts, end_ts, airport))
        go_arounds = results[0][0] if results else 0
        
        # Monthly breakdown
        query = """
            SELECT 
                strftime('%Y-%m', first_seen_ts, 'unixepoch') as month,
                SUM(CASE WHEN is_diversion = 1 THEN 1 ELSE 0 END) as diversions,
                0 as go_arounds
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (destination_airport = ? OR origin_airport = ?)
            GROUP BY month
            ORDER BY month
        """
        monthly_results = self._execute_query('tagged', query, (start_ts, end_ts, airport, airport))
        
        monthly_breakdown = [
            {'month': row[0], 'diversions': row[1] or 0, 'go_arounds': row[2] or 0}
            for row in monthly_results if row[0]
        ]
        
        # Hourly distribution of go-arounds
        # NOTE: Use COUNT(DISTINCT fm.flight_id) to avoid double-counting flights with multiple anomaly_reports
        query = """
            SELECT 
                CAST(strftime('%H', fm.first_seen_ts, 'unixepoch') AS INTEGER) as hour,
                COUNT(DISTINCT fm.flight_id) as count
            FROM flight_metadata fm
            JOIN anomaly_reports ar ON fm.flight_id = ar.flight_id
            WHERE fm.first_seen_ts BETWEEN ? AND ?
            AND fm.destination_airport = ?
            AND ar.matched_rule_ids LIKE '%6%'
            GROUP BY hour
            ORDER BY hour
        """
        hourly_results = self._execute_query('tagged', query, (start_ts, end_ts, airport))
        
        hourly_distribution = [
            {'hour': row[0], 'count': row[1]}
            for row in hourly_results
        ]
        
        return {
            'airport': airport,
            'diversions_to': diversions_to,
            'diversions_from': diversions_from,
            'go_arounds': go_arounds,
            'monthly_breakdown': monthly_breakdown,
            'hourly_distribution': hourly_distribution
        }
    
    # ============================================================================
    # NEW METHODS FOR DASHBOARD DEMANDS
    # ============================================================================
    
    def get_signal_loss_anomalies(self, start_ts: int, end_ts: int, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Detect unusual signal loss in areas that normally have good reception.
        
        Compares signal loss patterns against a historical baseline to identify
        anomalous signal loss events (e.g., sudden jamming or interference).
        
        Args:
            start_ts: Start of analysis period
            end_ts: End of analysis period
            lookback_days: Days to use for baseline calculation
        
        Returns:
            {
                'anomalous_zones': [{lat, lon, current_losses, baseline_losses, anomaly_score}],
                'total_anomalies': int,
                'insights': [str]
            }
        """
        import math
        from collections import defaultdict
        
        # Calculate baseline period
        baseline_start = start_ts - (lookback_days * 86400)
        baseline_end = start_ts
        
        def get_signal_loss_by_zone(period_start: int, period_end: int) -> Dict[str, Dict]:
            """Get signal loss counts per geographic zone."""
            zone_stats = defaultdict(lambda: {'count': 0, 'flights': set()})
            grid_size = 0.5  # degrees
            
            for table_name in ['anomalies_tracks', 'normal_tracks']:
                # Find gaps > 60 seconds in track data
                query = f"""
                    SELECT flight_id, lat, lon, timestamp,
                           LAG(timestamp) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_ts
                    FROM {table_name}
                    WHERE timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL AND lon IS NOT NULL
                """
                results = self._execute_query('research', query, (period_start, period_end))
                
                for row in results:
                    flight_id, lat, lon, ts, prev_ts = row
                    if prev_ts and ts - prev_ts > 60:  # Gap > 60 seconds
                        # Round to grid cell
                        grid_lat = round(lat / grid_size) * grid_size
                        grid_lon = round(lon / grid_size) * grid_size
                        zone_key = f"{grid_lat:.1f},{grid_lon:.1f}"
                        zone_stats[zone_key]['count'] += 1
                        zone_stats[zone_key]['flights'].add(flight_id)
                        zone_stats[zone_key]['lat'] = grid_lat
                        zone_stats[zone_key]['lon'] = grid_lon
            
            return dict(zone_stats)
        
        # Get baseline and current signal loss
        baseline_losses = get_signal_loss_by_zone(baseline_start, baseline_end)
        current_losses = get_signal_loss_by_zone(start_ts, end_ts)
        
        # Calculate anomaly scores
        anomalous_zones = []
        period_ratio = (end_ts - start_ts) / (baseline_end - baseline_start) if baseline_end > baseline_start else 1
        
        for zone_key, current_data in current_losses.items():
            baseline_data = baseline_losses.get(zone_key, {'count': 0})
            baseline_normalized = baseline_data['count'] * period_ratio
            
            # Anomaly: significant increase from baseline
            if baseline_normalized > 0:
                anomaly_score = (current_data['count'] - baseline_normalized) / baseline_normalized
            else:
                # No baseline - any current loss is anomalous
                anomaly_score = current_data['count'] / 10.0  # Scale factor
            
            if anomaly_score > 0.5:  # >50% increase from baseline
                anomalous_zones.append({
                    'lat': current_data.get('lat', 0),
                    'lon': current_data.get('lon', 0),
                    'current_losses': current_data['count'],
                    'baseline_losses': baseline_data['count'],
                    'affected_flights': len(current_data.get('flights', set())),
                    'anomaly_score': round(anomaly_score, 2)
                })
        
        # Sort by anomaly score
        anomalous_zones.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        # Generate insights
        insights = []
        if anomalous_zones:
            top_zone = anomalous_zones[0]
            insights.append(f"Highest anomaly at {top_zone['lat']:.1f}°N, {top_zone['lon']:.1f}°E with {top_zone['current_losses']} losses")
            insights.append(f"Total anomalous zones detected: {len(anomalous_zones)}")
        else:
            insights.append("No unusual signal loss patterns detected")
        
        return {
            'anomalous_zones': anomalous_zones[:20],
            'total_anomalies': len(anomalous_zones),
            'insights': insights
        }
    
    def get_gps_jamming_temporal(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze GPS jamming patterns by time of day and day of week.
        
        Answers: "When does GPS jamming happen most?"
        
        Returns:
            {
                'by_hour': [{hour, count}],
                'by_day_of_week': [{day, day_name, count}],
                'peak_hours': [int],
                'peak_days': [str],
                'total_events': int
            }
        """
        from datetime import datetime
        
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Query anomaly reports for GPS jamming (Rule 9 or signature-based)
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        total_events = 0
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                # Check for GPS jamming indicators
                has_gps_jamming = False
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    # Rule 9 is GPS jamming, also check for altitude anomalies (Rule 10)
                    if rule_id in [9, 10]:
                        has_gps_jamming = True
                        break
                    # Check rule details for jamming-like signatures
                    details = rule.get('details', {})
                    if 'altitude_spike' in str(details) or 'gps' in str(details).lower():
                        has_gps_jamming = True
                        break
                
                if has_gps_jamming:
                    dt = datetime.fromtimestamp(timestamp)
                    hourly_counts[dt.hour] += 1
                    daily_counts[dt.weekday()] += 1
                    total_events += 1
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Format results
        by_hour = [{'hour': h, 'count': hourly_counts.get(h, 0)} for h in range(24)]
        by_day = [{'day': d, 'day_name': day_names[d], 'count': daily_counts.get(d, 0)} for d in range(7)]
        
        # Find peaks
        peak_hours = sorted(hourly_counts.keys(), key=lambda h: hourly_counts[h], reverse=True)[:3]
        peak_days = sorted(daily_counts.keys(), key=lambda d: daily_counts[d], reverse=True)[:2]
        peak_day_names = [day_names[d] for d in peak_days]
        
        return {
            'by_hour': by_hour,
            'by_day_of_week': by_day,
            'peak_hours': list(peak_hours),
            'peak_days': peak_day_names,
            'total_events': total_events
        }
    
    def get_go_arounds_hourly(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get hourly distribution of go-around events.
        
        Answers: "What time of day are there the most go-arounds?"
        
        Returns:
            [{hour, count, airports: [str]}]
        """
        from datetime import datetime
        
        hourly_stats = defaultdict(lambda: {'count': 0, 'airports': set()})
        
        # Query anomaly reports for go-arounds (Rule 6)
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 6:  # Go-around
                        hour = datetime.fromtimestamp(timestamp).hour
                        hourly_stats[hour]['count'] += 1
                        
                        # Try to get airport from details or summary
                        details = rule.get('details', {})
                        airport = details.get('airport') or report.get('summary', {}).get('destination_airport')
                        if airport:
                            hourly_stats[hour]['airports'].add(airport)
                        
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Format results
        result = []
        for hour in range(24):
            data = hourly_stats.get(hour, {'count': 0, 'airports': set()})
            result.append({
                'hour': hour,
                'count': data['count'],
                'airports': list(data['airports'])[:5]  # Top 5 airports
            })
        
        return result
    
    def get_diversions_seasonal(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get seasonal breakdown of diversions.
        
        Answers: "What time of year has the most diversions?"
        
        Returns:
            {
                'by_season': [{season, count}],
                'by_quarter': [{quarter, count}],
                'by_month': [{month, month_name, count}],
                'peak_season': str,
                'insights': [str]
            }
        """
        from datetime import datetime
        
        season_names = {
            (12, 1, 2): 'Winter',
            (3, 4, 5): 'Spring',
            (6, 7, 8): 'Summer',
            (9, 10, 11): 'Fall'
        }
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_counts = defaultdict(int)
        quarterly_counts = defaultdict(int)
        seasonal_counts = defaultdict(int)
        
        # Query for diversions (Rule 8)
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    if rule.get('id') == 8:  # Diversion
                        dt = datetime.fromtimestamp(timestamp)
                        month = dt.month
                        quarter = (month - 1) // 3 + 1
                        
                        monthly_counts[month] += 1
                        quarterly_counts[quarter] += 1
                        
                        # Determine season
                        for months, season in season_names.items():
                            if month in months:
                                seasonal_counts[season] += 1
                                break
                        
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Format results
        by_month = [{'month': m, 'month_name': month_names[m-1], 'count': monthly_counts.get(m, 0)} 
                    for m in range(1, 13)]
        by_quarter = [{'quarter': f'Q{q}', 'count': quarterly_counts.get(q, 0)} for q in range(1, 5)]
        by_season = [{'season': s, 'count': seasonal_counts.get(s, 0)} 
                     for s in ['Winter', 'Spring', 'Summer', 'Fall']]
        
        # Find peak season
        peak_season = max(seasonal_counts.keys(), key=lambda s: seasonal_counts[s]) if seasonal_counts else 'Unknown'
        
        # Generate insights
        insights = []
        total = sum(monthly_counts.values())
        if total > 0:
            insights.append(f"Peak diversion season: {peak_season}")
            if monthly_counts:
                peak_month = max(monthly_counts.keys(), key=lambda m: monthly_counts[m])
                insights.append(f"Worst month: {month_names[peak_month-1]} ({monthly_counts[peak_month]} diversions)")
        
        return {
            'by_season': by_season,
            'by_quarter': by_quarter,
            'by_month': by_month,
            'peak_season': peak_season,
            'total_diversions': total,
            'insights': insights
        }
    
    def get_daily_incident_clusters(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze daily clustering of incidents to detect unusual days.
        
        Answers: "Were there multiple incidents in one day? Were they in the same area?"
        
        Returns:
            {
                'high_incident_days': [{date, total_incidents, incidents_by_type, geographic_cluster}],
                'average_daily_incidents': float,
                'max_incidents_day': {date, count},
                'insights': [str]
            }
        """
        from datetime import datetime
        import math
        
        daily_incidents = defaultdict(lambda: {
            'emergency': [],
            'near_miss': [],
            'go_around': [],
            'diversion': [],
            'locations': []
        })
        
        # Query anomaly reports
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                flight_id = report.get('summary', {}).get('flight_id', '')
                
                date_key = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                
                # Get location from summary
                summary = report.get('summary', {})
                if 'first_lat' in summary and 'first_lon' in summary:
                    daily_incidents[date_key]['locations'].append({
                        'lat': summary['first_lat'],
                        'lon': summary['first_lon']
                    })
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    if rule_id == 1:  # Emergency
                        daily_incidents[date_key]['emergency'].append(flight_id)
                    elif rule_id == 4:  # Near-miss
                        daily_incidents[date_key]['near_miss'].append(flight_id)
                    elif rule_id == 6:  # Go-around
                        daily_incidents[date_key]['go_around'].append(flight_id)
                    elif rule_id == 8:  # Diversion
                        daily_incidents[date_key]['diversion'].append(flight_id)
                        
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        def calculate_geographic_spread(locations):
            """Calculate geographic spread of incidents."""
            if len(locations) < 2:
                return 0
            
            lats = [loc['lat'] for loc in locations if loc.get('lat')]
            lons = [loc['lon'] for loc in locations if loc.get('lon')]
            
            if not lats or not lons:
                return 0
            
            lat_spread = max(lats) - min(lats)
            lon_spread = max(lons) - min(lons)
            
            # Return spread in degrees (rough geographic extent)
            return round(math.sqrt(lat_spread**2 + lon_spread**2), 2)
        
        # Analyze each day
        high_incident_days = []
        total_incidents = 0
        
        for date_key, data in daily_incidents.items():
            day_total = (len(data['emergency']) + len(data['near_miss']) + 
                        len(data['go_around']) + len(data['diversion']))
            total_incidents += day_total
            
            if day_total >= 3:  # Threshold for "high incident" day
                spread = calculate_geographic_spread(data['locations'])
                is_clustered = spread < 2.0  # Incidents within ~200km
                
                high_incident_days.append({
                    'date': date_key,
                    'total_incidents': day_total,
                    'emergency_count': len(data['emergency']),
                    'near_miss_count': len(data['near_miss']),
                    'go_around_count': len(data['go_around']),
                    'diversion_count': len(data['diversion']),
                    'geographic_spread_deg': spread,
                    'geographically_clustered': is_clustered
                })
        
        # Sort by total incidents
        high_incident_days.sort(key=lambda x: x['total_incidents'], reverse=True)
        
        # Calculate averages
        num_days = len(daily_incidents) if daily_incidents else 1
        avg_incidents = total_incidents / num_days
        
        # Find max day
        max_day = high_incident_days[0] if high_incident_days else {'date': 'N/A', 'total_incidents': 0}
        
        # Generate insights
        insights = []
        if high_incident_days:
            insights.append(f"Found {len(high_incident_days)} high-incident days (3+ events)")
            clustered_days = sum(1 for d in high_incident_days if d['geographically_clustered'])
            if clustered_days > 0:
                insights.append(f"{clustered_days} days had geographically clustered incidents")
        else:
            insights.append("No unusually high incident days detected")
        
        return {
            'high_incident_days': high_incident_days[:20],
            'average_daily_incidents': round(avg_incidents, 2),
            'max_incidents_day': {
                'date': max_day['date'],
                'count': max_day['total_incidents']
            },
            'total_days_analyzed': num_days,
            'insights': insights
        }

    def get_airline_safety_scorecard(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Build an airline safety scorecard based on various safety metrics.
        
        Aggregates data from emergency codes, near-misses, go-arounds, and diversions
        to create a comprehensive safety profile for each airline.
        
        Returns:
            {
                'scorecards': [{
                    'airline': str,
                    'airline_name': str,
                    'total_flights': int,
                    'emergencies': int,
                    'near_miss': int,
                    'go_arounds': int,
                    'diversions': int,
                    'safety_score': int (0-100),
                    'safety_grade': str (A-F),
                    'trend': str (improving/declining/stable),
                    'issues': [str]
                }],
                'summary': {
                    'total_airlines': int,
                    'average_score': float,
                    'best_performer': {airline, score},
                    'needs_attention': [{airline, issues}]
                }
            }
        """
        cache_key = _get_cache_key("get_airline_safety_scorecard", start_ts, end_ts)
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Airline name lookup - comprehensive mapping including Israeli carriers
        AIRLINE_NAMES = {
            # Israeli carriers (priority)
            'ELY': 'El Al', 'ISR': 'Israir', 'AIZ': 'Arkia', 'BLB': 'Blue Bird Airways',
            # Major Middle Eastern carriers
            'UAE': 'Emirates', 'QTR': 'Qatar Airways', 'ETH': 'Ethiopian', 'GFA': 'Gulf Air',
            'FDB': 'flydubai', 'RJA': 'Royal Jordanian', 'MSR': 'EgyptAir', 'MEA': 'Middle East Airlines',
            'SVA': 'Saudia', 'IAW': 'Iraqi Airways', 'SYR': 'Syrian Air', 'RAM': 'Royal Air Maroc',
            'ABY': 'Air Arabia',
            # European carriers
            'THY': 'Turkish Airlines', 'PGT': 'Pegasus',
            'RYR': 'Ryanair', 'WZZ': 'Wizz Air', 'WMT': 'Wizz Air Malta', 'W4U': 'Wizz Air Malta',
            'EZY': 'easyJet',
            'BAW': 'British Airways', 'DLH': 'Lufthansa', 'AFR': 'Air France',
            'KLM': 'KLM', 'SWR': 'Swiss', 'AUA': 'Austrian', 'TAP': 'TAP Portugal',
            'AZA': 'Alitalia/ITA', 'ITA': 'ITA Airways', 'IBE': 'Iberia', 
            'VLG': 'Vueling', 'AEE': 'Aegean', 'TUI': 'TUI',
            # Additional/Regional carriers
            'LOT': 'LOT Polish', 'AZG': 'Silk Way Airlines',
            'FDX': 'FedEx', 'UPS': 'UPS Airlines', 'CAL': 'Cargo Air Lines',
            'EWG': 'Eurowings', 'AXY': 'Axis Airways', 'LX': 'Swiss', 'OS': 'Austrian',
            # TLV-specific codes (may be regional or charter)
            'KNE': 'KNE', 'MSC': 'MSC', 'FAD': 'FAD', 'HFA': 'HFA',
            'XD': 'Unknown Carrier'  # Placeholder for unknown codes
        }
        
        # Priority airlines for TLV - 20 key carriers
        PRIORITY_AIRLINES = {
            'RJA', 'MSR', 'ELY', 'MEA', 'KNE', 'MSC', 'SVA', 'FDB', 'AIZ', 'ISR',
            'FAD', 'QTR', 'THY', 'UAE', 'ABY', 'WZZ', 'HFA', 'ETH', 'WMT', 'W4U', 'DLH'
        }
        
        airline_stats = defaultdict(lambda: {
            'total_flights': 0,
            'emergencies': 0,
            'near_miss': 0,
            'go_arounds': 0,
            'diversions': 0
        })
        
        # 1. Count total flights per airline from flight_metadata
        flight_query = """
            SELECT callsign, COUNT(DISTINCT flight_id) as flight_count
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ? 
            AND callsign IS NOT NULL AND callsign != ''
            GROUP BY SUBSTR(callsign, 1, 3)
        """
        flight_results = self._execute_query('research', flight_query, (start_ts, end_ts))
        
        for row in flight_results:
            callsign, count = row
            if callsign:
                airline = callsign[:3].upper()
                airline_stats[airline]['total_flights'] += count
        
        # 2. Query anomaly reports for safety events
        anomaly_query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        anomaly_results = self._execute_query('research', anomaly_query, (start_ts, end_ts))
        
        for row in anomaly_results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                # Get callsign from summary
                callsign = report.get('summary', {}).get('callsign', '')
                if not callsign:
                    continue
                    
                airline = callsign[:3].upper()
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    if rule_id == 1:  # Emergency code
                        airline_stats[airline]['emergencies'] += 1
                    elif rule_id == 4:  # Near-miss/proximity
                        airline_stats[airline]['near_miss'] += 1
                    elif rule_id == 6:  # Go-around
                        airline_stats[airline]['go_arounds'] += 1
                    elif rule_id == 8:  # Diversion
                        airline_stats[airline]['diversions'] += 1
                        
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        def calculate_safety_score(data: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate safety score and grade from metrics."""
            score = 100  # Start with perfect score
            issues = []
            
            total_flights = data['total_flights'] or 1  # Avoid div by zero
            
            # Emergency codes (highest weight) - 40 points max deduction
            emergencies = data['emergencies']
            if emergencies > 0:
                emergency_rate = emergencies / total_flights * 1000  # per 1000 flights
                deduction = min(40, int(emergency_rate * 20))
                score -= deduction
                issues.append(f"{emergencies} emergencies ({emergency_rate:.1f}/1000 flights)")
            
            # Near-miss events - 30 points max deduction
            near_miss = data['near_miss']
            if near_miss > 0:
                nm_rate = near_miss / total_flights * 1000
                deduction = min(30, int(nm_rate * 15))
                score -= deduction
                issues.append(f"{near_miss} near-miss events")
            
            # Go-arounds - 20 points max deduction
            go_arounds = data['go_arounds']
            if go_arounds > 0:
                ga_rate = go_arounds / total_flights * 1000
                deduction = min(20, int(ga_rate * 5))
                score -= deduction
                if deduction >= 10:
                    issues.append(f"{go_arounds} go-arounds ({ga_rate:.1f}/1000 flights)")
            
            # Diversions - 10 points max deduction
            diversions = data['diversions']
            if diversions > 0:
                div_rate = diversions / total_flights * 1000
                deduction = min(10, int(div_rate * 2))
                score -= deduction
                if deduction >= 5:
                    issues.append(f"{diversions} diversions")
            
            # Ensure score is in bounds
            score = max(0, min(100, score))
            
            # Assign grade
            if score >= 90:
                grade = 'A'
            elif score >= 80:
                grade = 'B'
            elif score >= 70:
                grade = 'C'
            elif score >= 60:
                grade = 'D'
            else:
                grade = 'F'
            
            return {
                'safety_score': score,
                'safety_grade': grade,
                'issues': issues if issues else ['No significant safety issues']
            }
        
        # Build scorecards for airlines with sufficient data
        scorecards = []
        
        # Calculate total flights across all airlines for market share
        total_all_flights = sum(d['total_flights'] for d in airline_stats.values() if d['total_flights'] >= 10)
        
        for airline, data in airline_stats.items():
            # Only include airlines with at least 10 flights
            if data['total_flights'] < 10:
                continue
            
            score_data = calculate_safety_score(data)
            
            # Calculate statistical confidence based on sample size
            # Higher flight counts = more reliable data
            flight_count = data['total_flights']
            if flight_count >= 500:
                confidence = 'high'
                confidence_factor = 1.0
            elif flight_count >= 100:
                confidence = 'medium'
                confidence_factor = 0.9
            elif flight_count >= 50:
                confidence = 'low'
                confidence_factor = 0.8
            else:
                confidence = 'very_low'
                confidence_factor = 0.6
            
            # Calculate weighted score (accounts for statistical reliability)
            # Airlines with few flights get penalized even with perfect scores
            raw_score = score_data['safety_score']
            weighted_score = int(raw_score * confidence_factor)
            
            # Calculate market share
            market_share = round((flight_count / total_all_flights * 100), 1) if total_all_flights > 0 else 0
            
            # Is this a priority airline for TLV?
            is_priority = airline in PRIORITY_AIRLINES
            
            scorecards.append({
                'airline': airline,
                'airline_name': AIRLINE_NAMES.get(airline, airline),
                'total_flights': data['total_flights'],
                'emergencies': data['emergencies'],
                'near_miss': data['near_miss'],
                'go_arounds': data['go_arounds'],
                'diversions': data['diversions'],
                'safety_score': score_data['safety_score'],
                'weighted_score': weighted_score,
                'confidence': confidence,
                'market_share': market_share,
                'is_priority': is_priority,
                'safety_grade': score_data['safety_grade'],
                'trend': 'stable',  # Could be expanded with historical comparison
                'issues': score_data['issues']
            })
        
        # Sort by: 1) Priority airlines first, 2) Flight volume (market relevance), 3) Safety score
        # This ensures major carriers appear at top for relevance
        scorecards.sort(key=lambda x: (
            -int(x['is_priority']),  # Priority airlines first
            -x['total_flights'],      # Then by flight volume
            -x['weighted_score']      # Then by weighted safety score
        ))
        
        # Build summary
        total_airlines = len(scorecards)
        avg_score = sum(s['safety_score'] for s in scorecards) / total_airlines if total_airlines > 0 else 0
        
        # Best performer should be from high-confidence airlines only
        high_confidence = [s for s in scorecards if s['confidence'] in ('high', 'medium') and s['total_flights'] >= 100]
        best_performer = max(high_confidence, key=lambda x: x['weighted_score']) if high_confidence else (scorecards[0] if scorecards else None)
        
        needs_attention = [s for s in scorecards if s['safety_grade'] in ('D', 'F') and s['total_flights'] >= 50]
        
        result = {
            'scorecards': scorecards[:40],  # Top 40 airlines
            'summary': {
                'total_airlines': total_airlines,
                'average_score': round(avg_score, 1),
                'best_performer': {
                    'airline': best_performer['airline'],
                    'airline_name': best_performer['airline_name'],
                    'score': best_performer['safety_score'],
                    'flights': best_performer['total_flights']
                } if best_performer else None,
                'needs_attention': [
                    {'airline': s['airline'], 'airline_name': s['airline_name'], 'issues': s['issues'][:3], 'flights': s['total_flights']}
                    for s in needs_attention[:5]
                ]
            }
        }
        
        _set_cached(cache_key, result)
        return result

