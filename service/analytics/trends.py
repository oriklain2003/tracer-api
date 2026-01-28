"""
Trend analysis and comparative analytics for Level 2 insights.

Provides:
- Airline efficiency comparisons
- Holding pattern analysis
- Alternate airport behavior
- Seasonal trends
"""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path

# Closed airports that should not be used for attribution
CLOSED_AIRPORTS = {'LLSD'}  # Sde Dov closed in 2019

# Load airport data from config
def _load_airports() -> List[Dict[str, Any]]:
    """Load airport data from rule_config.json, excluding closed airports."""
    config_path = Path(__file__).parent.parent.parent / 'rules' / 'rule_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            airports = config.get('airports', [])
            # Filter out closed airports
            return [apt for apt in airports if apt.get('code') not in CLOSED_AIRPORTS]
    return []

AIRPORTS = _load_airports()


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in nautical miles between two coordinates."""
    R = 3440.065  # Earth radius in nautical miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def _find_nearest_airport(lat: float, lon: float) -> Tuple[Optional[str], float]:
    """Find nearest airport to given coordinates.
    
    Returns:
        (airport_code, distance_nm) or (None, inf) if no airports loaded
    """
    best_code = None
    best_distance = float('inf')
    
    for airport in AIRPORTS:
        distance = _haversine_nm(lat, lon, airport['lat'], airport['lon'])
        if distance < best_distance:
            best_distance = distance
            best_code = airport['code']
    
    return best_code, best_distance


# Wide-body vs Narrow-body aircraft classification
# Wide-body: typically 2 aisles, >200 seats, longer range
# Narrow-body: single aisle, <200 seats
WIDE_BODY_AIRCRAFT = {
    # Boeing wide-body
    'B744', 'B747', 'B748', 'B74S',  # 747 variants
    'B762', 'B763', 'B764', 'B767',  # 767 variants
    'B772', 'B773', 'B77L', 'B77W', 'B779', 'B778', 'B77F',  # 777 variants
    'B788', 'B789', 'B78X', 'B787',  # 787 variants
    # Airbus wide-body
    'A332', 'A333', 'A339', 'A330', 'A338', 'A33F',  # A330 variants
    'A342', 'A343', 'A345', 'A346', 'A340',  # A340 variants
    'A359', 'A35K', 'A350', 'A35X',  # A350 variants
    'A388', 'A380',  # A380 variants
    # Other wide-body
    'A306', 'A310', 'A3ST',  # Legacy Airbus
    'DC10', 'MD11', 'MD1F',  # McDonnell Douglas
    'IL96', 'IL86',  # Ilyushin
    'B78K', 'B764', 'B76F',  # Freighter variants
}

NARROW_BODY_AIRCRAFT = {
    # Boeing narrow-body
    'B731', 'B732', 'B733', 'B734', 'B735', 'B736', 'B737',  # 737 classic
    'B738', 'B739', 'B38M', 'B39M', 'B37M', 'B3XM',  # 737 NG/MAX
    'B752', 'B753', 'B757',  # 757
    # Airbus narrow-body
    'A318', 'A319', 'A320', 'A321', 'A19N', 'A20N', 'A21N',  # A320 family
    # Regional/Narrow
    'E170', 'E175', 'E190', 'E195', 'E75L', 'E75S',  # Embraer E-Jet
    'CRJ2', 'CRJ7', 'CRJ9', 'CRJX',  # Bombardier CRJ
    'DH8A', 'DH8B', 'DH8C', 'DH8D', 'DHC8', 'Q400',  # Dash 8
    'AT43', 'AT72', 'AT76', 'ATR',  # ATR
    'A220', 'BCS1', 'BCS3',  # Airbus A220/Bombardier CSeries
    'MD80', 'MD81', 'MD82', 'MD83', 'MD87', 'MD88', 'MD90',  # MD-80/90
    'B712', 'B717',  # Boeing 717
    'F70', 'F100',  # Fokker
}


def _classify_aircraft_body_type(aircraft_type: Optional[str]) -> str:
    """
    Classify aircraft as wide-body, narrow-body, or unknown.
    
    Args:
        aircraft_type: ICAO aircraft type code (e.g., 'B738', 'A350')
    
    Returns:
        'wide_body', 'narrow_body', or 'unknown'
    """
    if not aircraft_type:
        return 'unknown'
    
    # Normalize - uppercase and take first 4 chars
    normalized = aircraft_type.upper().strip()[:4]
    
    # Check wide-body first (more specific)
    if normalized in WIDE_BODY_AIRCRAFT:
        return 'wide_body'
    
    # Check 3-character codes
    if normalized[:3] in {'B74', 'B76', 'B77', 'B78', 'A33', 'A34', 'A35', 'A38'}:
        return 'wide_body'
    
    if normalized in NARROW_BODY_AIRCRAFT:
        return 'narrow_body'
    
    # Check 3-character codes for narrow-body
    if normalized[:3] in {'B73', 'B75', 'A31', 'A32', 'E17', 'E19', 'CRJ', 'DH8', 'AT7', 'MD8', 'MD9'}:
        return 'narrow_body'
    
    return 'unknown'


class TrendsAnalyzer:
    """Analyzer for operational trends and comparative insights."""
    
    def __init__(self, db_paths: Dict[str, Path]):
        self.db_paths = db_paths
    
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
    
    def get_airline_efficiency(self, start_ts: int = 0, end_ts: int = 0, route: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Compare airline efficiency by analyzing holding patterns (360-degree turns) per airline.
        
        Holding time is extracted from Rule 3 events in anomaly reports.
        Flight duration is calculated from track timestamps.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            route: Optional route filter (e.g., "LLBG-EGLL")
        
        Returns:
            [{airline, avg_flight_time_min, avg_holding_time_min, sample_count}]
        """
        # If no timestamps provided, use last 30 days
        if not start_ts or not end_ts:
            import time
            end_ts = int(time.time())
            start_ts = end_ts - (30 * 24 * 60 * 60)
        
        # First, get flight durations by airline from track data
        airline_flight_durations = self._get_flight_durations_by_airline(start_ts, end_ts)
        
        # Query anomaly reports to get holding patterns per airline
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Aggregate holding time by airline (extracted from callsign prefix)
        airline_stats = defaultdict(lambda: {'total_holding_s': 0, 'flights': set(), 'holding_events': 0})
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                
                # Get callsign from summary
                callsign = report.get('summary', {}).get('callsign', '')
                if not callsign or len(callsign) < 2:
                    continue
                    
                # Extract airline code (first 2-3 letters)
                airline = ''.join(c for c in callsign[:3] if c.isalpha()).upper()
                if not airline:
                    continue
                
                flight_id = report.get('summary', {}).get('flight_id', '')
                
                # Look for Rule 3 (holding patterns / 360 turns)
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                for rule in matched_rules:
                    if rule.get('id') == 3:
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            duration_s = event.get('duration_s', 0)
                            if duration_s > 0:
                                airline_stats[airline]['total_holding_s'] += duration_s
                                airline_stats[airline]['flights'].add(flight_id)
                                airline_stats[airline]['holding_events'] += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Convert to output format
        result = []
        for airline, data in airline_stats.items():
            if data['holding_events'] > 0:
                avg_holding_min = (data['total_holding_s'] / data['holding_events']) / 60
                # Get avg flight time from pre-calculated durations
                avg_flight_time = airline_flight_durations.get(airline, {}).get('avg_duration_min', 0)
                result.append({
                    'airline': airline,
                    'avg_flight_time_min': round(avg_flight_time, 1),
                    'avg_holding_time_min': round(avg_holding_min, 1),
                    'sample_count': len(data['flights']),
                    'total_holding_events': data['holding_events']
                })
        
        # Sort by sample count descending
        result.sort(key=lambda x: x['sample_count'], reverse=True)
        
        # Return top 10 airlines
        return result[:10] if result else []
    
    def _get_flight_durations_by_airline(self, start_ts: int, end_ts: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate average flight duration by airline from track data.
        
        Returns:
            {airline_code: {'avg_duration_min': float, 'flight_count': int}}
        """
        airline_durations = defaultdict(lambda: {'total_duration_s': 0, 'flight_count': 0})
        
        # Query flight durations from track tables
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT callsign, 
                       MIN(timestamp) as start_time, 
                       MAX(timestamp) as end_time
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                  AND callsign IS NOT NULL
                  AND callsign != ''
                GROUP BY flight_id
                HAVING (MAX(timestamp) - MIN(timestamp)) > 300  -- At least 5 min flight
                   AND (MAX(timestamp) - MIN(timestamp)) < 86400  -- Less than 24 hours
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            for row in results:
                callsign, start_time, end_time = row
                if not callsign or len(callsign) < 2:
                    continue
                
                # Extract airline code (first 2-3 letters)
                airline = ''.join(c for c in callsign[:3] if c.isalpha()).upper()
                if not airline:
                    continue
                
                duration_s = end_time - start_time
                airline_durations[airline]['total_duration_s'] += duration_s
                airline_durations[airline]['flight_count'] += 1
        
        # Calculate averages
        result = {}
        for airline, data in airline_durations.items():
            if data['flight_count'] > 0:
                avg_duration_min = (data['total_duration_s'] / data['flight_count']) / 60
                result[airline] = {
                    'avg_duration_min': avg_duration_min,
                    'flight_count': data['flight_count']
                }
        
        return result
    
    def get_holding_pattern_analysis(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze holding patterns and estimate costs.
        
        Holding patterns are detected from Rule 3 (360-degree turns) in anomaly reports.
        Each holding event has a duration_s field that we aggregate.
        Events are attributed to nearest airport based on event coordinates.
        
        Returns:
            {
                total_time_hours: float,
                estimated_fuel_cost_usd: float,
                peak_hours: [int],
                events_by_airport: {airport_code: count}
            }
        """
        from datetime import datetime
        
        # Query for holding pattern detections (Rule 3 - abrupt turns/360s)
        # Query research.db which has the anomaly reports
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        hour_distribution = defaultdict(int)
        airport_events = defaultdict(int)
        
        # De-duplicate overlapping holding pattern detections
        # Rule 3 can detect overlapping patterns from different starting points
        # Key: (flight_id, event_ts rounded to 60s) -> max duration
        # This ensures we count each distinct holding event only once per flight
        flight_event_map = {}  # {(flight_id, rounded_ts): duration_s}
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                flight_id = report.get('summary', {}).get('flight_id', '')
                
                for rule in matched_rules:
                    # Look for holding patterns (360-degree turns)
                    if rule.get('id') == 3:
                        events = rule.get('details', {}).get('events', [])
                        for event in events:
                            # Only count actual holding patterns, NOT simple single-point turns
                            duration_s = event.get('duration_s')
                            
                            # Skip events without duration_s (simple turn detections)
                            if duration_s is None:
                                continue
                            
                            # Validate duration is reasonable (between 1 minute and 1 hour)
                            if duration_s <= 60 or duration_s > 3600:
                                continue
                            
                            # Get event timestamp and round to 60s for de-duplication
                            event_ts = event.get('timestamp', timestamp)
                            rounded_ts = (event_ts // 60) * 60  # Round to nearest minute
                            
                            # De-duplicate: keep the longest duration for each (flight, time_bucket)
                            key = (flight_id, rounded_ts)
                            if key not in flight_event_map or duration_s > flight_event_map[key]:
                                flight_event_map[key] = duration_s
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Now calculate totals from de-duplicated events
        total_holding_minutes = 0
        flight_holding_events = []  # [(flight_id, timestamp, duration_s)]
        
        for (flight_id, rounded_ts), duration_s in flight_event_map.items():
            total_holding_minutes += duration_s / 60
            
            # Track by hour
            hour = datetime.fromtimestamp(rounded_ts).hour
            hour_distribution[hour] += 1
            
            flight_holding_events.append((flight_id, rounded_ts, duration_s))
        
        # Get airport attribution for holding events by looking up track positions
        if flight_holding_events:
            airport_events = self._attribute_events_to_airports(flight_holding_events, start_ts, end_ts)
        
        # Estimate fuel cost: ~$50/min for commercial jets (approx $3000/hr)
        # Previous value of 3 was too low ($180/hr)
        fuel_cost = total_holding_minutes * 50
        
        # Find peak hours
        peak_hours = sorted(hour_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'total_time_hours': round(total_holding_minutes / 60, 2),
            'estimated_fuel_cost_usd': int(fuel_cost),
            'peak_hours': [h for h, _ in peak_hours],
            'events_by_airport': dict(airport_events)
        }
    
    def _attribute_events_to_airports(self, events: List[Tuple[str, int, int]], 
                                       start_ts: int, end_ts: int) -> Dict[str, int]:
        """
        Attribute holding events to nearest airports based on track positions.
        
        Args:
            events: List of (flight_id, event_timestamp, duration_s)
            start_ts, end_ts: Time range for track queries
            
        Returns:
            {airport_code: event_count}
        """
        airport_counts = defaultdict(int)
        
        # Get unique flight IDs
        flight_ids = list(set(e[0] for e in events if e[0]))
        if not flight_ids:
            return airport_counts
        
        # Build a mapping of flight_id -> [(timestamp, lat, lon)]
        flight_positions = defaultdict(list)
        
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            # Query in batches to avoid too many parameters
            for i in range(0, len(flight_ids), 50):
                batch = flight_ids[i:i+50]
                placeholders = ','.join(['?' for _ in batch])
                query = f"""
                    SELECT flight_id, timestamp, lat, lon
                    FROM {table_name}
                    WHERE flight_id IN ({placeholders})
                      AND timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL AND lon IS NOT NULL
                    ORDER BY flight_id, timestamp
                """
                params = tuple(batch) + (start_ts, end_ts)
                results = self._execute_query('research', query, params)
                
                for row in results:
                    fid, ts, lat, lon = row
                    flight_positions[fid].append((ts, lat, lon))
        
        # For each event, find the nearest airport based on position at event time
        for flight_id, event_ts, _ in events:
            if not flight_id or flight_id not in flight_positions:
                continue
            
            positions = flight_positions[flight_id]
            if not positions:
                continue
            
            # Find position closest to event timestamp
            closest_pos = min(positions, key=lambda p: abs(p[0] - event_ts))
            _, lat, lon = closest_pos
            
            # Find nearest airport
            airport_code, distance = _find_nearest_airport(lat, lon)
            if airport_code and distance < 100:  # Within 100nm of an airport
                airport_counts[airport_code] += 1
        
        return airport_counts
    
    def get_alternate_airports(self, airport: str, event_date: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Analyze where flights divert when primary airport is unavailable.
        
        Queries Rule 8 (diversion) matches from anomaly reports to find:
        - Which airports flights diverted TO when they couldn't land at the primary
        - What aircraft types were involved
        
        Args:
            airport: Primary airport code (e.g., "LLBG") - flights planned for this airport
            event_date: Optional timestamp to filter by date
            
        Returns:
            [{alternate_airport, count, aircraft_types: [...]}]
        """
        import time
        
        # Default time range: last 90 days if no date specified
        if event_date:
            start_ts = event_date - (24 * 60 * 60)  # One day before
            end_ts = event_date + (24 * 60 * 60)    # One day after
        else:
            end_ts = int(time.time())
            start_ts = end_ts - (90 * 24 * 60 * 60)  # Last 90 days
        
        # Query anomaly reports for Rule 8 (diversion) matches
        query = """
            SELECT full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Aggregate by alternate airport
        alternate_stats = defaultdict(lambda: {'count': 0, 'aircraft_types': set()})
        
        for row in results:
            report_json = row[0]
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                callsign = report.get('summary', {}).get('callsign', '')
                
                for rule in matched_rules:
                    if rule.get('id') == 8:  # Diversion rule
                        details = rule.get('details', {})
                        planned = details.get('planned', '')
                        actual = details.get('actual', '')
                        
                        # Filter by primary airport if specified
                        if airport and planned.upper() != airport.upper():
                            continue
                        
                        # Skip if landed at planned destination (no diversion)
                        if not actual or actual == planned:
                            continue
                        
                        alternate_stats[actual]['count'] += 1
                        
                        # Try to extract aircraft type from callsign
                        # Common patterns: first 2-3 chars are airline, rest is flight number
                        if callsign:
                            # Look for aircraft type in the report metadata
                            aircraft_type = self._extract_aircraft_type(report)
                            if aircraft_type:
                                alternate_stats[actual]['aircraft_types'].add(aircraft_type)
                                
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Convert to output format
        result = []
        for alt_airport, data in alternate_stats.items():
            result.append({
                'alternate_airport': alt_airport,
                'count': data['count'],
                'aircraft_types': list(data['aircraft_types'])[:5]  # Limit to 5 types
            })
        
        # Sort by count descending
        result.sort(key=lambda x: x['count'], reverse=True)
        
        return result[:10] if result else []
    
    def _extract_aircraft_type(self, report: Dict[str, Any]) -> Optional[str]:
        """
        Extract aircraft type from anomaly report.
        
        Looks for aircraft type in various report fields.
        """
        # Try to get from metadata if available
        metadata = report.get('metadata', {})
        if metadata:
            aircraft = metadata.get('aircraft_type') or metadata.get('aircraft')
            if aircraft:
                return aircraft
        
        # Try to infer from callsign pattern (less reliable)
        callsign = report.get('summary', {}).get('callsign', '')
        if callsign:
            # Some callsigns include aircraft type hints
            # This is a simplified heuristic
            airline_prefixes = {
                'ELY': 'B738',   # El Al typically uses 737/787
                'THY': 'A321',   # Turkish Airlines
                'MEA': 'A320',   # Middle East Airlines
                'UAE': 'B77W',   # Emirates
                'RJA': 'E190',   # Royal Jordanian
            }
            prefix = callsign[:3].upper()
            if prefix in airline_prefixes:
                return airline_prefixes[prefix]
        
        return None
    
    def get_monthly_trends(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get monthly aggregated trends for flights and anomalies.
        
        Returns:
            [{month, total_flights, anomalies, safety_events, busiest_hour}]
        """
        from datetime import datetime
        
        monthly_data = {}
        
        # Aggregate flights by month from research.db tables
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT 
                    strftime('%Y-%m', datetime(timestamp, 'unixepoch')) as month,
                    COUNT(DISTINCT flight_id) as flight_count,
                    strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                    COUNT(*) as hour_count
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY month, hour
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            for row in results:
                month, flight_count, hour, hour_count = row
                if month not in monthly_data:
                    monthly_data[month] = {
                        'month': month,
                        'total_flights': 0,
                        'anomalies': 0,
                        'safety_events': 0,
                        'hour_counts': {}
                    }
                monthly_data[month]['total_flights'] += flight_count or 0
                monthly_data[month]['hour_counts'][hour] = monthly_data[month]['hour_counts'].get(hour, 0) + (hour_count or 0)
        
        # Convert to list and find busiest hour per month
        result = []
        for month, data in sorted(monthly_data.items()):
            busiest_hour = max(data['hour_counts'].items(), key=lambda x: x[1])[0] if data['hour_counts'] else 0
            result.append({
                'month': month,
                'total_flights': data['total_flights'],
                'anomalies': data['anomalies'],
                'safety_events': data['safety_events'],
                'busiest_hour': int(busiest_hour)
            })
        
        return result
    
    def get_peak_hours_analysis(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze peak traffic hours and their correlation with safety events.
        
        Traffic is counted as DISTINCT FLIGHTS per hour (not track points).
        Safety events are counted as DISTINCT ANOMALY FLIGHTS per hour.
        
        Values are shown as DAILY AVERAGES for easier interpretation.
        
        The correlation score (0-100%) indicates how strongly traffic volume
        and safety events are related across the 24-hour cycle:
        - High (>50%): More flights = more safety events (expected pattern)
        - Moderate (20-50%): Some relationship exists
        - Low (<20%): Safety events occur regardless of traffic volume
        
        Returns:
            {
                'peak_traffic_hours': [hours],
                'peak_safety_hours': [hours],
                'correlation_score': float,
                'hourly_data': [{hour, traffic, safety_events, ...}]
            }
        """
        from collections import defaultdict
        from datetime import datetime
        
        # Calculate total days in period FIRST - use consistent value for all hours
        # This ensures sum(hourly_avg) * num_days == total_flights
        num_days = max(1, (end_ts - start_ts) / 86400)
        
        # Store raw totals per hour (not averages yet)
        traffic_totals_by_hour = defaultdict(int)
        anomaly_totals_by_hour = defaultdict(int)
        normal_totals_by_hour = defaultdict(int)
        
        # Use flight_metadata table which has ALL flights with first_seen_ts
        # JOIN with anomaly_reports to determine if flight is anomaly
        # NOTE: Use MAX(ar.is_anomaly) to ensure flights with ANY is_anomaly=1 record
        # are correctly classified as anomalies (handles flights with multiple records)
        query = """
            SELECT 
                strftime('%H', fl.first_seen_ts, 'unixepoch') AS hour_of_day,
                SUM(CASE WHEN fl.is_anomaly = 1 THEN 1 ELSE 0 END) AS total_anomalies,
                SUM(CASE WHEN fl.is_anomaly != 1 OR fl.is_anomaly IS NULL THEN 1 ELSE 0 END) AS total_normal,
                COUNT(*) AS total_flights
            FROM (
                SELECT fm.flight_id, MAX(ar.is_anomaly) AS is_anomaly, fm.first_seen_ts
                FROM flight_metadata fm
                LEFT JOIN anomaly_reports ar ON fm.flight_id = ar.flight_id
                WHERE fm.first_seen_ts BETWEEN ? AND ?
                GROUP BY fm.flight_id
            ) fl
            GROUP BY hour_of_day
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        total_anomalies_sum = 0
        total_normal_sum = 0
        total_flights_sum = 0
        
        for row in results:
            hour_str, anomaly_count, normal_count, flight_count = row
            if hour_str is not None:
                hour = int(hour_str)
                
                # Store raw totals
                anomaly_totals_by_hour[hour] = anomaly_count or 0
                normal_totals_by_hour[hour] = normal_count or 0
                traffic_totals_by_hour[hour] = flight_count or 0
                
                # Track totals
                total_anomalies_sum += anomaly_count or 0
                total_normal_sum += normal_count or 0
                total_flights_sum += flight_count or 0
        
        # Now calculate daily averages using consistent num_days
        traffic_by_hour = {h: round(t / num_days, 1) for h, t in traffic_totals_by_hour.items()}
        anomaly_by_hour = {h: round(t / num_days, 1) for h, t in anomaly_totals_by_hour.items()}
        normal_by_hour = {h: round(t / num_days, 1) for h, t in normal_totals_by_hour.items()}
        
        # Safety events = anomaly flights
        safety_by_hour = anomaly_by_hour
        
        # Find peak traffic hours (top 3) based on daily average
        peak_traffic_hours = sorted(traffic_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_traffic_hours = [h[0] for h in peak_traffic_hours]
        
        # Find peak anomaly hours (top 3) - hours with most anomaly flights on average
        peak_safety_hours = sorted(safety_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_safety_hours = [h[0] for h in peak_safety_hours]
        
        # Calculate Pearson correlation between traffic and anomalies by hour
        correlation_score = self._calculate_pearson_correlation(traffic_by_hour, safety_by_hour)
        
        # Calculate anomaly rate
        anomaly_rate = round((total_anomalies_sum / total_flights_sum * 100) if total_flights_sum > 0 else 0, 1)
        
        # Build hourly_data array for frontend charts
        hourly_data = []
        total_traffic_avg = sum(traffic_by_hour.values())
        total_anomaly_avg = sum(anomaly_by_hour.values())
        
        for hour in range(24):
            traffic_avg = traffic_by_hour.get(hour, 0)
            anomaly_avg = anomaly_by_hour.get(hour, 0)
            normal_avg = normal_by_hour.get(hour, 0)
            
            hourly_data.append({
                'hour': hour,
                # Daily averages (calculated with consistent num_days)
                'traffic': traffic_avg,
                'safety_events': anomaly_avg,
                'normal_flights': normal_avg,
                # Percentage of daily traffic/anomalies at this hour
                'traffic_pct': round((traffic_avg / total_traffic_avg * 100) if total_traffic_avg > 0 else 0, 1),
                'safety_pct': round((anomaly_avg / total_anomaly_avg * 100) if total_anomaly_avg > 0 else 0, 1),
                # Totals for this hour across entire period (for tooltip) - use actual totals
                'traffic_total': traffic_totals_by_hour.get(hour, 0),
                'safety_total': anomaly_totals_by_hour.get(hour, 0),
            })
        
        return {
            'peak_traffic_hours': peak_traffic_hours,
            'peak_safety_hours': peak_safety_hours,
            'correlation_score': correlation_score,
            'hourly_data': hourly_data,
            # Totals across entire period
            'total_flights': total_flights_sum,
            'total_normal_flights': total_normal_sum,
            'total_anomaly_flights': total_anomalies_sum,
            'anomaly_rate_pct': anomaly_rate,
            'num_days': round(num_days, 1),
            'explanation': {
                'traffic': 'Average number of ALL flights per day that started during each hour',
                'safety_events': 'Average number of ANOMALY flights per day during each hour',
                'correlation': 'How strongly traffic volume relates to anomaly occurrence (0-100%)'
            }
        }
    
    def _calculate_pearson_correlation(self, data1: Dict[int, int], data2: Dict[int, int]) -> float:
        """
        Calculate Pearson correlation coefficient between two hourly distributions.
        
        Returns:
            Correlation coefficient between -1 and 1, or 0 if insufficient data.
        """
        import math
        
        # Create aligned vectors for all 24 hours
        hours = range(24)
        x = [data1.get(h, 0) for h in hours]
        y = [data2.get(h, 0) for h in hours]
        
        n = len(x)
        if n == 0:
            return 0.0
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate covariance and standard deviations
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        
        # Avoid division by zero
        if std_x == 0 or std_y == 0:
            return 0.0
        
        correlation = covariance / (std_x * std_y)
        return round(correlation, 3)
    
    def get_alternate_airports_by_time(self, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get all alternate airports used during diversions in a time period.
        
        Unlike get_alternate_airports() which filters by primary airport, this returns
        ALL diversions in the time range with wide-body vs narrow-body breakdown.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            [{airport, count, aircraft_types: [...], last_used: timestamp,
              wide_body_count, narrow_body_count, body_type_preference}]
        """
        # Query anomaly reports for Rule 8 (diversion) matches
        query = """
            SELECT timestamp, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Aggregate by alternate airport with body type tracking
        alternate_stats = defaultdict(lambda: {
            'count': 0, 
            'aircraft_types': set(), 
            'last_used': 0,
            'wide_body_count': 0,
            'narrow_body_count': 0,
            'unknown_count': 0
        })
        
        for row in results:
            timestamp, report_json = row
            try:
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                callsign = report.get('summary', {}).get('callsign', '')
                
                for rule in matched_rules:
                    if rule.get('id') == 8:  # Diversion rule
                        details = rule.get('details', {})
                        actual = details.get('actual', '')
                        planned = details.get('planned', '')
                        
                        # Skip if no actual landing airport or same as planned
                        if not actual or actual == planned:
                            continue
                        
                        alternate_stats[actual]['count'] += 1
                        alternate_stats[actual]['last_used'] = max(
                            alternate_stats[actual]['last_used'], 
                            timestamp
                        )
                        
                        # Try to extract aircraft type
                        aircraft_type = self._extract_aircraft_type(report)
                        if aircraft_type:
                            alternate_stats[actual]['aircraft_types'].add(aircraft_type)
                            
                            # Classify body type
                            body_type = _classify_aircraft_body_type(aircraft_type)
                            if body_type == 'wide_body':
                                alternate_stats[actual]['wide_body_count'] += 1
                            elif body_type == 'narrow_body':
                                alternate_stats[actual]['narrow_body_count'] += 1
                            else:
                                alternate_stats[actual]['unknown_count'] += 1
                        else:
                            alternate_stats[actual]['unknown_count'] += 1
                            
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        # Convert to output format with body type preference
        result = []
        for airport, data in alternate_stats.items():
            # Determine body type preference
            wide = data['wide_body_count']
            narrow = data['narrow_body_count']
            total_known = wide + narrow
            
            if total_known > 0:
                wide_pct = (wide / total_known) * 100
                if wide_pct >= 70:
                    preference = 'wide_body_preferred'
                elif wide_pct <= 30:
                    preference = 'narrow_body_preferred'
                else:
                    preference = 'mixed'
            else:
                preference = 'unknown'
            
            result.append({
                'airport': airport,
                'count': data['count'],
                'aircraft_types': list(data['aircraft_types'])[:5],
                'last_used': data['last_used'],
                'wide_body_count': wide,
                'narrow_body_count': narrow,
                'body_type_preference': preference
            })
        
        # Sort by count descending
        result.sort(key=lambda x: x['count'], reverse=True)
        
        return result[:15] if result else []
    
    def get_airline_activity_trends(self, start_ts: int, end_ts: int, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Detect airlines that started or stopped flying in the region.
        
        Compares activity in the current period vs a lookback period to identify:
        - Airlines that stopped flying (were active before, not active now)
        - Airlines that started flying (not active before, active now)
        - Airlines with significant activity changes
        
        Answers the demand: "Did any airline that used to fly over Israel stop flying?"
        
        Args:
            start_ts: Start of current period
            end_ts: End of current period
            lookback_days: Days to look back for comparison (default 30)
        
        Returns:
            {
                'stopped_flying': [{airline, last_seen, flight_count_before}],
                'started_flying': [{airline, first_seen, flight_count}],
                'activity_changes': [{airline, change_percent, before_count, after_count}]
            }
        """
        from datetime import datetime
        
        # Calculate lookback period
        current_duration = end_ts - start_ts
        lookback_start = start_ts - (lookback_days * 86400)
        lookback_end = start_ts
        
        def get_airline_flights(period_start: int, period_end: int) -> Dict[str, Dict]:
            """Get flight counts per airline for a period."""
            airline_stats = defaultdict(lambda: {'count': 0, 'first_seen': None, 'last_seen': None})
            
            for table_name in ['anomalies_tracks', 'normal_tracks']:
                query = f"""
                    SELECT DISTINCT callsign, MIN(timestamp) as first_ts, MAX(timestamp) as last_ts, COUNT(*) as cnt
                    FROM {table_name}
                    WHERE timestamp BETWEEN ? AND ?
                      AND callsign IS NOT NULL
                      AND callsign != ''
                    GROUP BY callsign
                """
                results = self._execute_query('research', query, (period_start, period_end))
                
                for row in results:
                    callsign, first_ts, last_ts, cnt = row
                    # Extract airline code (first 2-3 letters)
                    airline = ''.join(c for c in callsign[:3] if c.isalpha()).upper()
                    if len(airline) < 2:
                        continue
                    
                    airline_stats[airline]['count'] += cnt
                    if airline_stats[airline]['first_seen'] is None or first_ts < airline_stats[airline]['first_seen']:
                        airline_stats[airline]['first_seen'] = first_ts
                    if airline_stats[airline]['last_seen'] is None or last_ts > airline_stats[airline]['last_seen']:
                        airline_stats[airline]['last_seen'] = last_ts
            
            return dict(airline_stats)
        
        # Get activity for both periods
        lookback_activity = get_airline_flights(lookback_start, lookback_end)
        current_activity = get_airline_flights(start_ts, end_ts)
        
        # Find airlines that stopped flying
        stopped_flying = []
        for airline, data in lookback_activity.items():
            if airline not in current_activity:
                stopped_flying.append({
                    'airline': airline,
                    'last_seen': data['last_seen'],
                    'last_seen_date': datetime.fromtimestamp(data['last_seen']).strftime('%Y-%m-%d') if data['last_seen'] else None,
                    'flight_count_before': data['count']
                })
        
        # Find airlines that started flying
        started_flying = []
        for airline, data in current_activity.items():
            if airline not in lookback_activity:
                started_flying.append({
                    'airline': airline,
                    'first_seen': data['first_seen'],
                    'first_seen_date': datetime.fromtimestamp(data['first_seen']).strftime('%Y-%m-%d') if data['first_seen'] else None,
                    'flight_count': data['count']
                })
        
        # Find significant activity changes (>50% change)
        activity_changes = []
        for airline in set(lookback_activity.keys()) & set(current_activity.keys()):
            before = lookback_activity[airline]['count']
            after = current_activity[airline]['count']
            
            if before > 0:
                change_percent = ((after - before) / before) * 100
                if abs(change_percent) >= 50:  # Significant change
                    activity_changes.append({
                        'airline': airline,
                        'change_percent': round(change_percent, 1),
                        'before_count': before,
                        'after_count': after,
                        'trend': 'increasing' if change_percent > 0 else 'decreasing'
                    })
        
        # Sort by significance
        stopped_flying.sort(key=lambda x: x['flight_count_before'], reverse=True)
        started_flying.sort(key=lambda x: x['flight_count'], reverse=True)
        activity_changes.sort(key=lambda x: abs(x['change_percent']), reverse=True)
        
        return {
            'stopped_flying': stopped_flying[:10],
            'started_flying': started_flying[:10],
            'activity_changes': activity_changes[:10],
            'analysis_period': {
                'current_start': start_ts,
                'current_end': end_ts,
                'lookback_start': lookback_start,
                'lookback_end': lookback_end,
                'lookback_days': lookback_days
            }
        }

