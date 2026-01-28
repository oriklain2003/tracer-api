"""
Intelligence gathering and pattern detection for Level 3 features.

Provides:
- GPS jamming detection and mapping
- Military aircraft tracking
- Anomaly DNA (pattern matching)
- Event correlation
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
_current_file = Path(__file__).resolve()
_parent_parent = _current_file.parent.parent.parent
if str(_parent_parent) not in sys.path:
    sys.path.insert(0, str(_parent_parent))

# Import common military detection
try:
    from core.military_detection import (
        is_military as core_is_military, 
        MILITARY_REFERENCE,
        MILITARY_PREFIXES
    )
except ImportError:
    # Fallback if import fails
    core_is_military = None
    MILITARY_REFERENCE = None
    MILITARY_PREFIXES = None


# Military callsign patterns and their associated metadata
MILITARY_CALLSIGN_PATTERNS = {
    # US Military - Transport
    'RCH': {'country': 'US', 'type': 'transport', 'name': 'REACH - USAF AMC'},
    'CNV': {'country': 'US', 'type': 'transport', 'name': 'Convoy - USAF'},
    'EVAC': {'country': 'US', 'type': 'transport', 'name': 'Medical Evacuation'},
    'SAM': {'country': 'US', 'type': 'vip', 'name': 'Special Air Mission'},
    'EXEC': {'country': 'US', 'type': 'vip', 'name': 'Executive Flight'},
    
    # US Military - Tankers (Aerial Refueling)
    'QUID': {'country': 'US', 'type': 'tanker', 'name': 'KC-135/KC-10 Tanker'},
    'NCHO': {'country': 'US', 'type': 'tanker', 'name': 'KC-135 Stratotanker'},
    'SHELL': {'country': 'US', 'type': 'tanker', 'name': 'Aerial Refueling'},
    'ARCO': {'country': 'US', 'type': 'tanker', 'name': 'KC-135 Tanker Track'},
    'TEXACO': {'country': 'US', 'type': 'tanker', 'name': 'KC-46 Tanker'},
    'ESSO': {'country': 'US', 'type': 'tanker', 'name': 'KC-10 Extender'},
    'GULF': {'country': 'US', 'type': 'tanker', 'name': 'Tanker Operations'},
    'MOBIL': {'country': 'US', 'type': 'tanker', 'name': 'Tanker Operations'},
    
    # US Military - ISR (Intelligence, Surveillance, Reconnaissance)
    'JAKE': {'country': 'US', 'type': 'ISR', 'name': 'RC-135 Rivet Joint'},
    'DOOM': {'country': 'US', 'type': 'ISR', 'name': 'E-8 JSTARS'},
    'HOMER': {'country': 'US', 'type': 'ISR', 'name': 'E-3 AWACS/Sentry'},
    'SENTRY': {'country': 'US', 'type': 'ISR', 'name': 'E-3 AWACS'},
    'FORTE': {'country': 'US', 'type': 'ISR', 'name': 'RQ-4 Global Hawk'},
    'REAPER': {'country': 'US', 'type': 'ISR', 'name': 'MQ-9 Reaper'},
    'DARK': {'country': 'US', 'type': 'ISR', 'name': 'ISR Operations'},
    'COBRA': {'country': 'US', 'type': 'ISR', 'name': 'RC-135 Cobra Ball'},
    'OLIVE': {'country': 'US', 'type': 'ISR', 'name': 'E-8 JSTARS'},
    
    # US Military - Fighters
    'VIPER': {'country': 'US', 'type': 'fighter', 'name': 'F-16 Fighting Falcon'},
    'RAGE': {'country': 'US', 'type': 'fighter', 'name': 'Fighter Aircraft'},
    'RAPTOR': {'country': 'US', 'type': 'fighter', 'name': 'F-22 Raptor'},
    'EAGLE': {'country': 'US', 'type': 'fighter', 'name': 'F-15 Eagle'},
    'STRIKE': {'country': 'US', 'type': 'fighter', 'name': 'F-15E Strike Eagle'},
    'HORNET': {'country': 'US', 'type': 'fighter', 'name': 'F/A-18 Hornet'},
    'WEASEL': {'country': 'US', 'type': 'fighter', 'name': 'Wild Weasel SEAD'},
    
    # UK Military
    'RAF': {'country': 'GB', 'type': 'transport', 'name': 'Royal Air Force'},
    'RFR': {'country': 'GB', 'type': 'tanker', 'name': 'RAF Voyager Tanker'},
    'ASCOT': {'country': 'GB', 'type': 'transport', 'name': 'RAF Transport'},
    'TWINSTAR': {'country': 'GB', 'type': 'tanker', 'name': 'RAF Voyager'},
    'TARTAN': {'country': 'GB', 'type': 'fighter', 'name': 'RAF Typhoon'},
    'BOXER': {'country': 'GB', 'type': 'ISR', 'name': 'RAF RC-135 Airseeker'},
    
    # Russian Military
    'RRR': {'country': 'RU', 'type': 'transport', 'name': 'Russian Air Force'},
    'RFF': {'country': 'RU', 'type': 'transport', 'name': 'Russian Federation'},
    'RSD': {'country': 'RU', 'type': 'transport', 'name': 'Russian State Transport'},
    
    # Israeli Military
    'IAF': {'country': 'IL', 'type': 'transport', 'name': 'Israeli Air Force'},
    'ISF': {'country': 'IL', 'type': 'fighter', 'name': 'Israeli Air Force'},
    
    # NATO/Allied
    'NATO': {'country': 'NATO', 'type': 'ISR', 'name': 'NATO Operations'},
    'MMF': {'country': 'NATO', 'type': 'tanker', 'name': 'Multinational MRTT Fleet'},
    'MAGIC': {'country': 'NATO', 'type': 'ISR', 'name': 'NATO E-3 AWACS'},
    
    # Other Allied Nations
    'GAF': {'country': 'DE', 'type': 'transport', 'name': 'German Air Force'},
    'FAF': {'country': 'FR', 'type': 'transport', 'name': 'French Air Force'},
    'AMI': {'country': 'IT', 'type': 'transport', 'name': 'Italian Air Force'},
    'RAAF': {'country': 'AU', 'type': 'transport', 'name': 'Royal Australian AF'},
    'CANFORCE': {'country': 'CA', 'type': 'transport', 'name': 'Canadian Forces'},
    
    # Middle East Military
    'SHAHD': {'country': 'JO', 'type': 'military', 'name': 'Royal Jordanian Air Force'},
    'RJAF': {'country': 'JO', 'type': 'military', 'name': 'Royal Jordanian Air Force'},
}

# Tanker-specific offshore holding areas (common refueling tracks)
TANKER_HOLDING_AREAS = [
    {'name': 'Eastern Med Track', 'lat': 34.0, 'lon': 33.0, 'radius_nm': 100},
    {'name': 'Gulf Track', 'lat': 27.0, 'lon': 52.0, 'radius_nm': 100},
    {'name': 'Red Sea Track', 'lat': 22.0, 'lon': 37.0, 'radius_nm': 100},
]

# ISR-typical racetrack areas
ISR_PATROL_AREAS = [
    {'name': 'Eastern Med Patrol', 'lat': 35.0, 'lon': 34.0, 'radius_nm': 150},
    {'name': 'Black Sea Patrol', 'lat': 43.0, 'lon': 34.0, 'radius_nm': 150},
    {'name': 'Baltic Patrol', 'lat': 56.0, 'lon': 18.0, 'radius_nm': 150},
]


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


class IntelligenceEngine:
    """Engine for intelligence gathering and pattern detection."""
    
    def __init__(self, db_paths: Dict[str, Path], use_optimized: bool = True):
        """
        Initialize intelligence engine.
        
        Args:
            db_paths: Dictionary mapping db names to paths
            use_optimized: Use optimized pandas/numpy methods (default True)
        """
        
        self.db_paths = db_paths
        self.use_optimized = use_optimized
        # Initialize optimized engine if enabled
        self._optimized_engine = None
        if self.use_optimized:
            try:
                
                from .intelligence_optimized import OptimizedIntelligenceEngine
                self._optimized_engine = OptimizedIntelligenceEngine(db_paths)
                print("[OK] Using optimized intelligence engine (pandas/numpy)")
            except ImportError as e:
                print(f"[WARNING] Optimized intelligence engine not available: {e}")
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
    
    def get_operational_tempo(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get military operational tempo over time - hourly and daily activity levels by country.
        
        Shows military activity buildup patterns that can predict upcoming operations.
        
        Returns:
            {
                'hourly_data': [{hour, US, RU, GB, IL, NATO, other, total}],
                'daily_data': [{date, US, RU, GB, IL, NATO, other, total}],
                'by_country_total': {country: count},
                'peak_activity': {timestamp, hour, country, count, description},
                'activity_spikes': [{timestamp, hour, country, count, average, increase_pct, description}],
                'trend_analysis': {country: 'increasing'|'stable'|'decreasing'},
                'alerts': [...],
                'total_flights': int
            }
        """
        from datetime import datetime
        
        # Query flight metadata for military flights from research database
        query = """
            SELECT 
                callsign, 
                first_seen_ts,
                military_type,
                category
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Initialize counters
        hourly_by_country = defaultdict(lambda: defaultdict(int))  # hour -> country -> count
        daily_by_country = defaultdict(lambda: defaultdict(int))   # date -> country -> count
        total_by_country = defaultdict(int)
        country_by_hour = defaultdict(lambda: defaultdict(int))    # country -> hour -> count (for spike detection)
        
        # Country detection function using MILITARY_CALLSIGN_PATTERNS and core detection
        def detect_country(callsign: str) -> str:
            """Detect country from callsign using comprehensive patterns."""
            if not callsign:
                return 'other'
            
            cs_upper = callsign.strip().upper()
            
            # Use core military detection if available
            if core_is_military is not None:
                is_mil, mil_info = core_is_military(callsign=callsign)
                if is_mil and mil_info:
                    info_lower = mil_info.lower()
                    # US detection
                    if any(x in info_lower for x in ['us ', 'usaf', 'us air force', 'us navy', 'us marine', 'us army', 'american']):
                        return 'US'
                    # UK detection
                    if any(x in info_lower for x in ['united kingdom', 'raf', 'royal air force', 'royal navy', 'british']):
                        return 'GB'
                    # Israel detection - expanded patterns
                    if any(x in info_lower for x in ['israel', 'idf', 'israeli']):
                        return 'IL'
                    # Russia detection
                    if any(x in info_lower for x in ['russia', 'russian']):
                        return 'RU'
                    # NATO detection
                    if 'nato' in info_lower:
                        return 'NATO'
                    # German -> NATO
                    if any(x in info_lower for x in ['german', 'luftwaffe']):
                        return 'NATO'
                    # French -> NATO
                    if 'french' in info_lower:
                        return 'NATO'
            
            # Fallback: Use local patterns dictionary for prefix matching
            for prefix, info in MILITARY_CALLSIGN_PATTERNS.items():
                if cs_upper.startswith(prefix.upper()):
                    return info.get('country', 'other')
            
            # Additional Israeli patterns not in the dictionary
            if cs_upper.startswith(('IAF', 'ISF', '4XA', '4XB', '4XC')):
                return 'IL'
            
            # Additional US patterns
            if cs_upper.startswith(('RCH', 'REACH', 'CNV', 'CONVOY', 'EVAC', 'SAM', 'NAVY', 'PAT', 'VM', 'FORTE', 'HOMER', 'DARK')):
                return 'US'
            
            # Additional UK patterns
            if cs_upper.startswith(('RAF', 'RRR', 'ASCOT', 'AAC', 'RFR', 'TARTN')):
                return 'GB'
            
            # Russian patterns
            if cs_upper.startswith(('RFF', 'RSD')):
                return 'RU'
            
            # NATO patterns (other European militaries)
            if cs_upper.startswith(('GAF', 'BAF', 'FAF', 'CTM', 'AME', 'PLF', 'RDAF', 'MMF', 'MAGIC')):
                return 'NATO'
            
            # Jordan patterns
            if cs_upper.startswith(('SHAHD', 'RJAF')):
                return 'JO'
            
            return 'other'
        
        # Process each flight
        total_flights = 0
        for row in results:
            callsign, first_seen_ts, military_type, category = row
            
            if first_seen_ts is None:
                continue
            
            total_flights += 1
            
            # Detect country
            country = detect_country(callsign)
            
            # Get hour and date
            try:
                dt = datetime.fromtimestamp(first_seen_ts)
                hour = dt.hour
                date_str = dt.strftime('%Y-%m-%d')
            except (ValueError, OSError):
                continue
            
            # Aggregate counts
            hourly_by_country[hour][country] += 1
            daily_by_country[date_str][country] += 1
            total_by_country[country] += 1
            country_by_hour[country][hour] += 1
        
        # Build hourly_data (all 24 hours)
        countries = ['US', 'RU', 'GB', 'IL', 'NATO', 'other']
        hourly_data = []
        for hour in range(24):
            entry = {
                'hour': hour,
                'hour_label': f"{hour:02d}:00",
                'US': hourly_by_country[hour].get('US', 0),
                'RU': hourly_by_country[hour].get('RU', 0),
                'GB': hourly_by_country[hour].get('GB', 0),
                'IL': hourly_by_country[hour].get('IL', 0),
                'NATO': hourly_by_country[hour].get('NATO', 0),
                'other': hourly_by_country[hour].get('other', 0),
            }
            entry['total'] = sum(entry[c] for c in countries)
            hourly_data.append(entry)
        
        # Build daily_data (sorted by date)
        daily_data = []
        for date_str in sorted(daily_by_country.keys()):
            entry = {
                'date': date_str,
                'US': daily_by_country[date_str].get('US', 0),
                'RU': daily_by_country[date_str].get('RU', 0),
                'GB': daily_by_country[date_str].get('GB', 0),
                'IL': daily_by_country[date_str].get('IL', 0),
                'NATO': daily_by_country[date_str].get('NATO', 0),
                'other': daily_by_country[date_str].get('other', 0),
            }
            entry['total'] = sum(entry[c] for c in countries)
            daily_data.append(entry)
        
        # Find peak activity - look for the country with highest single-day count
        # Exclude 'other' unless it's the only option
        peak_activity = None
        max_daily_count = 0
        peak_date = None
        peak_country = None
        
        for day in daily_data:
            # Check each specific country first (not 'other')
            for country in ['US', 'RU', 'GB', 'IL', 'NATO']:
                if day[country] > max_daily_count:
                    max_daily_count = day[country]
                    peak_date = day['date']
                    peak_country = country
        
        # If no specific country has activity, check 'other'
        if max_daily_count == 0:
            for day in daily_data:
                if day['other'] > max_daily_count:
                    max_daily_count = day['other']
                    peak_date = day['date']
                    peak_country = 'other'
        
        if peak_country and max_daily_count > 0:
            peak_activity = {
                'timestamp': int(datetime.strptime(peak_date, '%Y-%m-%d').timestamp()) if peak_date else 0,
                'hour': peak_date,
                'country': peak_country,
                'count': max_daily_count,
                'description': f"Peak activity: {max_daily_count} flights from {peak_country} on {peak_date}"
            }
        
        # Calculate activity spikes (days where country activity is significantly above average)
        # Requirements for a meaningful spike:
        # 1. Day count > 150% of average (>50% increase)
        # 2. Minimum average of 3 flights/day (avoid noise from low-activity countries)
        # 3. Minimum count of 5 flights on that day (absolute significance)
        # 4. Include 'other' category since it often has the most data
        activity_spikes = []
        for country in ['US', 'RU', 'GB', 'IL', 'NATO', 'other']:
            country_daily_counts = [day[country] for day in daily_data]
            if country_daily_counts:
                avg_count = sum(country_daily_counts) / len(country_daily_counts)
                # Only consider countries with meaningful baseline activity
                min_avg_threshold = 2.0  # Must have at least 2 flights/day on average
                min_count_threshold = 5   # Must have at least 5 flights on spike day
                
                if avg_count >= min_avg_threshold:
                    for day in daily_data:
                        count = day[country]
                        # Check both relative (>150% of avg) and absolute (>= min threshold) requirements
                        if count >= min_count_threshold and count > avg_count * 1.5:
                            increase_pct = int(((count - avg_count) / avg_count) * 100)
                            activity_spikes.append({
                                'timestamp': int(datetime.strptime(day['date'], '%Y-%m-%d').timestamp()),
                                'hour': day['date'],
                                'country': country,
                                'count': count,
                                'average': round(avg_count, 1),
                                'increase_pct': increase_pct,
                                'absolute_increase': count - avg_count,  # For better sorting
                                'description': f"{country} had {count} flights ({increase_pct}% above average of {round(avg_count, 1)})"
                            })
        
        # Sort spikes by a weighted score: prioritize both high percentage AND high absolute count
        # This prevents tiny countries with 2->4 flights showing as "+100%" above meaningful spikes
        for spike in activity_spikes:
            # Weighted score: 60% absolute increase, 40% percentage increase (normalized)
            spike['spike_score'] = spike['absolute_increase'] * 0.6 + (spike['increase_pct'] / 100) * spike['count'] * 0.4
        
        activity_spikes = sorted(activity_spikes, key=lambda x: x['spike_score'], reverse=True)[:20]
        
        # Calculate trend analysis - compare first half vs second half of period
        # A proper trend analysis looks at the direction of change over time
        trend_analysis = {}
        
        if len(daily_data) >= 4:
            # Split into two halves
            mid_point = len(daily_data) // 2
            first_half = daily_data[:mid_point]
            second_half = daily_data[mid_point:]
            
            for country in ['US', 'RU', 'GB', 'IL', 'NATO']:
                first_half_avg = sum(day[country] for day in first_half) / len(first_half) if first_half else 0
                second_half_avg = sum(day[country] for day in second_half) / len(second_half) if second_half else 0
                
                # Also look at the last few days vs average to detect recent decline
                last_days = daily_data[-min(7, len(daily_data)):]
                last_days_avg = sum(day[country] for day in last_days) / len(last_days) if last_days else 0
                
                if first_half_avg == 0 and second_half_avg == 0:
                    trend_analysis[country] = 'stable'
                elif second_half_avg > first_half_avg * 1.3:
                    # But check if recent activity is declining
                    if last_days_avg < second_half_avg * 0.5:
                        trend_analysis[country] = 'decreasing'  # Peaked and now declining
                    else:
                        trend_analysis[country] = 'increasing'
                elif second_half_avg < first_half_avg * 0.7:
                    trend_analysis[country] = 'decreasing'
                else:
                    # Check for recent decline from peak
                    overall_avg = sum(day[country] for day in daily_data) / len(daily_data) if daily_data else 0
                    if last_days_avg < overall_avg * 0.5 and overall_avg > 0:
                        trend_analysis[country] = 'decreasing'
                    else:
                        trend_analysis[country] = 'stable'
        else:
            # Not enough data for trend analysis
            for country in ['US', 'RU', 'GB', 'IL', 'NATO']:
                trend_analysis[country] = 'stable'
        
        # Generate alerts
        alerts = []
        for country in ['US', 'RU', 'GB', 'IL', 'NATO']:
            if total_by_country.get(country, 0) > 50:
                alerts.append({
                    'type': 'high_activity',
                    'severity': 'info',
                    'message': f"High {country} military activity: {total_by_country[country]} flights detected"
                })
        
        if trend_analysis.get('RU') == 'increasing':
            alerts.append({
                'type': 'trend_alert',
                'severity': 'warning',
                'message': 'Russian military activity showing increasing trend'
            })
        
        # by_country_total - filter out 0 counts to keep it clean
        by_country_total = {k: v for k, v in total_by_country.items() if v > 0}
        
        return {
            'hourly_data': hourly_data,
            'daily_data': daily_data,
            'by_country_total': by_country_total,
            'peak_activity': peak_activity,
            'activity_spikes': activity_spikes,
            'trend_analysis': trend_analysis,
            'alerts': alerts,
            'total_flights': total_flights
        }
    
    def get_tanker_activity(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Track aerial refueling tanker activity - indicates upcoming strike operations.
        
        Returns:
            {
                'active_tankers': [{flight_id, callsign, country, holding_area, duration_min, orbit_count, 
                                    point_count, start_time, end_time, last_position}],
                'tanker_count': int,
                'total_tanker_hours': float,
                'by_holding_area': {area: count},
                'by_country': {country: count},
                'alerts': [...]
            }
        """
        # Known refueling track areas (approximate centers)
        HOLDING_AREAS = {
            'TRACK_EAST': {'center': (34.5, 36.5), 'radius_nm': 50, 'name': 'Eastern Med'},
            'TRACK_WEST': {'center': (34.0, 32.0), 'radius_nm': 50, 'name': 'Western Med'},
            'TRACK_NORTH': {'center': (36.5, 35.0), 'radius_nm': 50, 'name': 'Northern Track'},
            'TRACK_SOUTH': {'center': (31.0, 34.5), 'radius_nm': 50, 'name': 'Southern Track'},
            'RED_SEA': {'center': (26.0, 36.0), 'radius_nm': 80, 'name': 'Red Sea'},
            'PERSIAN_GULF': {'center': (27.0, 51.0), 'radius_nm': 80, 'name': 'Persian Gulf'},
        }
        
        def get_holding_area(lat: float, lon: float) -> str:
            """Determine holding area based on position."""
            if lat is None or lon is None:
                return 'Unknown'
            for track_id, area in HOLDING_AREAS.items():
                center_lat, center_lon = area['center']
                # Simple distance check (approximate)
                dist = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5 * 60  # rough nm
                if dist < area['radius_nm']:
                    return area['name']
            return 'Other'
        
        # Query for tanker flights with flight_id for track lookup
        query = """
            SELECT 
                flight_id,
                callsign,
                first_seen_ts,
                last_seen_ts,
                flight_duration_sec,
                military_type
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (
                military_type LIKE '%tanker%' 
                OR military_type LIKE '%refuel%'
                OR callsign LIKE 'QUID%'
                OR callsign LIKE 'SHELL%'
                OR callsign LIKE 'ARCO%'
                OR callsign LIKE 'TEXACO%'
                OR callsign LIKE 'ESSO%'
                OR callsign LIKE 'RFR%'
                OR callsign LIKE 'TARTN%'
            )
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        active_tankers = []
        by_country = defaultdict(int)
        by_holding_area = defaultdict(int)
        total_tanker_hours = 0
        
        # Collect flight_ids for batch position lookup
        flight_ids = []
        flight_data = {}
        
        for row in results:
            flight_id, callsign, first_seen, last_seen, duration_sec, mil_type = row
            
            # Detect country from callsign
            country = 'Unknown'
            cs_upper = (callsign or '').upper()
            if cs_upper.startswith(('QUID', 'SHELL', 'ARCO', 'TEXACO', 'ESSO', 'GULF', 'MOBIL')):
                country = 'US'
            elif cs_upper.startswith(('RFR', 'TARTN', 'TWINSTAR')):
                country = 'GB'
            elif cs_upper.startswith('MMF'):
                country = 'NATO'
            
            duration_min = int((duration_sec or 0) / 60)
            duration_hours = (duration_sec or 0) / 3600
            total_tanker_hours += duration_hours
            by_country[country] += 1
            
            flight_ids.append(flight_id)
            flight_data[flight_id] = {
                'flight_id': flight_id,
                'callsign': callsign,
                'country': country,
                'start_time': first_seen,
                'end_time': last_seen,
                'duration_min': duration_min,
                'military_type': mil_type or 'tanker'
            }
        
        # Batch query for last positions and track point counts
        if flight_ids:
            # Get last position for each flight
            placeholders = ','.join(['?' for _ in flight_ids])
            position_query = f"""
                SELECT flight_id, lat, lon, alt, timestamp,
                       (SELECT COUNT(*) FROM anomalies_tracks t2 WHERE t2.flight_id = t.flight_id) as point_count
                FROM anomalies_tracks t
                WHERE flight_id IN ({placeholders})
                AND timestamp = (
                    SELECT MAX(timestamp) FROM anomalies_tracks t3 WHERE t3.flight_id = t.flight_id
                )
                GROUP BY flight_id
            """
            position_results = self._execute_query('research', position_query, tuple(flight_ids))
            
            # Also try normal_tracks if not found
            normal_query = f"""
                SELECT flight_id, lat, lon, alt, timestamp,
                       (SELECT COUNT(*) FROM normal_tracks t2 WHERE t2.flight_id = t.flight_id) as point_count
                FROM normal_tracks t
                WHERE flight_id IN ({placeholders})
                AND timestamp = (
                    SELECT MAX(timestamp) FROM normal_tracks t3 WHERE t3.flight_id = t.flight_id
                )
                GROUP BY flight_id
            """
            normal_results = self._execute_query('research', normal_query, tuple(flight_ids))
            
            # Merge results
            positions = {}
            for row in (position_results or []):
                fid, lat, lon, alt, ts, point_count = row
                if fid and lat and lon:
                    positions[fid] = {
                        'lat': lat, 'lon': lon, 'alt': alt or 0, 
                        'timestamp': ts, 'point_count': point_count or 0
                    }
            for row in (normal_results or []):
                fid, lat, lon, alt, ts, point_count = row
                if fid and lat and lon and fid not in positions:
                    positions[fid] = {
                        'lat': lat, 'lon': lon, 'alt': alt or 0, 
                        'timestamp': ts, 'point_count': point_count or 0
                    }
            
            # Update flight data with positions
            for flight_id, data in flight_data.items():
                pos = positions.get(flight_id)
                if pos:
                    holding_area = get_holding_area(pos['lat'], pos['lon'])
                    by_holding_area[holding_area] += 1
                    
                    # Estimate orbit count from point count and duration
                    point_count = pos.get('point_count', 0)
                    duration_min = data['duration_min']
                    # Rough estimate: 1 orbit every 15-20 minutes for tankers
                    orbit_count = max(1, duration_min // 17) if duration_min > 0 else 0
                    
                    active_tankers.append({
                        'flight_id': flight_id,
                        'callsign': data['callsign'],
                        'country': data['country'],
                        'holding_area': holding_area,
                        'duration_min': duration_min,
                        'orbit_count': orbit_count,
                        'point_count': point_count,
                        'start_time': data['start_time'],
                        'end_time': data['end_time'],
                        'last_position': {
                            'lat': pos['lat'],
                            'lon': pos['lon'],
                            'alt': pos['alt'],
                            'timestamp': pos['timestamp']
                        }
                    })
                else:
                    # No position data available - still include but without map marker capability
                    by_holding_area['Unknown'] += 1
                    active_tankers.append({
                        'flight_id': flight_id,
                        'callsign': data['callsign'],
                        'country': data['country'],
                        'holding_area': 'Unknown',
                        'duration_min': data['duration_min'],
                        'orbit_count': 0,
                        'point_count': 0,
                        'start_time': data['start_time'],
                        'end_time': data['end_time'],
                        'last_position': None
                    })
        
        # Sort by duration (most active first)
        active_tankers.sort(key=lambda x: x['duration_min'], reverse=True)
        
        alerts = []
        if len(active_tankers) > 5:
            alerts.append({
                'type': 'high_tanker_activity',
                'severity': 'warning',
                'message': f"High tanker activity: {len(active_tankers)} tanker flights detected"
            })
        
        return {
            'active_tankers': active_tankers[:50],  # Limit to 50
            'tanker_count': len(active_tankers),
            'total_tanker_hours': round(total_tanker_hours, 1),
            'by_holding_area': dict(by_holding_area),
            'by_country': dict(by_country),
            'alerts': alerts
        }
    
    def get_night_operations(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Analyze military night operations - often indicates sensitive/covert activity.
        Night is defined as 20:00-06:00 local time.
        
        Returns:
            {
                'night_flights': [{callsign, country, timestamp, hour}],
                'day_vs_night': {'day': count, 'night': count, 'night_percentage': float},
                'by_country': {country: {'day': count, 'night': count}},
                'alerts': [...]
            }
        """
        from datetime import datetime
        
        # Query military flights from research database
        query = """
            SELECT callsign, first_seen_ts, category
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        night_flights = []
        day_count = 0
        night_count = 0
        by_country = defaultdict(lambda: {'day': 0, 'night': 0})
        
        for row in results:
            callsign, first_seen, category = row
            
            if first_seen is None:
                continue
            
            try:
                dt = datetime.fromtimestamp(first_seen)
                hour = dt.hour
            except (ValueError, OSError):
                continue
            
            # Night = 20:00-06:00
            is_night = hour >= 20 or hour < 6
            
            # Detect country
            country = 'other'
            cs_upper = (callsign or '').upper()
            for prefix, info in MILITARY_CALLSIGN_PATTERNS.items():
                if cs_upper.startswith(prefix.upper()):
                    country = info.get('country', 'other')
                    break
            
            if is_night:
                night_count += 1
                by_country[country]['night'] += 1
                night_flights.append({
                    'callsign': callsign,
                    'country': country,
                    'timestamp': first_seen,
                    'hour': hour
                })
            else:
                day_count += 1
                by_country[country]['day'] += 1
        
        total = day_count + night_count
        night_percentage = round((night_count / total * 100) if total > 0 else 0, 1)
        
        alerts = []
        if night_percentage > 40:
            alerts.append({
                'type': 'high_night_activity',
                'severity': 'warning',
                'message': f"Elevated night operations: {night_percentage}% of military flights at night"
            })
        
        return {
            'night_flights': night_flights[:100],  # Limit to 100
            'day_vs_night': {
                'day': day_count,
                'night': night_count,
                'night_percentage': night_percentage
            },
            'by_country': dict(by_country),
            'alerts': alerts
        }
    
    def get_isr_patterns(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Detect ISR (Intelligence, Surveillance, Reconnaissance) flight patterns.
        Identifies figure-8s, racetracks, and orbit patterns typical of reconnaissance.
        
        Returns:
            {
                'patterns': [{callsign, pattern_type, country, duration, area}],
                'total_isr_flights': int,
                'by_pattern_type': {type: count},
                'by_country': {country: count},
                'alerts': [...]
            }
        """
        # Query for ISR flights
        query = """
            SELECT callsign, first_seen_ts, last_seen_ts, flight_duration_sec, military_type
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (
                military_type LIKE '%ISR%' 
                OR military_type LIKE '%recon%'
                OR callsign LIKE 'FORTE%'
                OR callsign LIKE 'HOMER%'
                OR callsign LIKE 'JAKE%'
                OR callsign LIKE 'DOOM%'
                OR callsign LIKE 'SENTRY%'
                OR callsign LIKE 'OLIVE%'
                OR callsign LIKE 'COBRA%'
                OR callsign LIKE 'BOXER%'
                OR callsign LIKE 'MAGIC%'
            )
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        patterns = []
        by_pattern_type = defaultdict(int)
        by_country = defaultdict(int)
        
        for row in results:
            callsign, first_seen, last_seen, duration_sec, mil_type = row
            
            # Detect country
            country = 'US'  # Default for ISR
            cs_upper = (callsign or '').upper()
            if cs_upper.startswith('BOXER'):
                country = 'GB'
            elif cs_upper.startswith('MAGIC'):
                country = 'NATO'
            
            # Detect pattern type based on duration
            duration_hours = (duration_sec or 0) / 3600
            if duration_hours > 8:
                pattern_type = 'extended_patrol'
            elif duration_hours > 4:
                pattern_type = 'racetrack'
            else:
                pattern_type = 'orbit'
            
            by_pattern_type[pattern_type] += 1
            by_country[country] += 1
            
            patterns.append({
                'callsign': callsign,
                'pattern_type': pattern_type,
                'country': country,
                'duration_hours': round(duration_hours, 1),
                'military_type': mil_type or 'ISR'
            })
        
        alerts = []
        if len(patterns) > 10:
            alerts.append({
                'type': 'high_isr_activity',
                'severity': 'info',
                'message': f"Elevated ISR activity: {len(patterns)} reconnaissance flights detected"
            })
        
        return {
            'patterns': patterns[:50],
            'total_isr_flights': len(patterns),
            'by_pattern_type': dict(by_pattern_type),
            'by_country': dict(by_country),
            'likely_collection_areas': [],  # Would require track analysis
            'alerts': alerts
        }
    
    def get_airspace_denial_zones(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Detect areas where commercial aircraft are avoiding - indicates active operations.
        
        Returns:
            {
                'denial_zones': [{lat, lon, radius_nm, confidence, reason}],
                'total_zones': int,
                'most_avoided_areas': [...],
                'alerts': [...]
            }
        """
        # This would require complex analysis comparing current vs historical traffic
        # For now, return empty/minimal structure
        return {
            'denial_zones': [],
            'total_zones': 0,
            'most_avoided_areas': [],
            'alerts': []
        }
    
    def get_border_crossings(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Track military aircraft border crossings with timeline.
        
        Returns:
            {
                'crossings': [{callsign, country, border_pair, timestamp, direction}],
                'total_crossings': int,
                'by_country_pair': {pair: count},
                'high_interest_crossings': [...],
                'alerts': [...]
            }
        """
        # Query military flights with border crossings from research database
        query = """
            SELECT callsign, first_seen_ts, crossed_borders, military_type
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
            AND crossed_borders > 0
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        crossings = []
        by_country_pair = defaultdict(int)
        
        for row in results:
            callsign, timestamp, borders, mil_type = row
            
            # Detect country from callsign
            country = 'Unknown'
            cs_upper = (callsign or '').upper()
            for prefix, info in MILITARY_CALLSIGN_PATTERNS.items():
                if cs_upper.startswith(prefix.upper()):
                    country = info.get('country', 'Unknown')
                    break
            
            crossings.append({
                'callsign': callsign,
                'country': country,
                'timestamp': timestamp,
                'borders_crossed': borders,
                'military_type': mil_type
            })
        
        alerts = []
        if len(crossings) > 20:
            alerts.append({
                'type': 'high_border_activity',
                'severity': 'info',
                'message': f"High military border crossing activity: {len(crossings)} crossings detected"
            })
        
        return {
            'crossings': crossings[:100],
            'total_crossings': len(crossings),
            'by_country_pair': dict(by_country_pair),
            'high_interest_crossings': crossings[:10],
            'alerts': alerts
        }
    
    def get_ew_correlation(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Electronic Warfare Correlation Map - correlate GPS jamming with military activity.
        
        Returns:
            {
                'jamming_zones': [{lat, lon, severity, radius_nm, affected_flights, indicators}],
                'military_paths_in_zones': [...],
                'estimated_ew_sources': [{lat, lon, confidence, likely_operator, severity, military_correlation}],
                'correlation_score': float,
                'total_jamming_zones': int,
                'zones_with_military': int,
                'alerts': [...]
            }
        """
        # Get GPS jamming data
        raw_zones = self.get_gps_jamming_heatmap(start_ts, end_ts, limit=30)
        
        # Transform to expected frontend format
        jamming_zones = []
        estimated_sources = []
        zones_with_military = 0
        
        for zone in raw_zones:
            # Map backend fields to frontend expected fields
            transformed_zone = {
                'lat': zone.get('lat', 0),
                'lon': zone.get('lon', 0),
                'radius_nm': 15,  # Approximate based on grid size (0.25 degrees)
                'severity': zone.get('jamming_score', zone.get('intensity', 50)),
                'affected_flights': zone.get('affected_flights', zone.get('event_count', 0)),
                'indicators': zone.get('jamming_indicators', zone.get('indicators', [])),
            }
            jamming_zones.append(transformed_zone)
            
            # If severity is high, estimate an EW source nearby
            if transformed_zone['severity'] >= 40:
                zones_with_military += 1
                # Create estimated source at zone center
                estimated_sources.append({
                    'lat': transformed_zone['lat'],
                    'lon': transformed_zone['lon'],
                    'confidence': 'High' if transformed_zone['severity'] >= 70 else 'Medium' if transformed_zone['severity'] >= 50 else 'Low',
                    'likely_operator': self._estimate_ew_operator(transformed_zone['lat'], transformed_zone['lon']),
                    'severity': int(transformed_zone['severity']),
                    'military_correlation': min(100, int(transformed_zone['severity'] * 1.2))
                })
        
        # Calculate overall correlation score
        correlation_score = 0
        if jamming_zones:
            avg_severity = sum(z['severity'] for z in jamming_zones) / len(jamming_zones)
            correlation_score = min(100, int(avg_severity * 1.5))
        
        alerts = []
        if zones_with_military > 3:
            alerts.append({
                'type': 'high_ew_activity',
                'severity': 'warning',
                'message': f"Elevated EW activity detected: {zones_with_military} high-severity jamming zones"
            })
        
        return {
            'jamming_zones': jamming_zones,
            'military_paths_in_zones': [],  # Would require track correlation
            'estimated_ew_sources': estimated_sources[:10],  # Limit to top 10
            'correlation_score': correlation_score,
            'total_jamming_zones': len(jamming_zones),
            'zones_with_military': zones_with_military,
            'alerts': alerts
        }
    
    def _estimate_ew_operator(self, lat: float, lon: float) -> str:
        """Estimate likely EW operator based on geographic location."""
        # Known EW activity regions
        if 32 <= lat <= 37 and 34 <= lon <= 42:
            return 'Russia (Syria)'
        elif 29 <= lat <= 33 and 33 <= lon <= 36:
            return 'Unknown (Israel region)'
        elif 35 <= lat <= 42 and 26 <= lon <= 45:
            return 'Turkey/Black Sea'
        elif 24 <= lat <= 32 and 44 <= lon <= 56:
            return 'Iran/Persian Gulf'
        elif 30 <= lat <= 35 and 30 <= lon <= 35:
            return 'Eastern Mediterranean'
        else:
            return 'Unknown'
    
    def get_mission_readiness(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Mission Readiness Assessment - overall situational awareness score.
        
        Returns:
            {
                'readiness_level': 'LOW'|'MODERATE'|'ELEVATED'|'HIGH'|'IMMINENT',
                'factors': [{factor, contribution, value}],
                'trend': 'increasing'|'stable'|'decreasing',
                'alerts': [...]
            }
        """
        # Get operational tempo data
        tempo = self.get_operational_tempo(start_ts, end_ts)
        
        # Calculate readiness score based on various factors
        score = 0
        factors = []
        
        # Factor 1: Total military flights
        total_flights = tempo.get('total_flights', 0)
        if total_flights > 100:
            score += 30
            factors.append({'factor': 'High flight volume', 'contribution': 30, 'value': total_flights})
        elif total_flights > 50:
            score += 20
            factors.append({'factor': 'Moderate flight volume', 'contribution': 20, 'value': total_flights})
        elif total_flights > 20:
            score += 10
            factors.append({'factor': 'Normal flight volume', 'contribution': 10, 'value': total_flights})
        
        # Factor 2: Activity spikes
        spikes = len(tempo.get('activity_spikes', []))
        if spikes > 5:
            score += 25
            factors.append({'factor': 'Multiple activity spikes', 'contribution': 25, 'value': spikes})
        elif spikes > 2:
            score += 15
            factors.append({'factor': 'Some activity spikes', 'contribution': 15, 'value': spikes})
        
        # Factor 3: Russian activity trend
        ru_trend = tempo.get('trend_analysis', {}).get('RU', 'stable')
        if ru_trend == 'increasing':
            score += 25
            factors.append({'factor': 'Russian activity increasing', 'contribution': 25, 'value': ru_trend})
        
        # Determine readiness level
        if score >= 70:
            readiness_level = 'IMMINENT'
        elif score >= 50:
            readiness_level = 'HIGH'
        elif score >= 35:
            readiness_level = 'ELEVATED'
        elif score >= 20:
            readiness_level = 'MODERATE'
        else:
            readiness_level = 'LOW'
        
        alerts = []
        if readiness_level in ('HIGH', 'IMMINENT'):
            alerts.append({
                'type': 'high_readiness',
                'severity': 'critical',
                'message': f"Mission readiness at {readiness_level} level"
            })
        
        return {
            'readiness_level': readiness_level,
            'readiness_score': score,
            'factors': factors,
            'trend': ru_trend,
            'alerts': alerts
        }
    
    def get_gps_jamming_heatmap(self, start_ts: int, end_ts: int, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Detect GPS jamming patterns using multiple signature analysis.
        
        IMPROVED Algorithm based on real GPS jamming case analysis (flight 3cc68900):
        
        GPS Jamming Signatures Detected:
        1. ALTITUDE OSCILLATION - Rapid jumps to/from known spoofed altitudes (34764ft, 44700ft, etc.)
        2. IMPOSSIBLE ALTITUDE RATES - >10,000 ft change in 2-4 seconds (physically impossible)
        3. SPEED ANOMALIES - Ground speed > 600 kts for commercial aircraft
        4. POSITION TELEPORTATION - Implied speed > 600 kts between consecutive points
        5. MLAT-ONLY SOURCE - GPS completely jammed, only multilateration works
        6. CLUSTERED ANOMALIES - Anomalies happen in bursts, not randomly
        7. TRACK GAPS - Traditional signal loss detection
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            limit: Max zones to return (default 30)
        
        Returns:
            [{lat, lon, intensity, jamming_score, jamming_indicators, first_seen, last_seen, event_count, affected_flights, ...}]
        """
        # Use optimized version if available (15-20x faster)
        if self.use_optimized and self._optimized_engine:
            return self._optimized_engine.get_gps_jamming_heatmap_optimized(start_ts, end_ts, limit)
        
        # Legacy implementation below
        # Configuration
        grid_size = 0.25  # ~28km cells for precision
        gap_threshold_seconds = 300  # Signal loss if gap > 5 minutes
        min_altitude_ft = 5000  # Exclude low altitude gaps
        airport_exclusion_nm = 5  # Exclude near airports
        temporal_cluster_window = 1800  # 30 minutes for correlation
        
        # Known GPS spoofing altitude values (common in Middle East jamming)
        SPOOFED_ALTITUDES = {34764, 44700, 44600, 44500, 40000, 40400, 40800, 36864, 42100, 42500, 42700}
        
        # Thresholds for anomaly detection
        ALTITUDE_JUMP_THRESHOLD_FT = 5000  # Altitude jump > 5000ft in short time
        SPEED_THRESHOLD_KTS = 600  # Speed > 600 kts is impossible for commercial
        POSITION_JUMP_THRESHOLD_KTS = 600  # Implied speed > 600 kts
        
        # Major airports to exclude
        airports = {
            'LLBG': (32.0114, 34.8867), 'LLER': (29.9403, 35.0004),
            'LLHA': (32.8094, 35.0431), 'LLOV': (31.2875, 34.7228),
            'OJAI': (31.7226, 35.9932), 'OJAM': (31.9726, 35.9916),
            'OLBA': (33.8209, 35.4884), 'LCRA': (34.5904, 32.9879),
            'LCLK': (34.8751, 33.6249), 'LLSD': (32.1147, 34.7822),
            'HECA': (30.1219, 31.4056), 'HEGN': (27.1783, 33.7994),
            'HESH': (27.9773, 34.3950), 'OMDB': (25.2528, 55.3644),
            'OERK': (24.9576, 46.6988), 'OTHH': (25.2731, 51.6081),
            'LTFM': (41.2753, 28.7519), 'LGAV': (37.9364, 23.9445),
        }
        
        def is_near_airport(lat: float, lon: float) -> bool:
            for icao, (apt_lat, apt_lon) in airports.items():
                if _haversine_nm(lat, lon, apt_lat, apt_lon) <= airport_exclusion_nm:
                    return True
            return False
        
        def calculate_implied_speed_kts(lat1: float, lon1: float, lat2: float, lon2: float, time_diff: int) -> float:
            """Calculate implied speed in knots from position jump."""
            if time_diff <= 0:
                return float('inf')
            dist_nm = _haversine_nm(lat1, lon1, lat2, lon2)
            return (dist_nm / time_diff) * 3600  # Convert to knots
        
        # Grid structure with enhanced jamming detection
        jamming_grid = defaultdict(lambda: {
            'events': [],
            'first_seen': None,
            'last_seen': None,
            'flights': set(),
            'total_gap_duration': 0,
            'lat_sum': 0.0,
            'lon_sum': 0.0,
            'correlated_events': 0,
            # New jamming-specific metrics
            'altitude_jumps': 0,
            'spoofed_altitude_hits': 0,
            'speed_anomalies': 0,
            'position_teleports': 0,
            'mlat_only_flights': set(),
            'anomaly_clusters': 0,
            # NEW: Heading-based jamming metrics
            'impossible_turn_rates': 0,
            'heading_oscillations': 0,
            'track_bearing_mismatches': 0,
        })
        
        # NEW: Heading-based jamming thresholds
        TURN_RATE_THRESHOLD_DEG_S = 8.0  # deg/sec - impossible for aircraft
        
        def _calc_initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate initial bearing in degrees from point 1 to point 2."""
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lon = math.radians(lon2 - lon1)
            
            x = math.sin(delta_lon) * math.cos(lat2_rad)
            y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
            
            bearing = math.atan2(x, y)
            return (math.degrees(bearing) + 360) % 360
        
        # Process flights for jamming signatures
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            # Query with window functions for consecutive point analysis
            # Added track column for heading-based detection
            query = f"""
                WITH ordered_tracks AS (
                    SELECT 
                        flight_id, timestamp, lat, lon, alt, gspeed, source, track,
                        LAG(timestamp) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_ts,
                        LAG(lat) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lat,
                        LAG(lon) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lon,
                        LAG(alt) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_alt,
                        LAG(track) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_track
                    FROM {table_name}
                    WHERE timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL 
                      AND lon IS NOT NULL
                )
                SELECT 
                    flight_id, timestamp, lat, lon, alt, gspeed, source, track,
                    prev_ts, prev_lat, prev_lon, prev_alt, prev_track,
                    (timestamp - prev_ts) as time_diff
                FROM ordered_tracks
                WHERE prev_ts IS NOT NULL
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            
            # Track per-flight jamming indicators
            flight_jamming_data = defaultdict(lambda: {
                'altitude_jumps': 0, 'spoofed_hits': 0, 'speed_anomalies': 0,
                'position_teleports': 0, 'mlat_count': 0, 'total_points': 0,
                'positions': [], 'anomaly_timestamps': [],
                # NEW: Heading-based indicators
                'impossible_turn_rates': 0, 'heading_oscillations': 0, 
                'track_bearing_mismatches': 0, 'prev_heading_change': 0
            })
            
            for row in results:
                (flight_id, timestamp, lat, lon, alt, gspeed, source, track,
                 prev_ts, prev_lat, prev_lon, prev_alt, prev_track, time_diff) = row
                
                if is_near_airport(lat, lon):
                    continue
                
                fd = flight_jamming_data[flight_id]
                fd['total_points'] += 1
                fd['positions'].append((lat, lon))
                
                # Track MLAT source
                if source and source.upper() == 'MLAT':
                    fd['mlat_count'] += 1
                
                # 1. Check for altitude jumps to spoofed values
                if alt is not None and prev_alt is not None and time_diff and time_diff > 0:
                    alt_diff = abs(alt - prev_alt)
                    alt_rate = alt_diff / time_diff  # ft per second
                    
                    # Impossible altitude rate (> 3000 ft/sec is physically impossible)
                    if alt_rate > 3000:
                        fd['altitude_jumps'] += 1
                        fd['anomaly_timestamps'].append(timestamp)
                    
                    # Jump to known spoofed altitude
                    if alt in SPOOFED_ALTITUDES or prev_alt in SPOOFED_ALTITUDES:
                        if alt_diff > ALTITUDE_JUMP_THRESHOLD_FT:
                            fd['spoofed_hits'] += 1
                            fd['anomaly_timestamps'].append(timestamp)
                
                # 2. Check for impossible speed
                if gspeed is not None and gspeed > SPEED_THRESHOLD_KTS:
                    fd['speed_anomalies'] += 1
                    fd['anomaly_timestamps'].append(timestamp)
                
                # 3. Check for position teleportation
                dist_nm = 0
                if prev_lat is not None and prev_lon is not None and time_diff and time_diff > 0:
                    implied_speed = calculate_implied_speed_kts(prev_lat, prev_lon, lat, lon, time_diff)
                    dist_nm = _haversine_nm(prev_lat, prev_lon, lat, lon)
                    if implied_speed > POSITION_JUMP_THRESHOLD_KTS:
                        fd['position_teleports'] += 1
                        fd['anomaly_timestamps'].append(timestamp)
                
                # 4. NEW: Heading-based jamming detection
                if track is not None and prev_track is not None and time_diff and time_diff > 0:
                    # Calculate signed heading change
                    heading_change = ((track - prev_track + 540) % 360) - 180
                    abs_heading_change = abs(heading_change)
                    
                    # 4a. Impossible turn rate (>8 deg/sec)
                    turn_rate = abs_heading_change / time_diff
                    if turn_rate > TURN_RATE_THRESHOLD_DEG_S:
                        fd['impossible_turn_rates'] += 1
                        fd['anomaly_timestamps'].append(timestamp)
                    
                    # 4b. Heading oscillation (rapid back-and-forth)
                    prev_hc = fd['prev_heading_change']
                    if prev_hc != 0:
                        if ((prev_hc > 30 and heading_change < -30) or
                            (prev_hc < -30 and heading_change > 30)):
                            fd['heading_oscillations'] += 1
                            fd['anomaly_timestamps'].append(timestamp)
                    fd['prev_heading_change'] = heading_change
                    
                    # 4c. Track vs actual bearing mismatch
                    if dist_nm > 0.1 and prev_lat is not None and prev_lon is not None:
                        actual_bearing = _calc_initial_bearing(prev_lat, prev_lon, lat, lon)
                        bearing_mismatch = abs(((track - actual_bearing + 540) % 360) - 180)
                        if bearing_mismatch > 90:
                            fd['track_bearing_mismatches'] += 1
                            fd['anomaly_timestamps'].append(timestamp)
            
            # Aggregate flight data into grid
            for flight_id, fd in flight_jamming_data.items():
                if not fd['positions']:
                    continue
                
                # Calculate jamming score for this flight
                total_points = max(fd['total_points'], 1)
                mlat_ratio = fd['mlat_count'] / total_points
                
                # Jamming indicators
                has_altitude_jumps = fd['altitude_jumps'] >= 3
                has_spoofed_altitude = fd['spoofed_hits'] >= 2
                has_speed_anomalies = fd['speed_anomalies'] >= 2
                has_position_teleports = fd['position_teleports'] >= 2
                is_mlat_dominant = mlat_ratio > 0.8
                # NEW: Heading-based indicators
                has_impossible_turns = fd['impossible_turn_rates'] >= 3
                has_heading_oscillations = fd['heading_oscillations'] >= 2
                
                # Calculate per-flight jamming score (0-100)
                flight_jamming_score = 0
                flight_jamming_score += min(20, fd['altitude_jumps'] * 2)  # Up to 20 for altitude jumps
                flight_jamming_score += min(15, fd['spoofed_hits'] * 4)    # Up to 15 for spoofed altitude
                flight_jamming_score += min(15, fd['speed_anomalies'] * 3) # Up to 15 for speed anomalies
                flight_jamming_score += min(15, fd['position_teleports'] * 4)  # Up to 15 for teleports
                flight_jamming_score += 8 if is_mlat_dominant else 0       # 8 for MLAT-only
                # NEW: Heading-based scoring (very reliable indicators)
                flight_jamming_score += min(12, fd['impossible_turn_rates'] * 1)  # Up to 12 for turn rates
                flight_jamming_score += min(10, fd['heading_oscillations'] * 2)   # Up to 10 for oscillations
                flight_jamming_score += min(5, fd['track_bearing_mismatches'] * 1)  # Up to 5 for mismatches
                
                # Only process flights with significant jamming indicators
                if flight_jamming_score < 15:
                    continue
                
                # Get centroid position for this flight
                avg_lat = sum(p[0] for p in fd['positions']) / len(fd['positions'])
                avg_lon = sum(p[1] for p in fd['positions']) / len(fd['positions'])
                
                # Grid the location
                grid_lat = round(avg_lat / grid_size) * grid_size
                grid_lon = round(avg_lon / grid_size) * grid_size
                grid_key = (grid_lat, grid_lon)
                
                cell = jamming_grid[grid_key]
                cell['flights'].add(flight_id)
                cell['lat_sum'] += avg_lat
                cell['lon_sum'] += avg_lon
                
                # Aggregate jamming metrics
                cell['altitude_jumps'] += fd['altitude_jumps']
                cell['spoofed_altitude_hits'] += fd['spoofed_hits']
                cell['speed_anomalies'] += fd['speed_anomalies']
                cell['position_teleports'] += fd['position_teleports']
                # NEW: Aggregate heading-based metrics
                cell['impossible_turn_rates'] += fd['impossible_turn_rates']
                cell['heading_oscillations'] += fd['heading_oscillations']
                cell['track_bearing_mismatches'] += fd['track_bearing_mismatches']
                
                if is_mlat_dominant:
                    cell['mlat_only_flights'].add(flight_id)
                
                # Track anomaly timestamps for clustering
                cell['events'].extend(fd['anomaly_timestamps'])
                
                # Update time bounds
                if fd['anomaly_timestamps']:
                    min_ts = min(fd['anomaly_timestamps'])
                    max_ts = max(fd['anomaly_timestamps'])
                    if cell['first_seen'] is None or min_ts < cell['first_seen']:
                        cell['first_seen'] = min_ts
                    if cell['last_seen'] is None or max_ts > cell['last_seen']:
                        cell['last_seen'] = max_ts
        
        # Also check for traditional track gaps (signal loss)
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                WITH ordered_tracks AS (
                    SELECT 
                        flight_id, timestamp, lat, lon, alt,
                        LAG(timestamp) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_ts,
                        LAG(lat) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lat,
                        LAG(lon) OVER (PARTITION BY flight_id ORDER BY timestamp) as prev_lon
                    FROM {table_name}
                    WHERE timestamp BETWEEN ? AND ?
                      AND lat IS NOT NULL AND lon IS NOT NULL
                )
                SELECT flight_id, prev_ts, timestamp, prev_lat, prev_lon, (timestamp - prev_ts) as gap_seconds
                FROM ordered_tracks
                WHERE prev_ts IS NOT NULL
                  AND (timestamp - prev_ts) > ?
            """
            results = self._execute_query('research', query, (start_ts, end_ts, gap_threshold_seconds))
            
            for row in results:
                flight_id, gap_start, gap_end, lat, lon, gap_seconds = row
                
                if is_near_airport(lat, lon):
                    continue
                
                grid_lat = round(lat / grid_size) * grid_size
                grid_lon = round(lon / grid_size) * grid_size
                grid_key = (grid_lat, grid_lon)
                
                cell = jamming_grid[grid_key]
                cell['events'].append(gap_start)
                cell['flights'].add(flight_id)
                cell['total_gap_duration'] += gap_seconds
                cell['lat_sum'] += lat
                cell['lon_sum'] += lon
                
                if cell['first_seen'] is None or gap_start < cell['first_seen']:
                    cell['first_seen'] = gap_start
                if cell['last_seen'] is None or gap_start > cell['last_seen']:
                    cell['last_seen'] = gap_start
        
        # Detect temporal clustering (multiple anomalies at same time = likely jamming)
        for grid_key, data in jamming_grid.items():
            events = sorted(set(data['events']))  # Deduplicate
            if len(events) >= 2:
                for i, t1 in enumerate(events):
                    for t2 in events[i+1:]:
                        if t2 - t1 <= temporal_cluster_window:
                            data['correlated_events'] += 1
                        else:
                            break
        
        # Format output with enhanced jamming scores
        result = []
        for (grid_lat, grid_lon), data in jamming_grid.items():
            num_flights = len(data['flights'])
            if num_flights == 0:
                continue
            
            num_events = len(set(data['events']))
            
            # Calculate comprehensive jamming score (0-100)
            # Based on learned patterns from flights 3cc68900, 3cc75595, etc.
            
            # Altitude anomaly score (reliable indicator)
            altitude_score = min(25, data['altitude_jumps'] * 2 + data['spoofed_altitude_hits'] * 4)
            
            # Position/speed anomaly score
            motion_score = min(20, data['speed_anomalies'] * 2 + data['position_teleports'] * 3)
            
            # NEW: Heading-based anomaly score (very reliable - from flight analysis)
            heading_score = min(25, 
                data['impossible_turn_rates'] * 1 + 
                data['heading_oscillations'] * 2 + 
                data['track_bearing_mismatches'] * 1
            )
            
            # MLAT dominance score (GPS completely jammed)
            mlat_ratio = len(data['mlat_only_flights']) / max(num_flights, 1)
            mlat_score = int(mlat_ratio * 10)
            
            # Correlation score (multiple aircraft affected)
            correlation_score = min(10, data['correlated_events'] * 2)
            
            # Multi-flight score (more flights = more confidence)
            flight_score = min(10, num_flights * 3)
            
            jamming_score = altitude_score + motion_score + heading_score + mlat_score + correlation_score + flight_score
            
            # Determine jamming confidence level
            if jamming_score >= 60:
                confidence = 'HIGH'
            elif jamming_score >= 35:
                confidence = 'MEDIUM'
            elif jamming_score >= 15:
                confidence = 'LOW'
            else:
                confidence = 'UNLIKELY'
            
            # Build jamming indicators list
            indicators = []
            if data['altitude_jumps'] > 0:
                indicators.append(f"altitude_oscillation:{data['altitude_jumps']}")
            if data['spoofed_altitude_hits'] > 0:
                indicators.append(f"spoofed_altitude:{data['spoofed_altitude_hits']}")
            if data['speed_anomalies'] > 0:
                indicators.append(f"speed_anomaly:{data['speed_anomalies']}")
            if data['position_teleports'] > 0:
                indicators.append(f"position_teleport:{data['position_teleports']}")
            # NEW: Heading-based indicators
            if data['impossible_turn_rates'] > 0:
                indicators.append(f"impossible_turn_rate:{data['impossible_turn_rates']}")
            if data['heading_oscillations'] > 0:
                indicators.append(f"heading_oscillation:{data['heading_oscillations']}")
            if data['track_bearing_mismatches'] > 0:
                indicators.append(f"track_bearing_mismatch:{data['track_bearing_mismatches']}")
            if len(data['mlat_only_flights']) > 0:
                indicators.append(f"mlat_only:{len(data['mlat_only_flights'])}")
            if data['correlated_events'] > 0:
                indicators.append(f"correlated:{data['correlated_events']}")
            if data['total_gap_duration'] > 0:
                indicators.append(f"signal_loss:{int(data['total_gap_duration'])}s")
            
            # Use centroid of actual event positions
            if data['lat_sum'] > 0 and data['lon_sum'] > 0:
                divisor = max(num_flights, 1)
                centroid_lat = data['lat_sum'] / divisor
                centroid_lon = data['lon_sum'] / divisor
            else:
                centroid_lat = grid_lat
                centroid_lon = grid_lon
            
            result.append({
                'lat': round(centroid_lat, 4),
                'lon': round(centroid_lon, 4),
                'intensity': jamming_score,  # Use jamming_score as intensity for compatibility
                'jamming_score': jamming_score,
                'jamming_confidence': confidence,
                'jamming_indicators': indicators,
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen'],
                'event_count': num_events,
                'affected_flights': num_flights,
                'avg_gap_duration_s': int(data['total_gap_duration'] / max(num_events, 1)) if data['total_gap_duration'] > 0 else 0,
                'correlated_events': data['correlated_events'],
                'altitude_anomalies': data['altitude_jumps'] + data['spoofed_altitude_hits'],
                'motion_anomalies': data['speed_anomalies'] + data['position_teleports'],
                # NEW: Heading anomalies
                'heading_anomalies': data['impossible_turn_rates'] + data['heading_oscillations'] + data['track_bearing_mismatches'],
                'mlat_only_flights': len(data['mlat_only_flights']),
                'likely_jamming': confidence in ('HIGH', 'MEDIUM')
            })
        
        # Sort by jamming score and limit
        sorted_result = sorted(result, key=lambda x: x['jamming_score'], reverse=True)
        return sorted_result[:limit]
    
    def get_signal_loss_zones(self, start_ts: int, end_ts: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Detect signal loss zones based on 5-minute time gaps between flight track points.
        
        This is a primary indicator of GPS jamming - when a flight's transponder
        stops reporting for 5+ minutes and then resumes, it suggests signal interference.
        
        Uses optimized pandas/numpy engine if available for better performance.
        
        Returns:
            List of signal loss zones with counts, affected flights, and gap statistics
        """
        # Use optimized version if available
        if self.use_optimized and self._optimized_engine:
            return self._optimized_engine.get_signal_loss_zones(start_ts, end_ts, limit)
        
        # Fallback to basic implementation
        return []
    
    def get_gps_jamming_clusters(self, start_ts: int, end_ts: int,
                                  cluster_threshold_nm: float = 50,
                                  min_points_for_polygon: int = 3) -> Dict[str, Any]:
        """
        Get GPS jamming clusters with polygon (convex hull) boundaries.
        
        Clusters GPS jamming points that are within cluster_threshold_nm of each other.
        For clusters with 3+ points, computes convex hull polygon coordinates.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            cluster_threshold_nm: Distance threshold for clustering in nautical miles
            min_points_for_polygon: Minimum points required to form a polygon
        
        Returns:
            Dict with 'clusters', 'singles', 'total_points', 'total_clusters'
        """
        # Use optimized version if available
        if self.use_optimized and self._optimized_engine:
            # Pass cached GPS jamming data if available (saves ~170s redundant query!)
            cached_data = getattr(self, '_cached_gps_jamming_extended', None)
            return self._optimized_engine.get_gps_jamming_clusters(
                start_ts, end_ts, cluster_threshold_nm, min_points_for_polygon,
                cached_jamming_data=cached_data
            )
        
        # Fallback: return empty structure
        return {
            'clusters': [],
            'singles': [],
            'total_points': 0,
            'total_clusters': 0
        }
    
    def get_gps_jamming_zones(self, start_ts: int, end_ts: int,
                               buffer_radius_nm: float = 15,
                               min_events_for_zone: int = 3,
                               merge_threshold_nm: float = 25,
                               limit: int = 30,
                               max_zone_radius_nm: float = 100) -> Dict[str, Any]:
        """
        Compute GPS JAMMING ZONES - estimated areas where GPS signal is compromised.
        
        IMPROVED ALGORITHM: Creates tighter, more meaningful zones instead of one giant polygon.
        - Uses DBSCAN-style density clustering with stricter thresholds
        - Limits maximum zone size to prevent mega-clusters
        - Splits large clusters into sub-zones based on density
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            buffer_radius_nm: Base buffer radius around jamming points (default 15nm - tighter)
            min_events_for_zone: Minimum events to form a zone (default 3)
            merge_threshold_nm: Distance to merge nearby points (default 25nm - stricter)
            limit: Maximum zones to return (default 30)
            max_zone_radius_nm: Maximum radius for a single zone (default 100nm)
        
        Returns:
            Dict with zones, total_events, total_zones, jamming_summary
        """
        import time as time_module
        
        start_time = time_module.perf_counter()
        print(f"[GPS_JAMMING_ZONES] Computing jamming zones with buffer={buffer_radius_nm}nm, merge={merge_threshold_nm}nm")
        
        # Get GPS jamming heatmap data (uses cached if available)
        if hasattr(self, '_cached_gps_jamming_extended') and self._cached_gps_jamming_extended:
            jamming_points = self._cached_gps_jamming_extended
        else:
            jamming_points = self.get_gps_jamming_heatmap(start_ts, end_ts, 200)
        
        if not jamming_points:
            return {
                'zones': [],
                'total_events': 0,
                'total_zones': 0,
                'jamming_summary': {
                    'total_jamming_area_sq_nm': 0,
                    'avg_jamming_score': 0,
                    'primary_type': 'none',
                    'hotspot_regions': []
                }
            }
        
        print(f"[GPS_JAMMING_ZONES] Found {len(jamming_points)} jamming points to cluster")
        
        import numpy as np
        from scipy.spatial import cKDTree
        from collections import defaultdict
        
        n_events = len(jamming_points)
        coords = np.array([[jp['lat'], jp['lon']] for jp in jamming_points])
        
        # IMPROVED: Use stricter, fixed threshold for initial clustering
        # Don't scale buffer by score - keep zones tight
        threshold_deg = merge_threshold_nm / 60.0  # Stricter: just merge_threshold, no buffer multiplication
        
        tree = cKDTree(coords)
        
        # Union-Find for clustering
        parent = list(range(n_events))
        rank = [0] * n_events
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
        
        # Find pairs within threshold
        pairs = tree.query_pairs(r=threshold_deg, output_type='ndarray')
        for i, j in pairs:
            union(i, j)
        
        # Group by cluster
        clusters_dict = defaultdict(list)
        for i in range(n_events):
            clusters_dict[find(i)].append(i)
        
        # IMPROVED: Split large clusters into sub-zones
        raw_zones = []
        max_radius_deg = max_zone_radius_nm / 60.0
        
        for indices in clusters_dict.values():
            if len(indices) < min_events_for_zone:
                continue
            
            cluster_coords = coords[indices]
            
            # Check if cluster is too large (spans too much area)
            lat_span = np.max(cluster_coords[:, 0]) - np.min(cluster_coords[:, 0])
            lon_span = np.max(cluster_coords[:, 1]) - np.min(cluster_coords[:, 1])
            max_span = max(lat_span, lon_span)
            
            if max_span > max_radius_deg * 2:
                # Cluster too large - split using k-means
                from scipy.cluster.vq import kmeans2
                n_subclusters = max(2, int(max_span / max_radius_deg))
                n_subclusters = min(n_subclusters, len(indices) // min_events_for_zone)
                
                if n_subclusters >= 2:
                    try:
                        _, labels = kmeans2(cluster_coords, n_subclusters, minit='++')
                        for label in range(n_subclusters):
                            sub_indices = [indices[i] for i in range(len(indices)) if labels[i] == label]
                            if len(sub_indices) >= min_events_for_zone:
                                raw_zones.append(sub_indices)
                    except Exception:
                        # Fallback: keep original cluster
                        raw_zones.append(indices)
                else:
                    raw_zones.append(indices)
            else:
                raw_zones.append(indices)
        
        print(f"[GPS_JAMMING_ZONES] Created {len(raw_zones)} zones (after splitting large clusters)")
        
        # Create zone polygons and statistics
        from scipy.spatial import ConvexHull
        zones = []
        total_area = 0
        
        # Helper to count indicators from either list or dict format
        def count_indicator(jp, keywords):
            """Count indicator from either list-of-strings or dict format."""
            ind = jp.get('jamming_indicators', [])
            if isinstance(ind, dict):
                return sum(ind.get(k, 0) for k in keywords)
            elif isinstance(ind, list):
                return sum(1 for s in ind if any(k.lower().replace('_', ' ') in s.lower() for k in keywords))
            return 0
        
        for zone_idx, indices in enumerate(raw_zones):
            zone_coords = coords[indices]
            zone_points = [jamming_points[i] for i in indices]
            
            # Centroid
            centroid_lat = float(np.mean(zone_coords[:, 0]))
            centroid_lon = float(np.mean(zone_coords[:, 1]))
            
            # Aggregate statistics
            total_event_count = sum(jp.get('event_count', 1) for jp in zone_points)
            total_affected_flights = sum(jp.get('affected_flights', 1) for jp in zone_points)
            avg_jamming_score = float(np.mean([jp.get('jamming_score', 50) for jp in zone_points]))
            
            # Aggregate indicators
            indicators = {
                'altitude_spikes': sum(count_indicator(jp, ['altitude_spikes', 'altitude jumps']) for jp in zone_points),
                'position_teleports': sum(count_indicator(jp, ['position_teleports', 'position teleports']) for jp in zone_points),
                'heading_anomalies': sum(count_indicator(jp, ['heading_anomalies', 'impossible_turn_rates', 'impossible turns']) for jp in zone_points),
                'mlat_only': sum(count_indicator(jp, ['mlat_dominant', 'mlat']) for jp in zone_points)
            }
            
            # Determine jamming type based on dominant indicator
            if indicators['altitude_spikes'] > indicators['position_teleports'] and indicators['altitude_spikes'] > 0:
                jamming_type = 'spoofing'
            elif indicators['mlat_only'] > total_event_count * 0.3:
                jamming_type = 'denial'
            else:
                jamming_type = 'mixed'
            
            # Confidence level
            confidences = [jp.get('jamming_confidence', 'MEDIUM') for jp in zone_points]
            confidence = max(set(confidences), key=confidences.count) if confidences else 'MEDIUM'
            
            # Timestamps
            timestamps_first = [jp.get('first_seen') for jp in zone_points if jp.get('first_seen')]
            timestamps_last = [jp.get('last_seen') for jp in zone_points if jp.get('last_seen')]
            
            # IMPROVED: Calculate buffer based on zone density, not arbitrary scaling
            # Tighter zones = smaller buffer, spread out = larger buffer
            lat_spread = np.max(zone_coords[:, 0]) - np.min(zone_coords[:, 0])
            lon_spread = np.max(zone_coords[:, 1]) - np.min(zone_coords[:, 1])
            zone_spread_nm = max(lat_spread, lon_spread) * 60  # Convert to nm
            
            # Buffer scales with jamming score but is capped
            score_factor = 1.0 + (avg_jamming_score / 100) * 0.5  # 1.0 to 1.5x
            zone_buffer = min(buffer_radius_nm * score_factor, 30)  # Cap at 30nm
            
            # Create polygon
            polygon = None
            area_sq_nm = 0
            
            try:
                if len(indices) >= 3:
                    hull_coords = np.column_stack([zone_coords[:, 1], zone_coords[:, 0]])  # lon, lat
                    hull = ConvexHull(hull_coords)
                    hull_points = hull_coords[hull.vertices].tolist()
                    
                    # Expand hull by buffer
                    expanded = self._expand_polygon_points(hull_points, centroid_lon, centroid_lat, zone_buffer)
                    expanded.append(expanded[0])  # Close polygon
                    polygon = expanded
                    
                    # Calculate area
                    lons = [p[0] for p in expanded]
                    lats = [p[1] for p in expanded]
                    area_sq_nm = (max(lons) - min(lons)) * 60 * (max(lats) - min(lats)) * 60 * 0.7
                else:
                    # Less than 3 points - create circle
                    polygon = self._generate_circle_polygon(centroid_lon, centroid_lat, zone_buffer, 16)
                    area_sq_nm = 3.14159 * zone_buffer * zone_buffer
            except Exception:
                # Fallback to circle
                polygon = self._generate_circle_polygon(centroid_lon, centroid_lat, zone_buffer, 16)
                area_sq_nm = 3.14159 * zone_buffer * zone_buffer
            
            total_area += area_sq_nm
            
            zones.append({
                'id': zone_idx,
                'polygon': polygon,
                'centroid': [centroid_lon, centroid_lat],
                'area_sq_nm': round(area_sq_nm, 1),
                'event_count': total_event_count,
                'affected_flights': total_affected_flights,
                'jamming_score': round(avg_jamming_score, 1),
                'jamming_type': jamming_type,
                'indicators': indicators,
                'first_seen': min(timestamps_first) if timestamps_first else None,
                'last_seen': max(timestamps_last) if timestamps_last else None,
                'confidence': confidence,
                'points': [{
                    'lat': jp['lat'], 'lon': jp['lon'],
                    'jamming_score': jp.get('jamming_score', 50),
                    'event_count': jp.get('event_count', 1)
                } for jp in zone_points[:12]]
            })
        
        # Sort by jamming score
        zones.sort(key=lambda z: z['jamming_score'], reverse=True)
        zones = zones[:limit]
        
        # Identify hotspot regions
        hotspot_regions = self._identify_jamming_regions(zones)
        
        # Determine primary jamming type
        type_counts = Counter(z['jamming_type'] for z in zones)
        primary_type = type_counts.most_common(1)[0][0] if type_counts else 'none'
        
        avg_score = sum(z['jamming_score'] for z in zones) / len(zones) if zones else 0
        
        total_time = time_module.perf_counter() - start_time
        print(f"[GPS_JAMMING_ZONES] Created {len(zones)} jamming zones in {total_time:.2f}s")
        
        return {
            'zones': zones,
            'total_events': sum(z['event_count'] for z in zones),
            'total_zones': len(zones),
            'jamming_summary': {
                'total_jamming_area_sq_nm': round(total_area, 1),
                'avg_jamming_score': round(avg_score, 1),
                'primary_type': primary_type,
                'hotspot_regions': hotspot_regions
            }
        }
    
    def _expand_polygon_points(self, hull_points: List[List[float]], centroid_lon: float, 
                                centroid_lat: float, buffer_nm: float) -> List[List[float]]:
        """Expand polygon points outward from centroid."""
        expanded = []
        buffer_deg = buffer_nm / 60.0
        
        for lon, lat in hull_points:
            delta_lon = lon - centroid_lon
            delta_lat = lat - centroid_lat
            dist = math.sqrt(delta_lon**2 + delta_lat**2)
            
            if dist == 0:
                expanded.append([lon, lat])
                continue
            
            scale = 1 + (buffer_deg / dist)
            expanded.append([
                centroid_lon + delta_lon * scale,
                centroid_lat + delta_lat * scale
            ])
        
        return expanded
    
    def _generate_circle_polygon(self, center_lon: float, center_lat: float, 
                                  radius_nm: float, num_points: int = 24) -> List[List[float]]:
        """Generate circular polygon."""
        radius_deg = radius_nm / 60.0
        points = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            lat = center_lat + radius_deg * math.cos(angle)
            lon = center_lon + (radius_deg / math.cos(math.radians(center_lat))) * math.sin(angle)
            points.append([lon, lat])
        
        points.append(points[0])
        return points
    
    def _identify_jamming_regions(self, zones: List[Dict[str, Any]]) -> List[str]:
        """Identify named regions with GPS jamming."""
        region_bounds = {
            'Syria': {'lat_min': 32.5, 'lat_max': 37.5, 'lon_min': 35.5, 'lon_max': 42.5},
            'Lebanon': {'lat_min': 33.0, 'lat_max': 34.7, 'lon_min': 35.0, 'lon_max': 36.7},
            'Eastern Mediterranean': {'lat_min': 32.0, 'lat_max': 37.0, 'lon_min': 30.0, 'lon_max': 36.0},
            'Northern Israel': {'lat_min': 32.5, 'lat_max': 33.5, 'lon_min': 34.5, 'lon_max': 36.0},
            'Gaza Border': {'lat_min': 31.0, 'lat_max': 32.0, 'lon_min': 34.0, 'lon_max': 35.0},
            'Sinai': {'lat_min': 28.0, 'lat_max': 31.5, 'lon_min': 32.5, 'lon_max': 35.0},
            'Cyprus': {'lat_min': 34.5, 'lat_max': 35.7, 'lon_min': 32.0, 'lon_max': 34.5},
            'Iraq Border': {'lat_min': 33.0, 'lat_max': 37.0, 'lon_min': 38.0, 'lon_max': 44.0},
        }
        
        region_scores = defaultdict(float)
        
        for zone in zones:
            lon, lat = zone['centroid']
            score = zone['jamming_score']
            
            for region_name, bounds in region_bounds.items():
                if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                    bounds['lon_min'] <= lon <= bounds['lon_max']):
                    region_scores[region_name] += score
        
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_regions[:3]]
    
    def get_military_patterns(self, start_ts: int, end_ts: int,
                             country: Optional[str] = None,
                             aircraft_type: Optional[str] = None,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Track military aircraft patterns.
        
        Identifies military aircraft by:
        1. Callsign patterns (RCH, REACH, IAF, RAF, etc.)
        2. category='Military_and_government' from FR24 data
        3. is_military=1 flag
        
        Args:
            country: Filter by country (e.g., "US", "RU", "GB", "IL", "NATO")
            aircraft_type: Filter by type (e.g., "tanker", "ISR", "transport", "fighter")
            limit: Max number of patterns to return (default None = no limit)
        
        Returns:
            [{flight_id, callsign, country, type, pattern_type, locations: [...], frequency}]
        """
        patterns = []
        
        # Build callsign pattern query dynamically using central MILITARY_PREFIXES
        # This ensures consistency with core.military_detection module
        callsign_prefixes = MILITARY_PREFIXES if MILITARY_PREFIXES else list(MILITARY_CALLSIGN_PATTERNS.keys())
        like_clauses = ' OR '.join([f"callsign LIKE '{prefix}%'" for prefix in callsign_prefixes])
        
        # Query flights with military callsign patterns
        all_results = []
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT DISTINCT flight_id, callsign
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                AND ({like_clauses})
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            all_results.extend(results)
        
        # Also query flight_metadata for category='Military_and_government' or is_military=1
        metadata_query = """
            SELECT DISTINCT flight_id, callsign
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
        """
        metadata_results = self._execute_query('research', metadata_query, (start_ts, end_ts))
        all_results.extend(metadata_results)
        
        # Remove duplicates
        seen_flights = set()
        unique_results = []
        for row in all_results:
            if row[0] not in seen_flights:
                seen_flights.add(row[0])
                unique_results.append(row)
        
        for row in unique_results:
            flight_id, callsign = row
            if not callsign:
                continue
            
            # Identify aircraft type and country from callsign
            mil_info = self._identify_military_aircraft(callsign)
            if not mil_info:
                continue
            
            # Apply filters
            if country and mil_info['country'] != country:
                continue
            if aircraft_type and mil_info['type'] != aircraft_type:
                continue
            
            # Get track data for pattern analysis and locations
            track_data = self._get_flight_track(flight_id, start_ts, end_ts)
            
            # Analyze flight pattern with enhanced ISR/tanker detection
            pattern_analysis = self._analyze_flight_pattern(track_data)
            
            # Extract key locations (sampled points)
            locations = self._extract_key_locations(track_data)
            
            # Determine final aircraft type (callsign-based or inferred from behavior)
            aircraft_type = mil_info['type']
            inferred_type = pattern_analysis.get('inferred_role')
            
            # If callsign type is generic ('military', 'unknown'), use inferred type
            if aircraft_type in ('military', 'unknown') and inferred_type:
                aircraft_type = inferred_type
            
            patterns.append({
                'flight_id': flight_id,
                'callsign': callsign,
                'country': mil_info['country'],
                'type': aircraft_type,
                'type_name': mil_info.get('name', ''),
                'pattern_type': pattern_analysis.get('pattern_type', 'unknown'),
                'inferred_role': inferred_type,
                'loiter_time_min': pattern_analysis.get('loiter_time_min', 0),
                'pattern_length_nm': pattern_analysis.get('pattern_length_nm', 0),
                'offshore': pattern_analysis.get('offshore', False),
                'locations': locations,
                'frequency': 1,
                'track_points': len(track_data)
            })
        
        # Sort by country then type
        patterns.sort(key=lambda x: (x['country'], x['type']))
        
        # Apply limit if specified
        if limit is not None:
            return patterns[:limit]
        return patterns
    
    def _identify_military_aircraft(self, callsign: str, registration: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Identify military aircraft type and country from callsign and/or registration.
        
        Now uses the common military detection function from core.military_detection.
        
        Args:
            callsign: Aircraft callsign
            registration: Aircraft registration (optional)
        
        Returns:
            {'country': str, 'type': str, 'name': str} or None
        """
        if not callsign and not registration:
            return None
        
        # Try using the common military detection function first
        if core_is_military is not None:
            is_mil, mil_info = core_is_military(callsign=callsign, aircraft_registration=registration)
            
            if is_mil and mil_info:
                # Parse mil_info to extract country and type
                info_lower = mil_info.lower()
                
                # Extract country
                if "us " in info_lower or "usaf" in info_lower or "us air force" in info_lower or "us navy" in info_lower or "us marine" in info_lower or "us army" in info_lower:
                    country = "US"
                elif "united kingdom" in info_lower or "raf" in info_lower or "royal air force" in info_lower or "royal navy" in info_lower:
                    country = "GB"
                elif "german" in info_lower or "luftwaffe" in info_lower:
                    country = "DE"
                elif "french" in info_lower:
                    country = "FR"
                elif "belgian" in info_lower:
                    country = "BE"
                elif "spanish" in info_lower:
                    country = "ES"
                elif "polish" in info_lower:
                    country = "PL"
                elif "danish" in info_lower:
                    country = "DK"
                elif "australian" in info_lower:
                    country = "AU"
                elif "canadian" in info_lower:
                    country = "CA"
                elif "israel" in info_lower or "idf" in info_lower:
                    country = "IL"
                elif "colombia" in info_lower:
                    country = "CO"
                elif "honduras" in info_lower:
                    country = "HN"
                elif "italy" in info_lower:
                    country = "IT"
                elif "jordan" in info_lower or "jordanian" in info_lower:
                    country = "JO"
                else:
                    country = "UNKNOWN"
                
                # Extract type
                if "transport" in info_lower or "mobility" in info_lower or "convoy" in info_lower:
                    mil_type = "transport"
                elif "tanker" in info_lower or "refuel" in info_lower:
                    mil_type = "tanker"
                elif "fighter" in info_lower:
                    mil_type = "fighter"
                elif "isr" in info_lower or "recce" in info_lower or "hawk" in info_lower or "rc-135" in info_lower:
                    mil_type = "ISR"
                elif "medical" in info_lower or "evac" in info_lower:
                    mil_type = "medical"
                elif "vip" in info_lower or "special air mission" in info_lower:
                    mil_type = "vip"
                elif "helicopter" in info_lower:
                    mil_type = "helicopter"
                else:
                    mil_type = "military"
                
                return {
                    'country': country,
                    'type': mil_type,
                    'name': mil_info
                }
        
        # Fallback to legacy patterns if common function not available
        if callsign:
            callsign_upper = callsign.upper()
            
            # Check against known military callsign patterns
            for prefix, info in MILITARY_CALLSIGN_PATTERNS.items():
                if callsign_upper.startswith(prefix):
                    return info.copy()
            
            # Additional heuristics for unrecognized patterns
            # Numeric-heavy callsigns with certain characteristics may be military
            if len(callsign) >= 5:
                # Check for all-caps with numbers pattern common in military
                alpha_count = sum(1 for c in callsign if c.isalpha())
                digit_count = sum(1 for c in callsign if c.isdigit())
                
                if digit_count >= 3 and alpha_count <= 3:
                    # Likely military or government
                    return {'country': 'UNKNOWN', 'type': 'unknown', 'name': 'Unidentified Military'}
        
        return None
    
    def _get_flight_track(self, flight_id: str, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """
        Get track points for a flight.
        
        Returns:
            List of {timestamp, lat, lon, alt, track, gspeed}
        """
        track_data = []
        
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT timestamp, lat, lon, alt, track, gspeed
                FROM {table_name}
                WHERE flight_id = ?
                  AND timestamp BETWEEN ? AND ?
                  AND lat IS NOT NULL AND lon IS NOT NULL
                ORDER BY timestamp
                LIMIT 500
            """
            results = self._execute_query('research', query, (flight_id, start_ts, end_ts))
            
            for row in results:
                ts, lat, lon, alt, heading, gspeed = row
                track_data.append({
                    'timestamp': ts,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt or 0,
                    'track': heading,
                    'gspeed': gspeed or 0
                })
            
            if track_data:
                break  # Found data, no need to check other table
        
        return track_data
    
    def get_military_flights_with_tracks(self, start_ts: int, end_ts: int, 
                                          flights_per_country: int = 30) -> Dict[str, Any]:
        """
        Get the last N military flights per country with their full tracks.
        
        This is designed for a simple map visualization showing recent military activity
        by country with flight paths.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            flights_per_country: Max number of flights per country (default 30)
        
        Returns:
            {
                'flights': [
                    {
                        'flight_id': str,
                        'callsign': str,
                        'country': str,
                        'type': str,  # tanker, ISR, transport, fighter, etc.
                        'type_name': str,  # Full description
                        'first_seen': int,  # timestamp
                        'track': [[lon, lat], [lon, lat], ...]  # GeoJSON-compatible coords
                    }
                ],
                'by_country': {country: count},
                'total_flights': int,
                'countries': [str]  # List of unique countries
            }
        """
        from datetime import datetime

        # Step 1: Get all military flight metadata, ordered by most recent first
        query = """
            SELECT flight_id, callsign, first_seen_ts, category, is_military
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
            ORDER BY first_seen_ts DESC
        """
        metadata_results = self._execute_query('research', query, (start_ts, end_ts))
        
        # Also get flights by callsign pattern (some may not have is_military flag set)
        callsign_prefixes = MILITARY_PREFIXES if MILITARY_PREFIXES else list(MILITARY_CALLSIGN_PATTERNS.keys())
        like_clauses = ' OR '.join([f"callsign LIKE '{prefix}%'" for prefix in callsign_prefixes])
        
        # Query distinct flights from track tables
        all_callsign_results = []
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT DISTINCT flight_id, callsign, MIN(timestamp) as first_seen
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                AND ({like_clauses})
                GROUP BY flight_id, callsign
                ORDER BY first_seen DESC
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
            all_callsign_results.extend(results)
        
        # Combine results, removing duplicates (prefer metadata results)
        seen_flight_ids = set()
        combined_flights = []
        
        # Add metadata results first
        for row in metadata_results:
            flight_id, callsign, first_seen_ts, category, is_mil = row
            if flight_id not in seen_flight_ids and callsign:
                seen_flight_ids.add(flight_id)
                combined_flights.append({
                    'flight_id': flight_id,
                    'callsign': callsign,
                    'first_seen': first_seen_ts
                })
        
        # Add callsign-pattern results that weren't in metadata
        for row in all_callsign_results:
            flight_id, callsign, first_seen = row
            if flight_id not in seen_flight_ids and callsign:
                seen_flight_ids.add(flight_id)
                combined_flights.append({
                    'flight_id': flight_id,
                    'callsign': callsign,
                    'first_seen': first_seen
                })
        
        # Sort by first_seen descending (most recent first)
        combined_flights.sort(key=lambda x: x['first_seen'] or 0, reverse=True)
        
        # Step 2: Group by country and take only flights_per_country per country
        flights_by_country = defaultdict(list)
        country_counts = defaultdict(int)
        
        for flight in combined_flights:
            callsign = flight['callsign']
            
            # Identify country from callsign
            mil_info = self._identify_military_aircraft(callsign)
            if not mil_info:
                continue
            
            country = mil_info.get('country', 'UNKNOWN')
            
            # Skip if we already have enough flights for this country
            if len(flights_by_country[country]) >= flights_per_country:
                country_counts[country] += 1  # Still count for stats
                continue
            
            country_counts[country] += 1
            
            flights_by_country[country].append({
                **flight,
                'country': country,
                'type': mil_info.get('type', 'military'),
                'type_name': mil_info.get('name', '')
            })
        
        # Step 3: Fetch tracks for selected flights
        final_flights = []
        
        for country, flights in flights_by_country.items():
            for flight in flights:
                flight_id = flight['flight_id']
                
                # Get track data
                track_data = self._get_flight_track(flight_id, start_ts, end_ts)
                
                if len(track_data) < 2:
                    continue  # Skip flights with insufficient track data
                
                # Convert to GeoJSON-compatible format [lon, lat] and sample to max 100 points
                raw_coords = [[p['lon'], p['lat']] for p in track_data 
                              if p.get('lon') is not None and p.get('lat') is not None]
                
                # Sample if too many points (keep start, end, and evenly spaced middle)
                max_points = 100
                if len(raw_coords) > max_points:
                    step = len(raw_coords) / max_points
                    track_coords = [raw_coords[int(i * step)] for i in range(max_points - 1)]
                    track_coords.append(raw_coords[-1])  # Always include last point
                else:
                    track_coords = raw_coords
                
                final_flights.append({
                    'flight_id': flight_id,
                    'callsign': flight['callsign'],
                    'country': country,
                    'type': flight['type'],
                    'type_name': flight['type_name'],
                    'first_seen': flight['first_seen'],
                    'track': track_coords,
                    'track_points': len(track_coords)
                })
        
        # Build summary
        countries = list(set(f['country'] for f in final_flights))
        countries.sort()
        
        return {
            'flights': final_flights,
            'by_country': dict(country_counts),
            'total_flights': len(final_flights),
            'countries': countries
        }
    
    def _analyze_flight_pattern(self, track_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze track data to determine flight pattern type with enhanced ISR/tanker detection.
        
        Returns:
            {
                'pattern_type': 'racetrack', 'orbit', 'transit', 'loiter', or 'unknown',
                'inferred_role': 'ISR', 'tanker', 'transport', 'fighter', or None,
                'loiter_time_min': float,
                'pattern_length_nm': float,
                'offshore': bool
            }
        """
        result = {
            'pattern_type': 'unknown',
            'inferred_role': None,
            'loiter_time_min': 0,
            'pattern_length_nm': 0,
            'offshore': False
        }
        
        if len(track_data) < 10:
            return result
        
        # Calculate total heading change
        total_heading_change = 0
        heading_changes = []
        
        for i in range(1, len(track_data)):
            prev_hdg = track_data[i-1].get('track')
            curr_hdg = track_data[i].get('track')
            
            if prev_hdg is not None and curr_hdg is not None:
                # Calculate signed heading change
                delta = ((curr_hdg - prev_hdg + 540) % 360) - 180
                heading_changes.append(delta)
                total_heading_change += delta
        
        if not heading_changes:
            result['pattern_type'] = 'transit'
            return result
        
        # Analyze the pattern
        abs_total = abs(total_heading_change)
        
        # Count significant turns (> 45 degrees accumulated)
        significant_turns = 0
        accumulated = 0
        for delta in heading_changes:
            accumulated += delta
            if abs(accumulated) >= 45:
                significant_turns += 1
                accumulated = 0
        
        # Calculate distance traveled vs displacement
        start = track_data[0]
        end = track_data[-1]
        displacement = _haversine_nm(start['lat'], start['lon'], end['lat'], end['lon'])
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(track_data)):
            p1, p2 = track_data[i-1], track_data[i]
            total_distance += _haversine_nm(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        
        result['pattern_length_nm'] = round(total_distance, 1)
        
        # Ratio of displacement to total distance
        efficiency = displacement / total_distance if total_distance > 0 else 1.0
        
        # Calculate loiter time (time in the pattern area)
        if track_data:
            start_ts = track_data[0].get('timestamp', 0)
            end_ts = track_data[-1].get('timestamp', 0)
            result['loiter_time_min'] = round((end_ts - start_ts) / 60, 1) if end_ts > start_ts else 0
        
        # Check if offshore (over water - simplified check for Mediterranean/coastal)
        avg_lat = sum(p['lat'] for p in track_data) / len(track_data)
        avg_lon = sum(p['lon'] for p in track_data) / len(track_data)
        
        # Check if in any tanker holding area
        for area in TANKER_HOLDING_AREAS:
            if _haversine_nm(avg_lat, avg_lon, area['lat'], area['lon']) < area['radius_nm']:
                result['offshore'] = True
                break
        
        # Pattern classification
        # Orbit: Multiple 360-degree turns, returns near start
        if abs_total >= 300 and efficiency < 0.3:
            result['pattern_type'] = 'orbit'
            # Orbiting patterns offshore often indicate tanker operations
            if result['offshore'] and result['loiter_time_min'] > 60:
                result['inferred_role'] = 'tanker'
        
        # Racetrack: 180-degree turns (back and forth pattern)
        elif significant_turns >= 2 and 0.2 < efficiency < 0.6:
            result['pattern_type'] = 'racetrack'
            # Long racetrack patterns are typical ISR behavior
            if result['loiter_time_min'] > 120 or total_distance > 200:
                result['inferred_role'] = 'ISR'
        
        # Loiter: Staying in small area for extended time
        elif efficiency < 0.4 and result['loiter_time_min'] > 60:
            result['pattern_type'] = 'loiter'
            if result['offshore']:
                result['inferred_role'] = 'tanker'
            else:
                result['inferred_role'] = 'ISR'
        
        # Transit: Relatively straight path
        elif efficiency > 0.7 or abs_total < 90:
            result['pattern_type'] = 'transit'
            result['inferred_role'] = 'transport'
        
        # Mixed or complex pattern
        elif significant_turns >= 3:
            result['pattern_type'] = 'racetrack'
            result['inferred_role'] = 'ISR'
        
        else:
            result['pattern_type'] = 'transit'
        
        return result
    
    def _extract_key_locations(self, track_data: List[Dict[str, Any]], max_points: int = 5) -> List[Dict[str, Any]]:
        """
        Extract key locations from track data.
        
        Returns up to max_points evenly sampled locations WITH timestamps.
        Timestamps are critical for bilateral proximity detection.
        """
        if not track_data:
            return []
        
        locations = []
        
        # Sample evenly from the track
        if len(track_data) <= max_points:
            indices = range(len(track_data))
        else:
            step = len(track_data) / max_points
            indices = [int(i * step) for i in range(max_points)]
        
        for idx in indices:
            point = track_data[idx]
            locations.append({
                'lat': round(point['lat'], 4),
                'lon': round(point['lon'], 4),
                'alt': point.get('alt', 0),
                'timestamp': point.get('timestamp', 0)  # Include timestamp for proximity checks
            })
        
        return locations
    
    def get_anomaly_dna(self, flight_id: str, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Find similar historical flights using route-based matching and trajectory comparison.
        
        Algorithm:
        1. Get origin/destination airports for the target flight
        2. Find flights with same origin AND/OR destination
        3. If target is an anomaly, filter to anomalies only
        4. If target has specific rule IDs, filter to same rules
        5. Calculate point-by-point trajectory match (5 NM threshold)
        6. Return flights with >= 80% match percentage
        
        Returns:
            {
                flight_info: {...},
                similar_flights: [{flight_id, similarity_score, match_percentage, date, pattern, ...}],
                recurring_pattern: str,
                risk_assessment: str,
                insights: [...]
            }
        """
        now = int(datetime.now().timestamp())
        lookback_ts = now - (lookback_days * 86400)
        
        DISTANCE_THRESHOLD_NM = 5.0  # Points within 5 NM are considered matching
        MATCH_THRESHOLD_PCT = 80.0   # Flights with >= 80% matching points are similar
        
        # Get the target flight's data
        flight_info = {'flight_id': flight_id}
        flight_path = []
        flight_anomalies = []
        flight_timestamp = None
        is_anomaly_flight = False
        origin_airport = None
        dest_airport = None
        
        # Step 1: Get flight metadata (origin/destination)
        query = """
            SELECT origin_airport, destination_airport, callsign, airline
            FROM flight_metadata
            WHERE flight_id = ?
        """
        metadata_results = self._execute_query('research', query, (flight_id,))
        if metadata_results:
            origin_airport, dest_airport, callsign, airline = metadata_results[0]
            if callsign:
                flight_info['callsign'] = callsign
            if airline:
                flight_info['airline'] = airline
            if origin_airport:
                flight_info['origin'] = origin_airport
            if dest_airport:
                flight_info['destination'] = dest_airport
        
        # Step 2: Get flight track points
        for table_name in ['anomalies_tracks', 'normal_tracks']:
            query = f"""
                SELECT timestamp, lat, lon, alt, track, callsign
                FROM {table_name}
                WHERE flight_id = ?
                ORDER BY timestamp
            """
            results = self._execute_query('research', query, (flight_id,))
            
            if results:
                is_anomaly_flight = (table_name == 'anomalies_tracks')
                for row in results:
                    ts, lat, lon, alt, track, callsign = row
                    if lat is not None and lon is not None:
                        flight_path.append({
                            'timestamp': ts,
                            'lat': lat, 
                            'lon': lon, 
                            'alt': alt or 0, 
                            'track': track
                        })
                    if not flight_info.get('callsign') and callsign:
                        flight_info['callsign'] = callsign
                    if not flight_timestamp:
                        flight_timestamp = ts
                break  # Found data
        
        if not flight_path:
            return {
                'flight_info': flight_info,
                'similar_flights': [],
                'recurring_pattern': 'No data found for this flight',
                'risk_assessment': 'Unknown',
                'insights': [],
                'anomalies_detected': []
            }
        
        # Step 3: Get anomaly rules for this flight
        target_rule_ids = set()
        target_rule_names = {}
        for db_name in ['research', 'anomalies']:
            query = """
                SELECT timestamp, full_report
                FROM anomaly_reports
                WHERE flight_id = ?
            """
            results = self._execute_query(db_name, query, (flight_id,))
            
            for row in results:
                ts, report_json = row
                is_anomaly_flight = True
                try:
                    report = json.loads(report_json) if isinstance(report_json, str) else report_json
                    matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                    for rule in matched_rules:
                        rule_id = rule.get('id')
                        rule_name = rule.get('name', 'Unknown')
                        target_rule_ids.add(rule_id)
                        target_rule_names[rule_id] = rule_name
                        flight_anomalies.append({
                            'rule_id': rule_id,
                            'rule_name': rule_name,
                            'timestamp': ts
                        })
                except:
                    pass
        
        flight_info['is_anomaly'] = is_anomaly_flight
        flight_info['rule_ids'] = list(target_rule_ids)
        
        # Step 4: Build candidate flight query based on route
        candidate_flight_ids = set()
        
        # If we have origin and/or destination, search by route
        if origin_airport or dest_airport:
            # Build WHERE clause for route matching
            route_conditions = []
            route_params = [flight_id, lookback_ts, now]
            
            if origin_airport and dest_airport:
                # Both origin and dest available - search for exact route match
                route_conditions.append("(origin_airport = ? AND destination_airport = ?)")
                route_params.extend([origin_airport, dest_airport])
            elif origin_airport:
                # Only origin available
                route_conditions.append("origin_airport = ?")
                route_params.append(origin_airport)
            elif dest_airport:
                # Only destination available
                route_conditions.append("destination_airport = ?")
                route_params.append(dest_airport)
            
            route_where = " OR ".join(route_conditions)
            
            # Query for candidate flights by route
            query = f"""
                SELECT DISTINCT flight_id
                FROM flight_metadata
                WHERE flight_id != ?
                  AND first_seen_ts BETWEEN ? AND ?
                  AND ({route_where})
            """
            results = self._execute_query('research', query, tuple(route_params))
            for row in results:
                candidate_flight_ids.add(row[0])
        
        # If no route info, fall back to geographic search
        if not candidate_flight_ids:
            lats = [p['lat'] for p in flight_path if p['lat']]
            lons = [p['lon'] for p in flight_path if p['lon']]
            if lats and lons:
                bbox = {
                    'min_lat': min(lats), 'max_lat': max(lats),
                    'min_lon': min(lons), 'max_lon': max(lons)
                }
                for table_name in ['anomalies_tracks', 'normal_tracks']:
                    query = f"""
                        SELECT DISTINCT flight_id
                        FROM {table_name}
                        WHERE flight_id != ?
                          AND timestamp BETWEEN ? AND ?
                          AND lat BETWEEN ? AND ?
                          AND lon BETWEEN ? AND ?
                        LIMIT 200
                    """
                    params = (
                        flight_id, lookback_ts, now,
                        bbox['min_lat'] - 0.5, bbox['max_lat'] + 0.5,
                        bbox['min_lon'] - 0.5, bbox['max_lon'] + 0.5
                    )
                    results = self._execute_query('research', query, params)
                    for row in results:
                        candidate_flight_ids.add(row[0])
        
        # Step 5: If target is anomaly, filter candidates to anomalies only
        if is_anomaly_flight and target_rule_ids:
            anomaly_candidates = set()
            for db_name in ['research', 'anomalies']:
                if candidate_flight_ids:
                    flight_ids_str = ','.join([f"'{fid}'" for fid in candidate_flight_ids])
                    query = f"""
                        SELECT DISTINCT flight_id, full_report
                        FROM anomaly_reports
                        WHERE flight_id IN ({flight_ids_str})
                    """
                    results = self._execute_query(db_name, query, ())
                    for row in results:
                        other_id, report_json = row
                        try:
                            report = json.loads(report_json) if isinstance(report_json, str) else report_json
                            matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                            other_rule_ids = {rule.get('id') for rule in matched_rules}
                            # Check if any rules match
                            if target_rule_ids.intersection(other_rule_ids):
                                anomaly_candidates.add(other_id)
                        except:
                            pass
            # Use filtered candidates if any found, otherwise keep all
            if anomaly_candidates:
                candidate_flight_ids = anomaly_candidates
        
        # Step 6: For each candidate, calculate trajectory match percentage
        similar_flights = []
        
        for candidate_id in list(candidate_flight_ids)[:100]:  # Limit to 100 candidates
            # Get candidate track
            candidate_path = []
            candidate_callsign = None
            candidate_timestamp = None
            candidate_is_anomaly = False
            
            for table_name in ['anomalies_tracks', 'normal_tracks']:
                query = f"""
                    SELECT timestamp, lat, lon, alt, track, callsign
                    FROM {table_name}
                    WHERE flight_id = ?
                    ORDER BY timestamp
                """
                results = self._execute_query('research', query, (candidate_id,))
                if results:
                    candidate_is_anomaly = (table_name == 'anomalies_tracks')
                    for row in results:
                        ts, lat, lon, alt, track, callsign = row
                        if lat is not None and lon is not None:
                            candidate_path.append({
                                'timestamp': ts,
                                'lat': lat,
                                'lon': lon,
                                'alt': alt or 0
                            })
                        if not candidate_callsign and callsign:
                            candidate_callsign = callsign
                        if not candidate_timestamp:
                            candidate_timestamp = ts
                    break
            
            if not candidate_path or len(candidate_path) < 3:
                continue
            
            # Calculate point-by-point match percentage
            # For each point in target flight, find closest point in candidate
            matching_points = 0
            total_points = len(flight_path)
            
            for target_point in flight_path:
                target_lat = target_point['lat']
                target_lon = target_point['lon']
                
                # Find minimum distance to any point in candidate path
                min_distance = float('inf')
                for cand_point in candidate_path:
                    dist = _haversine_nm(target_lat, target_lon, cand_point['lat'], cand_point['lon'])
                    if dist < min_distance:
                        min_distance = dist
                
                if min_distance <= DISTANCE_THRESHOLD_NM:
                    matching_points += 1
            
            match_percentage = (matching_points / total_points * 100) if total_points > 0 else 0
            
            # Only include flights with >= 80% match
            if match_percentage >= MATCH_THRESHOLD_PCT:
                # Get candidate metadata
                cand_origin = None
                cand_dest = None
                query = """
                    SELECT origin_airport, destination_airport
                    FROM flight_metadata
                    WHERE flight_id = ?
                """
                meta_results = self._execute_query('research', query, (candidate_id,))
                if meta_results:
                    cand_origin, cand_dest = meta_results[0]
                
                # Get candidate rule IDs
                cand_rule_ids = set()
                for db_name in ['research', 'anomalies']:
                    query = """
                        SELECT full_report
                        FROM anomaly_reports
                        WHERE flight_id = ?
                    """
                    results = self._execute_query(db_name, query, (candidate_id,))
                    for row in results:
                        try:
                            report = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                            matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                            for rule in matched_rules:
                                cand_rule_ids.add(rule.get('id'))
                        except:
                            pass
                
                # Determine pattern type
                pattern_desc = []
                if cand_origin == origin_airport and cand_dest == dest_airport:
                    pattern_desc.append('same_route')
                elif cand_origin == origin_airport:
                    pattern_desc.append('same_origin')
                elif cand_dest == dest_airport:
                    pattern_desc.append('same_destination')
                
                if target_rule_ids and cand_rule_ids:
                    common_rules = target_rule_ids.intersection(cand_rule_ids)
                    if common_rules:
                        pattern_desc.append('same_anomalies')
                
                if candidate_is_anomaly:
                    pattern_desc.append('anomaly')
                
                if not pattern_desc:
                    pattern_desc.append('trajectory_match')
                
                # Calculate similarity score (weighted)
                similarity_score = int(match_percentage)
                
                similar_flights.append({
                    'flight_id': candidate_id,
                    'callsign': candidate_callsign or 'Unknown',
                    'similarity_score': similarity_score,
                    'match_percentage': round(match_percentage, 1),
                    'matching_points': matching_points,
                    'total_points': total_points,
                    'date': datetime.fromtimestamp(candidate_timestamp).isoformat() if candidate_timestamp else None,
                    'pattern': '+'.join(pattern_desc),
                    'origin': cand_origin,
                    'destination': cand_dest,
                    'is_anomaly': candidate_is_anomaly,
                    'common_rules': list(target_rule_ids.intersection(cand_rule_ids)) if target_rule_ids and cand_rule_ids else []
                })
        
        # Sort by match percentage (highest first)
        similar_flights.sort(key=lambda x: x['match_percentage'], reverse=True)
        similar_flights = similar_flights[:15]  # Top 15
        
        # Calculate statistics
        lats = [p['lat'] for p in flight_path if p['lat']]
        lons = [p['lon'] for p in flight_path if p['lon']]
        alts = [p['alt'] for p in flight_path if p['alt']]
        avg_alt = sum(alts) / len(alts) if alts else 0
        max_alt = max(alts) if alts else 0
        
        bbox = None
        centroid = None
        if lats and lons:
            bbox = {
                'min_lat': min(lats), 'max_lat': max(lats),
                'min_lon': min(lons), 'max_lon': max(lons)
            }
            centroid = {
                'lat': sum(lats) / len(lats),
                'lon': sum(lons) / len(lons)
            }
        
        # Determine recurring pattern analysis
        high_match = [f for f in similar_flights if f['match_percentage'] >= 90]
        same_route = [f for f in similar_flights if 'same_route' in f['pattern']]
        same_anomalies = [f for f in similar_flights if 'same_anomalies' in f['pattern']]
        
        if len(high_match) >= 3 and len(same_anomalies) >= 2:
            recurring_pattern = f"Strong recurring pattern: {len(high_match)} flights with >90% trajectory match, {len(same_anomalies)} with same anomalies"
            risk_assessment = 'High - Systematic pattern requiring investigation'
        elif len(high_match) >= 3:
            recurring_pattern = f"Recurring pattern detected: {len(high_match)} flights with >90% trajectory match"
            risk_assessment = 'High - Possible reconnaissance or surveillance pattern'
        elif len(same_route) >= 2:
            recurring_pattern = f"Route pattern: {len(same_route)} flights on same route"
            risk_assessment = 'Medium - Repeated route activity'
        elif len(similar_flights) >= 3:
            recurring_pattern = f"Partial pattern: {len(similar_flights)} flights with similar trajectory"
            risk_assessment = 'Medium - Geographic overlap'
        else:
            recurring_pattern = 'No significant recurring pattern detected'
            risk_assessment = 'Low - Unique or infrequent flight path'
        
        # Generate insights
        insights = []
        route_str = f"{origin_airport or '?'} → {dest_airport or '?'}"
        insights.append(f"Route: {route_str}")
        
        if is_anomaly_flight:
            insights.append(f"This flight is flagged as an anomaly")
            if target_rule_ids:
                rule_names_list = [target_rule_names.get(rid, f'Rule {rid}') for rid in list(target_rule_ids)[:5]]
                insights.append(f"Triggered rules: {', '.join(rule_names_list)}")
        
        if similar_flights:
            insights.append(f"Found {len(similar_flights)} similar flights (≥{MATCH_THRESHOLD_PCT}% trajectory match)")
            avg_match = sum(f['match_percentage'] for f in similar_flights) / len(similar_flights)
            insights.append(f"Average trajectory match: {avg_match:.1f}%")
        
        if same_route:
            insights.append(f"{len(same_route)} flights on exact same route")
        
        if same_anomalies:
            insights.append(f"{len(same_anomalies)} flights triggered the same anomaly rules")
        
        if avg_alt > 0:
            insights.append(f"Average altitude: {avg_alt:.0f} ft, Max altitude: {max_alt:.0f} ft")
        
        return {
            'flight_info': flight_info,
            'similar_flights': similar_flights,
            'recurring_pattern': recurring_pattern,
            'risk_assessment': risk_assessment,
            'insights': insights,
            'anomalies_detected': flight_anomalies,
            'search_criteria': {
                'origin': origin_airport,
                'destination': dest_airport,
                'is_anomaly': is_anomaly_flight,
                'rule_ids': list(target_rule_ids),
                'match_threshold': MATCH_THRESHOLD_PCT,
                'distance_threshold_nm': DISTANCE_THRESHOLD_NM
            },
            'fingerprint': {
                'bbox': bbox,
                'centroid': centroid,
                'avg_altitude': avg_alt,
                'max_altitude': max_alt,
                'rule_ids': list(target_rule_ids),
                'flight_hour': datetime.fromtimestamp(flight_timestamp).hour if flight_timestamp else 12,
                'flight_weekday': datetime.fromtimestamp(flight_timestamp).weekday() if flight_timestamp else 0
            }
        }
    
    def detect_pattern_clusters(self, start_ts: int, end_ts: int,
                               min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Detect recurring suspicious patterns across multiple flights.
        
        Returns:
            [{pattern_id, description, flights: [...], first_seen, last_seen, risk_level}]
        """
        # Group flights by geographic hotspots and anomaly types
        patterns = []
        
        # Query anomaly reports and group by location
        location_clusters = defaultdict(list)
        
        for db_name in ['research', 'anomalies']:
            query = """
                SELECT flight_id, timestamp, full_report
                FROM anomaly_reports
                WHERE timestamp BETWEEN ? AND ?
            """
            results = self._execute_query(db_name, query, (start_ts, end_ts))
            
            for row in results:
                flight_id, timestamp, report_json = row
                try:
                    report = json.loads(report_json) if isinstance(report_json, str) else report_json
                    
                    # Get flight path centroid
                    flight_path = report.get('summary', {}).get('flight_path', [])
                    if flight_path:
                        avg_lon = sum(p[0] for p in flight_path) / len(flight_path)
                        avg_lat = sum(p[1] for p in flight_path) / len(flight_path)
                        
                        # Grid to 0.5 degree cells
                        grid_key = (round(avg_lat * 2) / 2, round(avg_lon * 2) / 2)
                        
                        location_clusters[grid_key].append({
                            'flight_id': flight_id,
                            'timestamp': timestamp,
                            'rules': [r.get('id') for r in report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])]
                        })
                except:
                    pass
        
        # Find clusters with min_occurrences
        pattern_id = 1
        for (lat, lon), flights in location_clusters.items():
            if len(flights) >= min_occurrences:
                timestamps = [f['timestamp'] for f in flights]
                patterns.append({
                    'pattern_id': f'CLUSTER_{pattern_id}',
                    'description': f'Anomaly cluster at {lat}°N, {lon}°E',
                    'location': {'lat': lat, 'lon': lon},
                    'flights': [f['flight_id'] for f in flights],
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'occurrence_count': len(flights),
                    'risk_level': 'High' if len(flights) >= 5 else 'Medium'
                })
                pattern_id += 1
        
        return sorted(patterns, key=lambda x: x['occurrence_count'], reverse=True)
    
    def get_military_routes(self, start_ts: int, end_ts: int, 
                           country: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze military aircraft preferred routes.
        
        Answers: "What are the preferred routes for US tankers/ISR aircraft?"
        
        Returns:
            {
                by_country: {country: {total_flights, routes: [{from_region, to_region, count}]}},
                by_type: {type: {total_flights, common_areas: [...]}},
                route_segments: [{start: {lat, lon}, end: {lat, lon}, count, countries: [...]}]
            }
        """
        # Get military patterns first
        patterns = self.get_military_patterns(start_ts, end_ts, country=country)
        
        # Analyze routes by country and type
        country_routes = defaultdict(lambda: {'total_flights': 0, 'routes': defaultdict(int), 'areas': []})
        type_routes = defaultdict(lambda: {'total_flights': 0, 'common_areas': Counter()})
        
        # Grid for route segments (0.5 degree cells)
        route_segments = defaultdict(lambda: {'count': 0, 'countries': set(), 'types': set()})
        
        # Named regions for route analysis
        REGIONS = {
            'Mediterranean Sea': {'lat_range': (31, 36), 'lon_range': (28, 36)},
            'Eastern Med': {'lat_range': (33, 37), 'lon_range': (33, 36)},
            'Levant Coast': {'lat_range': (31, 35), 'lon_range': (34, 36)},
            'Cyprus Area': {'lat_range': (34, 36), 'lon_range': (32, 35)},
            'Jordan': {'lat_range': (29, 33), 'lon_range': (35, 39)},
            'Syria': {'lat_range': (32, 37), 'lon_range': (35, 42)},
            'Iraq': {'lat_range': (29, 37), 'lon_range': (38, 48)},
            'Saudi Arabia': {'lat_range': (16, 32), 'lon_range': (34, 56)},
            'Egypt': {'lat_range': (22, 32), 'lon_range': (25, 35)},
            'Turkey': {'lat_range': (36, 42), 'lon_range': (26, 45)},
            'Israel': {'lat_range': (29, 34), 'lon_range': (34, 36)},
        }
        
        def get_region(lat: float, lon: float) -> str:
            """Get named region for coordinates."""
            for region, bounds in REGIONS.items():
                if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and
                    bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
                    return region
            return 'Other'
        
        for pattern in patterns:
            country_code = pattern['country']
            aircraft_type = pattern['type']
            locations = pattern.get('locations', [])
            
            country_routes[country_code]['total_flights'] += 1
            type_routes[aircraft_type]['total_flights'] += 1
            
            if len(locations) >= 2:
                # Get start and end regions
                start_loc = locations[0]
                end_loc = locations[-1]
                
                start_region = get_region(start_loc['lat'], start_loc['lon'])
                end_region = get_region(end_loc['lat'], end_loc['lon'])
                
                route_key = f"{start_region} → {end_region}"
                country_routes[country_code]['routes'][route_key] += 1
                
                # Track areas visited
                for loc in locations:
                    region = get_region(loc['lat'], loc['lon'])
                    type_routes[aircraft_type]['common_areas'][region] += 1
                    
                    # Grid cell for segment
                    grid_lat = round(loc['lat'] * 2) / 2
                    grid_lon = round(loc['lon'] * 2) / 2
                    segment_key = (grid_lat, grid_lon)
                    route_segments[segment_key]['count'] += 1
                    route_segments[segment_key]['countries'].add(country_code)
                    route_segments[segment_key]['types'].add(aircraft_type)
        
        # Format output
        result = {
            'by_country': {},
            'by_type': {},
            'route_segments': [],
            'total_military_flights': len(patterns)
        }
        
        # Format country routes
        for country_code, data in country_routes.items():
            top_routes = sorted(data['routes'].items(), key=lambda x: x[1], reverse=True)[:5]
            result['by_country'][country_code] = {
                'total_flights': data['total_flights'],
                'routes': [{'route': r, 'count': c} for r, c in top_routes]
            }
        
        # Format type routes
        for aircraft_type, data in type_routes.items():
            top_areas = data['common_areas'].most_common(5)
            result['by_type'][aircraft_type] = {
                'total_flights': data['total_flights'],
                'common_areas': [{'area': a, 'count': c} for a, c in top_areas]
            }
        
        # Format route segments (top 20 most used)
        sorted_segments = sorted(route_segments.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
        for (lat, lon), data in sorted_segments:
            result['route_segments'].append({
                'lat': lat,
                'lon': lon,
                'count': data['count'],
                'countries': list(data['countries']),
                'types': list(data['types'])
            })
        
        return result
    
    def get_holding_cost_analysis(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """Analyze holding patterns and calculate fuel cost using flight_phase_summary."""
        HOLDING_FUEL_COST_PER_MIN = 50
        query = """
            SELECT callsign, destination_airport, flight_phase_summary
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND flight_phase_summary IS NOT NULL AND flight_phase_summary LIKE '%holding_time%'
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        by_airport = defaultdict(lambda: {'count': 0, 'total_holding_min': 0, 'total_cost': 0})
        top_events = []
        total_holding_min = 0
        
        for row in results:
            callsign, dest_airport, phase_summary = row
            try:
                phases = json.loads(phase_summary) if isinstance(phase_summary, str) else phase_summary
                holding_time = phases.get('holding_time', 0)
                if holding_time > 0:
                    holding_min = holding_time / 60
                    holding_cost = holding_min * HOLDING_FUEL_COST_PER_MIN
                    total_holding_min += holding_min
                    if dest_airport:
                        by_airport[dest_airport]['count'] += 1
                        by_airport[dest_airport]['total_holding_min'] += holding_min
                        by_airport[dest_airport]['total_cost'] += holding_cost
                    top_events.append({
                        'callsign': callsign or 'Unknown',
                        'airport': dest_airport or 'Unknown',
                        'holding_min': round(holding_min, 1),
                        'cost_usd': round(holding_cost, 2)
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        
        airport_stats = [{'airport': airport, 'count': data['count'],
                         'avg_holding_min': round(data['total_holding_min'] / data['count'], 1),
                         'total_cost_usd': round(data['total_cost'], 2)}
                        for airport, data in by_airport.items()]
        
        return {
            'total_flights_with_holding': len(top_events),
            'total_holding_hours': round(total_holding_min / 60, 1),
            'estimated_fuel_cost_usd': round(total_holding_min * HOLDING_FUEL_COST_PER_MIN, 2),
            'by_airport': sorted(airport_stats, key=lambda x: x['total_cost_usd'], reverse=True),
            'top_holding_events': sorted(top_events, key=lambda x: x['holding_min'], reverse=True)[:20]
        }
    
    def get_gps_jamming_hotspots(self, start_ts: int, end_ts: int, grid_size: float = 0.5) -> List[Dict[str, Any]]:
        """Get GPS jamming hotspots using signal_loss_events table."""
        query = """
            SELECT flight_id, lat, lon, loss_duration_sec, timestamp
            FROM signal_loss_events
            WHERE timestamp BETWEEN ? AND ?
            AND lat IS NOT NULL AND lon IS NOT NULL
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        grid_cells = defaultdict(lambda: {'loss_count': 0, 'flights': set(), 'total_duration': 0, 'locations': []})
        
        for row in results:
            flight_id, lat, lon, duration, timestamp = row
            grid_lat = round(lat / grid_size) * grid_size
            grid_lon = round(lon / grid_size) * grid_size
            cell_key = (grid_lat, grid_lon)
            grid_cells[cell_key]['loss_count'] += 1
            grid_cells[cell_key]['flights'].add(flight_id)
            grid_cells[cell_key]['total_duration'] += duration or 0
            grid_cells[cell_key]['locations'].append({'lat': lat, 'lon': lon})
        
        hotspots = []
        for (grid_lat, grid_lon), data in grid_cells.items():
            if data['loss_count'] >= 3:
                avg_lat = sum(loc['lat'] for loc in data['locations']) / len(data['locations'])
                avg_lon = sum(loc['lon'] for loc in data['locations']) / len(data['locations'])
                severity_score = data['loss_count'] * len(data['flights']) * (data['total_duration'] / 3600)
                hotspots.append({
                    'lat': round(avg_lat, 3),
                    'lon': round(avg_lon, 3),
                    'loss_count': data['loss_count'],
                    'affected_flights': len(data['flights']),
                    'avg_duration_sec': round(data['total_duration'] / data['loss_count'], 1),
                    'severity_score': round(severity_score, 1)
                })
        
        return sorted(hotspots, key=lambda x: x['severity_score'], reverse=True)[:30]
    
    def get_military_presence_patterns(self, start_ts: int, end_ts: int,
                                      military_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyze military aircraft loitering patterns using is_military field and category='Military_and_government'."""
        if military_type:
            query = """
                SELECT callsign, military_type, flight_duration_sec, crossed_borders,
                       first_seen_ts, geographic_region, category
                FROM flight_metadata
                WHERE first_seen_ts BETWEEN ? AND ?
                AND (is_military = 1 OR category = 'Military_and_government') AND military_type = ?
            """
            results = self._execute_query('research', query, (start_ts, end_ts, military_type))
        else:
            query = """
                SELECT callsign, military_type, flight_duration_sec, crossed_borders,
                       first_seen_ts, geographic_region, category
                FROM flight_metadata
                WHERE first_seen_ts BETWEEN ? AND ?
                AND (is_military = 1 OR category = 'Military_and_government')
            """
            results = self._execute_query('research', query, (start_ts, end_ts))
        
        by_type = Counter()
        by_country = Counter()
        loitering = []
        
        for row in results:
            callsign, mil_type, duration, borders, timestamp, region, category = row
            if mil_type:
                by_type[mil_type] += 1
            elif category == 'Military_and_government':
                by_type['Government/Military'] += 1
            
            # Use core military detection for country identification
            country = 'Unknown'
            if callsign and core_is_military is not None:
                is_mil, mil_info = core_is_military(callsign=callsign)
                if is_mil and mil_info:
                    # Extract country from mil_info
                    info_lower = mil_info.lower()
                    if any(x in info_lower for x in ['us ', 'usaf', 'us air force', 'us navy', 'us marine', 'us army']):
                        country = "US"
                    elif any(x in info_lower for x in ['united kingdom', 'raf', 'royal air force', 'royal navy']):
                        country = "GB"
                    elif 'german' in info_lower or 'luftwaffe' in info_lower:
                        country = "DE"
                    elif 'french' in info_lower:
                        country = "FR"
                    elif any(x in info_lower for x in ['israel', 'idf']):
                        country = "IL"
                    elif 'russian' in info_lower:
                        country = "RU"
                    elif 'nato' in info_lower:
                        country = "NATO"
            elif callsign:
                # Fallback to local patterns if core module not available
                cs_upper = callsign.upper()
                for prefix, info in MILITARY_CALLSIGN_PATTERNS.items():
                    if cs_upper.startswith(prefix):
                        country = info['country']
                        break
            by_country[country] += 1
            if duration and duration > 7200:
                pattern = 'Unknown'
                if mil_type == 'ISR':
                    pattern = 'Intelligence/Surveillance Pattern'
                elif mil_type == 'tanker':
                    pattern = 'Refueling Orbit'
                elif duration > 14400:
                    pattern = 'Extended Loitering/CAP'
                loitering.append({
                    'callsign': callsign or 'Unknown',
                    'type': mil_type or category or 'Unknown',
                    'country': country,
                    'duration_hours': round(duration / 3600, 1),
                    'region': region or 'Unknown',
                    'pattern_description': pattern,
                    'timestamp': timestamp
                })
        
        return {
            'total_military_flights': len(results),
            'by_type': dict(by_type),
            'by_country': dict(by_country),
            'loitering_patterns': sorted(loitering, key=lambda x: x['duration_hours'], reverse=True)[:20],
            'analysis_period_days': (end_ts - start_ts) / 86400
        }
    
    def get_alternate_airport_preferences(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """Analyze which alternate airports are preferred during diversions."""
        query = """
            SELECT callsign, origin_airport, destination_airport, aircraft_type, full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
            AND (matched_rule_ids LIKE '%8,%' OR matched_rule_ids LIKE '%8' 
                 OR matched_rule_ids = '8' OR matched_rule_ids LIKE '%, 8%')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        by_alternate = Counter()
        by_aircraft_type = defaultdict(Counter)
        by_origin = defaultdict(Counter)
        
        for row in results:
            callsign, origin, planned_dest, aircraft, full_report = row
            try:
                report = json.loads(full_report) if isinstance(full_report, str) else full_report
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                for rule in matched_rules:
                    if rule.get('id') == 8:
                        details = rule.get('details', {})
                        actual_dest = details.get('actual_destination') or details.get('actual')
                        if actual_dest:
                            by_alternate[actual_dest] += 1
                            if aircraft:
                                by_aircraft_type[aircraft][actual_dest] += 1
                            if origin:
                                by_origin[origin][actual_dest] += 1
            except (json.JSONDecodeError, KeyError):
                continue
        
        aircraft_preferences = []
        for aircraft, alternates in by_aircraft_type.items():
            if alternates:
                preferred = alternates.most_common(1)[0]
                aircraft_preferences.append({
                    'aircraft_type': aircraft,
                    'preferred_alternate': preferred[0],
                    'count': preferred[1]
                })
        
        origin_alternates = {}
        for origin, alternates in by_origin.items():
            origin_alternates[origin] = [{'airport': apt, 'count': cnt} 
                                         for apt, cnt in alternates.most_common(5)]
        
        return {
            'total_diversions': sum(by_alternate.values()),
            'by_alternate': dict(by_alternate.most_common(10)),
            'by_aircraft_type': aircraft_preferences,
            'by_origin': origin_alternates,
            'critical_alternates': [{'airport': apt, 'usage_count': cnt} 
                                   for apt, cnt in by_alternate.most_common(10)]
        }
    
    def analyze_flight_jamming(self, flight_id: str) -> Dict[str, Any]:
        """
        Analyze a specific flight for GPS jamming indicators.
        
        Based on analysis of GPS jamming cases (flights 3cc68900, 3cc75595, 3cc7b2ab, etc.),
        this function detects the following jamming signatures:
        
        1. ALTITUDE OSCILLATION - Rapid jumps between real altitude and spoofed values
        2. SPOOFED ALTITUDE VALUES - Known GPS spoofing altitudes (34764ft, 44700ft, etc.)
        3. IMPOSSIBLE ALTITUDE RATES - >3000 ft/sec change (physically impossible)
        4. SPEED ANOMALIES - Ground speed > 600 kts for commercial aircraft
        5. POSITION TELEPORTATION - Implied speed > 600 kts between consecutive points
        6. MLAT-ONLY SOURCE - GPS completely jammed, only multilateration works
        7. CLUSTERED ANOMALIES - Anomalies happening in bursts
        8. HEADING OSCILLATION - Track alternating between opposite directions (e.g., 90° ↔ 270°)
        9. IMPOSSIBLE TURN RATES - >8 deg/sec instantaneous turn rate
        10. TRACK VS BEARING MISMATCH - Reported heading differs >90° from actual movement
        
        Args:
            flight_id: The flight ID to analyze
        
        Returns:
            {
                flight_id: str,
                jamming_score: int (0-100),
                jamming_confidence: str ('HIGH', 'MEDIUM', 'LOW', 'UNLIKELY'),
                indicators: {...},
                anomaly_details: [...],
                summary: str
            }
        """
        # Known GPS spoofing altitude values
        SPOOFED_ALTITUDES = {34764, 44700, 44600, 44500, 40000, 40400, 40800, 36864, 42100, 42500, 42700}
        
        # Thresholds
        ALTITUDE_JUMP_THRESHOLD_FT = 5000
        ALTITUDE_RATE_THRESHOLD = 3000  # ft/sec - physically impossible
        SPEED_THRESHOLD_KTS = 600
        POSITION_JUMP_THRESHOLD_KTS = 600
        TURN_RATE_THRESHOLD_DEG_S = 8.0  # deg/sec - impossible for aircraft
        HEADING_REVERSAL_THRESHOLD_DEG = 150  # Heading change > 150° in short time
        TRACK_BEARING_MISMATCH_DEG = 90  # Reported track differs >90° from actual bearing
        
        result = {
            'flight_id': flight_id,
            'jamming_score': 0,
            'jamming_confidence': 'UNLIKELY',
            'indicators': {
                'altitude_jumps': 0,
                'spoofed_altitude_hits': 0,
                'impossible_altitude_rates': 0,
                'speed_anomalies': 0,
                'position_teleports': 0,
                'mlat_ratio': 0.0,
                'anomaly_clusters': 0,
                # NEW indicators based on flight analysis
                'heading_oscillations': 0,
                'impossible_turn_rates': 0,
                'track_bearing_mismatches': 0,
            },
            'anomaly_details': [],
            'unique_altitudes': [],
            'summary': ''
        }
        
        # Get flight track data
        track_data = []
        for table_name in ['anomalies_tracks', 'normal_tracks', 'flight_tracks']:
            # Try different databases
            for db_name in ['research', 'last']:
                query = f"""
                    SELECT timestamp, lat, lon, alt, gspeed, vspeed, track, source
                    FROM {table_name}
                    WHERE flight_id = ?
                    ORDER BY timestamp
                """
                try:
                    results = self._execute_query(db_name, query, (flight_id,))
                    if results:
                        for row in results:
                            ts, lat, lon, alt, gspeed, vspeed, heading, source = row
                            if lat is not None and lon is not None:
                                track_data.append({
                                    'timestamp': ts,
                                    'lat': lat,
                                    'lon': lon,
                                    'alt': alt,
                                    'gspeed': gspeed,
                                    'vspeed': vspeed,
                                    'track': heading,
                                    'source': source
                                })
                        break
                except:
                    continue
            if track_data:
                break
        
        if not track_data:
            result['summary'] = 'No track data found for this flight'
            return result
        
        def _initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate initial bearing in degrees from point 1 to point 2."""
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lon = math.radians(lon2 - lon1)
            
            x = math.sin(delta_lon) * math.cos(lat2_rad)
            y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
            
            bearing = math.atan2(x, y)
            return (math.degrees(bearing) + 360) % 360
        
        total_points = len(track_data)
        anomaly_timestamps = []
        mlat_count = 0
        unique_alts = set()
        prev_heading_change = 0  # For detecting heading oscillation
        
        # Analyze each consecutive pair of points
        for i in range(1, total_points):
            prev = track_data[i-1]
            curr = track_data[i]
            
            time_diff = curr['timestamp'] - prev['timestamp']
            if time_diff <= 0:
                continue
            
            # Track MLAT source
            if curr.get('source') and curr['source'].upper() == 'MLAT':
                mlat_count += 1
            
            # Track unique altitudes
            if curr['alt'] is not None:
                unique_alts.add(curr['alt'])
            
            # 1. Check altitude anomalies
            if curr['alt'] is not None and prev['alt'] is not None:
                alt_diff = abs(curr['alt'] - prev['alt'])
                alt_rate = alt_diff / time_diff
                
                # Impossible altitude rate
                if alt_rate > ALTITUDE_RATE_THRESHOLD:
                    result['indicators']['impossible_altitude_rates'] += 1
                    anomaly_timestamps.append(curr['timestamp'])
                    result['anomaly_details'].append({
                        'type': 'impossible_altitude_rate',
                        'timestamp': curr['timestamp'],
                        'from_alt': prev['alt'],
                        'to_alt': curr['alt'],
                        'rate_ft_per_sec': round(alt_rate, 1),
                        'time_diff': time_diff
                    })
                
                # Large altitude jump
                if alt_diff > ALTITUDE_JUMP_THRESHOLD_FT:
                    result['indicators']['altitude_jumps'] += 1
                    anomaly_timestamps.append(curr['timestamp'])
                
                # Jump to/from known spoofed altitude
                if curr['alt'] in SPOOFED_ALTITUDES or prev['alt'] in SPOOFED_ALTITUDES:
                    if alt_diff > ALTITUDE_JUMP_THRESHOLD_FT:
                        result['indicators']['spoofed_altitude_hits'] += 1
                        anomaly_timestamps.append(curr['timestamp'])
                        result['anomaly_details'].append({
                            'type': 'spoofed_altitude',
                            'timestamp': curr['timestamp'],
                            'from_alt': prev['alt'],
                            'to_alt': curr['alt'],
                            'spoofed_value': curr['alt'] if curr['alt'] in SPOOFED_ALTITUDES else prev['alt']
                        })
            
            # 2. Check speed anomalies
            if curr['gspeed'] is not None and curr['gspeed'] > SPEED_THRESHOLD_KTS:
                result['indicators']['speed_anomalies'] += 1
                anomaly_timestamps.append(curr['timestamp'])
                result['anomaly_details'].append({
                    'type': 'impossible_speed',
                    'timestamp': curr['timestamp'],
                    'speed_kts': curr['gspeed']
                })
            
            # 3. Check position teleportation
            dist_nm = 0
            if prev['lat'] is not None and prev['lon'] is not None:
                dist_nm = _haversine_nm(prev['lat'], prev['lon'], curr['lat'], curr['lon'])
                implied_speed = (dist_nm / time_diff) * 3600  # knots
                
                if implied_speed > POSITION_JUMP_THRESHOLD_KTS:
                    result['indicators']['position_teleports'] += 1
                    anomaly_timestamps.append(curr['timestamp'])
                    result['anomaly_details'].append({
                        'type': 'position_teleport',
                        'timestamp': curr['timestamp'],
                        'distance_nm': round(dist_nm, 1),
                        'time_diff': time_diff,
                        'implied_speed_kts': round(implied_speed, 0)
                    })
            
            # 4. NEW: Check heading/track anomalies (GPS jamming signature)
            if curr['track'] is not None and prev['track'] is not None:
                # Calculate signed heading change
                heading_change = ((curr['track'] - prev['track'] + 540) % 360) - 180
                abs_heading_change = abs(heading_change)
                
                # 4a. Impossible turn rate (>8 deg/sec)
                turn_rate = abs_heading_change / time_diff if time_diff > 0 else 0
                if turn_rate > TURN_RATE_THRESHOLD_DEG_S:
                    result['indicators']['impossible_turn_rates'] += 1
                    anomaly_timestamps.append(curr['timestamp'])
                    result['anomaly_details'].append({
                        'type': 'impossible_turn_rate',
                        'timestamp': curr['timestamp'],
                        'from_track': prev['track'],
                        'to_track': curr['track'],
                        'turn_rate_deg_s': round(turn_rate, 1),
                        'time_diff': time_diff
                    })
                
                # 4b. Heading oscillation (rapid back-and-forth, e.g., 90° ↔ 270°)
                # Detect when heading changes direction significantly
                if prev_heading_change != 0:
                    # If heading changed direction significantly (opposite signs with large magnitude)
                    if ((prev_heading_change > 30 and heading_change < -30) or
                        (prev_heading_change < -30 and heading_change > 30)):
                        result['indicators']['heading_oscillations'] += 1
                        anomaly_timestamps.append(curr['timestamp'])
                        result['anomaly_details'].append({
                            'type': 'heading_oscillation',
                            'timestamp': curr['timestamp'],
                            'prev_change': round(prev_heading_change, 1),
                            'curr_change': round(heading_change, 1),
                            'track': curr['track']
                        })
                
                prev_heading_change = heading_change
                
                # 4c. Track vs actual bearing mismatch
                # If the aircraft is moving (dist > 0.1nm), check if reported track matches actual movement
                if dist_nm > 0.1 and prev['lat'] is not None and prev['lon'] is not None:
                    actual_bearing = _initial_bearing(prev['lat'], prev['lon'], curr['lat'], curr['lon'])
                    bearing_mismatch = abs(((curr['track'] - actual_bearing + 540) % 360) - 180)
                    
                    if bearing_mismatch > TRACK_BEARING_MISMATCH_DEG:
                        result['indicators']['track_bearing_mismatches'] += 1
                        anomaly_timestamps.append(curr['timestamp'])
                        result['anomaly_details'].append({
                            'type': 'track_bearing_mismatch',
                            'timestamp': curr['timestamp'],
                            'reported_track': curr['track'],
                            'actual_bearing': round(actual_bearing, 1),
                            'mismatch_deg': round(bearing_mismatch, 1)
                        })
        
        # Calculate MLAT ratio
        result['indicators']['mlat_ratio'] = round(mlat_count / total_points, 2) if total_points > 0 else 0
        
        # Detect anomaly clusters (anomalies within 60 seconds of each other)
        anomaly_timestamps = sorted(set(anomaly_timestamps))
        clusters = []
        current_cluster = []
        for ts in anomaly_timestamps:
            if not current_cluster or ts - current_cluster[-1] <= 60:
                current_cluster.append(ts)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [ts]
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        result['indicators']['anomaly_clusters'] = len(clusters)
        
        # Store unique altitudes (sorted)
        result['unique_altitudes'] = sorted(list(unique_alts))
        
        # Calculate jamming score (0-100)
        score = 0
        
        # Altitude anomalies (most reliable - up to 30 points)
        score += min(10, result['indicators']['altitude_jumps'] * 1)
        score += min(12, result['indicators']['spoofed_altitude_hits'] * 3)
        score += min(8, result['indicators']['impossible_altitude_rates'] * 2)
        
        # Motion anomalies (up to 25 points)
        score += min(12, result['indicators']['speed_anomalies'] * 2)
        score += min(13, result['indicators']['position_teleports'] * 3)
        
        # NEW: Heading/track anomalies (up to 25 points - very reliable indicators)
        # These are especially indicative of GPS jamming based on flight analysis
        score += min(10, result['indicators']['impossible_turn_rates'] * 1)  # Common in jamming
        score += min(10, result['indicators']['heading_oscillations'] * 2)   # Strong indicator
        score += min(5, result['indicators']['track_bearing_mismatches'] * 1)  # Supporting indicator
        
        # MLAT dominance (up to 10 points)
        if result['indicators']['mlat_ratio'] > 0.9:
            score += 10
        elif result['indicators']['mlat_ratio'] > 0.7:
            score += 7
        elif result['indicators']['mlat_ratio'] > 0.5:
            score += 4
        
        # Clustering bonus (up to 10 points)
        score += min(10, result['indicators']['anomaly_clusters'] * 2)
        
        result['jamming_score'] = min(100, score)
        
        # Determine confidence level
        if result['jamming_score'] >= 60:
            result['jamming_confidence'] = 'HIGH'
        elif result['jamming_score'] >= 35:
            result['jamming_confidence'] = 'MEDIUM'
        elif result['jamming_score'] >= 15:
            result['jamming_confidence'] = 'LOW'
        else:
            result['jamming_confidence'] = 'UNLIKELY'
        
        # Generate summary
        summary_parts = []
        if result['jamming_confidence'] == 'HIGH':
            summary_parts.append(f"HIGH CONFIDENCE GPS JAMMING DETECTED (score: {result['jamming_score']}/100)")
        elif result['jamming_confidence'] == 'MEDIUM':
            summary_parts.append(f"Possible GPS jamming detected (score: {result['jamming_score']}/100)")
        elif result['jamming_confidence'] == 'LOW':
            summary_parts.append(f"Minor GPS anomalies detected (score: {result['jamming_score']}/100)")
        else:
            summary_parts.append(f"No significant GPS jamming indicators (score: {result['jamming_score']}/100)")
        
        if result['indicators']['spoofed_altitude_hits'] > 0:
            summary_parts.append(f"Detected {result['indicators']['spoofed_altitude_hits']} jumps to known spoofed altitudes")
        if result['indicators']['altitude_jumps'] > 0:
            summary_parts.append(f"Found {result['indicators']['altitude_jumps']} sudden altitude changes")
        if result['indicators']['position_teleports'] > 0:
            summary_parts.append(f"Found {result['indicators']['position_teleports']} impossible position jumps")
        if result['indicators']['impossible_turn_rates'] > 0:
            summary_parts.append(f"Found {result['indicators']['impossible_turn_rates']} impossible turn rates (>8°/sec)")
        if result['indicators']['heading_oscillations'] > 0:
            summary_parts.append(f"Found {result['indicators']['heading_oscillations']} heading oscillations (back-and-forth)")
        if result['indicators']['track_bearing_mismatches'] > 0:
            summary_parts.append(f"Found {result['indicators']['track_bearing_mismatches']} track/bearing mismatches")
        if result['indicators']['mlat_ratio'] > 0.8:
            summary_parts.append(f"MLAT-only tracking ({int(result['indicators']['mlat_ratio']*100)}% of points)")
        
        result['summary'] = '. '.join(summary_parts)
        
        # Limit anomaly details to top 50 for response size
        result['anomaly_details'] = result['anomaly_details'][:50]
        
        return result

    def get_combined_threat_assessment(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Combined threat assessment - aggregates GPS jamming, military activity,
        unusual patterns, and conflict zone activity into a single score.
        
        Weights:
        - GPS Jamming: 30%
        - Military Activity: 25%
        - Unusual Patterns: 20%
        - Conflict Zones: 25%
        
        Returns:
            {
                'overall_score': int (0-100),
                'threat_level': 'LOW'|'MODERATE'|'ELEVATED'|'HIGH'|'CRITICAL',
                'threat_color': str (hex color),
                'components': {
                    'gps_jamming': {'score': int, 'raw_count': int},
                    'military_activity': {'score': int, 'total_flights': int},
                    'unusual_patterns': {'score': int, 'cluster_count': int},
                    'conflict_zone_activity': {'score': int, 'description': str, 'syria_flights': int}
                },
                'top_concerns': [{'name': str, 'score': int}],
                'recommendations': [str]
            }
        """
        components = {}
        concerns = []
        recommendations = []
        
        # =====================================================================
        # 1. GPS JAMMING COMPONENT (30% weight)
        # =====================================================================
        gps_score = 0
        gps_count = 0
        try:
            # Use cached data if available
            if hasattr(self, '_cached_gps_jamming') and self._cached_gps_jamming:
                gps_jamming = self._cached_gps_jamming
            else:
                gps_jamming = self.get_gps_jamming_heatmap(start_ts, end_ts, 50)
            
            gps_count = len(gps_jamming)
            
            # Score based on count and severity
            high_confidence = sum(1 for j in gps_jamming if j.get('jamming_confidence') == 'HIGH')
            medium_confidence = sum(1 for j in gps_jamming if j.get('jamming_confidence') == 'MEDIUM')
            total_affected = sum(j.get('affected_flights', 0) for j in gps_jamming)
            
            # Calculate GPS score (0-100)
            gps_score = min(100, (
                high_confidence * 15 +
                medium_confidence * 8 +
                min(30, gps_count * 3) +
                min(25, total_affected // 2)
            ))
            
            if gps_score >= 50:
                concerns.append({'name': 'Active GPS Jamming', 'score': gps_score})
                recommendations.append(f"GPS interference detected in {gps_count} zones - advise alternative navigation")
            
        except Exception as e:
            print(f"[ThreatAssessment] GPS jamming error: {e}")
        
        components['gps_jamming'] = {
            'score': gps_score,
            'raw_count': gps_count
        }
        
        # =====================================================================
        # 2. MILITARY ACTIVITY COMPONENT (25% weight)
        # =====================================================================
        mil_score = 0
        mil_flights = 0
        try:
            # Use cached data if available
            if hasattr(self, '_cached_military_by_country') and self._cached_military_by_country:
                mil_data = self._cached_military_by_country
                mil_flights = mil_data.get('summary', {}).get('total_military_flights', 0)
            else:
                # Fall back to military_patterns
                mil_patterns = self.get_military_patterns(start_ts, end_ts)
                mil_flights = len(mil_patterns)
            
            # Count foreign/hostile military (higher weight for Russia, Iran)
            hostile_count = 0
            if hasattr(self, '_cached_military_by_country') and self._cached_military_by_country:
                countries = self._cached_military_by_country.get('countries', {})
                hostile_count = countries.get('RU', {}).get('total_flights', 0)
                hostile_count += countries.get('IR', {}).get('total_flights', 0)
            
            # Calculate military score (0-100)
            mil_score = min(100, (
                min(40, mil_flights * 2) +
                min(40, hostile_count * 10) +
                min(20, mil_flights // 5)
            ))
            
            if mil_score >= 40:
                concerns.append({'name': 'High Military Presence', 'score': mil_score})
                if hostile_count > 0:
                    recommendations.append(f"Russian/Iranian military activity detected ({hostile_count} flights)")
            
        except Exception as e:
            print(f"[ThreatAssessment] Military activity error: {e}")
        
        components['military_activity'] = {
            'score': mil_score,
            'total_flights': mil_flights
        }
        
        # =====================================================================
        # 3. UNUSUAL PATTERNS COMPONENT (20% weight)
        # =====================================================================
        pattern_score = 0
        cluster_count = 0
        try:
            # Use cached pattern clusters if available
            if hasattr(self, '_cached_pattern_clusters') and self._cached_pattern_clusters:
                clusters = self._cached_pattern_clusters
            else:
                # Query anomaly clusters from database
                query = """
                    SELECT COUNT(*) as cnt, 
                           ROUND(lat, 1) as lat_grid, ROUND(lon, 1) as lon_grid
                    FROM anomaly_reports
                    WHERE timestamp BETWEEN ? AND ?
                    AND is_anomaly = 1
                    GROUP BY lat_grid, lon_grid
                    HAVING cnt >= 3
                """
                results = self._execute_query('research', query, (start_ts, end_ts))
                clusters = results if results else []
            
            cluster_count = len(clusters) if isinstance(clusters, list) else 0
            
            # Calculate pattern score
            pattern_score = min(100, cluster_count * 10)
            
            if pattern_score >= 30:
                concerns.append({'name': 'Anomaly Clusters Detected', 'score': pattern_score})
            
        except Exception as e:
            print(f"[ThreatAssessment] Pattern clusters error: {e}")
        
        components['unusual_patterns'] = {
            'score': pattern_score,
            'cluster_count': cluster_count
        }
        
        # =====================================================================
        # 4. CONFLICT ZONE ACTIVITY COMPONENT (25% weight)
        # =====================================================================
        conflict_score = 0
        syria_flights = 0
        conflict_desc = "No significant conflict zone activity"
        try:
            # Use cached data if available
            if hasattr(self, '_cached_military_by_destination') and self._cached_military_by_destination:
                dest_data = self._cached_military_by_destination
                syria_flights = len(dest_data.get('syria_flights', []))
                from_east = dest_data.get('syria_from_east_count', 0)
            else:
                # Estimate from military patterns - check for Syria/Gaza/Lebanon routes
                mil_patterns = self.get_military_patterns(start_ts, end_ts) if mil_flights == 0 else []
                for p in mil_patterns:
                    locs = p.get('locations', [])
                    for loc in locs:
                        lat, lon = loc.get('lat', 0), loc.get('lon', 0)
                        # Syria area roughly: lat 32-37, lon 35-42
                        if 32 <= lat <= 37 and 35 <= lon <= 42:
                            syria_flights += 1
                            break
                from_east = 0
            
            # Calculate conflict score
            conflict_score = min(100, (
                syria_flights * 15 +
                from_east * 25  # Higher weight for Russia/Iran -> Syria
            ))
            
            if syria_flights > 0:
                conflict_desc = f"{syria_flights} flights to Syria"
                if from_east > 0:
                    conflict_desc += f" ({from_east} from Russia/Iran)"
                concerns.append({'name': 'Conflict Zone Activity', 'score': conflict_score})
                recommendations.append(f"Monitor Syria airspace - {syria_flights} military flights detected")
            
        except Exception as e:
            print(f"[ThreatAssessment] Conflict zone error: {e}")
        
        components['conflict_zone_activity'] = {
            'score': conflict_score,
            'description': conflict_desc,
            'syria_flights': syria_flights
        }
        
        # =====================================================================
        # CALCULATE OVERALL SCORE (weighted average)
        # =====================================================================
        overall_score = int(
            gps_score * 0.30 +
            mil_score * 0.25 +
            pattern_score * 0.20 +
            conflict_score * 0.25
        )
        
        # Determine threat level and color
        if overall_score >= 80:
            threat_level = 'CRITICAL'
            threat_color = '#ef4444'  # red
        elif overall_score >= 60:
            threat_level = 'HIGH'
            threat_color = '#f97316'  # orange
        elif overall_score >= 40:
            threat_level = 'ELEVATED'
            threat_color = '#eab308'  # yellow
        elif overall_score >= 20:
            threat_level = 'MODERATE'
            threat_color = '#3b82f6'  # blue
        else:
            threat_level = 'LOW'
            threat_color = '#22c55e'  # green
        
        # Sort concerns by score
        concerns.sort(key=lambda x: x['score'], reverse=True)
        
        # Add default recommendation if none
        if not recommendations:
            if overall_score < 20:
                recommendations.append("Airspace conditions normal - continue standard operations")
            else:
                recommendations.append("Monitor situation and review component scores for details")
        
        return {
            'overall_score': overall_score,
            'threat_level': threat_level,
            'threat_color': threat_color,
            'components': components,
            'top_concerns': concerns[:4],
            'recommendations': recommendations[:3]
        }

    def get_military_by_country(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get military activity breakdown by country.
        
        IMPROVED: Queries flight_metadata directly for more complete data,
        including flight duration and border crossings from the database.
        
        Returns detailed statistics for each country's military presence.
        """
        countries = {}
        alerts = []
        total_flights = 0
        seen_flight_ids = set()
        
        # Query military flights from flight_metadata with additional info
        query = """
            SELECT flight_id, callsign, military_type, flight_duration_sec, 
                   crossed_borders, first_seen_ts, category
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
            ORDER BY first_seen_ts DESC
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            flight_id, callsign, military_type, duration_sec, crossed_borders, first_seen, category = row
            
            if not callsign or flight_id in seen_flight_ids:
                continue
            seen_flight_ids.add(flight_id)
            
            # Identify country and type from callsign
            mil_info = self._identify_military_aircraft(callsign)
            if not mil_info:
                continue
            
            country = mil_info.get('country', 'UNKNOWN')
            mil_type = mil_info.get('type', military_type or 'unknown')
            
            if country not in countries:
                countries[country] = {
                    'country_name': self._get_country_name(country),
                    'total_flights': 0,
                    'by_type': defaultdict(int),
                    'total_duration_hours': 0,
                    'avg_duration_hours': 0,
                    'crossed_borders': 0,
                    'recent_flights': [],
                    'anomalies': [],
                    'anomaly_count': 0
                }
            
            countries[country]['total_flights'] += 1
            total_flights += 1
            
            countries[country]['by_type'][mil_type] += 1
            
            # Add duration
            if duration_sec:
                countries[country]['total_duration_hours'] += duration_sec / 3600
            
            # Count border crossings
            if crossed_borders:
                border_count = len(crossed_borders.split(',')) if crossed_borders else 0
                if border_count > 1:
                    countries[country]['crossed_borders'] += 1
            
            # Add recent flight (up to 5 per country)
            if len(countries[country]['recent_flights']) < 5:
                countries[country]['recent_flights'].append({
                    'callsign': callsign,
                    'type': mil_type,
                    'type_name': mil_info.get('name', ''),
                    'duration_hours': round(duration_sec / 3600, 1) if duration_sec else None
                })
        
        # Calculate averages and convert defaultdicts
        for country in countries:
            countries[country]['by_type'] = dict(countries[country]['by_type'])
            if countries[country]['total_flights'] > 0:
                countries[country]['avg_duration_hours'] = round(
                    countries[country]['total_duration_hours'] / countries[country]['total_flights'], 1
                )
            del countries[country]['total_duration_hours']  # Remove intermediate field
        
        # Generate alerts for unusual activity
        ru_flights = countries.get('RU', {}).get('total_flights', 0)
        if ru_flights > 5:
            alerts.append({
                'severity': 'high' if ru_flights > 10 else 'medium',
                'message': f'Elevated Russian military activity: {ru_flights} flights detected'
            })
        
        # Alert for Iranian military
        ir_flights = countries.get('IR', {}).get('total_flights', 0)
        if ir_flights > 0:
            alerts.append({
                'severity': 'high',
                'message': f'Iranian military aircraft detected: {ir_flights} flights'
            })
        
        # Build top countries list
        top_countries = sorted(
            [{'country': k, 'flights': v['total_flights']} for k, v in countries.items()],
            key=lambda x: x['flights'],
            reverse=True
        )[:5]
        
        return {
            'countries': countries,
            'summary': {
                'total_military_flights': total_flights,
                'countries_detected': len(countries),
                'top_countries': top_countries,
                'analysis_period_days': (end_ts - start_ts) / 86400,
                'alerts': alerts
            }
        }
    
    def _get_country_name(self, code: str) -> str:
        """Get full country name from code."""
        names = {
            'US': 'United States',
            'GB': 'United Kingdom',
            'RU': 'Russia',
            'IL': 'Israel',
            'NATO': 'NATO Alliance',
            'DE': 'Germany',
            'FR': 'France',
            'IT': 'Italy',
            'AU': 'Australia',
            'CA': 'Canada',
            'TR': 'Turkey',
            'IR': 'Iran',
            'SA': 'Saudi Arabia',
            'AE': 'UAE',
            'JO': 'Jordan',
            'EG': 'Egypt'
        }
        return names.get(code, code)

    def get_military_by_destination(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Get military flights by destination region, with special focus on conflict zones.
        
        IMPROVED: Uses flight_metadata destination_airport when available,
        falls back to last track position for flights without airport data.
        
        Returns flights heading to Syria, Gaza, Lebanon with origin analysis.
        """
        by_destination = defaultdict(int)
        by_origin = defaultdict(int)
        syria_flights = []
        syria_from_east_count = 0
        total_flights = 0
        
        # Region boundaries (expanded for better coverage)
        regions = {
            'syria': {'lat_min': 32, 'lat_max': 37.5, 'lon_min': 35.5, 'lon_max': 42},
            'gaza': {'lat_min': 31.2, 'lat_max': 31.6, 'lon_min': 34.2, 'lon_max': 34.6},
            'lebanon': {'lat_min': 33, 'lat_max': 34.5, 'lon_min': 35, 'lon_max': 36.5},
            'israel': {'lat_min': 29, 'lat_max': 33.5, 'lon_min': 34, 'lon_max': 36},
            'jordan': {'lat_min': 29, 'lat_max': 33.5, 'lon_min': 35, 'lon_max': 39},
            'iraq': {'lat_min': 29, 'lat_max': 37.5, 'lon_min': 38.5, 'lon_max': 48.5},
            'iran': {'lat_min': 25, 'lat_max': 40, 'lon_min': 44, 'lon_max': 63},
            'turkey': {'lat_min': 36, 'lat_max': 42, 'lon_min': 26, 'lon_max': 45},
            'saudi_arabia': {'lat_min': 16, 'lat_max': 32, 'lon_min': 34.5, 'lon_max': 55},
            'egypt': {'lat_min': 22, 'lat_max': 32, 'lon_min': 24.5, 'lon_max': 36.5},
            'cyprus': {'lat_min': 34.5, 'lat_max': 35.7, 'lon_min': 32, 'lon_max': 34.6}
        }
        
        # Syrian airports (ICAO codes)
        syrian_airports = {'OSDI', 'OSLK', 'OSAP', 'OSPR', 'OSDZ', 'OSKL'}
        
        eastern_countries = {'RU', 'IR'}  # Russia, Iran
        
        # Query military flights from flight_metadata with airport info
        query = """
            SELECT flight_id, callsign, origin_airport, destination_airport,
                   origin_lat, origin_lon, dest_lat, dest_lon, is_military, category
            FROM flight_metadata
            WHERE first_seen_ts BETWEEN ? AND ?
            AND (is_military = 1 OR category = 'Military_and_government')
        """
        results = self._execute_query('research', query, (start_ts, end_ts))
        
        for row in results:
            flight_id, callsign, origin_apt, dest_apt, origin_lat, origin_lon, dest_lat, dest_lon, is_mil, category = row
            
            if not callsign:
                continue
            
            # Identify country from callsign
            mil_info = self._identify_military_aircraft(callsign)
            if not mil_info:
                continue
            
            country = mil_info.get('country', 'Unknown')
            total_flights += 1
            
            # Determine destination region
            dest_region = 'other'
            
            # Method 1: Use destination airport if available
            if dest_apt and dest_apt in syrian_airports:
                dest_region = 'syria'
            elif dest_lat and dest_lon:
                # Method 2: Use destination coordinates
                for region, bounds in regions.items():
                    if (bounds['lat_min'] <= dest_lat <= bounds['lat_max'] and
                        bounds['lon_min'] <= dest_lon <= bounds['lon_max']):
                        dest_region = region
                        break
            
            by_destination[dest_region] += 1
            
            # Determine origin region
            origin_region = 'other'
            if origin_lat and origin_lon:
                if origin_lon > 44:  # East of Iraq
                    origin_region = 'east'
                elif origin_lon < 30:  # West Mediterranean
                    origin_region = 'west'
                else:
                    origin_region = 'regional'
                by_origin[origin_region] += 1
            
            # Track Syria-bound flights specially
            if dest_region == 'syria':
                is_from_east = country in eastern_countries or origin_region == 'east'
                if is_from_east:
                    syria_from_east_count += 1
                
                syria_flights.append({
                    'callsign': callsign,
                    'country': country,
                    'type': mil_info.get('type', 'Unknown'),
                    'origin_region': origin_region,
                    'is_from_east': is_from_east,
                    'concern_level': 'high' if is_from_east else 'medium',
                    'origin_airport': origin_apt,
                    'destination_airport': dest_apt
                })
        
        return {
            'total_flights': total_flights,
            'by_destination': dict(by_destination),
            'by_origin': dict(by_origin),
            'syria_flights': syria_flights,
            'syria_from_east_count': syria_from_east_count
        }

    def get_bilateral_proximity_events(self, start_ts: int, end_ts: int,
                                        proximity_threshold_nm: float = 50,
                                        time_window_sec: int = 300) -> Dict[str, Any]:
        """
        Detect close approaches between military aircraft from different nations.
        
        CRITICAL FIX: Only detect proximity when aircraft were near each other 
        AT THE SAME TIME (within time_window_sec seconds).
        
        Args:
            proximity_threshold_nm: Distance threshold for proximity (default 50nm)
            time_window_sec: Time window to consider "same time" (default 5 minutes)
        
        Returns events where aircraft from potentially opposing nations came within
        proximity_threshold_nm of each other at approximately the same time.
        """
        mil_patterns = self.get_military_patterns(start_ts, end_ts)
        
        events = []
        by_pair = defaultdict(int)
        high_risk_events = 0
        alerts = []
        
        # High-interest country pairs
        high_interest_pairs = {
            frozenset(['US', 'RU']), frozenset(['US', 'IR']),
            frozenset(['IL', 'RU']), frozenset(['IL', 'IR']),
            frozenset(['NATO', 'RU'])
        }
        
        # Check each pair of military aircraft
        for i, p1 in enumerate(mil_patterns):
            for p2 in mil_patterns[i+1:]:
                c1, c2 = p1.get('country', ''), p2.get('country', '')
                if c1 == c2:
                    continue  # Same country
                
                locs1 = p1.get('locations', [])
                locs2 = p2.get('locations', [])
                
                # Skip if no locations with timestamps
                if not locs1 or not locs2:
                    continue
                
                # Find closest approach WITHIN THE SAME TIME WINDOW
                min_dist = float('inf')
                closest_loc = None
                event_timestamp = None
                
                for loc1 in locs1:
                    ts1 = loc1.get('timestamp', 0)
                    if not ts1:
                        continue
                        
                    for loc2 in locs2:
                        ts2 = loc2.get('timestamp', 0)
                        if not ts2:
                            continue
                        
                        # CRITICAL: Only consider if within time window
                        time_diff = abs(ts1 - ts2)
                        if time_diff > time_window_sec:
                            continue
                        
                        dist = _haversine_nm(
                            loc1.get('lat', 0), loc1.get('lon', 0),
                            loc2.get('lat', 0), loc2.get('lon', 0)
                        )
                        if dist < min_dist:
                            min_dist = dist
                            closest_loc = {
                                'lat': (loc1.get('lat', 0) + loc2.get('lat', 0)) / 2,
                                'lon': (loc1.get('lon', 0) + loc2.get('lon', 0)) / 2
                            }
                            event_timestamp = (ts1 + ts2) // 2
                
                if min_dist <= proximity_threshold_nm and closest_loc:
                    pair_key = '-'.join(sorted([c1, c2]))
                    is_high_interest = frozenset([c1, c2]) in high_interest_pairs
                    
                    # Calculate severity - adjusted for actual proximity events
                    if min_dist < 5:
                        severity = 'critical'
                        severity_score = 100
                    elif min_dist < 15:
                        severity = 'high'
                        severity_score = 75
                    elif min_dist < 30:
                        severity = 'medium'
                        severity_score = 50
                    else:
                        severity = 'low'
                        severity_score = 25
                    
                    if is_high_interest:
                        severity_score = min(100, severity_score + 25)
                    
                    by_pair[pair_key] += 1
                    
                    if severity in ['critical', 'high']:
                        high_risk_events += 1
                    
                    events.append({
                        'pair_name': pair_key,
                        'callsign1': p1.get('callsign', 'Unknown'),
                        'country1': c1,
                        'type1': p1.get('type', 'Unknown'),
                        'callsign2': p2.get('callsign', 'Unknown'),
                        'country2': c2,
                        'type2': p2.get('type', 'Unknown'),
                        'min_distance_nm': round(min_dist, 1),
                        'severity': severity,
                        'severity_score': severity_score,
                        'is_high_interest': is_high_interest,
                        'location': closest_loc,
                        'timestamp': event_timestamp
                    })
        
        # Generate alerts
        if high_risk_events > 0:
            alerts.append({
                'severity': 'critical' if high_risk_events > 3 else 'high',
                'message': f'{high_risk_events} high-risk proximity events detected between different nations'
            })
        
        # Sort events by severity
        events.sort(key=lambda x: x['severity_score'], reverse=True)
        
        return {
            'events': events[:50],  # Limit for response size
            'by_pair': dict(by_pair),
            'total_events': len(events),
            'high_risk_events': high_risk_events,
            'proximity_threshold_nm': proximity_threshold_nm,
            'time_window_sec': time_window_sec,
            'alerts': alerts
        }

    def get_combined_signal_map(self, start_ts: int, end_ts: int, limit: int = 50) -> Dict[str, Any]:
        """
        Get combined GPS Jamming and Signal Loss data for unified map visualization.
        
        Returns both datasets with distinct styling so they can be displayed on the same map:
        
        - GPS JAMMING (RED #ef4444): Active interference with altitude jumps, position teleports
        - SIGNAL LOSS (ORANGE #f97316): Coverage gaps where tracking was lost for 5+ minutes
        
        Returns:
            {
                'points': [{
                    'type': 'jamming' | 'signal_loss',
                    'lat': float,
                    'lon': float,
                    'intensity': int (0-100),
                    'event_count': int,
                    'affected_flights': int,
                    'color': str,
                    ... type-specific fields
                }],
                'summary': {...},
                'legend': {...}
            }
        """
        # Color definitions
        JAMMING_COLOR = '#ef4444'      # Red - active interference
        SIGNAL_LOSS_COLOR = '#f97316'  # Orange - passive coverage gaps
        
        points = []
        
        # =====================================================================
        # Get GPS Jamming Data
        # =====================================================================
        jamming_data = self.get_gps_jamming_heatmap(start_ts, end_ts, limit)
        
        for jp in jamming_data:
            points.append({
                'type': 'jamming',
                'lat': jp.get('lat', 0),
                'lon': jp.get('lon', 0),
                'intensity': jp.get('jamming_score', jp.get('intensity', 50)),
                'event_count': jp.get('event_count', 1),
                'affected_flights': jp.get('affected_flights', 1),
                'color': JAMMING_COLOR,
                'confidence': jp.get('jamming_confidence', 'MEDIUM'),
                'indicators': jp.get('jamming_indicators', []),
                'first_seen': jp.get('first_seen'),
                'last_seen': jp.get('last_seen'),
            })
        
        # =====================================================================
        # Get Signal Loss Data
        # =====================================================================
        signal_loss_data = self.get_signal_loss_zones(start_ts, end_ts, limit)
        
        for sl in signal_loss_data:
            count = sl.get('count', sl.get('event_count', 1))
            avg_duration = sl.get('avg_duration', sl.get('avgDuration', 300))
            # Intensity based on count and duration
            intensity = min(100, int(count * 5 + avg_duration / 60))
            
            points.append({
                'type': 'signal_loss',
                'lat': sl.get('lat', 0),
                'lon': sl.get('lon', 0),
                'intensity': intensity,
                'event_count': count,
                'affected_flights': sl.get('affected_flights', count),
                'color': SIGNAL_LOSS_COLOR,
                'avg_gap_duration_sec': int(avg_duration),
                'first_seen': sl.get('first_seen'),
                'last_seen': sl.get('last_seen'),
            })
        
        # Sort by intensity (highest first)
        points.sort(key=lambda p: p['intensity'], reverse=True)
        
        # Calculate summary
        jamming_points = [p for p in points if p['type'] == 'jamming']
        signal_loss_points = [p for p in points if p['type'] == 'signal_loss']
        
        total_jamming = sum(p['event_count'] for p in jamming_points)
        total_signal_loss = sum(p['event_count'] for p in signal_loss_points)
        
        return {
            'points': points,
            'summary': {
                'total_jamming_events': total_jamming,
                'total_signal_loss_events': total_signal_loss,
                'jamming_zones': len(jamming_points),
                'signal_loss_zones': len(signal_loss_points),
                'total_zones': len(points),
            },
            'legend': {
                'jamming': {
                    'color': JAMMING_COLOR,
                    'label': 'GPS Jamming',
                    'description': 'Active interference - altitude jumps, position teleports, heading anomalies',
                    'icon': '🔺'
                },
                'signal_loss': {
                    'color': SIGNAL_LOSS_COLOR,
                    'label': 'Signal Loss',
                    'description': 'Coverage gaps - tracking lost for 5+ minutes',
                    'icon': '📡'
                }
            }
        }

    def get_jamming_source_triangulation(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Estimate potential GPS jamming source locations based on affected flight patterns.
        
        Uses clustering of jamming events to triangulate likely source positions.
        """
        try:
            # Use cached GPS jamming data if available
            if hasattr(self, '_cached_gps_jamming_extended') and self._cached_gps_jamming_extended:
                gps_data = self._cached_gps_jamming_extended
            elif hasattr(self, '_cached_gps_jamming') and self._cached_gps_jamming:
                gps_data = self._cached_gps_jamming
            else:
                gps_data = self.get_gps_jamming_heatmap(start_ts, end_ts, 100)
            
            if not gps_data:
                return {
                    'estimated_sources': [],
                    'total_affected_flights': 0,
                    'triangulation_quality': 'insufficient_data'
                }
            
            # Group jamming events by rough area (0.5 degree grid)
            grid_size = 0.5
            clusters = defaultdict(list)
            total_affected = 0
            
            for point in gps_data:
                lat = point.get('lat', 0)
                lon = point.get('lon', 0)
                grid_key = (round(lat / grid_size) * grid_size, round(lon / grid_size) * grid_size)
                clusters[grid_key].append(point)
                total_affected += point.get('affected_flights', 0)
            
            # Estimate source for each cluster
            estimated_sources = []
            for (grid_lat, grid_lon), points in clusters.items():
                if len(points) < 2:
                    continue
                
                # Calculate weighted centroid
                total_weight = sum(p.get('jamming_score', 1) for p in points)
                if total_weight == 0:
                    continue
                
                avg_lat = sum(p.get('lat', 0) * p.get('jamming_score', 1) for p in points) / total_weight
                avg_lon = sum(p.get('lon', 0) * p.get('jamming_score', 1) for p in points) / total_weight
                avg_score = total_weight / len(points)
                
                # Estimate operator
                operator = self._estimate_ew_operator(avg_lat, avg_lon)
                
                estimated_sources.append({
                    'lat': round(avg_lat, 3),
                    'lon': round(avg_lon, 3),
                    'confidence': 'high' if len(points) >= 5 else 'medium' if len(points) >= 3 else 'low',
                    'affected_points': len(points),
                    'avg_jamming_score': round(avg_score, 1),
                    'estimated_operator': operator
                })
            
            # Sort by affected points
            estimated_sources.sort(key=lambda x: x['affected_points'], reverse=True)
            
            quality = 'good' if len(estimated_sources) >= 3 else 'moderate' if len(estimated_sources) >= 1 else 'insufficient_data'
            
            return {
                'estimated_sources': estimated_sources[:10],
                'total_affected_flights': total_affected,
                'triangulation_quality': quality
            }
            
        except Exception as e:
            print(f"[JammingTriangulation] Error: {e}")
            return {
                'estimated_sources': [],
                'total_affected_flights': 0,
                'triangulation_quality': 'error'
            }

