"""
Predictive analytics and risk scoring for Level 4 features.

Provides:
- Airspace risk scoring
- Trajectory prediction
- Threat assessment
- Safety forecasting
- Hostile intent prediction
"""
from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import math


# Restricted zones / sensitive areas for trajectory breach detection
# Format: {zone_id: {name, lat, lon, radius_nm, type, severity}}
RESTRICTED_ZONES = {
    'LLBG_TMA': {
        'name': 'Ben Gurion TMA',
        'lat': 32.011389,
        'lon': 34.886667,
        'radius_nm': 25,
        'type': 'airport_tma',
        'severity': 'medium'
    },
    'BORDER_NORTH': {
        'name': 'Northern Border Zone',
        'lat': 33.1,
        'lon': 35.5,
        'radius_nm': 15,
        'type': 'border',
        'severity': 'high'
    },
    'BORDER_EAST': {
        'name': 'Eastern Border Zone',
        'lat': 31.5,
        'lon': 35.5,
        'radius_nm': 10,
        'type': 'border',
        'severity': 'high'
    },
    'DIMONA': {
        'name': 'Dimona Restricted',
        'lat': 31.0,
        'lon': 35.15,
        'radius_nm': 20,
        'type': 'restricted',
        'severity': 'critical'
    },
    'GAZA_BUFFER': {
        'name': 'Gaza Buffer Zone',
        'lat': 31.4,
        'lon': 34.4,
        'radius_nm': 15,
        'type': 'conflict',
        'severity': 'critical'
    },
    'LEBANON_FIR': {
        'name': 'Lebanon FIR Border',
        'lat': 33.3,
        'lon': 35.5,
        'radius_nm': 10,
        'type': 'fir_boundary',
        'severity': 'high'
    },
    'SYRIA_FIR': {
        'name': 'Syria FIR Border',
        'lat': 32.8,
        'lon': 36.0,
        'radius_nm': 15,
        'type': 'fir_boundary',
        'severity': 'high'
    }
}


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


class PredictiveAnalytics:
    """Engine for predictive analytics and risk assessment."""
    
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
    
    def calculate_airspace_risk(self) -> Dict[str, Any]:
        """
        Calculate real-time airspace risk score.
        
        Returns:
            {
                risk_score: 0-100,
                risk_level: str ('low', 'medium', 'high', 'critical'),
                factors: [{name, weight, value, impact}],
                recommendation: str
            }
        """
        now = int(datetime.now().timestamp())
        last_hour = now - 3600
        
        factors = []
        total_score = 0
        
        # Factor 1: Traffic density (0-30 points)
        query = """
            SELECT COUNT(DISTINCT flight_id)
            FROM flight_tracks
            WHERE timestamp >= ?
        """
        results = self._execute_query('live', query, (last_hour,))
        traffic_count = results[0][0] if results else 0
        
        traffic_score = min(30, traffic_count * 0.5)  # Scale: 60 flights = max
        total_score += traffic_score
        factors.append({
            'name': 'Traffic Density',
            'weight': 30,
            'value': traffic_count,
            'impact': round(traffic_score, 1),
            'description': f'{traffic_count} active flights in last hour'
        })
        
        # Factor 2: Recent anomalies (0-40 points) - count distinct flights
        query = """
            SELECT COUNT(DISTINCT json_extract(full_report, '$.summary.flight_id'))
            FROM anomaly_reports
            WHERE timestamp >= ?
        """
        results = self._execute_query('research', query, (last_hour,))
        anomaly_count = results[0][0] if results else 0
        
        anomaly_score = min(40, anomaly_count * 8)  # Scale: 5 anomalies = max
        total_score += anomaly_score
        factors.append({
            'name': 'Recent Anomalies',
            'weight': 40,
            'value': anomaly_count,
            'impact': round(anomaly_score, 1),
            'description': f'{anomaly_count} anomalies detected in last hour'
        })
        
        # Factor 3: Time of day (0-15 points)
        hour = datetime.now().hour
        # Peak hours: 6-9am, 5-8pm local time = higher risk
        if hour in [6, 7, 8, 17, 18, 19]:
            time_score = 15
            time_desc = 'Peak traffic hours'
        elif hour in [0, 1, 2, 3, 4, 5]:
            time_score = 3
            time_desc = 'Night hours (low traffic)'
        else:
            time_score = 8
            time_desc = 'Normal operating hours'
        
        total_score += time_score
        factors.append({
            'name': 'Time of Day',
            'weight': 15,
            'value': hour,
            'impact': time_score,
            'description': time_desc
        })
        
        # Factor 4: Safety event trend (0-15 points)
        # Look at last 24h vs previous 24h
        day_ago = now - 86400
        two_days_ago = now - 172800
        
        query = """
            SELECT COUNT(DISTINCT json_extract(full_report, '$.summary.flight_id'))
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        recent = self._execute_query('research', query, (day_ago, now))
        previous = self._execute_query('research', query, (two_days_ago, day_ago))
        
        recent_count = recent[0][0] if recent else 0
        previous_count = previous[0][0] if previous else 1
        
        trend_ratio = recent_count / max(previous_count, 1)
        trend_score = min(15, (trend_ratio - 1) * 15) if trend_ratio > 1 else 0
        total_score += trend_score
        
        factors.append({
            'name': 'Safety Trend',
            'weight': 15,
            'value': round(trend_ratio, 2),
            'impact': round(trend_score, 1),
            'description': f'Events {"increasing" if trend_ratio > 1.1 else "stable"}'
        })
        
        # Factor 5: Weather Impact (0-15 points) - NEWLY ADDED
        # Get weather for central region (Tel Aviv area as reference)
        weather = self.get_weather_impact(32.0, 34.8)
        weather_score = (weather['impact_score'] / 100) * 15
        total_score += weather_score
        
        factors.append({
            'name': 'Weather Conditions',
            'weight': 15,
            'value': weather['conditions'],
            'impact': round(weather_score, 1),
            'description': f"Visibility: {weather['visibility_miles']}mi, Wind: {weather['wind_speed_kts']}kts"
        })
        
        # Determine risk level
        if total_score >= 75:
            risk_level = 'critical'
            recommendation = 'Consider adding additional ATC coverage and heightened alert status'
        elif total_score >= 50:
            risk_level = 'high'
            recommendation = 'Monitor closely and prepare contingency measures'
        elif total_score >= 25:
            risk_level = 'medium'
            recommendation = 'Maintain standard vigilance'
        else:
            risk_level = 'low'
            recommendation = 'Normal operations'
        
        return {
            'risk_score': round(total_score, 1),
            'risk_level': risk_level,
            'factors': factors,
            'recommendation': recommendation,
            'timestamp': now
        }
    
    def predict_trajectory(self, flight_id: str, 
                          current_position: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict flight trajectory for next 3-5 minutes (kinematic prediction).
        
        Includes restricted zone breach detection using predefined sensitive areas.
        
        Args:
            flight_id: Flight to predict
            current_position: {lat, lon, alt, heading, speed}
        
        Returns:
            {
                predicted_path: [[lon, lat, timestamp]],
                breach_warning: bool,
                breach_time_seconds: int,
                breach_zone: str,
                breach_severity: str,
                confidence: float,
                zones_checked: int
            }
        """
        lat = current_position.get('lat', 0)
        lon = current_position.get('lon', 0)
        heading = current_position.get('heading', 0)
        speed_kts = current_position.get('speed', 300)
        alt = current_position.get('alt', 0)
        
        # Kinematic prediction (straight-line with current heading)
        predicted_path = []
        time_steps = [30, 60, 90, 120, 180, 240, 300]  # Seconds (up to 5 minutes)
        
        for dt in time_steps:
            # Distance traveled (nautical miles)
            distance_nm = (speed_kts / 3600) * dt
            
            # Convert to lat/lon offset (simplified spherical)
            dlat = distance_nm * math.cos(math.radians(heading)) / 60
            dlon = distance_nm * math.sin(math.radians(heading)) / (60 * math.cos(math.radians(lat))) if lat != 0 else 0
            
            pred_lat = lat + dlat
            pred_lon = lon + dlon
            pred_time = int(datetime.now().timestamp()) + dt
            
            predicted_path.append([pred_lon, pred_lat, pred_time])
        
        # Check for restricted zone breaches
        breach_warning = False
        breach_time_seconds = None
        breach_zone = None
        breach_severity = None
        closest_zone = None
        closest_distance = float('inf')
        
        # Check current position and predicted positions against restricted zones
        all_positions = [(lat, lon, 0)] + [(p[1], p[0], time_steps[i]) for i, p in enumerate(predicted_path)]
        
        for zone_id, zone in RESTRICTED_ZONES.items():
            zone_lat = zone['lat']
            zone_lon = zone['lon']
            zone_radius = zone['radius_nm']
            
            for pos_lat, pos_lon, time_offset in all_positions:
                distance = _haversine_nm(pos_lat, pos_lon, zone_lat, zone_lon)
                
                # Track closest approach
                if distance < closest_distance:
                    closest_distance = distance
                    closest_zone = zone_id
                
                # Check for breach (entering the zone)
                if distance < zone_radius:
                    # Breach detected!
                    if not breach_warning or (time_offset < breach_time_seconds):
                        breach_warning = True
                        breach_time_seconds = time_offset
                        breach_zone = zone['name']
                        breach_severity = zone['severity']
                        break
            
            if breach_warning:
                break
        
        # Calculate confidence based on speed consistency and data quality
        confidence = self._calculate_trajectory_confidence(current_position, flight_id)
        
        return {
            'predicted_path': predicted_path,
            'breach_warning': breach_warning,
            'breach_time_seconds': breach_time_seconds,
            'breach_zone': breach_zone,
            'breach_severity': breach_severity,
            'closest_zone': closest_zone,
            'closest_distance_nm': round(closest_distance, 1) if closest_distance != float('inf') else None,
            'confidence': confidence,
            'zones_checked': len(RESTRICTED_ZONES)
        }
    
    def _calculate_trajectory_confidence(self, current_position: Dict[str, float], 
                                         flight_id: str) -> float:
        """
        Calculate confidence score for trajectory prediction.
        
        Based on:
        - Speed consistency (reasonable range)
        - Altitude (higher = more predictable)
        - Data completeness
        """
        confidence = 0.5  # Base confidence
        
        speed = current_position.get('speed', 0)
        alt = current_position.get('alt', 0)
        heading = current_position.get('heading')
        
        # Speed in reasonable range (150-550 kts for commercial)
        if 150 <= speed <= 550:
            confidence += 0.2
        elif 50 <= speed <= 700:
            confidence += 0.1
        
        # Higher altitude = more predictable
        if alt > 20000:
            confidence += 0.15
        elif alt > 10000:
            confidence += 0.1
        elif alt > 5000:
            confidence += 0.05
        
        # Has heading data
        if heading is not None:
            confidence += 0.1
        
        # Has all required data
        if all(current_position.get(k) for k in ['lat', 'lon', 'speed']):
            confidence += 0.05
        
        return min(0.95, round(confidence, 2))
    
    def forecast_safety_events(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """
        IMPROVED safety forecast using historical patterns and time-of-day analysis.
        
        Algorithm:
        1. Analyze last 30 days of safety events by hour and event type
        2. Calculate hourly patterns (which hours have more events)
        3. Apply trend analysis (increasing/decreasing pattern)
        4. Generate hour-by-hour forecast with confidence scores
        5. Include event type breakdown (go-arounds, near-miss, emergency)
        
        Returns:
            {
                'forecast': [{hour_offset, datetime, expected_events_by_type, risk_level, confidence}],
                'baseline_events_per_hour': float,
                'trend': 'increasing'|'stable'|'decreasing',
                'total_expected': int,
                'model': 'time_series_with_patterns',
                'confidence_score': float
            }
        """
        from datetime import timedelta
        
        now = int(datetime.now().timestamp())
        thirty_days_ago = now - (30 * 24 * 3600)
        
        # 1. Get historical safety event patterns by hour and type
        hourly_patterns = Counter()  # hour -> count
        hourly_by_type = {h: {'go_arounds': 0, 'near_miss': 0, 'emergency': 0} for h in range(24)}
        
        query = """
            SELECT 
                strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                full_report
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        results = self._execute_query('research', query, (thirty_days_ago, now))
        
        total_events = 0
        for row in results:
            hour_str, report_json = row
            if not hour_str:
                continue
                
            hour = int(hour_str)
            hourly_patterns[hour] += 1
            total_events += 1
            
            # Parse event type
            try:
                import json
                report = json.loads(report_json) if isinstance(report_json, str) else report_json
                matched_rules = report.get('layer_1_rules', {}).get('report', {}).get('matched_rules', [])
                
                for rule in matched_rules:
                    rule_id = rule.get('id')
                    if rule_id == 1:  # Emergency squawk
                        hourly_by_type[hour]['emergency'] += 1
                    elif rule_id == 4:  # Near-miss
                        hourly_by_type[hour]['near_miss'] += 1
                    elif rule_id == 6:  # Go-around
                        hourly_by_type[hour]['go_arounds'] += 1
            except:
                pass
        
        # 2. Calculate baseline (average events per hour)
        baseline = total_events / (30 * 24) if total_events > 0 else 0.1
        
        # 3. Analyze trend (last 7 days vs previous 23 days)
        seven_days_ago = now - (7 * 24 * 3600)
        
        query_trend = """
            SELECT COUNT(DISTINCT json_extract(full_report, '$.summary.flight_id'))
            FROM anomaly_reports
            WHERE timestamp BETWEEN ? AND ?
        """
        recent = self._execute_query('research', query_trend, (seven_days_ago, now))
        older = self._execute_query('research', query_trend, (thirty_days_ago, seven_days_ago))
        
        recent_count = recent[0][0] if recent and recent[0] else 0
        older_count = older[0][0] if older and older[0] else 0
        
        if older_count > 0:
            recent_rate = recent_count / 7
            older_rate = older_count / 23
            trend_ratio = recent_rate / older_rate if older_rate > 0 else 1.0
            
            if trend_ratio > 1.2:
                trend = 'increasing'
            elif trend_ratio < 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            trend_ratio = 1.0
        
        # 4. Generate hour-by-hour forecast
        forecast = []
        current_dt = datetime.now()
        total_expected = 0
        
        for offset in range(hours_ahead):
            target_time = current_dt + timedelta(hours=offset)
            target_hour = target_time.hour
            
            # Get historical average for this hour
            hour_historical_avg = hourly_patterns.get(target_hour, 0) / 30  # Per day average
            
            # Apply trend
            expected = hour_historical_avg * trend_ratio
            total_expected += expected
            
            # Get event type breakdown
            hour_types = hourly_by_type[target_hour]
            total_hour_events = sum(hour_types.values())
            
            if total_hour_events > 0:
                go_around_pct = hour_types['go_arounds'] / total_hour_events
                near_miss_pct = hour_types['near_miss'] / total_hour_events
                emergency_pct = hour_types['emergency'] / total_hour_events
            else:
                go_around_pct = 0.6
                near_miss_pct = 0.3
                emergency_pct = 0.1
            
            expected_go_arounds = max(0, int(round(expected * go_around_pct)))
            expected_near_miss = max(0, int(round(expected * near_miss_pct)))
            expected_emergency = max(0, int(round(expected * emergency_pct)))
            
            # Risk level based on expected events vs baseline
            if expected > baseline * 1.5:
                risk_level = 'high'
            elif expected > baseline:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Confidence decreases with forecast distance
            confidence = max(0.5, 1.0 - (offset / hours_ahead) * 0.4)
            
            forecast.append({
                'hour_offset': offset,
                'datetime': target_time.isoformat(),
                'hour': target_hour,
                'expected_go_arounds': expected_go_arounds,
                'expected_near_miss': expected_near_miss,
                'expected_emergency': expected_emergency,
                'expected_total': round(expected, 2),
                'risk_level': risk_level,
                'confidence': round(confidence, 2)
            })
        
        # Find peak risk hours in forecast
        peak_hours = sorted(hourly_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'forecast': forecast,
            'baseline_events_per_hour': round(baseline, 3),
            'trend': trend,
            'trend_ratio': round(trend_ratio, 2),
            'total_expected': int(round(total_expected)),
            'historical_data_days': 30,
            'total_historical_events': total_events,
            'peak_risk_hours': [h for h, _ in peak_hours],
            'model': 'time_series_with_hourly_patterns',
            'confidence_score': 0.75 if total_events > 100 else 0.5
        }
    
    def predict_trajectory(self, flight_id: str, current_position: Dict[str, float]) -> Dict[str, Any]:
        """
        IMPROVED trajectory prediction using kinematic model.
        Predicts flight path for next 5 minutes using physics-based model.
        
        Args:
            flight_id: Current flight identifier
            current_position: {lat, lon, alt, heading, speed}
        
        Returns:
            {predicted_path, breach_warnings, confidence, model_used}
        """
        lat = current_position.get('lat')
        lon = current_position.get('lon')
        alt = current_position.get('alt', 0)
        heading = current_position.get('heading', 0)
        speed = current_position.get('speed', 0)
        
        if not lat or not lon:
            return {'error': 'Missing position data'}
        
        # Predict every 15 seconds for next 5 minutes
        predicted_path = []
        for seconds in range(15, 301, 15):
            distance_nm = (speed * seconds) / 3600
            
            # Simple dead reckoning
            lat_change = distance_nm * math.cos(math.radians(heading)) / 60
            lon_change = distance_nm * math.sin(math.radians(heading)) / (60 * math.cos(math.radians(lat)))
            
            predicted_path.append({
                'time_offset_seconds': seconds,
                'lat': lat + lat_change,
                'lon': lon + lon_change,
                'alt': int(alt)
            })
        
        # Check for restricted airspace breaches
        breach_warnings = []
        restricted_zones = [
            {'name': 'Lebanese Airspace', 'lat_min': 33.05, 'lat_max': 34.65, 'lon_min': 35.1, 'lon_max': 36.6},
            {'name': 'Syrian Airspace', 'lat_min': 32.3, 'lat_max': 37.3, 'lon_min': 35.7, 'lon_max': 42.4}
        ]
        
        for point in predicted_path:
            for zone in restricted_zones:
                if (zone['lat_min'] <= point['lat'] <= zone['lat_max'] and
                    zone['lon_min'] <= point['lon'] <= zone['lon_max']):
                    breach_warnings.append({
                        'zone_name': zone['name'],
                        'time_to_breach_seconds': point['time_offset_seconds'],
                        'severity': 'critical' if point['time_offset_seconds'] < 120 else 'warning'
                    })
                    break
        
        return {
            'flight_id': flight_id,
            'predicted_path': predicted_path,
            'breach_warnings': breach_warnings,
            'confidence': 0.75,
            'model_used': 'kinematic_dead_reckoning'
        }
    
    def get_weather_impact(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get current weather impact using Aviation Weather API.
        
        Uses: https://aviationweather.gov/api/data/metar
        
        Args:
            lat, lon: Geographic coordinates
        
        Returns:
            {
                'conditions': str,
                'visibility_miles': float,
                'wind_speed_kts': int,
                'impact_score': float  # 0-100, higher = worse
            }
        """
        try:
            import requests
            
            # Aviation Weather API - METAR data
            url = f"https://aviationweather.gov/api/data/metar?format=json&bbox={lon-1},{lat-1},{lon+1},{lat+1}"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    metar = data[0]
                    
                    # Parse weather conditions
                    visibility = metar.get('vis_statute_mi', 10)
                    wind_speed = metar.get('wind_speed_kt', 0)
                    conditions = metar.get('wx_string', 'Clear')
                    
                    # Calculate impact score
                    impact = 0
                    if visibility < 3:
                        impact += 40  # Poor visibility
                    elif visibility < 5:
                        impact += 20
                    
                    if wind_speed > 30:
                        impact += 30  # Strong winds
                    elif wind_speed > 20:
                        impact += 15
                    
                    if 'TS' in conditions:  # Thunderstorm
                        impact += 50
                    elif 'RA' in conditions:  # Rain
                        impact += 10
                    
                    return {
                        'conditions': conditions,
                        'visibility_miles': visibility,
                        'wind_speed_kts': wind_speed,
                        'impact_score': min(100, impact),
                        'source': 'aviationweather.gov'
                    }
        except:
            pass
        
        # Default / fallback
        return {
            'conditions': 'Unknown',
            'visibility_miles': 10,
            'wind_speed_kts': 0,
            'impact_score': 0,
            'source': 'default'
        }
    
    def predict_hostile_intent(self, flight_id: str, 
                               track_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict hostile intent based on flight behavior analysis.
        
        Analyzes multiple factors:
        1. Deviation from filed/expected route
        2. Proximity to sensitive areas
        3. Sudden heading changes toward restricted zones
        4. Altitude drops without approach clearance
        5. Speed changes inconsistent with normal operations
        6. Historical pattern of callsign/operator
        
        Args:
            flight_id: Flight identifier
            track_data: List of {timestamp, lat, lon, alt, heading, speed}
        
        Returns:
            {
                intent_score: 0-100,
                risk_level: 'low'|'medium'|'high'|'critical',
                factors: [{name, score, description}],
                recommendation: str,
                confidence: float
            }
        """
        if not track_data or len(track_data) < 5:
            return {
                'intent_score': 0,
                'risk_level': 'unknown',
                'factors': [],
                'recommendation': 'Insufficient data for analysis',
                'confidence': 0.0
            }
        
        factors = []
        total_score = 0
        
        # Sort track data by timestamp
        sorted_track = sorted(track_data, key=lambda x: x.get('timestamp', 0))
        
        # Factor 1: Proximity to sensitive zones (0-30 points)
        proximity_score, proximity_desc = self._analyze_zone_proximity(sorted_track)
        factors.append({
            'name': 'Zone Proximity',
            'score': proximity_score,
            'description': proximity_desc
        })
        total_score += proximity_score
        
        # Factor 2: Heading changes toward sensitive areas (0-25 points)
        heading_score, heading_desc = self._analyze_heading_changes(sorted_track)
        factors.append({
            'name': 'Heading Behavior',
            'score': heading_score,
            'description': heading_desc
        })
        total_score += heading_score
        
        # Factor 3: Altitude profile anomalies (0-20 points)
        altitude_score, altitude_desc = self._analyze_altitude_profile(sorted_track)
        factors.append({
            'name': 'Altitude Profile',
            'score': altitude_score,
            'description': altitude_desc
        })
        total_score += altitude_score
        
        # Factor 4: Speed anomalies (0-15 points)
        speed_score, speed_desc = self._analyze_speed_anomalies(sorted_track)
        factors.append({
            'name': 'Speed Anomalies',
            'score': speed_score,
            'description': speed_desc
        })
        total_score += speed_score
        
        # Factor 5: Historical operator analysis (0-10 points)
        operator_score, operator_desc = self._analyze_operator_history(flight_id, sorted_track)
        factors.append({
            'name': 'Operator History',
            'score': operator_score,
            'description': operator_desc
        })
        total_score += operator_score
        
        # Determine risk level
        if total_score >= 70:
            risk_level = 'critical'
            recommendation = 'IMMEDIATE ATTENTION REQUIRED - Contact ATC and security immediately'
        elif total_score >= 50:
            risk_level = 'high'
            recommendation = 'High-priority monitoring - Prepare for possible interception'
        elif total_score >= 30:
            risk_level = 'medium'
            recommendation = 'Enhanced monitoring recommended - Track closely'
        else:
            risk_level = 'low'
            recommendation = 'Normal monitoring - No immediate concern'
        
        # Calculate confidence based on data quality
        confidence = min(0.95, 0.5 + (len(sorted_track) / 100) * 0.3)
        
        return {
            'intent_score': min(100, total_score),
            'risk_level': risk_level,
            'factors': factors,
            'recommendation': recommendation,
            'confidence': round(confidence, 2),
            'track_points_analyzed': len(sorted_track)
        }
    
    def _analyze_zone_proximity(self, track_data: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Analyze proximity to sensitive zones."""
        min_distance = float('inf')
        closest_zone = None
        critical_approach = False
        
        for point in track_data[-10:]:  # Check recent points
            lat = point.get('lat', 0)
            lon = point.get('lon', 0)
            
            for zone_id, zone in RESTRICTED_ZONES.items():
                distance = _haversine_nm(lat, lon, zone['lat'], zone['lon'])
                
                if distance < min_distance:
                    min_distance = distance
                    closest_zone = zone
                
                # Check if approaching critical zone
                if zone['severity'] == 'critical' and distance < zone['radius_nm'] * 1.5:
                    critical_approach = True
        
        if critical_approach:
            return 30, f"Approaching critical zone: {closest_zone['name']}"
        elif min_distance < 10:
            return 25, f"Very close to sensitive zone: {closest_zone['name']} ({min_distance:.1f}nm)"
        elif min_distance < 25:
            return 15, f"Near sensitive zone: {closest_zone['name']} ({min_distance:.1f}nm)"
        elif min_distance < 50:
            return 5, f"Moderate distance from zones ({min_distance:.1f}nm)"
        else:
            return 0, "Far from sensitive zones"
    
    def _analyze_heading_changes(self, track_data: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Analyze heading changes for suspicious patterns."""
        if len(track_data) < 3:
            return 0, "Insufficient data"
        
        sudden_turns = 0
        turns_toward_zones = 0
        
        for i in range(1, len(track_data)):
            prev = track_data[i-1]
            curr = track_data[i]
            
            prev_hdg = prev.get('track') or prev.get('heading')
            curr_hdg = curr.get('track') or curr.get('heading')
            
            if prev_hdg is None or curr_hdg is None:
                continue
            
            # Calculate heading change
            delta = ((curr_hdg - prev_hdg + 540) % 360) - 180
            
            # Count sudden turns (> 30 degrees in short time)
            dt = (curr.get('timestamp', 0) - prev.get('timestamp', 0))
            if dt > 0 and dt < 60 and abs(delta) > 30:
                sudden_turns += 1
                
                # Check if turn is toward a sensitive zone
                curr_lat = curr.get('lat', 0)
                curr_lon = curr.get('lon', 0)
                
                for zone_id, zone in RESTRICTED_ZONES.items():
                    if zone['severity'] in ['critical', 'high']:
                        # Calculate bearing to zone
                        bearing = self._calculate_bearing(curr_lat, curr_lon, zone['lat'], zone['lon'])
                        bearing_diff = abs(((curr_hdg - bearing + 540) % 360) - 180)
                        
                        if bearing_diff < 30:  # Heading toward zone
                            turns_toward_zones += 1
                            break
        
        if turns_toward_zones >= 2:
            return 25, f"Multiple heading changes toward sensitive zones ({turns_toward_zones})"
        elif turns_toward_zones == 1:
            return 15, "Single heading change toward sensitive zone"
        elif sudden_turns >= 3:
            return 10, f"Multiple sudden heading changes ({sudden_turns})"
        elif sudden_turns >= 1:
            return 5, "Some heading variability"
        else:
            return 0, "Normal heading profile"
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360
    
    def _analyze_altitude_profile(self, track_data: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Analyze altitude profile for suspicious patterns."""
        if len(track_data) < 3:
            return 0, "Insufficient data"
        
        alts = [p.get('alt', 0) for p in track_data if p.get('alt')]
        if not alts:
            return 0, "No altitude data"
        
        # Check for sudden drops
        sudden_drops = 0
        for i in range(1, len(alts)):
            drop = alts[i-1] - alts[i]
            if drop > 2000:  # More than 2000 ft drop
                sudden_drops += 1
        
        # Check for low-level approach
        recent_alts = alts[-5:] if len(alts) >= 5 else alts
        min_recent_alt = min(recent_alts)
        
        # Check if descending without being near an airport
        last_point = track_data[-1]
        near_airport = False
        for zone_id, zone in RESTRICTED_ZONES.items():
            if zone['type'] == 'airport_tma':
                dist = _haversine_nm(last_point.get('lat', 0), last_point.get('lon', 0), 
                                    zone['lat'], zone['lon'])
                if dist < zone['radius_nm']:
                    near_airport = True
                    break
        
        if min_recent_alt < 3000 and not near_airport:
            return 20, f"Low altitude ({min_recent_alt}ft) away from airports"
        elif sudden_drops >= 2:
            return 15, f"Multiple sudden altitude drops ({sudden_drops})"
        elif sudden_drops == 1:
            return 8, "Single significant altitude drop"
        elif min_recent_alt < 5000 and not near_airport:
            return 5, "Moderately low altitude away from airports"
        else:
            return 0, "Normal altitude profile"
    
    def _analyze_speed_anomalies(self, track_data: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Analyze speed for suspicious patterns."""
        speeds = [p.get('gspeed') or p.get('speed', 0) for p in track_data if p.get('gspeed') or p.get('speed')]
        if not speeds:
            return 0, "No speed data"
        
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
        min_speed = min(speeds)
        
        # Check for sudden speed changes
        speed_changes = 0
        for i in range(1, len(speeds)):
            change = abs(speeds[i] - speeds[i-1])
            if change > 50:  # More than 50 kts change
                speed_changes += 1
        
        # Suspicious patterns
        if max_speed > 500 and min_speed < 150:
            return 15, f"Extreme speed variation ({min_speed}-{max_speed} kts)"
        elif speed_changes >= 3:
            return 10, f"Multiple sudden speed changes ({speed_changes})"
        elif avg_speed < 100 and len(track_data) > 20:
            return 8, f"Sustained low speed ({avg_speed:.0f} kts) - possible loitering"
        elif speed_changes >= 1:
            return 3, "Some speed variability"
        else:
            return 0, "Normal speed profile"
    
    def _analyze_operator_history(self, flight_id: str, track_data: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Analyze operator/callsign history."""
        # Try to get callsign from track data
        callsign = None
        for point in track_data:
            if point.get('callsign'):
                callsign = point['callsign']
                break
        
        if not callsign:
            return 0, "Unknown operator"
        
        # Check for military callsign patterns
        callsign_upper = callsign.upper()
        
        # Known military patterns that may require attention
        military_prefixes = ['RCH', 'CNV', 'QUID', 'RRR', 'IAF']
        for prefix in military_prefixes:
            if callsign_upper.startswith(prefix):
                return 5, f"Military callsign pattern ({prefix})"
        
        # Check against historical anomaly reports
        query = """
            SELECT COUNT(*) FROM anomaly_reports
            WHERE json_extract(full_report, '$.summary.callsign') LIKE ?
        """
        results = self._execute_query('research', query, (f'{callsign[:3]}%',))
        
        if results and results[0][0] > 5:
            return 10, f"Operator has multiple historical anomalies ({results[0][0]})"
        elif results and results[0][0] > 2:
            return 5, f"Operator has some historical anomalies ({results[0][0]})"
        
        return 0, "No concerning operator history"

