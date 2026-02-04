"""
Detailed line-by-line profiler for API routes.

This provides more granular information about which specific lines are slow.

Usage:
    # Install line_profiler first: pip install line-profiler
    python profile_detailed.py --route track --flight-id <flight_id>
    python profile_detailed.py --route feedback-history
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
root_path = str(Path(__file__).resolve().parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Add api subdirectory to path for middleware imports
api_path = str(Path(__file__).resolve().parent / "api")
if api_path not in sys.path:
    sys.path.insert(0, api_path)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR

CACHE_DB_PATH = PROJECT_ROOT / "flight_cache.db"
DB_ANOMALIES_PATH = PROJECT_ROOT / "realtime/live_anomalies.db"
DB_TRACKS_PATH = PROJECT_ROOT / "realtime/live_tracks.db"
DB_RESEARCH_PATH = PROJECT_ROOT / "realtime/research_new.db"
DB_LIVE_RESEARCH_PATH = PROJECT_ROOT / "realtime/live_research.db"
PRESENT_DB_PATH = BASE_DIR / "api/present_anomalies.db"
FEEDBACK_TAGGED_DB_PATH = BASE_DIR / "api/feedback_tagged.db"
FEEDBACK_DB_PATH = PROJECT_ROOT / "training_ops/feedback.db"


def analyze_unified_track_performance(flight_id: str):
    """Manually analyze the unified track function with timing."""
    import sqlite3
    import json
    from routes import flights
    from flight_fetcher import serialize_flight, deserialize_flight, search_flight_path, get as get_flight
    from anomaly_pipeline import AnomalyPipeline
    
    # Configure
    def get_pipeline_dummy():
        return AnomalyPipeline()
    
    flights.configure(
        cache_db_path=CACHE_DB_PATH,
        db_anomalies_path=DB_ANOMALIES_PATH,
        db_tracks_path=DB_TRACKS_PATH,
        db_research_path=DB_RESEARCH_PATH,
        feedback_tagged_db_path=FEEDBACK_TAGGED_DB_PATH,
        get_pipeline_func=get_pipeline_dummy,
        serialize_flight_func=serialize_flight,
        deserialize_flight_func=deserialize_flight,
        search_flight_path_func=search_flight_path,
        get_flight_func=get_flight,
        db_live_research_path=DB_LIVE_RESEARCH_PATH,
    )
    
    print("\n" + "="*80)
    print(f"DETAILED PERFORMANCE ANALYSIS: /api/track/unified/{flight_id}")
    print("="*80 + "\n")
    
    timings = {}
    total_start = time.time()
    
    # Step 1: Try PostgreSQL
    step_start = time.time()
    result = None
    try:
        from service.pg_provider import get_unified_track as pg_get_unified_track
        result = pg_get_unified_track(flight_id)
        timings['postgresql_query'] = time.time() - step_start
        if result:
            print(f"✓ PostgreSQL: Found data ({len(result.get('points', []))} points) - {timings['postgresql_query']:.3f}s")
            total_time = time.time() - total_start
            print(f"\nTotal time: {total_time:.3f}s")
            print_timing_breakdown(timings, total_time)
            return
        else:
            print(f"  PostgreSQL: No data - {timings['postgresql_query']:.3f}s")
    except ImportError as e:
        timings['postgresql_query'] = time.time() - step_start
        print(f"  PostgreSQL: Not available - {timings['postgresql_query']:.3f}s")
    except Exception as e:
        timings['postgresql_query'] = time.time() - step_start
        print(f"✗ PostgreSQL: Error ({e}) - {timings['postgresql_query']:.3f}s")
    
    # Step 2: Try Live DB
    step_start = time.time()
    points = []
    if DB_TRACKS_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_TRACKS_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            conn.close()
            
            timings['live_db_query'] = time.time() - step_start
            
            if rows:
                points = [dict(r) for r in rows]
                print(f"✓ Live DB: Found {len(points)} points - {timings['live_db_query']:.3f}s")
                total_time = time.time() - total_start
                print(f"\nTotal time: {total_time:.3f}s")
                print_timing_breakdown(timings, total_time)
                return
            else:
                print(f"  Live DB: No data - {timings['live_db_query']:.3f}s")
        except Exception as e:
            timings['live_db_query'] = time.time() - step_start
            print(f"✗ Live DB: Error ({e}) - {timings['live_db_query']:.3f}s")
    else:
        timings['live_db_query'] = 0
        print(f"  Live DB: File not found")
    
    # Step 3: Try Research DB
    step_start = time.time()
    if not points and DB_RESEARCH_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_RESEARCH_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try anomalies_tracks
            query_start = time.time()
            cursor.execute("SELECT * FROM anomalies_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            anomalies_time = time.time() - query_start
            
            if not rows:
                # Try normal_tracks
                query_start = time.time()
                cursor.execute("SELECT * FROM normal_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
                rows = cursor.fetchall()
                normal_time = time.time() - query_start
            else:
                normal_time = 0
            
            conn.close()
            
            timings['research_db_anomalies'] = anomalies_time
            timings['research_db_normal'] = normal_time
            timings['research_db_query'] = time.time() - step_start
            
            if rows:
                points = [dict(r) for r in rows]
                print(f"✓ Research DB: Found {len(points)} points - {timings['research_db_query']:.3f}s")
                print(f"    - anomalies_tracks: {anomalies_time:.3f}s")
                print(f"    - normal_tracks: {normal_time:.3f}s")
                total_time = time.time() - total_start
                print(f"\nTotal time: {total_time:.3f}s")
                print_timing_breakdown(timings, total_time)
                return
            else:
                print(f"  Research DB: No data - {timings['research_db_query']:.3f}s")
        except Exception as e:
            timings['research_db_query'] = time.time() - step_start
            print(f"✗ Research DB: Error ({e}) - {timings['research_db_query']:.3f}s")
    else:
        timings['research_db_query'] = 0
        print(f"  Research DB: Skipped")
    
    # Step 4: Try Feedback Tagged DB
    step_start = time.time()
    if not points and FEEDBACK_TAGGED_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flight_tracks WHERE flight_id = ? ORDER BY timestamp ASC", (flight_id,))
            rows = cursor.fetchall()
            conn.close()
            
            timings['feedback_tagged_db_query'] = time.time() - step_start
            
            if rows:
                points = [dict(r) for r in rows]
                print(f"✓ Feedback Tagged DB: Found {len(points)} points - {timings['feedback_tagged_db_query']:.3f}s")
                total_time = time.time() - total_start
                print(f"\nTotal time: {total_time:.3f}s")
                print_timing_breakdown(timings, total_time)
                return
            else:
                print(f"  Feedback Tagged DB: No data - {timings['feedback_tagged_db_query']:.3f}s")
        except Exception as e:
            timings['feedback_tagged_db_query'] = time.time() - step_start
            print(f"✗ Feedback Tagged DB: Error ({e}) - {timings['feedback_tagged_db_query']:.3f}s")
    else:
        timings['feedback_tagged_db_query'] = 0
        print(f"  Feedback Tagged DB: Skipped")
    
    # Step 5: Try Cache DB
    step_start = time.time()
    if not points:
        try:
            conn = sqlite3.connect(str(CACHE_DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
            row = cursor.fetchone()
            conn.close()
            
            timings['cache_db_query'] = time.time() - step_start
            
            if row:
                data = json.loads(row[0])
                points = data.get("points", [])
                print(f"✓ Cache DB: Found {len(points)} points - {timings['cache_db_query']:.3f}s")
                total_time = time.time() - total_start
                print(f"\nTotal time: {total_time:.3f}s")
                print_timing_breakdown(timings, total_time)
                return
            else:
                print(f"  Cache DB: No data - {timings['cache_db_query']:.3f}s")
        except Exception as e:
            timings['cache_db_query'] = time.time() - step_start
            print(f"✗ Cache DB: Error ({e}) - {timings['cache_db_query']:.3f}s")
    else:
        timings['cache_db_query'] = 0
        print(f"  Cache DB: Skipped")
    
    # If we get here, no data was found
    print(f"\n✗ No data found in any source")
    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.3f}s")
    print_timing_breakdown(timings, total_time)


def analyze_feedback_history_performance(start_ts: int = 0, end_ts: int = None, 
                                        limit: int = 100, include_normal: bool = True):
    """Manually analyze the feedback history function with timing."""
    import sqlite3
    import json
    from routes import flights, feedback
    from flight_fetcher import serialize_flight, deserialize_flight, search_flight_path, get as get_flight
    from anomaly_pipeline import AnomalyPipeline
    from training_ops.db_utils import save_feedback
    
    if end_ts is None:
        end_ts = int(datetime.now().timestamp())
    
    # Configure
    def get_pipeline_dummy():
        return AnomalyPipeline()
    
    flights.configure(
        cache_db_path=CACHE_DB_PATH,
        db_anomalies_path=DB_ANOMALIES_PATH,
        db_tracks_path=DB_TRACKS_PATH,
        db_research_path=DB_RESEARCH_PATH,
        feedback_tagged_db_path=FEEDBACK_TAGGED_DB_PATH,
        get_pipeline_func=get_pipeline_dummy,
        serialize_flight_func=serialize_flight,
        deserialize_flight_func=deserialize_flight,
        search_flight_path_func=search_flight_path,
        get_flight_func=get_flight,
        db_live_research_path=DB_LIVE_RESEARCH_PATH,
    )
    
    def fetch_flight_details_dummy(flight_id, flight_time=None, callsign=None):
        return {"flight_id": flight_id}
    
    def update_flight_record_dummy(flight_id, **kwargs):
        pass
    
    feedback.configure(
        cache_db_path=CACHE_DB_PATH,
        db_anomalies_path=DB_ANOMALIES_PATH,
        db_tracks_path=DB_TRACKS_PATH,
        db_research_path=DB_RESEARCH_PATH,
        present_db_path=PRESENT_DB_PATH,
        feedback_db_path=FEEDBACK_DB_PATH,
        feedback_tagged_db_path=FEEDBACK_TAGGED_DB_PATH,
        project_root=PROJECT_ROOT,
        get_pipeline_func=get_pipeline_dummy,
        get_unified_track_func=flights.get_unified_track,
        fetch_flight_details_func=fetch_flight_details_dummy,
        update_flight_record_func=update_flight_record_dummy,
        save_feedback_func=save_feedback,
    )
    
    print("\n" + "="*80)
    print(f"DETAILED PERFORMANCE ANALYSIS: /api/feedback/history")
    print("="*80)
    print(f"Parameters: start_ts={start_ts}, end_ts={end_ts}, limit={limit}, include_normal={include_normal}\n")
    
    if not FEEDBACK_TAGGED_DB_PATH.exists():
        print("✗ feedback_tagged.db does not exist!")
        return
    
    timings = {}
    total_start = time.time()
    
    # Step 1: Open database connection
    step_start = time.time()
    try:
        conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        timings['db_connect'] = time.time() - step_start
        print(f"✓ Database connection: {timings['db_connect']:.3f}s")
    except Exception as e:
        timings['db_connect'] = time.time() - step_start
        print(f"✗ Database connection failed: {e}")
        return
    
    # Step 2: Execute main query
    step_start = time.time()
    try:
        label_filter = "" if include_normal else "AND uf.user_label = 1"
        
        cursor.execute(f"""
            SELECT 
                uf.flight_id,
                uf.first_seen_ts,
                uf.last_seen_ts,
                uf.tagged_at,
                COALESCE(uf.first_seen_ts, uf.tagged_at) as timestamp,
                uf.rule_id,
                uf.rule_name,
                uf.comments,
                uf.other_details,
                uf.user_label,
                fm.callsign,
                fm.flight_number,
                fm.airline,
                fm.origin_airport,
                fm.destination_airport,
                fm.aircraft_type,
                fm.aircraft_registration,
                fm.category,
                fm.is_military,
                fm.total_points,
                fm.flight_duration_sec,
                fm.max_altitude_ft,
                fm.avg_altitude_ft,
                fm.min_altitude_ft,
                fm.avg_speed_kts,
                fm.max_speed_kts,
                fm.min_speed_kts,
                fm.total_distance_nm,
                fm.scheduled_departure,
                fm.scheduled_arrival,
                fm.squawk_codes,
                fm.emergency_squawk_detected,
                ar.full_report,
                ar.severity_cnn,
                ar.severity_dense,
                ar.matched_rule_ids,
                ar.matched_rule_names,
                ar.matched_rule_categories,
                (SELECT COUNT(*) FROM flight_tracks ft WHERE ft.flight_id = uf.flight_id) as track_count
            FROM user_feedback uf
            LEFT JOIN flight_metadata fm ON uf.flight_id = fm.flight_id
            LEFT JOIN anomaly_reports ar ON uf.flight_id = ar.flight_id
            WHERE COALESCE(uf.first_seen_ts, uf.tagged_at) BETWEEN ? AND ?
              {label_filter}
            ORDER BY COALESCE(uf.first_seen_ts, uf.tagged_at) DESC
            LIMIT ?
        """, (start_ts, end_ts, limit))
        
        timings['main_query'] = time.time() - step_start
        print(f"✓ Main query execution: {timings['main_query']:.3f}s")
    except Exception as e:
        timings['main_query'] = time.time() - step_start
        print(f"✗ Main query failed: {e}")
        conn.close()
        return
    
    # Step 3: Fetch all rows
    step_start = time.time()
    try:
        rows = cursor.fetchall()
        timings['fetchall'] = time.time() - step_start
        print(f"✓ Fetch all rows: {len(rows)} rows in {timings['fetchall']:.3f}s")
    except Exception as e:
        timings['fetchall'] = time.time() - step_start
        print(f"✗ Fetchall failed: {e}")
        conn.close()
        return
    
    conn.close()
    
    # Step 4: Process results
    step_start = time.time()
    origin_dest_times = []
    result = []
    
    for row in rows:
        row_start = time.time()
        
        # Parse full_report
        full_report = row['full_report']
        if isinstance(full_report, (str, bytes)):
            try:
                full_report = json.loads(full_report)
            except:
                pass
        
        # Get origin and destination
        origin_airport = row['origin_airport']
        destination_airport = row['destination_airport']
        
        # Calculate if missing (this might be slow!)
        if not origin_airport or not destination_airport:
            calc_start = time.time()
            calculated_origin, calculated_dest = feedback._calculate_origin_destination_from_track(row['flight_id'])
            calc_time = time.time() - calc_start
            origin_dest_times.append(calc_time)
            
            if not origin_airport and calculated_origin:
                origin_airport = calculated_origin
            if not destination_airport and calculated_dest:
                destination_airport = calculated_dest
        
        # Build result dict
        result.append({
            'flight_id': row['flight_id'],
            'timestamp': row['timestamp'],
            'first_seen_ts': row['first_seen_ts'],
            'last_seen_ts': row['last_seen_ts'],
            'tagged_at': row['tagged_at'],
            'is_anomaly': bool(row['user_label']),
            'user_label': row['user_label'],
            'rule_id': row['rule_id'],
            'rule_name': row['rule_name'],
            'comments': row['comments'],
            'other_details': row['other_details'],
            'callsign': row['callsign'],
            'flight_number': row['flight_number'],
            'airline': row['airline'],
            'origin_airport': origin_airport,
            'destination_airport': destination_airport,
            'aircraft_type': row['aircraft_type'],
            'aircraft_registration': row['aircraft_registration'],
            'category': row['category'],
            'is_military': row['is_military'],
            'total_points': row['total_points'] or row['track_count'],
            'flight_duration_sec': row['flight_duration_sec'],
            'max_altitude_ft': row['max_altitude_ft'],
            'avg_altitude_ft': row['avg_altitude_ft'],
            'min_altitude_ft': row['min_altitude_ft'],
            'avg_speed_kts': row['avg_speed_kts'],
            'max_speed_kts': row['max_speed_kts'],
            'min_speed_kts': row['min_speed_kts'],
            'total_distance_nm': row['total_distance_nm'],
            'scheduled_departure': row['scheduled_departure'],
            'scheduled_arrival': row['scheduled_arrival'],
            'squawk_codes': row['squawk_codes'],
            'emergency_squawk_detected': row['emergency_squawk_detected'],
            'full_report': full_report,
            'severity_cnn': row['severity_cnn'],
            'severity_dense': row['severity_dense'],
            'matched_rule_ids': row['matched_rule_ids'],
            'matched_rule_names': row['matched_rule_names'],
            'matched_rule_categories': row['matched_rule_categories'],
            'track_count': row['track_count'],
            'source': 'tagged'
        })
    
    timings['process_results'] = time.time() - step_start
    print(f"✓ Process results: {timings['process_results']:.3f}s")
    
    if origin_dest_times:
        timings['origin_dest_calculation'] = sum(origin_dest_times)
        timings['origin_dest_count'] = len(origin_dest_times)
        timings['origin_dest_avg'] = timings['origin_dest_calculation'] / len(origin_dest_times)
        print(f"  - Origin/destination calculation: {timings['origin_dest_calculation']:.3f}s total")
        print(f"    ({timings['origin_dest_count']} calculations, avg {timings['origin_dest_avg']:.3f}s each)")
        print(f"    ⚠️  This is a major bottleneck!")
    
    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.3f}s")
    print(f"Results: {len(result)} flights")
    print_timing_breakdown(timings, total_time)


def print_timing_breakdown(timings: dict, total_time: float):
    """Print timing breakdown as percentages."""
    print("\n" + "-"*80)
    print("TIME BREAKDOWN:")
    print("-"*80)
    
    for key, value in sorted(timings.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
        if isinstance(value, (int, float)) and value > 0:
            percentage = (value / total_time) * 100
            print(f"  {key:30s}: {value:7.3f}s ({percentage:5.1f}%)")
    
    print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='Detailed line-by-line profiler for API routes')
    parser.add_argument('--route', required=True, choices=['track', 'feedback-history'],
                       help='Route to profile')
    
    # Track route arguments
    parser.add_argument('--flight-id', help='Flight ID for track route')
    
    # Feedback history arguments
    parser.add_argument('--start-ts', type=int, default=0, help='Start timestamp')
    parser.add_argument('--end-ts', type=int, help='End timestamp')
    parser.add_argument('--limit', type=int, default=100, help='Result limit')
    parser.add_argument('--exclude-normal', action='store_true', 
                       help='Exclude normal (non-anomaly) flights')
    
    args = parser.parse_args()
    
    if args.route == 'track':
        if not args.flight_id:
            parser.error('--flight-id is required for track route')
        analyze_unified_track_performance(args.flight_id)
    
    elif args.route == 'feedback-history':
        analyze_feedback_history_performance(
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            limit=args.limit,
            include_normal=not args.exclude_normal
        )


if __name__ == '__main__':
    main()
