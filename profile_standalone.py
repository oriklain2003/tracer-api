"""
Standalone profiler that directly measures the slow routes without circular imports.

This script replicates the route logic with detailed timing measurements.
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

load_dotenv()

# Add parent to path
root_path = str(Path(__file__).resolve().parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def profile_unified_track(flight_id: str):
    """Profile the /api/track/unified/{flight_id} route."""
    from service.pg_provider import get_pool
    
    print("\n" + "="*80)
    print(f"PROFILING: /api/track/unified/{flight_id}")
    print("="*80 + "\n")
    
    timings = {}
    total_start = time.time()
    pool = get_pool()
    
    # Step 1: Try PostgreSQL feedback schema
    step_start = time.time()
    result = None
    try:
        with pool.get_connection() as conn:
            with conn.cursor() as cursor:
                # This matches the query in pg_provider.get_unified_track
                cursor.execute("""
                    SELECT 
                        ft.flight_id,
                        ft.timestamp,
                        ft.lat,
                        ft.lon,
                        ft.alt,
                        ft.gspeed,
                        ft.track,
                        ft.vspeed,
                        ft.squawk,
                        ft.emergency,
                        ft.is_on_ground
                    FROM feedback.flight_tracks ft
                    WHERE ft.flight_id = %s
                    ORDER BY ft.timestamp ASC
                """, (flight_id,))
                
                rows = cursor.fetchall()
                timings['postgresql_feedback_query'] = time.time() - step_start
                
                if rows:
                    points = []
                    for row in rows:
                        points.append({
                            'flight_id': row[0],
                            'timestamp': row[1],
                            'lat': row[2],
                            'lon': row[3],
                            'alt': row[4],
                            'gspeed': row[5],
                            'track': row[6],
                            'vspeed': row[7],
                            'squawk': row[8],
                            'emergency': row[9],
                            'is_on_ground': row[10]
                        })
                    result = {'points': points}
                    print(f"✓ PostgreSQL feedback: Found {len(points)} points - {timings['postgresql_feedback_query']:.3f}s")
                else:
                    print(f"  PostgreSQL feedback: No data - {timings['postgresql_feedback_query']:.3f}s")
    except Exception as e:
        timings['postgresql_feedback_query'] = time.time() - step_start
        print(f"✗ PostgreSQL feedback error: {e} - {timings['postgresql_feedback_query']:.3f}s")
    
    # Step 2: Try research.normal_tracks if not found
    if not result:
        step_start = time.time()
        try:
            with pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            flight_id, timestamp, lat, lon, alt, 
                            gspeed, track, vspeed, squawk
                        FROM research.normal_tracks
                        WHERE flight_id = %s
                        ORDER BY timestamp ASC
                    """, (flight_id,))
                    
                    rows = cursor.fetchall()
                    timings['postgresql_research_normal'] = time.time() - step_start
                    
                    if rows:
                        points = []
                        for row in rows:
                            points.append({
                                'flight_id': row[0],
                                'timestamp': row[1],
                                'lat': row[2],
                                'lon': row[3],
                                'alt': row[4],
                                'gspeed': row[5],
                                'track': row[6],
                                'vspeed': row[7],
                                'squawk': row[8]
                            })
                        result = {'points': points}
                        print(f"✓ PostgreSQL research.normal: Found {len(points)} points - {timings['postgresql_research_normal']:.3f}s")
                    else:
                        print(f"  PostgreSQL research.normal: No data - {timings['postgresql_research_normal']:.3f}s")
        except Exception as e:
            timings['postgresql_research_normal'] = time.time() - step_start
            print(f"✗ PostgreSQL research.normal error: {e} - {timings['postgresql_research_normal']:.3f}s")
    
    # Step 3: Try research.anomalies_tracks if still not found
    if not result:
        step_start = time.time()
        try:
            with pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            flight_id, timestamp, lat, lon, alt, 
                            gspeed, track, vspeed, squawk
                        FROM research.anomalies_tracks
                        WHERE flight_id = %s
                        ORDER BY timestamp ASC
                    """, (flight_id,))
                    
                    rows = cursor.fetchall()
                    timings['postgresql_research_anomaly'] = time.time() - step_start
                    
                    if rows:
                        points = []
                        for row in rows:
                            points.append({
                                'flight_id': row[0],
                                'timestamp': row[1],
                                'lat': row[2],
                                'lon': row[3],
                                'alt': row[4],
                                'gspeed': row[5],
                                'track': row[6],
                                'vspeed': row[7],
                                'squawk': row[8]
                            })
                        result = {'points': points}
                        print(f"✓ PostgreSQL research.anomaly: Found {len(points)} points - {timings['postgresql_research_anomaly']:.3f}s")
                    else:
                        print(f"  PostgreSQL research.anomaly: No data - {timings['postgresql_research_anomaly']:.3f}s")
        except Exception as e:
            timings['postgresql_research_anomaly'] = time.time() - step_start
            print(f"✗ PostgreSQL research.anomaly error: {e} - {timings['postgresql_research_anomaly']:.3f}s")
    
    total_time = time.time() - total_start
    
    if not result:
        print(f"\n✗ No data found in any source")
    
    print(f"\nTotal time: {total_time:.3f}s")
    print_timing_breakdown(timings, total_time)


def profile_feedback_history(limit: int = 100):
    """Profile the /api/feedback/history route."""
    from service.pg_provider import get_pool
    
    print("\n" + "="*80)
    print(f"PROFILING: /api/feedback/history (limit={limit})")
    print("="*80 + "\n")
    
    timings = {}
    total_start = time.time()
    pool = get_pool()
    
    # Get current timestamp
    end_ts = int(datetime.now().timestamp())
    start_ts = 0
    
    # Step 1: Database connection
    step_start = time.time()
    try:
        with pool.get_connection() as conn:
            timings['db_connect'] = time.time() - step_start
            print(f"✓ Database connection: {timings['db_connect']:.3f}s")
            
            # Step 2: Main query execution
            step_start = time.time()
            with conn.cursor() as cursor:
                cursor.execute("""
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
                        fm.emergency_squawk_detected
                    FROM feedback.user_feedback uf
                    LEFT JOIN feedback.flight_metadata fm ON uf.flight_id = fm.flight_id
                    WHERE COALESCE(uf.first_seen_ts, uf.tagged_at) BETWEEN %s AND %s
                    ORDER BY COALESCE(uf.first_seen_ts, uf.tagged_at) DESC
                    LIMIT %s
                """, (start_ts, end_ts, limit))
                
                timings['main_query'] = time.time() - step_start
                print(f"✓ Main query execution: {timings['main_query']:.3f}s")
                
                # Step 3: Fetch all rows
                step_start = time.time()
                rows = cursor.fetchall()
                timings['fetchall'] = time.time() - step_start
                print(f"✓ Fetch all rows: {len(rows)} rows in {timings['fetchall']:.3f}s")
                
                # Step 4: Process results
                step_start = time.time()
                result = []
                origin_dest_calculations = 0
                
                for row in rows:
                    flight_data = {
                        'flight_id': row[0],
                        'first_seen_ts': row[1],
                        'last_seen_ts': row[2],
                        'tagged_at': row[3],
                        'timestamp': row[4],
                        'rule_id': row[5],
                        'rule_name': row[6],
                        'comments': row[7],
                        'other_details': row[8],
                        'user_label': row[9],
                        'callsign': row[10],
                        'flight_number': row[11],
                        'airline': row[12],
                        'origin_airport': row[13],
                        'destination_airport': row[14],
                        'aircraft_type': row[15],
                        'aircraft_registration': row[16],
                        'category': row[17],
                        'is_military': row[18],
                        'total_points': row[19],
                        'flight_duration_sec': row[20],
                        'max_altitude_ft': row[21],
                        'avg_altitude_ft': row[22],
                        'min_altitude_ft': row[23],
                        'avg_speed_kts': row[24],
                        'max_speed_kts': row[25],
                        'min_speed_kts': row[26],
                        'total_distance_nm': row[27],
                        'scheduled_departure': row[28],
                        'scheduled_arrival': row[29],
                        'squawk_codes': row[30],
                        'emergency_squawk_detected': row[31]
                    }
                    
                    # Check if origin/dest calculation would be needed
                    if not flight_data['origin_airport'] or not flight_data['destination_airport']:
                        origin_dest_calculations += 1
                    
                    result.append(flight_data)
                
                timings['process_results'] = time.time() - step_start
                print(f"✓ Process results: {timings['process_results']:.3f}s")
                
                if origin_dest_calculations > 0:
                    print(f"  ⚠️  {origin_dest_calculations} flights missing origin/dest")
                    print(f"     (would trigger additional queries in actual route)")
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
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
            print(f"  {key:35s}: {value:7.3f}s ({percentage:5.1f}%)")
    
    print("-"*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone profiler for slow routes')
    parser.add_argument('--route', required=True, choices=['track', 'feedback-history'],
                       help='Route to profile')
    parser.add_argument('--flight-id', help='Flight ID for track route')
    parser.add_argument('--limit', type=int, default=100, help='Limit for feedback-history')
    
    args = parser.parse_args()
    
    if args.route == 'track':
        if not args.flight_id:
            parser.error('--flight-id is required for track route')
        profile_unified_track(args.flight_id)
    
    elif args.route == 'feedback-history':
        profile_feedback_history(args.limit)


if __name__ == '__main__':
    main()
