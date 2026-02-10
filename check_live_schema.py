"""
Script to check the live schema and diagnose why no flights are being returned.
"""
import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

dsn = os.environ.get("POSTGRES_DSN")
if not dsn:
    print("ERROR: POSTGRES_DSN not set")
    exit(1)

conn = psycopg2.connect(dsn)
cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

print("=" * 80)
print("CHECKING LIVE SCHEMA")
print("=" * 80)

# 1. Check if live schema exists
print("\n1. Checking if 'live' schema exists...")
cursor.execute("""
    SELECT schema_name 
    FROM information_schema.schemata 
    WHERE schema_name = 'live'
""")
schema = cursor.fetchone()
if schema:
    print(f"   [OK] Schema 'live' exists")
else:
    print(f"   [ERROR] Schema 'live' does NOT exist!")
    print("\n   Available schemas:")
    cursor.execute("SELECT schema_name FROM information_schema.schemata ORDER BY schema_name")
    for row in cursor.fetchall():
        print(f"     - {row['schema_name']}")
    conn.close()
    exit(1)

# 2. Check what tables exist in live schema
print("\n2. Checking tables in 'live' schema...")
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'live'
    ORDER BY table_name
""")
tables = cursor.fetchall()
if tables:
    print(f"   Found {len(tables)} tables:")
    for row in tables:
        print(f"     - {row['table_name']}")
else:
    print("   [ERROR] No tables found in 'live' schema!")
    conn.close()
    exit(1)

# 3. Check if flight_metadata table exists
print("\n3. Checking 'live.flight_metadata' table...")
if any(t['table_name'] == 'flight_metadata' for t in tables):
    print("   [OK] Table 'live.flight_metadata' exists")
    
    # Check row count
    cursor.execute("SELECT COUNT(*) as count FROM live.flight_metadata")
    count = cursor.fetchone()['count']
    print(f"   Total rows in flight_metadata: {count}")
    
    if count > 0:
        # Check timestamp distribution
        cursor.execute("""
            SELECT 
                MIN(first_seen_ts) as min_first_seen,
                MAX(first_seen_ts) as max_first_seen,
                MIN(last_seen_ts) as min_last_seen,
                MAX(last_seen_ts) as max_last_seen
            FROM live.flight_metadata
        """)
        ts_info = cursor.fetchone()
        
        current_ts = int(time.time())
        cutoff_ts = current_ts - (15 * 60)  # 15 minutes ago
        
        print(f"\n   Timestamp analysis:")
        print(f"   Current time: {current_ts} ({datetime.fromtimestamp(current_ts)})")
        print(f"   Cutoff time (15 min ago): {cutoff_ts} ({datetime.fromtimestamp(cutoff_ts)})")
        print(f"   ")
        if ts_info['min_first_seen']:
            print(f"   Min first_seen_ts: {ts_info['min_first_seen']} ({datetime.fromtimestamp(ts_info['min_first_seen'])})")
        if ts_info['max_first_seen']:
            print(f"   Max first_seen_ts: {ts_info['max_first_seen']} ({datetime.fromtimestamp(ts_info['max_first_seen'])})")
        if ts_info['min_last_seen']:
            print(f"   Min last_seen_ts: {ts_info['min_last_seen']} ({datetime.fromtimestamp(ts_info['min_last_seen'])})")
        if ts_info['max_last_seen']:
            print(f"   Max last_seen_ts: {ts_info['max_last_seen']} ({datetime.fromtimestamp(ts_info['max_last_seen'])})")
        
        # Check recent flights (within cutoff)
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM live.flight_metadata 
            WHERE last_seen_ts > %s
        """, (cutoff_ts,))
        recent_count = cursor.fetchone()['count']
        print(f"\n   Flights with last_seen_ts > cutoff (within 15 min): {recent_count}")
        
        if recent_count == 0:
            print("\n   [ERROR] NO FLIGHTS found within the last 15 minutes!")
            print("   This is why /api/live/flights returns empty.")
            print("\n   Possible causes:")
            print("   - Monitor is not running")
            print("   - Monitor is not writing to database")
            print("   - Data is too old (last update was more than 15 minutes ago)")
            
            # Show most recent flights
            cursor.execute("""
                SELECT flight_id, callsign, last_seen_ts
                FROM live.flight_metadata
                ORDER BY last_seen_ts DESC
                LIMIT 10
            """)
            recent_flights = cursor.fetchall()
            if recent_flights:
                print("\n   Most recent 10 flights:")
                for f in recent_flights:
                    ts = f['last_seen_ts']
                    dt = datetime.fromtimestamp(ts) if ts else "None"
                    age_minutes = (current_ts - ts) / 60 if ts else 0
                    print(f"     - {f['flight_id']} ({f['callsign']}): {ts} ({dt}) - {age_minutes:.1f} min ago")
        else:
            print(f"\n   [OK] Found {recent_count} recent flights (should be returned by API)")
            cursor.execute("""
                SELECT flight_id, callsign, last_seen_ts, is_anomaly
                FROM live.flight_metadata
                WHERE last_seen_ts > %s
                ORDER BY last_seen_ts DESC
                LIMIT 5
            """, (cutoff_ts,))
            recent_flights = cursor.fetchall()
            print("\n   Sample of recent flights:")
            for f in recent_flights:
                ts = f['last_seen_ts']
                dt = datetime.fromtimestamp(ts) if ts else "None"
                age_minutes = (current_ts - ts) / 60 if ts else 0
                anomaly_str = " [ANOMALY]" if f.get('is_anomaly') else ""
                print(f"     - {f['flight_id']} ({f['callsign']}): {age_minutes:.1f} min ago{anomaly_str}")
else:
    print("   [ERROR] Table 'live.flight_metadata' does NOT exist!")

# 4. Check normal_tracks table
print("\n4. Checking 'live.normal_tracks' table...")
if any(t['table_name'] == 'normal_tracks' for t in tables):
    print("   [OK] Table 'live.normal_tracks' exists")
    cursor.execute("SELECT COUNT(*) as count FROM live.normal_tracks")
    count = cursor.fetchone()['count']
    print(f"   Total rows in normal_tracks: {count}")
else:
    print("   [ERROR] Table 'live.normal_tracks' does NOT exist!")

# 5. Check monitor status
print("\n5. Checking monitor status...")
try:
    cursor.execute("""
        SELECT is_active, last_update_time, started_by, stopped_by
        FROM public.monitor_status
        WHERE id = 1
    """)
    status = cursor.fetchone()
    if status:
        print(f"   Monitor is_active: {status['is_active']}")
        print(f"   Last update: {status['last_update_time']}")
        print(f"   Started by: {status.get('started_by')}")
        print(f"   Stopped by: {status.get('stopped_by')}")
    else:
        print("   [ERROR] No monitor status found in database")
except Exception as e:
    print(f"   [ERROR] Error checking monitor status: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

conn.close()
