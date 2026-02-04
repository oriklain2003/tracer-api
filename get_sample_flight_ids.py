"""
Helper script to find sample flight IDs from PostgreSQL for profiling.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR

print("="*80)
print("SAMPLE FLIGHT IDs FOR PROFILING")
print("="*80)

sample_flight_id = None

# Try to get from PostgreSQL
try:
    from service.pg_provider import get_pool
    pool = get_pool()
    
    # Check research.normal_tracks
    with pool.get_connection() as conn:
        with conn.cursor() as cursor:
            print("\n[PostgreSQL] research.normal_tracks")
            cursor.execute("""
                SELECT DISTINCT flight_id, MIN(timestamp) as first_seen, COUNT(*) as points
                FROM research.normal_tracks
                GROUP BY flight_id
                ORDER BY MIN(timestamp) DESC
                LIMIT 5
            """)
            rows = cursor.fetchall()
            
            if rows:
                for i, row in enumerate(rows, 1):
                    print(f"  {i}. {row[0]} ({row[2]} points)")
                sample_flight_id = rows[0][0]
                print(f"\nExample commands:")
                print(f"  python profile_routes.py --route track --flight-id {sample_flight_id}")
                print(f"  python profile_detailed.py --route track --flight-id {sample_flight_id}")
            else:
                print("  No data found")
                sample_flight_id = None
    
    # Check feedback.user_feedback
    with pool.get_connection() as conn:
        with conn.cursor() as cursor:
            print("\n[PostgreSQL] feedback.user_feedback")
            cursor.execute("""
                SELECT flight_id, first_seen_ts, rule_name, user_label
                FROM feedback.user_feedback
                ORDER BY first_seen_ts DESC
                LIMIT 5
            """)
            rows = cursor.fetchall()
            
            if rows:
                for i, row in enumerate(rows, 1):
                    label = "anomaly" if row[3] == 1 else "normal"
                    print(f"  {i}. {row[0]} - {row[2]} ({label})")
                    
                # Get count
                cursor.execute("SELECT COUNT(*) FROM feedback.user_feedback")
                count = cursor.fetchone()[0]
                print(f"\nTotal feedback entries: {count}")
                
                print(f"\nExample commands for feedback history:")
                print(f"  python profile_routes.py --route feedback-history")
                print(f"  python profile_detailed.py --route feedback-history")
            else:
                print("  No data found")
    
except Exception as e:
    print(f"\n[X] PostgreSQL error: {e}")
    import traceback
    traceback.print_exc()
    sample_flight_id = None

print("\n" + "="*80)
print("PROFILING COMMANDS")
print("="*80)
print("\nTo profile the unified track route:")
print("  python profile_routes.py --route track --flight-id <FLIGHT_ID>")
print("  python profile_detailed.py --route track --flight-id <FLIGHT_ID>")
print("\nTo profile the feedback history route:")
print("  python profile_routes.py --route feedback-history")
print("  python profile_detailed.py --route feedback-history")
print("\nSee PROFILING_README.md for detailed instructions.")
print("="*80)
