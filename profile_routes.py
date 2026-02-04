"""
Profile slow API routes to identify performance bottlenecks.

This script profiles:
1. /api/track/unified/{flight_id} - Get unified track data
2. /api/feedback/history - Get feedback history

Usage:
    python profile_routes.py --route track --flight-id <flight_id>
    python profile_routes.py --route feedback-history [--start-ts <ts>] [--end-ts <ts>] [--limit <n>]
"""
import cProfile
import pstats
import io
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import route modules
from routes import flights, feedback
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# Configure paths (same as api.py)
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


def profile_unified_track(flight_id: str, output_file: str = None):
    """Profile the unified track endpoint."""
    logger.info(f"Profiling unified track for flight_id: {flight_id}")
    
    # Import necessary functions
    from flight_fetcher import serialize_flight, deserialize_flight, search_flight_path, get as get_flight
    from anomaly_pipeline import AnomalyPipeline
    
    # Configure flights router
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
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the function
    logger.info("Starting profiling...")
    start_time = time.time()
    
    profiler.enable()
    try:
        result = flights.get_unified_track(flight_id)
        profiler.disable()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ Function completed in {elapsed_time:.2f} seconds")
        
        if result:
            if isinstance(result, dict):
                points_count = len(result.get('points', []))
                logger.info(f"  Result: {points_count} track points returned")
            else:
                logger.info(f"  Result: {type(result)}")
        else:
            logger.info("  Result: No data found")
            
    except Exception as e:
        profiler.disable()
        logger.error(f"✗ Error during profiling: {e}", exc_info=True)
        return
    
    # Print statistics
    print("\n" + "="*80)
    print("PROFILING RESULTS: /api/track/unified/{flight_id}")
    print("="*80)
    print(f"Flight ID: {flight_id}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print("="*80 + "\n")
    
    # Create stats object
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Sort by cumulative time and print top 30 functions
    print("Top 30 functions by CUMULATIVE time:")
    print("-" * 80)
    ps.sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Sort by total time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    print("\n" + "="*80)
    print("Top 30 functions by TOTAL time:")
    print("-" * 80)
    ps.sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Save to file if requested
    if output_file:
        profiler.dump_stats(output_file)
        logger.info(f"Profile data saved to: {output_file}")
        logger.info(f"You can analyze it with: python -m pstats {output_file}")


def profile_feedback_history(start_ts: int = 0, end_ts: int = None, limit: int = 100, 
                             include_normal: bool = True, output_file: str = None):
    """Profile the feedback history endpoint."""
    if end_ts is None:
        end_ts = int(datetime.now().timestamp())
    
    logger.info(f"Profiling feedback history")
    logger.info(f"  Parameters: start_ts={start_ts}, end_ts={end_ts}, limit={limit}, include_normal={include_normal}")
    
    # Import necessary functions
    from flight_fetcher import serialize_flight, deserialize_flight, search_flight_path, get as get_flight
    from anomaly_pipeline import AnomalyPipeline
    from training_ops.db_utils import save_feedback
    
    # Configure flights router (needed for get_unified_track)
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
    
    # Configure feedback router
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
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the function
    logger.info("Starting profiling...")
    start_time = time.time()
    
    profiler.enable()
    try:
        result = feedback.get_feedback_history(
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
            include_normal=include_normal
        )
        profiler.disable()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ Function completed in {elapsed_time:.2f} seconds")
        
        if result:
            logger.info(f"  Result: {len(result)} flights returned")
        else:
            logger.info("  Result: Empty list")
            
    except Exception as e:
        profiler.disable()
        logger.error(f"✗ Error during profiling: {e}", exc_info=True)
        return
    
    # Print statistics
    print("\n" + "="*80)
    print("PROFILING RESULTS: /api/feedback/history")
    print("="*80)
    print(f"Parameters: start_ts={start_ts}, end_ts={end_ts}, limit={limit}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print("="*80 + "\n")
    
    # Create stats object
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Sort by cumulative time and print top 30 functions
    print("Top 30 functions by CUMULATIVE time:")
    print("-" * 80)
    ps.sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Sort by total time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    print("\n" + "="*80)
    print("Top 30 functions by TOTAL time:")
    print("-" * 80)
    ps.sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Save to file if requested
    if output_file:
        profiler.dump_stats(output_file)
        logger.info(f"Profile data saved to: {output_file}")
        logger.info(f"You can analyze it with: python -m pstats {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Profile slow API routes')
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
    
    # Output options
    parser.add_argument('--output', help='Output file for profile data (.prof)')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.route == 'track':
            args.output = f'profile_track_{timestamp}.prof'
        else:
            args.output = f'profile_feedback_history_{timestamp}.prof'
    
    if args.route == 'track':
        if not args.flight_id:
            parser.error('--flight-id is required for track route')
        profile_unified_track(args.flight_id, args.output)
    
    elif args.route == 'feedback-history':
        profile_feedback_history(
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            limit=args.limit,
            include_normal=not args.exclude_normal,
            output_file=args.output
        )


if __name__ == '__main__':
    main()
