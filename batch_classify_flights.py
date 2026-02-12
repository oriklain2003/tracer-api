"""
Batch Flight Classification Script

Processes a list of flight IDs from the research schema and generates
AI classifications for each flight. Useful for classifying historical
anomalies that were detected before the AI system was implemented.

Usage:
    python batch_classify_flights.py --flight-ids 3d7211ef 3cf959dd 3ad2166a
    python batch_classify_flights.py --file flight_ids.txt
    python batch_classify_flights.py --all-anomalies --limit 100
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_classify.log")
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
from service.pg_provider import get_connection, create_ai_classifications_table, save_ai_classification
from ai_classify import AIClassifier


def fetch_flight_data(flight_id: str, schema: str = 'research') -> Optional[Dict[str, Any]]:
    """
    Fetch all necessary data for a flight from the database.
    
    Args:
        flight_id: Flight identifier
        schema: Database schema (default: 'research')
    
    Returns:
        Dict containing flight_data, metadata, and anomaly_report, or None if not found
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Fetch flight metadata
                cursor.execute(f"""
                    SELECT 
                        flight_id, callsign, flight_number, airline, airline_code,
                        aircraft_type, aircraft_model, aircraft_registration,
                        origin_airport, destination_airport,
                        first_seen_ts, last_seen_ts, flight_duration_sec,
                        total_distance_nm, total_points,
                        min_altitude_ft, max_altitude_ft, avg_altitude_ft, cruise_altitude_ft,
                        min_speed_kts, max_speed_kts, avg_speed_kts,
                        start_lat, start_lon, end_lat, end_lon,
                        squawk_codes, emergency_squawk_detected,
                        is_anomaly, is_military, military_type,
                        flight_phase_summary, nearest_airport_start, nearest_airport_end,
                        crossed_borders, signal_loss_events, data_quality_score,
                        category
                    FROM {schema}.flight_metadata
                    WHERE flight_id = %s
                """, (flight_id,))
                
                metadata_row = cursor.fetchone()
                if not metadata_row:
                    logger.warning(f"Flight {flight_id} not found in {schema}.flight_metadata")
                    return None
                
                # Convert to dict
                metadata_cols = [
                    'flight_id', 'callsign', 'flight_number', 'airline', 'airline_code',
                    'aircraft_type', 'aircraft_model', 'aircraft_registration',
                    'origin_airport', 'destination_airport',
                    'first_seen_ts', 'last_seen_ts', 'flight_duration_sec',
                    'total_distance_nm', 'total_points',
                    'min_altitude_ft', 'max_altitude_ft', 'avg_altitude_ft', 'cruise_altitude_ft',
                    'min_speed_kts', 'max_speed_kts', 'avg_speed_kts',
                    'start_lat', 'start_lon', 'end_lat', 'end_lon',
                    'squawk_codes', 'emergency_squawk_detected',
                    'is_anomaly', 'is_military', 'military_type',
                    'flight_phase_summary', 'nearest_airport_start', 'nearest_airport_end',
                    'crossed_borders', 'signal_loss_events', 'data_quality_score',
                    'category'
                ]
                metadata = dict(zip(metadata_cols, metadata_row))
                
                # Fetch track points
                cursor.execute(f"""
                    SELECT timestamp, lat, lon, alt, gspeed, vspeed, track, squawk, callsign
                    FROM {schema}.{"flight_tracks" if schema == "feedback" else "normal_tracks"}
                    WHERE flight_id = %s
                    ORDER BY timestamp
                """, (flight_id,))
                
                track_rows = cursor.fetchall()
                if not track_rows:
                    logger.warning(f"No track points found for flight {flight_id}")
                    return None
                
                flight_data = [
                    {
                        'timestamp': row[0],
                        'lat': row[1],
                        'lon': row[2],
                        'alt': row[3],
                        'gspeed': row[4],
                        'vspeed': row[5],
                        'track': row[6],
                        'squawk': row[7],
                        'callsign': row[8]
                    }
                    for row in track_rows
                ]
                
                # Fetch anomaly report
                cursor.execute(f"""
                    SELECT full_report
                    FROM {schema}.anomaly_reports
                    WHERE flight_id = %s
                """, (flight_id,))
                
                report_row = cursor.fetchone()
                if not report_row or not report_row[0]:
                    logger.warning(f"No anomaly report found for flight {flight_id}")
                    # Create minimal report structure
                    anomaly_report = {
                        'summary': {
                            'flight_id': flight_id,
                            'is_anomaly': metadata.get('is_anomaly', False),
                            'confidence_score': 0,
                            'triggers': []
                        }
                    }
                else:
                    import json
                    anomaly_report = json.loads(report_row[0]) if isinstance(report_row[0], str) else report_row[0]
                
                return {
                    'flight_id': flight_id,
                    'flight_data': flight_data,
                    'metadata': metadata,
                    'anomaly_report': anomaly_report
                }
                
    except Exception as e:
        logger.error(f"Error fetching data for flight {flight_id}: {e}")
        return None


def check_already_classified(flight_id: str, schema: str = 'research') -> bool:
    """
    Check if a flight has already been classified.
    
    Args:
        flight_id: Flight identifier
        schema: Database schema
    
    Returns:
        True if already classified, False otherwise
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM {schema}.ai_classifications
                    WHERE flight_id = %s
                """, (flight_id,))
                count = cursor.fetchone()[0]
                return count > 0
    except Exception as e:
        logger.error(f"Error checking classification status for {flight_id}: {e}")
        return False


def fetch_unclassified_anomalies(schema: str = 'research', limit: int = 100) -> List[str]:
    """
    Fetch flight IDs of anomalies that haven't been classified yet.
    
    Args:
        schema: Database schema
        limit: Maximum number of flights to fetch
    
    Returns:
        List of flight IDs
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT ar.flight_id
                    FROM {schema}.anomaly_reports ar
                    LEFT JOIN {schema}.ai_classifications ac ON ar.flight_id = ac.flight_id
                    WHERE ar.is_anomaly = TRUE
                    AND ac.flight_id IS NULL
                    ORDER BY ar.timestamp DESC
                    LIMIT %s
                """, (limit,))
                
                flight_ids = [row[0] for row in cursor.fetchall()]
                logger.info(f"Found {len(flight_ids)} unclassified anomalies in {schema} schema")
                return flight_ids
                
    except Exception as e:
        logger.error(f"Error fetching unclassified anomalies: {e}")
        return []


def classify_flight(
    classifier: AIClassifier,
    flight_id: str,
    schema: str = 'research',
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Classify a single flight.
    
    Args:
        classifier: AIClassifier instance
        flight_id: Flight identifier
        schema: Database schema
        skip_existing: Skip if already classified
    
    Returns:
        Dict with classification result
    """
    result = {
        'flight_id': flight_id,
        'success': False,
        'skipped': False,
        'error': None,
        'classification_text': None
    }
    
    try:
        # Check if already classified
        if skip_existing and check_already_classified(flight_id, schema):
            logger.info(f"✓ Flight {flight_id} already classified, skipping")
            result['skipped'] = True
            result['success'] = True
            return result
        
        # Fetch flight data
        logger.info(f"→ Fetching data for flight {flight_id}...")
        data = fetch_flight_data(flight_id, schema)
        
        if not data:
            result['error'] = "Flight data not found"
            return result
        
        # Classify synchronously (we're already in a worker thread)
        logger.info(f"→ Classifying flight {flight_id}...")
        classification_result = classifier._classify_sync(
            flight_id=data['flight_id'],
            flight_data=data['flight_data'],
            anomaly_report=data['anomaly_report'],
            metadata=data['metadata']
        )
        
        if classification_result.get('error_message'):
            result['error'] = classification_result['error_message']
        else:
            result['success'] = True
            result['classification_text'] = classification_result.get('classification_text')
            logger.info(f"✓ Flight {flight_id} classified: '{result['classification_text']}'")
        
    except Exception as e:
        logger.error(f"✗ Error classifying flight {flight_id}: {e}")
        result['error'] = str(e)
    
    return result


def batch_classify(
    flight_ids: List[str],
    schema: str = 'research',
    api_key: Optional[str] = None,
    max_workers: int = 2,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Classify multiple flights in parallel.
    
    Args:
        flight_ids: List of flight IDs to classify
        schema: Database schema
        api_key: Gemini API key (reads from env if not provided)
        max_workers: Number of parallel workers
        skip_existing: Skip already classified flights
    
    Returns:
        Dict with summary statistics
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY" , "AIzaSyBArSFAlxqm-9q1hWbaNgeT7f3WMOqF5Go")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Set environment variable or pass as argument.")
    
    # Ensure table exists
    logger.info(f"Ensuring ai_classifications table exists in {schema} schema...")
    create_ai_classifications_table(schema)
    
    # Initialize classifier (we'll use it synchronously in threads)
    logger.info("Initializing AI Classifier...")
    classifier = AIClassifier(api_key, schema=schema, max_workers=1)
    
    # Process flights
    logger.info(f"Starting batch classification of {len(flight_ids)} flights...")
    logger.info(f"Schema: {schema}")
    logger.info(f"Workers: {max_workers}")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info("=" * 80)
    
    start_time = time.time()
    results = {
        'total': len(flight_ids),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': [],
        'duration_sec': 0
    }
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_flight = {
            executor.submit(classify_flight, classifier, fid, schema, skip_existing): fid
            for fid in flight_ids
        }
        
        # Process as they complete
        for future in as_completed(future_to_flight):
            flight_id = future_to_flight[future]
            try:
                result = future.result()
                
                if result['skipped']:
                    results['skipped'] += 1
                elif result['success']:
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'flight_id': flight_id,
                        'error': result['error']
                    })
                    
            except Exception as e:
                logger.error(f"✗ Exception processing flight {flight_id}: {e}")
                results['failed'] += 1
                results['errors'].append({
                    'flight_id': flight_id,
                    'error': str(e)
                })
    
    results['duration_sec'] = time.time() - start_time
    
    # Print summary
    logger.info("=" * 80)
    logger.info("BATCH CLASSIFICATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total flights: {results['total']}")
    logger.info(f"Successfully classified: {results['success']}")
    logger.info(f"Skipped (already classified): {results['skipped']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Duration: {results['duration_sec']:.1f} seconds")
    logger.info(f"Average time per flight: {results['duration_sec']/results['total']:.1f} seconds")
    
    if results['errors']:
        logger.info("\nERRORS:")
        for err in results['errors'][:10]:  # Show first 10 errors
            logger.info(f"  - {err['flight_id']}: {err['error']}")
        if len(results['errors']) > 10:
            logger.info(f"  ... and {len(results['errors']) - 10} more errors")
    
    logger.info("=" * 80)
    
    # Cleanup
    classifier.shutdown(wait=True)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch classify flights using AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify specific flights
  python batch_classify_flights.py --flight-ids 3d7211ef 3cf959dd 3ad2166a

  # Classify from file (one flight ID per line)
  python batch_classify_flights.py --file flight_ids.txt

  # Classify all unclassified anomalies (limit 100)
  python batch_classify_flights.py --all-anomalies --limit 100

  # Use different schema
  python batch_classify_flights.py --all-anomalies --schema live --limit 50

  # Force re-classify even if already classified
  python batch_classify_flights.py --flight-ids 3d7211ef --no-skip-existing
        """
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--flight-ids',
        nargs='+',
        help='Flight IDs to classify (space-separated)'
    )
    input_group.add_argument(
        '--file',
        type=str,
        help='File containing flight IDs (one per line)'
    )
    input_group.add_argument(
        '--all-anomalies',
        action='store_true',
        help='Classify all unclassified anomalies in the schema'
    )
    
    # Options
    parser.add_argument(
        '--schema',
        type=str,
        default='research',
        help='Database schema to use (default: research)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of flights to process with --all-anomalies (default: 100)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=2,
        help='Number of parallel workers (default: 2)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-classify flights even if already classified'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Gemini API key (reads from GEMINI_API_KEY env var if not provided)'
    )
    
    args = parser.parse_args()
    
    # Get flight IDs based on input source
    flight_ids = []
    
    if args.flight_ids:
        flight_ids = args.flight_ids
        logger.info(f"Processing {len(flight_ids)} flight IDs from command line")
        
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                flight_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(flight_ids)} flight IDs from {args.file}")
        except Exception as e:
            logger.error(f"Error reading file {args.file}: {e}")
            sys.exit(1)
            
    elif args.all_anomalies:
        logger.info(f"Fetching unclassified anomalies from {args.schema} schema...")
        flight_ids = fetch_unclassified_anomalies(args.schema, args.limit)
        if not flight_ids:
            logger.info("No unclassified anomalies found!")
            sys.exit(0)
    
    if not flight_ids:
        logger.error("No flight IDs to process")
        sys.exit(1)
    
    # Run batch classification
    try:
        results = batch_classify(
            flight_ids=flight_ids,
            schema=args.schema,
            api_key=args.api_key,
            max_workers=args.max_workers,
            skip_existing=not args.no_skip_existing
        )
        
        # Exit with error code if any failures
        sys.exit(0 if results['failed'] == 0 else 1)
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
