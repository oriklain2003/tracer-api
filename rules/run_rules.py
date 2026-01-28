from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory to path so we can import core
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.db import DbConfig, FlightRepository
from core.models import FlightMetadata, FlightTrack
# If running from root, rule_engine is in rules.rule_engine, but if running from rules dir, it is rule_engine
try:
    from rules.rule_engine import AnomalyRuleEngine
except ImportError:
    from rule_engine import AnomalyRuleEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import argparse

# Default configuration values
# Check if files exist in current directory (running from root) or parent (running from rules)
DEFAULT_DB_PATH = Path("last.db") if Path("last.db").exists() else Path("../last.db")
RULES_PATH = Path("anomaly_rule.json") if Path("anomaly_rule.json").exists() else Path("../anomaly_rule.json")
METADATA_PATH = None  # Optional
FILTERED_IDS_FILE = Path("filtered_flight_ids.json")
ANOMALOUS_IDS_FILE = Path("anomalous_flight_ids.json")


def load_metadata(path: Path | None) -> FlightMetadata | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return FlightMetadata(
        planned_destination=raw.get("planned_destination"),
        planned_route=raw.get("planned_route"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    args = parser.parse_args()
    
    db_path = args.db if args.db else DEFAULT_DB_PATH
    logger.info(f"Using database: {db_path}")

    repository = FlightRepository(DbConfig(path=db_path))
    engine = AnomalyRuleEngine(repository, RULES_PATH)
    metadata = load_metadata(METADATA_PATH)
    
    # Process all flights
    results = []
    skipped_flights = []
    anomalous_flights = []
    
    for flight_track in repository.iter_flights():
        # Apply gateway filters from rule engine
        should_filter, reason = engine.apply_gateway_filters(flight_track)
        if should_filter:
            skipped_flights.append({
                "flight_id": flight_track.flight_id,
                "reason": reason
            })
            logger.info(f"Filtered flight {flight_track.flight_id}: {reason}")
            continue
        
        report = engine.evaluate_flight(flight_track.flight_id, metadata=metadata)
        if report["matched_rules"]:
            results.append({
                "flight_id": report["flight_id"],
                "matched_rules": report["matched_rules"],
                "total_rules": report["total_rules"],
                "matched_count": len(report["matched_rules"])
            })
            anomalous_flights.append(report["flight_id"])

    # Save filtered flight IDs to file
    if skipped_flights:
        # Use absolute path to ensure file is created in the rules directory
        filtered_file_path = Path(__file__).parent / FILTERED_IDS_FILE
        with filtered_file_path.open("w", encoding="utf-8") as f:
            json.dump({"filtered_flights": skipped_flights}, f, indent=2)
        logger.info(f"Saved {len(skipped_flights)} filtered flight IDs to {filtered_file_path}")
    
    # Save anomalous flight IDs to file
    if anomalous_flights:
        anomalous_file_path = Path(__file__).parent / ANOMALOUS_IDS_FILE
        with anomalous_file_path.open("w", encoding="utf-8") as f:
            json.dump({"anomalous_flights": anomalous_flights}, f, indent=2)
        logger.info(f"Saved {len(anomalous_flights)} anomalous flight IDs to {anomalous_file_path}")

    output = {
        "total_flights": len(results) + len(skipped_flights) + (len([f for f in repository.iter_flights()]) - len(results) - len(skipped_flights)), # Approximate
        "flights_processed": len(results) + len(skipped_flights) + (len([f for f in repository.iter_flights()]) - len(results) - len(skipped_flights)),
        "flights_with_anomalies": len(anomalous_flights),
        "skipped_flights_count": len(skipped_flights),
        "results": results
    }
    # Recalculate total because generator was consumed
    output["total_flights"] = "N/A" 
    output["flights_processed"] = "N/A"
    
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
