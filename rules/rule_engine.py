from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.db import DbConfig, FlightRepository
from core.models import FlightMetadata, FlightTrack, RuleContext, RuleResult
from rules.rule_logic import evaluate_rule, has_point_above_altitude

logger = logging.getLogger(__name__)

FILTER_MIN_ALTITUDE_FT = 5600.0
FILTER_EXCLUDED_PREFIXES = ("4XC", "4XB", "CHLE", "4XA", "HMR")


class AnomalyRuleEngine:
    def __init__(self, repository: Optional[FlightRepository], rules_path: Path):
        self.repository = repository
        self.rules_path = rules_path
        self._rules = self._load_rules(rules_path)

    @staticmethod
    def _load_rules(path: Path) -> List[Dict[str, object]]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def apply_gateway_filters(self, track: FlightTrack) -> Tuple[bool, Optional[str]]:
        """
        Check if a flight should be filtered out before processing.
        
        Returns:
            Tuple of (should_filter, reason)
        """
        # Check altitude
        if not has_point_above_altitude(track, altitude_ft=FILTER_MIN_ALTITUDE_FT):
            return True, f"No point above {FILTER_MIN_ALTITUDE_FT} ft"

        # Check callsign prefixes
        for point in track.points:
            if point.callsign:
                callsign = point.callsign.strip().upper()
                if callsign.startswith(FILTER_EXCLUDED_PREFIXES):
                    return True, f"Callsign starts with excluded prefix: {callsign}"

        return False, None

    def evaluate_track(
        self,
        track: FlightTrack,
        metadata: Optional[FlightMetadata] = None,
    ) -> Dict[str, object]:
        """
        Evaluate rules against a provided FlightTrack object.
        """
        ctx = RuleContext(track=track, metadata=metadata, repository=self.repository)
        evaluations: List[Dict[str, object]] = []
        
        import time
        for rule_definition in self._rules:
            t_rule = time.time()
            rule_id = int(rule_definition["id"])
            result: RuleResult = evaluate_rule(ctx, rule_id)
            
            duration = time.time() - t_rule
            if duration > 1.0:
                print(f"  [Timer] Rule {rule_id} ({rule_definition.get('name')}): {duration:.4f}s")

            # Add to results (always include full evaluation; UI can filter)
            evaluations.append(
                {
                    "id": rule_id,
                    "name": rule_definition.get("name"),
                    "definition": rule_definition.get("definition"),
                    "operational_significance": rule_definition.get("operational_significance"),
                    # Optional modern metadata for downstream consumers
                    "severity": rule_definition.get("severity"),
                    "category": rule_definition.get("category"),
                    "matched": result.matched,
                    "summary": result.summary,
                    "details": result.details,
                }
            )
            
        return {
            "flight_id": track.flight_id,
            "total_rules": len(self._rules),
            "matched_rules": [rule for rule in evaluations if rule["matched"]],
            "evaluations": evaluations,
        }

    def evaluate_flight(
        self,
        flight_id: str,
        metadata: Optional[FlightMetadata] = None,
    ) -> Dict[str, object]:
        if not self.repository:
            raise ValueError("Repository not initialized")
            
        track = self.repository.fetch_flight(flight_id)
        if not track:
             raise ValueError(f"Flight {flight_id} not found")
             
        return self.evaluate_track(track, metadata)


def load_engine(db_path: Path, rules_path: Path) -> AnomalyRuleEngine:
    repository = FlightRepository(DbConfig(path=db_path))
    return AnomalyRuleEngine(repository, rules_path)
