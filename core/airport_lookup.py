"""
Airport lookup from docs/airports.csv (OpenFlights-style).
Resolves ICAO (or ident/gps_code) to country, region, municipality for UI display.
"""

import csv
import os
from typing import Dict, Optional

# In-memory lookup: code (ICAO/ident/gps) -> {country, iso_region, municipality, name, place}
_LOOKUP: Dict[str, Dict[str, str]] = {}
_LOADED = False

# Path to CSV relative to project root
_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "airports.csv")


def _load() -> None:
    global _LOOKUP, _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = os.path.abspath(_CSV_PATH)
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            icao = (row.get("icao_code") or "").strip()
            ident = (row.get("ident") or "").strip()
            gps = (row.get("gps_code") or "").strip()
            iso_country = (row.get("iso_country") or "").strip()
            iso_region = (row.get("iso_region") or "").strip()
            municipality = (row.get("municipality") or "").strip()
            name = (row.get("name") or "").strip()
            if not iso_country:
                continue
            info = {
                "country": iso_country,
                "iso_region": iso_region,
                "municipality": municipality,
                "name": name,
                "place": _place_str(municipality, iso_country, iso_region),
            }
            for code in (icao, ident, gps):
                if code and code not in _LOOKUP:
                    _LOOKUP[code] = info


def _place_str(municipality: str, iso_country: str, iso_region: str) -> str:
    """Human-readable place, e.g. 'Tel Aviv, Israel' or 'US-NY'."""
    if municipality and iso_country:
        return f"{municipality}, {iso_country}"
    if iso_region:
        return iso_region
    return iso_country or ""


def get_airport_place(code: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Resolve airport code (ICAO, or ident/gps from CSV) to country/region/place.
    Returns None if not found or code is empty.
    """
    code = (code or "").strip().upper()
    if not code:
        return None
    _load()
    return _LOOKUP.get(code)


def enrich_origin_destination(
    origin_airport: Optional[str],
    destination_airport: Optional[str],
) -> Dict[str, Optional[Dict[str, str]]]:
    """Return dict with origin_place and destination_place (each None or {country, iso_region, municipality, name, place})."""
    return {
        "origin_place": get_airport_place(origin_airport),
        "destination_place": get_airport_place(destination_airport),
    }
