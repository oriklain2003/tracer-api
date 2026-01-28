"""
Military Aircraft Detection

Common military detection logic used across the system.
Detects military aircraft based on callsigns, aircraft registrations, and category.
"""

from typing import Tuple, Optional, Dict, List

# Military reference database
# Maps callsign prefixes and registration patterns to military organizations
MILITARY_REFERENCE = {
    "callsigns": {
        # USA
        "RCH": "US Air Force (Air Mobility Command)",
        "REACH": "US Air Force (Air Mobility Command)",
        "TOPCAT": "US Air Force (Refueling)",
        "SPAR": "US Military (Senior Presence, Airborne)",
        "SAM": "US Air Force (Special Air Mission - VIP)",
        "PAT": "US Army (Priority Air Transport)",
        "NAVY": "US Navy",
        "VM": "US Marine Corps",
        "CNV": "US Navy (Convoy)",
        "CONVOY": "US Navy (Convoy)",
        "EVAC": "US Air Force (Medical Evacuation)",
        "TABOO": "US Air Force (Tanker)",
        "QID": "US Air Force (KC-135 Tanker)",
        "QUID": "US Air Force (KC-135 Tanker)",
        "LAGR": "US Air Force (Fighter)",
        "DARK": "US Air Force (ISR)",
        "FORTE": "US Air Force (RQ-4 Global Hawk)",
        "HOMER": "US Air Force (RC-135)",
        "DUKE": "Military (General)",
        "KING": "Military (General)",
        "VIPER": "Military (Fighter)",
        "HAWK": "Military (General)",
        "EAGLE": "Military (Fighter)",
        "N00": "US Navy",
        
        # United Kingdom
        "RRR": "Royal Air Force (ASCOT)",
        "ASCOT": "Royal Air Force (Transport)",
        "SHF": "Royal Navy / RAF (Support Helicopter Force)",
        "AAC": "Army Air Corps",
        "SYS": "RAF Syerston (Training)",
        "TARTN": "Royal Air Force (Tanker)",
        "RAF": "Royal Air Force",
        "RFR": "Royal Air Force (Tanker)",
        
        # Other NATO / International
        "GAF": "German Air Force",
        "BAF": "Belgian Air Force",
        "FAF": "French Air Force",
        "CTM": "French Air Force (Transport)",
        "AME": "Spanish Air Force",
        "PLF": "Polish Air Force",
        "RDAF": "Royal Danish Air Force",
        "ASY": "Royal Australian Air Force",
        "CFC": "Canadian Armed Forces",
        
        # Russian Military
        "RFF": "Russian Air Force (Transport)",
        "RSD": "Russian State Flight",
        
        # Jordanian Military
        "SHAHD": "Royal Jordanian Air Force",
        
        # Iranian Military Drones
        "SHAHED": "Iranian Military (Shahed Drone)",
        
        # Israeli Military
        "IAF": "Israeli Air Force",
        "ISF": "Israeli Air Force",
        
        # Singapore
        "RSAF": "Republic of Singapore Air Force",
    },
    "registration_prefixes": {
        "ZZ": "United Kingdom (RAF)",
        "ZM": "United Kingdom (RAF)",
        "ZH": "United Kingdom (RAF)",
        "10+": "Germany (Luftwaffe)",
        "11+": "Germany (Luftwaffe)",
        "2+": "Germany (Helicopters/Jets)",
        "MM": "Italy (Aeronautica Militare)",
        "FAC": "Colombia (Fuerza Aérea Colombiana)",
        "FAH": "Honduras",
        "4XA": "Israel (IDF Aircraft)",
        "4XB": "Israel (IDF Aircraft)",
        "4XC": "Israel (IDF Aircraft)",
    }
}

# Export list of prefixes for vectorized operations
MILITARY_PREFIXES = list(MILITARY_REFERENCE["callsigns"].keys())

# Known civilian categories that should NEVER be classified as military
CIVILIAN_CATEGORIES = {
    'passenger',
    'cargo',
    'general_aviation',
    'private',
    'charter',
    'business',
    'commercial',
    'airline',
}

def is_military(callsign: Optional[str] = None, 
                aircraft_registration: Optional[str] = None,
                category: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Detect if an aircraft is military based on callsign, registration, or category.
    
    This function checks:
    1. Category (from data source) - civilian categories are immediately rejected
    2. Callsign prefixes (comprehensive database)
    3. Registration patterns
    4. Heuristics (numeric-heavy callsigns) - only if not a known civilian category
    
    Args:
        callsign: Aircraft callsign (e.g., "RCH123", "NAVY45")
        aircraft_registration: Aircraft registration/tail number (e.g., "ZZ123", "4XA-001")
        category: Flight category description (e.g., "military", "military_and_government", "Passenger")
        
    Returns:
        Tuple of (is_military: bool, organization_info: str or None)
        - is_military: True if identified as military
        - organization_info: Description of military organization (e.g., "US Air Force")
    """
    # Normalize category for comparison
    category_lower = category.lower() if category else None
    
    # 1. Check if category explicitly indicates military
    if category_lower in ('military_and_government', 'military'):
        return True, "Military (Category)"
    
    # 2. Check if category explicitly indicates civilian - skip heuristics for these
    is_known_civilian = category_lower in CIVILIAN_CATEGORIES if category_lower else False

    # 3. Check callsign
    if callsign:
        callsign_upper = str(callsign).strip().upper()
        
        # Check if callsign starts with known military prefix
        for prefix, info in MILITARY_REFERENCE["callsigns"].items():
            if callsign_upper.startswith(prefix.upper()):
                return True, info
        
        # Heuristics for unknown patterns - ONLY apply if NOT a known civilian category
        # This prevents false positives like UAL140 (United Airlines) being flagged as military
        # if not is_known_civilian:
        #     # Numeric-heavy callsigns (common in military, e.g. "12345", "A1234")
        #     digit_count = sum(1 for c in callsign_upper if c.isdigit())
        #     alpha_count = sum(1 for c in callsign_upper if c.isalpha())
            
        #     if len(callsign_upper) >= 5:
        #         if digit_count >= 3 and alpha_count >= 1:
        #             return True, "Unidentified Military (Pattern)"
        #         if digit_count > 3:  # Mostly numbers
        #             return True, "Unidentified Military (Numeric)"

    # 4. Check aircraft registration
    if aircraft_registration:
        reg_upper = str(aircraft_registration).strip().upper()
        
        # Check if registration matches military patterns
        for prefix, info in MILITARY_REFERENCE["registration_prefixes"].items():
            if reg_upper.startswith(prefix.upper()):
                return True, info
    
    # Not identified as military
    return False, None


def get_military_type(military_info: Optional[str]) -> Optional[str]:
    """
    Extract military type category from organization info string.
    
    Args:
        military_info: Organization info string from is_military()
        
    Returns:
        Military type category (transport, tanker, fighter, ISR, etc.) or None
    """
    if not military_info:
        return None
    
    info_lower = military_info.lower()
    
    # Type classification based on info string
    if "transport" in info_lower or "mobility" in info_lower or "convoy" in info_lower:
        return "transport"
    elif "tanker" in info_lower or "refuel" in info_lower:
        return "tanker"
    elif "fighter" in info_lower:
        return "fighter"
    elif "isr" in info_lower or "recce" in info_lower or "hawk" in info_lower or "rc-135" in info_lower:
        return "ISR"
    elif "medical" in info_lower or "evac" in info_lower:
        return "medical"
    elif "vip" in info_lower or "special air mission" in info_lower or "executive" in info_lower:
        return "vip"
    elif "helicopter" in info_lower:
        return "helicopter"
    elif "training" in info_lower:
        return "training"
    elif "drone" in info_lower or "shahed" in info_lower:
        return "drone"
    else:
        return "military"
