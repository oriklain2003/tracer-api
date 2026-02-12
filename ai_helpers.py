"""
AI Helper Functions

Provides helper functions for building context and generating visualizations
for AI classification of flight anomalies.
"""
from __future__ import annotations

import io
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def build_anomaly_context(
    anomaly_report: Dict[str, Any],
    metadata: Dict[str, Any],
    flight_data: List[Dict[str, Any]]
) -> str:
    """
    Build formatted context text for AI classification.
    
    Args:
        anomaly_report: Anomaly report from detection pipeline
        metadata: Flight metadata dictionary
        flight_data: List of track point dictionaries
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    # Flight identification
    context_parts.append("=== FLIGHT IDENTIFICATION ===")
    context_parts.append(f"Flight ID: {metadata.get('flight_id', 'Unknown')}")
    context_parts.append(f"Callsign: {metadata.get('callsign', 'Unknown')}")
    
    if metadata.get('flight_number'):
        context_parts.append(f"Flight Number: {metadata.get('flight_number')}")
    if metadata.get('airline'):
        context_parts.append(f"Airline: {metadata.get('airline')}")
    
    # Aircraft details
    if metadata.get('aircraft_type') or metadata.get('aircraft_model'):
        context_parts.append("\n=== AIRCRAFT ===")
        if metadata.get('aircraft_type'):
            context_parts.append(f"Type: {metadata.get('aircraft_type')}")
        if metadata.get('aircraft_model'):
            context_parts.append(f"Model: {metadata.get('aircraft_model')}")
        if metadata.get('aircraft_registration'):
            context_parts.append(f"Registration: {metadata.get('aircraft_registration')}")
    
    # Route information
    context_parts.append("\n=== ROUTE ===")
    context_parts.append(f"Origin: {metadata.get('origin_airport', 'Unknown')}")
    context_parts.append(f"Destination: {metadata.get('destination_airport', 'Unknown')}")
    
    # Flight statistics
    context_parts.append("\n=== FLIGHT STATISTICS ===")
    if metadata.get('flight_duration_sec'):
        duration_min = int(metadata['flight_duration_sec'] / 60)
        context_parts.append(f"Duration: {duration_min} minutes")
    
    if metadata.get('total_distance_nm'):
        context_parts.append(f"Distance: {metadata['total_distance_nm']:.1f} NM")
    
    if metadata.get('total_points'):
        context_parts.append(f"Track Points: {metadata['total_points']}")
    
    # Altitude information
    if metadata.get('cruise_altitude_ft'):
        context_parts.append(f"Cruise Altitude: {metadata['cruise_altitude_ft']:.0f} ft")
    if metadata.get('min_altitude_ft') and metadata.get('max_altitude_ft'):
        context_parts.append(f"Altitude Range: {metadata['min_altitude_ft']:.0f} - {metadata['max_altitude_ft']:.0f} ft")
    
    # Speed information
    if metadata.get('avg_speed_kts'):
        context_parts.append(f"Average Speed: {metadata['avg_speed_kts']:.0f} kts")
    if metadata.get('min_speed_kts') and metadata.get('max_speed_kts'):
        context_parts.append(f"Speed Range: {metadata['min_speed_kts']:.0f} - {metadata['max_speed_kts']:.0f} kts")
    
    # Special indicators
    if metadata.get('is_military'):
        context_parts.append("\n⚠️ MILITARY FLIGHT DETECTED")
        if metadata.get('military_type'):
            context_parts.append(f"Military Type: {metadata['military_type']}")
    
    if metadata.get('emergency_squawk_detected'):
        context_parts.append("\n🚨 EMERGENCY SQUAWK DETECTED")
        if metadata.get('squawk_codes'):
            context_parts.append(f"Squawk Codes: {metadata['squawk_codes']}")
    
    # Anomaly analysis
    if anomaly_report:
        context_parts.append("\n=== ANOMALY ANALYSIS ===")
        
        summary = anomaly_report.get('summary', {})
        if summary:
            context_parts.append(f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}")
            
            if summary.get('confidence_score'):
                context_parts.append(f"Confidence: {summary.get('confidence_score')}%")
            
            if summary.get('severity_cnn'):
                context_parts.append(f"CNN Severity: {summary.get('severity_cnn')}")
            
            if summary.get('severity_dense'):
                context_parts.append(f"Dense Severity: {summary.get('severity_dense')}")
            
            triggers = summary.get('triggers', [])
            if triggers:
                context_parts.append(f"Triggers: {', '.join(str(t) for t in triggers)}")
        
        # Matched rules
        layer1 = anomaly_report.get('layer_1_rules', {})
        if layer1:
            matched_rules = layer1.get('report', {}).get('matched_rules', [])
            if matched_rules:
                context_parts.append("\n=== MATCHED RULES ===")
                for rule in matched_rules:
                    rule_name = rule.get('name', f"Rule {rule.get('id')}")
                    rule_summary = rule.get('summary', '')
                    if rule_summary:
                        context_parts.append(f"  • {rule_name}: {rule_summary}")
                    else:
                        context_parts.append(f"  • {rule_name}")
    
    # Flight data summary
    if flight_data:
        context_parts.append(f"\n=== TRACK DATA ===")
        context_parts.append(f"Total Points: {len(flight_data)}")
        
        if len(flight_data) >= 2:
            first_point = flight_data[0]
            last_point = flight_data[-1]
            
            context_parts.append(f"Start: ({first_point.get('lat', 0):.4f}, {first_point.get('lon', 0):.4f}) at {first_point.get('alt', 0):.0f} ft")
            context_parts.append(f"End: ({last_point.get('lat', 0):.4f}, {last_point.get('lon', 0):.4f}) at {last_point.get('alt', 0):.0f} ft")
    
    return "\n".join(context_parts)


def generate_flight_map(
    flight_data: List[Dict[str, Any]],
    width: int = 800,
    height: int = 600
) -> Optional[bytes]:
    """
    Generate a map visualization of the flight path.
    
    Args:
        flight_data: List of track point dictionaries with lat, lon, alt
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        PNG image bytes, or None if generation fails
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not flight_data or len(flight_data) < 2:
            logger.warning("Insufficient flight data for map generation")
            return None
        
        # Extract coordinates
        lats = [p.get('lat', 0) for p in flight_data if p.get('lat') is not None]
        lons = [p.get('lon', 0) for p in flight_data if p.get('lon') is not None]
        alts = [p.get('alt', 0) for p in flight_data if p.get('alt') is not None]
        
        if not lats or not lons:
            logger.warning("No valid coordinates in flight data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Plot flight path colored by altitude
        if len(alts) == len(lats):
            # Normalize altitudes for color mapping
            norm_alts = np.array(alts)
            scatter = ax.scatter(lons, lats, c=norm_alts, cmap='viridis', s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Altitude (ft)')
        else:
            ax.plot(lons, lats, 'b-', linewidth=1, alpha=0.6)
        
        # Mark start and end points
        ax.plot(lons[0], lats[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(lons[-1], lats[-1], 'ro', markersize=12, label='End', zorder=5)
        
        # Labels and grid
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Flight Path')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add some padding to the view
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        padding = max(lat_range, lon_range) * 0.1
        
        ax.set_xlim(min(lons) - padding, max(lons) + padding)
        ax.set_ylim(min(lats) - padding, max(lats) + padding)
        
        # Save to bytes
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        buf.seek(0)
        image_bytes = buf.read()
        buf.close()
        
        logger.info(f"Generated flight map: {len(image_bytes)} bytes")
        return image_bytes
        
    except ImportError as e:
        logger.error(f"Missing required library for map generation: {e}")
        return None
    except Exception as e:
        logger.error(f"Error generating flight map: {e}")
        return None
