"""
Routes package for the Anomaly Detection Service API.

This package contains all API route handlers split by domain:
- flights: Flight data, tracks, analysis, rules
- feedback: User feedback, tagging, history  
- analytics: Statistics, trends, intelligence, predictions
- ai_routes: Chat, AI analysis, reasoning
- route_planning: Route planning and conflict detection
"""

# Note: Routers are imported directly in api.py to avoid circular imports
# This module is primarily for documentation purposes

__all__ = [
    'flights',
    'feedback', 
    'analytics',
    'ai_routes',
    'route_planning',
]

