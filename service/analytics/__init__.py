"""
Analytics module for FiveAir Intelligence Dashboard.

Provides statistical aggregation, trend analysis, intelligence gathering,
and predictive analytics for flight data.
"""

from .statistics import StatisticsEngine
from .queries import QueryBuilder
from .trends import TrendsAnalyzer
from .intelligence import IntelligenceEngine
from .predictive import PredictiveAnalytics

__all__ = [
    'StatisticsEngine',
    'QueryBuilder',
    'TrendsAnalyzer',
    'IntelligenceEngine',
    'PredictiveAnalytics'
]

