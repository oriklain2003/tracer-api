"""
Performance benchmarking suite for dashboard statistics endpoints.

This script profiles the current implementation to identify bottlenecks
and measures improvements after optimization.
"""
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Callable
from datetime import datetime, timedelta

from .statistics import StatisticsEngine


class StatisticsBenchmark:
    """Benchmark suite for statistics methods."""
    
    def __init__(self, stats_engine: StatisticsEngine):
        self.engine = stats_engine
        self.results: List[Dict[str, Any]] = []
    
    def benchmark_method(self, method_name: str, method: Callable, 
                        *args, iterations: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a single method.
        
        Args:
            method_name: Name for display
            method: The method to benchmark
            iterations: Number of iterations to average
            *args, **kwargs: Arguments to pass to method
        
        Returns:
            {method, avg_time_ms, min_time_ms, max_time_ms, result_count}
        """
        times = []
        result = None
        
        print(f"\n{'='*60}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*60}")
        
        for i in range(iterations):
            start = time.perf_counter()
            try:
                result = method(*args, **kwargs)
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000
                times.append(elapsed_ms)
                print(f"  Run {i+1}/{iterations}: {elapsed_ms:.2f}ms")
            except Exception as e:
                print(f"  ERROR: {e}")
                return {
                    'method': method_name,
                    'error': str(e),
                    'status': 'failed'
                }
        
        avg_time = sum(times) / len(times)
        result_count = len(result) if isinstance(result, (list, dict)) else 1
        
        benchmark_result = {
            'method': method_name,
            'avg_time_ms': round(avg_time, 2),
            'min_time_ms': round(min(times), 2),
            'max_time_ms': round(max(times), 2),
            'result_count': result_count,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n  ✅ Average: {avg_time:.2f}ms")
        print(f"  📊 Results: {result_count} items")
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def run_full_benchmark(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Run comprehensive benchmark of all dashboard endpoints.
        
        Args:
            start_ts: Start timestamp (Unix)
            end_ts: End timestamp (Unix)
        
        Returns:
            Summary with all results and total time
        """
        print("\n" + "="*60)
        print(f"DASHBOARD API PERFORMANCE BENCHMARK")
        print(f"Date Range: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)}")
        print("="*60)
        
        overall_start = time.perf_counter()
        
        # HIGH PRIORITY - Overview Stats
        print("\n\n🎯 HIGH PRIORITY ENDPOINTS")
        print("-" * 60)
        
        self.benchmark_method(
            "get_overview_stats",
            self.engine.get_overview_stats,
            start_ts, end_ts, use_cache=False
        )
        
        self.benchmark_method(
            "get_emergency_codes_stats",
            self.engine.get_emergency_codes_stats,
            start_ts, end_ts
        )
        
        self.benchmark_method(
            "get_near_miss_events",
            self.engine.get_near_miss_events,
            start_ts, end_ts
        )
        
        self.benchmark_method(
            "get_go_around_stats",
            self.engine.get_go_around_stats,
            start_ts, end_ts
        )
        
        self.benchmark_method(
            "get_flights_per_day",
            self.engine.get_flights_per_day,
            start_ts, end_ts, use_cache=False
        )
        
        self.benchmark_method(
            "get_busiest_airports",
            self.engine.get_busiest_airports,
            start_ts, end_ts, limit=10, use_cache=False
        )
        
        # MEDIUM PRIORITY
        print("\n\n📊 MEDIUM PRIORITY ENDPOINTS")
        print("-" * 60)
        
        self.benchmark_method(
            "get_signal_loss_stats",
            self.engine.get_signal_loss_stats,
            start_ts, end_ts
        )
        
        self.benchmark_method(
            "get_deviations_by_type",
            self.engine.get_deviations_by_type,
            start_ts, end_ts
        )
        
        self.benchmark_method(
            "get_safety_by_phase",
            self.engine.get_safety_by_phase,
            start_ts, end_ts
        )
        
        self.benchmark_method(
            "get_flights_per_month",
            self.engine.get_flights_per_month,
            start_ts, end_ts
        )
        
        # Calculate overall metrics
        overall_end = time.perf_counter()
        total_time = (overall_end - overall_start) * 1000
        
        successful = [r for r in self.results if r.get('status') == 'success']
        failed = [r for r in self.results if r.get('status') == 'failed']
        
        total_avg_time = sum(r['avg_time_ms'] for r in successful)
        
        summary = {
            'total_endpoints_tested': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'total_benchmark_time_ms': round(total_time, 2),
            'total_avg_endpoint_time_ms': round(total_avg_time, 2),
            'results': self.results,
            'date_range': {
                'start': datetime.fromtimestamp(start_ts).isoformat(),
                'end': datetime.fromtimestamp(end_ts).isoformat()
            }
        }
        
        # Print summary
        print("\n\n" + "="*60)
        print("📈 BENCHMARK SUMMARY")
        print("="*60)
        print(f"  Total Endpoints Tested: {summary['total_endpoints_tested']}")
        print(f"  ✅ Successful: {summary['successful']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  ⏱️  Total Benchmark Time: {total_time:.2f}ms")
        print(f"  📊 Sum of Avg Times: {total_avg_time:.2f}ms")
        print("\n  🐌 SLOWEST ENDPOINTS:")
        
        # Sort by avg time and show top 10
        sorted_results = sorted(successful, key=lambda x: x['avg_time_ms'], reverse=True)[:10]
        for i, result in enumerate(sorted_results, 1):
            print(f"     {i}. {result['method']}: {result['avg_time_ms']}ms")
        
        print("="*60)
        
        return summary
    
    def save_results(self, filepath: Path) -> None:
        """Save benchmark results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'benchmark_type': 'dashboard_statistics',
                'timestamp': datetime.now().isoformat(),
                'results': self.results
            }, f, indent=2)
        print(f"\n✅ Results saved to: {filepath}")
    
    def compare_with_baseline(self, baseline_file: Path) -> Dict[str, Any]:
        """
        Compare current results with baseline benchmark.
        
        Args:
            baseline_file: Path to baseline benchmark JSON
        
        Returns:
            Comparison report with improvements/regressions
        """
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        baseline_results = {r['method']: r for r in baseline.get('results', [])}
        
        comparison = {
            'improvements': [],
            'regressions': [],
            'unchanged': []
        }
        
        for current in self.results:
            if current.get('status') != 'success':
                continue
            
            method = current['method']
            if method not in baseline_results:
                continue
            
            baseline_time = baseline_results[method]['avg_time_ms']
            current_time = current['avg_time_ms']
            
            improvement_pct = ((baseline_time - current_time) / baseline_time) * 100
            speedup = baseline_time / current_time if current_time > 0 else 0
            
            comparison_entry = {
                'method': method,
                'baseline_ms': baseline_time,
                'current_ms': current_time,
                'improvement_pct': round(improvement_pct, 1),
                'speedup': round(speedup, 2)
            }
            
            if improvement_pct > 10:
                comparison['improvements'].append(comparison_entry)
            elif improvement_pct < -10:
                comparison['regressions'].append(comparison_entry)
            else:
                comparison['unchanged'].append(comparison_entry)
        
        # Print comparison
        print("\n" + "="*60)
        print("📊 PERFORMANCE COMPARISON VS BASELINE")
        print("="*60)
        
        if comparison['improvements']:
            print(f"\n✅ IMPROVEMENTS ({len(comparison['improvements'])} endpoints)")
            for item in sorted(comparison['improvements'], key=lambda x: x['improvement_pct'], reverse=True):
                print(f"  {item['method']}:")
                print(f"    {item['baseline_ms']}ms → {item['current_ms']}ms")
                print(f"    🚀 {item['speedup']}x faster ({item['improvement_pct']:+.1f}%)")
        
        if comparison['regressions']:
            print(f"\n❌ REGRESSIONS ({len(comparison['regressions'])} endpoints)")
            for item in sorted(comparison['regressions'], key=lambda x: x['improvement_pct']):
                print(f"  {item['method']}:")
                print(f"    {item['baseline_ms']}ms → {item['current_ms']}ms")
                print(f"    🐌 {item['speedup']}x slower ({item['improvement_pct']:+.1f}%)")
        
        print("="*60)
        
        return comparison


def run_benchmark_cli():
    """CLI entry point for running benchmarks."""
    import sys
    from pathlib import Path
    
    # Default: last 7 days
    end_ts = int(time.time())
    start_ts = end_ts - (7 * 86400)
    
    # Initialize engine
    service_dir = Path(__file__).parent.parent
    db_paths = {
        'research': service_dir.parent / 'research_new.db',
        'live': service_dir.parent / 'realtime' / 'live_tracks.db',
        'tagged': service_dir / 'feedback_tagged.db'
    }
    
    engine = StatisticsEngine(db_paths)
    benchmark = StatisticsBenchmark(engine)
    
    # Run benchmark
    results = benchmark.run_full_benchmark(start_ts, end_ts)
    
    # Save results
    output_file = service_dir.parent / f"benchmark_baseline_{int(time.time())}.json"
    benchmark.save_results(output_file)
    
    return results


if __name__ == '__main__':
    run_benchmark_cli()
