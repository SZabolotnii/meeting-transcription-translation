"""
Performance profiler and optimization utilities.
Provides tools for measuring and optimizing system performance.
"""

import time
import threading
import psutil
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
from datetime import datetime, timedelta


@dataclass
class PerformanceMetric:
    """Single performance metric measurement"""
    name: str
    value: float
    timestamp: float
    unit: str = "seconds"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentStats:
    """Statistics for a system component"""
    name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    
    def add_measurement(self, duration: float):
        """Add a new measurement"""
        self.total_calls += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.total_calls
        self.recent_times.append(duration)
    
    def add_error(self):
        """Record an error"""
        self.error_count += 1
    
    def get_recent_avg(self) -> float:
        """Get average of recent measurements"""
        if not self.recent_times:
            return 0.0
        return statistics.mean(self.recent_times)
    
    def get_recent_p95(self) -> float:
        """Get 95th percentile of recent measurements"""
        if not self.recent_times:
            return 0.0
        sorted_times = sorted(self.recent_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]


class PerformanceProfiler:
    """
    Performance profiler for monitoring system performance.
    Tracks timing, resource usage, and provides optimization recommendations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: List[PerformanceMetric] = []
        self.component_stats: Dict[str, ComponentStats] = defaultdict(ComponentStats)
        self.system_stats: Dict[str, deque] = {
            'cpu_percent': deque(maxlen=300),  # 5 minutes at 1s intervals
            'memory_percent': deque(maxlen=300),
            'memory_mb': deque(maxlen=300),
        }
        
        # Performance thresholds
        self.thresholds = {
            'transcription_latency': 3.0,  # seconds
            'translation_latency': 2.0,    # seconds
            'total_latency': 5.0,          # seconds
            'cpu_percent': 80.0,           # percent
            'memory_percent': 85.0,        # percent
            'error_rate': 0.05,            # 5%
        }
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Start system monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system_resources,
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_system_resources(self):
        """Monitor system resources in background thread"""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1.0)
                self.system_stats['cpu_percent'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_stats['memory_percent'].append(memory.percent)
                self.system_stats['memory_mb'].append(memory.used / 1024 / 1024)
                
                # Log warnings if thresholds exceeded
                if cpu_percent > self.thresholds['cpu_percent']:
                    self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                if memory.percent > self.thresholds['memory_percent']:
                    self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Error monitoring system resources: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def measure_component(self, component_name: str):
        """Decorator for measuring component performance"""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Update component stats
                    if component_name not in self.component_stats:
                        self.component_stats[component_name] = ComponentStats(component_name)
                    
                    self.component_stats[component_name].add_measurement(duration)
                    
                    # Add metric
                    self.add_metric(f"{component_name}_duration", duration, "seconds")
                    
                    # Check thresholds
                    self._check_component_thresholds(component_name, duration)
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    if component_name not in self.component_stats:
                        self.component_stats[component_name] = ComponentStats(component_name)
                    
                    self.component_stats[component_name].add_error()
                    self.logger.error(f"Error in {component_name}: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    def _check_component_thresholds(self, component_name: str, duration: float):
        """Check if component performance exceeds thresholds"""
        threshold_key = f"{component_name}_latency"
        if threshold_key in self.thresholds:
            threshold = self.thresholds[threshold_key]
            if duration > threshold:
                self.logger.warning(
                    f"{component_name} latency exceeded threshold: "
                    f"{duration:.2f}s > {threshold:.2f}s"
                )
    
    def add_metric(self, name: str, value: float, unit: str = "seconds", 
                   metadata: Optional[Dict[str, Any]] = None):
        """Add a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            unit=unit,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        
        # Keep only recent metrics (last hour)
        cutoff_time = time.time() - 3600
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
    
    def get_component_stats(self, component_name: str) -> Optional[ComponentStats]:
        """Get statistics for a specific component"""
        return self.component_stats.get(component_name)
    
    def get_all_component_stats(self) -> Dict[str, ComponentStats]:
        """Get statistics for all components"""
        return dict(self.component_stats)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        stats = {}
        
        for stat_name, values in self.system_stats.items():
            if values:
                stats[stat_name] = {
                    'current': values[-1],
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                stats[stat_name] = {
                    'current': 0,
                    'avg': 0,
                    'min': 0,
                    'max': 0,
                    'count': 0
                }
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system': self.get_system_stats(),
            'components': {},
            'recommendations': self.get_optimization_recommendations()
        }
        
        # Component statistics
        for name, stats in self.component_stats.items():
            summary['components'][name] = {
                'total_calls': stats.total_calls,
                'avg_time': stats.avg_time,
                'recent_avg': stats.get_recent_avg(),
                'recent_p95': stats.get_recent_p95(),
                'min_time': stats.min_time if stats.min_time != float('inf') else 0,
                'max_time': stats.max_time,
                'error_count': stats.error_count,
                'error_rate': stats.error_count / max(stats.total_calls, 1)
            }
        
        return summary
    
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on current performance"""
        recommendations = []
        
        # Check system resources
        system_stats = self.get_system_stats()
        
        if system_stats['cpu_percent']['avg'] > 70:
            recommendations.append({
                'type': 'cpu',
                'severity': 'high' if system_stats['cpu_percent']['avg'] > 85 else 'medium',
                'message': f"High CPU usage ({system_stats['cpu_percent']['avg']:.1f}%)",
                'suggestion': "Consider using smaller Whisper model or reducing concurrent tasks"
            })
        
        if system_stats['memory_percent']['avg'] > 75:
            recommendations.append({
                'type': 'memory',
                'severity': 'high' if system_stats['memory_percent']['avg'] > 90 else 'medium',
                'message': f"High memory usage ({system_stats['memory_percent']['avg']:.1f}%)",
                'suggestion': "Reduce history size or queue sizes in configuration"
            })
        
        # Check component performance
        for name, stats in self.component_stats.items():
            if stats.total_calls > 0:
                error_rate = stats.error_count / stats.total_calls
                
                if error_rate > self.thresholds['error_rate']:
                    recommendations.append({
                        'type': 'reliability',
                        'severity': 'high',
                        'message': f"High error rate in {name} ({error_rate:.1%})",
                        'suggestion': f"Check {name} configuration and logs for issues"
                    })
                
                if name == 'transcription' and stats.get_recent_avg() > self.thresholds['transcription_latency']:
                    recommendations.append({
                        'type': 'performance',
                        'severity': 'medium',
                        'message': f"Slow transcription ({stats.get_recent_avg():.1f}s avg)",
                        'suggestion': "Use smaller Whisper model or enable GPU acceleration"
                    })
                
                if name == 'translation' and stats.get_recent_avg() > self.thresholds['translation_latency']:
                    recommendations.append({
                        'type': 'performance',
                        'severity': 'medium',
                        'message': f"Slow translation ({stats.get_recent_avg():.1f}s avg)",
                        'suggestion': "Check internet connection or increase batch size"
                    })
        
        return recommendations
    
    def export_performance_report(self, filepath: str):
        """Export detailed performance report to file"""
        report = {
            'export_time': datetime.now().isoformat(),
            'summary': self.get_performance_summary(),
            'detailed_metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'timestamp': m.timestamp,
                    'unit': m.unit,
                    'metadata': m.metadata
                }
                for m in self.metrics[-1000:]  # Last 1000 metrics
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Performance report exported to {filepath}")
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """Generate optimized configuration based on current performance"""
        recommendations = self.get_optimization_recommendations()
        system_stats = self.get_system_stats()
        
        optimized_config = {}
        
        # CPU optimization
        if system_stats['cpu_percent']['avg'] > 80:
            optimized_config['WHISPER_MODEL_SIZE'] = 'tiny'
            optimized_config['MAX_CONCURRENT_TASKS'] = 2
            optimized_config['AUDIO_BUFFER_DURATION'] = 7.0
        elif system_stats['cpu_percent']['avg'] > 60:
            optimized_config['WHISPER_MODEL_SIZE'] = 'base'
            optimized_config['MAX_CONCURRENT_TASKS'] = 2
        
        # Memory optimization
        if system_stats['memory_percent']['avg'] > 80:
            optimized_config['MAX_HISTORY_ENTRIES'] = 500
            optimized_config['HISTORY_CLEANUP_THRESHOLD'] = 250
            optimized_config['AUDIO_QUEUE_SIZE'] = 5
            optimized_config['TRANSCRIPTION_QUEUE_SIZE'] = 5
            optimized_config['RESULT_QUEUE_SIZE'] = 25
        
        # Performance optimization based on component stats
        transcription_stats = self.get_component_stats('transcription')
        if transcription_stats and transcription_stats.get_recent_avg() > 3.0:
            optimized_config['WHISPER_DEVICE'] = 'mps'  # Try GPU acceleration
            optimized_config['AUDIO_BUFFER_DURATION'] = 3.0  # Smaller buffers
        
        translation_stats = self.get_component_stats('translation')
        if translation_stats and translation_stats.get_recent_avg() > 2.0:
            optimized_config['TRANSLATION_BATCH_SIZE'] = 20
            optimized_config['TRANSLATION_CACHE_ENABLED'] = 'true'
        
        return optimized_config
    
    def cleanup(self):
        """Clean up profiler resources"""
        self.stop_monitoring()
        self.metrics.clear()
        self.component_stats.clear()


# Global profiler instance
profiler = PerformanceProfiler()


def measure_performance(component_name: str):
    """Decorator for measuring function performance"""
    return profiler.measure_component(component_name)


def get_performance_summary():
    """Get current performance summary"""
    return profiler.get_performance_summary()


def get_optimization_recommendations():
    """Get optimization recommendations"""
    return profiler.get_optimization_recommendations()


def export_performance_report(filepath: str):
    """Export performance report"""
    return profiler.export_performance_report(filepath)