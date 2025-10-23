"""
Performance optimization utility.
Analyzes system performance and provides optimization recommendations.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .performance_profiler import profiler, get_performance_summary, get_optimization_recommendations
from ..config import config, validate_environment


def analyze_system_performance() -> Dict[str, Any]:
    """Analyze current system performance"""
    print("🔍 Analyzing system performance...")
    
    # Get system information
    import psutil
    import platform
    
    system_info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
    }
    
    # Check for GPU availability
    gpu_info = check_gpu_availability()
    system_info.update(gpu_info)
    
    return system_info


def check_gpu_availability() -> Dict[str, Any]:
    """Check for GPU availability and capabilities"""
    gpu_info = {
        'gpu_available': False,
        'gpu_type': None,
        'gpu_memory_gb': 0,
        'mps_available': False,
        'cuda_available': False
    }
    
    try:
        import torch
        
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            gpu_info['mps_available'] = True
            gpu_info['gpu_available'] = True
            gpu_info['gpu_type'] = 'Apple Silicon (MPS)'
        
        # Check for CUDA
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_available'] = True
            gpu_info['gpu_type'] = 'NVIDIA CUDA'
            gpu_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    except ImportError:
        pass
    
    return gpu_info


def benchmark_whisper_models() -> Dict[str, Dict[str, float]]:
    """Benchmark different Whisper model sizes"""
    print("🎯 Benchmarking Whisper models...")
    
    models_to_test = ["tiny", "base", "small"]
    results = {}
    
    # Create test audio (5 seconds of sine wave)
    import numpy as np
    sample_rate = 16000
    duration = 5.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    for model_size in models_to_test:
        print(f"  Testing {model_size} model...")
        
        try:
            from ..transcription.whisper_transcriber import WhisperTranscriber
            import time
            
            # Initialize transcriber
            transcriber = WhisperTranscriber(model_size=model_size)
            
            # Warm-up run
            transcriber.transcribe_segment(test_audio[:sample_rate])  # 1 second
            
            # Benchmark runs
            times = []
            for i in range(3):
                start_time = time.time()
                result = transcriber.transcribe_segment(test_audio)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results[model_size] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'throughput_ratio': duration / avg_time  # Real-time factor
            }
            
            print(f"    Average: {avg_time:.2f}s, Throughput: {results[model_size]['throughput_ratio']:.1f}x")
            
            # Clean up
            transcriber.cleanup()
            
        except Exception as e:
            print(f"    Error testing {model_size}: {e}")
            results[model_size] = {'error': str(e)}
    
    return results


def generate_optimized_config(system_info: Dict[str, Any], 
                            benchmark_results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Generate optimized configuration based on system analysis"""
    print("⚙️ Generating optimized configuration...")
    
    optimized_config = {}
    
    # CPU-based optimizations
    cpu_count = system_info.get('cpu_count', 4)
    memory_gb = system_info.get('memory_total_gb', 8)
    
    # Whisper model selection based on performance and resources
    if memory_gb < 4:
        # Low memory system
        optimized_config['WHISPER_MODEL_SIZE'] = 'tiny'
        optimized_config['MAX_CONCURRENT_TASKS'] = '1'
        optimized_config['AUDIO_BUFFER_DURATION'] = '3.0'
    elif memory_gb < 8:
        # Medium memory system
        optimized_config['WHISPER_MODEL_SIZE'] = 'base'
        optimized_config['MAX_CONCURRENT_TASKS'] = '2'
    else:
        # High memory system - choose based on benchmark results
        best_model = 'base'  # Default
        
        if benchmark_results:
            # Find the best model that can process in real-time
            for model, results in benchmark_results.items():
                if 'throughput_ratio' in results and results['throughput_ratio'] > 1.0:
                    if model == 'small' and results['throughput_ratio'] > 0.8:
                        best_model = 'small'
                    elif model == 'base' and results['throughput_ratio'] > 1.2:
                        best_model = 'base'
        
        optimized_config['WHISPER_MODEL_SIZE'] = best_model
        optimized_config['MAX_CONCURRENT_TASKS'] = str(min(cpu_count // 2, 3))
    
    # GPU optimizations
    if system_info.get('mps_available'):
        optimized_config['WHISPER_DEVICE'] = 'mps'
    elif system_info.get('cuda_available'):
        optimized_config['WHISPER_DEVICE'] = 'cuda'
    else:
        optimized_config['WHISPER_DEVICE'] = 'cpu'
    
    # Memory optimizations
    if memory_gb < 8:
        optimized_config['MAX_HISTORY_ENTRIES'] = '500'
        optimized_config['HISTORY_CLEANUP_THRESHOLD'] = '250'
        optimized_config['AUDIO_QUEUE_SIZE'] = '5'
        optimized_config['TRANSCRIPTION_QUEUE_SIZE'] = '5'
        optimized_config['RESULT_QUEUE_SIZE'] = '25'
    
    # Audio buffer optimization based on CPU performance
    if cpu_count >= 8:
        optimized_config['AUDIO_BUFFER_DURATION'] = '3.0'  # Smaller buffers for faster response
    elif cpu_count >= 4:
        optimized_config['AUDIO_BUFFER_DURATION'] = '5.0'  # Default
    else:
        optimized_config['AUDIO_BUFFER_DURATION'] = '7.0'  # Larger buffers for slower CPUs
    
    # UI optimizations
    if memory_gb < 4:
        optimized_config['UI_REFRESH_INTERVAL'] = '1.0'  # Slower refresh
    else:
        optimized_config['UI_REFRESH_INTERVAL'] = '0.5'  # Default
    
    return optimized_config


def apply_optimized_config(optimized_config: Dict[str, str], backup: bool = True):
    """Apply optimized configuration to .env file"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found. Please create it from .env.example first.")
        return False
    
    # Backup current .env file
    if backup:
        backup_path = Path(f'.env.backup.{int(time.time())}')
        import shutil
        shutil.copy2(env_path, backup_path)
        print(f"📁 Backup created: {backup_path}")
    
    # Read current .env file
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update configuration
    updated_lines = []
    updated_keys = set()
    
    for line in lines:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key = line.split('=')[0]
            if key in optimized_config:
                updated_lines.append(f"{key}={optimized_config[key]}\n")
                updated_keys.add(key)
            else:
                updated_lines.append(line + '\n')
        else:
            updated_lines.append(line + '\n')
    
    # Add new configuration keys
    for key, value in optimized_config.items():
        if key not in updated_keys:
            updated_lines.append(f"{key}={value}\n")
    
    # Write updated .env file
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print("✅ Configuration updated successfully!")
    return True


def print_optimization_report(system_info: Dict[str, Any], 
                            benchmark_results: Dict[str, Dict[str, float]],
                            optimized_config: Dict[str, str]):
    """Print comprehensive optimization report"""
    print("\n" + "="*60)
    print("📊 PERFORMANCE OPTIMIZATION REPORT")
    print("="*60)
    
    # System Information
    print("\n🖥️  System Information:")
    print(f"  Platform: {system_info['platform']} {system_info['platform_version']}")
    print(f"  Architecture: {system_info['architecture']}")
    print(f"  CPU Cores: {system_info['cpu_count']} physical, {system_info['cpu_count_logical']} logical")
    print(f"  Memory: {system_info['memory_total_gb']:.1f} GB total, {system_info['memory_available_gb']:.1f} GB available")
    
    if system_info['gpu_available']:
        print(f"  GPU: {system_info['gpu_type']}")
        if system_info['gpu_memory_gb'] > 0:
            print(f"  GPU Memory: {system_info['gpu_memory_gb']:.1f} GB")
    else:
        print("  GPU: Not available")
    
    # Benchmark Results
    if benchmark_results:
        print("\n🎯 Whisper Model Benchmarks:")
        for model, results in benchmark_results.items():
            if 'error' in results:
                print(f"  {model}: Error - {results['error']}")
            else:
                throughput = results['throughput_ratio']
                status = "✅ Real-time" if throughput > 1.0 else "⚠️ Slower than real-time"
                print(f"  {model}: {results['avg_time']:.2f}s avg, {throughput:.1f}x throughput {status}")
    
    # Optimized Configuration
    print("\n⚙️ Recommended Configuration:")
    for key, value in optimized_config.items():
        current_value = getattr(config, key.lower(), 'Not set')
        if hasattr(config, key.lower().replace('_', '.')):
            # Handle nested config attributes
            parts = key.lower().split('_')
            current_obj = config
            for part in parts[:-1]:
                current_obj = getattr(current_obj, part, None)
                if current_obj is None:
                    break
            if current_obj:
                current_value = getattr(current_obj, parts[-1], 'Not set')
        
        change_indicator = "🔄" if str(current_value) != value else "✓"
        print(f"  {change_indicator} {key}={value} (current: {current_value})")
    
    # Performance Recommendations
    recommendations = []
    
    if system_info['memory_total_gb'] < 8:
        recommendations.append("💾 Consider upgrading RAM for better performance")
    
    if not system_info['gpu_available']:
        recommendations.append("🎮 Consider using a system with GPU acceleration (Apple Silicon or NVIDIA)")
    
    if system_info['cpu_count'] < 4:
        recommendations.append("⚡ Consider upgrading to a multi-core CPU for better parallel processing")
    
    if benchmark_results:
        slow_models = [model for model, results in benchmark_results.items() 
                      if 'throughput_ratio' in results and results['throughput_ratio'] < 1.0]
        if slow_models:
            recommendations.append(f"🐌 Models {', '.join(slow_models)} may be too slow for real-time use")
    
    if recommendations:
        print("\n💡 Additional Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
    
    print("\n" + "="*60)


def main():
    """Main optimization function"""
    parser = argparse.ArgumentParser(description="Performance optimization utility")
    parser.add_argument("--analyze", action="store_true", help="Analyze system performance")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark Whisper models")
    parser.add_argument("--optimize", action="store_true", help="Generate and apply optimized configuration")
    parser.add_argument("--apply", action="store_true", help="Apply optimized configuration to .env file")
    parser.add_argument("--no-backup", action="store_true", help="Don't backup .env file when applying changes")
    parser.add_argument("--export", type=str, help="Export optimization report to file")
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.benchmark, args.optimize]):
        # Default: run full optimization
        args.analyze = True
        args.benchmark = True
        args.optimize = True
    
    system_info = {}
    benchmark_results = {}
    optimized_config = {}
    
    if args.analyze:
        system_info = analyze_system_performance()
    
    if args.benchmark:
        benchmark_results = benchmark_whisper_models()
    
    if args.optimize:
        if not system_info:
            system_info = analyze_system_performance()
        if not benchmark_results:
            benchmark_results = benchmark_whisper_models()
        
        optimized_config = generate_optimized_config(system_info, benchmark_results)
        
        # Print report
        print_optimization_report(system_info, benchmark_results, optimized_config)
        
        if args.apply:
            apply_optimized_config(optimized_config, backup=not args.no_backup)
    
    # Export report if requested
    if args.export:
        report = {
            'timestamp': time.time(),
            'system_info': system_info,
            'benchmark_results': benchmark_results,
            'optimized_config': optimized_config
        }
        
        with open(args.export, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report exported to {args.export}")


if __name__ == "__main__":
    import time
    main()