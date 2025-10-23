"""
Configuration validation utility.
Provides tools for validating and testing system configuration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, validate_environment, get_config_summary


def print_config_summary():
    """Print a formatted configuration summary"""
    summary = get_config_summary()
    
    print("=" * 60)
    print(f"Configuration Summary - {summary['app']['name']} v{summary['app']['version']}")
    print("=" * 60)
    
    print("\n📊 Audio Configuration:")
    print(f"  Sample Rate: {summary['audio']['sample_rate']} Hz")
    print(f"  Channels: {summary['audio']['channels']}")
    print(f"  Buffer Duration: {summary['audio']['buffer_duration']}s")
    
    print("\n🎯 Whisper Configuration:")
    print(f"  Model Size: {summary['whisper']['model_size']}")
    print(f"  Device: {summary['whisper']['device']}")
    print(f"  Language: {summary['whisper']['language']}")
    
    print("\n🌐 Translation Configuration:")
    print(f"  Target Language: {summary['translation']['target_language']}")
    print(f"  Cache Enabled: {summary['translation']['cache_enabled']}")
    print(f"  API Configured: {'✅' if summary['translation']['api_configured'] else '❌'}")
    
    print("\n🖥️  UI Configuration:")
    print(f"  Host: {summary['ui']['host']}")
    print(f"  Port: {summary['ui']['port']}")
    print(f"  Share: {summary['ui']['share']}")
    
    print("\n⚡ Performance Configuration:")
    print(f"  Max Concurrent Tasks: {summary['performance']['max_concurrent_tasks']}")
    print(f"  Processing Timeout: {summary['performance']['processing_timeout']}s")


def validate_config() -> bool:
    """Validate configuration and print results"""
    print("🔍 Validating Configuration...")
    
    is_valid, issues = validate_environment()
    
    if is_valid:
        print("✅ Configuration is valid!")
        return True
    else:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available"""
    dependencies = {
        "whisper": False,
        "gradio": False,
        "google-cloud-translate": False,
        "pyaudio": False,
        "numpy": False,
        "python-dotenv": False,
        "loguru": False
    }
    
    # Check each dependency
    try:
        import whisper
        dependencies["whisper"] = True
    except ImportError:
        pass
    
    try:
        import gradio
        dependencies["gradio"] = True
    except ImportError:
        pass
    
    try:
        from google.cloud import translate
        dependencies["google-cloud-translate"] = True
    except ImportError:
        pass
    
    try:
        import pyaudio
        dependencies["pyaudio"] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass
    
    try:
        from dotenv import load_dotenv
        dependencies["python-dotenv"] = True
    except ImportError:
        pass
    
    try:
        from loguru import logger
        dependencies["loguru"] = True
    except ImportError:
        pass
    
    return dependencies


def print_dependency_status():
    """Print dependency status"""
    print("\n📦 Dependency Status:")
    dependencies = check_dependencies()
    
    for dep, available in dependencies.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    missing = [dep for dep, available in dependencies.items() if not available]
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies are available!")


def check_api_keys():
    """Check API key configuration"""
    print("\n🔑 API Key Status:")
    
    # Google Translate API
    if config.translation.api_key:
        # Mask the key for security
        masked_key = config.translation.api_key[:8] + "..." + config.translation.api_key[-4:]
        print(f"  ✅ Google Translate API Key: {masked_key}")
    else:
        print("  ❌ Google Translate API Key: Not configured")
    
    if config.translation.project_id:
        print(f"  ✅ Google Cloud Project ID: {config.translation.project_id}")
    else:
        print("  ❌ Google Cloud Project ID: Not configured")


def check_directories():
    """Check directory structure and permissions"""
    print("\n📁 Directory Status:")
    
    directories = [
        config.storage.session_history_path,
        config.storage.cache_path,
        config.storage.logs_path,
        config.storage.temp_audio_path,
        config.storage.export_path,
    ]
    
    for directory in directories:
        path = Path(directory)
        try:
            path.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.is_dir():
                print(f"  ✅ {directory}")
            else:
                print(f"  ❌ {directory} (not accessible)")
        except PermissionError:
            print(f"  ❌ {directory} (permission denied)")
        except Exception as e:
            print(f"  ❌ {directory} (error: {e})")


def run_full_check():
    """Run a complete system check"""
    print("🚀 Running Full System Check")
    print("=" * 60)
    
    # Print configuration summary
    print_config_summary()
    
    # Check dependencies
    print_dependency_status()
    
    # Check API keys
    check_api_keys()
    
    # Check directories
    check_directories()
    
    # Validate configuration
    print()
    is_valid = validate_config()
    
    print("\n" + "=" * 60)
    if is_valid:
        print("🎉 System is ready to run!")
    else:
        print("⚠️  Please fix the issues above before running the system.")
    print("=" * 60)
    
    return is_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration validation utility")
    parser.add_argument("--summary", action="store_true", help="Show configuration summary")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--dependencies", action="store_true", help="Check dependencies")
    parser.add_argument("--full", action="store_true", help="Run full system check")
    
    args = parser.parse_args()
    
    if args.summary:
        print_config_summary()
    elif args.validate:
        validate_config()
    elif args.dependencies:
        print_dependency_status()
    elif args.full:
        run_full_check()
    else:
        # Default: run full check
        run_full_check()