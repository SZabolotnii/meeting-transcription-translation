#!/usr/bin/env python3
"""
Main entry point for the Meeting Transcription and Translation System.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.config import config
    from loguru import logger
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Dependencies not installed: {e}")
    print("Please run: pip install -r requirements.txt")
    DEPENDENCIES_AVAILABLE = False


def setup_logging():
    """Configure logging for the application"""
    if not DEPENDENCIES_AVAILABLE:
        return
        
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level=config.log_level,
        colorize=True
    )
    
    # Add file logger if specified
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            config.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level=config.log_level,
            rotation="10 MB",
            retention="1 week"
        )


def create_directories():
    """Create necessary directories for the application"""
    directories = [
        "logs",
        "data/sessions", 
        "data/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        if DEPENDENCIES_AVAILABLE:
            logger.info(f"Created directory: {directory}")
        else:
            print(f"Created directory: {directory}")


def main():
    """Main application entry point"""
    if not DEPENDENCIES_AVAILABLE:
        print("Meeting Transcription and Translation System")
        print("Project structure created successfully!")
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy .env.example to .env and configure API keys")
        print("3. Run the system: python main.py")
        return
    
    logger.info(f"Starting {config.app_name} v{config.app_version}")
    
    # Setup logging
    setup_logging()
    
    # Validate environment and configuration
    from src.config import validate_environment, get_config_summary
    from src.utils.performance_profiler import profiler
    
    is_valid, issues = validate_environment()
    if not is_valid:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        print("Configuration issues detected. Please check the logs and fix the issues.")
        return
    
    # Create necessary directories
    create_directories()
    
    # Log configuration summary
    config_summary = get_config_summary()
    logger.info("System initialized successfully")
    logger.info(f"Audio: {config_summary['audio']['sample_rate']}Hz, {config_summary['audio']['channels']} channel(s)")
    logger.info(f"Whisper: {config_summary['whisper']['model_size']} model on {config_summary['whisper']['device']}")
    logger.info(f"Translation: {config_summary['translation']['target_language']} ({'enabled' if config_summary['translation']['api_configured'] else 'disabled'})")
    logger.info(f"UI: {config_summary['ui']['host']}:{config_summary['ui']['port']}")
    
    # Initialize the orchestrator
    try:
        from src.orchestrator import TranscriptionOrchestrator
        from src.ui import create_interface
        
        logger.info("Initializing transcription orchestrator...")
        orchestrator = TranscriptionOrchestrator()
        
        logger.info("Creating Gradio interface...")
        interface = create_interface(orchestrator=orchestrator)
        
        # Connect orchestrator to UI callbacks
        def on_subtitle_update(result):
            """Handle new subtitle from orchestrator"""
            # This will be handled by the UI's live update mechanism
            pass
            
        def on_status_update(status, data):
            """Handle status updates from orchestrator"""
            logger.info(f"Status update: {status} - {data}")
            
        def on_error(error):
            """Handle errors from orchestrator"""
            logger.error(f"Orchestrator error: {error}")
        
        orchestrator.set_callbacks(
            subtitle_callback=on_subtitle_update,
            status_callback=on_status_update,
            error_callback=on_error
        )
        
        logger.info("Launching Gradio interface...")
        print("Starting web interface...")
        print(f"Open your browser and go to: http://{config.ui.host}:{config.ui.port}")
        
        # Launch the interface
        interface.launch()
        
    except Exception as e:
        logger.error(f"Failed to launch interface: {e}")
        print(f"Error launching interface: {e}")
        return
    finally:
        # Clean up performance profiler
        profiler.cleanup()


if __name__ == "__main__":
    main()