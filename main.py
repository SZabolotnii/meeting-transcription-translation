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
    
    logger.info("Starting Meeting Transcription and Translation System")
    
    # Setup logging
    setup_logging()
    
    # Create necessary directories
    create_directories()
    
    logger.info("System initialized successfully")
    logger.info(f"Configuration loaded: Audio={config.audio.sample_rate}Hz, UI Port={config.ui.port}")
    
    # TODO: Initialize and start the system components
    # This will be implemented in subsequent tasks
    print("Meeting Transcription and Translation System")
    print("System initialized. Ready for component implementation.")


if __name__ == "__main__":
    main()