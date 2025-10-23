"""
Configuration module for the meeting transcription and translation system.
Handles environment variables, API keys, and system settings.
"""

import os
from typing import Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class AudioConfig:
    """Audio capture configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    buffer_duration: float = 5.0  # seconds
    format: str = "int16"


@dataclass
class WhisperConfig:
    """Whisper transcription configuration"""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # Auto-detect if None
    device: str = "auto"  # auto, cpu, cuda, mps
    compute_type: str = "default"


@dataclass
class TranslationConfig:
    """Translation service configuration"""
    api_key: Optional[str] = None
    project_id: Optional[str] = None
    target_language: str = "uk"  # Ukrainian by default
    cache_enabled: bool = True
    batch_size: int = 10


@dataclass
class UIConfig:
    """Gradio UI configuration"""
    port: int = 7860
    host: str = "127.0.0.1"
    share: bool = False
    debug: bool = False
    auto_refresh_interval: float = 0.5  # seconds


@dataclass
class SystemConfig:
    """Main system configuration"""
    audio: AudioConfig
    whisper: WhisperConfig
    translation: TranslationConfig
    ui: UIConfig
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/transcription.log"
    
    # Performance
    max_concurrent_tasks: int = 3
    processing_timeout: float = 30.0  # seconds
    
    # Data storage
    session_history_path: str = "data/sessions"
    cache_path: str = "data/cache"


def load_config() -> SystemConfig:
    """Load configuration from environment variables and defaults"""
    
    # Audio configuration
    audio_config = AudioConfig(
        sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
        channels=int(os.getenv("AUDIO_CHANNELS", "1")),
        chunk_size=int(os.getenv("AUDIO_CHUNK_SIZE", "1024")),
        buffer_duration=float(os.getenv("AUDIO_BUFFER_DURATION", "5.0")),
    )
    
    # Whisper configuration
    whisper_config = WhisperConfig(
        model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
        language=os.getenv("WHISPER_LANGUAGE"),
        device=os.getenv("WHISPER_DEVICE", "auto"),
    )
    
    # Translation configuration
    translation_config = TranslationConfig(
        api_key=os.getenv("GOOGLE_TRANSLATE_API_KEY"),
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
        target_language=os.getenv("TARGET_LANGUAGE", "uk"),
        cache_enabled=os.getenv("TRANSLATION_CACHE_ENABLED", "true").lower() == "true",
    )
    
    # UI configuration
    ui_config = UIConfig(
        port=int(os.getenv("UI_PORT", "7860")),
        host=os.getenv("UI_HOST", "127.0.0.1"),
        share=os.getenv("UI_SHARE", "false").lower() == "true",
        debug=os.getenv("UI_DEBUG", "false").lower() == "true",
    )
    
    return SystemConfig(
        audio=audio_config,
        whisper=whisper_config,
        translation=translation_config,
        ui=ui_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/transcription.log"),
    )


# Global configuration instance
config = load_config()


# Supported languages for translation
SUPPORTED_LANGUAGES = {
    "uk": "Українська",
    "en": "English", 
    "ru": "Русский",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "pl": "Polski",
    "cs": "Čeština",
    "sk": "Slovenčina",
    "bg": "Български",
    "hr": "Hrvatski",
    "sl": "Slovenščina",
    "et": "Eesti",
    "lv": "Latviešu",
    "lt": "Lietuvių",
    "hu": "Magyar",
    "ro": "Română",
    "pt": "Português",
    "nl": "Nederlands",
    "da": "Dansk",
    "sv": "Svenska",
    "no": "Norsk",
    "fi": "Suomi",
}

# Audio source types
AUDIO_SOURCE_TYPES = {
    "microphone": "Мікрофон",
    "system_audio": "Системне аудіо",
}