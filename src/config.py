"""
Configuration module for the meeting transcription and translation system.
Handles environment variables, API keys, and system settings.
"""

import os
from typing import Optional, List, Dict, Tuple, Any
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
class PerformanceConfig:
    """Performance and optimization configuration"""
    max_concurrent_tasks: int = 3
    processing_timeout: float = 30.0  # seconds
    audio_queue_size: int = 10
    transcription_queue_size: int = 10
    result_queue_size: int = 50
    max_history_entries: int = 1000
    history_cleanup_threshold: int = 500


@dataclass
class StorageConfig:
    """Data storage configuration"""
    session_history_path: str = "data/sessions"
    cache_path: str = "data/cache"
    logs_path: str = "logs"
    temp_audio_path: str = "data/temp"
    export_path: str = "data/exports"


@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    enable_api_key_validation: bool = True
    max_session_duration: int = 14400  # 4 hours in seconds
    auto_cleanup_temp_files: bool = True
    log_sensitive_data: bool = False


@dataclass
class SystemConfig:
    """Main system configuration"""
    audio: AudioConfig
    whisper: WhisperConfig
    translation: TranslationConfig
    ui: UIConfig
    performance: PerformanceConfig
    storage: StorageConfig
    security: SecurityConfig
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/transcription.log"
    
    # Application metadata
    app_name: str = "Meeting Transcription System"
    app_version: str = "1.0.0"
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate API key if translation is enabled
        if not self.translation.api_key and self.translation.target_language != "en":
            issues.append("Google Translate API key is required for translation")
        
        # Validate audio settings
        if self.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            issues.append(f"Unsupported audio sample rate: {self.audio.sample_rate}")
        
        if self.audio.buffer_duration < 1.0 or self.audio.buffer_duration > 30.0:
            issues.append(f"Audio buffer duration should be between 1-30 seconds: {self.audio.buffer_duration}")
        
        # Validate Whisper model
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if self.whisper.model_size not in valid_models:
            issues.append(f"Invalid Whisper model: {self.whisper.model_size}")
        
        # Validate UI settings
        if self.ui.port < 1024 or self.ui.port > 65535:
            issues.append(f"Invalid UI port: {self.ui.port}")
        
        return issues


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
        batch_size=int(os.getenv("TRANSLATION_BATCH_SIZE", "10")),
    )
    
    # UI configuration
    ui_config = UIConfig(
        port=int(os.getenv("UI_PORT", "7860")),
        host=os.getenv("UI_HOST", "127.0.0.1"),
        share=os.getenv("UI_SHARE", "false").lower() == "true",
        debug=os.getenv("UI_DEBUG", "false").lower() == "true",
        auto_refresh_interval=float(os.getenv("UI_REFRESH_INTERVAL", "0.5")),
    )
    
    # Performance configuration
    performance_config = PerformanceConfig(
        max_concurrent_tasks=int(os.getenv("MAX_CONCURRENT_TASKS", "3")),
        processing_timeout=float(os.getenv("PROCESSING_TIMEOUT", "30.0")),
        audio_queue_size=int(os.getenv("AUDIO_QUEUE_SIZE", "10")),
        transcription_queue_size=int(os.getenv("TRANSCRIPTION_QUEUE_SIZE", "10")),
        result_queue_size=int(os.getenv("RESULT_QUEUE_SIZE", "50")),
        max_history_entries=int(os.getenv("MAX_HISTORY_ENTRIES", "1000")),
        history_cleanup_threshold=int(os.getenv("HISTORY_CLEANUP_THRESHOLD", "500")),
    )
    
    # Storage configuration
    storage_config = StorageConfig(
        session_history_path=os.getenv("SESSION_HISTORY_PATH", "data/sessions"),
        cache_path=os.getenv("CACHE_PATH", "data/cache"),
        logs_path=os.getenv("LOGS_PATH", "logs"),
        temp_audio_path=os.getenv("TEMP_AUDIO_PATH", "data/temp"),
        export_path=os.getenv("EXPORT_PATH", "data/exports"),
    )
    
    # Security configuration
    security_config = SecurityConfig(
        enable_api_key_validation=os.getenv("ENABLE_API_KEY_VALIDATION", "true").lower() == "true",
        max_session_duration=int(os.getenv("MAX_SESSION_DURATION", "14400")),
        auto_cleanup_temp_files=os.getenv("AUTO_CLEANUP_TEMP_FILES", "true").lower() == "true",
        log_sensitive_data=os.getenv("LOG_SENSITIVE_DATA", "false").lower() == "true",
    )
    
    return SystemConfig(
        audio=audio_config,
        whisper=whisper_config,
        translation=translation_config,
        ui=ui_config,
        performance=performance_config,
        storage=storage_config,
        security=security_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/transcription.log"),
        app_name=os.getenv("APP_NAME", "Meeting Transcription System"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
    )


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration (without sensitive data)"""
    cfg = config
    return {
        "app": {
            "name": cfg.app_name,
            "version": cfg.app_version,
        },
        "audio": {
            "sample_rate": cfg.audio.sample_rate,
            "channels": cfg.audio.channels,
            "buffer_duration": cfg.audio.buffer_duration,
        },
        "whisper": {
            "model_size": cfg.whisper.model_size,
            "device": cfg.whisper.device,
            "language": cfg.whisper.language or "auto-detect",
        },
        "translation": {
            "target_language": cfg.translation.target_language,
            "cache_enabled": cfg.translation.cache_enabled,
            "api_configured": bool(cfg.translation.api_key),
        },
        "ui": {
            "host": cfg.ui.host,
            "port": cfg.ui.port,
            "share": cfg.ui.share,
        },
        "performance": {
            "max_concurrent_tasks": cfg.performance.max_concurrent_tasks,
            "processing_timeout": cfg.performance.processing_timeout,
        }
    }


def validate_environment() -> Tuple[bool, List[str]]:
    """Validate environment configuration"""
    issues = config.validate()
    
    # Check for required dependencies
    try:
        import whisper
    except ImportError:
        issues.append("OpenAI Whisper not installed (pip install openai-whisper)")
    
    try:
        import gradio
    except ImportError:
        issues.append("Gradio not installed (pip install gradio)")
    
    try:
        from google.cloud import translate
    except ImportError:
        issues.append("Google Cloud Translate not installed (pip install google-cloud-translate)")
    
    # Check directory permissions
    import os
    from pathlib import Path
    
    directories_to_check = [
        config.storage.session_history_path,
        config.storage.cache_path,
        config.storage.logs_path,
        config.storage.temp_audio_path,
        config.storage.export_path,
    ]
    
    for directory in directories_to_check:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            issues.append(f"Cannot create directory: {directory}")
    
    return len(issues) == 0, issues


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