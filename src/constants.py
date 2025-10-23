"""
Constants for the meeting transcription and translation system.
"""

# Audio processing constants
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT = "int16"

# Whisper model constants
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
WHISPER_SEGMENT_DURATION = 5.0  # seconds
WHISPER_MAX_PROCESSING_TIME = 10.0  # seconds

# Translation constants
MAX_TRANSLATION_LENGTH = 5000  # characters
TRANSLATION_TIMEOUT = 5.0  # seconds
TRANSLATION_RETRY_ATTEMPTS = 3

# UI constants
UI_REFRESH_INTERVAL = 0.5  # seconds
MAX_SUBTITLE_LINES = 50
SUBTITLE_DISPLAY_DURATION = 30.0  # seconds

# Performance thresholds
MAX_LATENCY_WARNING = 10.0  # seconds
MAX_PROCESSING_TIME = 30.0  # seconds
MAX_MEMORY_USAGE_MB = 1024

# File paths
DEFAULT_LOG_PATH = "logs/transcription.log"
DEFAULT_SESSION_PATH = "data/sessions"
DEFAULT_CACHE_PATH = "data/cache"

# Error messages
ERROR_MESSAGES = {
    "audio_device_not_found": "Аудіо пристрій не знайдено",
    "whisper_model_load_failed": "Не вдалося завантажити модель Whisper",
    "translation_api_error": "Помилка API перекладу",
    "network_connection_error": "Помилка мережевого з'єднання",
    "insufficient_permissions": "Недостатньо дозволів для доступу до аудіо",
    "processing_timeout": "Перевищено час обробки",
}

# Status messages
STATUS_MESSAGES = {
    "initializing": "Ініціалізація...",
    "ready": "Готово",
    "capturing_audio": "Захоплення аудіо...",
    "transcribing": "Транскрибування...",
    "translating": "Переклад...",
    "error": "Помилка",
    "stopped": "Зупинено",
}

# API endpoints and configuration
GOOGLE_TRANSLATE_API_URL = "https://translation.googleapis.com/language/translate/v2"
GOOGLE_TRANSLATE_DETECT_URL = "https://translation.googleapis.com/language/translate/v2/detect"

# Logging configuration
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "1 week"