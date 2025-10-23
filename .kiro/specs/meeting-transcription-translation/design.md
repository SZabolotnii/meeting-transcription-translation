# Дизайн Системи

## Огляд

Система транскрибування та автоматичного перекладу мітингів в режимі реального часу складається з кількох взаємопов'язаних компонентів, що забезпечують захоплення аудіо, транскрибування, переклад та відображення живих субтитрів з мінімальною затримкою.

## Архітектура

### Високорівнева архітектура

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Аудіо Джерело │───▶│  Аудіо Захоплення │───▶│ Аудіо Буфер     │
│ (Мікрофон/Zoom) │    │     Модуль       │    │ (3-5 сек)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Gradio UI       │◀───│   Контролер      │◀───│ Whisper Модуль  │
│ (Живі Субтитри) │    │   Оркестратор    │    │ (Транскрибування)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲                       │
         │                       │                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         └──────────────│ Translate Модуль │◀───│ Черга Обробки   │
                        │ (Google API)    │    │ (Async Queue)   │
                        └─────────────────┘    └─────────────────┘
```

### Потік Даних

1. **Захоплення Аудіо** → Безперервний потік аудіо даних
2. **Буферизація** → Сегментація на 3-5 секундні блоки
3. **Транскрибування** → Whisper обробляє кожен сегмент
4. **Переклад** → Google Translate API перекладає текст
5. **Відображення** → Gradio UI оновлює живі субтитри

## Компоненти та Інтерфейси

### 1. Аудіо Захоплення Модуль (`audio_capture.py`)

**Призначення**: Захоплення аудіо з різних джерел на macOS

**Ключові функції**:
```python
class AudioCapture:
    def __init__(self, source_type: str, device_id: Optional[int] = None)
    def start_capture(self) -> None
    def stop_capture(self) -> None
    def get_audio_devices(self) -> List[Dict]
    def set_audio_source(self, source_type: str, device_id: int) -> None
```

**Технічна реалізація**:
- **Мікрофон**: PyAudio або sounddevice для захоплення з мікрофону
- **Системне аудіо**: BlackHole (віртуальний аудіо драйвер) + PyAudio для захоплення системного звуку
- **Формат аудіо**: 16kHz, 16-bit, mono (оптимально для Whisper)
- **Буферизація**: Кільцевий буфер для безперервного захоплення

### 2. Whisper Транскрибування Модуль (`whisper_transcriber.py`)

**Призначення**: Перетворення аудіо сегментів в текст

**Ключові функції**:
```python
class WhisperTranscriber:
    def __init__(self, model_size: str = "base")
    def transcribe_segment(self, audio_data: np.ndarray) -> str
    def set_language(self, language: str) -> None
    def is_processing(self) -> bool
```

**Технічна реалізація**:
- **Модель**: Whisper "base" для балансу швидкості/якості
- **Оптимізація**: Використання GPU якщо доступно (MPS на macOS)
- **Сегментація**: Обробка 3-5 секундних аудіо блоків
- **Кешування**: Завантаження моделі один раз при ініціалізації

### 3. Переклад Модуль (`translator.py`)

**Призначення**: Переклад транскрибованого тексту

**Ключові функції**:
```python
class Translator:
    def __init__(self, api_key: str)
    def translate_text(self, text: str, target_language: str) -> str
    def get_supported_languages(self) -> List[Dict]
    def set_target_language(self, language: str) -> None
```

**Технічна реалізація**:
- **API**: Google Cloud Translation API v2
- **Кешування**: Локальний кеш для повторюваних фраз
- **Обробка помилок**: Fallback до оригінального тексту при помилках
- **Батчинг**: Групування коротких сегментів для ефективності

### 4. Контролер Оркестратор (`orchestrator.py`)

**Призначення**: Координація всіх компонентів та управління потоком даних

**Ключові функції**:
```python
class TranscriptionOrchestrator:
    def __init__(self)
    def start_session(self, audio_source: str, target_language: str) -> None
    def stop_session(self) -> None
    def get_live_subtitles(self) -> Iterator[Dict]
    def get_session_history(self) -> List[Dict]
```

**Технічна реалізація**:
- **Асинхронність**: asyncio для паралельної обробки
- **Черги**: Queue для буферизації між компонентами
- **Стан сесії**: Управління життєвим циклом транскрибування
- **Метрики**: Відстеження затримки та продуктивності

### 5. Gradio Інтерфейс (`gradio_ui.py`)

**Призначення**: Веб-інтерфейс для користувача

**Ключові компоненти**:
```python
def create_interface() -> gr.Interface:
    # Елементи управління
    start_button = gr.Button("Старт")
    stop_button = gr.Button("Стоп")
    language_dropdown = gr.Dropdown(choices=languages)
    audio_source_radio = gr.Radio(choices=["Мікрофон", "Системне аудіо"])
    
    # Відображення результатів
    live_subtitles = gr.Textbox(label="Живі Субтитри", lines=10)
    status_indicator = gr.HTML()
    
    return gr.Interface(...)
```

**Технічна реалізація**:
- **Живе оновлення**: gr.Interface з автоматичним оновленням кожні 0.5 секунд
- **Стан**: Збереження налаштувань сесії в gr.State
- **Експорт**: Можливість завантаження історії субтитрів
- **Адаптивність**: Responsive дизайн для різних екранів

## Моделі Даних

### AudioSegment
```python
@dataclass
class AudioSegment:
    data: np.ndarray
    timestamp: float
    duration: float
    sample_rate: int
    segment_id: str
```

### TranscriptionResult
```python
@dataclass
class TranscriptionResult:
    original_text: str
    translated_text: Optional[str]
    timestamp: float
    confidence: float
    language: str
    target_language: Optional[str]
    processing_time: float
```

### SessionConfig
```python
@dataclass
class SessionConfig:
    audio_source_type: str  # "microphone" | "system_audio"
    audio_device_id: Optional[int]
    target_language: str
    whisper_model: str
    buffer_duration: float
```

## Обробка Помилок

### Стратегії Відновлення

1. **Аудіо Помилки**:
   - Автоматичне перепідключення до аудіо пристрою
   - Fallback до пристрою за замовчуванням
   - Повідомлення користувача про проблеми з аудіо

2. **Whisper Помилки**:
   - Пропуск пошкоджених сегментів
   - Retry механізм для тимчасових помилок
   - Логування проблемних сегментів

3. **Переклад Помилки**:
   - Відображення оригінального тексту
   - Кешування для offline режиму
   - Індикація статусу перекладу

4. **Мережеві Помилки**:
   - Exponential backoff для API викликів
   - Локальне збереження для синхронізації пізніше
   - Graceful degradation функціональності

### Логування та Моніторинг

```python
class PerformanceMonitor:
    def track_latency(self, component: str, duration: float) -> None
    def track_error(self, component: str, error: Exception) -> None
    def get_metrics(self) -> Dict[str, Any]
    def export_logs(self) -> str
```

## Стратегія Тестування

### Модульні Тести
- **AudioCapture**: Тестування захоплення з mock аудіо пристроїв
- **WhisperTranscriber**: Тестування з заздалегідь записаними аудіо файлами
- **Translator**: Тестування з mock Google API відповідями
- **Orchestrator**: Тестування координації компонентів

### Інтеграційні Тести
- **End-to-End**: Повний цикл від аудіо до субтитрів
- **Performance**: Вимірювання затримки та пропускної здатності
- **Error Handling**: Тестування сценаріїв помилок

### Тестування Продуктивності
- **Latency Benchmarks**: Вимірювання затримки кожного компонента
- **Memory Usage**: Моніторинг використання пам'яті при тривалих сесіях
- **CPU Usage**: Оптимізація навантаження на процесор

## Вимоги до Розгортання

### Системні Вимоги
- **OS**: macOS 10.15+ (для підтримки системного аудіо)
- **Python**: 3.8+
- **RAM**: Мінімум 4GB (рекомендовано 8GB)
- **Дисковий простір**: 2GB для Whisper моделей

### Залежності
```
openai-whisper>=20230314
gradio>=3.40.0
google-cloud-translate>=3.11.0
pyaudio>=0.2.11
sounddevice>=0.4.6
numpy>=1.21.0
asyncio
queue
```

### Конфігурація
- **API ключі**: Google Cloud Translation API
- **Аудіо драйвери**: BlackHole для системного аудіо на macOS
- **Дозволи**: Мікрофон та системне аудіо в macOS Security & Privacy

## Безпека та Приватність

### Обробка Даних
- **Локальна обробка**: Whisper працює локально без передачі аудіо в хмару
- **API безпека**: Шифрування Google Translate API викликів
- **Тимчасові дані**: Автоматичне очищення аудіо буферів

### Дозволи
- **Мікрофон**: Запит дозволу через macOS Security framework
- **Системне аудіо**: Інструкції для налаштування BlackHole
- **Мережа**: Тільки для Google Translate API викликів