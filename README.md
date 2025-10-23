# Meeting Transcription and Translation System

🎤 Система транскрибування та автоматичного перекладу мітингів в режимі реального часу для macOS.

## ✨ Особливості

- 🎤 **Захоплення аудіо** з мікрофону або системного аудіо (Zoom, Teams, Meet)
- 🗣️ **Транскрибування в реальному часі** за допомогою OpenAI Whisper
- 🌍 **Автоматичний переклад** на 20+ мов через Google Translate API
- 💻 **Зручний веб-інтерфейс** на базі Gradio з живими субтитрами
- ⚡ **Мінімальна затримка** (менше 5 секунд від мовлення до субтитрів)
- 📝 **Збереження історії** субтитрів у різних форматах (TXT, JSON, SRT)
- 🔧 **Гнучкі налаштування** продуктивності та якості
- 📊 **Моніторинг продуктивності** в реальному часі

## 🖥️ Системні вимоги

- **macOS**: 10.15+ (Catalina або новіша)
- **Python**: 3.8+ (рекомендовано 3.9+)
- **RAM**: 4GB мінімум (рекомендовано 8GB для кращої продуктивності)
- **Дисковий простір**: 2GB для Whisper моделей та кешу
- **Інтернет**: Стабільне з'єднання для Google Translate API

## 🚀 Швидкий старт

### 1. Встановлення

```bash
# Клонуйте репозиторій
git clone https://github.com/SZabolotnii/meeting-transcription-translation.git
cd meeting-transcription-translation

# Створіть віртуальне середовище
python -m venv venv
source venv/bin/activate  # На macOS/Linux

# Встановіть залежності
pip install -r requirements.txt
```

### 2. Конфігурація

```bash
# Скопіюйте файл конфігурації
cp .env.example .env

# Відредагуйте .env файл та додайте ваші API ключі
nano .env  # або використайте будь-який текстовий редактор
```

### 3. Перевірка системи

```bash
# Запустіть перевірку конфігурації
python src/utils/config_validator.py --full
```

### 4. Запуск

```bash
python main.py
```

Система буде доступна за адресою: **http://localhost:7860**

## ⚙️ Детальне налаштування

### Google Cloud Translation API

1. **Створіть проект в Google Cloud Console**:
   - Перейдіть на https://console.cloud.google.com/
   - Створіть новий проект або оберіть існуючий

2. **Увімкніть Translation API**:
   - Перейдіть в "APIs & Services" > "Library"
   - Знайдіть "Cloud Translation API" та увімкніть його

3. **Створіть API ключ**:
   - Перейдіть в "APIs & Services" > "Credentials"
   - Натисніть "Create Credentials" > "API Key"
   - Скопіюйте ключ

4. **Додайте ключ до .env файлу**:
   ```env
   GOOGLE_TRANSLATE_API_KEY=your_api_key_here
   GOOGLE_CLOUD_PROJECT_ID=your_project_id_here
   ```

### Налаштування аудіо для macOS

#### Для захоплення мікрофону
Система автоматично використовує системний мікрофон. Переконайтеся, що:
- Дозволено доступ до мікрофону в System Preferences > Security & Privacy > Microphone

#### Для захоплення системного аудіо (Zoom, Teams, Meet)

**Крок 1: Встановіть BlackHole**
```bash
# Завантажте BlackHole
curl -L -o BlackHole2ch.pkg https://existential.audio/blackhole/BlackHole2ch.pkg

# Встановіть (потребує пароль адміністратора)
sudo installer -pkg BlackHole2ch.pkg -target /
```

**Крок 2: Налаштуйте Audio MIDI Setup**
1. Відкрийте **Audio MIDI Setup** (Applications > Utilities)
2. Натисніть "+" та оберіть "Create Multi-Output Device"
3. Назвіть його "Meeting Audio"
4. Оберіть ваші динаміки/навушники та BlackHole 2ch
5. Встановіть ваші динаміки як "Master Device"

**Крок 3: Налаштуйте систему**
1. System Preferences > Sound > Output > оберіть "Meeting Audio"
2. У додатку для мітингів (Zoom/Teams) оберіть BlackHole як мікрофон
3. У нашій системі оберіть "Системне аудіо" як джерело

## 📁 Структура проекту

```
meeting-transcription-translation/
├── 📁 src/                          # Вихідний код
│   ├── 📁 audio_capture/            # Модуль захоплення аудіо
│   │   ├── audio_manager.py         # Менеджер аудіо пристроїв
│   │   ├── microphone_capture.py    # Захоплення з мікрофону
│   │   └── system_audio_capture.py  # Захоплення системного аудіо
│   ├── 📁 transcription/            # Модуль транскрибування
│   │   └── whisper_transcriber.py   # Whisper транскрибер
│   ├── 📁 translation/              # Модуль перекладу
│   │   └── translator.py            # Google Translate інтеграція
│   ├── 📁 orchestrator/             # Координатор системи
│   │   └── transcription_orchestrator.py
│   ├── 📁 ui/                       # Веб-інтерфейс
│   │   └── gradio_interface.py      # Gradio UI компоненти
│   ├── 📁 utils/                    # Допоміжні утиліти
│   │   └── config_validator.py      # Валідатор конфігурації
│   ├── config.py                    # Конфігурація системи
│   └── constants.py                 # Системні константи
├── 📁 data/                         # Дані та кеш
│   ├── 📁 sessions/                 # Історія сесій
│   ├── 📁 cache/                    # Кеш перекладів
│   └── 📁 exports/                  # Експортовані файли
├── 📁 logs/                         # Системні логи
├── 📁 tests/                        # Тести
├── main.py                          # Точка входу
├── requirements.txt                 # Python залежності
├── .env.example                     # Приклад конфігурації
└── README.md                        # Ця документація
```

## 🔧 Конфігурація

### Основні параметри (.env файл)

```env
# Аудіо налаштування
AUDIO_SAMPLE_RATE=16000              # Частота дискретизації (Hz)
AUDIO_BUFFER_DURATION=5.0            # Тривалість буфера (секунди)

# Whisper налаштування
WHISPER_MODEL_SIZE=base              # tiny, base, small, medium, large
WHISPER_DEVICE=auto                  # auto, cpu, cuda, mps

# Переклад
TARGET_LANGUAGE=uk                   # Цільова мова за замовчуванням
TRANSLATION_CACHE_ENABLED=true       # Увімкнути кеш перекладів

# UI налаштування
UI_PORT=7860                         # Порт веб-інтерфейсу
UI_HOST=127.0.0.1                    # Хост (0.0.0.0 для доступу ззовні)

# Продуктивність
MAX_CONCURRENT_TASKS=3               # Максимум паралельних задач
PROCESSING_TIMEOUT=30.0              # Таймаут обробки (секунди)
```

### Підтримувані мови

| Код | Мова | Код | Мова |
|-----|------|-----|------|
| uk | Українська | en | English |
| ru | Русский | de | Deutsch |
| fr | Français | es | Español |
| it | Italiano | pl | Polski |
| cs | Čeština | sk | Slovenčina |
| bg | Български | hr | Hrvatski |
| pt | Português | nl | Nederlands |
| da | Dansk | sv | Svenska |
| no | Norsk | fi | Suomi |

## 🎯 Використання

### Веб-інтерфейс

1. **Налаштування**:
   - Оберіть джерело аудіо (мікрофон або системне аудіо)
   - Виберіть цільову мову для перекладу
   - Налаштуйте аудіо пристрій при необхідності

2. **Запуск сесії**:
   - Натисніть "▶️ Старт" для початку транскрибування
   - Говоріть або запустіть мітинг
   - Спостерігайте за живими субтитрами

3. **Експорт результатів**:
   - Натисніть "📥 Завантажити історію"
   - Оберіть формат (TXT, JSON, SRT)
   - Налаштуйте параметри експорту

### Командний рядок

```bash
# Перевірка конфігурації
python src/utils/config_validator.py --full

# Перевірка тільки залежностей
python src/utils/config_validator.py --dependencies

# Показати поточну конфігурацію
python src/utils/config_validator.py --summary
```

## 🔍 Діагностика проблем

### Проблеми з аудіо

**Мікрофон не працює**:
- Перевірте дозволи в System Preferences > Security & Privacy > Microphone
- Переконайтеся, що мікрофон не використовується іншим додатком

**Системне аудіо не захоплюється**:
- Перевірте, чи встановлено BlackHole
- Переконайтеся, що Multi-Output Device налаштовано правильно
- Перевірте, чи обрано правильне джерело в системі

### Проблеми з перекладом

**Переклад не працює**:
- Перевірте API ключ Google Translate
- Переконайтеся в наявності інтернет-з'єднання
- Перевірте квоти API в Google Cloud Console

### Проблеми з продуктивністю

**Висока затримка**:
- Зменшіть розмір Whisper моделі (tiny замість base)
- Збільшіть AUDIO_BUFFER_DURATION для кращої стабільності
- Закрийте інші ресурсомісткі додатки

**Високе використання CPU**:
- Використайте GPU якщо доступно (MPS на Apple Silicon)
- Зменшіть MAX_CONCURRENT_TASKS
- Оберіть меншу Whisper модель

## 🧪 Розробка

### Встановлення для розробки

```bash
# Клонуйте репозиторій
git clone https://github.com/SZabolotnii/meeting-transcription-translation.git
cd meeting-transcription-translation

# Встановіть в режимі розробки
pip install -e .
pip install -r requirements-dev.txt
```

### Запуск тестів

```bash
# Всі тести
pytest

# Тести з покриттям
pytest --cov=src

# Тести конкретного модуля
pytest tests/test_audio_capture.py
```

### Форматування коду

```bash
# Форматування
black src/ tests/

# Перевірка стилю
flake8 src/ tests/

# Сортування імпортів
isort src/ tests/
```

### Структура тестів

```
tests/
├── test_audio_capture.py           # Тести аудіо захоплення
├── test_transcription.py           # Тести транскрибування
├── test_translation.py             # Тести перекладу
├── test_orchestrator.py            # Тести оркестратора
├── test_ui.py                      # Тести інтерфейсу
└── conftest.py                     # Конфігурація pytest
```

## 📊 Моніторинг та логи

### Логи системи

Логи зберігаються в `logs/transcription.log` та включають:
- Інформацію про запуск/зупинку сесій
- Метрики продуктивності
- Помилки та попередження
- Статистику використання API

### Метрики продуктивності

Веб-інтерфейс показує:
- **Затримка транскрибування**: час обробки Whisper
- **Затримка перекладу**: час обробки Google Translate
- **Загальна затримка**: від аудіо до субтитрів
- **Кількість оброблених сегментів**
- **Рівень аудіо сигналу**

## 🤝 Внесок у проект

1. Fork репозиторій
2. Створіть feature branch (`git checkout -b feature/amazing-feature`)
3. Commit зміни (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Створіть Pull Request

## 📄 Ліцензія

Цей проект ліцензовано під MIT License - дивіться файл [LICENSE](LICENSE) для деталей.

## 🙏 Подяки

- [OpenAI Whisper](https://github.com/openai/whisper) за чудовий speech-to-text
- [Google Cloud Translation](https://cloud.google.com/translate) за API перекладу
- [Gradio](https://gradio.app/) за простий веб-інтерфейс
- [BlackHole](https://existential.audio/blackhole/) за віртуальний аудіо драйвер

## 📚 Документація

Детальна документація доступна в папці `docs/`:

- **[BlackHole Setup Guide](docs/blackhole-setup.md)** - Налаштування системного аудіо
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Вирішення проблем
- **[Documentation Index](docs/README.md)** - Повний список документації

## 📞 Підтримка

Якщо у вас виникли проблеми:

1. Перевірте [Troubleshooting Guide](docs/troubleshooting.md)
2. Запустіть діагностику: `python src/utils/config_validator.py --full`
3. Перевірте [Issues](https://github.com/SZabolotnii/meeting-transcription-translation/issues)
4. Створіть новий Issue з детальним описом проблеми

## 🔗 Корисні посилання

- [BlackHole Audio Driver](https://existential.audio/blackhole/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Google Cloud Translation](https://cloud.google.com/translate)
- [Gradio Documentation](https://gradio.app/docs/)

---

**Зроблено з ❤️ для української спільноти**