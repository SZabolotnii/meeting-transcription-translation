# Meeting Transcription and Translation System

Система транскрибування та автоматичного перекладу мітингів в режимі реального часу для macOS.

## Особливості

- 🎤 Захоплення аудіо з мікрофону або системного аудіо (Zoom, Teams)
- 🗣️ Транскрибування в реальному часі за допомогою OpenAI Whisper
- 🌍 Автоматичний переклад на 20+ мов через Google Translate API
- 💻 Зручний веб-інтерфейс на базі Gradio
- ⚡ Мінімальна затримка (менше 5 секунд)
- 📝 Збереження історії субтитрів

## Системні вимоги

- macOS 10.15+
- Python 3.8+
- 4GB RAM (рекомендовано 8GB)
- 2GB вільного місця на диску

## Встановлення

1. Клонуйте репозиторій:
```bash
git clone <repository-url>
cd meeting-transcription-translation
```

2. Створіть віртуальне середовище:
```bash
python -m venv venv
source venv/bin/activate
```

3. Встановіть залежності:
```bash
pip install -r requirements.txt
```

4. Налаштуйте змінні середовища:
```bash
cp .env.example .env
# Відредагуйте .env файл та додайте ваші API ключі
```

5. (Опціонально) Встановіть BlackHole для захоплення системного аудіо:
   - Завантажте з https://existential.audio/blackhole/
   - Встановіть та налаштуйте згідно з інструкціями

## Конфігурація

### Google Cloud Translation API

1. Створіть проект в Google Cloud Console
2. Увімкніть Translation API
3. Створіть API ключ або service account
4. Додайте ключ до .env файлу

### Аудіо налаштування

Для захоплення системного аудіо (Zoom, Teams) потрібно:
1. Встановити BlackHole
2. Налаштувати Audio MIDI Setup
3. Створити Multi-Output Device

## Запуск

```bash
python main.py
```

Система буде доступна за адресою http://localhost:7860

## Структура проекту

```
├── src/
│   ├── audio_capture/     # Модуль захоплення аудіо
│   ├── transcription/     # Модуль Whisper транскрибування
│   ├── translation/       # Модуль перекладу
│   ├── orchestrator/      # Координатор системи
│   ├── ui/               # Gradio інтерфейс
│   ├── utils/            # Допоміжні функції
│   ├── config.py         # Конфігурація системи
│   └── constants.py      # Константи
├── data/                 # Дані та кеш
├── logs/                 # Логи системи
├── main.py              # Точка входу
├── requirements.txt     # Залежності
└── README.md           # Документація
```

## Розробка

Для розробки встановіть додаткові залежності:
```bash
pip install pytest pytest-asyncio black flake8
```

Запуск тестів:
```bash
pytest
```

Форматування коду:
```bash
black src/
```

## Ліцензія

MIT License