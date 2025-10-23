# Performance Optimization Guide

Посібник з оптимізації продуктивності системи транскрибування та перекладу.

## 🚀 Швидкий старт оптимізації

### Автоматична оптимізація

```bash
# Запустіть автоматичну оптимізацію
python src/utils/performance_optimizer.py --optimize --apply

# Або тільки аналіз без застосування змін
python src/utils/performance_optimizer.py --analyze --benchmark
```

### Ручна перевірка продуктивності

```bash
# Перевірка поточної продуктивності
python src/utils/config_validator.py --full

# Детальний аналіз системи
python src/utils/performance_optimizer.py --analyze
```

## 📊 Розуміння метрик продуктивності

### Ключові показники

| Метрика | Оптимальне значення | Опис |
|---------|-------------------|------|
| **Затримка транскрибування** | < 3 секунди | Час обробки Whisper |
| **Затримка перекладу** | < 2 секунди | Час обробки Google Translate |
| **Загальна затримка** | < 5 секунд | Від аудіо до субтитрів |
| **Використання CPU** | < 80% | Навантаження на процесор |
| **Використання RAM** | < 85% | Використання пам'яті |
| **Пропускна здатність** | > 1.0x | Реальний час обробки |

### Моніторинг в реальному часі

Веб-інтерфейс показує:
- 📊 **Метрики продуктивності** - поточні показники
- 🔄 **Статус компонентів** - стан обробки
- 📈 **Рівень аудіо** - якість сигналу
- ⚠️ **Рекомендації** - пропозиції з оптимізації

## ⚙️ Оптимізація конфігурації

### Whisper модель

**Вибір моделі залежно від ресурсів:**

```env
# Для слабких систем (< 4GB RAM)
WHISPER_MODEL_SIZE=tiny

# Для середніх систем (4-8GB RAM)
WHISPER_MODEL_SIZE=base

# Для потужних систем (> 8GB RAM)
WHISPER_MODEL_SIZE=small
```

**Порівняння моделей:**

| Модель | Розмір | Швидкість | Якість | RAM |
|--------|--------|-----------|--------|-----|
| tiny | 39 MB | Найшвидша | Базова | 1GB |
| base | 74 MB | Швидка | Хороша | 2GB |
| small | 244 MB | Середня | Відмінна | 4GB |
| medium | 769 MB | Повільна | Дуже хороша | 8GB |

### Пристрій обробки

```env
# Для Apple Silicon Mac (M1/M2/M3)
WHISPER_DEVICE=mps

# Для NVIDIA GPU
WHISPER_DEVICE=cuda

# Для CPU (за замовчуванням)
WHISPER_DEVICE=cpu
```

### Аудіо налаштування

```env
# Для швидких систем - менші буфери
AUDIO_BUFFER_DURATION=3.0

# Для повільних систем - більші буфери
AUDIO_BUFFER_DURATION=7.0

# Якість аудіо (вища частота = краща якість, але більше навантаження)
AUDIO_SAMPLE_RATE=16000  # Стандарт
AUDIO_SAMPLE_RATE=48000  # Висока якість
```

### Паралельна обробка

```env
# Кількість паралельних задач (залежить від CPU)
MAX_CONCURRENT_TASKS=2   # Для 4-ядерних CPU
MAX_CONCURRENT_TASKS=3   # Для 8-ядерних CPU

# Розміри черг (менші значення = менше пам'яті)
AUDIO_QUEUE_SIZE=5
TRANSCRIPTION_QUEUE_SIZE=5
RESULT_QUEUE_SIZE=25
```

## 🔧 Системні оптимізації

### macOS налаштування

**Енергозбереження:**
```bash
# Вимкніть App Nap для кращої продуктивності
sudo defaults write NSGlobalDomain NSAppSleepDisabled -bool YES

# Встановіть високу продуктивність
sudo pmset -a powernap 0
sudo pmset -a standby 0
```

**Аудіо налаштування:**
```bash
# Збільшіть буфер аудіо системи
sudo sysctl -w kern.audio.buffer_size=4096
```

### Оптимізація пам'яті

**Налаштування Python:**
```bash
# Запуск з оптимізацією пам'яті
export PYTHONOPTIMIZE=1
export MALLOC_ARENA_MAX=2
python main.py
```

**Обмеження історії:**
```env
# Зменшіть розмір історії для економії пам'яті
MAX_HISTORY_ENTRIES=500
HISTORY_CLEANUP_THRESHOLD=250
```

## 📈 Профілювання продуктивності

### Вбудований профайлер

Система автоматично відстежує:
- Час обробки кожного компонента
- Використання системних ресурсів
- Помилки та їх частоту
- Рекомендації з оптимізації

### Експорт звітів

```bash
# Експорт детального звіту
python -c "
from src.utils.performance_profiler import profiler
profiler.export_performance_report('performance_report.json')
"

# Експорт звіту оптимізації
python src/utils/performance_optimizer.py --export optimization_report.json
```

### Аналіз вузьких місць

**Повільна транскрибування:**
1. Зменшіть модель Whisper
2. Увімкніть GPU прискорення
3. Збільшіть розмір аудіо буфера

**Повільний переклад:**
1. Перевірте інтернет-з'єднання
2. Увімкніть кешування
3. Збільшіть розмір батча

**Високе використання CPU:**
1. Зменшіть кількість паралельних задач
2. Використайте меншу модель Whisper
3. Збільшіть інтервал оновлення UI

## 🎯 Рекомендації для різних сценаріїв

### Демонстрації та презентації

```env
# Максимальна швидкість, прийнятна якість
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=mps
AUDIO_BUFFER_DURATION=2.0
MAX_CONCURRENT_TASKS=1
UI_REFRESH_INTERVAL=0.3
```

### Професійні мітинги

```env
# Баланс швидкості та якості
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=mps
AUDIO_BUFFER_DURATION=5.0
MAX_CONCURRENT_TASKS=2
TRANSLATION_CACHE_ENABLED=true
```

### Архівування та документування

```env
# Максимальна якість, швидкість не критична
WHISPER_MODEL_SIZE=small
AUDIO_BUFFER_DURATION=10.0
TRANSLATION_BATCH_SIZE=20
MAX_HISTORY_ENTRIES=2000
```

### Слабкі системи

```env
# Мінімальні ресурси
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
AUDIO_BUFFER_DURATION=7.0
MAX_CONCURRENT_TASKS=1
MAX_HISTORY_ENTRIES=200
AUDIO_QUEUE_SIZE=3
UI_REFRESH_INTERVAL=1.0
```

## 🔍 Діагностика проблем продуктивності

### Високе використання CPU

**Симптоми:**
- Вентилятор працює на повну
- Система гальмує
- Затримка > 10 секунд

**Рішення:**
```env
WHISPER_MODEL_SIZE=tiny
MAX_CONCURRENT_TASKS=1
AUDIO_BUFFER_DURATION=7.0
```

### Високе використання пам'яті

**Симптоми:**
- Система використовує > 90% RAM
- Повідомлення про нестачу пам'яті
- Система swap активна

**Рішення:**
```env
MAX_HISTORY_ENTRIES=200
AUDIO_QUEUE_SIZE=3
TRANSCRIPTION_QUEUE_SIZE=3
RESULT_QUEUE_SIZE=10
```

### Повільна мережа

**Симптоми:**
- Переклад займає > 5 секунд
- Помилки мережі в логах
- Субтитри з'являються рывками

**Рішення:**
```env
TRANSLATION_CACHE_ENABLED=true
TRANSLATION_BATCH_SIZE=5
PROCESSING_TIMEOUT=60.0
```

## 📊 Бенчмарки

### Типові показники

**MacBook Air M2 (8GB RAM):**
- tiny: 0.3s (16.7x швидкість)
- base: 0.8s (6.3x швидкість)
- small: 2.1s (2.4x швидкість)

**MacBook Pro M3 (16GB RAM):**
- tiny: 0.2s (25x швидкість)
- base: 0.5s (10x швидкість)
- small: 1.2s (4.2x швидкість)

**Intel Mac (16GB RAM):**
- tiny: 1.2s (4.2x швидкість)
- base: 3.1s (1.6x швидкість)
- small: 8.2s (0.6x швидкість)

### Запуск бенчмарків

```bash
# Тест всіх моделей
python src/utils/performance_optimizer.py --benchmark

# Тест конкретної моделі
python -c "
from src.transcription.whisper_transcriber import WhisperTranscriber
import time, numpy as np

# Створіть тестове аудіо
audio = np.random.random(80000).astype(np.float32)  # 5 секунд

# Тест
transcriber = WhisperTranscriber('base')
start = time.time()
result = transcriber.transcribe_segment(audio)
print(f'Час: {time.time() - start:.2f}s')
"
```

## 🛠️ Інструменти моніторингу

### Системні утиліти

```bash
# Моніторинг CPU та пам'яті
top -o CPU

# Детальна інформація про процеси
htop  # Якщо встановлено

# Моніторинг мережі
nettop

# Температура CPU (потребує додаткових утиліт)
sudo powermetrics -n 1 -i 1000 | grep -i temp
```

### Вбудовані інструменти

```bash
# Аналіз продуктивності
python src/utils/performance_optimizer.py --analyze

# Поточні метрики
python -c "
from src.utils.performance_profiler import get_performance_summary
import json
print(json.dumps(get_performance_summary(), indent=2))
"
```

## 📝 Логування продуктивності

### Налаштування логів

```env
# Детальне логування продуктивності
LOG_LEVEL=DEBUG

# Окремий файл для метрик
PERFORMANCE_LOG_FILE=logs/performance.log
```

### Аналіз логів

```bash
# Пошук повільних операцій
grep "Slow transcription" logs/transcription.log

# Статистика помилок
grep "ERROR" logs/transcription.log | wc -l

# Середня затримка
grep "transcription_latency" logs/transcription.log | awk '{sum+=$NF; count++} END {print sum/count}'
```

---

**Примітка**: Оптимальні налаштування залежать від вашої системи та вимог. Використовуйте автоматичну оптимізацію як відправну точку та налаштовуйте параметри відповідно до ваших потреб.