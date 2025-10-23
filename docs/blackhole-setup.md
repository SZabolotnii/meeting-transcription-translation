# BlackHole Setup Guide for macOS

Детальна інструкція з налаштування BlackHole для захоплення системного аудіо на macOS.

## Що таке BlackHole?

BlackHole - це віртуальний аудіо драйвер для macOS, який дозволяє перенаправляти аудіо між додатками. Це необхідно для захоплення звуку з додатків для відеоконференцій (Zoom, Teams, Meet) без використання зовнішніх мікрофонів.

## Встановлення BlackHole

### Метод 1: Завантаження з офіційного сайту

1. Перейдіть на https://existential.audio/blackhole/
2. Завантажте BlackHole 2ch (рекомендовано для більшості випадків)
3. Відкрийте завантажений .pkg файл
4. Слідуйте інструкціям встановлення (потрібен пароль адміністратора)

### Метод 2: Через Homebrew

```bash
# Встановіть Homebrew якщо ще не встановлено
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Встановіть BlackHole
brew install blackhole-2ch
```

### Метод 3: Завантаження через командний рядок

```bash
# Завантажте BlackHole
curl -L -o BlackHole2ch.pkg https://existential.audio/blackhole/BlackHole2ch.pkg

# Встановіть (потребує пароль адміністратора)
sudo installer -pkg BlackHole2ch.pkg -target /

# Видаліть файл встановлення
rm BlackHole2ch.pkg
```

## Налаштування Multi-Output Device

Після встановлення BlackHole потрібно створити Multi-Output Device для одночасного відтворення звуку через динаміки та захоплення через BlackHole.

### Крок 1: Відкрийте Audio MIDI Setup

1. Натисніть `Cmd + Space` та введіть "Audio MIDI Setup"
2. Або перейдіть: Applications > Utilities > Audio MIDI Setup

### Крок 2: Створіть Multi-Output Device

1. У Audio MIDI Setup натисніть кнопку "+" (плюс) внизу лівої панелі
2. Оберіть "Create Multi-Output Device"
3. Новий пристрій з'явиться в списку

### Крок 3: Налаштуйте Multi-Output Device

1. Оберіть створений Multi-Output Device
2. У правій панелі поставте галочки біля:
   - Ваших основних динаміків/навушників (наприклад, "MacBook Pro Speakers")
   - "BlackHole 2ch"
3. Встановіть ваші основні динаміки як "Master Device" (галочка в колонці "Master")
4. Перейменуйте пристрій на щось зрозуміле, наприклад "Meeting Audio"

### Крок 4: Налаштуйте систему

1. Перейдіть в System Preferences > Sound
2. На вкладці "Output" оберіть ваш Multi-Output Device ("Meeting Audio")
3. Тепер звук буде відтворюватися через динаміки і одночасно доступний через BlackHole

## Налаштування додатків для мітингів

### Zoom

1. Відкрийте Zoom
2. Перейдіть в Settings (Zoom > Preferences)
3. На вкладці "Audio":
   - **Microphone**: оберіть "BlackHole 2ch"
   - **Speaker**: залиште ваші звичайні динаміки або оберіть Multi-Output Device

### Microsoft Teams

1. Відкрийте Teams
2. Натисніть на ваш профіль > Settings
3. У розділі "Devices":
   - **Microphone**: оберіть "BlackHole 2ch"
   - **Speaker**: залиште ваші звичайні динаміки

### Google Meet

1. У браузері перейдіть на meet.google.com
2. Перед входом в мітинг натисніть на іконку налаштувань (шестерня)
3. У розділі "Audio":
   - **Microphone**: оберіть "BlackHole 2ch"
   - **Speakers**: залиште ваші звичайні динаміки

## Налаштування нашої системи

1. Запустіть систему транскрибування: `python main.py`
2. У веб-інтерфейсі:
   - **Джерело аудіо**: оберіть "Системне аудіо"
   - **Аудіо пристрій**: оберіть "BlackHole 2ch"
3. Натисніть "Старт" та почніть мітинг

## Перевірка налаштувань

### Тест 1: Перевірка BlackHole

1. Відкрийте Audio MIDI Setup
2. Оберіть "BlackHole 2ch" в списку пристроїв
3. Натисніть кнопку "Configure Speakers" (іконка динаміка)
4. Якщо ви бачите налаштування - BlackHole встановлено правильно

### Тест 2: Перевірка Multi-Output

1. Увімкніть музику або відео
2. Перейдіть в System Preferences > Sound > Output
3. Переключіться між вашими динаміками та Multi-Output Device
4. Звук повинен відтворюватися в обох випадках

### Тест 3: Перевірка захоплення

1. Запустіть нашу систему
2. Оберіть "Системне аудіо" як джерело
3. Увімкніть музику або відео
4. У системі повинен з'явитися рівень аудіо сигналу

## Усунення проблем

### BlackHole не з'являється в списку пристроїв

**Рішення**:
1. Перезавантажте комп'ютер
2. Переконайтеся, що встановлення пройшло успішно
3. Перевірте System Preferences > Security & Privacy > General на наявність заблокованого ПЗ

### Немає звуку через динаміки після налаштування

**Рішення**:
1. Перевірте, що ваші динаміки обрані в Multi-Output Device
2. Переконайтеся, що динаміки встановлені як "Master Device"
3. Перевірте рівень гучності в System Preferences > Sound

### Система не захоплює аудіо з мітингів

**Рішення**:
1. Переконайтеся, що в додатку для мітингів обрано BlackHole як мікрофон
2. Перевірте, що в нашій системі обрано "Системне аудіо"
3. Переконайтеся, що Multi-Output Device активний в системних налаштуваннях

### Ехо або зворотний зв'язок

**Рішення**:
1. У додатку для мітингів вимкніть "Original Sound" або подібні опції
2. Переконайтеся, що мікрофон в додатку встановлено на BlackHole, а не на фізичний мікрофон
3. Зменшіть рівень гучності системи

### Низька якість звуку

**Рішення**:
1. У Audio MIDI Setup оберіть BlackHole 2ch
2. Встановіть Format на "2 ch 48000 Hz"
3. У нашій системі встановіть AUDIO_SAMPLE_RATE=48000 в .env файлі

## Альтернативи BlackHole

### Soundflower (застарілий)
- Більше не підтримується
- Може працювати на старіших версіях macOS
- Не рекомендується для нових установок

### Loopback (платний)
- Професійний інструмент від Rogue Amoeba
- Більше функцій та простіше налаштування
- Коштує $99, але має пробний період

### Background Music (безкоштовний)
- Відкритий код
- Більше функцій для управління аудіо
- Складніше в налаштуванні

## Безпека та приватність

### Дозволи macOS

Після встановлення BlackHole може знадобитися:
1. System Preferences > Security & Privacy > Privacy > Microphone
2. Додати нашу систему до списку дозволених додатків

### Конфіденційність

- BlackHole працює локально на вашому Mac
- Не передає дані в інтернет
- Аудіо обробляється тільки на вашому пристрої

## Видалення BlackHole

Якщо потрібно видалити BlackHole:

```bash
# Завантажте uninstaller
curl -L -o BlackHoleUninstaller.pkg https://existential.audio/blackhole/BlackHoleUninstaller.pkg

# Запустіть видалення
sudo installer -pkg BlackHoleUninstaller.pkg -target /

# Видаліть файл
rm BlackHoleUninstaller.pkg
```

Або вручну:
```bash
sudo rm -rf /Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver
sudo killall coreaudiod
```

## Додаткові ресурси

- [Офіційний сайт BlackHole](https://existential.audio/blackhole/)
- [GitHub репозиторій](https://github.com/ExistentialAudio/BlackHole)
- [Документація Apple Audio](https://developer.apple.com/documentation/coreaudio)
- [Форум підтримки](https://github.com/ExistentialAudio/BlackHole/discussions)

---

**Примітка**: Цей гайд написано для macOS 10.15+ та BlackHole 2ch. Для інших версій кроки можуть відрізнятися.