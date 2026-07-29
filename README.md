# Neural Audio Censor by Zombak (Zavtracast Podcast)

Утилита с графическим интерфейсом для автоматического цензурирования плохих слов в `.wav` аудиофайлах. Использует нейросетевые модели распознавания речи для транскрибации и точной замены нецензурных выражений на звуковой сигнал (бип) или пользовательский звук.

## Возможности

- **Три модели распознавания речи:**
  - **Whisper** (faster-whisper) — tiny / base / small / medium / large-v3, word-level timestamps, до 9 языков + автоопределение
  - **GigaAM-Multilingual** — Russian/English, segment-level обработка
  - **Parakeet TDT 0.6B v3** (NVIDIA) — Russian/English, word-level timestamps, точная цензура как у Whisper (**по умолчанию**)

- **Три уровня фильтрации:** blacklist.txt, whitelist.txt и regex-паттерн для матовых слов
- **Гибкая настройка звука цензуры:**
  - Стандартный бип (1000 Гц) с настраиваемой длительностью (% от слова)
  - Пользовательский .wav файл — зацикливается или обрезается под длину слова
  - Контроль громкости: -40 dB до 0 dB
- **Пакетная обработка** всех `.wav` файлов в папке
- **Экспорт транскрипций** для каждого файла (с таймкодами)
- **Логи и статистика:** `censor-log.txt`, `stats.txt`
- **Мультиязычный интерфейс:** русский, английский, немецкий, французский, испанский, китайский, японский, корейский, украинский, польский

## Системные требования

- Windows (тестировалось на Win10/Win11)
- Python 3.10+
- NVIDIA GPU рекомендуется для ускорения (CUDA), но работает и на CPU

## Установка

1. Клонируй или скачай репозиторий:
   ```bash
   git clone https://github.com/zombak/neural-audio-censor
   
   ```

2. Создай виртуальное окружение (если ещё нет):
   ```bash
   python -m venv venv
   call venv\Scripts\activate.bat
   ```

3. Установи зависимости:
   ```bash
   pip install torch torchaudio pydub python-dotenv faster-whisper transformers
   ```

4. Для использования **Parakeet** дополнительно установи NeMo:
   ```bash
   pip install -U "nemo_toolkit[asr]"
   ```
   > ⚠️ Пакет большой (~2–3 ГБ с зависимостями). Если Parakeet не нужен — этот шаг можно пропустить.

5. Скачай модели (по желанию, для оффлайн-режима):
   - Whisper: `faster-whisper-{tiny|base|small|medium|large-v3}` в корень проекта
   - GigaAM: папку `GigaAM-Multilingual` из Hugging Face
   - Parakeet: папку `parakeet-tdt-0.6b-v3` из Hugging Face

## Настройка (.env)

Скопируй `.env.example` в `.env` и отредактируй пути:

```ini
UI_LANGUAGE=russian
AUDIO_FOLDER_PATH=c:\Utils\_audio
GIGAAM_MODEL_PATH=c:\Utils\audio-censor\GigaAM-Multilingual
WHISPER_TINY_PATH=c:\Utils\audio-censor\faster-whisper-tiny
WHISPER_BASE_PATH=c:\Utils\audio-censor\faster-whisper-base
WHISPER_SMALL_PATH=c:\Utils\audio-censor\faster-whisper-small
WHISPER_MEDIUM_PATH=c:\Utils\audio-censor\faster-whisper-medium
WHISPER_LARGE_V3_PATH=c:\Utils\audio-censor\faster-whisper-large-v3
PARAKEET_MODEL_PATH=C:\Utils\audio-censor\parakeet-tdt-0.6b-v3
```

## Запуск

**Вариант 1 (быстрый):**
```bash
censor.bat
```

**Вариант 2 (вручную):**
```bash
call venv\Scripts\activate.bat
python main.py
```

## Как использовать

1. Положи `.wav` файлы в папку, указанную в поле "Audio folder" (или выбери через Browse)
2. Выбери модель (Whisper / GigaAM / Parakeet) и версию (для Whisper)
3. Выбери режим загрузки: Online (скачать с Hugging Face) или Offline (локальная модель)
4. Выбери язык распознавания речи
5. Выбери язык цензуры (Russian / English) — определяет, по какому списку слов искать маты
6. Настрой процент запикивания слова ползунком (0–100%)
7. Опционально: включи "Custom sound" и выбери свой .wav файл для замены бипа
8. Настрой громкость цензуры (-40 до 0 dB)
9. Выбери уровень логов: Detailed или Basic
10. Нажми **🚀 Start** для обработки с цензурой или **📄 Transcribe** только для транскрипции без цензуры

## Структура файлов

- `main.py` — основной код приложения
- `censor.bat` — быстрый запуск (активирует venv + запускает main.py)
- `.env` — конфигурация путей и настроек
- `blacklist.txt` — точные слова для цензуры (по одному на строку)
- `whitelist.txt` — исключения из цензуры (например, допустимые сокращения)
- `words-russian.txt` / `words-english.txt` — regex-паттерны матовых слов
- `ui-{lang}.txt` — переводы интерфейса

## Лицензия

Pet-project для [Zavtracast](https://zavtracast.ru/) Podcast. Не является коммерческим продуктом.
