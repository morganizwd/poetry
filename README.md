# Интеллектуальная система генерации стихов

Проект генерирует русскоязычные стихи двумя способами:

1. `Markov` генерирует строки по статистике переходов между словами.
2. `LSTM` учится продолжать текст и собирать более связные стихотворные строки.

Дополнительно проект умеет использовать локальную `Ollama` как редактор:

- улучшать черновик после генерации;
- сравнивать версию `raw` и версию `после LLM`;
- сохранять подробные отчёты в `Excel (.xlsx)` и резервные копии в `CSV`.

Сейчас основной сценарий проекта: локальный запуск без Google Colab.

## Самый простой запуск

Если вы просто хотите запустить проект и посмотреть результат, делайте так.

### 1. Установите Python

Для проекта лучше всего подходит `Python 3.11` или `Python 3.12`.

Важно:

- `TensorFlow` для этой LSTM-части не ставится на `Python 3.14`;
- если у вас уже стоит несколько Python, лучше явно использовать `3.11`.

### 2. Создайте виртуальное окружение

Пример для Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Если хотите LLM-редактирование, установите Ollama

Если `Ollama` вам не нужна, этот шаг можно пропустить.

После установки запустите локальный сервер и скачайте модель:

```powershell
ollama serve
ollama pull qwen2.5:7b-instruct
```

### 4. Запустите desktop-приложение

```powershell
.\.venv\Scripts\python.exe poetry_desktop_app.py
```

Или двойным кликом по:

```text
poetry_desktop_app.pyw
```

## Что делать в окне программы

1. Выберите готовый сценарий запуска.
2. Если нужен LLM, включите `Ollama` и нажмите `Проверить Ollama`.
3. Нажмите `Запустить по текущим настройкам`.
4. Следите за логом справа.
5. После завершения откройте вкладки `Стихи`, `Сравнение` и `Отчёты`.

## Готовые сценарии в GUI

### Быстрый старт (рекомендуется)

Для первого запуска.

Что делает:

- подготавливает данные;
- запускает `Markov`;
- запускает быструю `LSTM`;
- не тратит время на долгую full LSTM и batch-сравнение.

### Обычный локальный запуск

Для более серьёзного локального прогона без LLM.

Что делает:

- подготавливает данные;
- запускает `Markov`;
- запускает основную `LSTM`;
- пропускает лишний batch, чтобы запуск не был слишком долгим.

### Черновик + улучшение через Ollama

Для генерации и последующего улучшения стиха локальной LLM.

Что делает:

- генерирует стих локально;
- отдаёт черновик в `Ollama`;
- позволяет потом сравнить вариант `до` и `после LLM`.

Перед запуском этого сценария должны работать:

- `ollama serve`
- загруженная модель, например `qwen2.5:7b-instruct`

### Только подготовить данные

Для случая, когда нужно просто скачать датасет и собрать `poems_clean.txt`.

Что делает:

- скачивает исходный JSON-датасет;
- собирает `data/poems_clean.txt`;
- копирует его в папки моделей;
- не обучает ничего и не генерирует стихи.

## Если не хотите пользоваться GUI

Можно запускать проект напрямую из консоли.

### Быстрый запуск

```powershell
python poetry_local_pipeline.py --fast
```

### Обычный запуск

```powershell
python poetry_local_pipeline.py
```

### Только подготовка данных

```powershell
python poetry_local_pipeline.py --prepare-data-only
```

### Запуск со своим датасетом

Если у вас уже есть готовый `poems_clean.txt`:

```powershell
python poetry_local_pipeline.py --dataset-txt "D:\data\poems_clean.txt"
```

## Где смотреть результаты

После запуска полезны вот эти папки:

- `data/poems_clean.txt` — подготовленный датасет;
- `reports/` — подробные и краткие отчёты в `Excel (.xlsx)` плюс резервные `CSV`;
- `cache/poetry_lstm_cache` — кэш локальной LSTM;
- `LSTM_generation_2/` — рабочие файлы LSTM;
- `Markov_chain_2/` — рабочие файлы Markov.

В desktop-приложении всё это можно открыть через вкладки и кнопки, не лазая по папкам вручную.

## Важное про GPU на Windows

Если у вас видеокарта NVIDIA, но `LSTM` всё равно идёт на `CPU`, это ожидаемо для текущей связки:

- `native Windows`
- `TensorFlow 2.11+`

Для этой конфигурации `TensorFlow` не использует GPU для LSTM на Windows.

Это не ошибка GUI и не ошибка проекта.

Что это значит на практике:

- `LSTM` на вашем Windows будет работать на `CPU`;
- локальная `Ollama` работает отдельно от `TensorFlow` и использует свой собственный runtime;
- если нужен именно `GPU` для `LSTM`, нужен `WSL2/Linux` или перенос модели на `PyTorch`.

## Частые проблемы

### `Could not find a version that satisfies the requirement tensorflow`

Обычно это значит, что вы создали `.venv` не на том Python.

Чаще всего причина такая:

- окружение было создано на `Python 3.14`

Решение:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### `Ollama probe failed` или `WinError 10061`

Это обычно значит, что локальный сервер `Ollama` не запущен.

Проверьте:

```powershell
ollama serve
```

И убедитесь, что модель скачана:

```powershell
ollama pull qwen2.5:7b-instruct
```

### GUI запущен, но использует не тот Python

Запускайте так:

```powershell
.\.venv\Scripts\python.exe poetry_desktop_app.py
```

Тогда и пайплайн, и проверки будут идти через правильное окружение.

## Структура проекта

```text
poetry/
├── Markov_chain_2/
│   └── Markov_chain.py
├── LSTM_generation_2/
│   └── lstm_generation.py
├── llm_poetry_tools.py
├── poetry_desktop_app.py
├── poetry_desktop_app.pyw
├── poetry_local_pipeline.py
├── poetry_colab_final.ipynb
├── requirements.txt
└── README.md
```

## Запуск в Google Colab

Colab теперь не основной, а дополнительный вариант.

Если он всё же нужен, используйте:

```text
poetry_colab_final.ipynb
```

Но для обычной локальной работы лучше использовать:

- `poetry_desktop_app.py`
- или `poetry_local_pipeline.py`
