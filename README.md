# RAG Data Preparation & Chunking Pipeline

Production-ready пайплайн для подготовки данных и нарезки на чанки (Chunking) для систем Retrieval-Augmented Generation (RAG).

## 🏗️ Архитектура проекта

Проект состоит из двух независимых пайплайнов, построенных по паттерну **Facade** (разделение ответственности):

### 1. Data Preparation Pipeline (Подготовка данных)
Состоит из следующих стадий:
1. **Loading** (`LlamaIndex`) — сканирование директории, фильтрация файлов, генерация SHA-256 хешей.
2. **Parsing** (`Unstructured`) — семантическое извлечение элементов (Title, NarrativeText).
3. **Cleaning** — удаление boilerplate (cookie, меню), расчет Quality Signals.
4. **Normalization** (`spaCy`) — Unicode NFKC, приведение к единому виду, подсчет предложений.
5. **Deduplication** (`MinHash LSH` + `SHA-256`) — удаление точных и нечетких дубликатов.
6. **Structuring** — сборка документов, генерация Lineage и Hierarchy.
7. **Exporting** — атомарная запись в JSON, JSONL и генерация Manifest.

### 2. Chunking Pipeline (Нарезка на чанки)
Пайплайн подготовки данных к этапу embeddings:
1. **Loading** — чтение подготовленных документов из `dataset.jsonl`.
2. **Splitting** — Token-Aware нарезка с использованием `tiktoken` и `SentenceSplitter` (LlamaIndex) с сохранением границ предложений.
3. **Validation** — проверка на пустые, undersized и oversized чанки, расчет метрик качества.
4. **Exporting** — сохранение чанков с обогащенной метадатой (Offsets, Lineage, Hierarchy) в JSON/JSONL и генерация Manifest.

## 🛠️ Стек технологий
* **Python 3.10+**
* **Pydantic** (строгая валидация конфигурации и метадаты)
* **Prefect** (оркестрация пайплайнов)
* **MLflow** (трекинг экспериментов и метрик)
* **LlamaIndex / Unstructured** (загрузка, парсинг и сплиттеры)
* **Tiktoken** (подсчет токенов)
* **Datasketch** (поиск near-duplicates)

## 🚀 Как запустить

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Запустите пайплайны последовательно (по умолчанию используется Prefect):
```bash
# Шаг 1: Подготовка данных (создаст dataset.jsonl в data/prepared/)
python -m src.rag_prep.cli prepare

# Шаг 2: Нарезка на чанки (создаст chunks.jsonl в data/chunks/)
python -m src.rag_prep.cli chunk
```

3. Для отладки можно запускать пайплайны без оркестратора (напрямую):
```bash
python -m src.rag_prep.cli prepare --no-prefect
python -m src.rag_prep.cli chunk --no-prefect
```

## 📊 Мониторинг (MLflow)
Для просмотра метрик экспериментов (параметры нарезки, количество чанков, ошибки валидации) запустите локальный сервер MLflow:
```bash
mlflow ui
```
Затем откройте в браузере `http://127.0.0.1:5000`.

## 📌 Особенности реализации
* **Token-Aware Chunking:** Размер чанка и overlap считаются строго в токенах (`tiktoken`), а не в символах.
* **Offsets & Attribution:** Каждый чанк хранит `chunk_start_char` и `chunk_end_char` для точного цитирования.
* **Воспроизводимость:** Все параметры задаются через YAML-конфигурацию. Стабильные ID генерируются на основе хэшей.