# RAG Data Preparation Pipeline

Production-ready пайплайн для подготовки данных к Retrieval-Augmented Generation (RAG).

## 🏗️ Архитектура
Пайплайн построен по паттерну **Facade** и состоит из независимых стадий:
1. **Loading** (`LlamaIndex`) — сканирование директории, фильтрация файлов, генерация хешей.
2. **Parsing** (`Unstructured`) — семантическое извлечение элементов (Title, NarrativeText).
3. **Cleaning** — удаление boilerplate (cookie, меню), расчет Quality Signals.
4. **Normalization** (`spaCy`) — Unicode NFKC, приведение к единому виду, подсчет предложений.
5. **Deduplication** (`MinHash LSH` + `SHA-256`) — удаление точных и нечетких дубликатов.
6. **Structuring** — сборка документов, генерация Lineage и Hierarchy.
7. **Exporting** — атомарная запись в JSON, JSONL и генерация Manifest.

## 🛠️ Стек технологий
* **Python 3.10+**
* **Pydantic** (валидация конфигурации и метадаты)
* **Prefect** (оркестрация пайплайна)
* **MLflow** (трекинг экспериментов и метрик)
* **LlamaIndex / Unstructured** (загрузка и парсинг)
* **Datasketch** (поиск near-duplicates)

## 🚀 Как запустить

1. Установите зависимости:
```bash
pip install -r requirements.txt