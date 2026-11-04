# --- ИМПОРТЫ ДЛЯ ПАЙПЛАЙНА ПОДГОТОВКИ ДАННЫХ (ЭТАП 1) ---
from src.rag_prep.config import PipelineConfig, ChunkingPipelineConfig
from src.rag_prep.stages.loading import LlamaIndexLoadingStage
from src.rag_prep.stages.parsing import UnstructuredParsingStage
from src.rag_prep.stages.cleaning import TextCleaningStage
from src.rag_prep.stages.normalization import TextNormalizationStage
from src.rag_prep.stages.deduplication import DeduplicationStage
from src.rag_prep.stages.structuring import StructuringStage
from src.rag_prep.stages.exporting import ExportStage
from src.rag_prep.tracking import MLflowTracker

# --- ИМПОРТЫ ДЛЯ ПАЙПЛАЙНА ЧАНКИНГА (ЭТАП 2) ---
from src.rag_prep.chunking_stages.loading import PreparedDocumentLoadingStage
from src.rag_prep.chunking_stages.splitting import ChunkSplittingStage
from src.rag_prep.chunking_stages.validation import ChunkValidationStage
from src.rag_prep.chunking_stages.exporting import ChunkExportStage


class RagPreparationPipeline:
    """Пайплайн подготовки данных (Loading -> Parsing -> ... -> Export)"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Инициализация всех стадий (Слайд 10)
        self.loader = LlamaIndexLoadingStage(config.loader)
        self.parser = UnstructuredParsingStage(config.parser)
        self.cleaner = TextCleaningStage(config.cleaning)
        self.normalizer = TextNormalizationStage(config.normalization)
        self.deduplicator = DeduplicationStage(config.deduplication)
        self.structurer = StructuringStage(config.structuring)
        self.exporter = ExportStage(config)

    def run(self) -> dict:
        print("🚀 Запуск пайплайна подготовки данных для RAG...")
        
        # 1. Loading
        print(f"📂 Загрузка из {self.config.paths.input_dir}...")
        sources = self.loader.run(self.config.paths.input_dir)
        print(f"   Найдено файлов: {len(sources)}")
        
        # 2. Parsing
        print("📄 Парсинг документов...")
        raw_elements = self.parser.run(sources)
        print(f"   Извлечено элементов: {len(raw_elements)}")
        
        # 3. Cleaning
        print("🧹 Очистка текста и расчет Quality Signals...")
        cleaned_elements = self.cleaner.run(raw_elements)
        print(f"   Очищено элементов: {len(cleaned_elements)}")
        
        # 4. Normalization
        print("🔤 Нормализация текста (Unicode + spaCy)...")
        normalized_elements = self.normalizer.run(cleaned_elements)
        
        # 5. Deduplication
        print("👯 Поиск дубликатов (Exact + MinHash)...")
        unique_elements = self.deduplicator.run(normalized_elements)
        print(f"   Осталось уникальных элементов: {len(unique_elements)}")
        
        # 6. Structuring
        print("🏗️ Структурирование и сборка метадаты (Lineage)...")
        prepared_docs = self.structurer.run(unique_elements)
        print(f"   Сформировано документов: {len(prepared_docs)}")
        
        # Собираем статистику для Manifest
        counts = {
            "sources_count": len(sources),
            "raw_elements_count": len(raw_elements),
            "cleaned_elements_count": len(cleaned_elements),
            "unique_elements_count": len(unique_elements),
            "prepared_docs_count": len(prepared_docs)
        }
        
        # 7. Export
        print("💾 Сохранение результатов...")
        manifest = self.exporter.run(prepared_docs, counts)
        
        # 8. Tracking (MLflow)
        tracker = MLflowTracker(self.config)
        manifest_path = f"{self.config.paths.output_dir}/manifest.json"
        tracker.log_run(counts, manifest_path)
        
        print("✅ Пайплайн подготовки данных успешно завершен!")
        return manifest


class RagChunkingPipeline:
    """Пайплайн чанкинга (Loading -> Splitting -> Validation -> Export)"""
    
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config
        
        # Инициализация стадий чанкинга
        self.loader = PreparedDocumentLoadingStage(config)
        self.splitter = ChunkSplittingStage(config)
        self.validator = ChunkValidationStage(config)
        self.exporter = ChunkExportStage(config)

    def run(self) -> dict:
        print("🚀 Запуск пайплайна Chunking...")
        
        # 1. Loading (Чтение подготовленных JSONL)
        documents = self.loader.run()
        
        # 2. Splitting (Нарезка на Semantic Blocks / Token-Aware)
        chunks = self.splitter.run(documents)
        
        # 3. Validation (Проверка на пустоту, oversized и метрики)
        valid_chunks, metrics = self.validator.run(chunks)
        
        # 4. Export
        manifest = self.exporter.run(valid_chunks, metrics)
        
        # 5. Tracking (MLflow) - ИСПРАВЛЕНАЯ СТРОКА
        tracker = MLflowTracker(self.config)
        manifest_path = f"{self.config.paths.output_dir}/manifest.json"
        tracker.log_chunking_run(metrics, manifest_path) # <--- Вызываем новый метод
        
        print("✅ Пайплайн Chunking успешно завершен!")
        return manifest

            