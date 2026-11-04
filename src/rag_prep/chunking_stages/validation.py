from src.rag_prep.models import PreparedChunk
from src.rag_prep.config import ChunkingPipelineConfig

class ChunkValidationStage:
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config

    def run(self, chunks: list[PreparedChunk]) -> tuple[list[PreparedChunk], dict]:
        print("🔍 Валидация чанков...")
        
        valid_chunks = []
        
        # Счетчики метрик (Слайд 86)
        empty_chunks_count = 0
        undersized_chunks_count = 0
        oversized_chunks_count = 0
        missing_lineage_count = 0
        estimated_offsets_count = 0

        for chunk in chunks:
            text = chunk.text.strip()
            token_count = chunk.metadata.chunk_token_count
            
            # 1. Проверка на пустоту
            if not text:
                empty_chunks_count += 1
                continue # Выбрасываем пустой чанк
                
            # 2. Проверка на минимальный размер (например, меньше 10 токенов)
            if token_count < 10:
                undersized_chunks_count += 1
                # По ТЗ можно логировать, но мы оставим, чтобы не терять смысл коротких фактов
                
            # 3. Проверка на максимальный жесткий лимит (Oversized)
            if token_count > self.config.chunking.max_chunk_tokens:
                oversized_chunks_count += 1
                # В идеале здесь нужен fallback-сплиттер, но для валидации просто фиксируем
                
            # 4. Проверка метадаты (Lineage)
            if not chunk.metadata.lineage:
                missing_lineage_count += 1
                
            # 5. Проверка стратегии offsets
            if "estimated" in chunk.metadata.offset_strategy:
                estimated_offsets_count += 1
            
            # Если чанк прошел базовые проверки (не пустой), добавляем в валидные
            valid_chunks.append(chunk)

        # Собираем отчет (попадет в manifest.json)
        validation_metrics = {
            "total_chunks_processed": len(chunks),
            "valid_chunks_count": len(valid_chunks),
            "empty_chunks_count": empty_chunks_count,
            "undersized_chunks_count": undersized_chunks_count,
            "oversized_chunks_count": oversized_chunks_count,
            "missing_lineage_count": missing_lineage_count,
            "estimated_offsets_count": estimated_offsets_count
        }
        
        print(f"✅ Валидация завершена. Валидных чанков: {len(valid_chunks)} из {len(chunks)}.")
        if oversized_chunks_count > 0:
            print(f"⚠️ Внимание: {oversized_chunks_count} чанков превышают max_chunk_tokens ({self.config.chunking.max_chunk_tokens}).")
            
        return valid_chunks, validation_metrics