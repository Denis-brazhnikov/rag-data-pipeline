from src.rag_prep.models import PreparedChunk
from src.rag_prep.config import ChunkingPipelineConfig

class ChunkValidationStage:
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config

    def _calculate_quality(self, text: str, token_count: int) -> dict:
        """Считает Quality Signals (плотность текста, мусор)"""
        if token_count == 0:
            return {"is_low_quality_chunk": True, "garbage_score": 1.0}
            
        chars = len(text)
        # Плотность: сколько символов на токен. Если меньше 2, значит это мусор/спецсимволы.
        density = chars / token_count 
        
        # Простая эвристика на мусор (много знаков препинания подряд)
        punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        punct_ratio = punct_count / chars if chars > 0 else 1.0
        
        is_low_quality = density < 1.5 or punct_ratio > 0.4
        
        return {
            "token_density": round(density, 2),
            "punctuation_ratio": round(punct_ratio, 2),
            "is_low_quality_chunk": is_low_quality
        }

    def run(self, chunks: list[PreparedChunk]) -> tuple[list[PreparedChunk], dict]:
        print("🔍 Валидация чанков (Fail-fast & Quality Scoring)...")
        
        valid_chunks = []
        empty_chunks_count = 0
        oversized_chunks_count = 0
        low_quality_chunks_count = 0

        for chunk in chunks:
            text = chunk.text.strip()
            token_count = chunk.metadata.chunk_token_count
            
            if not text:
                empty_chunks_count += 1
                continue 
                
            if token_count > self.config.chunking.max_chunk_tokens:
                oversized_chunks_count += 1
                
            # Считаем качество
            quality_signals = self._calculate_quality(text, token_count)
            chunk.metadata.quality = quality_signals
            
            if quality_signals["is_low_quality_chunk"]:
                low_quality_chunks_count += 1
                
            valid_chunks.append(chunk)

        # FAIL-FAST: Если пустых или огромных чанков слишком много, роняем пайплайн
        if empty_chunks_count > len(chunks) * 0.5:
            raise ValueError("Критическая ошибка: Более 50% чанков пустые.")
        if oversized_chunks_count > 0:
            print(f"⚠️ Внимание: {oversized_chunks_count} чанков превышают лимит!")

        validation_metrics = {
            "total_chunks_processed": len(chunks),
            "valid_chunks_count": len(valid_chunks),
            "empty_chunks_count": empty_chunks_count,
            "oversized_chunks_count": oversized_chunks_count,
            "low_quality_chunks_count": low_quality_chunks_count
        }
        
        return valid_chunks, validation_metrics