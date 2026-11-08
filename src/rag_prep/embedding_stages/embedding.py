import numpy as np
import hashlib
import tiktoken
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from src.rag_prep.models import PreparedChunk, EmbeddedChunk, EmbeddingMetadata
from src.rag_prep.config import EmbeddingsPipelineConfig

logger = logging.getLogger(__name__)

class TextEmbeddingStage:
    def __init__(self, config: EmbeddingsPipelineConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        print(f"🧠 Загрузка модели {config.embedding.model}...")
        self.model = SentenceTransformer(config.embedding.model)
        
        # Инициализация токенизатора для проверки лимитов (Слайд 24)
        try:
            # Используем базовый энкодер для подсчета токенов
            self.encoding = tiktoken.get_encoding("cl100k_base") 
        except Exception as e:
            logger.warning(f"Не удалось загрузить tiktoken: {e}. Пропуск валидации токенов.")

    def _validate_tokens(self, text: str) -> bool:
        """Проверка длины текста (Слайд 24)"""
        if not hasattr(self, 'encoding'):
            return True
        token_count = len(self.encoding.encode(text))
        return token_count <= self.config.embedding.max_input_tokens

    def _build_batches(self, chunks: list[PreparedChunk]) -> list[list[PreparedChunk]]:
        """Динамический batching (Слайд 25)"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for chunk in chunks:
            # Пропускаем слишком длинные чанки
            if not self._validate_tokens(chunk.text):
                logger.warning(f"⚠️ Чанк {chunk.metadata.lineage.get('chunk_id')} превышает max_input_tokens и будет пропущен.")
                continue
                
            chunk_tokens = chunk.metadata.chunk_token_count
            
            # Если добавление чанка превысит лимиты батча, закрываем текущий батч
            if (len(current_batch) >= self.config.embedding.batch_size or 
                current_tokens + chunk_tokens > self.config.embedding.max_batch_tokens):
                if current_batch:
                    batches.append(current_batch)
                current_batch = []
                current_tokens = 0
                
            current_batch.append(chunk)
            current_tokens += chunk_tokens
            
        if current_batch:
            batches.append(current_batch)
            
        return batches

    @retry(
        stop=stop_after_attempt(3), # Можно брать из конфига
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Вызов модели с retry-механизмом через tenacity (Слайд 27)"""
        return self.model.encode(texts, convert_to_numpy=True)

    def run(self, chunks: list[PreparedChunk]) -> list[EmbeddedChunk]:
        print(f"🔢 Расчет эмбеддингов...")
        embedded_results = []
        
        batches = self._build_batches(chunks)
        print(f"   Сформировано батчей: {len(batches)}")
        
        for batch_idx, batch_chunks in enumerate(batches):
            batch_texts = [c.text for c in batch_chunks]
            
            # Вызов с retry
            embeddings = self._embed_batch(batch_texts)

            # L2-Нормализация (Слайд 28)
            if self.config.embedding.l2_normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-10)

            for chunk, emb in zip(batch_chunks, embeddings):
                # Считаем норму и хэш (Слайд 29)
                norm_val = float(np.linalg.norm(emb))
                emb_hash = hashlib.sha256(emb.tobytes()).hexdigest()

                # Формируем строгую метадату (Слайд 36)
                meta = EmbeddingMetadata(
                    source=chunk.metadata.source or "unknown",
                    document_id=chunk.metadata.document_id,
                    model=self.config.embedding.model,
                    dimensions=len(emb),
                    embedding_hash=emb_hash,
                    embedding_norm=norm_val,
                    pipeline_run_id=self.run_id,
                    created_at=datetime.now(),
                    chunk_id=chunk.metadata.lineage.get("chunk_id"),
                    text_hash=chunk.metadata.text_hash,
                    lineage=chunk.metadata.lineage,
                    hierarchy=chunk.metadata.hierarchy
                )
                
                embedded_results.append(EmbeddedChunk(
                    text=chunk.text,
                    embedding=emb.tolist(),
                    metadata=meta
                ))
            
            print(f"   Батч {batch_idx + 1}/{len(batches)} обработан (Чанков: {len(batch_chunks)}).")
            
        print(f"✅ Успешно рассчитано {len(embedded_results)} эмбеддингов.")
        return embedded_results