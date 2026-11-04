import hashlib
import tiktoken
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from src.rag_prep.models import PreparedDocument, PreparedChunk, ChunkMetadata
from src.rag_prep.config import ChunkingPipelineConfig

class ChunkSplittingStage:
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config
        
        # Инициализация tiktoken (Слайд 17)
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.chunking.tokenizer_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding('cl100k_base')

        self.splitter = self._build_splitter()

    def _tokenize(self, text: str) -> list[int]:
        """Считает токены для передачи в LlamaIndex"""
        return self.encoding.encode(text)

    def _build_splitter(self):
        """Выбор стратегии сплиттера (Слайд 22)"""
        kwargs = {
            'chunk_size': self.config.chunking.chunk_size,
            'chunk_overlap': self.config.chunking.chunk_overlap,
            'tokenizer': self._tokenize  # Передаем функцию токенизации
        }
        
        if self.config.chunking.strategy == 'token':
            return TokenTextSplitter(**kwargs)
        # По умолчанию используем SentenceSplitter (semantic-aware)
        return SentenceSplitter(**kwargs)

    def run(self, documents: list[PreparedDocument]) -> list[PreparedChunk]:
        all_chunks = []
        print("✂️ Нарезка документов на чанки (Token-Aware)...")
        
        for doc in documents:
            # LlamaIndex Splitter разбивает текст на список строк
            text_chunks = self.splitter.split_text(doc.text)
            
            cursor = 0  # Курсор для поиска offsets (чтобы не искать с начала каждый раз)
            
            for pos, chunk_text in enumerate(text_chunks):
                # 1. Считаем количество токенов в чанке
                token_count = len(self._tokenize(chunk_text))
                
                # 2. Вычисляем offsets (Слайд 68, 71)
                start_char = doc.text.find(chunk_text, cursor)
                
                if start_char == -1:
                    # Fallback, если сплиттер изменил текст (например, убрал двойные пробелы)
                    start_char = cursor
                    offset_strategy = "estimated_fallback"
                else:
                    cursor = start_char + len(chunk_text)
                    offset_strategy = "semantic_block_span"
                    
                end_char = start_char + len(chunk_text)
                
                # 3. Генерируем стабильный ID чанка (Слайд 4 подсказок: hash(doc_id + pos + text))
                chunk_id_str = f"{doc.metadata.id}_{pos}_{chunk_text[:30]}"
                chunk_id = hashlib.sha256(chunk_id_str.encode('utf-8')).hexdigest()[:16]

                # 4. Собираем метадату
                metadata = ChunkMetadata(
                    document_id=doc.metadata.id,
                    position=pos,
                    chunk_token_count=token_count,
                    chunk_size=self.config.chunking.chunk_size,
                    chunk_overlap=self.config.chunking.chunk_overlap,
                    chunking_strategy=self.config.chunking.strategy,
                    source=doc.metadata.source,
                    section=doc.metadata.section,
                    text_hash=hashlib.sha256(chunk_text.encode('utf-8')).hexdigest(),
                    parent_ids=[doc.metadata.id],
                    lineage=doc.metadata.lineage,   # Передаем lineage дальше (Слайд 75)
                    hierarchy=doc.metadata.hierarchy, # Передаем hierarchy дальше
                    chunk_start_char=start_char,
                    chunk_end_char=end_char,
                    offset_strategy=offset_strategy
                )
                
                # Добавляем ID чанка в lineage для полной трассировки
                metadata.lineage['chunk_id'] = chunk_id
                
                all_chunks.append(PreparedChunk(text=chunk_text, metadata=metadata))
                
        print(f"✅ Создано {len(all_chunks)} чанков.")
        return all_chunks