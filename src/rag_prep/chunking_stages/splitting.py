import hashlib
import re
import tiktoken
from llama_index.core.node_parser import SentenceSplitter
from src.rag_prep.models import PreparedDocument, PreparedChunk, ChunkMetadata, SemanticBlock
from src.rag_prep.config import ChunkingPipelineConfig

class ChunkSplittingStage:
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.chunking.tokenizer_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding('cl100k_base')
            
        # Сплиттер нужен только для fallback-режима (если один блок больше chunk_size)
        self.fallback_splitter = SentenceSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            tokenizer=self._tokenize
        )

    def _tokenize(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def _create_semantic_blocks(self, text: str) -> list[SemanticBlock]:
        """Разбивает сырой текст на смысловые блоки (абзацы) с точными координатами."""
        blocks = []
        # Регулярка для поиска абзацев (разделенных двойным переносом строки)
        pattern = re.compile(r'([^\n]+(?:\n[^\n]+)*)')
        
        position = 0
        for match in pattern.finditer(text):
            block_text = match.group(1).strip()
            if not block_text:
                continue
                
            start_char = match.start(1)
            end_char = match.end(1)
            token_count = len(self._tokenize(block_text))
            
            blocks.append(SemanticBlock(
                id=hashlib.sha256(f"{start_char}_{block_text[:20]}".encode()).hexdigest()[:12],
                text=block_text,
                position=position,
                start_char=start_char,
                end_char=end_char,
                token_count=token_count
            ))
            position += 1
            
        return blocks

    def _pack_blocks(self, doc: PreparedDocument, blocks: list[SemanticBlock]) -> list[PreparedChunk]:
        """Упаковывает Semantic Blocks в чанки, не превышая chunk_size."""
        chunks = []
        current_blocks = []
        current_tokens = 0
        
        for block in blocks:
            # Если сам по себе блок больше chunk_size, режем его через fallback
            if block.token_count > self.config.chunking.chunk_size:
                if current_blocks:
                    chunks.append(self._finalize_chunk(doc, current_blocks, len(chunks)))
                    current_blocks, current_tokens = [], 0
                
                split_texts = self.fallback_splitter.split_text(block.text)
                for s_text in split_texts:
                    s_tokens = len(self._tokenize(s_text))
                    # Ищем точное совпадение, чтобы сохранить offset_strategy = semantic_block_span
                    s_start = doc.text.find(s_text, block.start_char)
                    if s_start == -1: s_start = block.start_char
                    
                    # Создаем фейковый блок для fallback
                    temp_block = SemanticBlock(
                        id=hashlib.sha256(f"{s_start}_{s_text}".encode()).hexdigest()[:12],
                        text=s_text, position=block.position, start_char=s_start,
                        end_char=s_start + len(s_text), token_count=s_tokens
                    )
                    chunks.append(self._finalize_chunk(doc, [temp_block], len(chunks)))
                continue

            # Проверяем, влезает ли блок в текущий чанк
            if current_tokens + block.token_count > self.config.chunking.chunk_size and current_blocks:
                chunks.append(self._finalize_chunk(doc, current_blocks, len(chunks)))
                
                # Формируем Overlap (берем последние блоки предыдущего чанка)
                overlap_blocks = []
                overlap_tokens = 0
                for prev_b in reversed(current_blocks):
                    if overlap_tokens + prev_b.token_count <= self.config.chunking.chunk_overlap:
                        overlap_blocks.insert(0, prev_b)
                        overlap_tokens += prev_b.token_count
                    else:
                        break
                current_blocks = overlap_blocks
                current_tokens = overlap_tokens

            current_blocks.append(block)
            current_tokens += block.token_count

        # Добавляем последний чанк
        if current_blocks:
            chunks.append(self._finalize_chunk(doc, current_blocks, len(chunks)))

        return chunks

    def _finalize_chunk(self, doc: PreparedDocument, blocks: list[SemanticBlock], position: int) -> PreparedChunk:
        """Собирает чанк из блоков и формирует метадату."""
        chunk_text = "\n\n".join(b.text for b in blocks)
        start_char = blocks[0].start_char
        end_char = blocks[-1].end_char
        token_count = sum(b.token_count for b in blocks)
        
        chunk_id = hashlib.sha256(f"{doc.metadata.id}_{position}_{chunk_text[:30]}".encode('utf-8')).hexdigest()[:16]

        metadata = ChunkMetadata(
            document_id=doc.metadata.id,
            position=position,
            chunk_token_count=token_count,
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            chunking_strategy="semantic_block_packing", # Честно указываем стратегию
            source=doc.metadata.source,
            section=doc.metadata.section,
            text_hash=hashlib.sha256(chunk_text.encode('utf-8')).hexdigest(),
            parent_ids=[doc.metadata.id],
            lineage=doc.metadata.lineage,
            hierarchy=doc.metadata.hierarchy,
            chunk_start_char=start_char,
            chunk_end_char=end_char,
            offset_strategy="semantic_block_span" # Точные координаты!
        )
        metadata.lineage['chunk_id'] = chunk_id
        
        return PreparedChunk(text=chunk_text, metadata=metadata)

    def run(self, documents: list[PreparedDocument]) -> list[PreparedChunk]:
        all_chunks = []
        print("✂️ Нарезка документов на Semantic Blocks (Production)...")
        
        for doc in documents:
            blocks = self._create_semantic_blocks(doc.text)
            doc_chunks = self._pack_blocks(doc, blocks)
            all_chunks.extend(doc_chunks)
                
        print(f"✅ Создано {len(all_chunks)} чанков (Semantic Packing).")
        return all_chunks