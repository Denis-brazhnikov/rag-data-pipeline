from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class SourceFile(BaseModel):
    path: Path
    source: str
    file_name: str
    file_type: str
    source_hash: str
    size_bytes: int
    modified_at: datetime

class RawElement(BaseModel):
    source_file: SourceFile
    element_id: str
    element_index: int
    text: str
    element_type: str
    section: str
    section_path: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentMetadata(BaseModel):
    id: str
    source: str
    section: str
    file_name: str
    file_type: str
    source_hash: str
    text_hash: str
    parent_ids: List[str] = Field(default_factory=list)
    origin_element_ids: List[str] = Field(default_factory=list)
    lineage: Dict[str, Any] = Field(default_factory=dict)
    hierarchy: Dict[str, Any] = Field(default_factory=dict)
    quality_signals: Dict[str, Any] = Field(default_factory=dict)

class PreparedDocument(BaseModel):
    text: str
    metadata: DocumentMetadata

    # --- НОВЫЕ МОДЕЛИ ДЛЯ CHUNKING PIPELINE ---

class SemanticBlock(BaseModel):
    id: str
    text: str
    position: int
    start_char: int
    end_char: int
    token_count: int
    origin_element_ids: List[str] = Field(default_factory=list)

class ChunkMetadata(BaseModel):
    # Обязательные поля из ТЗ
    document_id: str
    position: int
    chunk_token_count: int
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    
    # Дополнительные поля из ТЗ (Lineage и Tracing)
    source: Optional[str] = None
    section: Optional[str] = None
    text_hash: Optional[str] = None
    parent_ids: List[str] = Field(default_factory=list)
    
    # Сложные структуры для attribution
    lineage: Dict[str, Any] = Field(default_factory=dict)
    hierarchy: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
    
    # Offsets для точного цитирования
    chunk_start_char: int
    chunk_end_char: int
    offset_strategy: str = "semantic_block_span"

class PreparedChunk(BaseModel):
    text: str
    metadata: ChunkMetadata