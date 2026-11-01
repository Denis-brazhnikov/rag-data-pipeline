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