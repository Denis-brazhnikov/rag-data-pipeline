import hashlib
from src.rag_prep.models import RawElement, PreparedDocument, DocumentMetadata
from src.rag_prep.config import StructuringConfig

class StructuringStage:
    def __init__(self, config: StructuringConfig):
        self.config = config

    def run(self, elements: list[RawElement]) -> list[PreparedDocument]:
        documents = []
        
        for element in elements:
            # Генерируем стабильные хеши (Слайд 100)
            text_hash = hashlib.sha256(element.text.encode('utf-8')).hexdigest()
            doc_id = f"{element.source_file.source_hash}_{text_hash[:8]}"
            
            # Формируем Lineage (Слайд 102)
            lineage = {
                "source_id": element.source_file.source_hash,
                "source_hash": element.source_file.source_hash,
                "origin_element_ids": [element.element_id],
                "element_range": [element.element_index, element.element_index],
                "pipeline_stage": "prepared_document"
            }
            
            # Формируем Hierarchy (Слайд 102)
            hierarchy = {
                "section_path": element.section_path,
                "section_depth": len(element.section_path),
                "document_order": element.element_index
            }
            
            # Собираем метадату по строгой Pydantic-модели
            metadata = DocumentMetadata(
                id=doc_id,
                source=element.source_file.source,
                section=element.section,
                file_name=element.source_file.file_name,
                file_type=element.source_file.file_type,
                source_hash=element.source_file.source_hash,
                text_hash=text_hash,
                parent_ids=[], 
                origin_element_ids=[element.element_id],
                lineage=lineage,
                hierarchy=hierarchy,
                quality_signals=element.metadata.get("quality_signals", {})
            )
            
            # Создаем итоговый документ
            doc = PreparedDocument(text=element.text, metadata=metadata)
            documents.append(doc)
            
        return documents