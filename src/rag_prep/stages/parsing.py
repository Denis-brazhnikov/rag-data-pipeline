from pathlib import Path
from unstructured.partition.auto import partition
from src.rag_prep.models import SourceFile, RawElement
from src.rag_prep.config import ParserConfig

class UnstructuredParsingStage:
    def __init__(self, config: ParserConfig):
        self.config = config

    def run(self, sources: list[SourceFile]) -> list[RawElement]:
        elements = []
        
        for source in sources:
            try:
                # Используем Unstructured для семантического парсинга
                raw_elements = partition(
                    filename=str(source.path),
                    strategy=self.config.strategy
                )
                
                current_section = ""
                section_path = []
                
                for idx, el in enumerate(raw_elements):
                    el_type = el.category
                    text = el.text.strip()
                    
                    if not text:
                        continue

                    # Отслеживаем иерархию секций (слайд 39)
                    if el_type in {'Title', 'Header'}:
                        current_section = text
                        section_path = [text]
                    
                    # Генерируем стабильный ID (упрощенно)
                    element_id = f"{source.source_hash}_{idx}"
                    
                    elements.append(RawElement(
                        source_file=source,
                        element_id=element_id,
                        element_index=idx,
                        text=text,
                        element_type=el_type,
                        section=current_section,
                        section_path=section_path,
                        metadata=el.metadata.to_dict() if hasattr(el, 'metadata') else {}
                    ))
            except Exception as e:
                if self.config.fail_on_error:
                    raise e
                print(f"Ошибка парсинга {source.file_name}: {e}")
                continue
                
        return elements