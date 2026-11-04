import pytest
from src.rag_prep.utils import stable_id
from src.rag_prep.models import SemanticBlock

def test_stable_id_determinism():
    """Проверяем, что ID всегда одинаковый для одних и тех же данных"""
    id1 = stable_id("doc1", 0, "Текст чанка")
    id2 = stable_id("doc1", 0, "Текст чанка")
    id3 = stable_id("doc1", 1, "Текст чанка")
    
    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 24

def test_semantic_block_model():
    """Проверяем валидацию модели SemanticBlock"""
    block = SemanticBlock(
        id="123", text="Тест", position=0, start_char=0, end_char=4, token_count=1
    )
    assert block.text == "Тест"