import unicodedata
import spacy
from src.rag_prep.models import RawElement
from src.rag_prep.config import NormalizationConfig

class TextNormalizationStage:
    def __init__(self, config: NormalizationConfig):
        self.config = config
        # Инициализируем облегченную модель spaCy (Слайд 72)
        self.nlp = spacy.blank(self.config.spacy_language)
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')

    def run(self, elements: list[RawElement]) -> list[RawElement]:
        normalized_elements = []
        
        for element in elements:
            text = element.text
            
            # 1. Unicode-нормализация (Слайд 73)
            text = unicodedata.normalize(self.config.unicode_form, text)
            
            # 2. Работа с регистром (Слайд 74)
            if self.config.lowercase:
                text = text.lower()
                
            # 3. Статистика через spaCy (Слайд 75)
            doc = self.nlp(text)
            sentence_count = sum(1 for _ in doc.sents)
            token_count = len(doc)
            
            # Сохраняем статистику в metadata
            if "quality_signals" not in element.metadata:
                element.metadata["quality_signals"] = {}
            element.metadata["quality_signals"]["sentence_count"] = sentence_count
            element.metadata["quality_signals"]["token_count"] = token_count
            
            element.text = text
            normalized_elements.append(element)
            
        return normalized_elements