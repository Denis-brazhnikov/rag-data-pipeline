import hashlib
import os
from bs4 import BeautifulSoup
import pdfplumber

def generate_document_id(text: str) -> str:
    """Создает уникальный хеш на основе текста документа."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_html(html_content: str) -> str:
    """Удаляет HTML-теги, скрипты и стили (удаление boilerplate)."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Удаляем скрипты, стили, навигацию, футеры
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    
    text = soup.get_text(separator=' ')
    # Очищаем от лишних пробелов и переносов
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def extract_text_from_pdf(file_path: str) -> str:
    """Извлекает текст из PDF файла."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Читает обычный текстовый файл или Markdown."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_document(file_path: str, file_type: str) -> str:
    """Маршрутизатор парсеров в зависимости от формата."""
    if file_type == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_type in [".html", ".htm"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return clean_html(f.read())
    elif file_type in [".txt", ".md"]:
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {file_type}")