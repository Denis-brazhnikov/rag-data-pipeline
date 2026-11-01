import os

# Создаем папку, если ее нет
os.makedirs('data/raw', exist_ok=True)

# 1. Обычный текст
base_text = "Это тестовый документ для проверки нашего RAG пайплайна. Он содержит достаточно символов, чтобы пройти фильтрацию минимальной длины. Здесь описаны основные бизнес-процессы компании."
with open('data/raw/article_1.txt', 'w', encoding='utf-8') as f: f.write(base_text)

# 2. Точный дубликат (Exact duplicate)
with open('data/raw/article_1_duplicate.txt', 'w', encoding='utf-8') as f: f.write(base_text)

# 3. Почти дубликат (Near-duplicate - изменены 2 слова)
near_text = "Это тестовый файл для проверки нашего RAG пайплайна. Он содержит достаточно символов, чтобы пройти фильтрацию минимальной длины. Здесь описаны основные бизнес-процессы предприятия."
with open('data/raw/article_1_near_duplicate.txt', 'w', encoding='utf-8') as f: f.write(near_text)

# 4. HTML с мусором (Boilerplate)
html_text = """<!DOCTYPE html><html><head><title>Test</title><style>body{color:red}</style></head><body><header>Меню сайта</header><nav>Главная, О нас, Контакты</nav><main><h1>Заголовок статьи</h1><p>Полезный текст внутри HTML документа, который должен остаться после очистки. Убираем весь мусор.</p></main><footer>Подвал сайта 2026</footer><script>console.log('spam')</script></body></html>"""
with open('data/raw/web_page_1.html', 'w', encoding='utf-8') as f: f.write(html_text)

# 5. Второй HTML
html_text_2 = """<html><body><nav>Home | About</nav><div><h2>Другая статья</h2><p>Второй HTML документ для проверки парсера.</p></div><footer>Все права защищены.</footer></body></html>"""
with open('data/raw/web_page_2.html', 'w', encoding='utf-8') as f: f.write(html_text_2)

# 6. Markdown
md_text = "# Заголовок\n\nЭто **markdown** документ.\n\n- Список 1\n- Список 2\n\nОн полезен для проверки."
with open('data/raw/notes.md', 'w', encoding='utf-8') as f: f.write(md_text)

# 7. Шум (Noise)
noise = "!!!@@@###$$$%%%^^^&&&***((()))___+++===---,,,....////\\\\\\ ||||| 1234567890 " * 10
with open('data/raw/noise.txt', 'w', encoding='utf-8') as f: f.write(noise)

# 8. Слишком короткий текст (должен отсеяться)
with open('data/raw/short.txt', 'w', encoding='utf-8') as f: f.write("Привет!")

# 9. Еще один нормальный текст
with open('data/raw/random_article.txt', 'w', encoding='utf-8') as f: f.write("Совершенно другой текст. Он рассказывает о том, как важно правильно готовить данные для нейросетей. Мусор на входе — мусор на выходе.")

# 10. Создаем фейковый PDF для теста (текстовый файл с расширением .pdf)
# В реальности лучше положить настоящий PDF, но для теста парсера сойдет и так.
with open('data/raw/document_10.pdf', 'w', encoding='utf-8') as f: f.write("Псевдо-PDF для теста. В реальности нужно положить настоящий PDF файл.")

print("✅ 10 тестовых файлов успешно созданы в папке data/raw!")