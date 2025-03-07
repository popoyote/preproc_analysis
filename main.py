# main.py
from for_text import (
    read_text,
    tokenize_with_spacy,
    tokenize_with_nltk,
    remove_stopwords_spacy,
    remove_stopwords_nltk,
    lemmatize_with_spacy,
    lemmatize_with_nltk,
    extract_entities_spacy,
    extract_entities_nltk,
)
from visualisation import plot_token_comparison

# Чтение текста
text = read_text("txt/manual_juke.txt")

# Токенизация
spacy_tokens = tokenize_with_spacy(text)
nltk_tokens = tokenize_with_nltk(text)

# Удаление стоп-слов
spacy_cleaned = remove_stopwords_spacy(spacy_tokens)
nltk_cleaned = remove_stopwords_nltk(nltk_tokens)

# Лемматизация
spacy_lemmas = lemmatize_with_spacy(spacy_cleaned)
nltk_lemmas = lemmatize_with_nltk(nltk_cleaned)

# Извлечение сущностей
spacy_entities = extract_entities_spacy(text)
nltk_entities = extract_entities_nltk(text)

# Визуализация
plot_token_comparison(spacy_tokens, nltk_tokens, spacy_cleaned, nltk_cleaned)

# Вывод результатов
print("Токены (spaCy):", spacy_tokens[:10])
print("Токены (NLTK):", nltk_tokens[:10])
print("Токены без стоп-слов (spaCy):", spacy_cleaned[:10])
print("Токены без стоп-слов (NLTK):", nltk_cleaned[:10])
print("Леммы (spaCy):", spacy_lemmas[:10])
print("Леммы (NLTK):", nltk_lemmas[:10])
print("Сущности (spaCy):", spacy_entities[:5])
print("Сущности (NLTK):", nltk_entities[:5])
