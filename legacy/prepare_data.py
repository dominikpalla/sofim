import os
import pymysql
import numpy as np
import requests

OPENAI_API_KEY= "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"
DB_HOST="dominikpalla.cz"
DB_NAME="pyto_1"
DB_USER="pyto.1"
DB_PASSWORD="^CiQoyGxYtO3O;zU7"

EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"

# Funkce pro připojení k databázi MySQL
def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# Funkce pro vložení dat do databáze
def insert_embedding_to_db(chunk, embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_str = str(embedding)
    cursor.execute("INSERT INTO embeddings (chunk, embedding) VALUES (%s, %s)", (chunk, embedding_str))
    conn.commit()
    conn.close()

# Funkce pro získání embeddingu z OpenAI API
def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": EMBEDDING_MODEL
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        raise Exception(f"Chyba při získávání embeddingu: {response.status_code}, {response.text}")

# Funkce pro rozdělení textu na chunky
def split_text_into_chunks(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# Zpracování všech souborů ve složce
def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    insert_embedding_to_db(chunk, embedding)
                    print(f"Chunk z {filename} uložen do databáze.")

# Hlavní funkce
if __name__ == "__main__":
    folder_path = "data"  # Změň na cestu ke složce s .txt soubory
    process_files_in_folder(folder_path)
    print("Zpracování souborů dokončeno.")