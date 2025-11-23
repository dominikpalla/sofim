import pymysql
import numpy as np
import json
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True
    )

def init_db():
    """Vytvoří tabulku, pokud neexistuje (pro jistotu)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            chunk TEXT,
            embedding JSON,
            source_file VARCHAR(255)
        )
    """)
    conn.close()

# Vložit nový záznam (použije ingest.py)
def insert_embedding_to_db(title, chunk, embedding, source_file):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_json = json.dumps(embedding.tolist())
    cursor.execute(
        "INSERT INTO embeddings (title, chunk, embedding, source_file) VALUES (%s, %s, %s, %s)",
        (title, chunk, embedding_json, source_file)
    )
    conn.close()

# Načíst všechny záznamy (použije application.py)
def load_embeddings_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Načítáme i source_file pro lepší citace
    cursor.execute("SELECT id, title, chunk, embedding, source_file FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    embeddings = []
    for row in rows:
        record_id, title, chunk, embedding_str, source_file = row
        try:
            embedding_array = np.array(json.loads(embedding_str))
        except (json.JSONDecodeError, TypeError):
            continue # Přeskočíme vadné záznamy

        embeddings.append({
            "id": record_id,
            "title": title,
            "text": chunk,
            "vector": embedding_array,
            "source": source_file
        })

    return embeddings

# Vymazání databáze před novým nahráním (volitelné)
def clear_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE embeddings")
    conn.close()