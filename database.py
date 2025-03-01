import pymysql
import numpy as np
import json
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

def get_db_connection():
    return pymysql.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, autocommit=True)

# Vložit nový záznam
def insert_embedding_to_db(title, chunk, embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_json = json.dumps(embedding.tolist())  # ✅ Ukládáme embedding jako JSON string
    cursor.execute("INSERT INTO embeddings (title, chunk, embedding) VALUES (%s, %s, %s)", (title, chunk, embedding_json))
    conn.close()

# Načíst všechny záznamy
def load_embeddings_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, chunk, embedding FROM embeddings")  # ✅ Přidán `embedding`
    rows = cursor.fetchall()
    conn.close()

    embeddings = []
    for row in rows:
        record_id, title, chunk, embedding_str = row
        try:
            embedding_array = np.array(json.loads(embedding_str))  # ✅ Opraveno: Převod JSON stringu na numpy array
        except (json.JSONDecodeError, TypeError):
            embedding_array = np.array([])  # ✅ Bezpečná fallback hodnota

        embeddings.append((record_id, title, chunk, embedding_array))

    return embeddings

# Aktualizace záznamu
def update_embedding_in_db(record_id, title, chunk):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE embeddings SET title=%s, chunk=%s WHERE id=%s", (title, chunk, record_id))
    conn.close()

# Smazání záznamu
def delete_embedding_from_db(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM embeddings WHERE id=%s", (record_id,))
    conn.close()