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


# --- STANDARDNÍ ČTENÍ (PRO CHATBOTA) ---
# Chatbot vždy čte z tabulky 'embeddings' (bez ohledu na to, co se děje na pozadí)
def load_embeddings_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Zkontrolujeme, jestli tabulka existuje (pro první spuštění)
    cursor.execute("SHOW TABLES LIKE 'embeddings'")
    if not cursor.fetchone():
        conn.close()
        return []

    cursor.execute("SELECT id, title, chunk, embedding, source_file FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    embeddings = []
    for row in rows:
        record_id, title, chunk, embedding_str, source_file = row
        try:
            embedding_array = np.array(json.loads(embedding_str))
        except (json.JSONDecodeError, TypeError):
            continue

        embeddings.append({
            "id": record_id,
            "title": title,
            "text": chunk,
            "vector": embedding_array,
            "source": source_file
        })

    return embeddings


# --- LOGIKA PRO ZERO-DOWNTIME INGEST ---

def init_next_table():
    """Vytvoří prázdnou stínovou tabulku 'embeddings_next'."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Smažeme případné pozůstatky z minulého nepovedeného běhu
    cursor.execute("DROP TABLE IF EXISTS embeddings_next")

    # 2. Vytvoříme novou tabulku se stejnou strukturou
    cursor.execute("""
        CREATE TABLE embeddings_next (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            chunk TEXT,
            embedding JSON,
            source_file VARCHAR(255)
        )
    """)
    conn.close()


def insert_into_next_table(title, chunk, embedding, source_file):
    """Vkládá data do STÍNOVÉ tabulky."""
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_json = json.dumps(embedding.tolist())
    cursor.execute(
        "INSERT INTO embeddings_next (title, chunk, embedding, source_file) VALUES (%s, %s, %s, %s)",
        (title, chunk, embedding_json, source_file)
    )
    conn.close()


def swap_tables_atomic():
    """Provede bleskové prohození tabulek."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Zkontrolujeme, jestli existuje ostrá tabulka 'embeddings'
    cursor.execute("SHOW TABLES LIKE 'embeddings'")
    exists = cursor.fetchone()

    if exists:
        # Pokud existuje, provedeme rotaci: Live -> Backup, Next -> Live
        cursor.execute("DROP TABLE IF EXISTS embeddings_backup")
        cursor.execute("RENAME TABLE embeddings TO embeddings_backup, embeddings_next TO embeddings")
        cursor.execute("DROP TABLE embeddings_backup")
    else:
        # Pokud je to úplně první běh, jen přejmenujeme Next -> Live
        cursor.execute("RENAME TABLE embeddings_next TO embeddings")

    conn.close()