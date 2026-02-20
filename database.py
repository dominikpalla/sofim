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


# --- INICIALIZACE STRUKTURY DATABÁZE (PRO ADMIN PANEL) ---

def init_db_schema():
    """Vytvoří nezbytné tabulky pro chod admin panelu a sledování indexace, pokud neexistují."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Tabulka pro URL adresy z crawleru
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crawler_urls (
            id INT AUTO_INCREMENT PRIMARY KEY,
            url VARCHAR(500) NOT NULL UNIQUE,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Tabulka pro sledování času a průběhu aktualizací
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sync_status (
            sync_type VARCHAR(10) PRIMARY KEY,
            last_updated DATETIME,
            status VARCHAR(50),
            total_items INT DEFAULT 0,
            processed_items INT DEFAULT 0,
            last_error TEXT
        )
    """)

    # Založíme výchozí stavy, ignoruje se, pokud už záznamy existují
    cursor.execute("INSERT IGNORE INTO sync_status (sync_type, status) VALUES ('WEB', 'idle'), ('CSV', 'idle')")
    conn.commit()
    conn.close()


# --- FUNKCE PRO SLEDOVÁNÍ PRŮBĚHU INDEXACE ---

def get_sync_status():
    """Vrátí aktuální stavy aktualizací pro admin panel."""
    init_db_schema()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sync_type, last_updated, status, total_items, processed_items, last_error FROM sync_status")
    rows = cursor.fetchall()
    conn.close()

    return {
        row[0]: {
            "last_updated": row[1],
            "status": row[2],
            "total_items": row[3],
            "processed_items": row[4],
            "last_error": row[5]
        } for row in rows
    }


def set_sync_status(sync_type, status, total=0):
    """Při startu nastaví status, vynuluje progress a chyby. Při úspěchu uloží čas."""
    init_db_schema()
    conn = get_db_connection()
    cursor = conn.cursor()

    if status == 'running':
        cursor.execute(
            "UPDATE sync_status SET status = 'running', total_items = %s, processed_items = 0, last_error = NULL WHERE sync_type = %s",
            (total, sync_type)
        )
    elif status == 'success':
        cursor.execute(
            "UPDATE sync_status SET status = 'idle', last_updated = NOW() WHERE sync_type = %s",
            (sync_type,)
        )
    else:
        cursor.execute(
            "UPDATE sync_status SET status = %s WHERE sync_type = %s",
            (status, sync_type)
        )

    conn.commit()
    conn.close()


def update_sync_progress(sync_type, processed_count):
    """Aktualizuje počet zpracovaných položek pro progress bar."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE sync_status SET processed_items = %s WHERE sync_type = %s", (processed_count, sync_type))
    conn.commit()
    conn.close()


def log_sync_error(sync_type, error_msg):
    """Zapíše chybovou hlášku do databáze (zřetězí k existujícím)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sync_status SET last_error = CONCAT(IFNULL(last_error, ''), %s, '\n') WHERE sync_type = %s",
        (error_msg, sync_type)
    )
    conn.commit()
    conn.close()


# --- STANDARDNÍ ČTENÍ (PRO CHATBOTA) ---

def load_embeddings_from_db():
    """Chatbot vždy čte z tabulky 'embeddings' bez ohledu na to, co se děje na pozadí."""
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


# --- LOGIKA PRO ZERO-DOWNTIME INGEST (VČETNĚ ČÁSTEČNÉHO UPDATE) ---

def prepare_next_table_for_update(mode="all"):
    """Vytvoří stínovou tabulku - buď prázdnou, nebo jako kopii živé pro částečný update."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Zkontrolujeme, jestli už existuje hlavní "ostrá" tabulka
    cursor.execute("SHOW TABLES LIKE 'embeddings'")
    live_exists = cursor.fetchone()

    # Smažeme případné pozůstatky z minulého nepovedeného běhu
    cursor.execute("DROP TABLE IF EXISTS embeddings_next")

    if not live_exists or mode == "all":
        # Čistý stůl (Kompletní reload nebo úplně první spuštění databáze)
        cursor.execute("""
            CREATE TABLE embeddings_next (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255),
                chunk TEXT,
                embedding JSON,
                source_file VARCHAR(255)
            )
        """)
    else:
        # Částečný update: Vytvoříme stínovou tabulku jako přesnou kopii té stávající
        cursor.execute("CREATE TABLE embeddings_next LIKE embeddings")
        cursor.execute("INSERT INTO embeddings_next SELECT * FROM embeddings")

        # Nyní vymažeme z kopie ta data, která se chystáme nahradit čerstvými
        if mode == "web":
            # Aktualizujeme jen weby, takže smažeme vše, co NENÍ STAG Export
            cursor.execute("DELETE FROM embeddings_next WHERE source_file != 'STAG Export'")
        elif mode == "csv":
            # Aktualizujeme jen STAG CSV, takže smažeme záznamy ze STAGu (ponecháme weby)
            cursor.execute("DELETE FROM embeddings_next WHERE source_file = 'STAG Export'")

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