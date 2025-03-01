import pymysql
import numpy as np
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

def get_db_connection():
    return pymysql.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)

def insert_embedding_to_db(chunk, embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO embeddings (chunk, embedding) VALUES (%s, %s)", (chunk, str(embedding)))
    conn.commit()
    conn.close()

def load_embeddings_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT chunk, embedding FROM embeddings")
    rows = cursor.fetchall()
    conn.close()
    return [(row[0], np.fromstring(row[1].strip("[]"), sep=",")) for row in rows]