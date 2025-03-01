import pymysql
import numpy as np
from flask import Flask, request, render_template
import requests
import os

os.environ["FLASK_SKIP_DOTENV"] = "1"

OPENAI_API_KEY= "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"
DB_HOST="dominikpalla.cz"
DB_NAME="pyto_1"
DB_USER="pyto.1"
DB_PASSWORD="^CiQoyGxYtO3O;zU7"

EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
LLM_API_URL = "https://api.openai.com/v1/chat/completions"

# Flask aplikace
app = Flask(__name__)

# Funkce pro připojení k databázi MySQL
def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# Funkce pro načtení embeddingů z MySQL
def load_embeddings_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT chunk, embedding FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    embeddings = []
    for row in rows:
        chunk, embedding_str = row
        embedding = np.fromstring(embedding_str.strip("[]"), sep=",")
        embeddings.append((chunk, embedding))
    return embeddings

# Funkce pro vložení embeddingu do MySQL
def insert_embedding_to_db(chunk, embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_str = str(embedding.tolist())
    cursor.execute("INSERT INTO embeddings (chunk, embedding) VALUES (%s, %s)", (chunk, embedding_str))
    conn.commit()
    conn.close()

# Získání embeddingu pro dotaz
def get_query_embedding(query):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": query,
        "model": EMBEDDING_MODEL
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        print(f"Chyba při získávání embeddingu: {response.status_code}, {response.text}")
        return None

# Vyhledání nejlepšího záznamu
def find_best_match(query_embedding, embeddings):
    best_similarity = -1
    best_chunk = None
    for chunk, embedding in embeddings:
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_chunk = chunk
    return best_chunk

# Výpočet kosinové podobnosti
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Získání odpovědi od LLM
def get_response_from_llm(context, query):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",  # Nebo použij "gpt-4"
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Na základě následujícího kontextu odpověz na dotaz:\n\nKontext: {context}\n\nDotaz: {query}"}
        ],
        "max_tokens": 500
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"Chyba při generování odpovědi: {response.status_code}, {response.text}")
        return "Chyba při generování odpovědi."

# Základní stránka
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# API endpoint pro vyhledávání
@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return render_template("index.html", error="Dotaz je vyžadován.")

    query_embedding = get_query_embedding(query)
    if not query_embedding:
        return render_template("index.html", error="Chyba při získávání embeddingu.")

    embeddings = load_embeddings_from_db()
    best_chunk = find_best_match(query_embedding, embeddings)
    if not best_chunk:
        return render_template("index.html", error="Nebyl nalezen vhodný záznam.")

    response = get_response_from_llm(best_chunk, query)
    return render_template("results.html", query=query, response=response)

# Spusť server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)