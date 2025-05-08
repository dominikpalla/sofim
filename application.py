import pymysql
import numpy as np
import json
from flask import Flask, request, render_template, redirect, url_for
import requests
import os
from database import get_db_connection, insert_embedding_to_db, load_embeddings_from_db, update_embedding_in_db, \
    delete_embedding_from_db
from config import OPENAI_API_KEY, EMBEDDING_MODEL, OPENAI_API_URL, LLM_API_URL

app = Flask(__name__)



# Získání embeddingu pro dotaz
def get_query_embedding(query):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": query, "model": EMBEDDING_MODEL}
    response = requests.post(OPENAI_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return np.array(response.json()["data"][0]["embedding"])  # ✅ Ujistíme se, že výstup je `numpy.array`

    return None


# Výpočet kosinové podobnosti
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# Vyhledání nejlepšího záznamu
def find_best_match(query_embedding, embeddings):
    valid_embeddings = [(e[0], e[1], e[2], e[3]) for e in embeddings if
                        len(e[3]) > 0]  # ✅ Filtrování prázdných embeddingů
    if not valid_embeddings:
        return None
    return max(valid_embeddings, key=lambda x: cosine_similarity(query_embedding, x[3]),
               default=None)  # ✅ Porovnáváme pouze `numpy.array`


# Získání odpovědi od LLM
def get_response_from_llm(context, query):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "Jsi chatbot studijního oddělení a pomáháš studentům s odpovědí na dotaz, pokud k němu máš potřebné informace."},
            {"role": "user",
             "content": f"Na základě následujícího kontextu odpověz na dotaz:\n\nKontext: {context}\n\nDotaz: {query}"}
        ],
        "max_tokens": 500
    }
    response = requests.post(LLM_API_URL, headers=headers, json=data)
    return response.json()["choices"][0]["message"][
        "content"].strip() if response.status_code == 200 else "Chyba při generování odpovědi."


# Hlavní stránka s chatem
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form.get("query")
        query_embedding = get_query_embedding(query)
        embeddings = load_embeddings_from_db()
        best_match = find_best_match(query_embedding, embeddings)

        response = get_response_from_llm(best_match[2], query) if best_match else "Nebyl nalezen vhodný kontext."
        return render_template("index.html", query=query, response=response)
    return render_template("index.html")


# Administrace pro nahrávání textů
@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        title = request.form.get("title")
        text = request.form.get("text")
        embedding = get_query_embedding(text)

        if embedding is None:
            return render_template("admin.html", embeddings=load_embeddings_from_db(),
                                   error="Chyba: Nepodařilo se získat embedding.")

        insert_embedding_to_db(title, text, embedding)

    embeddings = load_embeddings_from_db()
    return render_template("admin.html", embeddings=embeddings)


# Aktualizace záznamu
@app.route("/admin/edit/<int:record_id>", methods=["GET", "POST"])
def edit_record(record_id):
    if request.method == "POST":
        title = request.form.get("title")
        text = request.form.get("text")
        update_embedding_in_db(record_id, title, text)
        return redirect(url_for("admin"))

    embeddings = load_embeddings_from_db()
    record = next((e for e in embeddings if e[0] == record_id), None)
    return render_template("edit.html", record=record)


# Smazání záznamu
@app.route("/admin/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    delete_embedding_from_db(record_id)
    return redirect(url_for("admin"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)