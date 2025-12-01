import numpy as np
import json
from flask import Flask, request, render_template
import requests
from database import load_embeddings_from_db
from config import OPENAI_API_KEY, EMBEDDING_MODEL, OPENAI_EMBEDDING_URL, LLM_API_URL
from flask import jsonify

app = Flask(__name__)


# --- Pomocné funkce ---

def get_query_embedding(query):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": query, "model": EMBEDDING_MODEL}
    response = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=data)
    if response.status_code == 200:
        return np.array(response.json()["data"][0]["embedding"])
    return None


def cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)


def find_top_k_matches(query_embedding, embeddings, k=3):
    """Najde K nejlepších shod v databázi."""
    if not embeddings:
        return []

    scored_embeddings = []
    for item in embeddings:
        score = cosine_similarity(query_embedding, item["vector"])
        scored_embeddings.append((score, item))

    # Seřadit sestupně podle skóre
    scored_embeddings.sort(key=lambda x: x[0], reverse=True)

    # Vrátit top K (jen pokud je skóre aspoň trochu relevantní, např. > 0.4)
    return [item for score, item in scored_embeddings[:k] if score > 0.0]


def get_response_from_llm(context_list, query):
    # Sestavení kontextu z více chunků
    context_text = ""
    for idx, item in enumerate(context_list):
        context_text += f"\n--- ZDROJ {idx + 1}: {item['title']} ({item['title']}) ---\n"
        context_text += item['text'] + "\n"

    system_prompt = """
    Jsi nápomocný AI asistent 'Sofim' pro Studijní oddělení FIM UHK. 
    Odpovídej na otázky studentů POUZE na základě poskytnutého kontextu.
    Pokud odpověď v kontextu není, slušně řekni, že tuto informaci nemáš, nevymýšlej si.
    Odpovídej stručně, jasně a přátelsky (tykání/vykání dle dotazu, defaultně vykání).
    """

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",  # Nebo gpt-4o-mini
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Kontext:\n{context_text}\n\nDotaz studenta: {query}"}
        ],
        "temperature": 0.5
    }

    response = requests.post(LLM_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    return "Omlouvám se, momentálně nejsem ve spojení se svým mozkem (chyba API)."



# --- Routes ---

# --- API Endpoint pro moderní chat (AJAX) ---
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    query_embedding = get_query_embedding(query)
    embeddings = load_embeddings_from_db()
    best_matches = find_top_k_matches(query_embedding, embeddings, k=3)

    response_sources = []
    if best_matches:
        response_text = get_response_from_llm(best_matches, query)

        # --- NOVÉ: Vytáhneme unikátní názvy souborů ---
        # Použijeme set(), abychom neměli ten stejný soubor 3x, pokud se našly 3 chunky z jednoho PDF
        seen_sources = set()
        for match in best_matches:
            src = match.get('source', 'Neznámý zdroj')
            if src not in seen_sources:
                response_sources.append(src)
                seen_sources.add(src)
    else:
        response_text = "Bohužel k tomuto dotazu nemám v databázi žádné informace."

    # Vracíme text I zdroje
    return jsonify({"response": response_text, "sources": response_sources})


@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    query = None

    if request.method == "POST":
        query = request.form.get("query")
        if query:
            query_embedding = get_query_embedding(query)

            # Načíst DB (v produkci by se to dělalo jen jednou při startu appky a drželo v paměti)
            embeddings = load_embeddings_from_db()

            # Najít nejlepší shody (Top-3)
            best_matches = find_top_k_matches(query_embedding, embeddings, k=5)

            if best_matches:
                response = get_response_from_llm(best_matches, query)
            else:
                response = "Bohužel k tomuto dotazu nemám v databázi žádné informace. Zkuste se zeptat jinak nebo kontaktujte studijní oddělení."

    return render_template("index.html", query=query, response=response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)