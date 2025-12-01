import numpy as np
import json
from flask import Flask, request, render_template, jsonify
import requests
from database import load_embeddings_from_db
from config import OPENAI_API_KEY, EMBEDDING_MODEL, OPENAI_EMBEDDING_URL, LLM_API_URL
import re

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


def find_top_k_matches(query_embedding, embeddings, query_text, k=3):
    """
    Najde K nejlepších shod s podporou Keyword Boostingu.
    Řeší problém, kdy 'OA1' a 'OA2' mají podobný embedding, ale uživatel chce přesně jeden.
    """
    if not embeddings:
        return []

    # Rozbijeme dotaz na slova (tokeny) pro keyword search
    # ZMĚNA: Povolíme i slova od 2 znaků (např. "AJ", "TV", "C#")
    # Používáme r'\w+' což bere alfanumerické znaky
    query_tokens = set(word.lower() for word in re.findall(r'\w+', query_text) if len(word) >= 2)

    scored_embeddings = []
    for item in embeddings:
        # 1. Základní skóre (Cosine Similarity - Sémantika)
        score = cosine_similarity(query_embedding, item["vector"])

        # 2. Keyword Boost (Tvrdá shoda kódů)
        item_title_lower = item["title"].lower()

        boost = 0.0
        for token in query_tokens:
            # Regex \bTOKEN\b zajistí, že najdeme "OA1" i v textu "(OA1)" nebo "OA1,",
            # ale nenajdeme ho v "OA12" (což je jiné).
            # Závorka '(' se pro regex chová jako hranice slova, takže to funguje perfektně.
            if re.search(r'\b' + re.escape(token) + r'\b', item_title_lower):
                boost += 0.4  # Velký boost! (0.4 je ve světě embeddingů hodně)

        final_score = score + boost
        scored_embeddings.append((final_score, item))

    # Seřadit sestupně podle finálního skóre
    scored_embeddings.sort(key=lambda x: x[0], reverse=True)

    # Vrátit top K
    # Práh 0.2 stačí, boostnuté dokumenty budou mít třeba 1.1, takže projdou snadno
    return [item for score, item in scored_embeddings[:k] if score > 0.2]


def rewrite_query_for_search(user_query):
    """LLM přepis dotazu pro lepší vyhledávání."""
    system_prompt = """
    Jsi expertní AI pro optimalizaci vyhledávacích dotazů v univerzitní databázi (RAG).
    Tvým úkolem je přeformulovat dotaz studenta tak, aby byl co nejlepší pro sémantické vyhledávání.

    Zdroje obsahují:
    1. Informace o předmětech (kódy např. ALG1, OA1, ZPRO; názvy, garanti, kredity).
    2. Směrnice a vyhlášky.

    Pravidla:
    - Pokud dotaz obsahuje zkratku předmětu (např. OA1), MUSÍŠ ji zachovat v přesném znění!
    - Rozšiř dotaz o synonyma (např. "kdy odevzdat" -> "termín odevzdání").
    - Odstraň zdvořilostní fráze.
    """

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Dotaz: {user_query}"}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    return user_query


def get_response_from_llm(context_list, query):
    context_text = ""
    for idx, item in enumerate(context_list):
        source_info = item.get('source', 'Neznámý soubor')
        title_info = item.get('title', 'Bez názvu')
        context_text += f"\n--- ZDROJ {idx + 1}: {title_info} (Soubor: {source_info}) ---\n"
        context_text += item['text'] + "\n"

    system_prompt = """
    Jsi nápomocný AI asistent 'Sofim' pro Studijní oddělení FIM UHK. 
    Odpovídej na otázky studentů POUZE na základě poskytnutého kontextu.
    Pokud odpověď v kontextu není, slušně řekni, že tuto informaci nemáš.
    Odpovídej stručně, jasně a přátelsky. Používej formátování pro lepší čitelnost.
    """

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Kontext:\n{context_text}\n\nDotaz studenta: {query}"}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return "Omlouvám se, chyba API."


# --- Routes ---

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    # 1. LLM Rewrite (Sémantika)
    search_query = rewrite_query_for_search(user_query)

    # 2. Embedding
    query_embedding = get_query_embedding(search_query)
    embeddings = load_embeddings_from_db()

    # 3. Hybridní Search (Sémantika + Regex Boost pro zkratky)
    # Posíláme 'search_query', protože LLM tam tu zkratku zachová/zvýrazní
    best_matches = find_top_k_matches(query_embedding, embeddings, search_query, k=3)

    response_sources = []
    response_text = ""

    if best_matches:
        response_text = get_response_from_llm(best_matches, user_query)

        seen_sources = set()
        for match in best_matches:
            # Prioritně zobrazujeme Title (např. "Předmět: ...")
            source_to_show = match.get('title')
            if not source_to_show:
                source_to_show = match.get('source', 'Neznámý zdroj')

            if source_to_show and source_to_show not in seen_sources:
                response_sources.append(source_to_show)
                seen_sources.add(source_to_show)
    else:
        response_text = "Bohužel k tomuto dotazu nemám v databázi žádné informace. Zkuste se zeptat jinak nebo kontaktujte studijní oddělení."

    return jsonify({"response": response_text, "sources": response_sources})


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)