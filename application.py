import numpy as np
import json
from flask import Flask, request, render_template, jsonify
import requests
from database import load_embeddings_from_db
from config import OPENAI_API_KEY, EMBEDDING_MODEL, OPENAI_EMBEDDING_URL, LLM_API_URL

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

    # Vrátit top K
    return [item for score, item in scored_embeddings[:k] if score > 0.2]


def rewrite_query_for_search(user_query):
    """
    Přepíše dotaz uživatele tak, aby byl optimalizovaný pro sémantické vyhledávání.
    Doplní kontext, klíčová slova a synonyma.
    """
    print(f"🔄 Původní dotaz: {user_query}")

    system_prompt = """
    Jsi expertní AI pro optimalizaci vyhledávacích dotazů v univerzitní databázi (RAG).
    Tvým úkolem je přeformulovat dotaz studenta tak, aby byl co nejlepší pro sémantické vyhledávání (embeddingy).

    Zdroje obsahují:
    1. Informace o předmětech (kódy, názvy, garanti, kredity, anotace).
    2. Směrnice a vyhlášky (termíny, pravidla, omluvy).

    Pravidla:
    - Pokud dotaz zmiňuje název předmětu, přidej slova jako "předmět", "sylabus", "garant", "kredity".
    - Pokud je dotaz na směrnici, přidej formální termíny (např. "omluvenka" -> "omluva z výuky", "lékařské potvrzení").
    - Odstraň balast ("ahoj", "prosím tě", "chtěl bych vědět").
    - Výstup musí být POUZE vylepšený vyhledávací dotaz, nic jiného.
    """

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",  # Nebo gpt-4o-mini pro rychlost
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Dotaz: {user_query}"}
        ],
        "temperature": 0  # Chceme deterministický výstup
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            optimized_query = response.json()["choices"][0]["message"]["content"].strip()
            print(f"✨ Optimalizovaný dotaz: {optimized_query}")
            return optimized_query
    except Exception as e:
        print(f"⚠️ Chyba při optimalizaci dotazu: {e}")

    return user_query  # Fallback na původní dotaz


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
    Odpovídej stručně, jasně a přátelsky.
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
        else:
            return "Omlouvám se, chyba API."
    except Exception:
        return "Omlouvám se, chyba komunikace."


# --- Routes ---

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    # 1. KROK: Přeformulování dotazu pro lepší vyhledávání
    search_query = rewrite_query_for_search(user_query)

    # 2. KROK: Hledání v DB pomocí VYLEPŠENÉHO dotazu
    query_embedding = get_query_embedding(search_query)
    embeddings = load_embeddings_from_db()
    best_matches = find_top_k_matches(query_embedding, embeddings, k=3)

    response_sources = []
    response_text = ""

    if best_matches:
        # 3. KROK: Odpověď generujeme na původní dotaz uživatele (aby to znělo přirozeně),
        # ale s kontextem nalezeným pomocí vylepšeného dotazu.
        response_text = get_response_from_llm(best_matches, user_query)

        seen_sources = set()
        for match in best_matches:
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