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


def is_subject_code(word):
    """
    Rozpozná, zda slovo vypadá jako kód předmětu (např. ALG1, OA1, KP/ALG).
    Vyloučí běžná slova jako 'kontakt', 'katedra', 'na'.
    """
    # Musí mít 2-8 znaků
    if not (2 <= len(word) <= 8):
        return False

    # Musí obsahovat alespoň jedno velké písmeno nebo číslo (pokud je zadáno velkými)
    # Ale my dostaneme 'word' už z tokenizace, takže musíme být opatrní.

    # Seznam zakázaných slov (běžná slova, která by se mohla splést s kódy)
    stopwords = {'pro', 'kde', 'kdy', 'jak', 'co', 'na', 'do', 'se', 'ze', 'ke', 've',
                 'test', 'info', 'data', 'stag', 'fim', 'uhk', 'pan', 'pani',
                 'doc', 'prof', 'ing', 'mgr', 'bc', 'phd', 'kontakt', 'vedouci'}

    if word.lower() in stopwords:
        return False

    # Musí obsahovat alespoň jedno písmeno (ne jen čísla, i když na FIMu jsou i kódy s čísly)
    # Ale hlavně: Kódy bývají 'OA1', 'ALG', '4IT101'.
    # Pokud je to jen "Dominik", tak to projde jako validní slovo, ale my chceme jen KÓDY.

    # Zkusíme přísnější pravidlo:
    # 1. Obsahuje číslo? (OA1, 4IT) -> JASNÝ KÓD
    if any(char.isdigit() for char in word):
        return True

    # 2. Je to celé velkými písmeny a má to 2-5 znaků? (ALG, ZPRO) -> ASI KÓD
    # (Tady spoléháme na to, že uživatel napíše ALG, ne alg. Pokud napíše alg, boostneme to taky, nevadí).
    if word.isalpha() and len(word) <= 5:
        return True

    return False


def find_top_k_matches(query_embedding, embeddings, query_text, k=3):
    """Najde K nejlepších shod s CHYTRÝM boostem pro kódy."""
    if not embeddings:
        return []

    # Rozbijeme dotaz na slova. Zachováme původní velikost písmen pro detekci kódů!
    raw_tokens = re.findall(r'\b\w+\b', query_text)

    scored_embeddings = []
    for item in embeddings:
        # 1. Základní skóre (Sémantika)
        score = cosine_similarity(query_embedding, item["vector"])

        # 2. Smart Keyword Boost
        item_title = item["title"]  # Původní title s velkými písmeny

        boost = 0.0
        for token in raw_tokens:
            # Aplikujeme boost JENOM pokud to vypadá jako kód předmětu
            if is_subject_code(token):
                # Hledáme přesnou shodu kódu v titulku (case-insensitive, ale boundary-sensitive)
                # \bTOKEN\b zajistí, že ALG nenajde v "Algebra", ale najde v "(ALG)"
                if re.search(r'\b' + re.escape(token) + r'\b', item_title, re.IGNORECASE):
                    # Je to kód a je v nadpisu! Boost!
                    print(f"🚀 Boostuji: {item['title']} kvůli kódu '{token}'")
                    boost += 0.5  # Masivní boost

        final_score = score + boost
        scored_embeddings.append((final_score, item))

    scored_embeddings.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored_embeddings[:k] if score > 0.2]


def rewrite_query_for_search(user_query):
    """LLM přepis dotazu."""
    system_prompt = """
    Jsi expertní AI pro optimalizaci vyhledávacích dotazů v univerzitní databázi (RAG).

    Pravidla:
    - Pokud dotaz obsahuje zkratku (např. OA1, ZPRO), ZACHOVEJ JI v přesném znění!
    - Pokud je dotaz obecný ("kdy je zápis"), rozšiř ho o klíčová slova ("harmonogram", "termín").
    - Odstraň balast.
    """

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
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
        if response.status_code == 200: return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"Chyba API (Status {response.status_code}): {response.text}"
    return f"Chyba API (Status {response.status_code}): {response.text}"


# --- Routes ---

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query: return jsonify({"error": "Empty query"}), 400

    search_query = rewrite_query_for_search(user_query)
    query_embedding = get_query_embedding(search_query)
    embeddings = load_embeddings_from_db()

    # Hledání s chytrým boostem (předáváme původní dotaz pro detekci kódů)
    best_matches = find_top_k_matches(query_embedding, embeddings, user_query, k=3)

    response_sources = []
    response_text = ""

    if best_matches:
        response_text = get_response_from_llm(best_matches, user_query)
        seen = set()
        for match in best_matches:
            src = match.get('title') or match.get('source', 'Zdroj')
            if src not in seen:
                response_sources.append(src)
                seen.add(src)
    else:
        response_text = "Bohužel k tomuto dotazu nemám v databázi žádné informace."

    return jsonify({"response": response_text, "sources": response_sources})


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)