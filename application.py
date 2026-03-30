import numpy as np
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import requests
import threading
from database import load_embeddings_from_db, get_db_connection, get_sync_status
from ingest import run_ingest
from config import OPENAI_API_KEY, EMBEDDING_MODEL, OPENAI_EMBEDDING_URL, LLM_API_URL
import re

app = Flask(__name__)

app.secret_key = "super_tajny_klic_pro_session"  # Tajný klíč pro session (v produkci dej do .env)
ADMIN_PASSWORD = "studijkojede"


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
    if not (2 <= len(word) <= 8):
        return False

    stopwords = {'pro', 'kde', 'kdy', 'jak', 'co', 'na', 'do', 'se', 'ze', 'ke', 've',
                 'test', 'info', 'data', 'stag', 'fim', 'uhk', 'pan', 'pani',
                 'doc', 'prof', 'ing', 'mgr', 'bc', 'phd', 'kontakt', 'vedouci'}

    if word.lower() in stopwords:
        return False

    if any(char.isdigit() for char in word):
        return True

    if word.isalpha() and len(word) <= 5:
        return True

    return False


def find_top_k_matches(query_embedding, embeddings, query_text, k=8):
    """Najde K nejlepších shod s CHYTRÝM boostem pro kódy i klíčová slova."""
    if not embeddings:
        return []

    # Očištění dotazu na jednotlivá smysluplná slova
    raw_tokens = [t for t in re.findall(r'\b\w+\b', query_text) if len(t) > 3]

    scored_embeddings = []
    for item in embeddings:
        score = cosine_similarity(query_embedding, item["vector"])
        item_title = item["title"]
        item_text = item["text"]

        boost = 0.0
        for token in raw_tokens:
            # Menší boost pro shodu jmen nebo klíčových slov v textu/názvu
            if token.lower() in item_title.lower() or token.lower() in item_text.lower():
                boost += 0.05

                # Tvůj původní masivní boost pro kódy předmětů
            if is_subject_code(token):
                if re.search(r'\b' + re.escape(token) + r'\b', item_title, re.IGNORECASE):
                    # print(f"🚀 Boostuji: {item['title']} kvůli kódu '{token}'")
                    boost += 0.5

        final_score = score + boost
        scored_embeddings.append((final_score, item))

    scored_embeddings.sort(key=lambda x: x[0], reverse=True)
    # Snížila jsem hranici na 0.15, protože při k=8 chceme pustit i širší kontext
    return [item for score, item in scored_embeddings[:k] if score > 0.15]


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
        url_info = item.get('url', '')

        # Důležité: Přidáváme jasný index zdroje pro LLM
        context_text += f"\n--- [ZDROJ_ID: {idx}] {title_info} (Soubor: {source_info}) ---\n"
        if url_info:
            context_text += f"Odkaz na zdroj: {url_info}\n"
        context_text += item['text'] + "\n"

    system_prompt = """
    Jsi nápomocný AI asistent 'Sofim' pro Studijní oddělení FIM UHK. 
    Odpovídej na otázky studentů POUZE na základě poskytnutého kontextu.
    Pokud odpověď v kontextu není, slušně řekni, že tuto informaci nemáš.

    MUSÍŠ odpovědět ve validním JSON formátu s následující strukturou:
    {
      "odpoved": "Tvoje kompletní odpověď pro studenta formátovaná v Markdownu...",
      "pouzite_zdroje": [0, 2] // Pole obsahující POUZE ID zdrojů (čísla), ze kterých jsi skutečně čerpal. Pokud nemáš info, vrať prázdné pole [].
    }
    """

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Kontext:\n{context_text}\n\nDotaz studenta: {query}"}
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}  # Vynutí JSON výstup
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            import json
            content = response.json()["choices"][0]["message"]["content"]
            try:
                # Rozparsujeme JSON od GPT
                parsed = json.loads(content)
                return {
                    "text": parsed.get("odpoved", "Omlouvám se, ale nepodařilo se mi vygenerovat smysluplnou odpověď."),
                    "used_indices": parsed.get("pouzite_zdroje", [])
                }
            except json.JSONDecodeError:
                return {"text": content, "used_indices": []}
    except Exception as e:
        return {"text": f"Chyba API: {str(e)}", "used_indices": []}

    return {"text": f"Chyba API (Status {response.status_code})", "used_indices": []}


# --- Routes pro Chatbota ---

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query: return jsonify({"error": "Empty query"}), 400

    search_query = rewrite_query_for_search(user_query)
    query_embedding = get_query_embedding(search_query)
    embeddings = load_embeddings_from_db()

    # Zvýšené K na 8 dokumentů
    best_matches = find_top_k_matches(query_embedding, embeddings, user_query, k=8)

    response_sources = []
    response_text = ""

    if best_matches:
        llm_result = get_response_from_llm(best_matches, user_query)
        response_text = llm_result["text"]
        used_indices = llm_result["used_indices"]

        seen = set()
        # Projdeme jen ty indexy, které GPT vrátilo jako reálně použité
        for idx in used_indices:
            # Kontrola, jestli si GPT nevymyslelo index mimo rozsah
            if isinstance(idx, int) and 0 <= idx < len(best_matches):
                match = best_matches[idx]
                src_name = match.get('title') or match.get('source', 'Zdroj')
                src_url = match.get('url', '')

                if src_name not in seen:
                    response_sources.append({
                        "name": src_name,
                        "url": src_url
                    })
                    seen.add(src_name)
    else:
        response_text = "Bohužel k tomuto dotazu nemám v databázi žádné informace."

    return jsonify({"response": response_text, "sources": response_sources})


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


# --- Routes pro Admin Panel ---

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if session.get("logged_in"):
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        password = request.form.get("password")
        if password == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            return render_template("admin_login.html", error="Špatné heslo!")

    return render_template("admin_login.html")


@app.route("/admin/dashboard", methods=["GET", "POST"])
def admin_dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == "POST":
        new_url = request.form.get("new_url")
        if new_url:
            try:
                cursor.execute("INSERT INTO crawler_urls (url) VALUES (%s)", (new_url,))
                conn.commit()
            except:
                pass  # Ignorujeme duplikáty

    cursor.execute("SELECT id, url FROM crawler_urls")
    urls = cursor.fetchall()
    conn.close()

    # Získáme aktuální stav aktualizací pro zobrazení na dashboardu
    status_data = get_sync_status()

    return render_template("admin_dashboard.html", urls=urls, status_data=status_data)


@app.route("/admin/delete/<int:url_id>")
def admin_delete_url(url_id):
    if not session.get("logged_in"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM crawler_urls WHERE id = %s", (url_id,))
    conn.commit()
    conn.close()

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/api/status")
def admin_api_status():
    """Vrací aktuální stav indexace jako JSON pro AJAX polling ve frontendu."""
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify(get_sync_status())


@app.route("/admin/trigger_sync/<mode>")
def admin_trigger_sync(mode):
    """Spustí ingest na pozadí jako asynchronní vlákno."""
    if not session.get("logged_in"):
        return redirect(url_for("admin_login"))

    status_data = get_sync_status()

    # Zkontrolujeme, jestli už indexace zrovna neběží
    is_running = any(data['status'] == 'running' for data in status_data.values())

    if mode in ["all", "web", "csv"] and not is_running:
        # Pustíme to na pozadí, ať tě to nezdržuje
        thread = threading.Thread(target=run_ingest, args=(mode,))
        thread.daemon = True
        thread.start()

    # Hned se vrátíme na dashboard, kde se chytí AJAX a ukáže ti hezký progress
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/logout")
def admin_logout():
    session.pop("logged_in", None)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)