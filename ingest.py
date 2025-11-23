import os
import json
import requests
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import docx
from pypdf import PdfReader
from config import OPENAI_API_KEY, GOOGLE_DRIVE_FOLDER_ID, GOOGLE_CREDENTIALS_FILE, EMBEDDING_MODEL, \
    OPENAI_EMBEDDING_URL
from database import insert_embedding_to_db, clear_database, init_db


# --- 1. Připojení ke Google Disku ---
def get_drive_service():
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        print(f"❌ Chyba: Soubor {GOOGLE_CREDENTIALS_FILE} nenalezen.")
        return None
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly'])
    return build('drive', 'v3', credentials=creds)


def process_file_content(service, file_item):
    print(f"  📄 Stahuji soubor: {file_item['name']}...")

    supported_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']

    # Kontrola, zda jde o podporovaný typ
    is_supported = file_item['mimeType'] in supported_types or file_item['name'].endswith(('.pdf', '.docx'))

    if not is_supported:
        return None

    try:
        request = service.files().get_media(fileId=file_item['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        fh.seek(0)
        text = ""

        if file_item['name'].endswith('.docx'):
            doc = docx.Document(fh)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])

        elif file_item['name'].endswith('.pdf'):
            reader = PdfReader(fh)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        if text:
            return {"filename": file_item['name'], "text": text}

    except Exception as e:
        print(f"⚠️ Chyba při stahování {file_item['name']}: {e}")

    return None


def get_files_recursive(service, folder_id):
    results_list = []
    page_token = None

    while True:
        try:
            response = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token
            ).execute()
        except Exception as e:
            print(f"⚠️ Chyba při listování složky: {e}")
            break

        items = response.get('files', [])

        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                print(f"📂 Vstupuji do podsložky: {item['name']}")
                results_list.extend(get_files_recursive(service, item['id']))
            else:
                processed_file = process_file_content(service, item)
                if processed_file:
                    results_list.append(processed_file)

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return results_list


# --- 2. Inteligentní Chunking ---
def semantic_chunking(text, filename):
    print(f"🧠 Sémantické řezání souboru: {filename}...")

    # Fallback pro prázdný text
    if not text or len(text.strip()) < 10:
        return []

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    # Zkrácení textu, pokud je moc dlouhý (GPT limit)
    shortened_text = text[:12000]

    prompt = f"""
    Jsi expertní analytik. Rozděl text na logické celky (chunky).
    Vstupní soubor: {filename}

    Pravidla:
    1. Výstup MUSÍ být validní JSON.
    2. Formát: {{"chunks": [ {{"title": "...", "content": "..."}} ]}}

    Text k analýze:
    {shortened_text}
    """

    data = {
        "model": "gpt-4o-mini",  # Používáme levnější model pro chunking
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

        if response.status_code != 200:
            print(f"⚠️ API Error {response.status_code}: {response.text}")
            raise Exception("API call failed")

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        json_content = json.loads(content)

        # Různé formáty, co může AI vrátit (zajištění kompatibility)
        if "chunks" in json_content:
            return json_content["chunks"]
        if "items" in json_content:
            return json_content["items"]
        if isinstance(json_content, list):
            return json_content

    except Exception as e:
        print(f"⚠️ Chyba AI chunkingu u {filename}: {e}. Používám Fallback.")

    # --- ZÁCHRANNÁ BRZDA (FALLBACK) ---
    # Pokud cokoliv selže, vrátíme celý text jako jeden chunk.
    return [{"title": filename, "content": text}]


# --- 3. Embedding ---
def get_embedding(text):
    # Ochrana před prázdným vstupem
    if not text or not text.strip():
        return None

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": text, "model": EMBEDDING_MODEL}

    try:
        response = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=data)
        if response.status_code == 200:
            return np.array(response.json()["data"][0]["embedding"])
        else:
            print(f"⚠️ Chyba Embeddings API: {response.text}")
    except Exception as e:
        print(f"⚠️ Chyba při embeddingu: {e}")

    return None


# --- HLAVNÍ LOOP ---
if __name__ == "__main__":
    init_db()

    print("🚀 Startuji indexaci Google Disku...")
    service = get_drive_service()

    if service:
        files_data = get_files_recursive(service, GOOGLE_DRIVE_FOLDER_ID)

        print(f"✅ Nalezeno a staženo celkem {len(files_data)} souborů.")

        if files_data:
            # Vyčistíme DB, ať tam nemáme duplicity
            clear_database()
            print("🧹 Databáze vyčištěna.")

            for i, file_item in enumerate(files_data):
                # Info o postupu
                print(f"[{i + 1}/{len(files_data)}] Zpracovávám: {file_item['filename']}")

                chunks = semantic_chunking(file_item['text'], file_item['filename'])

                # Kontrola proti NoneType erroru
                if not chunks:
                    print("   ⚠️ Žádné chunky nevráceny, přeskakuji.")
                    continue

                for chunk in chunks:
                    title = chunk.get("title", file_item['filename'])  # Fallback na název souboru
                    text_content = chunk.get("content", "")

                    if text_content:
                        # --- ZDE JE TA KLÍČOVÁ ZMĚNA: Obohacení kontextu ---
                        # Vytváříme "bohatý text" jen pro výpočet embeddingu (vektoru).
                        # Do databáze ale uložíme čistý text_content, aby se uživateli zobrazoval hezky.
                        enriched_text_for_embedding = (
                            f"Zdrojový soubor: {file_item['filename']}\n"
                            f"Téma: {title}\n"
                            f"Obsah: {text_content}"
                        )

                        emb = get_embedding(enriched_text_for_embedding)

                        if emb is not None:
                            insert_embedding_to_db(title, text_content, emb, file_item['filename'])
                            print(f"   💾 Uloženo: {title[:40]}...")

            print("🎉 Hotovo! Všechna data jsou v databázi.")
        else:
            print("⚠️ Žádné relevantní soubory (PDF/DOCX) nenalezeny.")