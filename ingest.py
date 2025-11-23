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
        print(f"Chyba: Soubor {GOOGLE_CREDENTIALS_FILE} nenalezen.")
        return None
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly'])
    return build('drive', 'v3', credentials=creds)


# Pomocná funkce: Stáhne a vytáhne text z jednoho souboru
def process_file_content(service, file_item):
    print(f"  📄 Stahuji soubor: {file_item['name']}...")

    # Filtrování: Zajímají nás jen Word a PDF (můžeš přidat další)
    supported_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
    is_google_doc = file_item['mimeType'] == 'application/vnd.google-apps.document'

    # Přeskočíme soubory, které neumíme zpracovat (obrázky, zipy atd.)
    # Pozn.: Google Docs (online) by se musely exportovat, zde řešíme primárně nahrané DOCX/PDF
    if file_item['mimeType'] not in supported_types and not file_item['name'].endswith(('.pdf', '.docx')):
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

        # Parser pro DOCX
        if file_item['name'].endswith('.docx'):
            doc = docx.Document(fh)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])

        # Parser pro PDF
        elif file_item['name'].endswith('.pdf'):
            reader = PdfReader(fh)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        if text:
            return {"filename": file_item['name'], "text": text}

    except Exception as e:
        print(f"⚠️ Chyba při zpracování souboru {file_item['name']}: {e}")

    return None


# Hlavní rekurzivní funkce
def get_files_recursive(service, folder_id):
    results_list = []
    page_token = None

    while True:
        # Dotaz na soubory v konkrétní složce (folder_id)
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()

        items = response.get('files', [])

        for item in items:
            # KDYŽ JE TO SLOŽKA -> REKURZE (Voláme sami sebe)
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                print(f"📂 Vstupuji do podsložky: {item['name']}")
                # Přidáme výsledky z podsložky do našeho hlavního listu
                results_list.extend(get_files_recursive(service, item['id']))

            # KDYŽ JE TO SOUBOR -> ZPRACUJEME
            else:
                processed_file = process_file_content(service, item)
                if processed_file:
                    results_list.append(processed_file)

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return results_list


# --- 2. Inteligentní Chunking přes LLM ---
# (Tato funkce zůstává stejná jako minule)
def semantic_chunking(text, filename):
    # ... (zde vlož kód funkce semantic_chunking z předchozí odpovědi) ...
    # Pro úsporu místa ji sem nekopíruji celou, ale v souboru musí být!
    pass


# Aby to fungovalo, zkopíruj si sem tu funkci semantic_chunking z minula.
# Tady je jen zjednodušený mock pro testování, POKUD BYS NEMĚL OPENAI KREDIT:
# def semantic_chunking(text, filename):
#     return [{"title": filename, "content": text}]

# --- 3. Embedding ---
# (Zůstává stejná jako minule)
def get_embedding(text):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": text, "model": EMBEDDING_MODEL}
    response = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=data)
    if response.status_code == 200:
        return np.array(response.json()["data"][0]["embedding"])
    return None


# --- HLAVNÍ LOOP ---
if __name__ == "__main__":
    init_db()

    # Pozor: Musíš mít definovanou funkci semantic_chunking (buď z minula, nebo importovanou)
    from ingest import semantic_chunking  # Pokud bys to měl rozdělené, nebo ji definuj výše

    print("🚀 Startuji indexaci Google Disku...")
    service = get_drive_service()

    if service:
        # Tady voláme tu novou rekurzivní funkci
        files_data = get_files_recursive(service, GOOGLE_DRIVE_FOLDER_ID)

        print(f"✅ Nalezeno a staženo celkem {len(files_data)} souborů.")

        if files_data:
            clear_database()
            print("🧹 Databáze vyčištěna.")

            for file_item in files_data:
                # Dále je to stejné...
                chunks = semantic_chunking(file_item['text'], file_item['filename'])

                for chunk in chunks:
                    title = chunk.get("title", "Bez názvu")
                    text_content = chunk.get("content", "")

                    if text_content:
                        emb = get_embedding(text_content)
                        if emb is not None:
                            insert_embedding_to_db(title, text_content, emb, file_item['filename'])
                            print(f"💾 Uloženo: {title}")

            print("🎉 Hotovo! Všechna data jsou v databázi.")
        else:
            print("⚠️ Žádné relevantní soubory (PDF/DOCX) nenalezeny.")