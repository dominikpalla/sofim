import os
import json
import requests
import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import docx
from pypdf import PdfReader
from config import OPENAI_API_KEY, GOOGLE_DRIVE_FOLDER_ID, GOOGLE_CREDENTIALS_FILE, EMBEDDING_MODEL, \
    OPENAI_EMBEDDING_URL
# Importujeme funkce pro stínovou tabulku (Zero Downtime)
from database import init_next_table, insert_into_next_table, swap_tables_atomic


# --- 1. Připojení ke Google Disku ---
def get_drive_service():
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        print(f"❌ Chyba: Soubor {GOOGLE_CREDENTIALS_FILE} nenalezen.")
        return None
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly'])
    return build('drive', 'v3', credentials=creds)


def read_csv_smart(fh):
    """Načte CSV s důrazem na zachování všech dat, poradí si s kódováním i oddělovači."""
    encodings = ['utf-8', 'cp1250', 'latin1']

    for encoding in encodings:
        fh.seek(0)
        try:
            # Přečteme CSV, automatická detekce oddělovače (sep=None) vyžaduje engine='python'
            df = pd.read_csv(fh, sep=None, engine='python', encoding=encoding, on_bad_lines='skip')

            # --- Validace hlavičky ---
            # Hledáme klíčová slova z tvého souboru (podle tvého uploadu)
            keywords = ['zkratka', 'zkr_predm', 'nazev_cz', 'kredity', 'anotace_cz']

            # Pokud v současných sloupcích není nic z klíčových slov, zkusíme najít hlavičku níže
            # (Někdy exporty začínají prázdnými řádky nebo metadaty)
            col_str = str(list(df.columns)).lower()
            if not any(k in col_str for k in keywords):
                print(f"   🕵️‍♀️ Hledám hlavičku tabulky v {encoding}...")
                fh.seek(0)
                # Načteme kousek bez hlavičky
                df_raw = pd.read_csv(fh, sep=None, engine='python', encoding=encoding, header=None, on_bad_lines='skip',
                                     nrows=15)

                header_index = -1
                for i in range(len(df_raw)):
                    row_str = str(df_raw.iloc[i].values).lower()
                    if any(k in row_str for k in keywords):
                        header_index = i
                        break

                if header_index != -1:
                    fh.seek(0)
                    df = pd.read_csv(fh, sep=None, engine='python', encoding=encoding, header=header_index,
                                     on_bad_lines='skip')
                    print(f"   ✅ Hlavička nalezena na řádku {header_index}.")

            # Vyčištění
            df = df.dropna(how='all')  # Smaže prázdné řádky
            df = df.fillna("")  # NaN -> ""

            # Normalizace názvů sloupců (odstranění mezer na začátku/konci názvu sloupce)
            df.columns = [str(c).strip() for c in df.columns]

            return df

        except Exception:
            continue

    return None


def process_file_content(service, file_item):
    print(f"  📄 Stahuji soubor: {file_item['name']}...")

    supported_types = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/csv',
        'application/vnd.ms-excel'
    ]

    # Rychlá kontrola koncovky a MIME typu
    if not (file_item['mimeType'] in supported_types or file_item['name'].endswith(('.pdf', '.docx', '.csv'))):
        return None

    try:
        request = service.files().get_media(fileId=file_item['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        fh.seek(0)

        # --- CSV ---
        if file_item['name'].endswith('.csv'):
            df = read_csv_smart(fh)
            if df is not None:
                return {"filename": file_item['name'], "type": "csv", "data": df}
            else:
                print(f"   ❌ Nepodařilo se přečíst CSV {file_item['name']} (ani utf-8, ani cp1250).")
                return None

        # --- DOCX ---
        text = ""
        if file_item['name'].endswith('.docx'):
            try:
                doc = docx.Document(fh)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
            except Exception as e:
                print(f"   ❌ Chyba čtení DOCX {file_item['name']}: {e}")

        # --- PDF ---
        elif file_item['name'].endswith('.pdf'):
            try:
                reader = PdfReader(fh)
                count = 0
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                        count += 1

                # Detekce "prázdného" PDF (sken)
                if len(text.strip()) == 0 and count > 0:
                    print(f"   ⚠️ PDF {file_item['name']} má stránky, ale žádný text. Asi sken?")
            except Exception as e:
                print(f"   ❌ Chyba čtení PDF {file_item['name']}: {e}")

        # Validace textu (pro PDF/DOCX)
        if text:
            text_len = len(text.strip())
            if text_len < 10:
                print(f"   ⚠️ VAROVÁNÍ: Soubor {file_item['name']} obsahuje jen {text_len} znaků! (Ignoruji)")
                return None

            return {"filename": file_item['name'], "type": "text", "text": text}

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


# --- 2. Chunking funkce ---

# A) Sémantické řezání pro dokumenty (PDF/DOCX)
def semantic_chunking(text, filename):
    print(f"🧠 Sémantické řezání souboru: {filename}...")

    if not text or len(text.strip()) < 10:
        return []

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
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
        "model": "gpt-4o-mini",
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

        if "chunks" in json_content: return json_content["chunks"]
        if "items" in json_content: return json_content["items"]
        if isinstance(json_content, list): return json_content

    except Exception as e:
        print(f"⚠️ Chyba AI chunkingu u {filename}: {e}. Používám Fallback.")

    return [{"title": filename, "content": text}]


# B) Řádkové řezání pro tabulky (CSV) - NASTAVENO PRO TVŮJ EXPORT
def csv_row_chunking(df, filename):
    print(f"📊 Zpracovávám tabulku předmětů: {filename} ({len(df)} řádků)...")
    chunks = []

    for index, row in df.iterrows():
        row_dict = row.to_dict()

        # 1. Identifikace předmětu (Název + Kód)
        # Hledáme konkrétní sloupce z tvého souboru
        nazev = row_dict.get('NAZEV_CZ', '')
        if not nazev:
            # Fallback
            nazev = row_dict.get('NAZEV_AN', 'Neznámý předmět')

        kod = row_dict.get('ZKR_PREDM', '')
        if not kod:
            # Fallback pro jiné názvy sloupců
            for k, v in row_dict.items():
                if 'zkr' in str(k).lower() and not kod: kod = str(v)

        # Pokud nemáme ani název, ani kód, řádek přeskočíme (asi prázdný)
        if nazev == 'Neznámý předmět' and not kod:
            continue

        title = f"Předmět: {nazev} ({kod})".strip()

        # 2. Sestavení obsahu (Formátovaný text)
        content_lines = [f"--- Detail předmětu: {title} ---"]

        # Definujeme pole, která chceme vytáhnout PŘEDNOSTNĚ a jejich české popisky
        priority_fields = {
            'NAZEV_AN': 'Anglický název',
            'GARANTI': 'Garanti',
            'VYUCUJICI': 'Vyučující',
            'KREDITY': 'Kredity',
            'ROK_VARIANTY': 'Rok varianty',
            'ANOTACE_CZ': 'Anotace',
            'CIL_CZ': 'Cíle předmětu',
            'OSNOVA_CZ': 'Osnova',
            'LITERATURA': 'Literatura',
            'POZADAVKY_CZ': 'Požadavky na studenta',
            'METODY_VYUKY_CZ': 'Metody výuky',
            'URL': 'Odkaz'
        }

        # Nejprve vypíšeme prioritní pole (pokud v řádku jsou a nejsou prázdná)
        for key, label in priority_fields.items():
            if key in row_dict:
                val = str(row_dict[key]).strip()
                if val and val.lower() != 'nan':
                    content_lines.append(f"{label}: {val}")

        # Potom projedeme zbytek sloupců, abychom o nic nepřišli
        # (Vynecháme ty, co už jsme vypsali, a technické sloupce)
        ignored_cols = ['FAKULTA', 'PRAC_ZKR', 'STAV_AKREDITACE', 'ZKR_PREDM', 'NAZEV_CZ']

        for k, v in row_dict.items():
            k_str = str(k)
            # Pokud už jsme to vypsali nebo to chceme ignorovat -> přeskočit
            if k_str in priority_fields or k_str in ignored_cols:
                continue
            # Pokud je to "Unnamed" nebo prázdné -> přeskočit
            if "unnamed" in k_str.lower():
                continue

            val = str(v).strip()
            if val and val.lower() != 'nan':
                # Hezké formátování názvu sloupce (např. TYP_ZK -> Typ Zk)
                nice_k = k_str.replace('_', ' ').title()
                content_lines.append(f"{nice_k}: {val}")

        content = "\n".join(content_lines)

        chunks.append({
            "title": title,
            "content": content
        })

    return chunks


# --- 3. Embedding ---
def get_embedding(text):
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


# --- HLAVNÍ LOOP (ZERO DOWNTIME) ---
if __name__ == "__main__":
    print("🚀 Startuji Zero-Downtime Indexaci...")

    # 1. Připravíme stínovou tabulku (funkce sama smaže starou a vytvoří novou)
    init_next_table()
    print("👻 Stínová tabulka (embeddings_next) připravena.")

    service = get_drive_service()

    if service:
        files_data = get_files_recursive(service, GOOGLE_DRIVE_FOLDER_ID)

        print(f"✅ Nalezeno a staženo celkem {len(files_data)} souborů.")

        if files_data:
            success_count = 0

            for i, file_item in enumerate(files_data):
                chunks = []

                # Rozhodování typu
                if file_item.get("type") == "csv":
                    # CSV -> Řádkový chunking s prioritními poli
                    chunks = csv_row_chunking(file_item['data'], file_item['filename'])
                else:
                    # Text/PDF -> AI chunking
                    print(f"[{i + 1}/{len(files_data)}] AI Zpracování: {file_item['filename']}")
                    chunks = semantic_chunking(file_item['text'], file_item['filename'])

                if not chunks:
                    continue

                for chunk in chunks:
                    title = chunk.get("title", file_item['filename'])
                    text_content = chunk.get("content", "")

                    if text_content:
                        # Obohacený kontext pro lepší vyhledávání
                        enriched_text_for_embedding = (
                            f"Zdrojový soubor: {file_item['filename']}\n"
                            f"Téma: {title}\n"
                            f"Obsah: {text_content}"
                        )

                        emb = get_embedding(enriched_text_for_embedding)

                        if emb is not None:
                            # 2. Vkládáme do STÍNOVÉ tabulky
                            insert_into_next_table(title, text_content, emb, file_item['filename'])

                            if file_item.get("type") != "csv":
                                print(f"   💾 Uloženo: {title[:40]}...")

                if file_item.get("type") == "csv":
                    print(f"   ✅ Uloženo {len(chunks)} záznamů z CSV tabulky.")

                if len(chunks) > 0:
                    success_count += 1

            # 3. Pokud proběhlo zpracování úspěšně, prohodíme tabulky
            if success_count > 0:
                print("🔄 Provádím atomické prohození tabulek (Swap)...")
                swap_tables_atomic()
                print("🎉 Hotovo! Nová data jsou LIVE. Uživatelé nic nepoznali.")
            else:
                print("⚠️ Nebyla zpracována žádná data, tabulky neprohazuji.")
        else:
            print("⚠️ Žádné relevantní soubory nenalezeny.")