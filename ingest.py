import os
import json
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import io
from pypdf import PdfReader
import docx  # Ponecháváme, kdybychom v budoucnu chtěli importovat lokální DOCX

from config import OPENAI_API_KEY, EMBEDDING_MODEL, OPENAI_EMBEDDING_URL
from database import (
    prepare_next_table_for_update,
    insert_into_next_table,
    swap_tables_atomic,
    get_db_connection,
    set_sync_status,
    update_sync_progress,
    log_sync_error
)


# --- 1. Pomocné funkce pro CRAWLER ---

def get_urls_from_db():
    """Načte seznam URL k indexaci z databáze."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Ověříme, zda tabulka existuje (pro jistotu)
    try:
        cursor.execute("SELECT url FROM crawler_urls")
        urls = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"⚠️ Tabulka crawler_urls asi neexistuje nebo je prázdná: {e}")
        urls = []
    finally:
        conn.close()
    return urls


def scrape_uhk_page(url):
    """Stáhne stránku, vyčistí HTML a najde PDF odkazy."""
    print(f"🕸️ Crawluji: {url}")
    try:
        headers = {"User-Agent": "SofimBot/1.0 (UHK Internal)"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"   ❌ Chyba HTTP {response.status_code}")
            return None, []

        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Hledání PDF odkazů PŘEDTÍM, než promažeme DOM
        pdf_urls = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # UHK ukládá soubory často přes /file/ nebo končí .pdf
            if '/file/' in href or href.lower().endswith('.pdf'):
                full_pdf_url = urljoin(url, href)
                if full_pdf_url not in pdf_urls:
                    pdf_urls.append(full_pdf_url)

        # 2. Agresivní čištění balastu
        for element in soup(["header", "footer", "nav", "script", "style", "noscript", "iframe"]):
            element.decompose()

        # Zacílení na UHK specifické třídy
        main_content = soup.find(class_="main__content") or soup.find("main") or soup.find("article")
        target_soup = main_content if main_content else soup.body

        if not target_soup:
            return None, pdf_urls

        # Odstranění dalšího balastu
        for noise in target_soup.find_all(class_=["share-buttons", "sidebar", "breadcrumb", "cookies-bar"]):
            noise.decompose()

        # 3. Extrakce čistého textu
        raw_text = target_soup.get_text(separator='\n', strip=True)
        clean_text = "\n".join([line.strip() for line in raw_text.splitlines() if line.strip()])

        return clean_text, pdf_urls

    except Exception as e:
        print(f"⚠️ Chyba při stahování {url}: {e}")
        return None, []


def process_pdf_from_url(pdf_url):
    """Stáhne a přečte PDF z URL do paměti."""
    print(f"   📄 Stahuji PDF: {pdf_url}")
    try:
        headers = {"User-Agent": "SofimBot/1.0 (UHK Internal)"}
        response = requests.get(pdf_url, headers=headers, timeout=15)

        if response.status_code == 200:
            fh = io.BytesIO(response.content)
            reader = PdfReader(fh)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            # Validace, zda to není jen sken (obrázek)
            if len(text.strip()) < 10:
                print(f"   ⚠️ PDF {pdf_url} je pravděpodobně sken bez textové vrstvy.")
                return None

            return text
        else:
            print(f"   ❌ Nelze stáhnout PDF (Status {response.status_code})")
    except Exception as e:
        print(f"   ❌ Chyba čtení PDF {pdf_url}: {e}")
    return None


# --- 2. Pomocné funkce pro CSV (Hybridní model) ---

def read_csv_smart(fh):
    """Načte CSV s důrazem na zachování všech dat, poradí si s kódováním i oddělovači."""
    encodings = ['utf-8', 'cp1250', 'latin1']

    for encoding in encodings:
        fh.seek(0)
        try:
            # Přečteme CSV
            df = pd.read_csv(fh, sep=None, engine='python', encoding=encoding, on_bad_lines='skip')

            # Validace hlavičky podle klíčových slov
            keywords = ['zkratka', 'zkr_predm', 'nazev_cz', 'kredity', 'anotace_cz']

            # Pokud hlavička nesedí, zkusíme ji najít níže
            col_str = str(list(df.columns)).lower()
            if not any(k in col_str for k in keywords):
                fh.seek(0)
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

            # Vyčištění
            df = df.dropna(how='all')
            df = df.fillna("")
            df.columns = [str(c).strip() for c in df.columns]

            return df

        except Exception:
            continue
    return None


# --- 3. Chunking funkce (Nezměněno) ---

def semantic_chunking(text, filename):
    """Inteligentní řezání textu pomocí GPT-4o-mini."""
    if not text or len(text.strip()) < 10:
        return []

    print(f"🧠 Sémantické řezání obsahu: {filename}...")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    shortened_text = text[:12000]  # Limit tokenů

    prompt = f"""
    Jsi expertní analytik. Rozděl text na logické celky (chunky).
    Zdroj: {filename}
    Pravidla:
    1. Výstup MUSÍ být validní JSON.
    2. Formát: {{"chunks": [ {{"title": "...", "content": "..."}} ]}}
    Text k analýze:
    {shortened_text}
    """

    data = {
        "model": "gpt-4o-mini",  # Levný model na chunking
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            json_content = json.loads(content)

            if "chunks" in json_content: return json_content["chunks"]
            if "items" in json_content: return json_content["items"]
    except Exception as e:
        print(f"⚠️ Chyba AI chunkingu: {e}. Používám Fallback.")

    # Fallback: Vrátíme to jako jeden kus
    return [{"title": f"Obsah z {filename}", "content": text}]


def csv_row_chunking(df, filename):
    """Řádkové zpracování tabulky předmětů."""
    print(f"📊 Zpracovávám tabulku předmětů: {filename} ({len(df)} řádků)...")
    chunks = []

    for index, row in df.iterrows():
        row_dict = row.to_dict()

        # Identifikace
        nazev = row_dict.get('NAZEV_CZ', row_dict.get('NAZEV_AN', 'Neznámý předmět'))
        kod = row_dict.get('ZKR_PREDM', '')

        # Hledání kódu jinde
        if not kod:
            for k, v in row_dict.items():
                if 'zkr' in str(k).lower() and not kod: kod = str(v)

        if nazev == 'Neznámý předmět' and not kod:
            continue

        title = f"Předmět: {nazev} ({kod})".strip()
        content_lines = [f"--- Detail předmětu: {title} ---"]

        priority_fields = {
            'NAZEV_AN': 'Anglický název', 'GARANTI': 'Garanti', 'VYUCUJICI': 'Vyučující',
            'KREDITY': 'Kredity', 'ROK_VARIANTY': 'Rok varianty', 'ANOTACE_CZ': 'Anotace',
            'CIL_CZ': 'Cíle předmětu', 'OSNOVA_CZ': 'Osnova', 'LITERATURA': 'Literatura',
            'POZADAVKY_CZ': 'Požadavky', 'METODY_VYUKY_CZ': 'Metody', 'URL': 'Odkaz'
        }

        for key, label in priority_fields.items():
            if key in row_dict:
                val = str(row_dict[key]).strip()
                if val and val.lower() != 'nan':
                    content_lines.append(f"{label}: {val}")

        chunks.append({"title": title, "content": "\n".join(content_lines)})

    return chunks


# --- 4. Embedding (Nezměněno) ---

def get_embedding(text):
    if not text or not text.strip():
        return None

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": text, "model": EMBEDDING_MODEL}

    try:
        response = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=data)
        if response.status_code == 200:
            return np.array(response.json()["data"][0]["embedding"])
    except Exception:
        pass
    return None


# --- 5. HLAVNÍ LOGIKA INDEXACE ---

def run_ingest(mode="all"):
    """
    Spustí proces ingestu. Režimy: 'all', 'web', 'csv'.
    Propojeno s databází pro sledování průběhu v admin panelu.
    """
    print(f"🚀 Startuji indexaci na pozadí (Režim: {mode})...")

    # Nastavíme status v DB na "běží" (zatím bez celkového počtu, ten se updatne hned jak ho zjistíme)
    if mode in ["all", "web"]: set_sync_status("WEB", "running")
    if mode in ["all", "csv"]: set_sync_status("CSV", "running")

    try:
        # Připravíme stínovou tabulku (vyčistí vše / zkopíruje a připraví pro částečný update podle módu)
        prepare_next_table_for_update(mode)
        success_count = 0

        # --- FÁZE A: CRAWLER (Web UHK) ---
        if mode in ["all", "web"]:
            urls = get_urls_from_db()
            total_urls = len(urls)

            # Nastavíme celkový počet URL do databáze pro progress bar
            set_sync_status("WEB", "running", total=total_urls)

            if urls:
                print(f"🌍 Nalezeno {total_urls} URL adres k indexaci.")
                for idx, url in enumerate(urls, 1):
                    try:
                        web_text, pdf_links = scrape_uhk_page(url)

                        if web_text:
                            chunks = semantic_chunking(web_text, f"Web: {url}")
                            for chunk in chunks:
                                title = chunk.get("title", "Webová stránka")
                                content = chunk.get("content", "")
                                emb = get_embedding(f"URL: {url}\n{content}")
                                if emb is not None:
                                    insert_into_next_table(title, content, emb, url)
                                    print(f"   💾 Web uložen: {title[:30]}...")
                                    success_count += 1

                        if pdf_links:
                            print(f"   📎 Nalezeno {len(pdf_links)} PDF dokumentů na odkazu {url}.")
                            for pdf_url in pdf_links:
                                pdf_text = process_pdf_from_url(pdf_url)
                                if pdf_text:
                                    chunks = semantic_chunking(pdf_text, f"PDF: {pdf_url.split('/')[-1]}")
                                    for chunk in chunks:
                                        title = chunk.get("title", "PDF Dokument")
                                        content = chunk.get("content", "")
                                        emb = get_embedding(f"Zdroj PDF: {pdf_url}\n{content}")
                                        if emb is not None:
                                            insert_into_next_table(title, content, emb, pdf_url)
                                            success_count += 1

                    except Exception as e:
                        log_sync_error("WEB", f"Chyba na {url}: {str(e)}")
                        print(f"   ❌ Chyba zpracování {url}: {e}")

                    # 📢 Průběžný report postupu do databáze
                    update_sync_progress("WEB", idx)
            else:
                print("⚠️ Žádná URL v databázi. Přidej je přes /admin.")

        # --- FÁZE B: LOKÁLNÍ CSV (Studijní plány) ---
        if mode in ["all", "csv"]:
            csv_path = "data/predmety.csv"

            if os.path.exists(csv_path):
                print(f"📊 Načítám lokální CSV: {csv_path}")
                try:
                    with open(csv_path, "rb") as f:
                        df = read_csv_smart(f)

                    if df is not None:
                        csv_chunks = csv_row_chunking(df, "Lokální Databáze Předmětů")
                        total_rows = len(csv_chunks)

                        # Nastavíme celkový počet pro progress bar
                        set_sync_status("CSV", "running", total=total_rows)

                        for idx, chunk in enumerate(csv_chunks, 1):
                            emb = get_embedding(chunk["content"])
                            if emb is not None:
                                # Klíčové: Udržíme identifikátor "STAG Export" pro parciální mazání
                                insert_into_next_table(chunk["title"], chunk["content"], emb, "STAG Export")
                                success_count += 1

                            # 📢 Průběžný report postupu
                            update_sync_progress("CSV", idx)

                        print(f"✅ CSV zpracováno: {total_rows} předmětů.")
                    else:
                        set_sync_status("CSV", "running", total=0)
                        log_sync_error("CSV", "Nelze načíst obsah CSV.")
                except Exception as e:
                    log_sync_error("CSV", f"Chyba při čtení CSV: {str(e)}")
                    print(f"❌ Chyba při čtení CSV: {e}")
            else:
                set_sync_status("CSV", "running", total=0)
                log_sync_error("CSV", f"Soubor nenalezen: {csv_path}")
                print(f"⚠️ CSV soubor nenalezen na cestě: {csv_path}. Přeskočeno.")

        # --- FINÁLE: PROHOZENÍ TABULEK ---
        print(f"🔄 Provádím atomické prohození tabulek (Zpracováno celkem {success_count} záznamů)...")
        # Prohodíme tabulky i kdyby success_count byl 0 (např. při smazání url se musí live db aktualizovat)
        swap_tables_atomic()

        # Nastavíme status na úspěch a získáme hezký timestamp aktuálního času
        if mode in ["all", "web"]: set_sync_status("WEB", "success")
        if mode in ["all", "csv"]: set_sync_status("CSV", "success")
        print("🎉 Indexace úspěšně dokončena. Data jsou LIVE.")

    except Exception as e:
        print(f"❌ Krizová chyba při indexaci: {e}")
        # Při krizové chybě to zalogujeme a hodíme do stavu error/idle
        if mode in ["all", "web"]:
            log_sync_error("WEB", f"Kritická chyba: {str(e)}")
            set_sync_status("WEB", "error")
        if mode in ["all", "csv"]:
            log_sync_error("CSV", f"Kritická chyba: {str(e)}")
            set_sync_status("CSV", "error")


if __name__ == "__main__":
    # Pokud spustíš ingest.py ručně z konzole, spustí se kompletní indexace
    run_ingest("all")