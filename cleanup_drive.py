import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from config import GOOGLE_DRIVE_FOLDER_ID, GOOGLE_CREDENTIALS_FILE

# --- Nastavení ---
PREFIXES_TO_REMOVE = ["Copy of ", "Kopie - ", "Kopie souboru "]  # Co chceme mazat


# --- Připojení ---
def get_drive_service():
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        print(f"❌ Chyba: Soubor {GOOGLE_CREDENTIALS_FILE} nenalezen.")
        return None
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_FILE,
        scopes=['https://www.googleapis.com/auth/drive'])  # Pozor: Změna scope na plný přístup (nejen readonly)
    return build('drive', 'v3', credentials=creds)


def rename_files_recursive(service, folder_id):
    page_token = None
    count_renamed = 0

    while True:
        try:
            # Načteme soubory i složky
            response = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token
            ).execute()
        except Exception as e:
            print(f"⚠️ Chyba při listování složky {folder_id}: {e}")
            break

        items = response.get('files', [])

        for item in items:
            original_name = item['name']
            new_name = original_name

            # 1. Kontrola a přejmenování
            for prefix in PREFIXES_TO_REMOVE:
                if original_name.startswith(prefix):
                    new_name = original_name[len(prefix):]  # Ořízneme prefix
                    break  # Stačí odstranit jeden prefix

            # Pokud se název změnil, provedeme update na Disku
            if new_name != original_name:
                try:
                    print(f"✏️ Přejmenovávám: '{original_name}' -> '{new_name}'")
                    service.files().update(
                        fileId=item['id'],
                        body={'name': new_name}
                    ).execute()
                    count_renamed += 1
                except Exception as e:
                    print(f"❌ Chyba při přejmenování {original_name}: {e}")

            # 2. Pokud je to složka, zanoříme se (Rekurze)
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                # print(f"📂 Vstupuji do: {item['name']}")
                count_renamed += rename_files_recursive(service, item['id'])

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return count_renamed


if __name__ == "__main__":
    print("🚀 Startuji čištění názvů na Google Disku...")

    # Varování pro jistotu
    print("⚠️ POZOR: Tento skript reálně přejmenuje soubory na tvém Google Disku.")
    confirm = input("Chceš pokračovat? (ano/ne): ")

    if confirm.lower() in ['ano', 'yes', 'y']:
        service = get_drive_service()
        if service:
            total = rename_files_recursive(service, GOOGLE_DRIVE_FOLDER_ID)
            print(f"\n🎉 Hotovo! Přejmenováno celkem {total} položek.")
    else:
        print("Operace zrušena.")