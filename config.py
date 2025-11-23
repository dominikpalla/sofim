import os
from dotenv import load_dotenv

# Načte proměnné z .env souboru
load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small" # Novější a levnější model
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
LLM_API_URL = "https://api.openai.com/v1/chat/completions"

# Database
DB_HOST = "localhost"
DB_NAME = "sofim"
DB_USER = "root"
DB_PASSWORD = "ChatAiSprv_1614*"

# Database BACKUP
#DB_HOST = "dominikpalla.cz"
#DB_NAME = "pyto_1"
#DB_USER = "pyto.1"
#DB_PASSWORD = "^CiQoyGxYtO3O;zU7"

# Google Drive nastavení
# ID složky na Google Disku, kterou má robot sledovat
GOOGLE_DRIVE_FOLDER_ID = "1VHfrmsyhP3qDnnExvtMDxqutEeNKvd4f"
# Cesta k souboru s credentials od Googlu (stáhneš z Google Cloud Console)
GOOGLE_CREDENTIALS_FILE = "credentials.json"