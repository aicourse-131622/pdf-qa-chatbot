import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)

# Paths
DOCUMENTS_DIR = "documents"
VECTORDB_DIR  = "vectordb"

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K_RESULTS = 4

# Model (akan diset secara dinamis oleh update_config)
LLM_MODEL       = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

def update_config(api_key):
    global OPENAI_API_KEY, OPENAI_API_BASE, LLM_MODEL, EMBEDDING_MODEL
    OPENAI_API_KEY = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Deteksi otomatis OpenRouter jika key berawalan sk-or-
    if api_key and (api_key.startswith("sk-or-") or (os.getenv("OPENAI_API_BASE") and "openrouter.ai" in os.getenv("OPENAI_API_BASE"))):
        OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or "https://openrouter.ai/api/v1"
        LLM_MODEL       = "openai/gpt-4o-mini"
        EMBEDDING_MODEL = "openai/text-embedding-3-small"
        os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
    else:
        OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)
        LLM_MODEL       = "gpt-4o-mini"
        EMBEDDING_MODEL = "text-embedding-3-small"
        # Hapus environment variable jika bukan OpenRouter agar tidak mengganggu default OpenAI
        if "OPENAI_API_BASE" in os.environ and not os.getenv("OPENAI_API_BASE"):
            del os.environ["OPENAI_API_BASE"]

# Jalankan update konfigurasi awal
if OPENAI_API_KEY:
    update_config(OPENAI_API_KEY)

# System prompt - dikustomisasi untuk dokumen PPA 2025
SYSTEM_PROMPT = """
Kamu adalah asisten pendidikan yang ahli dalam Panduan Pembelajaran dan 
Asesmen (PPA) Edisi Revisi 2025 dari Kementerian Pendidikan Dasar dan 
Menengah Republik Indonesia.

Tugas kamu:
- Menjawab pertanyaan berdasarkan HANYA konten dari dokumen yang diunggah
- Memberikan jawaban yang jelas, terstruktur, dan mudah dipahami oleh pendidik
- Menyebutkan bagian/bab/halaman yang relevan jika tersedia
- Jika informasi tidak ada dalam dokumen, katakan dengan jujur

Bahasa: Gunakan Bahasa Indonesia yang formal namun mudah dipahami.
"""