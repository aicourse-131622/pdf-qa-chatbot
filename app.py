import streamlit as st
import os
import tempfile
import shutil
from embeddings import load_documents, split_documents, create_vectorstore, load_vectorstore
from rag        import build_qa_chain, get_answer
import config

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title = "📚 Asisten ChatBot",
    page_icon  = "📚",
    layout     = "wide",
)

# ─── Google Fonts (Outfit & Inter) ───────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ─── CSS Kustom Premium ──────────────────────────────────────────
st.markdown("""
<style>
    /* Styling Fonts */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Premium */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #3b82f6 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin: 0;
        font-size: 2.3rem;
        background: linear-gradient(to right, #ffffff, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin-top: 0.6rem;
        margin-bottom: 0;
        opacity: 0.85;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Stats Box */
    .stats-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }
    .stats-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-color: #3b82f6;
    }
    .stats-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .stats-lbl {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Status Badge */
    .status-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-active {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .status-inactive {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }
</style>
""", unsafe_allow_html=True)

# Helper untuk memeriksa apakah database vektor sudah ada
def check_vectorstore_exists():
    db_path = config.VECTORDB_DIR
    if os.path.exists(db_path) and os.path.isdir(db_path):
        contents = os.listdir(db_path)
        if any(f.endswith(".sqlite3") or f == "chroma.sqlite3" for f in contents):
            return True
    return False

# Mencoba memuat database yang sudah ada
def try_load_existing_db():
    if check_vectorstore_exists() and config.OPENAI_API_KEY:
        try:
            vs = load_vectorstore()
            count = vs._collection.count()
            if count > 0:
                return vs, count
        except Exception as e:
            print(f"Error loading existing vectorstore: {e}")
    return None, 0

# ─── Session State ────────────────────────────────────────────────
def init_session():
    defaults = {
        "qa_chain"      : None,
        "chat_history"  : [],
        "docs_processed": False,
        "doc_stats"     : {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# Jalankan pendeteksian database otomatis saat startup jika kunci API tersedia
if st.session_state.qa_chain is None and config.OPENAI_API_KEY:
    vs, count = try_load_existing_db()
    if vs:
        try:
            chain = build_qa_chain(vs)
            st.session_state.qa_chain = chain
            st.session_state.docs_processed = True
            
            # Cari jumlah file di folder documents/ jika ada
            local_pdfs = [
                f for f in os.listdir(config.DOCUMENTS_DIR)
                if f.endswith(".pdf")
            ] if os.path.exists(config.DOCUMENTS_DIR) else []
            
            st.session_state.doc_stats = {
                "files"  : len(local_pdfs) if local_pdfs else 1,
                "pages"  : "Tersimpan",
                "chunks" : count,
            }
        except Exception as e:
            st.warning(f"Gagal memuat database otomatis: {e}")

# ─── Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 Asisten Chatbot</h1>
    <p>Tanya jawab cerdas berbasis dokumen resmi</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")

    # API Key Input
    api_key = st.text_input(
        "🔑 API Key (OpenAI / OpenRouter)",
        type     = "password",
        value    = config.OPENAI_API_KEY or "",
        help     = "Masukkan API Key Anda. Mendukung OpenAI (sk-...) dan OpenRouter (sk-or-...).",
    )
    if api_key:
        config.update_config(api_key)

    # Indikator Host & Model yang aktif
    if config.OPENAI_API_KEY:
        is_or = config.OPENAI_API_KEY.startswith("sk-or-")
        host_name = "OpenRouter" if is_or else "OpenAI"
        st.caption(f"🤖 **Host:** {host_name} | **Model:** `{config.LLM_MODEL}`")
    else:
        st.caption("⚠️ Masukkan API Key untuk mengaktifkan model.")

    st.divider()

    # Upload PDF
    st.header("📂 Upload Dokumen")
    uploaded_files = st.file_uploader(
        "Upload PDF tambahan jika perlu",
        type            = ["pdf"],
        accept_multiple_files = True,
        help            = "Secara default menggunakan Panduan PPA 2025 di folder documents/",
    )

    # Tombol proses
    process_btn = st.button(
        "🚀 Proses Dokumen",
        use_container_width = True,
        type                = "primary",
        disabled            = not config.OPENAI_API_KEY,
    )

    if process_btn:
        if not uploaded_files:
            # Coba load dari folder documents/
            local_pdfs = [
                os.path.join(config.DOCUMENTS_DIR, f)
                for f in os.listdir(config.DOCUMENTS_DIR)
                if f.endswith(".pdf")
            ] if os.path.exists(config.DOCUMENTS_DIR) else []

            if local_pdfs:
                file_paths = local_pdfs
            else:
                st.error("❌ Upload minimal 1 file PDF atau masukkan ke folder `documents/`!")
                st.stop()
        else:
            # Simpan file upload ke temp dir
            tmp_dir    = tempfile.mkdtemp()
            file_paths = []
            for uf in uploaded_files:
                path = os.path.join(tmp_dir, uf.name)
                with open(path, "wb") as f:
                    f.write(uf.read())
                file_paths.append(path)

        with st.spinner("⏳ Sedang memproses & mengekstrak teks PDF..."):
            try:
                # Pipeline RAG
                docs   = load_documents(file_paths)
                chunks = split_documents(docs)
                vs     = create_vectorstore(chunks)
                chain  = build_qa_chain(vs)

                st.session_state.qa_chain       = chain
                st.session_state.docs_processed = True
                st.session_state.doc_stats      = {
                    "files"  : len(file_paths),
                    "pages"  : len(docs),
                    "chunks" : len(chunks),
                }
                st.success("✅ Dokumen berhasil diproses ke Database Vektor!")

                # Cleanup temp
                if uploaded_files and os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")

    # Statistik & Status Database Vektor
    st.divider()
    st.markdown("### 📊 Status & Statistik")
    if st.session_state.docs_processed:
        st.markdown(
            '<div class="status-badge status-active">🟢 Database Siap</div>',
            unsafe_allow_html=True
        )
        st.write("")
        stats = st.session_state.doc_stats
        cols  = st.columns(3)
        with cols[0]:
            st.markdown(
                f'<div class="stats-box">'
                f'<div class="stats-val">{stats["files"]}</div>'
                f'<div class="stats-lbl">File</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with cols[1]:
            st.markdown(
                f'<div class="stats-box">'
                f'<div class="stats-val">{stats["pages"]}</div>'
                f'<div class="stats-lbl">Hal</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with cols[2]:
            st.markdown(
                f'<div class="stats-box">'
                f'<div class="stats-val">{stats["chunks"]}</div>'
                f'<div class="stats-lbl">Chunks</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="status-badge status-inactive">🔴 Database Belum Siap</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # Tombol reset
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.qa_chain:
            st.session_state.qa_chain.memory.clear()
        st.rerun()


# ─── Main Chat Area ───────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    # Status dokumen info awal
    if not st.session_state.docs_processed:
        st.info(
            "### 💡 Cara Penggunaan:\n"
            "1. **Masukkan API Key** di panel sebelah kiri.\n"
            "2. Klik tombol **🚀 Proses Dokumen** untuk membaca PDF bawaan atau PDF yang baru diunggah.\n"
            "3. Setelah status menjadi **🟢 Database Siap**, silakan mulai mengetikkan pertanyaan Anda!"
        )

    # Tampilkan Riwayat Chat dengan st.chat_message
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Lihat Sumber Referensi"):
                    st.markdown(msg["sources"])

    # Input pertanyaan menggunakan st.chat_input (lebih modern daripada form text_area)
    user_input = st.chat_input("Tanyakan sesuatu tentang dokumen yang diupload...")

    if user_input:
        if not st.session_state.docs_processed:
            st.warning("⚠️ Proses dokumen terlebih dahulu atau masukkan API Key yang valid!")
        else:
            # Tambah pesan user ke history
            st.session_state.chat_history.append({
                "role"   : "user",
                "content": user_input,
            })
            
            # Tampilkan langsung pertanyaan user agar responsif
            st.rerun()

    # Cek jika ada input terakhir yang belum dijawab oleh AI
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        last_question = st.session_state.chat_history[-1]["content"]
        
        # Jalankan animasi loading RAG
        with st.chat_message("assistant"):
            with st.spinner("🔍 Sedang menganalisis dokumen dan menyusun jawaban..."):
                try:
                    result = get_answer(
                        st.session_state.qa_chain,
                        last_question,
                    )
                    # Tambah respons AI ke history
                    st.session_state.chat_history.append({
                        "role"   : "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat mengambil jawaban: {e}")


# ─── Kolom Kanan: Info & Contoh Pertanyaan ───────────────────────
with col2:
    st.header("💡 Contoh Pertanyaan")
    sample_questions = [
        "Jelaskan prinsip-prinsip pembelajaran dan asesmen?",
        "Apa perbedaan asesmen formatif dan sumatif?",
        "Bagaimana merumuskan tujuan pembelajaran?",
        "Bagaimana alur tujuan pembelajaran dibuat?",
        "Bagaimana mekanisme kenaikan kelas dan kelulusan?",
        "Apa itu Kriteria Ketercapaian Tujuan Pembelajaran?",
        "Jelaskan pengolahan hasil asesmen untuk rapor!",
    ]
    for q in sample_questions:
        if st.button(f"💬 {q}", use_container_width=True, key=f"sq_{q}"):
            if not st.session_state.docs_processed:
                st.warning("⚠️ Silakan proses dokumen terlebih dahulu!")
            else:
                # Daftarkan langsung ke chat history
                st.session_state.chat_history.append({
                    "role"   : "user",
                    "content": q,
                })
                st.rerun()

    st.divider()

    st.header("ℹ️ Tentang Sistem")
    st.markdown("""
    **Teknologi Utama:**
    - **LLM:** `gpt-4o-mini` (OpenAI / OpenRouter)
    - **Embeddings:** `text-embedding-3-small`
    - **Vector Store:** ChromaDB
    - **Framework:** LangChain (RAG)
    - **Antarmuka:** Streamlit Premium
    
    **Dokumen Referensi:**
    *Panduan Pembelajaran dan Asesmen (PPA) Edisi Revisi 2025* dari Kementerian Pendidikan Dasar dan Menengah RI.
    """)