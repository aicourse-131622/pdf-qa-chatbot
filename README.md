# 📚 PDF QA Chatbot - Asisten Panduan Pembelajaran & Asesmen (PPA) 2025

Aplikasi RAG (Retrieval-Augmented Generation) berbasis kecerdasan buatan untuk melakukan tanya jawab cerdas menggunakan dokumen resmi **Panduan Pembelajaran dan Asesmen (PPA) Edisi Revisi 2025** dari Kementerian Pendidikan Dasar dan Menengah RI.

Aplikasi ini menggunakan antarmuka premium **Streamlit**, framework **LangChain**, dan **ChromaDB** sebagai basis data vektor untuk pencarian konteks yang presisi.

---

## 🌟 Fitur Utama
* **Antarmuka Premium & Modern**: Desain antarmuka interaktif menggunakan komponen chat bawaan Streamlit (`st.chat_message`) yang dilengkapi tipografi Google Fonts (*Outfit & Inter*) serta animasi pemrosesan dinamis.
* **Auto-Load Vector Database**: Mendeteksi secara otomatis jika basis data vektor sudah ada di lokal (`vectordb/`) untuk langsung digunakan tanpa perlu melakukan pengunggahan/pemrosesan ulang PDF.
* **Deteksi API Key OpenAI & OpenRouter**: Mendukung API Key standard OpenAI (`sk-...`) serta OpenRouter (`sk-or-...`) secara cerdas dengan pemetaan otomatis base URL dan model yang sesuai.
* **Akurasi Tinggi**: Dilengkapi dengan pembatas konteks yang ketat agar chatbot menjawab hanya berdasarkan dokumen PPA 2025 resmi.
* **Sitasi Referensi**: Menampilkan nomor halaman dan nama file sumber rujukan jawaban di dalam tab ekspansi yang rapi di bawah gelembung percakapan.
* **Tombol Pertanyaan Cepat**: Akses cepat contoh pertanyaan edukatif di panel sidebar untuk langsung diuji sekali klik.

---

## 🛠️ Prasyarat
Sebelum memulai, pastikan perangkat Anda sudah terpasang:
* **Python 3.8 - 3.13**
* **Git**
* API Key aktif dari **OpenAI** atau **OpenRouter**

---

## ⚙️ Panduan Instalasi & Penggunaan

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di komputer lokal Anda:

### 1. Klon Repositori
Klon repositori ini terlebih dahulu, lalu masuk ke direktori proyek:
```bash
git clone https://github.com/aicourse-131622/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

### 2. Buat & Aktifkan Virtual Environment
Sangat disarankan membuat virtual environment Python agar dependensi proyek tidak berbenturan dengan library Python global Anda:

* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Pasang Dependensi
Pasang semua pustaka yang diperlukan proyek dengan perintah:
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi API Key (Environment Variables)
Anda dapat memasang API Key melalui dua cara:

* **Cara A (Melalui berkas `.env`):**
  Salin contoh berkas `.env.example` menjadi `.env`:
  ```bash
  cp .env.example .env
  ```
  Buka berkas `.env` tersebut dan ganti nilai `your_api_key_here` dengan API Key OpenAI atau OpenRouter Anda:
  ```env
  OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
  ```

* **Cara B (Langsung di UI Web):**
  Anda juga bisa memasukkan API Key secara aman langsung di form input **🔑 API Key** pada panel kiri halaman web saat aplikasi berjalan.

### 5. Jalankan Aplikasi
Jalankan server aplikasi Streamlit menggunakan perintah:
```bash
streamlit run app.py
```

Setelah berjalan, buka tautan berikut di peramban (browser) Anda:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📁 Struktur Direktori Proyek
```
pdf-qa-chatbot/
├── documents/                      # Direktori penyimpan berkas PDF referensi RAG
│   └── Panduan_Pembelajaran_dan_Asesmen_2025.pdf
├── vectordb/                       # Folder penyimpanan database vektor ChromaDB
├── app.py                          # Kode program utama untuk antarmuka Streamlit
├── config.py                       # Pengaturan program & deteksi API Key dinamis
├── embeddings.py                   # Modul loading PDF, split teks, & database vektor
├── rag.py                          # Modul pembuatan LLM chain & pencarian jawaban
├── requirements.txt                # Daftar pustaka dependensi Python proyek
├── .env.example                    # Template berkas environment key
├── .gitignore                      # Berkas pengabaian Git untuk keamanan data
└── README.md                       # Panduan petunjuk dokumentasi ini
```
