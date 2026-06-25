from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import config

def load_documents(uploaded_files: list) -> list:
    """Load dan parse semua PDF yang diupload."""
    all_documents = []

    for file_path in uploaded_files:
        try:
            loader    = PyPDFLoader(file_path)
            documents = loader.load()

            # Tambahkan metadata nama file
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(file_path)

            all_documents.extend(documents)
            print(f"✅ Loaded: {os.path.basename(file_path)} "
                  f"({len(documents)} halaman)")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")

    return all_documents


def split_documents(documents: list) -> list:
    """Split dokumen menjadi chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"📄 Total chunks: {len(chunks)}")
    return chunks


def create_vectorstore(chunks: list) -> Chroma:
    """Buat vector database dari chunks."""
    embeddings = OpenAIEmbeddings(
        model   = config.EMBEDDING_MODEL,
        api_key = config.OPENAI_API_KEY,
        openai_api_base = config.OPENAI_API_BASE,
    )

    vectorstore = Chroma.from_documents(
        documents       = chunks,
        embedding       = embeddings,
        persist_directory = config.VECTORDB_DIR,
    )
    print(f"🗄️ Vector database berhasil dibuat!")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load vector database yang sudah ada."""
    embeddings = OpenAIEmbeddings(
        model   = config.EMBEDDING_MODEL,
        api_key = config.OPENAI_API_KEY,
        openai_api_base = config.OPENAI_API_BASE,
    )
    return Chroma(
        persist_directory = config.VECTORDB_DIR,
        embedding_function = embeddings,
    )