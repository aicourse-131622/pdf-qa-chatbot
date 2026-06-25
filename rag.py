from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import Chroma
import config


def build_qa_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """Bangun QA chain dengan memory percakapan."""

    # LLM
    llm = ChatOpenAI(
        model       = config.LLM_MODEL,
        temperature = 0.1,          # rendah agar jawaban konsisten & faktual
        api_key     = config.OPENAI_API_KEY,
        openai_api_base = config.OPENAI_API_BASE,
    )

    # Retriever dengan similarity search
    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": config.TOP_K_RESULTS},
    )

    # Memory: simpan 5 ronde percakapan terakhir
    memory = ConversationBufferWindowMemory(
        k                    = 5,
        memory_key           = "chat_history",
        return_messages      = True,
        output_key           = "answer",
    )

    # Prompt untuk menjawab berdasarkan konteks
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            config.SYSTEM_PROMPT + """

Konteks dari dokumen:
{context}
"""
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    # Bangun chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm                    = llm,
        retriever              = retriever,
        memory                 = memory,
        combine_docs_chain_kwargs = {"prompt": qa_prompt},
        return_source_documents   = True,
        verbose                   = False,
    )

    return qa_chain


def format_sources(source_documents: list) -> str:
    """Format referensi sumber untuk ditampilkan."""
    if not source_documents:
        return ""

    sources = []
    seen    = set()

    for doc in source_documents:
        meta      = doc.metadata
        file_name = meta.get("source_file", "Dokumen")
        page      = meta.get("page", "?")
        key       = f"{file_name}_p{page}"

        if key not in seen:
            seen.add(key)
            sources.append(f"📄 **{file_name}** — Halaman {int(page)+1}")

    return "\n".join(sources)


def get_answer(qa_chain: ConversationalRetrievalChain,
               question: str) -> dict:
    """Dapatkan jawaban dari chain."""
    result = qa_chain.invoke({"question": question})
    return {
        "answer"  : result["answer"],
        "sources" : format_sources(result.get("source_documents", [])),
    }