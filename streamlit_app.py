"""
Demo Streamlit per il chatbot RAG su AI literacy.
Riusa le funzioni di app/rag_query.py per retrieval, reranking e gating.
Lanciare con:  streamlit run streamlit_app.py
"""

from pathlib import Path
import sys

# Permette di importare le funzioni da app/rag_query.py
APP_DIR = Path(__file__).resolve().parent / "app"
sys.path.insert(0, str(APP_DIR))

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from rag_query import (
    EMBED_MODEL,
    LLM_MODEL,
    COLLECTION_NAME,
    make_query_variants,
    dedup_docs,
    rerank,
    diversify_by_source,
    context_is_relevant,
    build_numbered_context,
    format_sources,
)

VECTOR_DIR = Path(__file__).resolve().parent / "vector_db"


# ===== System prompt =====

SYSTEM_PROMPT = """
Sei un assistente che aiuta adulti non esperti a capire come funzionano i chatbot,
i loro rischi e come usarli con consapevolezza.

REGOLE STRINGENTI:
1. Usa SOLO le informazioni nel CONTENUTO qui sotto. Non aggiungere nulla.
2. Non inventare nomi di professioni, leggi, dati, percentuali che non ci sono.
3. Se la domanda non è coperta, rispondi: "Su questo i documenti non dicono di più."
4. Rispondi SOLO alla domanda posta. Non divagare verso altri temi.
5. Lunghezza: 2-5 frasi per domande chiuse, max 8 per domande aperte.
6. Tra i passaggi del CONTENUTO può capitare che ci siano chunk su temi adiacenti ma non
   pertinenti alla domanda. In tal caso ignorali. Non cucire insieme informazioni di temi
   diversi solo perché compaiono nei passaggi forniti. Meglio una risposta breve che usa
   uno o due chunk pertinenti, che una risposta lunga che mescola temi.

STILE:
- Dai del "tu". Tono diretto, niente formule cortesi.
- Niente "è importante", "è bene", "vale la pena", "in conclusione", "inoltre".
- Niente aforismi a chiusura. Niente liste se la prosa basta.
- Vai dritto al punto. Per domande sì/no apri con la risposta.
"""


# ===== Inizializzazione (cached) =====
@st.cache_resource(show_spinner="Carico modello e vector DB...")
def init_resources():
    embedding = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
    )
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
    return db, llm


def answer_query(query, db, llm):
    """Replica la logica di rag_query.main() per una singola domanda."""
    variants = make_query_variants(query)
    retrieved = []
    for v in variants:
        try:
            docs_scores = db.similarity_search_with_score(v, k=10)
            retrieved.extend([d for d, _ in docs_scores])
        except Exception:
            retrieved.extend(db.similarity_search(v, k=10))

    retrieved = dedup_docs(retrieved)
    if not retrieved:
        return "Non lo trovo nei documenti.", ""

    ranked = rerank(retrieved, query)
    docs = diversify_by_source(ranked, max_per_source=2, max_total=3)
    if not context_is_relevant(docs, query, min_hits=1):
        return (
            "Non trovo un passaggio pertinente nei documenti. "
            "Prova a riformulare con parole più specifiche.",
            format_sources(docs),
        )

    contesto = build_numbered_context(docs, max_chars_total=7000)
    sources = format_sources(docs)

    prompt = f"""{SYSTEM_PROMPT}

DOMANDA:
{query}

CONTENUTO:
{contesto}
"""
    answer = llm.invoke(prompt)
    return answer, sources


# ===== UI =====

st.set_page_config(
    page_title="Chatbot AI Literacy",
    layout="centered",
)

st.title("Chatbot AI Literacy")
st.caption(
    "Demo basata su RAG. Le risposte vengono costruite a partire dalla "
    "knowledge base sui chatbot, i loro rischi e l'uso consapevole."
)

db, llm = init_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra cronologia
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Fonti utilizzate"):
                st.markdown(msg["sources"])

# Input nuovo messaggio
if query := st.chat_input("Scrivi la tua domanda sui chatbot..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Sto cercando nei documenti e formulando la risposta..."):
            answer, sources = answer_query(query, db, llm)
        st.markdown(answer)
        if sources:
            with st.expander("Fonti utilizzate"):
                st.markdown(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

# Sidebar
with st.sidebar:
    st.header("Comandi")
    if st.button("Cancella conversazione"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption(f"Modello LLM: `{LLM_MODEL}`")
    st.caption(f"Embedding: `{EMBED_MODEL}`")