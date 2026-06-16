from pathlib import Path
import re

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vector_db"

EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3.1:8b"
COLLECTION_NAME = "knowledge_base"


# ---------- Utility ----------

def normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def tokenize_it(text: str):
    """Tokenizzazione minimale (italiano): parole alfabetiche + numeri."""
    text = (text or "").lower()
    return re.findall(r"[a-zàèéìòù0-9]+", text)


STOPWORDS_IT = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "a", "da", "in", "su", "con", "per", "tra", "fra",
    "e", "o", "ma", "che", "come", "cosa", "perché", "perche",
    "del", "della", "dei", "degli", "delle", "al", "allo", "alla",
    "ai", "agli", "alle", "nel", "nello", "nella", "nei", "negli", "nelle",
    "mi", "ti", "si", "ci", "vi", "non", "è", "sono", "essere",
    "posso", "puoi", "può", "fare", "faccio", "fai", "uso", "usare",
    "una", "un", "quale", "quali"
}


def keywords_from_query(q: str):
    toks = tokenize_it(q)
    kws = [t for t in toks if t not in STOPWORDS_IT and len(t) >= 3]
    return kws


def format_sources(docs):
    seen = set()
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        if page is not None:
            lines.append(f"- {src} (pag. {page})")
        else:
            lines.append(f"- {src}")
    return "\n".join(lines)


def make_query_variants(q: str):
    """Varianti che aumentano recall per domande su chatbot e AI literacy."""
    q = q.strip()
    variants = [
    q,
    f"{q} chatbot ChatGPT",
    f"{q} allucinazione inventare falso",
    f"{q} bias neutralità stereotipi opinioni",
    f"{q} sycophancy assecondare lode compiacere",
    f"{q} rischio errore",
    f"{q} come funziona",
    f"{q} verificare affidabile",
    f"{q} esempio",
]
    return list(dict.fromkeys(variants))


def dedup_docs(docs):
    seen = set()
    out = []
    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            (d.page_content or "")[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def diversify_by_source(docs, max_per_source=1, max_total=4):
    out = []
    counts = {}
    for d in docs:
        src = d.metadata.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
        if counts[src] <= max_per_source:
            out.append(d)
        if len(out) >= max_total:
            break
    return out


def lexical_score(doc_text: str, query_keywords):
    """Reranking leggero: quante keyword della query compaiono nel chunk."""
    if not query_keywords:
        return 0
    text = (doc_text or "").lower()
    hits = 0
    for kw in query_keywords:
        if kw in text:
            hits += 1
    return hits


def rerank(docs, query: str):
    """Ordina i chunk combinando ordine embedding (già buono) + overlap lessicale."""
    qk = keywords_from_query(query)
    scored = []
    for idx, d in enumerate(docs):
        ls = lexical_score(d.page_content, qk)
        # penalità leggera per risultati molto "bassi" nella lista originale
        # (idx piccolo = meglio)
        score = ls * 10 - idx
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


def context_is_relevant(docs, query: str, min_hits=1):
    """Gating: se nessuna keyword compare, meglio non rispondere."""
    qk = keywords_from_query(query)
    if not qk:
        return True  # query troppo corta/generica: non blocco
    best = 0
    for d in docs:
        best = max(best, lexical_score(d.page_content, qk))
    return best >= min_hits


def build_numbered_context(docs, max_chars_total=7000):
    """Costruisce contesto numerato [1].. e taglia se troppo lungo."""
    parts = []
    total = 0
    for i, d in enumerate(docs, start=1):
        txt = normalize_ws(d.page_content)
        block = f"{txt}"
        if total + len(block) > max_chars_total:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


# ---------- Main ----------

def main():
    embedding = OllamaEmbeddings(model=EMBED_MODEL)

    db = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
    )

    llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.0
    )

    while True:
        query = input("\nDomanda (scrivi 'exit' per uscire): ").strip()
        if query.lower() == "exit":
            print("Chiusura.")
            break
        if not query:
            continue

        # 1) Multi-query retrieval (k alto)
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
            print("Non lo trovo nei documenti.")
            continue

        # 2) Rerank leggero per pertinenza (keyword overlap)
        ranked = rerank(retrieved, query)

        # 3) Diversifica e limita contesto
        docs = diversify_by_source(ranked, max_per_source=1, max_total=3)

        # 4) Gating: se contesto non è pertinente, chiedi riformulazione
        if not context_is_relevant(docs, query, min_hits=1):
            print("Non trovo un passaggio pertinente nei documenti. Prova a riformulare usando parole più vicine ai testi (es. 'policy', 'consentito', 'divieto', 'privacy', ecc.).")
            print("\nFonti candidate:\n", format_sources(docs))
            continue

        contesto = build_numbered_context(docs, max_chars_total=7000)
        sources = format_sources(docs)

        SYSTEM_PROMPT = """
Sei un assistente che aiuta adulti non esperti a capire come funzionano i chatbot,
i loro rischi e come usarli con consapevolezza.

REGOLE STRINGENTI:
1. Usa SOLO le informazioni nel CONTENUTO qui sotto. Non aggiungere nulla.
2. Non inventare nomi di professioni, leggi, dati, percentuali che non ci sono.
3. Se la domanda non è coperta, rispondi: "Su questo i documenti non dicono di più."
4. Rispondi SOLO alla domanda posta. Non divagare verso altri temi.
5. Lunghezza: 2-5 frasi per domande chiuse, max 8 per domande aperte.

STILE:
- Dai del "tu". Tono diretto, niente formule cortesi.
- Niente "è importante", "è bene", "vale la pena", "in conclusione", "inoltre".
- Niente aforismi a chiusura. Niente liste se la prosa basta.
- Vai dritto al punto. Per domande sì/no apri con la risposta.
"""

        prompt = f"""
{SYSTEM_PROMPT}

DOMANDA:
{query}

CONTENUTO:
{contesto}
"""

        answer = llm.invoke(prompt)

        print("\nRISPOSTA:\n", answer)
        print("\nFONTI RECUPERATE:\n", sources)


if __name__ == "__main__":
    main()