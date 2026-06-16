from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_demo_db():
    # 1) Modello LLM (generativo)
    llm = OllamaLLM(model="mistral")

    # 2) Modello embedding (per ricerca semantica)
    embedding = OllamaEmbeddings(model="all-minilm:l6-v2")

    # 3) Documento demo (poi lo sostituiremo con PDF/DOCX/TXT)
    documento = """
    Ollama è un framework che permette di eseguire modelli di large language model in locale,
    includendo sia modelli generativi (per scrivere risposte) che modelli di embedding
    (per trasformare testi in vettori e fare ricerca semantica).
    """

    # 4) Chunking (spezzare in pezzi)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(documento)

    # 5) Vector store in memoria (demo)
    db = Chroma.from_texts(chunks, embedding)

    return llm, db


def rag(llm, db, query: str) -> str:
    risultati = db.similarity_search(query, k=3)
    contesto = "\n\n".join([r.page_content for r in risultati])

    prompt = f"""
Rispondi usando SOLO queste informazioni, senza inventare nulla.

CONTENUTO:
{contesto}

DOMANDA:
{query}
"""
    return llm.invoke(prompt)


if __name__ == "__main__":
    llm, db = build_demo_db()

    domanda = "Che cos'è Ollama?"
    risposta = rag(llm, db, domanda)

    print("\nDOMANDA:", domanda)
    print("\nRISPOSTA:\n", risposta)