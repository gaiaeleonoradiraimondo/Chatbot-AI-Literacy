from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# === PATH BASE PROGETTO ===
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "data" / "source_docs"
VECTOR_DIR = BASE_DIR / "vector_db"

# Modello embedding Ollama
EMBED_MODEL = "bge-m3"

# Parametri "safe" per evitare errori di contesto dell'embedding
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
MAX_CHARS = 600  # hard cap per essere certi di non sforare il context


def file_last_modified(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    """Riduce whitespace e caratteri 'strani' che possono aumentare i token."""
    return " ".join((text or "").split())


def load_documents_with_metadata(folder: Path):
    documents = []

    if not folder.exists():
        print(f"❌ Cartella non trovata: {folder}")
        return documents

    for file_path in folder.iterdir():
        if file_path.name.startswith("."):
            continue  # ignora file nascosti macOS (.DS_Store)

        suffix = file_path.suffix.lower()

        common_metadata = {
            "source": file_path.name,
            "path": str(file_path),
            "doc_type": suffix.replace(".", ""),
            "last_modified": file_last_modified(file_path),
        }

        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(file_path)).load()
            elif suffix == ".docx":
                loaded = Docx2txtLoader(str(file_path)).load()
            elif suffix == ".txt":
                loaded = TextLoader(str(file_path), encoding="utf-8").load()
            else:
                print(f"⚠️ File non supportato: {file_path.name}")
                continue

            # aggiungi metadati comuni e normalizza testo
            for doc in loaded:
                doc.metadata.update(common_metadata)
                doc.page_content = normalize_text(doc.page_content)

            documents.extend(loaded)

        except Exception as e:
            print(f"❌ Errore caricando {file_path.name}: {e}")

    return documents


def enforce_max_chars(chunks, max_chars: int = MAX_CHARS):
    """Se un chunk supera max_chars, lo spezza in subchunk mantenendo i metadati."""
    fixed = []

    for c in chunks:
        text = normalize_text(c.page_content)

        if len(text) <= max_chars:
            c.page_content = text
            fixed.append(c)
            continue

        meta = dict(c.metadata)
        meta["was_split_by_hard_cap"] = True

        start = 0
        part = 0
        while start < len(text):
            piece = text[start : start + max_chars]
            part_meta = dict(meta)
            part_meta["subchunk"] = part

            fixed.append(type(c)(page_content=piece, metadata=part_meta))

            start += max_chars
            part += 1

    return fixed


def main():
    print(f"📁 Source directory: {SOURCE_DIR}")
    print(f"💾 Vector DB directory: {VECTOR_DIR}")

    docs = load_documents_with_metadata(SOURCE_DIR)
    print(f"✅ Documenti caricati: {len(docs)}")

    if not docs:
        print("⚠️ Nessun documento trovato in data/source_docs.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"✅ Chunks generati (pre hard-cap): {len(chunks)}")

    chunks = enforce_max_chars(chunks, MAX_CHARS)
    print(f"✅ Chunks dopo hard-cap (<= {MAX_CHARS} chars): {len(chunks)}")

    # sanity check: trova il chunk più lungo per essere sicuri
    max_len = max(len(c.page_content or "") for c in chunks)
    print(f"🔎 Lunghezza massima chunk finale: {max_len} chars")

    embedding = OllamaEmbeddings(model=EMBED_MODEL)

    # Chroma salva automaticamente su disco nelle versioni recenti
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(VECTOR_DIR),
        collection_name="knowledge_base",
    )

    print("🎉 Ingestion completata con successo!")
    print("La knowledge base è stata salvata in 'vector_db'.")


if __name__ == "__main__":
    main()