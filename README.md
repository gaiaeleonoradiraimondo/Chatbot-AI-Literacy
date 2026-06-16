# Chatbot-AI-Literacy
Prototipo di chatbot divulgativo sull'AI literacy, sviluppato nell'ambito di una tesi magistrale in Linguistica Teorica e Applicata. Il sistema adotta un'architettura RAG (Retrieval-Augmented Generation) in locale: le risposte vengono generate ancorandole a una knowledge base curata su temi quali il funzionamento dei Large Language Model, le allucinazioni, i bias e la valutazione critica degli output. L'obiettivo è offrire uno strumento di consultazione affidabile e tracciabile, pensato per colmare i divari di comprensione emersi da un'indagine sul pubblico adulto non specialista.

Le scelte di design privilegiano il grounding sulle fonti (per contenere le allucinazioni) e un registro sobrio e non compiacente. Si tratta di uno strumento di consultazione, non di un tutor adattivo.

Documenti di sintesi originali, redatti dall'autrice sulla base della letteratura scientifica citata nella tesi

**Stack**: Python · Ollama (LLM in locale) · LangChain · ChromaDB · embeddings nomic-embed-text · Streamlit (interfaccia).
## Installazione
pip install -r requirements.txt
# serve Ollama installato e in esecuzione: https://ollama.com

## Avvio
# 1) costruisci la knowledge base
python app/ingestion.py
# 2) lancia l'interfaccia
streamlit run streamlit_app.py
