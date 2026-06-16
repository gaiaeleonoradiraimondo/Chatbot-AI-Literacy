# Chatbot-AI-Literacy
Prototipo di chatbot divulgativo sull'AI literacy, sviluppato nell'ambito di una tesi magistrale in Linguistica Teorica e Applicata. Il sistema adotta un'architettura RAG (Retrieval-Augmented Generation) in locale: le risposte vengono generate ancorandole a una knowledge base curata su temi quali il funzionamento dei Large Language Model, le allucinazioni, i bias e la valutazione critica degli output. L'obiettivo è offrire uno strumento di consultazione affidabile e tracciabile, pensato per colmare i divari di comprensione emersi da un'indagine sul pubblico adulto non specialista.

Le scelte di design privilegiano il grounding sulle fonti (per contenere le allucinazioni) e un registro sobrio e non compiacente. Si tratta di uno strumento di consultazione, non di un tutor adattivo.

**Stack**: Python · Ollama (LLM in locale) · LangChain · ChromaDB · embeddings nomic-embed-text · Streamlit (interfaccia).
