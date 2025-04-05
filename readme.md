# Haven - Empathetic RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Pinecone, and Ollama, designed to provide empathetic and supportive conversations while sharing relevant information from documents.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Copy `.env_example` to `.env`
   - Add your Pinecone API key
   - Set your desired Pinecone index name (default: calm_index)

3. Make sure Ollama is running locally with the phi4-mini model:
```bash
ollama pull phi4-mini
```

## Usage

### Document Ingestion
To ingest documents into the calm_index:
```bash
python calm_ingestion.py
```

### Running the Chatbot
To start the chatbot interface:
```bash
streamlit run chatbot_rag.py
```

### Testing with Sample Data
To test with sample data:
```bash
python sample_ingestion.py
python sample_retrieval.py
```

## Features

- Document ingestion with PDF support
- Vector storage using Pinecone (calm_index)
- Empathetic conversation with emotional support
- Interactive chat interface with Streamlit
- Source document display for transparency
- Support for multiple document types 
