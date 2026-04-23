# RAG PDF Summarizer
 
A Retrieval-Augmented Generation (RAG) based PDF summarizer that uses Claude to generate summaries and answer questions about PDF documents.
 
## Overview
 
This project implements a RAG pipeline that:
1. Loads and extracts text from PDF documents
2. Chunks the text into manageable segments
3. Generates vector embeddings for each chunk
4. Stores embeddings in an in-memory vector store
5. Retrieves relevant chunks based on a query
6. Uses Claude to generate summaries or answer questions based on retrieved context
## Architecture
 
```
PDF → Text Extraction → Chunking → Embeddings → Vector Store
                                                      ↓
User Query → Query Embedding → Retrieval → Claude → Answer/Summary
```
 
## Installation
 
```bash
pip install -r requirements.txt
```
 
Set your Anthropic API key:
 
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```
 
## Usage
 
### Command Line
 
Summarize a PDF:
 
```bash
python main.py summarize path/to/document.pdf
```
 
Ask a question about a PDF:
 
```bash
python main.py ask path/to/document.pdf "What are the main findings?"
```
 
### Python API
 
```python
from rag_summarizer import RAGSummarizer
 
summarizer = RAGSummarizer()
summarizer.load_pdf("path/to/document.pdf")
 
# Get a summary
summary = summarizer.summarize()
print(summary)
 
# Ask a question
answer = summarizer.ask("What are the key conclusions?")
print(answer)
```
 
## Project Structure
 
```
Document_Summarizer/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                    # CLI entry point
└── rag_summarizer/
    ├── __init__.py
    ├── pdf_loader.py          # PDF text extraction
    ├── chunker.py             # Text chunking
    ├── embeddings.py          # Embedding generation
    ├── vector_store.py        # Vector storage and retrieval
    └── summarizer.py          # Main RAG summarizer
```
 
## Requirements
 
- Python 3.9+
- Anthropic API key
- Internet connection (for initial embedding model download)