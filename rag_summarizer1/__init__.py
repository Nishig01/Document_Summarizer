"""RAG PDF Summarizer package."""

from rag_summarizer.summarizer import RAGSummarizer
from rag_summarizer.pdf_loader import PDFLoader
from rag_summarizer.chunker import TextChunker
from rag_summarizer.embeddings import EmbeddingGenerator
from rag_summarizer.vector_store import VectorStore

__all__ = [
    "RAGSummarizer",
    "PDFLoader",
    "TextChunker",
    "EmbeddingGenerator",
    "VectorStore",
]

__version__ = "0.1.0"