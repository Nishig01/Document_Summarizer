"""Main RAG summarizer that ties all components together."""

import os
from typing import List, Optional

from anthropic import Anthropic

from rag_summarizer.chunker import TextChunker
from rag_summarizer.embeddings import EmbeddingGenerator
from rag_summarizer.pdf_loader import PDFLoader
from rag_summarizer.vector_store import SearchResult, VectorStore


DEFAULT_MODEL = "claude-opus-4-6"

SUMMARY_PROMPT = """You are an expert document summarizer. Based on the following \
excerpts from a PDF document, produce a clear, well-structured summary that captures \
the main ideas, key findings, and important details.

Document excerpts:
{context}

Write a comprehensive summary of the document. Use sections or bullet points where \
appropriate. Focus only on information present in the excerpts."""

QA_PROMPT = """You are a helpful assistant answering questions about a PDF document. \
Use only the information in the provided excerpts to answer. If the answer is not in \
the excerpts, say so clearly.

Document excerpts:
{context}

Question: {question}

Answer:"""


class RAGSummarizer:
    """Retrieval-Augmented Generation PDF summarizer powered by Claude."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY or pass api_key."
            )

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.store = VectorStore()
        self._loaded_pdf: Optional[str] = None

    def load_pdf(self, file_path: str) -> int:
        """Load a PDF, chunk it, embed the chunks, and index them.

        Returns the number of chunks indexed.
        """
        self.store.clear()
        loader = PDFLoader(file_path)
        pages = loader.load()
        if not pages:
            raise ValueError(f"No extractable text found in {file_path}")

        chunks = self.chunker.chunk_pages(pages)
        texts = [c["text"] for c in chunks]
        metadata = [{"page": c["page"], "chunk_index": c["chunk_index"]} for c in chunks]

        vectors = self.embedder.embed(texts)
        self.store.add(texts, vectors, metadata)
        self._loaded_pdf = file_path
        return len(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve the most relevant chunks for a query."""
        if len(self.store) == 0:
            raise RuntimeError("No PDF loaded. Call load_pdf() first.")
        query_vector = self.embedder.embed_one(query)
        return self.store.search(query_vector, top_k=top_k)

    def ask(self, question: str, top_k: int = 5, max_tokens: int = 1024) -> str:
        """Answer a question about the loaded PDF using retrieved context."""
        results = self.retrieve(question, top_k=top_k)
        context = self._format_context(results)
        prompt = QA_PROMPT.format(context=context, question=question)
        return self._complete(prompt, max_tokens=max_tokens)

    def summarize(self, top_k: int = 10, max_tokens: int = 2048) -> str:
        """Generate a summary of the loaded PDF.

        Retrieves representative chunks using a generic summary query, then asks
        Claude to produce a summary grounded in those chunks.
        """
        query = (
            "main topic, key findings, conclusions, important details, and overall "
            "purpose of the document"
        )
        results = self.retrieve(query, top_k=top_k)
        context = self._format_context(results)
        prompt = SUMMARY_PROMPT.format(context=context)
        return self._complete(prompt, max_tokens=max_tokens)

    def _format_context(self, results: List[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results, start=1):
            page = r.metadata.get("page", "?")
            parts.append(f"[Excerpt {i} | page {page}]\n{r.text}")
        return "\n\n".join(parts)

    def _complete(self, prompt: str, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")