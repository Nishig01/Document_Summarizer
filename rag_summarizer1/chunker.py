"""Text chunking utilities."""

from typing import List, Dict


class TextChunker:
    """Split text into overlapping chunks suitable for embedding.

    Uses a simple character-based sliding window with overlap. Chunks attempt
    to break at paragraph or sentence boundaries when possible.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split a plain string into overlapping chunks."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to end at a natural break point if we're not at the end.
            if end < len(text):
                for sep in ("\n\n", "\n", ". ", " "):
                    last = chunk.rfind(sep)
                    if last > self.chunk_size // 2:
                        chunk = chunk[: last + len(sep)]
                        end = start + len(chunk)
                        break

            chunks.append(chunk.strip())
            start += max(step, len(chunk) - self.chunk_overlap)

        return [c for c in chunks if c]

    def chunk_pages(self, pages: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Chunk a list of pages into chunks that retain page metadata."""
        results: List[Dict[str, object]] = []
        for page in pages:
            page_num = page["page"]
            for idx, chunk in enumerate(self.chunk_text(page["text"])):
                results.append(
                    {
                        "text": chunk,
                        "page": page_num,
                        "chunk_index": idx,
                    }
                )
        return results