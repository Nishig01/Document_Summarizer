"""PDF loading and text extraction."""

from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


class PDFLoader:
    """Loads PDF files and extracts text content per page."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {self.file_path.suffix}")

    def load(self) -> List[Dict[str, object]]:
        """Extract text from every page of the PDF.

        Returns a list of dicts, one per page, each containing:
            - page: 1-indexed page number
            - text: extracted text
        """
        reader = PdfReader(str(self.file_path))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": idx, "text": text})
        return pages

    def load_full_text(self) -> str:
        """Return the entire PDF text as a single string."""
        pages = self.load()
        return "\n\n".join(p["text"] for p in pages)