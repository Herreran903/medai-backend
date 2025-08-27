import io
import os
import re
import tempfile
from typing import Optional

import docx2txt
import fitz


def _clean_pdf_text(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Une palabras cortadas por guion al final de línea: "hiper-\n tensión" -> "hipertensión"
    t = re.sub(r"(\w)[-‐-–—]\n(\w)", r"\1\2", t)

    # Colapsa espacios/tabs repetidos
    t = re.sub(r"[ \t]+", " ", t)

    # Divide por párrafos (>= 2 saltos); dentro de cada párrafo, salto simple -> espacio
    paragraphs = [re.sub(r"\s*\n\s*", " ", p.strip()) for p in re.split(r"\n{2,}", t)]

    cleaned = "\n\n".join(p for p in paragraphs if p)
    return cleaned


def _extract_pdf_text_pymupdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text: list[str] = []

    for page in doc:
        # Bloques: (x0, y0, x1, y1, text, block_no, ...)
        blocks = page.get_text("blocks")
        # Ordena por top (y0) y luego left (x0)
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))
        page_text = "\n".join((b[4] or "").strip() for b in blocks_sorted if b[4])
        pages_text.append(page_text.strip())

    return "\n\n".join(p for p in pages_text if p)


def read_any_to_text(filename: str, content: bytes) -> str:
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        raw = _extract_pdf_text_pymupdf(content)
        return _clean_pdf_text(raw)

    if name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            text = docx2txt.process(tmp_path) or ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return text

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")
