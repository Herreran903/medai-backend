"""
Text Extraction Utilities for Document Processing.

This module provides utilities for extracting and cleaning text content from
various document formats, enabling the MedAI backend to process clinical notes
in multiple file formats.

Architecture Context:
    The text utilities service handles document-to-text conversion for the
    extraction API. When users upload files (PDF, DOCX, TXT), this module
    converts them to plain text for NER processing.

Supported Formats:
    - **PDF** (.pdf): Extracted using PyMuPDF (fitz) with block-level ordering
    - **DOCX** (.docx): Extracted using docx2txt
    - **Plain Text** (.txt): Direct decoding with encoding fallback

Text Cleaning:
    PDF extraction includes post-processing to:

    - Normalize line endings
    - Reconstruct hyphenated words split across lines
    - Collapse multiple spaces
    - Preserve paragraph structure

Usage:
    >>> from app.services.text_utils import read_any_to_text
    >>>
    >>> # From file upload in FastAPI
    >>> content = await file.read()
    >>> text = read_any_to_text(file.filename, content)
    >>>
    >>> # Process extracted text
    >>> result = extract_from_text(text, model="transformer")

See Also:
    - :mod:`app.routers.extract` for file upload handling
    - :mod:`app.services.pipeline` for text processing
"""

import io
import os
import re
import tempfile
from typing import Optional

import docx2txt
import fitz


def _clean_pdf_text(text: str) -> str:
    """
    Clean and normalize text extracted from PDF documents.

    Applies multiple cleaning steps to improve text quality:

    1. Normalize line endings (CRLF, CR → LF)
    2. Reconstruct hyphenated words split across lines
    3. Collapse multiple spaces/tabs to single space
    4. Preserve paragraph structure (double newlines)

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Cleaned text with normalized formatting.

    Example:
        >>> raw = "This is a hyphen-\\nated word.\\n\\nNew paragraph."
        >>> clean = _clean_pdf_text(raw)
        >>> print(clean)
        This is a hyphenated word.

        New paragraph.
    """
    if not text:
        return ""

    # Normalize line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Reconstruct hyphenated words (handles various hyphen characters)
    t = re.sub(r"(\w)[-‐-–—]\n(\w)", r"\1\2", t)

    # Collapse multiple spaces/tabs
    t = re.sub(r"[ \t]+", " ", t)

    # Split into paragraphs and clean each
    paragraphs = [re.sub(r"\s*\n\s*", " ", p.strip()) for p in re.split(r"\n{2,}", t)]

    # Rejoin paragraphs with double newlines
    cleaned = "\n\n".join(p for p in paragraphs if p)
    return cleaned


def _extract_pdf_text_pymupdf(content: bytes) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).

    Uses block-level extraction with spatial ordering to maintain
    logical reading order across columns and sections.

    Args:
        content: PDF file content as bytes.

    Returns:
        Extracted text with page separation.

    Algorithm:
        1. Open PDF from byte stream
        2. For each page, extract text blocks
        3. Sort blocks by vertical then horizontal position
        4. Join blocks with newlines
        5. Separate pages with double newlines

    Note:
        Block sorting approximates reading order but may not be perfect
        for complex multi-column layouts.
    """
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text: list[str] = []

    for page in doc:
        # Get text blocks with position information
        blocks = page.get_text("blocks")

        # Sort by vertical position (y), then horizontal (x)
        # Round to handle minor alignment variations
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))

        # Extract text from each block
        page_text = "\n".join((b[4] or "").strip() for b in blocks_sorted if b[4])
        pages_text.append(page_text.strip())

    return "\n\n".join(p for p in pages_text if p)


def read_any_to_text(filename: str, content: bytes) -> str:
    """
    Convert file content to plain text based on file extension.

    This is the primary interface for document-to-text conversion.
    It automatically detects the file format from the filename extension
    and applies the appropriate extraction method.

    Args:
        filename: Original filename with extension (e.g., "note.pdf").
            Used to determine file format. Case-insensitive.
        content: File content as bytes.

    Returns:
        Extracted plain text content.

    Supported Formats:
        ``.pdf``:
            Extracted using PyMuPDF with block ordering and text cleaning.
            Handles multi-page documents with paragraph preservation.

        ``.docx``:
            Extracted using docx2txt library.
            Preserves basic text structure from Word documents.

        Other extensions:
            Treated as plain text with encoding detection.
            Tries UTF-8 first, falls back to Latin-1.

    Example:
        >>> # PDF extraction
        >>> with open("clinical_note.pdf", "rb") as f:
        ...     content = f.read()
        >>> text = read_any_to_text("clinical_note.pdf", content)
        >>> print(text[:100])
        'Patient presents with respiratory distress...'

        >>> # DOCX extraction
        >>> text = read_any_to_text("report.docx", docx_bytes)

        >>> # Plain text with encoding fallback
        >>> text = read_any_to_text("note.txt", text_bytes)

    Note:
        For DOCX files, a temporary file is created during extraction
        and automatically cleaned up afterward.

    Warning:
        Binary files other than PDF/DOCX may produce garbled output.
        Ensure files are in supported formats before processing.
    """
    name = (filename or "").lower()

    # PDF extraction
    if name.endswith(".pdf"):
        raw = _extract_pdf_text_pymupdf(content)
        return _clean_pdf_text(raw)

    # DOCX extraction
    if name.endswith(".docx"):
        # docx2txt requires a file path, so use temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            text = docx2txt.process(tmp_path) or ""
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return text

    # Plain text with encoding fallback
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        # Fall back to Latin-1 (covers all byte values)
        return content.decode("latin-1", errors="ignore")
