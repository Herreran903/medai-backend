import io

import docx2txt
from pypdf import PdfReader


def read_any_to_text(filename: str, content: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        # docx2txt necesita archivo temporal
        tmp = f"/tmp/{filename}"
        with open(tmp, "wb") as f:
            f.write(content)
        return docx2txt.process(tmp) or ""
    # txt o fallback
    return content.decode("utf-8", errors="ignore")
