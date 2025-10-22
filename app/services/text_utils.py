# Este archivo contiene utilidades para extraer y limpiar texto desde archivos en formatos PDF, DOCX y texto plano.
# Proporciona funciones para manejar diferentes tipos de contenido y convertirlos en texto procesable.

import io
import os
import re
import tempfile
from typing import Optional

import docx2txt
import fitz


# Función para limpiar texto extraído de un archivo PDF.
# Elimina saltos de línea innecesarios, normaliza espacios y reconstruye palabras divididas por guiones.
def _clean_pdf_text(text: str) -> str:
    if not text:
        return ""

    # Normaliza saltos de línea para unificar el formato.
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Reconstruye palabras divididas por guiones al final de una línea.
    t = re.sub(r"(\w)[-‐-–—]\n(\w)", r"\1\2", t)

    # Reemplaza múltiples espacios o tabulaciones por un único espacio.
    t = re.sub(r"[ \t]+", " ", t)

    # Divide el texto en párrafos usando dos o más saltos de línea como delimitador.
    # Luego, elimina espacios innecesarios dentro de cada párrafo.
    paragraphs = [re.sub(r"\s*\n\s*", " ", p.strip()) for p in re.split(r"\n{2,}", t)]

    # Une los párrafos limpios con dos saltos de línea entre ellos.
    cleaned = "\n\n".join(p for p in paragraphs if p)
    return cleaned


# Función para extraer texto de un archivo PDF utilizando la biblioteca PyMuPDF (fitz).
# Procesa el contenido del PDF y organiza los bloques de texto en orden lógico.
def _extract_pdf_text_pymupdf(content: bytes) -> str:
    # Abre el archivo PDF desde un flujo de bytes.
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text: list[str] = []

    # Itera sobre cada página del documento.
    for page in doc:
        # Obtiene los bloques de texto de la página.
        blocks = page.get_text("blocks")
        # Ordena los bloques por su posición vertical y horizontal para mantener el flujo lógico del texto.
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))
        # Une el contenido de los bloques, eliminando espacios innecesarios.
        page_text = "\n".join((b[4] or "").strip() for b in blocks_sorted if b[4])
        # Agrega el texto de la página a la lista, eliminando espacios adicionales.
        pages_text.append(page_text.strip())

    # Une el texto de todas las páginas, separándolas con dos saltos de línea.
    return "\n\n".join(p for p in pages_text if p)


# Función principal para leer y convertir el contenido de un archivo en texto.
# Soporta archivos PDF, DOCX y texto plano, manejando diferentes codificaciones.
def read_any_to_text(filename: str, content: bytes) -> str:
    # Convierte el nombre del archivo a minúsculas para facilitar la comparación de extensiones.
    name = (filename or "").lower()

    # Manejo de archivos PDF.
    if name.endswith(".pdf"):
        # Extrae el texto crudo del PDF y lo limpia antes de devolverlo.
        raw = _extract_pdf_text_pymupdf(content)
        return _clean_pdf_text(raw)

    # Manejo de archivos DOCX.
    if name.endswith(".docx"):
        # Crea un archivo temporal para procesar el contenido del DOCX.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            # Extrae el texto del archivo DOCX utilizando la biblioteca docx2txt.
            text = docx2txt.process(tmp_path) or ""
        finally:
            # Elimina el archivo temporal para liberar recursos.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return text

    # Manejo de archivos de texto plano.
    try:
        # Intenta decodificar el contenido como UTF-8.
        return content.decode("utf-8")
    except UnicodeDecodeError:
        # Si falla, utiliza la codificación Latin-1 como alternativa, ignorando errores.
        return content.decode("latin-1", errors="ignore")
