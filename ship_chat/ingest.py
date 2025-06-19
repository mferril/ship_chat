# Builder only
"""
PDF to raw UTF-8 text.
Keeps every page.
"""
import fitz, pathlib

def extract_text(pdf_path: pathlib.Path) -> str:
    doc = fitz.open(pdf_path)
    pages = [p.get_text().strip() for p in doc if p.get_text().strip()]
    return "\n\n".join(pages)
