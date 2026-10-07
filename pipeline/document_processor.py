from pathlib import Path
import fitz

def extract_pdf(path):
    path = Path(path)
    pages = []
    with fitz.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            pages.append({
                "page": page_number,
                "text": page.get_text("text"),
                "source": path.name,
            })
    return pages
