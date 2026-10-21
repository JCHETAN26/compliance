"""Document ingestion (Phase 1): PDF text layer with per-character boxes and per-page
Tesseract OCR fallback, DOCX paragraphs/styles, XLSX/CSV structured rows with header
detection, sentence-preserving layout-aware chunking, and local bge-small-en-v1.5
embeddings (with bounded encoder windows for chunks longer than the model context).
"""

IMPLEMENTED = True
