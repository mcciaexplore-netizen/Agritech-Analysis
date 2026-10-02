"""Pluggable OCR connector. Select engine with OCR_ENGINE in .env:

  rapid  (default) - RapidOCR / ONNX, lightweight, Render-friendly
  paddle           - PaddleOCR (needs: pip install -r requirements-paddle.txt)
  none             - native PDF text only, no OCR (scanned pages are skipped)

To add an engine: write a module exposing
extract_pdf_with_ocr(file_path, force_ocr=False, max_pages=None) -> str and register it below.
"""

import importlib
import os

ENGINES = {
    "rapid": "rapid_pdf",
    "paddle": "paddle_pdf",
    "none": None,
}


def extract_pdf(file_path: str, force_ocr: bool = False, max_pages: int | None = None) -> str:
    name = os.getenv("OCR_ENGINE", "rapid").strip().lower()
    if name not in ENGINES:
        raise ValueError(f"Unknown OCR_ENGINE '{name}'. Choose from: {', '.join(ENGINES)}")
    if ENGINES[name] is None:
        import pymupdf

        with pymupdf.open(file_path) as doc:
            return "\n\n".join(
                f"--- Page {i} ---\n{t}"
                for i, p in enumerate(doc, 1)
                if not (max_pages and i > max_pages)
                if (t := p.get_text("text").strip())
            )
    module = importlib.import_module(ENGINES[name])  # imported only when selected
    return module.extract_pdf_with_ocr(file_path, force_ocr=force_ocr, max_pages=max_pages)
