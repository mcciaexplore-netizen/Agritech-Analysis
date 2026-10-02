"""RapidOCR (ONNX) connector. Same interface as paddle_pdf.extract_pdf_with_ocr."""

import threading

import numpy as np
import pymupdf

_ocr = None
_ocr_lock = threading.Lock()


def get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr import RapidOCR  # lazy: keeps import cost off other engines

        _ocr = RapidOCR()
    return _ocr


def _ocr_page(page) -> str:
    scale = min(2.0, 1800 / max(page.rect.width, page.rect.height))
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csRGB,
        alpha=False,
    )
    image = (
        np.frombuffer(pixmap.samples, dtype=np.uint8)
        .reshape(pixmap.height, pixmap.width, 3)
        .copy()
    )
    result = get_ocr()(image)
    if result is None or result.txts is None or result.boxes is None:
        return ""
    # Reading order: top-to-bottom, then left-to-right (line height tolerance).
    items = sorted(
        zip(result.boxes, result.txts),
        key=lambda it: (round(float(np.min(it[0][:, 1])) / 12), float(np.min(it[0][:, 0]))),
    )
    return "\n".join(t for _, t in items).strip()


def extract_pdf_with_ocr(file_path: str, force_ocr: bool = False, max_pages: int | None = None) -> str:
    pages = []
    with _ocr_lock:
        with pymupdf.open(file_path) as document:
            for page_number, page in enumerate(document, start=1):
                if max_pages and page_number > max_pages:
                    break
                text = page.get_text("text").strip()
                if force_ocr or not text:
                    text = _ocr_page(page)
                if text:
                    pages.append(f"--- Page {page_number} ---\n{text}")
    return "\n\n".join(pages)
