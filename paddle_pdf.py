import threading

import numpy as np
import pymupdf
from paddleocr import PaddleOCR


_ocr = None
_ocr_lock = threading.Lock()


def get_ocr():
    global _ocr

    if _ocr is None:
        _ocr = PaddleOCR(
            device="cpu",
            enable_mkldnn=False,
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name="PP-OCRv6_medium_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_recognition_batch_size=1,
            cpu_threads=2,
        )

    return _ocr


def extract_pdf_with_ocr(
    file_path: str,
    force_ocr: bool = False,
) -> str:
    pages = []

    # Limit concurrent OCR work and protect the shared engine.
    with _ocr_lock:
        with pymupdf.open(file_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if force_ocr or not text:
                    # Process one page at a bounded resolution.
                    scale = min(
                        2.0,
                        1800 / max(page.rect.width, page.rect.height),
                    )

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

                    lines = []

                    for result in get_ocr().predict(image):
                        lines.extend(result["rec_texts"])

                    text = "\n".join(lines).strip()

                    del image, pixmap

                if text:
                    pages.append(
                        f"--- Page {page_number} ---\n{text}"
                    )

    return "\n\n".join(pages)