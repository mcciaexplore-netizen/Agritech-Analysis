import os
import json
import tempfile
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

from groq_extractor import extract_with_groq
from content import extract_ppt_text
from ocr_connector import extract_pdf
from insights import build_insights, build_references, document_path, normalize_metrics, review_flags

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Max pages (PDF) or slides (PPTX) read per document; 0 = unlimited.
MAX_PAGES = int(os.getenv("MAX_PAGES", "70"))

app = FastAPI()

# CORS: set ALLOWED_ORIGINS to the deployed frontend URL(s), comma-separated,
# e.g. https://agritech.vercel.app. Defaults to "*" for local development.
# ALLOWED_ORIGIN_REGEX optionally allows preview URLs, e.g.
# https://agritech-.*\.vercel\.app
ALLOWED_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


@app.get("/")
@app.get("/health")
async def health():
    """Render health check; also lets the frontend verify the API is reachable."""
    return {"status": "ok", "ocr_engine": os.getenv("OCR_ENGINE", "rapid"), "max_pages": MAX_PAGES}


# ---------------------------------------------------------------------------
# 1. Target Extraction Schema
# ---------------------------------------------------------------------------

class FinancialMetrics(BaseModel):
    startup_name: str = Field(
        description="Name of the startup/company, or 'N/A' if missing."
    )
    profit_loss: str = Field(
        description=(
            "Net profit, net loss, or net income reported, "
            "or 'N/A' if missing."
        )
    )
    gross_profit: str = Field(
        description="Gross profit reported, or 'N/A' if missing."
    )
    stage_of_startup: str = Field(
        description=(
            "Current stage of the startup (e.g., Idea, Seed, Early-stage, "
            "Growth, Pre-revenue), or 'N/A' if missing."
        )
    )
    revenue_generation: str = Field(
        description=(
            "Details on revenue generated, total sales, or monetization "
            "model, or 'N/A' if missing."
        )
    )
    pre_revenue: str = Field(
        description=(
            "Explicitly state whether the startup is pre-revenue "
            "(e.g., 'Yes', 'No', or details provided), or 'N/A' if missing."
        )
    )
    pre_revenue_amount: str = Field(
        description=(
            "All money the startup has actually received so far: any revenue "
            "plus every kind of funding (grants, equity, loans, awards, "
            "subsidies), with amount and source. If none stated, return 'N/A'."
        )
    )
    Financial_ask: str = Field(
        description="Financial ask or funding request details, or 'N/A' if missing."
    )


# ---------------------------------------------------------------------------
# 2. Document Text Extractor
# ---------------------------------------------------------------------------

def extract_full_text(file_path: str, file_ext: str) -> str:
    if file_ext == "pdf":
        return extract_pdf(file_path, max_pages=MAX_PAGES)

    if file_ext == "pptx":
        return extract_ppt_text(file_path, max_slides=MAX_PAGES)

    raise ValueError(f"Unsupported file format: .{file_ext}")


# ---------------------------------------------------------------------------
# 3. Extraction Endpoint
# ---------------------------------------------------------------------------

@app.post("/extract")
async def extract_metrics(file: UploadFile = File(...)):
    # Keep only the filename, excluding any supplied directory path.
    filename = os.path.basename(
        (file.filename or "").replace("\\", "/")
    )
    file_ext = os.path.splitext(filename)[1].lower().lstrip(".")

    if file_ext not in {"pdf", "pptx"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF or PPTX file. Convert older PPT files to PPTX.",
        )

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file_ext}",
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        raw_text = await run_in_threadpool(
            extract_full_text,
            tmp_path,
            file_ext,
        )

        if not raw_text.strip():
            raise HTTPException(
                status_code=422,
                detail="No readable text was found in this document.",
            )

        # Process all extracted text in bounded chunks through Groq.
        result: FinancialMetrics = await extract_with_groq(
            raw_text,
            FinancialMetrics,
        )

        data_dict = normalize_metrics(result.model_dump())

        os.makedirs("data", exist_ok=True)
        json_filename = f"{os.path.splitext(filename)[0]}_metrics.json"
        json_path = os.path.join("data", json_filename)

        with open(json_path, "w", encoding="utf-8") as output_file:
            json.dump(
                data_dict,
                output_file,
                indent=4,
                ensure_ascii=False,
            )

        # Insight cards + fundability. A failure here (e.g. Groq rate limit)
        # must not discard the financial metrics already extracted.
        insights, insights_error = None, None
        try:
            insights = await build_insights(raw_text, data_dict)
        except HTTPException as exc:
            insights_error = exc.detail

        review = review_flags(raw_text, data_dict, insights)
        references = build_references(raw_text, data_dict, insights, review, tmp_path, file_ext)

        return {
            "filename": filename,
            "saved_to": json_path,
            "data": data_dict,
            "insights": insights,
            "insights_error": insights_error,
            "review": review,
            "references": references,
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        await file.close()


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Serve the yellow-highlighted PDF copy produced for an upload."""
    path = document_path(doc_id)
    if not path:
        raise HTTPException(status_code=404, detail="Document not found or expired.")
    return FileResponse(path, media_type="application/pdf")
