import os
import json
import tempfile
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from content import extract_pdf_text, extract_ppt_text

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 1. Target Extraction Schema
# ---------------------------------------------------------------------------
class FinancialMetrics(BaseModel):
    startup_name: str = Field(description="Name of the startup/Company, or 'N/A' if missing.")
    profit_loss: str = Field(description="Net profit, net loss, or net income reported, or 'N/A' if missing.")
    gross_profit: str = Field(description="Gross Profit reported, or 'N/A' if missing.")
    stage_of_startup: str = Field(description="Current stage of the startup (e.g., Idea, Seed, Early-stage, Growth, Pre-revenue), or 'N/A' if missing.")
    revenue_generation: str = Field(description="Details on revenue generated, total sales, or monetization model, or 'N/A' if missing.")
    pre_revenue: str = Field(description="Explicitly state whether the startup is pre-revenue (e.g., 'Yes', 'No', or details provided), or 'N/A' if missing.")
    pre_revenue_amount: str = Field(description="If pre-revenue, specify the amount of revenue generated and summarize pre-revenue funding received so far. If missing, return 'N/A'.")
    Financial_ask: str = Field(description="Financial ask or funding request details, or 'N/A' if missing.")

# ---------------------------------------------------------------------------
# 2. Document Text Extractor
# ---------------------------------------------------------------------------
def extract_full_text(file_path: str, file_ext: str) -> str:
    if file_ext == "pdf":
        return extract_pdf_text(file_path)
    elif file_ext in ["ppt", "pptx"]:
        return extract_ppt_text(file_path)
    else:
        raise ValueError(f"Unsupported file format extension: .{file_ext}")

# ---------------------------------------------------------------------------
# 3. Extraction Endpoint
# ---------------------------------------------------------------------------
@app.post("/extract")
async def extract_metrics(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "ppt", "pptx"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        raw_text = extract_full_text(tmp_path, file_ext)

        # Cap text length safely to fit within Ollama's local context window (~35k chars)
        max_chars = 35000 
        context_text = raw_text[:max_chars]

        # Bind directly to your local computer's Ollama instance
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        llm = ChatOllama(
            model="llama3.1:8b",
            base_url=ollama_url,
            temperature=0
        )

        structured_llm = llm.with_structured_output(FinancialMetrics)

        prompt = f"""
        You are an expert financial analyst. Analyze the following document context extracted from a report/presentation.
        Extract the requested metrics precisely as stated in the text. 
        If a metric or status is not explicitly mentioned, return "N/A".

        Document Content:
        {context_text}
        """

        result: FinancialMetrics = structured_llm.invoke(prompt)
        data_dict = result.model_dump()

        os.makedirs("data", exist_ok=True)
        json_filename = f"{os.path.splitext(file.filename)[0]}_metrics.json"
        json_path = os.path.join("data", json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)

        return {
            "filename": file.filename,
            "saved_to": json_path,
            "data": data_dict
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)