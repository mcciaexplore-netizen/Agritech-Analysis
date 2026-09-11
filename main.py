import os
import json
import tempfile
import warnings

# Suppress deprecation warnings from LangChain ecosystem
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

# Import extraction functions directly from content.py
from content import extract_pdf_text, extract_ppt_text

app = FastAPI()

# Retain CORS middleware for React frontend connectivity
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
    pre_revenue_amount: str = Field(description="If pre-revenue, specify the amount of revenue generated (e.g., '0', 'N/A', or details provided) and summarize all pre-revenue funding received so far, "
            "including personal contributions, competition prize money, CSR funding, non-dilutive capital, "
            "and fellowships/grants. If missing, return 'N/A'.")
    Financial_ask: str = Field(description="Financial ask or funding request details, or 'N/A' if missing.")
# ---------------------------------------------------------------------------
# 2. Document Loader via content.py
# ---------------------------------------------------------------------------
def load_document_via_content_py(file_path: str, file_ext: str) -> list[Document]:
    """Uses text extraction functions from content.py to create LangChain Documents."""
    if file_ext == "pdf":
        raw_text = extract_pdf_text(file_path)
    elif file_ext in ["ppt", "pptx"]:
        raw_text = extract_ppt_text(file_path)
    else:
        raise ValueError(f"Unsupported file format extension: .{file_ext}")

    return [Document(page_content=raw_text, metadata={"source": file_path})]

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
        # Load document using content.py logic
        documents = load_document_via_content_py(tmp_path, file_ext)

        # Chunk document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # Vector Store & Embedding Retrieval
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma.from_documents(chunks, embeddings)

        query = "financial ask funding requirement projections revenue generation profit loss stage gross profit investment needed"
        relevant_docs = vector_store.similarity_search(query, k=8)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # LLM Inference
        llm = ChatOllama(model="llama3.1", temperature=0)
        structured_llm = llm.with_structured_output(FinancialMetrics)

        prompt = f"""
        You are an expert financial analyst. Analyze the following document context extracted from a report/presentation.
        Extract the requested metrics precisely as stated in the text. 
        If a metric or status is missing, explicitly state "N/A".

        Context:
        {context}
        """

        result: FinancialMetrics = structured_llm.invoke(prompt)
        data_dict = result.model_dump()

        # Save output to local data/ directory
        os.makedirs("data", exist_ok=True)
        json_filename = f"{os.path.splitext(file.filename)[0]}_metrics.json"
        json_path = os.path.join("data", json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)

        # Response body matching frontend contract
        return {
            "filename": file.filename,
            "saved_to": json_path,
            "data": data_dict
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)