import os
import json
import warnings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from pptx import Presentation
from langchain_ollama import ChatOllama



# Load environment variables from .env file
load_dotenv()

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA_FOLDER = "./data"
OUTPUT_JSON_FILE = "extracted_financial_metrics.json"

# ---------------------------------------------------------------------------
# 1. Target Extraction Schema
# ---------------------------------------------------------------------------
class FinancialMetrics(BaseModel):
    revenue: str = Field(description="Revenue or Total Sales reported, with currency and period, or 'N/A' if missing.")
    gross_profit: str = Field(description="Gross Profit reported, or 'N/A' if missing.")
    net_profit: str = Field(description="Net Profit or Net Income/Loss reported, or 'N/A' if missing.")
    pre_revenue_status: str = Field(description="Indicates whether the company is pre-revenue, or 'N/A' if missing.")

# ---------------------------------------------------------------------------
# 2. Document Loader (PDF & PPT/PPTX)
# ---------------------------------------------------------------------------
def load_document(file_path: str, file_ext: str):
    """Parses text content from PDF and PowerPoint files."""
    if file_ext == "pdf":
        reader = PdfReader(file_path)
        pages_content = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_content.append(text)
        full_pdf_text = "\n\n".join(pages_content)
        return [Document(page_content=full_pdf_text, metadata={"source": file_path})]

    elif file_ext in ["ppt", "pptx"]:
        prs = Presentation(file_path)
        text_content = []
        for slide_idx, slide in enumerate(prs.slides, start=1):
            text_content.append(f"--- Slide {slide_idx} ---")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        if paragraph.text.strip():
                            text_content.append(paragraph.text.strip())
                elif shape.has_table:
                    for row in shape.table.rows:
                        text_content.append(" | ".join([cell.text.strip() for cell in row.cells]))
        full_text = "\n".join(text_content)
        return [Document(page_content=full_text, metadata={"source": file_path})]
    else:
        raise ValueError(f"Unsupported file format extension: .{file_ext}")

# ---------------------------------------------------------------------------
# 3. RAG Pipeline (Ollama Embeddings + Gemini LLM)
# ---------------------------------------------------------------------------
def process_single_file(file_path: str, file_ext: str, api_key: str) -> dict:
    """Extracts target financial fields using Ollama embeddings + FAISS + Gemini."""
    documents = load_document(file_path, file_ext)

    # Chunk document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # Local Ollama Embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Retrieve context
    query = "financial metrics income statement revenue sales gross profit net profit net income earnings pre-revenue stage"
    relevant_docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Structured Output LLM via Supported Gemini Endpoint
    llm = ChatOllama(
    model="llama3.1",
    temperature=0
)
    structured_llm = llm.with_structured_output(FinancialMetrics)

    prompt = f"""
    You are an expert financial analyst. Analyze the following document context extracted from a report/presentation.
    Extract the requested financial metrics precisely as stated in the text. 
    If a metric is missing, explicitly state "N/A".

    Context:
    {context}
    """
    
    result: FinancialMetrics = structured_llm.invoke(prompt)
    return result.model_dump()

# ---------------------------------------------------------------------------
# 4. Batch Folder Processing & JSON Export
# ---------------------------------------------------------------------------
def run_batch_extraction():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API Key: ").strip()

    if not os.path.exists(DATA_FOLDER):
        print(f"Directory '{DATA_FOLDER}' does not exist. Creating it now...")
        os.makedirs(DATA_FOLDER)
        print(f"Please place your PDF and PPTX files into '{DATA_FOLDER}' and re-run.")
        return

    valid_extensions = ["pdf", "ppt", "pptx"]
    files = [f for f in os.listdir(DATA_FOLDER) if f.split(".")[-1].lower() in valid_extensions]

    if not files:
        print(f"No valid PDF or PPTX files found inside '{DATA_FOLDER}'.")
        return

    print(f"\nFound {len(files)} file(s) in '{DATA_FOLDER}'. Processing...\n")
    all_results = {}

    for idx, filename in enumerate(files, start=1):
        file_path = os.path.join(DATA_FOLDER, filename)
        file_ext = filename.split(".")[-1].lower()

        print(f"[{idx}/{len(files)}] Processing: {filename}...")

        try:
            metrics = process_single_file(file_path, file_ext, api_key)
            all_results[filename] = metrics
            print("    ✔ Successfully extracted metrics.")
        except Exception as e:
            all_results[filename] = {"error": str(e)}
            print(f"    ❌ Error: {e}")

    # Export merged results into JSON
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print("\n==========================================")
    print("✅ Processing complete!")
    print(f"📄 JSON output saved to: {os.path.abspath(OUTPUT_JSON_FILE)}")
    print("==========================================")

if __name__ == "__main__":
    run_batch_extraction()