import os
import json
import warnings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS

# Import extraction functions directly from content.py
from content import extract_pdf_text, extract_ppt_text

# Load environment variables
load_dotenv()

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA_FOLDER = "./data"
OUTPUT_JSON_FILE = "extracted_financial_metrics.json"

# ---------------------------------------------------------------------------
# 1. Target Extraction Schema
# ---------------------------------------------------------------------------
class FinancialMetrics(BaseModel):
    profit_loss: str = Field(description="Net profit, net loss, or net income reported, or 'N/A' if missing.")
    gross_profit: str = Field(description="Gross Profit reported, or 'N/A' if missing.")
    stage_of_startup: str = Field(description="Current stage of the startup (e.g., Idea, Seed, Early-stage, Growth, Pre-revenue), or 'N/A' if missing.")
    revenue_generation: str = Field(description="Details on revenue generated, total sales, or monetization model, or 'N/A' if missing.")
    pre_revenue: str = Field(description="Explicitly state whether the startup is pre-revenue (e.g., 'Yes', 'No', or details provided), or 'N/A' if missing.")

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
# 3. RAG Pipeline (Ollama Embeddings + Ollama LLM)
# ---------------------------------------------------------------------------
def process_single_file(file_path: str, file_ext: str) -> dict:
    """Extracts target financial fields using content.py + Ollama embeddings + FAISS + Llama3.1."""
    documents = load_document_via_content_py(file_path, file_ext)

    # Chunk document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # Local Ollama Embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Retrieve context based on updated fields
    query = "profit loss gross profit startup stage revenue generation pre revenue business metrics"
    relevant_docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Structured Output via ChatOllama
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
    return result.model_dump()

# ---------------------------------------------------------------------------
# 4. Batch Folder Processing & JSON Export
# ---------------------------------------------------------------------------
def run_batch_extraction():
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
            metrics = process_single_file(file_path, file_ext)
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