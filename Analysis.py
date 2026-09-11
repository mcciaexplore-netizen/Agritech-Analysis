import os
import tempfile
import warnings
import streamlit as st

# Suppress standard community deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# 1. Page Configuration & UI Title
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Financial Extractor", page_icon="📊", layout="wide")
st.title("📊 Financial Metrics Extractor (FAISS RAG)")
st.write("Upload a PDF or PowerPoint file to extract key financial indicators.")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API Key starting with sk-")

# ---------------------------------------------------------------------------
# 2. Define Extraction Target Schema (Pydantic)
# ---------------------------------------------------------------------------
class FinancialMetrics(BaseModel):
    revenue: str = Field(description="Revenue or Total Sales reported (with currency/period if present), or 'N/A' if missing.")
    gross_profit: str = Field(description="Gross Profit reported, or 'N/A' if missing.")
    net_profit: str = Field(description="Net Profit or Net Income/Loss reported, or 'N/A' if missing.")
    pre_revenue_status: str = Field(description="Indicates whether the company is pre-revenue (e.g., 'Yes', 'No', or details on current stage), or 'N/A' if missing.")

# ---------------------------------------------------------------------------
# 3. Document Loader Selector
# ---------------------------------------------------------------------------
def load_document(file_path: str, file_ext: str):
    """Loads document contents depending on extension."""
    if file_ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext in ["ppt", "pptx"]:
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format extension: .{file_ext}")
    return loader.load()

# ---------------------------------------------------------------------------
# 4. RAG Pipeline Execution
# ---------------------------------------------------------------------------
def process_document(file_path: str, file_ext: str, key: str) -> FinancialMetrics:
    # 1. Extract raw text pages/slides
    documents = load_document(file_path, file_ext)

    # 2. Split into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # 3. Embed & store in FAISS Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Perform target similarity search
    query = "financial metrics income statement revenue sales gross profit net profit net income earnings pre-revenue stage"
    relevant_docs = vector_store.similarity_search(query, k=5)
    
    # 5. Combine extracted relevant context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 6. Extract structured data via OpenAI model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key)
    structured_llm = llm.with_structured_output(FinancialMetrics)

    prompt = f"""
    You are an expert financial analyst. Analyze the following document context extracted from a report/presentation.
    Extract the requested financial metrics precisely as stated in the text. 
    If a metric is missing, explicitly state "N/A".

    Context:
    {context}
    """

    return structured_llm.invoke(prompt)

# ---------------------------------------------------------------------------
# 5. File Upload & Processing Interface
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a PDF or PowerPoint File", 
    type=["pdf", "ppt", "pptx"]
)

if uploaded_file is not None:
    if not api_key:
        st.warning("⚠️ Please provide your OpenAI API Key in the sidebar to run the extraction.")
    else:
        if st.button("Extract Financial Fields", type="primary"):
            with st.spinner("Processing document using FAISS Vector Index & GPT..."):
                file_extension = uploaded_file.name.split(".")[-1].lower()

                # Write uploaded bytes to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    # Run RAG Process
                    metrics = process_document(tmp_path, file_extension, api_key)
                    
                    st.success("✅ Extraction Completed!")
                    st.divider()
                    
                    # Display extracted results in nice metrics blocks
                    st.subheader("Extracted Metrics")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="1. Revenue", value=metrics.revenue)
                        st.metric(label="3. Net Profit", value=metrics.net_profit)
                    
                    with col2:
                        st.metric(label="2. Gross Profit", value=metrics.gross_profit)
                        st.metric(label="4. Pre-Revenue Status", value=metrics.pre_revenue_status)

                except Exception as e:
                    st.error(f"An error occurred while processing: {e}")
                finally:
                    # Cleanup temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)