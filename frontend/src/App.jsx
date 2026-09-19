import { useRef, useState } from "react";
import axios from "axios";
import { UploadCloud, FileText, Loader2, RefreshCw, Trash2, ArrowRight, CheckCircle2, BarChart3, Leaf } from "lucide-react";
import "./App.css";

// Dynamic API URL for local development and cloud production deployment
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

const App = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [flippedCards, setFlippedCards] = useState({});
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);

  const selectFile = (selected) => {
    if (!selected || loading) return;
    if (!/\.(pdf|pptx)$/i.test(selected.name)) {
      setError("Please choose a PDF or PPTX file. Convert older PPT files to PPTX.");
      if (fileInput.current) fileInput.current.value = "";
      return;
    }
    setFile(selected);
    setResult(null);
    setFlippedCards({});
    setError("");
  };

  const removeFile = () => {
    setFile(null);
    setResult(null);
    setFlippedCards({});
    setError("");
    if (fileInput.current) fileInput.current.value = "";
  };

  const handleUpload = async () => {
    if (!file || loading) return;
    setLoading(true);
    setResult(null);
    setError("");
    setFlippedCards({});
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await axios.post(`${API_BASE_URL}/extract`, formData);
      setResult(response.data);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string" ? detail :
        typeof detail?.message === "string" ? detail.message :
        err.response ? "The server could not process this document. Please retry." :
        "Cannot connect to the backend. Check that the API is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="brand">
          <img src={`${import.meta.env.BASE_URL}mccia-logo.png`} alt="MCCIA" className="brand-logo" />
          <span className="brand-name">Mahratta Chamber of Commerce,<br />Industries and Agriculture</span>
        </div>
        <a className="header-link" href="#upload">Document Insights <ArrowRight size={16} /></a>
      </header>

      <main className="main-content">
        <section className="intro" aria-labelledby="page-title">
          <span className="eyebrow"><Leaf size={15} /> AGRI-TECH ANALYSIS</span>
          <h1 id="page-title">Your documents.<br /><span>Clearer business insights.</span></h1>
          <p>Turn presentations and reports into key financial insights.<br className="desktop-break" /> Upload a document to discover the numbers that matter.</p>
        </section>

        <section className="upload-panel" id="upload" aria-labelledby="upload-title" aria-busy={loading}>
          <div className="panel-heading">
            <div className="section-icon"><FileText size={22} /></div>
            <div><h2 id="upload-title">Upload your document</h2><p>Start with a report or company presentation.</p></div>
            <span className="step-label">STEP 01</span>
          </div>
          <div className={`drop-zone ${dragging ? "is-dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); if (!loading) setDragging(true); }}
            onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }}
            onDrop={(event) => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files[0]); }}>
            <span className="upload-icon"><UploadCloud size={32} strokeWidth={1.6} /></span>
            <h3>Drag & drop your file here</h3>
            <p>or browse your device to choose a document</p>
            <input ref={fileInput} id="file-upload" className="file-input" type="file" accept=".pdf,.pptx" disabled={loading} onChange={(event) => selectFile(event.target.files?.[0])} />
            <button className="browse-button" type="button" disabled={loading} onClick={() => fileInput.current?.click()}>Browse files <ArrowRight size={16} /></button>
            <span className="file-types">Supported formats: PDF, PPTX</span>
          </div>

          {file && <div className="selected-file">
            <span className="file-icon"><FileText size={23} /></span>
            <div className="file-description"><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · {loading ? "Processing" : "Ready to extract"}</span></div>
            <button className="remove-button" type="button" onClick={removeFile} disabled={loading} aria-label={`Remove ${file.name}`}><Trash2 size={16} /> Remove</button>
          </div>}
          {error && <p className="error-message" role="alert">{error}</p>}
          <div className="upload-actions">
            <p><CheckCircle2 size={16} /> One document. Key metrics at a glance.</p>
            <button className="extract-button" type="button" onClick={handleUpload} disabled={!file || loading}>
              {loading ? <><Loader2 size={18} className="spin" /> Processing document...</> : <>Extract & Generate Cards <ArrowRight size={17} /></>}
            </button>
          </div>
          <span className="sr-only" role="status">{loading ? "Processing your document. Please wait." : file ? `Selected ${file.name}` : "No file selected"}</span>
        </section>

        {result && <div className="success-message" role="status"><CheckCircle2 size={20} /><div><strong>Your document insights are ready.</strong>{result.saved_to && <p>Saved JSON locally to: <span>{result.saved_to}</span></p>}</div></div>}
        
        {result?.data ? <section className="results" aria-labelledby="results-title">
          <div className="results-heading"><h2 id="results-title">Financial insights</h2><p>Select a card to view its full details.</p></div>
          <div className="cards-grid">
            {[
              { title: "Profit / Loss", val: result.data.profit_loss },
              { title: "Gross Profit", val: result.data.gross_profit },
              { title: "Startup Stage", val: result.data.stage_of_startup },
              { title: "Revenue Generation", val: result.data.revenue_generation },
              { title: "Pre-Revenue Status", val: result.data.pre_revenue },
              { title: "Pre-Revenue Amount", val: result.data.pre_revenue_amount },
              { title: "Startup Name", val: result.data.startup_name },
              { title: "Financial Ask", val: result.data.Financial_ask }
            ].map((card, idx) => <button key={idx} type="button" className={`insight-card ${flippedCards[idx] ? "is-expanded" : ""}`} aria-expanded={!!flippedCards[idx]} onClick={() => setFlippedCards((prev) => ({ ...prev, [idx]: !prev[idx] }))}>
              <span className="card-label">{card.title}</span><p className="card-value">{card.val ?? "Not available"}</p>
              <span className="card-action"><RefreshCw size={12} />{flippedCards[idx] ? "Show less" : "View details"}</span>
            </button>)}
          </div>
        </section> : <div className="insights-note"><BarChart3 size={20} /><p><strong>From information to insight.</strong> Startup stage, revenue status, gross profit, and profit/loss details — in one place.</p></div>}
      </main>
      <footer className="site-footer"><span>MCCIA <span className="footer-divider">/</span> Document Insights</span><span>Commerce. Industry. Agriculture.</span></footer>
    </div>
  );
};

export default App;