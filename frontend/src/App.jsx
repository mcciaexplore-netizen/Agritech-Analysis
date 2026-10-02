import { useRef, useState } from "react";
import axios from "axios";
import { UploadCloud, FileText, Loader2, RefreshCw, Trash2, ArrowRight, CheckCircle2, BarChart3, Leaf, X, AlertTriangle, Lightbulb, Cpu, Target, TrendingUp } from "lucide-react";
import "./App.css";

// Dynamic API URL for local development and cloud production deployment
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

const bullets = (text) => (text || "").split(" | ").map((t) => t.trim()).filter((t) => t && t.toUpperCase() !== "N/A");

const markQuote = (text, quote) => {
  const words = (quote || "").trim().split(/\s+/).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const match = words.length ? new RegExp(words.join("\\s+"), "i").exec(text) : null;
  if (!match) return text;
  return <>{text.slice(0, match.index)}<mark>{match[0]}</mark>{text.slice(match.index + match[0].length)}</>;
};

const SourceLinks = ({ sources, onOpen }) => sources?.length ? (
  <div className="source-links">
    {sources.map((src, i) => <button key={i} type="button" className="go-to" onClick={() => onOpen(src)}>Go to {src.kind} {src.page} <ArrowRight size={11} /></button>)}
  </div>
) : <span className="no-source">No verifiable source found in the document</span>;

const App = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [flippedCards, setFlippedCards] = useState({});
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [viewer, setViewer] = useState(null);
  const [reviewed, setReviewed] = useState({});
  const fileInput = useRef(null);

  const refs = result?.references;
  const open = (src) => setViewer({ src, doc: refs?.document, text: refs?.page_text?.[String(src.page)] || "" });
  const pageLabel = (list) => list?.length ? `${list[0].kind} ${list.map((p) => p.page).join(", ")}` : "";
  const flags = Object.fromEntries((result?.review || []).map((f) => [f.key, f]));
  const badge = (key, label, hasValue = true) => {
    if (flags[key]) {
      return reviewed[key]
        ? <span className="verify-badge reviewed"><CheckCircle2 size={11} /> {label} reviewed by human</span>
        : <span className="verify-badge unsure"><AlertTriangle size={11} /> Not sure of {label} - needs human review{flags[key].pages?.length ? ` (look at ${pageLabel(flags[key].pages)})` : ""}</span>;
    }
    return hasValue ? <span className="verify-badge sure"><CheckCircle2 size={11} /> Verified in document</span> : null;
  };

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
      setReviewed({});
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
        
        {result?.review?.length > 0 && <section className="review-panel" aria-labelledby="review-title">
          <h2 id="review-title"><AlertTriangle size={16} /> Human review needed ({result.review.filter((f) => !reviewed[f.key]).length} open)</h2>
          <ul>{result.review.map((f) => <li key={f.key} className={reviewed[f.key] ? "done" : ""}>
            <div><strong>Not sure of {f.label}</strong><span>{f.reason}</span>
              <span className="source-links">{f.pages?.length ? f.pages.map((src, i) => <button key={i} type="button" className="go-to" onClick={() => open(src)}>Look at {src.kind} {src.page} <ArrowRight size={11} /></button>) : <em className="no-source">No specific page found - check the whole document</em>}</span></div>
            <button type="button" className="go-to" onClick={() => setReviewed((prev) => ({ ...prev, [f.key]: !prev[f.key] }))}>{reviewed[f.key] ? "Undo" : "Mark reviewed"}</button>
          </li>)}</ul>
        </section>}

        {result?.data ? <section className="results" aria-labelledby="results-title">
          <div className="results-heading"><h2 id="results-title">Financial insights</h2><p>Select a card to view its full details.</p></div>
          <div className="cards-grid">
            {[
              { key: "profit_loss", title: "Profit / Loss", val: result.data.profit_loss },
              { key: "gross_profit", title: "Gross Profit", val: result.data.gross_profit },
              { key: "stage_of_startup", title: "Startup Stage", val: result.data.stage_of_startup },
              { key: "revenue_generation", title: "Revenue Generation", val: result.data.revenue_generation },
              { key: "pre_revenue", title: "Pre-Revenue Status", val: result.data.pre_revenue },
              { key: "pre_revenue_amount", title: "Pre-Revenue Amount", val: result.data.pre_revenue_amount },
              { key: "startup_name", title: "Startup Name", val: result.data.startup_name },
              { key: "Financial_ask", title: "Financial Ask", val: result.data.Financial_ask }
            ].map((card, idx) => <button key={idx} type="button" className={`insight-card ${flippedCards[idx] ? "is-expanded" : ""}`} aria-expanded={!!flippedCards[idx]} onClick={() => setFlippedCards((prev) => ({ ...prev, [idx]: !prev[idx] }))}>
              <span className="card-label">{card.title}</span><p className="card-value">{card.val ?? "Not available"}</p>{badge(card.key, card.title, card.val && card.val.toUpperCase() !== "N/A")}
              {refs?.financial_sources?.[card.key]?.length > 0 && <span className="source-links">{refs.financial_sources[card.key].map((src, i) => <span key={i} role="button" tabIndex={0} className="go-to" onClick={(e) => { e.stopPropagation(); open(src); }} onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); open(src); } }}>Go to {src.kind} {src.page} <ArrowRight size={11} /></span>)}</span>}
              <span className="card-action"><RefreshCw size={12} />{flippedCards[idx] ? "Show less" : "View details"}</span>
            </button>)}
          </div>
        </section> : <div className="insights-note"><BarChart3 size={20} /><p><strong>From information to insight.</strong> Startup stage, revenue status, gross profit, and profit/loss details — in one place.</p></div>}

        {result?.insights_error && <p className="error-message" role="alert">Insight cards unavailable: {result.insights_error}</p>}
        {result?.insights && (() => {
          const { cards, fundability: fund } = result.insights;
          const pct = Math.round(fund.probability * 100);
          const label = (k) => k.replace(/_/g, " ");
          const list = (text) => bullets(text).length ? <ul>{bullets(text).map((t, i) => <li key={i}>{t}</li>)}</ul> : <p className="card-value">Not available</p>;
          return <section className="results" aria-labelledby="insight-title">
            <div className="results-heading"><h2 id="insight-title">Startup insights</h2><p>Every claim links to the exact place it was found.</p></div>
            <div className="detail-grid">
              <article className="insight-card detail-card"><span className="card-label"><Lightbulb size={13} /> Innovation</span>
                {badge("innovation", "Innovation", bullets(cards.innovation.text).length > 0)}
                {list(cards.innovation.text)}
                <SourceLinks sources={cards.innovation.sources} onOpen={open} /></article>
              <article className="insight-card detail-card"><span className="card-label"><Cpu size={13} /> Tech in Agriculture</span>
                {badge("tech_in_agriculture", "Tech in Agriculture", bullets(cards.tech_in_agriculture.text).length > 0)}
                {list(cards.tech_in_agriculture.text)}
                <SourceLinks sources={cards.tech_in_agriculture.sources} onOpen={open} /></article>
              <article className="insight-card detail-card"><span className="card-label"><Target size={13} /> Strategy</span>
                {badge("strategy", "Strategy", bullets(cards.strategy.text).length > 0)}
                <span className={`strategy-badge ${cards.strategy.is_stated ? "stated" : "inferred"}`}>{cards.strategy.is_stated ? "Stated in document" : "No stated strategy - what they are building"}</span>
                {list(cards.strategy.text)}
                {bullets(cards.strategy.impact).length > 0 && <><span className="sub-label">Potential impact</span><ul>{bullets(cards.strategy.impact).map((t, i) => <li key={i}>{t}</li>)}</ul></>}
                <SourceLinks sources={cards.strategy.sources} onOpen={open} /></article>
            </div>
            <div className="fund-panel">
              <div className="fund-head"><span className="card-label"><TrendingUp size={13} /> Fundability probability</span>
                <strong className={`fund-pct band-${fund.band.toLowerCase()}`}>{pct}%</strong><span className="fund-band">{fund.band}</span></div>
              <div className="fund-bar" aria-hidden="true"><span style={{ width: `${pct}%` }} /></div>
              <table className="fund-table"><thead><tr><th>Factor</th><th>Adds</th></tr></thead>
                <tbody>{Object.keys(fund.adds_pct).map((k) => <tr key={k}><td>{label(k)}</td><td>{fund.adds_pct[k].toFixed(1)}%</td></tr>)}
                  <tr className="fund-total"><td>Total probability</td><td>{pct}%</td></tr></tbody></table>
              <p className="fund-note">{fund.note}</p>
            </div>
          </section>;
        })()}
      </main>

      {viewer && <div className="viewer-backdrop" role="dialog" aria-modal="true" onClick={() => setViewer(null)}>
        <div className="viewer" onClick={(e) => e.stopPropagation()}>
          <div className="viewer-head"><strong>{viewer.src.kind === "slide" ? "Slide" : "Page"} {viewer.src.page}</strong>
            <button type="button" className="viewer-close" onClick={() => setViewer(null)} aria-label="Close"><X size={18} /></button></div>
          {viewer.doc?.id && viewer.src.highlighted
            ? <iframe title="Highlighted document" className="viewer-frame" src={`${API_BASE_URL}/documents/${viewer.doc.id}#page=${viewer.src.page}`} />
            : <pre className="viewer-text">{markQuote(viewer.text, viewer.src.quote)}</pre>}
        </div>
      </div>}
      <footer className="site-footer"><span>MCCIA <span className="footer-divider">/</span> Document Insights</span><span>Commerce. Industry. Agriculture.</span></footer>
    </div>
  );
};

export default App;