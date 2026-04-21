import streamlit as st
import numpy as np
import wikipediaapi
import chromadb
import json
import re
import os
from typing import TypedDict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ── LangGraph orchestration ────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="FactSphere QA", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem;
    text-align: center; box-shadow: 0 8px 32px rgba(233,69,96,.15);
}
.main-header h1 { color:#e94560; font-size:2.8rem; margin:0; letter-spacing:-1px; }
.main-header p  { color:#a8b2d8; margin:.5rem 0 0; font-size:1rem; }
.result-card {
    background:#0d0d1a; border:1px solid #1e2a45; border-radius:14px;
    padding:1.5rem; margin-bottom:1.5rem; box-shadow:0 4px 16px rgba(0,0,0,.4);
}
.q-label { color:#a8b2d8; font-size:.8rem; font-weight:600;
           text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.q-text  { color:#e2e8f0; font-size:1.1rem; font-weight:600; margin-bottom:1rem; }
.verdict-answer  { background:#0d2318; border-left:4px solid #2ecc71;
                   padding:1rem 1.2rem; border-radius:8px; color:#c6f6d5; }
.verdict-clarify { background:#2a2000; border-left:4px solid #f39c12;
                   padding:1rem 1.2rem; border-radius:8px; color:#fefcbf; }
.verdict-refuse  { background:#2a0a0a; border-left:4px solid #e74c3c;
                   padding:1rem 1.2rem; border-radius:8px; color:#fed7d7; }
.badge { display:inline-block; padding:.2rem .7rem; border-radius:20px;
         font-size:.78rem; font-weight:700; margin-right:6px; }
.badge-factual     { background:#0f3460; color:#90cdf4; }
.badge-speculative { background:#3d1f00; color:#fbd38d; }
.badge-ambiguous   { background:#2d1558; color:#d6bcfa; }
.badge-supported   { background:#0d2318; color:#68d391; }
.badge-unsupported { background:#2a0a0a; color:#fc8181; }
hr.div { border:none; border-top:1px solid #1e2a45; margin:2rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🔮 FactSphere</h1>
  <p>Hallucination-Aware Multi-Agent QA · Powered by Groq Cloud LLM (Free)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — API Key + Model
# ─────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Configuration")

GROQ_API_KEY = st.sidebar.text_input(
    "🔑 Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="Get your free key at https://console.groq.com"
)

GROQ_MODEL = st.sidebar.selectbox(
    "🤖 Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b"],
    help="All free on Groq's free tier"
)

st.sidebar.info(
    "**Free Groq setup:**\n"
    "1. Go to [console.groq.com](https://console.groq.com)\n"
    "2. Sign up (free)\n"
    "3. API Keys → Create key\n"
    "4. Paste it above"
)

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""**🔁 Agent Pipeline**
1. 🧠 Planner — classify query
2. 📚 Retriever — fetch context
3. ✍️ Generator — draft answer
4. ✅ Verifier — check facts
5. 🎯 Confidence — final score""")

# ─────────────────────────────────────────────
# Shared Typed State  (LangGraph requires a TypedDict)
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    # ── inputs ──────────────────────────────
    query: str
    api_key: str
    model: str
    # ── planner outputs ─────────────────────
    query_type: str          # "factual" | "speculative" | "ambiguous"
    planner_reasoning: str
    # ── retriever outputs ───────────────────
    chunks: List[str]
    similarities: List[float]
    iterations: int
    # ── generator outputs ───────────────────
    answer: str
    # ── verifier outputs ────────────────────
    verdict: str             # "SUPPORTED" | "NOT_SUPPORTED"
    verifier_reason: str
    avg_similarity: float
    # ── confidence outputs ──────────────────
    confidence_score: float
    decision: str            # "ANSWER" | "CLARIFY" | "REFUSE"
    final_response: str

# ─────────────────────────────────────────────
# Groq LLM call  (fast, cloud, free)
# ─────────────────────────────────────────────
def groq_call(api_key: str, model: str, system: str, user: str,
              as_json: bool = False) -> str:
    client = Groq(api_key=api_key)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=400,
        temperature=0,
    )
    if as_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def safe_json(text: str, keys: list, defaults: dict) -> dict:
    for src in [text, re.search(r'\{.*?\}', text, re.DOTALL)]:
        try:
            raw  = src if isinstance(src, str) else src.group()
            data = json.loads(raw)
            return {k: data.get(k, defaults[k]) for k in keys}
        except Exception:
            pass
    return defaults

# ─────────────────────────────────────────────
# Embedding resources (cached — runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    wiki     = wikipediaapi.Wikipedia(user_agent='FactSphere/1.0', language='en')
    chroma   = chromadb.Client()
    docs = [
        "Machine learning is a subset of AI that builds systems learning from data.",
        "Neural networks mimic the human brain to recognize patterns in data.",
        "Large Language Models (LLMs) are deep learning models trained on vast text data.",
        "Albert Einstein developed the theory of relativity.",
        "Marie Curie pioneered research on radioactivity and won Nobel Prizes.",
        "Nikola Tesla designed the modern alternating current (AC) electricity supply system.",
        "The Industrial Revolution occurred roughly between 1760 and 1840.",
        "The Great Wall of China protected ancient Chinese states from nomadic groups.",
        "Mount Everest is Earth's highest mountain above sea level, in the Himalayas.",
        "The Amazon River is the largest river by discharge volume, in South America.",
    ]
    try:
        col = chroma.create_collection("factsphere")
        col.add(embeddings=embedder.encode(docs).tolist(),
                documents=docs,
                ids=[f"d{i}" for i in range(len(docs))])
    except Exception:
        col = chroma.get_collection("factsphere")
    return embedder, wiki, col

# ─────────────────────────────────────────────
# AGENT NODE FUNCTIONS  (each reads + writes AgentState)
# ─────────────────────────────────────────────

def run_planner(state: AgentState) -> AgentState:
    """
    Agent 1 — Planner
    Responsibility: Classify the query as factual / speculative / ambiguous.
    Decision made: The query_type it writes routes the pipeline differently
                   (speculative/ambiguous queries skip vector retrieval).
    Input:  state["query"], state["api_key"], state["model"]
    Output: state["query_type"], state["planner_reasoning"]
    """
    try:
        raw = groq_call(state["api_key"], state["model"],
            system='Classify the query as factual, speculative, or ambiguous. '
                   'Return ONLY JSON: {"query_type":"factual","reasoning":"..."}',
            user=f"Query: {state['query']}",
            as_json=True)
        parsed = safe_json(raw, ["query_type", "reasoning"],
                           {"query_type": "ambiguous", "reasoning": "N/A"})
    except Exception as e:
        parsed = {"query_type": "ambiguous", "reasoning": str(e)}

    return {**state,
            "query_type": parsed["query_type"].lower().strip(),
            "planner_reasoning": parsed["reasoning"]}


def run_retriever(state: AgentState) -> AgentState:
    """
    Agent 2 — Retriever
    Responsibility: Fetch relevant context from ChromaDB (vector search) and
                    Wikipedia (external knowledge fallback) for factual queries.
    Tools called:   embedder.encode(), collection.query(), cosine_similarity(),
                    wiki.page().summary
    Input:  state["query"], state["query_type"], state["iterations"]
    Output: state["chunks"], state["similarities"], state["iterations"]
    """
    embedder, wiki, collection = load_resources()

    # Routing decision: skip retrieval for speculative / ambiguous queries
    if state["query_type"] in ("speculative", "ambiguous"):
        return {**state, "chunks": [], "similarities": [], "iterations": 0}

    # On retry (iterations > 0), broaden the query
    search_query = state["query"] + " explanation" if state["iterations"] > 0 else state["query"]

    try:
        qemb   = embedder.encode(search_query).tolist()                    # sentence-transformers
        res    = collection.query(query_embeddings=[qemb], n_results=3)    # chromadb
        chunks = res["documents"][0] if res["documents"] else []
        sims: List[float] = []
        if chunks:
            cembs = embedder.encode(chunks)
            sims  = cosine_similarity([qemb], cembs)[0].tolist()           # scikit-learn
        best = max(sims) if sims else 0.0

        # External knowledge fallback
        if best < 0.4:
            page = wiki.page(state["query"])                               # wikipediaapi
            if page.exists():
                chunks.append("Wikipedia: " + page.summary[:500])
                sims.append(best)

        # Merge with existing chunks on retry (deduplication)
        old_chunks = state.get("chunks") or []
        old_sims   = state.get("similarities") or []
        merged_chunks = list(dict.fromkeys(old_chunks + chunks))
        merged_sims   = (old_sims + sims)[:len(merged_chunks)]

        return {**state,
                "chunks": merged_chunks,
                "similarities": merged_sims,
                "iterations": state["iterations"] + 1}
    except Exception:
        return {**state,
                "chunks": state.get("chunks") or [],
                "similarities": state.get("similarities") or [],
                "iterations": state["iterations"] + 1}


def run_generator(state: AgentState) -> AgentState:
    """
    Agent 3 — Generator
    Responsibility: Produce a grounded answer using ONLY the retrieved context.
    Tool called:    groq_call() → Groq LLM inference
    Input:  state["query"], state["chunks"], state["api_key"], state["model"]
    Output: state["answer"]
    """
    chunks = state.get("chunks") or []
    ctx    = "\n\n".join(chunks) if chunks else "No context available."
    try:
        answer = groq_call(state["api_key"], state["model"],
            system="Answer using ONLY the provided context. Be concise. "
                   "If unsupported by context say: UNSUPPORTED",
            user=f"Context:\n{ctx}\n\nQuestion: {state['query']}",
            as_json=False)
    except Exception as e:
        answer = f"UNSUPPORTED (error: {e})"
    return {**state, "answer": answer}


def run_verifier(state: AgentState) -> AgentState:
    """
    Agent 4 — Verifier
    Responsibility: Check whether the generated answer is supported by the context.
    Tools called:   groq_call() → Groq LLM, embedder.encode() + cosine_similarity()
    Decision made:  If avg_similarity < 0.35, override verdict to NOT_SUPPORTED,
                    which triggers a LangGraph conditional edge back to Retriever.
    Input:  state["answer"], state["chunks"], state["api_key"], state["model"]
    Output: state["verdict"], state["verifier_reason"], state["avg_similarity"]
    """
    embedder, _, _ = load_resources()
    chunks  = state.get("chunks") or []
    answer  = state.get("answer", "")
    ctx     = "\n\n".join(chunks) if chunks else ""

    try:
        raw = groq_call(state["api_key"], state["model"],
            system='Check if the answer is supported by context. '
                   'Return ONLY JSON: {"verdict":"SUPPORTED","reason":"..."}',
            user=f"Context:\n{ctx}\n\nAnswer: {answer}",
            as_json=True)
        parsed  = safe_json(raw, ["verdict", "reason"],
                            {"verdict": "NOT_SUPPORTED", "reason": "Parse failed."})
        verdict = parsed["verdict"].upper()
        reason  = parsed["reason"]
    except Exception as e:
        verdict, reason = "NOT_SUPPORTED", str(e)

    # Cosine similarity double-check (scikit-learn)
    aemb    = embedder.encode([answer])
    cembs   = embedder.encode(chunks) if chunks else np.array([])
    avg_sim = float(np.mean(cosine_similarity(aemb, cembs)[0])) if len(cembs) > 0 else 0.0
    if avg_sim < 0.35:
        verdict = "NOT_SUPPORTED"

    return {**state,
            "verdict": verdict,
            "verifier_reason": reason,
            "avg_similarity": avg_sim}


def run_confidence(state: AgentState) -> AgentState:
    """
    Agent 5 — Confidence
    Responsibility: Compute a final confidence score and make the ANSWER / CLARIFY / REFUSE
                    decision based on all accumulated evidence.
    Input:  state["query_type"], state["verdict"], state["iterations"]
    Output: state["confidence_score"], state["decision"], state["final_response"]
    """
    q_type     = state.get("query_type", "ambiguous")
    verdict    = state.get("verdict", "NOT_SUPPORTED")
    iterations = state.get("iterations", 1)
    answer     = state.get("answer", "")

    score = 1.0
    if q_type == "speculative": score -= 0.3
    elif q_type == "ambiguous": score -= 0.2
    if verdict == "NOT_SUPPORTED": score -= 0.3
    if iterations > 1: score -= 0.1 * (iterations - 1)
    score = max(0.0, min(1.0, score))

    if score >= 0.6:
        decision, fallback = "ANSWER",  None
    elif score >= 0.3:
        decision, fallback = "CLARIFY", "Please provide more context or clarify your query."
    else:
        decision, fallback = "REFUSE",  "No reliable evidence found for this query."

    final_response = answer if decision == "ANSWER" else fallback

    return {**state,
            "confidence_score": score,
            "decision": decision,
            "final_response": final_response}

# ─────────────────────────────────────────────
# LangGraph Conditional Routing Functions
# ─────────────────────────────────────────────

def route_after_planner(state: AgentState) -> str:
    """
    Conditional edge 1 — After Planner.
    Routes directly to Generator (skipping Retriever) for speculative/ambiguous queries.
    Routes to Retriever for factual queries.
    """
    if state["query_type"] in ("speculative", "ambiguous"):
        return "generator"
    return "retriever"


def route_after_verifier(state: AgentState) -> str:
    """
    Conditional edge 2 — After Verifier (self-correction / reflection loop).
    If verdict is NOT_SUPPORTED and we haven't retried yet, route BACK to Retriever.
    Otherwise proceed to Confidence.
    """
    if state["verdict"] == "NOT_SUPPORTED" and state.get("iterations", 0) < 2:
        return "retriever"          # ← reflection loop back
    return "confidence"

# ─────────────────────────────────────────────
# Build the LangGraph StateGraph
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register agent nodes
    g.add_node("planner",    run_planner)
    g.add_node("retriever",  run_retriever)
    g.add_node("generator",  run_generator)
    g.add_node("verifier",   run_verifier)
    g.add_node("confidence", run_confidence)

    # Entry point
    g.set_entry_point("planner")

    # Conditional edge 1: planner → retriever OR generator
    g.add_conditional_edges("planner", route_after_planner,
                            {"retriever": "retriever", "generator": "generator"})

    # Fixed edge: retriever always feeds generator
    g.add_edge("retriever", "generator")

    # Fixed edge: generator always feeds verifier
    g.add_edge("generator", "verifier")

    # Conditional edge 2: verifier → retriever (reflection) OR confidence
    g.add_conditional_edges("verifier", route_after_verifier,
                            {"retriever": "retriever", "confidence": "confidence"})

    # Fixed edge: confidence → END
    g.add_edge("confidence", END)

    return g.compile()


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# Input UI
# ─────────────────────────────────────────────
if not GROQ_API_KEY:
    st.info("👈 **Enter your free Groq API key in the sidebar to get started.**\n\n"
            "Get one in 1 minute at [console.groq.com](https://console.groq.com)")
    st.stop()

c1, c2 = st.columns([4, 1])
with c1:
    query = st.text_input("💬 Enter your question:",
                          placeholder="e.g. Who was Marie Curie?")
with c2:
    st.write(""); st.write("")
    submit = st.button("🚀 Ask FactSphere", use_container_width=True)

# ─────────────────────────────────────────────
# Pipeline  (LangGraph orchestrated)
# ─────────────────────────────────────────────
if submit and query.strip():
    # Ensure ChromaDB is initialised before graph runs
    load_resources()

    # Build and compile the graph each run (lightweight; stateless compilation)
    pipeline = build_graph()

    # Initial shared state
    initial_state: AgentState = {
        "query":            query.strip(),
        "api_key":          GROQ_API_KEY,
        "model":            GROQ_MODEL,
        "query_type":       "",
        "planner_reasoning": "",
        "chunks":           [],
        "similarities":     [],
        "iterations":       0,
        "answer":           "",
        "verdict":          "",
        "verifier_reason":  "",
        "avg_similarity":   0.0,
        "confidence_score": 0.0,
        "decision":         "",
        "final_response":   "",
    }

    with st.status("🤖 Running 5-agent LangGraph pipeline...", expanded=True) as status:
        try:
            st.write("🧠 **Agent 1/5 — Planner**: classifying query...")
            st.write("📚 **Agent 2/5 — Retriever**: fetching context...")
            st.write("✍️ **Agent 3/5 — Generator**: drafting answer...")
            st.write("✅ **Agent 4/5 — Verifier**: checking hallucination...")
            st.write("🎯 **Agent 5/5 — Confidence**: scoring...")

            # ── LangGraph invoke ─────────────────────────────────────────────
            result: AgentState = pipeline.invoke(initial_state)
            # ─────────────────────────────────────────────────────────────────

            q_type    = result["query_type"]
            reasoning = result["planner_reasoning"]
            chunks    = result["chunks"]
            sims      = result["similarities"]
            answer    = result["answer"]
            verdict   = result["verdict"]
            ver_reason= result["verifier_reason"]
            avg_sim   = result["avg_similarity"]
            iterations= result["iterations"]
            score     = result["confidence_score"]
            decision  = result["decision"]
            final_response = result["final_response"]

            st.write(f"   ✔ Type: **{q_type}** | Verdict: **{verdict}** | "
                     f"Iterations: **{iterations}** | Decision: **{decision}** | "
                     f"Confidence: **{score:.0%}**")

            status.update(label="✅ Done!", state="complete", expanded=False)

            st.session_state.history.insert(0, {
                "query": query, "model": GROQ_MODEL,
                "query_type": q_type, "reasoning": reasoning,
                "chunks": chunks, "sims": sims, "answer": answer,
                "verdict": verdict, "ver_reason": ver_reason,
                "avg_sim": avg_sim, "iterations": iterations,
                "score": score, "decision": decision,
                "final_response": final_response,
            })

        except Exception as e:
            status.update(label="❌ Error", state="error")
            st.error(f"❌ {e}")
            st.info("Check your Groq API key in the sidebar.")

elif submit:
    st.warning("Please enter a question first.")

# ─────────────────────────────────────────────
# Display multiple outputs
# ─────────────────────────────────────────────
if st.session_state.history:
    n = len(st.session_state.history)
    st.markdown(f"### 📋 Results — {n} {'query' if n==1 else 'queries'}")

    for idx, r in enumerate(st.session_state.history):
        qt   = r['query_type']
        qcls = f"badge-{qt}" if qt in ['factual','speculative','ambiguous'] else "badge-factual"
        vcls = "badge-supported" if r['verdict']=="SUPPORTED" else "badge-unsupported"

        st.markdown(f"""
<div class="result-card">
  <div class="q-label">Query #{n-idx}</div>
  <div class="q-text">💬 {r['query']}</div>
  <span class="badge {qcls}">{qt.capitalize()}</span>
  <span class="badge {vcls}">{r['verdict']}</span>
  <span class="badge" style="background:#1a1a3e;color:#90cdf4">⚡ {r['model']}</span>
</div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Query Type",  qt.capitalize())
        c2.metric("Verdict",     r['verdict'])
        c3.metric("Iterations",  r['iterations'])
        c4.metric("Decision",    r['decision'])

        score = r['score']
        st.write(f"**🎯 Confidence: {score:.0%}**")
        st.progress(score)

        d = r['decision']
        t = r['final_response']
        if d == 'ANSWER':
            st.markdown(f'<div class="verdict-answer">✅ <strong>Answer</strong><br><br>{t}</div>',
                        unsafe_allow_html=True)
        elif d == 'CLARIFY':
            st.markdown(f'<div class="verdict-clarify">⚠️ <strong>Needs Clarification</strong><br><br>{t}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-refuse">❌ <strong>Refused</strong><br><br>{t}</div>',
                        unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            with st.expander("📚 Retrieved Chunks"):
                if r['chunks']:
                    for i, ch in enumerate(r['chunks']):
                        sim = r['sims'][i] if i < len(r['sims']) else None
                        st.markdown(f"**Chunk {i+1}**" + (f" · sim={sim:.3f}" if sim else ""))
                        st.write(ch)
                        if i < len(r['chunks'])-1: st.divider()
                else:
                    st.write("No chunks retrieved.")
        with cb:
            with st.expander("🔍 Agent Details"):
                st.write(f"**Verdict:** {r['verdict']}")
                st.write(f"**Reason:** {r['ver_reason']}")
                st.write(f"**Avg Similarity:** {r['avg_sim']:.3f}")
                st.write(f"**Planner Reasoning:** {r['reasoning']}")

        if idx < n-1:
            st.markdown('<hr class="div">', unsafe_allow_html=True)
