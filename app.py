import streamlit as st
import numpy as np
import wikipediaapi
import chromadb
import json
import re
import os
from typing import TypedDict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from langgraph.graph import StateGraph, END

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="FactSphere · Multi-Agent QA",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLES  +  THREE.JS BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(r"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --cyan:   #00f5ff;
    --purple: #7b2ff7;
    --pink:   #ff2d78;
    --green:  #00ff87;
    --gold:   #ffd700;
    --bg:     #02020f;
    --card:   rgba(8, 8, 28, 0.80);
    --border: rgba(0, 245, 255, 0.12);
}

/* ── Global reset ── */
* { box-sizing: border-box; }
body { background: var(--bg) !important; font-family: 'Inter', sans-serif; }

/* ── Make Streamlit transparent so canvas shows through ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
.block-container { background: transparent !important; }

/* ── Sidebar glassmorphism ── */
[data-testid="stSidebar"] > div:first-child {
    background: rgba(3, 3, 16, 0.94) !important;
    backdrop-filter: blur(28px) saturate(180%) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] input {
    background: rgba(0, 245, 255, 0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-size: 1.05rem !important;
    padding: 0.85rem 1.2rem !important;
    transition: all 0.35s;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(0,245,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,245,255,0.08), 0 0 28px rgba(0,245,255,0.18) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #2d3748 !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, rgba(0,245,255,0.08), rgba(123,47,247,0.10)) !important;
    border: 1px solid rgba(0,245,255,0.38) !important;
    border-radius: 14px !important;
    color: var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    padding: 0.9rem 1.5rem !important;
    transition: all 0.3s cubic-bezier(.25,.8,.25,1) !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, rgba(0,245,255,0.16), rgba(123,47,247,0.18)) !important;
    box-shadow: 0 0 24px rgba(0,245,255,0.32), 0 0 60px rgba(123,47,247,0.18) !important;
    transform: translateY(-2px) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--purple)) !important;
    border-radius: 4px !important;
    animation: shimmer 2s ease-in-out infinite;
}
@keyframes shimmer { 0%,100%{filter:brightness(1)} 50%{filter:brightness(1.35)} }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(0,245,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    transition: all 0.3s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0,245,255,0.22) !important;
    box-shadow: 0 0 18px rgba(0,245,255,0.07) !important;
}
[data-testid="stMetricValue"] {
    color: var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
}
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.73rem !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.25) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary { color: #64748b !important; }
[data-testid="stExpander"] summary:hover { color: var(--cyan) !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.18); border-radius: 2px; }

/* ── CRT scanlines overlay ── */
body::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 3px,
        rgba(0,245,255,0.006) 3px, rgba(0,245,255,0.006) 4px
    );
    pointer-events: none;
    z-index: 9998;
}

/* ════════════════════════════
   HERO HEADER
════════════════════════════ */
.hero-wrap   { text-align:center; padding:2.5rem 1rem 1.2rem; position:relative; }
.hero-logo   {
    font-family:'Orbitron',monospace; font-size:3.8rem; font-weight:900;
    background:linear-gradient(135deg,#00f5ff 0%,#7b2ff7 50%,#ff2d78 100%);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
    display:inline-block; letter-spacing:3px;
    animation: logoFloat 4s ease-in-out infinite;
}
@keyframes logoFloat {
    0%,100% { filter:drop-shadow(0 0 16px rgba(0,245,255,.5)) drop-shadow(0 0 40px rgba(123,47,247,.3)); transform:translateY(0); }
    50%      { filter:drop-shadow(0 0 32px rgba(0,245,255,.9)) drop-shadow(0 0 70px rgba(123,47,247,.5)); transform:translateY(-5px); }
}
.hero-sub {
    margin-top:.6rem; color:#334155; font-size:.78rem;
    letter-spacing:4px; text-transform:uppercase; font-weight:500;
}
.hero-div {
    max-width:520px; margin:1.4rem auto 0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(0,245,255,.35),rgba(123,47,247,.35),transparent);
}

/* ════════════════════════════
   AGENT PIPELINE
════════════════════════════ */
.pipeline-wrap {
    padding:1.6rem 1.2rem 1rem;
    background:rgba(4,4,18,0.65);
    border:1px solid var(--border);
    border-radius:22px;
    backdrop-filter:blur(16px);
    margin:.6rem 0 1.2rem;
}
.pipeline-row {
    display:flex; align-items:center; justify-content:center;
    gap:0; flex-wrap:nowrap; overflow-x:auto;
}

/* individual agent node */
.ag { display:flex; flex-direction:column; align-items:center; gap:.45rem; position:relative; flex-shrink:0; }

/* rotating gradient ring (visible when active) */
.ag-ring {
    position:absolute;
    top:-6px; left:-6px; right:-6px; bottom:-6px;
    border-radius:50%;
    background:conic-gradient(from 0deg,var(--cyan),var(--purple),var(--pink),var(--cyan));
    opacity:0;
    animation:ringSpin 1.8s linear infinite;
    transition:opacity .4s;
    pointer-events:none;
}
@keyframes ringSpin { to{transform:rotate(360deg)} }

/* mask ring to look like a border */
.ag-ring-mask {
    position:absolute;
    top:-3px; left:-3px; right:-3px; bottom:-3px;
    border-radius:50%;
    background:rgba(4,4,18,0.9);
    pointer-events:none;
}

.ag-inner {
    width:68px; height:68px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.65rem;
    background:rgba(6,6,22,0.85);
    border:1px solid rgba(0,245,255,0.14);
    position:relative; z-index:1;
    transition:all .5s cubic-bezier(.34,1.56,.64,1);
}

.ag-name {
    font-family:'Orbitron',monospace; font-size:.52rem; color:#1e293b;
    letter-spacing:1.5px; text-transform:uppercase;
    text-align:center; max-width:72px;
    transition:color .35s;
}

/* idle */
.ag.idle .ag-inner { opacity:.42; }

/* active */
.ag.active .ag-ring   { opacity:1; }
.ag.active .ag-inner  {
    transform:scale(1.22);
    border-color:rgba(0,245,255,.75);
    background:rgba(0,245,255,0.09);
    box-shadow:0 0 22px rgba(0,245,255,.45),0 0 55px rgba(0,245,255,.18),inset 0 0 18px rgba(0,245,255,.06);
}
.ag.active .ag-inner::after {
    content:'';
    position:absolute; inset:-10px; border-radius:50%;
    background:radial-gradient(circle,rgba(0,245,255,.16) 0%,transparent 70%);
    animation:nodePulse 1.1s ease-in-out infinite;
}
@keyframes nodePulse {
    0%,100%{transform:scale(1);opacity:1}
    50%{transform:scale(1.35);opacity:.45}
}
.ag.active .ag-name { color:var(--cyan); }

/* done */
.ag.done .ag-inner {
    border-color:rgba(0,255,135,.5);
    background:rgba(0,255,135,0.06);
    box-shadow:0 0 14px rgba(0,255,135,.28);
}
.ag.done .ag-name { color:var(--green); }

/* wire connector between nodes */
.ag-wire {
    width:48px; height:2px;
    background:rgba(0,245,255,.06);
    position:relative; flex-shrink:0;
    margin-bottom:26px; overflow:hidden;
}
.ag-pulse {
    position:absolute; top:0; left:-60%; width:60%; height:100%;
    background:linear-gradient(90deg,transparent,var(--cyan),transparent);
    animation:wirePulse 3.5s ease-in-out infinite;
}
.ag-wire.lit .ag-pulse { animation-duration:.9s; }
@keyframes wirePulse  { to{left:100%} }

/* status line */
.ag-status {
    text-align:center; margin-top:.9rem;
    font-family:'JetBrains Mono',monospace; font-size:.76rem;
    color:#1e293b; min-height:18px; transition:color .3s;
}
.ag-status.live  { color:var(--cyan); }
.ag-status.retry { color:var(--gold); animation:blink .9s step-end infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.4} }

/* ════════════════════════════
   RESULT CARDS
════════════════════════════ */
.r-card {
    background:var(--card);
    border:1px solid var(--border);
    border-radius:22px; padding:2rem;
    margin:1.4rem 0;
    backdrop-filter:blur(18px);
    transition:border-color .3s,box-shadow .35s,transform .3s;
    position:relative; overflow:hidden;
}
.r-card::before {
    content:''; position:absolute;
    top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,var(--cyan),var(--purple),var(--pink));
    opacity:.55;
}
.r-card:hover {
    border-color:rgba(0,245,255,.25);
    box-shadow:0 22px 55px rgba(0,0,0,.65),0 0 28px rgba(0,245,255,.06);
    transform:translateY(-4px) perspective(1000px) rotateX(-1deg);
}
.r-qlabel { font-family:'Orbitron',monospace; font-size:.68rem; color:var(--cyan); opacity:.55; letter-spacing:3px; text-transform:uppercase; margin-bottom:.35rem; }
.r-qtext  { font-size:1.08rem; color:#e2e8f0; font-weight:600; margin-bottom:1rem; }

/* Verdict boxes */
.v-answer  { background:rgba(0,255,135,.06);  border-left:3px solid var(--green);  border-radius:10px; padding:1rem 1.3rem; color:#a7f3d0; font-size:.93rem; line-height:1.65; }
.v-clarify { background:rgba(255,215,0,.06);  border-left:3px solid var(--gold);   border-radius:10px; padding:1rem 1.3rem; color:#fde68a; font-size:.93rem; line-height:1.65; }
.v-refuse  { background:rgba(255,45,120,.06); border-left:3px solid var(--pink);   border-radius:10px; padding:1rem 1.3rem; color:#fecaca; font-size:.93rem; line-height:1.65; }

/* Badges */
.badge { display:inline-flex; align-items:center; padding:.22rem .78rem; border-radius:20px; font-size:.7rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-right:6px; font-family:'JetBrains Mono',monospace; }
.b-factual     { background:rgba(0,245,255,.08);  color:var(--cyan);   border:1px solid rgba(0,245,255,.25); }
.b-speculative { background:rgba(255,215,0,.08);  color:var(--gold);   border:1px solid rgba(255,215,0,.25); }
.b-ambiguous   { background:rgba(123,47,247,.08); color:#a78bfa;       border:1px solid rgba(123,47,247,.25); }
.b-supported   { background:rgba(0,255,135,.08);  color:var(--green);  border:1px solid rgba(0,255,135,.25); }
.b-unsupported { background:rgba(255,45,120,.08); color:var(--pink);   border:1px solid rgba(255,45,120,.25); }
.b-model       { background:rgba(8,8,28,.6);      color:#475569;       border:1px solid rgba(255,255,255,.07); }

/* Query wrapper */
.q-wrap { background:rgba(4,4,18,.6); border:1px solid var(--border); border-radius:18px; padding:1.2rem 1.5rem; backdrop-filter:blur(14px); margin:.6rem 0; }

/* Section heading */
.sec-head { display:flex; align-items:center; gap:1rem; margin:2rem 0 .8rem; }
.sec-head-label { font-family:'Orbitron',monospace; font-size:.75rem; color:var(--cyan); letter-spacing:3px; white-space:nowrap; }
.sec-head-count { background:rgba(0,245,255,.08); border:1px solid rgba(0,245,255,.2); border-radius:20px; padding:.15rem .7rem; font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--cyan); }
.sec-head-line  { flex:1; height:1px; background:linear-gradient(90deg,rgba(0,245,255,.2),transparent); }

hr.sec { border:none; border-top:1px solid rgba(0,245,255,.07); margin:1.6rem 0; }

/* Sidebar */
.sid-title { font-family:'Orbitron',monospace; font-size:.72rem; color:var(--cyan); letter-spacing:3px; text-transform:uppercase; padding:.5rem 0; border-bottom:1px solid var(--border); margin-bottom:1rem; }
.sid-list  { list-style:none; padding:0; margin:0; }
.sid-list li { padding:.45rem 0; font-size:.82rem; color:#334155; display:flex; align-items:center; gap:.55rem; border-bottom:1px solid rgba(255,255,255,.04); transition:color .2s; }
.sid-list li:hover { color:#64748b; }
</style>

<!-- ═══════════════ THREE.JS NEURAL BACKGROUND ═══════════════ -->
<script>
(function () {
    if (window._fsThreeReady) return;
    window._fsThreeReady = true;

    /* Create canvas on document.body so it persists across Streamlit re-renders */
    var cv = document.createElement('canvas');
    cv.id = 'fs-bg';
    cv.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:-9999;pointer-events:none;';
    document.body.appendChild(cv);

    function boot() {
        var T = window.THREE;
        var W = innerWidth, H = innerHeight;
        var renderer = new T.WebGLRenderer({ canvas: cv, alpha: false, antialias: true });
        renderer.setSize(W, H);
        renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
        renderer.setClearColor(0x02020f, 1);

        var scene = new T.Scene();
        var cam   = new T.PerspectiveCamera(65, W / H, 0.1, 600);
        cam.position.set(0, 0, 42);

        /* ── Neural nodes ── */
        var N = 90, nodes = [], sgeo = new T.SphereGeometry(0.13, 7, 7);
        var palette = [0x00f5ff, 0x7b2ff7, 0xff2d78, 0x00ff87];
        for (var i = 0; i < N; i++) {
            var mat = new T.MeshBasicMaterial({ color: palette[i % palette.length], transparent: true, opacity: .7 });
            var m = new T.Mesh(sgeo, mat);
            m.position.set((Math.random() - .5) * 100, (Math.random() - .5) * 60, (Math.random() - .5) * 45 - 8);
            m.userData = { spd: Math.random() * .014 + .004, off: Math.random() * 6.28 };
            scene.add(m); nodes.push(m);
        }

        /* ── Connection lines ── */
        var lg = new T.Group(); scene.add(lg);
        function buildLines() {
            while (lg.children.length) lg.remove(lg.children[0]);
            var maxD = 20;
            for (var i = 0; i < N; i++) for (var j = i + 1; j < N; j++) {
                var d = nodes[i].position.distanceTo(nodes[j].position);
                if (d < maxD) {
                    var g = new T.BufferGeometry().setFromPoints([nodes[i].position.clone(), nodes[j].position.clone()]);
                    lg.add(new T.Line(g, new T.LineBasicMaterial({ color: 0x00f5ff, transparent: true, opacity: (1 - d / maxD) * .16 })));
                }
            }
        }
        buildLines();

        /* ── Wireframe polyhedra ── */
        var shapes = [], geos = [
            new T.IcosahedronGeometry(2.8, 0), new T.OctahedronGeometry(2.2, 0),
            new T.TetrahedronGeometry(2.0, 0),  new T.IcosahedronGeometry(1.8, 1),
            new T.OctahedronGeometry(3.2, 0),
        ], scols = [0x00f5ff, 0x7b2ff7, 0xff2d78, 0x00ff87, 0x7b2ff7];
        for (var i = 0; i < 5; i++) {
            var s = new T.Mesh(geos[i], new T.MeshBasicMaterial({ color: scols[i], wireframe: true, transparent: true, opacity: .18 }));
            s.position.set((Math.random() - .5) * 80, (Math.random() - .5) * 45, (Math.random() - .5) * 22 - 12);
            s.userData = { rx: (Math.random() - .5) * .013, ry: (Math.random() - .5) * .016, rz: (Math.random() - .5) * .009 };
            scene.add(s); shapes.push(s);
        }

        /* ── Ambient particles ── */
        var pn = 350, pg = new T.BufferGeometry(), pp = new Float32Array(pn * 3);
        for (var i = 0; i < pn; i++) {
            pp[i * 3]     = (Math.random() - .5) * 140;
            pp[i * 3 + 1] = (Math.random() - .5) * 90;
            pp[i * 3 + 2] = (Math.random() - .5) * 70;
        }
        pg.setAttribute('position', new T.BufferAttribute(pp, 3));
        var pts = new T.Points(pg, new T.PointsMaterial({ color: 0x00f5ff, size: .07, transparent: true, opacity: .38 }));
        scene.add(pts);

        /* ── Grid floor ── */
        var grid = new T.GridHelper(200, 48, 0x00f5ff, 0x00f5ff);
        grid.material.opacity = .04; grid.material.transparent = true;
        grid.position.y = -28; scene.add(grid);

        /* ── Mouse parallax ── */
        var mx = 0, my = 0;
        window.addEventListener('mousemove', function (e) {
            mx = (e.clientX / innerWidth  - .5) * 2;
            my = -(e.clientY / innerHeight - .5) * 2;
        });

        window.addEventListener('resize', function () {
            W = innerWidth; H = innerHeight;
            cam.aspect = W / H; cam.updateProjectionMatrix();
            renderer.setSize(W, H);
        });

        var frame = 0;
        (function animate() {
            requestAnimationFrame(animate);
            frame++;
            /* Float nodes */
            nodes.forEach(function (n, i) {
                n.position.y   += Math.sin(frame * n.userData.spd + n.userData.off) * .014;
                n.material.opacity = .3 + Math.abs(Math.sin(frame * .028 + i * .68)) * .55;
            });
            if (frame % 100 === 0) buildLines();
            /* Rotate shapes */
            shapes.forEach(function (s) {
                s.rotation.x += s.userData.rx;
                s.rotation.y += s.userData.ry;
                s.rotation.z += s.userData.rz;
            });
            pts.rotation.y += .0004; pts.rotation.x += .00018;
            grid.rotation.y += .0006;
            /* Camera drift */
            cam.position.x += (mx * 9 - cam.position.x) * .022;
            cam.position.y += (my * 5 - cam.position.y) * .022;
            cam.lookAt(scene.position);
            renderer.render(scene, cam);
        })();
    }

    /* Dynamically load Three.js r160 */
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js';
    s.onload = function () { if (window.THREE) boot(); };
    document.head.appendChild(s);
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-logo">🔮 FactSphere</div>
  <div class="hero-sub">Hallucination-Aware · 5-Agent LangGraph QA · Powered by Groq LLM</div>
  <div class="hero-div"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT STATE + BACKEND  (logic identical to original)
# ══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    query: str; api_key: str; model: str
    query_type: str; planner_reasoning: str
    chunks: List[str]; similarities: List[float]; iterations: int
    answer: str
    verdict: str; verifier_reason: str; avg_similarity: float
    confidence_score: float; decision: str; final_response: str


def groq_call(api_key, model, system, user, as_json=False):
    client = Groq(api_key=api_key)
    kw = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=400, temperature=0
    )
    if as_json: kw["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kw).choices[0].message.content.strip()


def safe_json(text, keys, defaults):
    for src in [text, re.search(r'\{.*?\}', text, re.DOTALL)]:
        try:
            raw  = src if isinstance(src, str) else src.group()
            data = json.loads(raw)
            return {k: data.get(k, defaults[k]) for k in keys}
        except Exception:
            pass
    return defaults


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
        col.add(embeddings=embedder.encode(docs).tolist(), documents=docs, ids=[f"d{i}" for i in range(len(docs))])
    except Exception:
        col = chroma.get_collection("factsphere")
    return embedder, wiki, col


def run_planner(state):
    try:
        raw = groq_call(state["api_key"], state["model"],
            system='Classify as factual, speculative, or ambiguous. Return ONLY JSON: {"query_type":"factual","reasoning":"..."}',
            user=f"Query: {state['query']}", as_json=True)
        parsed = safe_json(raw, ["query_type", "reasoning"], {"query_type": "ambiguous", "reasoning": "N/A"})
    except Exception as e:
        parsed = {"query_type": "ambiguous", "reasoning": str(e)}
    return {**state, "query_type": parsed["query_type"].lower().strip(), "planner_reasoning": parsed["reasoning"]}


def run_retriever(state):
    embedder, wiki, col = load_resources()
    if state["query_type"] in ("speculative", "ambiguous"):
        return {**state, "chunks": [], "similarities": [], "iterations": 0}
    sq = state["query"] + " explanation" if state["iterations"] > 0 else state["query"]
    try:
        qemb  = embedder.encode(sq).tolist()
        res   = col.query(query_embeddings=[qemb], n_results=3)
        chunks = res["documents"][0] if res["documents"] else []
        sims: List[float] = []
        if chunks:
            cembs = embedder.encode(chunks)
            sims  = cosine_similarity([qemb], cembs)[0].tolist()
        best = max(sims) if sims else 0.0
        if best < 0.4:
            page = wiki.page(state["query"])
            if page.exists():
                chunks.append("Wikipedia: " + page.summary[:500])
                sims.append(best)
        old_c = state.get("chunks") or []
        old_s = state.get("similarities") or []
        merged_c = list(dict.fromkeys(old_c + chunks))
        merged_s = (old_s + sims)[:len(merged_c)]
        return {**state, "chunks": merged_c, "similarities": merged_s, "iterations": state["iterations"] + 1}
    except Exception:
        return {**state, "chunks": state.get("chunks") or [], "similarities": state.get("similarities") or [], "iterations": state["iterations"] + 1}


def run_generator(state):
    chunks = state.get("chunks") or []
    ctx    = "\n\n".join(chunks) if chunks else "No context available."
    try:
        answer = groq_call(state["api_key"], state["model"],
            system="Answer using ONLY the provided context. Be concise. If unsupported say: UNSUPPORTED",
            user=f"Context:\n{ctx}\n\nQuestion: {state['query']}")
    except Exception as e:
        answer = f"UNSUPPORTED (error: {e})"
    return {**state, "answer": answer}


def run_verifier(state):
    embedder, _, _ = load_resources()
    chunks  = state.get("chunks") or []
    answer  = state.get("answer", "")
    qt      = state.get("query_type", "factual")
    ctx     = "\n\n".join(chunks) if chunks else ""

    # For speculative/ambiguous queries there is no retrieved context;
    # skip cosine check and just ask the LLM to verify the answer alone.
    if qt in ("speculative", "ambiguous") or not chunks:
        try:
            raw = groq_call(state["api_key"], state["model"],
                system='Rate whether the answer is reasonable. Return ONLY JSON: {"verdict":"SUPPORTED","reason":"..."}',
                user=f"Question: {state['query']}\n\nAnswer: {answer}", as_json=True)
            parsed  = safe_json(raw, ["verdict", "reason"], {"verdict": "SUPPORTED", "reason": "No context to check."})
            verdict = parsed["verdict"].upper()
            reason  = parsed["reason"]
        except Exception as e:
            verdict, reason = "SUPPORTED", str(e)
        return {**state, "verdict": verdict, "verifier_reason": reason, "avg_similarity": 0.0}

    # Factual query with retrieved context — full cosine + LLM check
    try:
        raw = groq_call(state["api_key"], state["model"],
            system='Check if answer is supported by context. Return ONLY JSON: {"verdict":"SUPPORTED","reason":"..."}',
            user=f"Context:\n{ctx}\n\nAnswer: {answer}", as_json=True)
        parsed  = safe_json(raw, ["verdict", "reason"], {"verdict": "NOT_SUPPORTED", "reason": "Parse failed."})
        verdict = parsed["verdict"].upper()
        reason  = parsed["reason"]
    except Exception as e:
        verdict, reason = "NOT_SUPPORTED", str(e)
    aemb    = embedder.encode([answer])
    cembs   = embedder.encode(chunks)
    avg_sim = float(np.mean(cosine_similarity(aemb, cembs)[0]))
    if avg_sim < 0.35:
        verdict = "NOT_SUPPORTED"
    return {**state, "verdict": verdict, "verifier_reason": reason, "avg_similarity": avg_sim}


def run_confidence(state):
    q, v, it = state.get("query_type","ambiguous"), state.get("verdict","NOT_SUPPORTED"), state.get("iterations",1)
    ans  = state.get("answer", "")
    score = 1.0
    if q == "speculative": score -= 0.3
    elif q == "ambiguous": score -= 0.2
    if v == "NOT_SUPPORTED": score -= 0.3
    if it > 1: score -= 0.1 * (it - 1)
    score = max(0.0, min(1.0, score))
    if score >= 0.6:   decision, fb = "ANSWER",  None
    elif score >= 0.3: decision, fb = "CLARIFY", "Please provide more context or clarify your query."
    else:              decision, fb = "REFUSE",  "No reliable evidence found for this query."
    return {**state, "confidence_score": score, "decision": decision, "final_response": ans if decision == "ANSWER" else fb}


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown('<div class="sid-title">⚙ Configuration</div>', unsafe_allow_html=True)

GROQ_API_KEY = st.sidebar.text_input(
    "🔑 Groq API Key", type="password", placeholder="gsk_...",
    help="Get your free key at https://console.groq.com"
)
GROQ_MODEL = st.sidebar.selectbox(
    "🤖 Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
     "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b"],
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
st.sidebar.markdown('<div class="sid-title">🔁 Agent Pipeline</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<ul class="sid-list">
  <li>🧠 <strong>Planner</strong> — classify query</li>
  <li>📚 <strong>Retriever</strong> — fetch context</li>
  <li>✍️ <strong>Generator</strong> — draft answer</li>
  <li>✅ <strong>Verifier</strong> — fact-check</li>
  <li>🎯 <strong>Confidence</strong> — final score</li>
</ul>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE HTML RENDERER
# ══════════════════════════════════════════════════════════════════════════════
_AGENTS = [("🧠","PLANNER"),("📚","RETRIEVER"),("✍️","GENERATOR"),("✅","VERIFIER"),("🎯","CONFIDENCE")]

def render_pipeline(active: int = -1, done: list = [], retry: bool = False, status: str = "") -> str:
    nodes_html = ""
    for i, (icon, name) in enumerate(_AGENTS):
        cls = "active" if i == active else ("done" if i in done else "idle")
        lit = " lit" if (i in done or i == active) else ""
        connector = "" if i == len(_AGENTS) - 1 else f'<div class="ag-wire{lit}"><div class="ag-pulse"></div></div>'
        nodes_html += f"""
        <div class="ag {cls}">
            <div class="ag-ring"></div>
            <div class="ag-ring-mask"></div>
            <div class="ag-inner">{icon}</div>
            <div class="ag-name">{name}</div>
        </div>{connector}"""

    sc = " retry" if retry else (" live" if active >= 0 else "")
    return f"""
    <div class="pipeline-wrap">
        <div class="pipeline-row">{nodes_html}</div>
        <div class="ag-status{sc}">{status}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []


# ══════════════════════════════════════════════════════════════════════════════
#  GATE: API KEY CHECK
# ══════════════════════════════════════════════════════════════════════════════
if not GROQ_API_KEY:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 2rem;background:rgba(0,245,255,0.025);
                border:1px solid rgba(0,245,255,0.1);border-radius:22px;margin:2rem 0;">
      <div style="font-size:3.2rem;margin-bottom:1rem;">🔑</div>
      <div style="font-family:'Orbitron',monospace;color:#00f5ff;font-size:1rem;letter-spacing:3px;margin-bottom:.6rem;">
        API KEY REQUIRED
      </div>
      <div style="color:#334155;font-size:.88rem;line-height:1.6;">
        Enter your free Groq API key in the sidebar<br>to initialize the FactSphere 5-agent pipeline.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY INPUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="q-wrap">', unsafe_allow_html=True)
col_in, col_btn = st.columns([5, 1])
with col_in:
    query = st.text_input(
        "q", label_visibility="collapsed",
        placeholder="⚡  Ask anything — e.g. 'Who was Marie Curie?'  ·  'What is an LLM?'"
    )
with col_btn:
    st.write(""); st.write("")
    submit = st.button("LAUNCH ⚡", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PLACEHOLDER  (always visible)
# ══════════════════════════════════════════════════════════════════════════════
pipeline_ph = st.empty()

if not submit and not st.session_state.history:
    pipeline_ph.markdown(
        render_pipeline(status="⬆  Enter a question above and click LAUNCH to run the 5-agent pipeline"),
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
#  EXECUTE PIPELINE  (step-by-step with live UI updates)
# ══════════════════════════════════════════════════════════════════════════════
if submit and query.strip():
    try:
        load_resources()

        state: AgentState = {
            "query": query.strip(), "api_key": GROQ_API_KEY, "model": GROQ_MODEL,
            "query_type": "",        "planner_reasoning": "",
            "chunks":  [],           "similarities": [], "iterations": 0,
            "answer":  "",
            "verdict": "",           "verifier_reason": "", "avg_similarity": 0.0,
            "confidence_score": 0.0, "decision": "",        "final_response": "",
        }

        # ── Agent 1: Planner ──────────────────────────────────────────────
        pipeline_ph.markdown(
            render_pipeline(active=0, status="🧠  Planner · classifying your query..."),
            unsafe_allow_html=True
        )
        state = run_planner(state)
        qt = state["query_type"]
        done_nodes = [0]

        # ── Agent 2: Retriever (factual queries only) ─────────────────────
        if qt == "factual":
            pipeline_ph.markdown(
                render_pipeline(active=1, done=done_nodes, status="📚  Retriever · searching knowledge base + Wikipedia..."),
                unsafe_allow_html=True
            )
            state = run_retriever(state)
            done_nodes = [0, 1]

        # ── Agent 3: Generator ────────────────────────────────────────────
        pipeline_ph.markdown(
            render_pipeline(active=2, done=done_nodes, status="✍️  Generator · drafting grounded answer..."),
            unsafe_allow_html=True
        )
        state = run_generator(state)
        done_nodes = list(set(done_nodes + [2]))

        # ── Agent 4: Verifier ─────────────────────────────────────────────
        pipeline_ph.markdown(
            render_pipeline(active=3, done=done_nodes, status="✅  Verifier · cross-checking for hallucinations..."),
            unsafe_allow_html=True
        )
        state = run_verifier(state)
        done_nodes = list(set(done_nodes + [3]))

        # ── Self-correction loop (retry) ──────────────────────────────────
        if state["verdict"] == "NOT_SUPPORTED" and state.get("iterations", 0) < 2:
            pipeline_ph.markdown(
                render_pipeline(active=1, done=[0], retry=True,
                                status="🔄  Retry Loop · low confidence — broadening search..."),
                unsafe_allow_html=True
            )
            state = run_retriever(state)

            pipeline_ph.markdown(
                render_pipeline(active=2, done=[0, 1], retry=True,
                                status="✍️  Re-generating with expanded context..."),
                unsafe_allow_html=True
            )
            state = run_generator(state)

            pipeline_ph.markdown(
                render_pipeline(active=3, done=[0, 1, 2], retry=True,
                                status="✅  Re-verifying answer..."),
                unsafe_allow_html=True
            )
            state = run_verifier(state)
            done_nodes = [0, 1, 2, 3]

        # ── Agent 5: Confidence ───────────────────────────────────────────
        pipeline_ph.markdown(
            render_pipeline(active=4, done=done_nodes, status="🎯  Confidence · computing final score..."),
            unsafe_allow_html=True
        )
        state = run_confidence(state)

        # ── All done ──────────────────────────────────────────────────────
        pipeline_ph.markdown(
            render_pipeline(done=[0, 1, 2, 3, 4], status="✨  Pipeline complete!"),
            unsafe_allow_html=True
        )

        st.session_state.history.insert(0, {
            "query":      query,                         "model":      GROQ_MODEL,
            "query_type": state["query_type"],           "reasoning":  state["planner_reasoning"],
            "chunks":     state["chunks"],               "sims":       state["similarities"],
            "answer":     state["answer"],               "verdict":    state["verdict"],
            "ver_reason": state["verifier_reason"],      "avg_sim":    state["avg_similarity"],
            "iterations": state["iterations"],           "score":      state["confidence_score"],
            "decision":   state["decision"],             "final_response": state["final_response"],
        })
        st.rerun()   # ← force re-render so results section below picks up new history

    except Exception as e:
        pipeline_ph.empty()
        st.error(f"❌ Pipeline error: {e}")
        st.info("👈 Check your Groq API key in the sidebar.")

elif submit:
    st.warning("⚠️ Please enter a question before launching.")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.history:
    n = len(st.session_state.history)
    st.markdown(f"""
    <div class="sec-head">
      <div class="sec-head-label">Results</div>
      <div class="sec-head-count">{n} {'Query' if n==1 else 'Queries'}</div>
      <div class="sec-head-line"></div>
    </div>""", unsafe_allow_html=True)

    for idx, r in enumerate(st.session_state.history):
        qt  = r["query_type"]
        qcls = f"b-{qt}" if qt in ["factual", "speculative", "ambiguous"] else "b-factual"
        vcls = "b-supported" if r["verdict"] == "SUPPORTED" else "b-unsupported"
        d    = r["decision"]
        t    = r["final_response"]
        vbox = "v-answer" if d == "ANSWER" else ("v-clarify" if d == "CLARIFY" else "v-refuse")
        vico = "✅ Answer" if d == "ANSWER" else ("⚠️ Needs Clarification" if d == "CLARIFY" else "❌ Refused")
        model_short = r["model"].split("/")[-1][:22]

        st.markdown(f"""
        <div class="r-card">
          <div class="r-qlabel">Query #{n - idx}</div>
          <div class="r-qtext">💬 {r['query']}</div>
          <span class="badge {qcls}">{qt.upper()}</span>
          <span class="badge {vcls}">{r['verdict']}</span>
          <span class="badge b-model">⚡ {model_short}</span>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Query Type",  qt.capitalize())
        m2.metric("Verdict",     r["verdict"])
        m3.metric("Iterations",  r["iterations"])
        m4.metric("Decision",    d)

        score = r["score"]
        st.markdown(
            f'<div style="font-family:\'Orbitron\',monospace;font-size:.72rem;'
            f'color:#334155;letter-spacing:2px;margin:.8rem 0 .35rem;">'
            f'CONFIDENCE — {score:.0%}</div>',
            unsafe_allow_html=True
        )
        st.progress(score)

        st.markdown(f'<div class="{vbox}"><strong>{vico}</strong><br><br>{t}</div>',
                    unsafe_allow_html=True)

        st.write("")
        ca, cb = st.columns(2)
        with ca:
            with st.expander("📚 Retrieved Chunks"):
                if r["chunks"]:
                    for i, ch in enumerate(r["chunks"]):
                        sim = r["sims"][i] if i < len(r["sims"]) else None
                        st.markdown(f"**Chunk {i+1}**" + (f" · `sim={sim:.3f}`" if sim else ""))
                        st.write(ch)
                        if i < len(r["chunks"]) - 1: st.divider()
                else:
                    st.write("No chunks retrieved.")
        with cb:
            with st.expander("🔍 Agent Details"):
                st.write(f"**Verdict:** {r['verdict']}")
                st.write(f"**Reason:** {r['ver_reason']}")
                st.write(f"**Avg Similarity:** {r['avg_sim']:.3f}")
                st.write(f"**Planner Reasoning:** {r['reasoning']}")

        if idx < n - 1:
            st.markdown('<hr class="sec">', unsafe_allow_html=True)
