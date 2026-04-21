"""
search_backend.py  –  FactSphere Search + Hallucination Rater
=============================================================
Runs a Flask API on http://localhost:5050

Endpoints
---------
POST /search   body: { "query": "...", "model": "llama3-8b-8192", "api_key": "gsk_..." }
GET  /models   returns list of Groq models
"""

import json
import re
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup

# ── DuckDuckGo search (no API key) ──────────────────────────────────────────
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    try:
        from ddgs import DDGS
        DDG_AVAILABLE = True
    except ImportError:
        DDG_AVAILABLE = False

# ── Wikipedia (reliable, always English) ─────────────────────────────────────
try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False

# ── Groq Cloud LLM ──────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

import os as _os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder=".", static_url_path="/static")
CORS(app)  # allow the HTML file to call this from any origin

_FRONTEND = _os.path.join(_os.path.dirname(__file__), "search_frontend.html")

@app.route("/")
def serve_index():
    from flask import send_file
    return send_file(_FRONTEND)

@app.route("/search_frontend.html")
def serve_html():
    from flask import send_file
    return send_file(_FRONTEND)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ── Available Groq Models (all free-tier) ────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def search_duckduckgo(query: str, max_results: int = 6) -> list[dict]:
    """Return list of {title, url, snippet, source} from DuckDuckGo (English)."""
    results = []
    if not DDG_AVAILABLE:
        return results
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region='en-us', max_results=max_results):
                results.append({
                    "source": "DuckDuckGo",
                    "title":   r.get("title", ""),
                    "url":     r.get("href",  ""),
                    "snippet": r.get("body",  ""),
                })
    except Exception as e:
        print(f"[DDG error] {e}")
    return results


def search_wikipedia(query: str, max_results: int = 3) -> list[dict]:
    """Fetch Wikipedia summary as a reliable English source."""
    results = []
    if not WIKI_AVAILABLE:
        return results
    try:
        import urllib.parse
        wiki = wikipediaapi.Wikipedia(user_agent='FactSphere/1.0', language='en')
        
        # Search for page titles using Wikimedia API
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
        
        resp = requests.get(url, headers={'User-Agent': 'FactSphere/1.0 (someone@example.com)'}, timeout=5)
        data = resp.json()
        search_hits = data.get("query", {}).get("search", [])
        
        for hit in search_hits[:max_results]:
            title = hit["title"]
            page = wiki.page(title)
            if page.exists():
                summary = page.summary[:600]
                results.append({
                    "source":  "Wikipedia",
                    "title":   page.title,
                    "url":     page.fullurl,
                    "snippet": summary,
                })
    except Exception as e:
        print(f"[Wikipedia error] {e}")
    return results


def search_google(query: str, api_key: str = "", cx: str = "", max_results: int = 6) -> list[dict]:
    """Search Google via JSON API if keys are provided, else fallback to scraping."""
    results = []
    if api_key and cx:
        try:
            url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={requests.utils.quote(query)}&num={max_results}"
            resp = requests.get(url, timeout=8)
            data = resp.json()
            for item in data.get("items", []):
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                if title and snippet:
                    results.append({
                        "source": "Google",
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                    })
        except Exception as e:
            print(f"[Google API error] {e}")
        return results

    try:
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}&num={max_results}"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")

        for g in soup.select("div.tF2Cxc, div.g"):
            title_el   = g.select_one("h3")
            link_el    = g.select_one("a[href]")
            snippet_el = g.select_one("div.VwiC3b, div.s, span.st")

            title   = title_el.get_text()   if title_el   else ""
            href    = link_el["href"]        if link_el    else ""
            snippet = snippet_el.get_text()  if snippet_el else ""

            if title and snippet and len(snippet) > 30:
                results.append({
                    "source":  "Google",
                    "title":   title,
                    "url":     href if href.startswith("http") else f"https://www.google.com{href}",
                    "snippet": snippet,
                })
            if len(results) >= max_results:
                break
    except Exception as e:
        print(f"[Google error] {e}")
    return results


def search_bing(query: str, max_results: int = 4) -> list[dict]:
    """Scrape Bing search result snippets."""
    results = []
    try:
        url = f"https://www.bing.com/search?q={requests.utils.quote(query)}"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")

        for item in soup.select("li.b_algo"):
            title_el   = item.select_one("h2 a")
            snippet_el = item.select_one("p, .b_caption p")

            title   = title_el.get_text()   if title_el   else ""
            href    = title_el["href"]       if title_el   else ""
            snippet = snippet_el.get_text()  if snippet_el else ""

            if title and snippet and len(snippet) > 30:
                results.append({
                    "source":  "Bing",
                    "title":   title,
                    "url":     href,
                    "snippet": snippet,
                })
            if len(results) >= max_results:
                break
    except Exception as e:
        print(f"[Bing error] {e}")
    return results


def rate_hallucination_groq(query: str, snippet: str, model: str, api_key: str) -> dict:
    """
    Ask Groq Cloud LLM to rate how likely the snippet contains false info.
    Returns { score: int (0-100), reason: str, verdict: str }
    """
    if not GROQ_AVAILABLE or not api_key:
        return {"score": 50, "reason": "Groq API key not provided.", "verdict": "Uncertain"}

    prompt = (
        "You are a fact-checking AI. Given a search query and a text snippet from a web result, "
        "assess how likely the snippet contains hallucinated, false, or misleading information.\n\n"
        f"Query: {query}\n\n"
        f"Snippet: {snippet}\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"score": <integer 0-100>, "reason": "<one sentence>", "verdict": "<Reliable|Uncertain|Likely Hallucinated>"}\n\n'
        "Score guide:\n"
        "  0-29  = Reliable (likely accurate, well-supported)\n"
        " 30-59  = Uncertain (partially supported, needs verification)\n"
        " 60-100 = Likely Hallucinated (false, misleading, or unsupported claims)\n"
    )

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant. Respond only with valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=200,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON from response
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score   = max(0, min(100, int(data.get("score",   50))))
            reason  = data.get("reason",  "Could not determine.")
            verdict = data.get("verdict", "Uncertain")
            return {"score": score, "reason": reason, "verdict": verdict}
    except Exception as e:
        print(f"[Groq error] {e}")

    return {"score": 50, "reason": "Could not rate (Groq unavailable).", "verdict": "Uncertain"}


def deduplicate(results: list[dict]) -> list[dict]:
    seen_snippets = set()
    unique = []
    for r in results:
        key = r["snippet"][:80].lower().strip()
        if key not in seen_snippets and len(r["snippet"]) > 30:
            seen_snippets.add(key)
            unique.append(r)
    return unique


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/models", methods=["GET"])
def get_models():
    """Return list of available Groq models."""
    return jsonify({"models": GROQ_MODELS})


@app.route("/search", methods=["POST"])
def search():
    """Main endpoint: search all engines, rate hallucination per result."""
    body    = request.get_json(force=True)
    query   = body.get("query", "").strip()
    model   = body.get("model", "llama-3.3-70b-versatile")
    api_key = body.get("api_key", "").strip()
    google_api_key = body.get("google_api_key", "").strip()
    google_cx = body.get("google_cx", "").strip()
    
    # Fallback to environment variable if API key not provided in request
    if not api_key:
        api_key = _os.getenv("GROQ_API_KEY", "").strip()
    if not google_api_key:
        google_api_key = _os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_cx:
        google_cx = _os.getenv("GOOGLE_CX", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not api_key:
        return jsonify({"error": "No Groq API key provided. Set GROQ_API_KEY in .env or provide in request."}), 400

    # 1. Collect results from all English sources
    all_results: list[dict] = []
    all_results += search_duckduckgo(query, max_results=1)   # forced en-us, limit 1
    all_results += search_wikipedia(query,  max_results=3)   # always English
    all_results += search_google(query, google_api_key, google_cx, max_results=5)
    all_results += search_bing(query,       max_results=4)

    # 2. Deduplicate
    unique = deduplicate(all_results)[:12]   # cap at 12 cards

    if not unique:
        return jsonify({
            "query":   query,
            "model":   model,
            "results": [],
            "message": "No results found. Check your connection."
        })

    # 3. Rate each snippet via Groq
    rated = []
    for item in unique:
        rating = rate_hallucination_groq(query, item["snippet"], model, api_key)
        rated.append({
            "source":  item["source"],
            "title":   item["title"],
            "url":     item["url"],
            "snippet": item["snippet"],
            "score":   rating["score"],
            "reason":  rating["reason"],
            "verdict": rating["verdict"],
        })
        time.sleep(0.1)   # tiny throttle

    # 4. Compute average score
    avg = round(sum(r["score"] for r in rated) / len(rated), 1) if rated else 0

    return jsonify({
        "query":   query,
        "model":   model,
        "results": rated,
        "avg_hallucination": avg,
    })


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FactSphere Search + Hallucination Rater Backend")
    print("  http://localhost:5050")
    print("  Powered by Groq Cloud LLM (Free)")
    print("  Get your API key: https://console.groq.com")
    print("=" * 60)
    app.run(port=5050, debug=False)
