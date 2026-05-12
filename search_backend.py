"""
search_backend.py  –  FactSphere Search + Hallucination Rater
=============================================================
Runs a Flask API on http://localhost:5050

Endpoints
---------
POST /search   body: { "query": "...", "model": "llama3-8b-8192", "api_key": "gsk_..." }
GET  /models   returns list of Groq models
"""

import csv
import io
import json
import re
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup

# ── DuckDuckGo search (no API key) ──────────────────────────────────────────
# Prefer the newer 'ddgs' package; fall back to legacy 'duckduckgo_search'
DDG_AVAILABLE = False
try:
    from ddgs import DDGS
    DDG_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
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
        ddgs = DDGS()
        for r in ddgs.text(query, region='en-us', max_results=max_results):
            results.append({
                "source":  "DuckDuckGo",
                "title":   r.get("title", ""),
                "url":     r.get("href",  "") or r.get("url", ""),
                "snippet": r.get("body",  "") or r.get("snippet", ""),
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


def search_bing(query: str, max_results: int = 5) -> list[dict]:
    """Scrape Bing search result snippets (updated selectors for 2025 Bing HTML)."""
    results = []
    try:
        url = f"https://www.bing.com/search?q={requests.utils.quote(query)}&setlang=en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try multiple Bing result container selectors (Bing changes HTML often)
        candidates = (
            soup.select("li.b_algo")
            or soup.select("div.b_algo")
            or soup.select(".b_results .b_algo")
        )
        print(f"[Bing] found {len(candidates)} result containers")

        for item in candidates:
            # Title + URL
            title_el = item.select_one("h2 a") or item.select_one("a[href]")
            # Snippet — try many selectors Bing uses
            snippet_el = (
                item.select_one(".b_caption p")
                or item.select_one(".b_algoSlug")
                or item.select_one(".b_snippet")
                or item.select_one("p")
                or item.select_one(".b_dList")
            )

            title   = title_el.get_text(strip=True)  if title_el   else ""
            href    = title_el.get("href", "")        if title_el   else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

            if title and snippet and len(snippet) > 20:
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
    all_results += search_duckduckgo(query, max_results=5)   # en-us results
    all_results += search_wikipedia(query,  max_results=3)   # always English
    all_results += search_google(query, google_api_key, google_cx, max_results=5)
    all_results += search_bing(query,       max_results=5)

    # 2. Deduplicate
    unique = deduplicate(all_results)[:12]   # cap at 12 cards

    if not unique:
        return jsonify({
            "query":   query,
            "model":   model,
            "results": [],
            "message": "No results found. Check your connection."
        })

    # 3. Generate a direct answer using Groq (uses search snippets as context)
    direct_answer = ""
    if GROQ_AVAILABLE and api_key:
        try:
            context_snippets = "\n\n".join(
                f"[{r['source']}] {r['title']}: {r['snippet'][:300]}"
                for r in unique[:6]
            )
            answer_prompt = (
                "You are a knowledgeable assistant. Based on the search results below, "
                "provide a clear, concise, and direct answer to the user's question.\n\n"
                f"Question: {query}\n\n"
                f"Search Results:\n{context_snippets}\n\n"
                "Instructions:\n"
                "- Answer the question directly in 2-4 sentences.\n"
                "- If asking 'who is X', state clearly who that person is.\n"
                "- If asking 'what is X', define it clearly.\n"
                "- Be factual and cite information from the search results.\n"
                "- Do NOT use JSON format. Just write the answer as plain text.\n"
            )
            client = Groq(api_key=api_key)
            answer_resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful, accurate assistant. Provide direct factual answers."},
                    {"role": "user",   "content": answer_prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )
            direct_answer = answer_resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Direct Answer error] {e}")
            direct_answer = ""

    # 4. Rate each snippet via Groq
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

    # 5. Compute average score and category counts
    avg = round(sum(r["score"] for r in rated) / len(rated), 1) if rated else 0
    categories = {
        "reliable": sum(1 for r in rated if r["score"] < 30),
        "uncertain": sum(1 for r in rated if 30 <= r["score"] < 60),
        "hallucinated": sum(1 for r in rated if r["score"] >= 60),
    }

    return jsonify({
        "query":   query,
        "model":   model,
        "results": rated,
        "avg_hallucination": avg,
        "direct_answer": direct_answer,
        "categories": categories,
    })


# ── Person / LinkedIn Search ─────────────────────────────────────────────────

def find_linkedin_url(name: str) -> str | None:
    """Search DuckDuckGo for a person's LinkedIn profile URL."""
    if not DDG_AVAILABLE:
        return None
    try:
        ddgs = DDGS()
        hits = ddgs.text(
            f"{name} LinkedIn profile site:linkedin.com/in/",
            region="en-us", max_results=5
        )
        for r in hits:
            url = r.get("href", "") or r.get("url", "")
            if "linkedin.com/in/" in url:
                return url
    except Exception as e:
        print(f"[LinkedIn URL search] {e}")
    return None


def fetch_linkedin_profile(linkedin_url: str, rapidapi_key: str) -> dict:
    """
    Fetch full LinkedIn profile via RapidAPI LinkedIn scraper.
    Uses 'linkedin-api8' on RapidAPI (free tier available).
    """
    try:
        resp = requests.get(
            "https://linkedin-api8.p.rapidapi.com/",
            headers={
                "X-RapidAPI-Key": rapidapi_key,
                "X-RapidAPI-Host": "linkedin-api8.p.rapidapi.com",
            },
            params={"username": linkedin_url.rstrip("/").split("/in/")[-1].split("/")[0]},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"API returned {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def _format_profile(raw: dict) -> dict:
    """Normalise raw API response into a clean profile dict."""
    if "error" in raw:
        return raw

    education = []
    for edu in raw.get("educations", []) or raw.get("education", []) or []:
        school_name = edu.get("schoolName", "") or edu.get("school", "")
        if isinstance(school_name, dict):
            school_name = school_name.get("name", "")
        education.append({
            "school":  school_name,
            "degree":  edu.get("degreeName", "") or edu.get("degree_name", ""),
            "field":   edu.get("fieldOfStudy", "") or edu.get("field_of_study", ""),
            "start":   str(edu.get("dateRange", {}).get("start", {}).get("year", "")) if edu.get("dateRange") else str(edu.get("starts_at", {}).get("year", "") if edu.get("starts_at") else ""),
            "end":     str(edu.get("dateRange", {}).get("end", {}).get("year", "")) if edu.get("dateRange") else str(edu.get("ends_at", {}).get("year", "") if edu.get("ends_at") else "Present"),
        })

    experience = []
    for exp in raw.get("position", []) or raw.get("experiences", []) or []:
        company = exp.get("companyName", "") or exp.get("company", "")
        if isinstance(company, dict):
            company = company.get("name", "")
        experience.append({
            "company":     company,
            "title":       exp.get("title", ""),
            "location":    exp.get("location", "") or exp.get("locationName", ""),
            "description": (exp.get("description", "") or "")[:250],
            "start":       str(exp.get("dateRange", {}).get("start", {}).get("year", "")) if exp.get("dateRange") else "",
            "end":         str(exp.get("dateRange", {}).get("end", {}).get("year", "")) if exp.get("dateRange") else "Present",
        })

    certifications = []
    for cert in raw.get("certifications", []) or []:
        certifications.append({
            "name":      cert.get("name", ""),
            "authority": cert.get("authority", "") or cert.get("company", {}).get("name", "") if isinstance(cert.get("company"), dict) else cert.get("authority", ""),
        })

    skills = []
    for s in raw.get("skills", []) or []:
        if isinstance(s, dict):
            skills.append(s.get("name", str(s)))
        else:
            skills.append(str(s))

    languages = []
    for lang in raw.get("languages", []) or []:
        if isinstance(lang, dict):
            languages.append(lang.get("name", str(lang)))
        else:
            languages.append(str(lang))

    return {
        "name":           raw.get("fullName", "") or raw.get("full_name", "") or f"{raw.get('firstName', '')} {raw.get('lastName', '')}".strip(),
        "headline":       raw.get("headline", ""),
        "summary":        raw.get("summary", "") or raw.get("about", ""),
        "location":       raw.get("geo", {}).get("full", "") if isinstance(raw.get("geo"), dict) else raw.get("location", "") or raw.get("locationName", ""),
        "profile_pic":    raw.get("profilePicture", "") or raw.get("profile_pic_url", ""),
        "linkedin_url":   raw.get("linkedInUrl", "") or raw.get("public_identifier", ""),
        "occupation":     raw.get("headline", ""),
        "connections":    raw.get("connectionsCount", 0) or raw.get("connections", 0),
        "education":      education,
        "experience":     experience,
        "certifications": certifications,
        "skills":         skills,
        "languages":      languages,
    }


@app.route("/person-search", methods=["POST"])
def person_search_route():
    """Lookup a single person's LinkedIn profile."""
    body = request.get_json(force=True)
    name          = body.get("name", "").strip()
    rapidapi_key  = body.get("rapidapi_key", "").strip() or _os.getenv("RAPIDAPI_KEY", "").strip()
    linkedin_url  = body.get("linkedin_url", "").strip()

    if not name and not linkedin_url:
        return jsonify({"error": "Provide a person's name or LinkedIn URL."}), 400
    if not rapidapi_key:
        return jsonify({"error": "RapidAPI key required. Get one free at rapidapi.com → subscribe to \"LinkedIn Data API\"."}), 400

    if not linkedin_url:
        linkedin_url = find_linkedin_url(name)
        if not linkedin_url:
            return jsonify({"error": f"Could not find LinkedIn profile for '{name}'. Try providing the LinkedIn URL directly."}), 404

    raw = fetch_linkedin_profile(linkedin_url, rapidapi_key)
    profile = _format_profile(raw)
    if "error" in profile:
        return jsonify(profile), 400

    return jsonify({"profile": profile, "linkedin_url": linkedin_url})


@app.route("/person-bulk", methods=["POST"])
def person_bulk_route():
    """Bulk lookup: accept CSV of names, return CSV of profile data."""
    rapidapi_key = request.form.get("rapidapi_key", "").strip() or _os.getenv("RAPIDAPI_KEY", "").strip()
    if not rapidapi_key:
        return jsonify({"error": "RapidAPI key required."}), 400

    f = request.files.get("csv_file")
    if not f:
        return jsonify({"error": "No CSV file uploaded."}), 400

    content = f.read().decode("utf-8", errors="replace")
    reader  = csv.reader(io.StringIO(content))
    names   = [row[0].strip() for row in reader if row and row[0].strip()]
    # skip header row
    if names and names[0].lower() in ("name", "names", "full_name", "person", "linkedin"):
        names = names[1:]
    if not names:
        return jsonify({"error": "CSV has no names."}), 400

    results = []
    for name in names[:25]:                       # cap to 25
        url = find_linkedin_url(name)
        if url:
            raw = fetch_linkedin_profile(url, rapidapi_key)
            p   = _format_profile(raw)
            p["search_name"]       = name
            p["linkedin_url_found"] = url
            results.append(p)
        else:
            results.append({"search_name": name, "error": "LinkedIn profile not found"})
        time.sleep(0.5)

    # Build CSV response
    out = io.StringIO()
    fields = [
        "search_name", "name", "headline", "location", "occupation",
        "education_summary", "certifications_summary", "skills_summary",
        "linkedin_url_found",
    ]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        edu = " | ".join(
            f"{e.get('degree','')} in {e.get('field','')} from {e.get('school','')}"
            for e in r.get("education", [])
        ) if r.get("education") else ""
        cert = " | ".join(
            f"{c.get('name','')} ({c.get('authority','')})"
            for c in r.get("certifications", [])
        ) if r.get("certifications") else ""
        sk = ", ".join(r.get("skills", [])[:15]) if r.get("skills") else ""
        writer.writerow({
            "search_name":            r.get("search_name", ""),
            "name":                   r.get("name", ""),
            "headline":               r.get("headline", ""),
            "location":               r.get("location", ""),
            "occupation":             r.get("occupation", ""),
            "education_summary":      edu,
            "certifications_summary": cert,
            "skills_summary":         sk,
            "linkedin_url_found":     r.get("linkedin_url_found", ""),
        })

    from flask import Response
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=person_search_results.csv"},
    )


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FactSphere Search + Hallucination Rater Backend")
    print("  http://localhost:5050")
    print("  Powered by Groq Cloud LLM (Free)")
    print("  Get your API key: https://console.groq.com")
    print("=" * 60)
    app.run(port=5050, debug=False)
