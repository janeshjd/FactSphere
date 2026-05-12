import os, requests
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('GROQ_API_KEY', '')
print(f"GROQ key present: {bool(api_key)} ({api_key[:15]}...)")

# ── DuckDuckGo ──────────────────────────────────────────────────
print("\n--- Testing DuckDuckGo ---")
try:
    from duckduckgo_search import DDGS as DDGS1
    with DDGS1() as ddgs:
        results = list(ddgs.text('Who invented the telephone', region='en-us', max_results=3))
    print(f"DDG (old pkg) results: {len(results)}")
    for r in results:
        print(f"  {r.get('title','')[:60]}")
except Exception as e:
    print(f"DDG old error: {e}")
    try:
        from ddgs import DDGS as DDGS2
        with DDGS2() as ddgs:
            results = list(ddgs.text('Who invented the telephone', region='en-us', max_results=3))
        print(f"DDGS (new pkg) results: {len(results)}")
        for r in results:
            print(f"  {r.get('title','')[:60]}")
    except Exception as e2:
        print(f"DDGS new error: {e2}")

# ── Bing scraping ────────────────────────────────────────────────
print("\n--- Testing Bing ---")
try:
    from bs4 import BeautifulSoup
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    resp = requests.get("https://www.bing.com/search?q=Who+invented+the+telephone", headers=HEADERS, timeout=8)
    soup = BeautifulSoup(resp.text, "html.parser")
    items = soup.select("li.b_algo")
    print(f"Bing raw items: {len(items)}")
    for item in items[:3]:
        t = item.select_one("h2 a")
        s = item.select_one("p, .b_caption p")
        title   = t.get_text()[:60] if t else "N/A"
        snippet = s.get_text()[:60] if s else "N/A"
        print(f"  Title: {title} | Snippet: {snippet}")
except Exception as e:
    print(f"Bing error: {e}")

# ── Wikipedia ────────────────────────────────────────────────────
print("\n--- Testing Wikipedia ---")
try:
    import urllib.parse
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&list=search"
        "&srsearch=" + urllib.parse.quote("telephone")
        + "&utf8=&format=json"
    )
    resp = requests.get(url, headers={"User-Agent": "FactSphere/1.0"}, timeout=5)
    hits = resp.json().get("query", {}).get("search", [])
    print(f"Wikipedia hits: {len(hits)}")
    for h in hits[:3]:
        print(f"  {h['title']}")
except Exception as e:
    print(f"Wikipedia error: {e}")

# ── Full /search endpoint test ────────────────────────────────────
print("\n--- Testing /search endpoint ---")
try:
    r = requests.post("http://localhost:5050/search", json={
        "query": "Who invented the telephone",
        "model": "llama-3.3-70b-versatile",
        "api_key": api_key
    }, timeout=60)
    d = r.json()
    print(f"Status: {r.status_code}")
    print(f"Num results: {len(d.get('results', []))}")
    print(f"Avg hallucination: {d.get('avg_hallucination')}")
    for x in d.get("results", []):
        print(f"  [{x['verdict']}] {x['score']}% | [{x['source']}] {x['title'][:60]}")
except Exception as e:
    print(f"/search error: {e}")
