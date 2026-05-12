chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "FACTSPHERE_SEARCH") return;

  const {
    backendBase,
    query,
    model,
    apiKey,
    googleKey = "",
    googleCx = "",
  } = message;

  (async () => {
    try {
      const base = (backendBase || "http://localhost:5050").replace(/\/$/, "");
      const res = await fetch(`${base}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          model: model || "llama-3.3-70b-versatile",
          api_key: apiKey,
          google_api_key: googleKey,
          google_cx: googleCx,
        }),
      });
      const data = await res.json().catch(() => ({}));
      sendResponse({ ok: res.ok, status: res.status, data });
    } catch (e) {
      sendResponse({
        ok: false,
        error: e instanceof Error ? e.message : String(e),
        data: null,
      });
    }
  })();

  return true;
});
