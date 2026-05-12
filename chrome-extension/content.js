(function () {
  const WIDGET_ID = "factsphere-google-widget";
  const STORAGE = {
    async get() {
      return new Promise((resolve) => {
        chrome.storage.sync.get(
          {
            groqApiKey: "",
            backendUrl: "http://localhost:5050",
            model: "llama-3.3-70b-versatile",
            googleApiKey: "",
            googleCx: "",
          },
          resolve
        );
      });
    },
  };

  function getSearchQuery() {
    const u = new URL(location.href);
    return (u.searchParams.get("q") || "").trim();
  }

  let lastFetchedKey = "";
  let latestRun = 0;

  function verdictClass(score) {
    if (score < 30) return "reliable";
    if (score < 60) return "uncertain";
    return "danger";
  }

  function verdictLabel(score) {
    if (score < 30) return "Reliable";
    if (score < 60) return "Uncertain";
    return "Likely hallucinated";
  }

  function removeWidget() {
    document.getElementById(WIDGET_ID)?.remove();
  }

  function buildFactSphereUrl(settings, query, filter) {
    const base = (settings.backendUrl || "http://localhost:5050").replace(/\/$/, "");
    const p = new URLSearchParams();
    p.set("q", query);
    p.set("autosearch", "1");
    if (filter) p.set("filter", filter);
    return `${base}/?${p.toString()}`;
  }

  function renderWidget(settings, state) {
    removeWidget();
    const root = document.createElement("div");
    root.id = WIDGET_ID;

    const q = state.query || "";
    const avg = typeof state.avg === "number" ? state.avg : null;
    const loading = state.loading;
    const err = state.error || "";
    const noKey = state.noKey;

    const vc = avg != null ? verdictClass(avg) : "";
    const label = avg != null ? verdictLabel(avg) : "";

    const pctText =
      noKey
        ? "!"
        : loading
          ? "…"
          : avg != null
            ? `${Math.round(avg)}%`
            : "?";

    root.innerHTML = `
      <button type="button" class="fs-fab fs-fab--${vc || "idle"}" aria-label="FactSphere hallucination hint" title="FactSphere">
        <span class="fs-fab__ring"></span>
        <span class="fs-fab__pct">${pctText}</span>
        <span class="fs-fab__hint">risk</span>
      </button>
      <div class="fs-panel" hidden>
        <div class="fs-panel__head">
          <strong>FactSphere</strong>
          <span class="fs-panel__sub">Hallucination risk (avg across sources)</span>
        </div>
        ${
          noKey
            ? `<p class="fs-panel__msg">Add your Groq API key in the extension popup (toolbar icon) so scores can load.</p>`
            : err
              ? `<p class="fs-panel__msg fs-panel__msg--err">${err}</p>`
              : loading
                ? `<p class="fs-panel__msg">Scoring snippets with your backend…</p>`
                : avg != null
                  ? `<p class="fs-panel__score"><span class="fs-panel__num">${avg.toFixed(1)}%</span> · <span class="fs-tag fs-tag--${vc}">${label}</span></p>
                     <p class="fs-panel__tiny">Lower % = more reliable. Open FactSphere to see each source and reasons.</p>`
                  : `<p class="fs-panel__msg">No score yet.</p>`
        }
        <div class="fs-panel__types">
          <span class="fs-panel__types-label">Open in FactSphere (filter)</span>
          <div class="fs-type-btns">
            <a class="fs-type fs-type--reliable" href="${buildFactSphereUrl(settings, q, "reliable")}" target="_blank" rel="noopener">Reliable</a>
            <a class="fs-type fs-type--uncertain" href="${buildFactSphereUrl(settings, q, "uncertain")}" target="_blank" rel="noopener">Uncertain</a>
            <a class="fs-type fs-type--hallucinated" href="${buildFactSphereUrl(settings, q, "hallucinated")}" target="_blank" rel="noopener">Likely hallucinating</a>
          </div>
        </div>
        <a class="fs-panel__link" href="${buildFactSphereUrl(settings, q, "all")}" target="_blank" rel="noopener">Open full results (all categories) ↗</a>
      </div>
    `;

    document.documentElement.appendChild(root);

    const fab = root.querySelector(".fs-fab");
    const panel = root.querySelector(".fs-panel");
    fab.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const open = panel.hasAttribute("hidden");
      panel.toggleAttribute("hidden", !open);
    });
    document.addEventListener("click", (e) => {
      if (!root.contains(e.target)) panel.setAttribute("hidden", "");
    });
  }

  async function run() {
    const q = getSearchQuery();
    if (!q) {
      removeWidget();
      return;
    }

    const key = `${location.host}|${q}`;
    if (key === lastFetchedKey && document.getElementById(WIDGET_ID)) return;

    const settings = await STORAGE.get();
    const noKey = !settings.groqApiKey?.trim();

    renderWidget(settings, { query: q, loading: !noKey, avg: null, error: "", noKey });

    if (noKey) {
      lastFetchedKey = key;
      return;
    }

    const runId = ++latestRun;

    chrome.runtime.sendMessage(
      {
        type: "FACTSPHERE_SEARCH",
        backendBase: settings.backendUrl,
        query: q,
        model: settings.model,
        apiKey: settings.groqApiKey.trim(),
        googleKey: settings.googleApiKey?.trim() || "",
        googleCx: settings.googleCx?.trim() || "",
      },
      (res) => {
        if (runId !== latestRun) return;
        if (chrome.runtime.lastError) {
          renderWidget(settings, {
            query: q,
            loading: false,
            avg: null,
            error: chrome.runtime.lastError.message || "Extension error",
            noKey: false,
          });
          lastFetchedKey = key;
          return;
        }
        if (!res?.ok) {
          const msg =
            res?.data?.error ||
            (res?.status ? `Server ${res.status}` : null) ||
            res?.error ||
            "Request failed. Is search_backend.py running on port 5050?";
          renderWidget(settings, {
            query: q,
            loading: false,
            avg: null,
            error: String(msg),
            noKey: false,
          });
          lastFetchedKey = key;
          return;
        }
        const avg = res.data?.avg_hallucination;
        const n = typeof avg === "number" ? avg : null;
        renderWidget(settings, {
          query: q,
          loading: false,
          avg: n,
          error: "",
          noKey: false,
        });
        lastFetchedKey = key;
      }
    );
  }

  let t = null;
  function schedule() {
    clearTimeout(t);
    t = setTimeout(run, 400);
  }

  run();
  window.addEventListener("popstate", schedule);
  const mo = new MutationObserver(schedule);
  mo.observe(document.body, { childList: true, subtree: true });

  chrome.storage.onChanged.addListener(() => {
    lastFetchedKey = "";
    schedule();
  });
})();
