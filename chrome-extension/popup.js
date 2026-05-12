const DEFAULTS = {
  groqApiKey: "",
  backendUrl: "http://localhost:5050",
  model: "llama-3.3-70b-versatile",
  googleApiKey: "",
  googleCx: "",
};

function $(id) {
  return document.getElementById(id);
}

function load() {
  chrome.storage.sync.get(DEFAULTS, (s) => {
    $("groq").value = s.groqApiKey || "";
    $("backend").value = s.backendUrl || DEFAULTS.backendUrl;
    $("model").value = s.model || DEFAULTS.model;
    $("gkey").value = s.googleApiKey || "";
    $("gcx").value = s.googleCx || "";
  });
}

$("save").addEventListener("click", () => {
  const groqApiKey = $("groq").value.trim();
  const backendUrl = $("backend").value.trim() || DEFAULTS.backendUrl;
  const model = $("model").value;
  const googleApiKey = $("gkey").value.trim();
  const googleCx = $("gcx").value.trim();

  chrome.storage.sync.set(
    { groqApiKey, backendUrl, model, googleApiKey, googleCx },
    () => {
      $("status").textContent = "Saved.";
      setTimeout(() => {
        $("status").textContent = "";
      }, 2000);
    }
  );
});

const openOpts = document.getElementById("open-options");
if (openOpts) {
  openOpts.addEventListener("click", (e) => {
    e.preventDefault();
    if (chrome.runtime.openOptionsPage) chrome.runtime.openOptionsPage();
  });
}

load();
