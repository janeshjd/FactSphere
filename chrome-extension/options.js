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

chrome.storage.sync.get(DEFAULTS, (s) => {
  $("groq").value = s.groqApiKey || "";
  $("backend").value = s.backendUrl || DEFAULTS.backendUrl;
  $("model").value = s.model || DEFAULTS.model;
  $("gkey").value = s.googleApiKey || "";
  $("gcx").value = s.googleCx || "";
});

$("save").addEventListener("click", () => {
  chrome.storage.sync.set(
    {
      groqApiKey: $("groq").value.trim(),
      backendUrl: $("backend").value.trim() || DEFAULTS.backendUrl,
      model: $("model").value,
      googleApiKey: $("gkey").value.trim(),
      googleCx: $("gcx").value.trim(),
    },
    () => {
      $("status").textContent = "Saved.";
    }
  );
});
