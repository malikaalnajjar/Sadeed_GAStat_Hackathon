const API_URLS = [
  "http://localhost:8000/anomalies",
  "https://sadeed.aldharrab.co.uk/anomalies",
];
let API_BASE = API_URLS[0];
const g = id => document.getElementById(id);

async function resolveApiBase() {
  for (const url of API_URLS) {
    try {
      const r = await fetch(`${url}/health`, { method: "GET", signal: AbortSignal.timeout(3000) });
      if (r.ok) { API_BASE = url; return; }
    } catch { /* try next */ }
  }
}

// Health check
async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`);
    const data = await resp.json();
    const ok = data.great_expectations !== false && data.svm !== false && data.llm !== false && data.redis !== false;
    g("healthDot").className = "health-dot " + (ok ? "online" : "offline");
    g("healthDot").title = ok ? "Server connected" : "Some services unavailable";
  } catch {
    g("healthDot").className = "health-dot offline";
    g("healthDot").title = "Server offline";
  }
}
resolveApiBase().then(() => checkHealth());
setInterval(checkHealth, 10000);

// Upload zone — click to select only
const uploadZone = g("uploadZone");
const fileInput = g("fileInput");

uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) processFile(fileInput.files[0]);
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatScore(r) {
  if (!r) return '<span class="score">\u2014</span>';
  const pct = r.score != null ? (r.score * 100).toFixed(1) + "%" : "\u2014";
  const cls = r.is_anomaly ? "score-bad" : "score-ok";
  const icon = r.is_anomaly ? "\u26A0" : "\u2714";
  return `<span class="score ${cls}">${icon} ${pct}</span>`;
}

async function processFile(file) {
  /* global XLSX */
  const data = await file.arrayBuffer();
  const workbook = XLSX.read(data, { type: "array" });
  const rows = XLSX.utils.sheet_to_json(workbook.Sheets[workbook.SheetNames[0]], { defval: null });

  if (rows.length === 0) return;

  // Show progress, hide previous results
  g("progressSection").style.display = "block";
  g("summaryCards").classList.remove("visible");
  g("resultsSection").classList.remove("visible");
  g("resultsBody").innerHTML = "";
  uploadZone.style.display = "none";

  let anomalyCount = 0;
  const total = rows.length;

  for (let i = 0; i < total; i++) {
    const row = rows[i];
    const cleaned = {};
    for (const [k, v] of Object.entries(row)) {
      if (v !== null && v !== undefined && String(v) !== "NULL" && !k.startsWith("_")) {
        cleaned[k] = v;
      }
    }

    g("progressBar").style.width = (((i + 1) / total) * 100).toFixed(1) + "%";
    g("progressText").textContent = `Processing ${i + 1} of ${total}...`;

    try {
      const resp = await fetch(`${API_BASE}/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ record_id: crypto.randomUUID(), data: cleaned }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const result = await resp.json();

      if (result.is_anomaly) anomalyCount++;

      const strategies = {};
      if (result.results) {
        for (const r of result.results) strategies[r.strategy] = r;
      }

      let badge;
      if (result.severity === "hard_error") {
        badge = '<span class="badge badge-hard">\u26D4 Hard Error</span>';
      } else if (result.severity === "warning") {
        badge = '<span class="badge badge-warning">\u26A0 Warning</span>';
      } else if (result.is_anomaly) {
        badge = '<span class="badge badge-warning">Anomaly</span>';
      } else {
        badge = '<span class="badge badge-clean">\u2714 Clean</span>';
      }

      const explanation = result.explanation
        ? escapeHtml(result.explanation)
        : '<span style="color:#aaa">\u2014</span>';

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>${badge}</td>
        <td>${formatScore(strategies.great_expectations)}</td>
        <td>${formatScore(strategies.svm)}</td>
        <td>${formatScore(strategies.llm)}</td>
        <td class="explanation-cell">${explanation}</td>
      `;
      g("resultsBody").appendChild(tr);

    } catch (err) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${i + 1}</td><td colspan="5" style="color:#c5221f">Error: ${escapeHtml(err.message)}</td>`;
      g("resultsBody").appendChild(tr);
    }
  }

  // Show summary
  g("progressSection").style.display = "none";
  g("statTotal").textContent = total;
  g("statAnomalies").textContent = anomalyCount;
  g("statClean").textContent = total - anomalyCount;
  g("summaryCards").classList.add("visible");
  g("resultsSection").classList.add("visible");
}
