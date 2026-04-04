// ------------------------------------------------------------------ //
// Sadeed — Popup Script                                              //
// Two tabs: page scan results + Excel file upload                    //
// ------------------------------------------------------------------ //

(function () {
  "use strict";

  const STRATEGY_LABELS = {
    great_expectations: "Business Rules (GE)",
    svm: "Statistical Model (SVM)",
    llm: "LLM Review",
  };

  // ---- Tab switching ---- //

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll(".tab-btn")
        .forEach((b) => b.classList.remove("active"));
      document
        .querySelectorAll(".tab-content")
        .forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  });

  // ---- Health polling ---- //

  const healthDot = document.getElementById("healthDot");

  async function checkHealth() {
    try {
      const result = await chrome.runtime.sendMessage({ type: "HEALTH_CHECK" });
      if (result && !result.error) {
        // null = not configured (OK), false = failed (error)
        const allUp =
          result.great_expectations !== false &&
          result.svm !== false &&
          result.llm !== false &&
          result.redis !== false;
        healthDot.className = "health-dot " + (allUp ? "online" : "offline");
        healthDot.title = allUp ? "Server connected" : "Some services unavailable";
      } else {
        healthDot.className = "health-dot offline";
        healthDot.title = "Server offline";
      }
    } catch {
      healthDot.className = "health-dot offline";
      healthDot.title = "Server offline";
    }
  }

  checkHealth();
  setInterval(checkHealth, 10000);

  // If opened as a full tab (not popup), auto-switch to upload tab
  if (window.innerWidth > 420) {
    const params = new URLSearchParams(window.location.search);
    if (params.get("tab") === "upload") {
      document.querySelector('[data-tab="upload"]').click();
    }
  }

  // ---- Scan tab: load last result ---- //

  const scanVerdict = document.getElementById("scanVerdict");
  const scanStrategies = document.getElementById("scanStrategies");

  function renderScanResult(result) {
    if (!result) return;

    const severityClass = result.severity === "hard_error" ? "hard-error" :
                          result.severity === "warning" ? "warning" : "";
    scanVerdict.className =
      "verdict-banner " + (result.is_anomaly ? "anomaly" : "clean") +
      (severityClass ? " " + severityClass : "");
    if (result.is_anomaly && result.severity === "hard_error") {
      scanVerdict.textContent = "\u26D4 Hard Error — Data entry error detected";
    } else if (result.is_anomaly && result.severity === "warning") {
      scanVerdict.textContent = "\u26A0 Warning — Suspicious data, needs review";
    } else if (result.is_anomaly) {
      scanVerdict.textContent = "\u26A0 Anomaly detected";
    } else {
      scanVerdict.textContent = "\u2714 Clean — No anomalies detected";
    }

    scanStrategies.innerHTML = "";

    // LLM explanation (top-level, prominent)
    if (result.explanation) {
      const explDiv = document.createElement("div");
      explDiv.className = "strategy-card flagged";
      explDiv.style.borderRightColor = "#1a73e8";
      explDiv.style.marginBottom = "14px";
      explDiv.innerHTML = `
        <div class="strategy-header">
          <span class="strategy-name">\u{1F4AC} LLM Explanation</span>
        </div>
        <div class="strategy-explanation" style="font-size:12px;line-height:1.7">${escapeHtml(result.explanation)}</div>
      `;
      scanStrategies.appendChild(explDiv);
    }

    // Strategy scores (compact, no explanations)
    if (result.results) {
      for (const r of result.results) {
        const card = document.createElement("div");
        card.className =
          "strategy-card " + (r.is_anomaly ? "flagged" : "passed");
        card.innerHTML = `
          <div class="strategy-header">
            <span class="strategy-name">${STRATEGY_LABELS[r.strategy] || r.strategy}</span>
            <span class="strategy-score">${r.is_anomaly ? "\u26A0" : "\u2714"} ${r.score != null ? (r.score * 100).toFixed(1) + "%" : "\u2014"}</span>
          </div>
        `;
        scanStrategies.appendChild(card);
      }
    }
  }

  // Load last result on popup open
  chrome.runtime.sendMessage({ type: "GET_LAST_RESULT" }, (result) => {
    if (result) renderScanResult(result);
  });

  // Listen for new results from background
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.lastResult && changes.lastResult.newValue) {
      renderScanResult(changes.lastResult.newValue);
    }
  });

  // ---- Upload tab ---- //

  const uploadZone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");
  const uploadProgress = document.getElementById("uploadProgress");
  const progressBar = document.getElementById("progressBar");
  const progressText = document.getElementById("progressText");
  const summaryBox = document.getElementById("summaryBox");
  const resultsTableWrap = document.getElementById("resultsTableWrap");
  const resultsBody = document.getElementById("resultsBody");

  uploadZone.addEventListener("click", () => {
    chrome.tabs.create({ url: chrome.runtime.getURL("upload.html") });
    window.close();
  });

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "#1a73e8";
    uploadZone.style.background = "#e8f0fe";
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.style.borderColor = "";
    uploadZone.style.background = "";
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "";
    uploadZone.style.background = "";
    if (e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      processFile(fileInput.files[0]);
    }
  });

  async function processFile(file) {
    const data = await file.arrayBuffer();
    /* global XLSX */
    const workbook = XLSX.read(data, { type: "array" });
    const firstSheet = workbook.SheetNames[0];
    const rows = XLSX.utils.sheet_to_json(workbook.Sheets[firstSheet], {
      defval: null,
    });

    if (rows.length === 0) return;

    // Reset UI
    resultsBody.innerHTML = "";
    summaryBox.style.display = "none";
    resultsTableWrap.style.display = "block";
    uploadProgress.style.display = "block";

    let anomalyCount = 0;
    const total = rows.length;

    for (let i = 0; i < total; i++) {
      const row = rows[i];
      // Clean row: remove null values, convert to strings where needed
      const cleaned = {};
      for (const [k, v] of Object.entries(row)) {
        if (v !== null && v !== undefined && String(v) !== "NULL") {
          cleaned[k] = v;
        }
      }

      // Update progress
      progressBar.style.width = (((i + 1) / total) * 100).toFixed(1) + "%";
      progressText.textContent = `${i + 1} / ${total}`;

      try {
        const result = await chrome.runtime.sendMessage({
          type: "DETECT_ROW",
          data: cleaned,
        });

        if (result.is_anomaly) anomalyCount++;

        const tr = document.createElement("tr");
        if (result.is_anomaly) tr.className = "row-anomaly";

        const strategyScores = {};
        if (result.results) {
          for (const r of result.results) {
            strategyScores[r.strategy] = r;
          }
        }

        let badge;
        if (result.severity === "hard_error") {
          badge = '<span class="badge-anomaly badge-hard">\u26D4 Hard Error</span>';
        } else if (result.severity === "warning") {
          badge = '<span class="badge-anomaly badge-warning">\u26A0 Warning</span>';
        } else if (result.is_anomaly) {
          badge = '<span class="badge-anomaly">Anomaly</span>';
        } else {
          badge = '<span class="badge-clean">Clean</span>';
        }

        tr.innerHTML = `
          <td>${i + 1}</td>
          <td>${badge}</td>
          <td>${formatScore(strategyScores.great_expectations)}</td>
          <td>${formatScore(strategyScores.svm)}</td>
          <td>${formatScore(strategyScores.llm)}</td>
        `;
        resultsBody.appendChild(tr);
      } catch (err) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${i + 1}</td>
          <td colspan="4" style="color:#c5221f">خطأ: ${escapeHtml(err.message)}</td>
        `;
        resultsBody.appendChild(tr);
      }
    }

    // Show summary
    summaryBox.style.display = "block";
    document.getElementById("statTotal").textContent = total;
    document.getElementById("statAnomalies").textContent = anomalyCount;
    document.getElementById("statClean").textContent = total - anomalyCount;
    uploadProgress.style.display = "none";
  }

  function formatScore(strategyResult) {
    if (!strategyResult) return "\u2014";
    const pct =
      strategyResult.score != null
        ? (strategyResult.score * 100).toFixed(0) + "%"
        : "\u2014";
    const icon = strategyResult.is_anomaly ? "\u26A0" : "\u2714";
    return `${icon} ${pct}`;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();
