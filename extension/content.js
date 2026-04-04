// ------------------------------------------------------------------ //
// Sadeed — Content Script                                            //
// Scans form fields, sends to background for detection, shows results //
// ------------------------------------------------------------------ //

(function () {
  "use strict";

  if (window.__sadeedInjected) return;
  window.__sadeedInjected = true;

  const SELECTORS = "input, select, textarea";

  // ----- Field name extraction (priority chain) ----- //

  function getFieldName(el) {
    if (el.name) return el.name;
    if (el.id) return el.id;
    if (el.getAttribute("aria-label")) return el.getAttribute("aria-label");
    if (el.placeholder) return el.placeholder;

    // Nearest <label>
    if (el.id) {
      const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (label) return label.textContent.trim();
    }
    const parent = el.closest("label");
    if (parent) return parent.textContent.trim();

    return null;
  }

  function getFieldValue(el) {
    if (el.type === "checkbox" || el.type === "radio") {
      return el.checked ? (el.value || "true") : null;
    }
    return el.value || null;
  }

  // ----- Google Forms support ----- //

  function isGoogleForms() {
    return location.hostname === "docs.google.com" && location.pathname.includes("/forms/");
  }

  function scanGoogleForms() {
    const fields = {};
    // Each question lives inside a div with data-params or [role="listitem"]
    const questionBlocks = document.querySelectorAll(
      '[role="listitem"], .freebirdFormviewerComponentsQuestionBaseRoot'
    );

    for (const block of questionBlocks) {
      // Get question title — the visible label text
      const titleEl =
        block.querySelector('[role="heading"]') ||
        block.querySelector(".freebirdFormviewerComponentsQuestionBaseTitle") ||
        block.querySelector('[data-initial-value]');
      const title = titleEl ? titleEl.textContent.trim() : null;
      if (!title) continue;

      // Text / number inputs
      const textInput = block.querySelector('input[type="text"], input[type="number"], input:not([type])');
      if (textInput && textInput.value) {
        fields[title] = textInput.value;
        continue;
      }

      // Textarea (long answer)
      const textArea = block.querySelector("textarea");
      if (textArea && textArea.value) {
        fields[title] = textArea.value;
        continue;
      }

      // Dropdown — Google Forms uses a hidden input + visible div
      const dropdown = block.querySelector('[data-value]:not([data-value=""])');
      if (dropdown) {
        fields[title] = dropdown.getAttribute("data-value");
        continue;
      }

      // Radio buttons — find the checked one
      const checkedRadio = block.querySelector('[role="radio"][aria-checked="true"]');
      if (checkedRadio) {
        const label = checkedRadio.getAttribute("data-value") ||
                      checkedRadio.getAttribute("aria-label") ||
                      checkedRadio.textContent.trim();
        if (label) { fields[title] = label; continue; }
      }

      // Checkboxes — collect all checked
      const checkedBoxes = block.querySelectorAll('[role="checkbox"][aria-checked="true"]');
      if (checkedBoxes.length > 0) {
        const vals = Array.from(checkedBoxes).map(
          (cb) => cb.getAttribute("data-value") || cb.getAttribute("aria-label") || cb.textContent.trim()
        );
        fields[title] = vals.join(", ");
        continue;
      }

      // Fallback — try any visible selected option text
      const selected = block.querySelector('[aria-selected="true"]');
      if (selected) {
        fields[title] = selected.textContent.trim();
      }
    }
    return fields;
  }

  function getGoogleFormsElements() {
    const map = {};
    const blocks = document.querySelectorAll(
      '[role="listitem"], .freebirdFormviewerComponentsQuestionBaseRoot'
    );
    for (const block of blocks) {
      const titleEl =
        block.querySelector('[role="heading"]') ||
        block.querySelector(".freebirdFormviewerComponentsQuestionBaseTitle");
      const title = titleEl ? titleEl.textContent.trim() : null;
      if (title) map[title] = [block];
    }
    return map;
  }

  // ----- Scan all form fields ----- //

  function scanFields() {
    // Google Forms path
    if (isGoogleForms()) return scanGoogleForms();

    const fields = {};
    document.querySelectorAll(SELECTORS).forEach((el) => {
      if (el.closest("#sadeed-overlay")) return;
      const name = getFieldName(el);
      const value = getFieldValue(el);
      if (name && value !== null && value !== "") {
        fields[name] = value;
      }
    });
    return fields;
  }

  // ----- Map field names back to DOM elements ----- //

  function getFieldElements() {
    if (isGoogleForms()) return getGoogleFormsElements();

    const map = {};
    document.querySelectorAll(SELECTORS).forEach((el) => {
      if (el.closest("#sadeed-overlay")) return;
      const name = getFieldName(el);
      if (name) {
        if (!map[name]) map[name] = [];
        map[name].push(el);
      }
    });
    return map;
  }

  // ----- Floating scan button ----- //

  function injectButton() {
    if (document.getElementById("sadeed-scan-btn")) return;

    const btn = document.createElement("button");
    btn.id = "sadeed-scan-btn";
    btn.innerHTML = `<span class="sadeed-icon">\u2714</span> Scan Form`;
    btn.addEventListener("click", handleScan);
    document.body.appendChild(btn);
  }

  // ----- Scan handler ----- //

  async function handleScan() {
    const btn = document.getElementById("sadeed-scan-btn");
    const fields = scanFields();

    if (Object.keys(fields).length === 0) {
      showOverlay(null, null);
      return;
    }

    btn.classList.add("sadeed-loading");
    btn.innerHTML = `<span class="sadeed-icon">\u23F3</span> Scanning...`;

    try {
      const response = await chrome.runtime.sendMessage({
        type: "DETECT",
        data: fields,
      });
      showOverlay(response, fields);
      highlightFields(response, fields);
    } catch (err) {
      console.error("Sadeed: detection failed", err);
      showOverlay({ error: err.message }, fields);
    } finally {
      btn.classList.remove("sadeed-loading");
      btn.innerHTML = `<span class="sadeed-icon">\u2714</span> Scan Form`;
    }
  }

  // ----- Highlight fields ----- //

  function clearHighlights() {
    document
      .querySelectorAll(".sadeed-field-ok, .sadeed-field-anomaly")
      .forEach((el) => {
        el.classList.remove("sadeed-field-ok", "sadeed-field-anomaly");
      });
  }

  function highlightFields(response, fields) {
    clearHighlights();
    if (!response || response.error) return;

    const fieldMap = getFieldElements();
    const anomalyFields = new Set();

    // Collect field names mentioned in failed expectations
    if (response.results) {
      for (const r of response.results) {
        if (r.is_anomaly && r.raw && r.raw.failed_expectations) {
          // GE failed expectations may reference column names
          for (const exp of r.raw.failed_expectations) {
            for (const fname of Object.keys(fields)) {
              if (exp.includes(fname)) anomalyFields.add(fname);
            }
          }
        }
        // If any strategy flags anomaly, mark all fields we can detect
        if (r.is_anomaly && r.explanation) {
          for (const fname of Object.keys(fields)) {
            if (r.explanation.includes(fname)) anomalyFields.add(fname);
          }
        }
      }
    }

    // If consensus is anomaly but no specific fields identified, flag all
    if (response.is_anomaly && anomalyFields.size === 0) {
      for (const fname of Object.keys(fields)) anomalyFields.add(fname);
    }

    for (const [fname, elements] of Object.entries(fieldMap)) {
      const cls = anomalyFields.has(fname)
        ? "sadeed-field-anomaly"
        : "sadeed-field-ok";
      elements.forEach((el) => el.classList.add(cls));
    }
  }

  // ----- Results overlay ----- //

  function removeOverlay() {
    const existing = document.getElementById("sadeed-overlay");
    if (existing) existing.remove();
    clearHighlights();
  }

  const STRATEGY_LABELS = {
    great_expectations: "Business Rules (GE)",
    svm: "Statistical Model (SVM)",
    llm: "LLM Review",
  };

  function showOverlay(response, fields) {
    removeOverlay();

    const overlay = document.createElement("div");
    overlay.id = "sadeed-overlay";

    const panel = document.createElement("div");
    panel.id = "sadeed-overlay-panel";

    // Close button
    const closeBtn = document.createElement("button");
    closeBtn.id = "sadeed-overlay-close";
    closeBtn.textContent = "\u2715";
    closeBtn.addEventListener("click", removeOverlay);
    panel.appendChild(closeBtn);

    // Title
    const title = document.createElement("h2");
    title.textContent = "Sadeed Detection Results";
    panel.appendChild(title);

    if (!fields || Object.keys(fields).length === 0) {
      const empty = document.createElement("div");
      empty.className = "sadeed-empty";
      empty.textContent = "No filled form fields found on this page.";
      panel.appendChild(empty);
    } else if (response && response.error) {
      const errDiv = document.createElement("div");
      errDiv.className = "sadeed-verdict anomaly";
      errDiv.textContent = "Connection error: " + response.error;
      panel.appendChild(errDiv);
    } else if (response) {
      // Verdict banner
      const verdict = document.createElement("div");
      verdict.className =
        "sadeed-verdict " + (response.is_anomaly ? "anomaly" : "clean");
      verdict.textContent = response.is_anomaly
        ? "\u26A0 Anomaly detected"
        : "\u2714 Clean — No anomalies detected";
      panel.appendChild(verdict);

      // LLM explanation (prominent)
      if (response.explanation) {
        const explCard = document.createElement("div");
        explCard.className = "sadeed-strategy flagged";
        explCard.style.borderRightColor = "#1a73e8";
        explCard.style.marginBottom = "14px";
        explCard.innerHTML = "";

        const explHeader = document.createElement("div");
        explHeader.className = "sadeed-strategy-header";
        const explTitle = document.createElement("span");
        explTitle.className = "sadeed-strategy-name";
        explTitle.textContent = "\u{1F4AC} LLM Explanation";
        explHeader.appendChild(explTitle);
        explCard.appendChild(explHeader);

        const explBody = document.createElement("div");
        explBody.className = "sadeed-strategy-explanation";
        explBody.style.fontSize = "13px";
        explBody.style.lineHeight = "1.7";
        explBody.textContent = response.explanation;
        explCard.appendChild(explBody);

        panel.appendChild(explCard);
      }

      // Strategy scores (compact — just name + score badge)
      if (response.results) {
        for (const r of response.results) {
          const card = document.createElement("div");
          card.className =
            "sadeed-strategy " + (r.is_anomaly ? "flagged" : "passed");

          const header = document.createElement("div");
          header.className = "sadeed-strategy-header";

          const name = document.createElement("span");
          name.className = "sadeed-strategy-name";
          name.textContent = STRATEGY_LABELS[r.strategy] || r.strategy;
          header.appendChild(name);

          const score = document.createElement("span");
          score.className = "sadeed-strategy-score";
          score.textContent =
            (r.is_anomaly ? "\u26A0 " : "\u2714 ") +
            (r.score !== null && r.score !== undefined
              ? (r.score * 100).toFixed(1) + "%"
              : "\u2014");
          header.appendChild(score);

          card.appendChild(header);
          panel.appendChild(card);
        }
      }
    }

    overlay.appendChild(panel);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) removeOverlay();
    });
    document.body.appendChild(overlay);
  }

  // ----- MutationObserver: re-inject button if removed ----- //

  const observer = new MutationObserver(() => {
    if (!document.getElementById("sadeed-scan-btn")) {
      injectButton();
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });

  // ----- Init ----- //

  injectButton();
})();
