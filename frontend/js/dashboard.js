/**
 * dashboard.js — Main dashboard controller.
 *
 * Responsibilities:
 *   - Poll the backend every 10 seconds for recent anomaly events
 *   - Render strategy summary cards (counts, last-seen timestamps)
 *   - Populate the recent-anomalies table
 *   - Feed new data points to charts.js
 *   - Update the status bar with detector health indicators
 *
 * Entry point: DOMContentLoaded event listener at the bottom of this file.
 */

const POLL_INTERVAL_MS = 10_000;

/** @type {Chart|null} */
let anomalyChart = null;

/**
 * Render three summary cards: Total, Anomalous, Clean.
 * @param {Object[]} results - Recent DetectionResponse objects
 */
function renderSummaryCards(results) {
  const container = document.getElementById("summary-cards");
  const total = results.length;
  const anomalous = results.filter((r) => r.is_anomaly).length;
  const clean = total - anomalous;

  container.innerHTML = `
    <div class="card card--total">
      <span class="card__label">Total Scanned</span>
      <span class="card__value">${total}</span>
    </div>
    <div class="card card--anomaly">
      <span class="card__label">Anomalous</span>
      <span class="card__value">${anomalous}</span>
    </div>
    <div class="card card--clean">
      <span class="card__label">Clean</span>
      <span class="card__value">${clean}</span>
    </div>
  `;
}

/**
 * Populate the recent-anomalies table with the latest results.
 * @param {Object[]} results - Recent DetectionResponse objects
 */
function renderAnomaliesTable(results) {
  const container = document.getElementById("recent-anomalies");

  if (!results.length) {
    container.innerHTML = `<p class="empty-state">No recent detections.</p>`;
    return;
  }

  const headerRow = `
    <tr>
      <th>Record ID</th>
      <th>Verdict</th>
      <th>GE Score</th>
      <th>SVM Score</th>
      <th>LLM Score</th>
      <th>Explanation</th>
    </tr>`;

  const bodyRows = results
    .map((r) => {
      const ge = r.results.find((s) => s.strategy === "great_expectations");
      const svm = r.results.find((s) => s.strategy === "svm");
      const llm = r.results.find((s) => s.strategy === "llm");

      const scoreCell = (s) =>
        s != null ? s.score.toFixed(3) : "&mdash;";

      const verdictClass = r.is_anomaly ? "verdict--anomaly" : "verdict--clean";
      const verdictLabel = r.is_anomaly ? "ANOMALY" : "CLEAN";

      const explanation = r.explanation
        ? escapeHtml(truncate(r.explanation, 120))
        : "&mdash;";

      return `
      <tr>
        <td class="mono">${escapeHtml(r.record_id)}</td>
        <td><span class="verdict ${verdictClass}">${verdictLabel}</span></td>
        <td>${scoreCell(ge)}</td>
        <td>${scoreCell(svm)}</td>
        <td>${scoreCell(llm)}</td>
        <td class="explanation-cell">${explanation}</td>
      </tr>`;
    })
    .join("");

  container.innerHTML = `
    <table class="anomalies-table">
      <thead>${headerRow}</thead>
      <tbody>${bodyRows}</tbody>
    </table>`;
}

/**
 * Update the status bar with dependency health indicators.
 * @param {Object} health - { great_expectations, svm, llm, redis }
 */
function renderStatusBar(health) {
  const bar = document.getElementById("status-bar");

  const items = [
    { label: "Great Expectations", ok: health.great_expectations },
    { label: "OC-SVM", ok: health.svm },
    { label: "LLM (Ollama)", ok: health.llm },
    { label: "Redis", ok: health.redis },
  ];

  bar.innerHTML = items
    .map((item) => {
      const dotClass = item.ok ? "dot--green" : "dot--red";
      return `<span class="status-item"><span class="dot ${dotClass}"></span>${item.label}</span>`;
    })
    .join("");
}

/**
 * Main polling loop — fetches data and refreshes all UI sections.
 */
async function poll() {
  try {
    const [results, health] = await Promise.all([
      listRecentAnomalies(20),
      checkHealth(),
    ]);

    renderSummaryCards(results);
    renderAnomaliesTable(results);
    renderStatusBar(health);

    if (anomalyChart) {
      const anomalous = results.filter((r) => r.is_anomaly).length;
      const clean = results.length - anomalous;
      updateChart(anomalyChart, { anomalous, clean });
    }
  } catch (err) {
    console.error("Poll error:", err);
    // Show connectivity issue in status bar
    renderStatusBar({
      great_expectations: false,
      svm: false,
      llm: false,
      redis: false,
    });
  }
}

// --- Helpers ---

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function truncate(str, max) {
  return str.length > max ? str.slice(0, max) + "..." : str;
}

// --- Initialisation ---

document.addEventListener("DOMContentLoaded", () => {
  anomalyChart = initChart("anomaly-chart");
  poll();
  setInterval(poll, POLL_INTERVAL_MS);
});
