/**
 * api.js — HTTP client for the Sadeed FastAPI backend.
 *
 * Thin wrapper around fetch() that handles base URL configuration,
 * JSON serialisation, and error normalisation.
 *
 * Exports:
 *   detectAnomaly(payload)       → Promise<DetectionResponse>
 *   getResult(recordId)          → Promise<DetectionResponse>
 *   listRecentAnomalies(limit)   → Promise<DetectionResponse[]>
 *   checkHealth()                → Promise<{great_expectations: bool, svm: bool, llm: bool, redis: bool}>
 */

const API_BASE_URL = "http://localhost:8000";

/**
 * Submit a data record for anomaly detection.
 * @param {Object} payload - { record_id, data }
 * @returns {Promise<Object>} DetectionResponse
 */
async function detectAnomaly(payload) {
  const res = await fetch(`${API_BASE_URL}/anomalies/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`detectAnomaly failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch a cached detection result by record ID.
 * @param {string} recordId
 * @returns {Promise<Object>} DetectionResponse
 */
async function getResult(recordId) {
  const res = await fetch(
    `${API_BASE_URL}/anomalies/detect/${encodeURIComponent(recordId)}`
  );
  if (!res.ok) {
    throw new Error(`getResult failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

/**
 * List the most recent anomaly detection results.
 * @param {number} [limit=20]
 * @returns {Promise<Object[]>} Array of DetectionResponse
 */
async function listRecentAnomalies(limit = 20) {
  const res = await fetch(
    `${API_BASE_URL}/anomalies/detect/recent?limit=${limit}`
  );
  if (!res.ok) {
    throw new Error(
      `listRecentAnomalies failed: ${res.status} ${res.statusText}`
    );
  }
  return res.json();
}

/**
 * Check the readiness of backend dependencies.
 * @returns {Promise<Object>} { great_expectations, svm, llm, redis }
 */
async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/anomalies/health`);
  if (!res.ok) {
    throw new Error(`checkHealth failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
