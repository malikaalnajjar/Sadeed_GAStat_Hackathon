// ------------------------------------------------------------------ //
// Sadeed — Background Service Worker                                 //
// Relays detection requests to the FastAPI backend                    //
// ------------------------------------------------------------------ //

const API_URLS = [
  "http://localhost:8000/anomalies",
  "https://sadeed.aldharrab.co.uk/anomalies",
];
let API_BASE = API_URLS[0];

async function resolveApiBase() {
  for (const url of API_URLS) {
    try {
      const r = await fetch(`${url}/health`, { method: "GET", signal: AbortSignal.timeout(3000) });
      if (r.ok) { API_BASE = url; return; }
    } catch { /* try next */ }
  }
}
resolveApiBase();

function generateUUID() {
  return crypto.randomUUID();
}

// ----- Badge helpers ----- //

function setBadgeAnomaly(tabId) {
  chrome.action.setBadgeText({ text: "!", tabId });
  chrome.action.setBadgeBackgroundColor({ color: "#ea4335", tabId });
}

function setBadgeClean(tabId) {
  chrome.action.setBadgeText({ text: "\u2713", tabId });
  chrome.action.setBadgeBackgroundColor({ color: "#34a853", tabId });
}

function setBadgeError(tabId) {
  chrome.action.setBadgeText({ text: "E", tabId });
  chrome.action.setBadgeBackgroundColor({ color: "#f9ab00", tabId });
}

function clearBadge(tabId) {
  chrome.action.setBadgeText({ text: "", tabId });
}

// ----- Message listener ----- //

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "DETECT") {
    handleDetect(message.data, sender.tab?.id)
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse({ error: err.message }));
    return true; // async
  }

  if (message.type === "DETECT_ROW") {
    handleDetect(message.data, null)
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse({ error: err.message }));
    return true;
  }

  if (message.type === "HEALTH_CHECK") {
    fetchHealth()
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse({ error: err.message }));
    return true;
  }

  if (message.type === "GET_LAST_RESULT") {
    chrome.storage.session.get("lastResult", (data) => {
      sendResponse(data.lastResult || null);
    });
    return true;
  }
});

// ----- Detection ----- //

async function handleDetect(data, tabId) {
  const recordId = generateUUID();
  const payload = { record_id: recordId, data };

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    const resp = await fetch(`${API_BASE}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }

    const result = await resp.json();

    // Store last result
    chrome.storage.session.set({ lastResult: result });

    // Update badge for the originating tab
    if (tabId) {
      if (result.is_anomaly) {
        setBadgeAnomaly(tabId);
      } else {
        setBadgeClean(tabId);
      }
    }

    return result;
  } catch (err) {
    if (tabId) setBadgeError(tabId);
    throw err;
  }
}

// ----- Health ----- //

async function fetchHealth() {
  const resp = await fetch(`${API_BASE}/health`, { method: "GET" });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}
