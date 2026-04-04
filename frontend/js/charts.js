/**
 * charts.js — Anomaly ratio doughnut chart using Chart.js.
 *
 * Exports:
 *   initChart(canvasId)              → Chart instance
 *   updateChart(chart, data)         → void
 *   clearChart(chart)                → void
 */

/**
 * Initialise a doughnut chart on the given canvas element.
 * @param {string} canvasId - ID of the <canvas> element
 * @returns {Object} Chart.js instance
 */
function initChart(canvasId) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");

  return new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Anomalous", "Clean"],
      datasets: [
        {
          data: [0, 0],
          backgroundColor: [
            getComputedStyle(document.documentElement)
              .getPropertyValue("--color-anomaly")
              .trim() || "#f56565",
            getComputedStyle(document.documentElement)
              .getPropertyValue("--color-normal")
              .trim() || "#68d391",
          ],
          borderColor: "transparent",
          hoverOffset: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "65%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#e2e8f0",
            font: { size: 13 },
            padding: 16,
          },
        },
        title: {
          display: true,
          text: "Anomaly Ratio",
          color: "#e2e8f0",
          font: { size: 16, weight: "600" },
          padding: { bottom: 12 },
        },
      },
    },
  });
}

/**
 * Update the doughnut chart with new anomaly / clean counts.
 * @param {Object} chart - Chart.js instance from initChart
 * @param {{ anomalous: number, clean: number }} data
 */
function updateChart(chart, data) {
  chart.data.datasets[0].data = [data.anomalous, data.clean];
  chart.update();
}

/**
 * Clear the chart data and reset to zeros.
 * @param {Object} chart - Chart.js instance from initChart
 */
function clearChart(chart) {
  chart.data.datasets[0].data = [0, 0];
  chart.update();
}
