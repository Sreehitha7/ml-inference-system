/**
 * ML Inference Console — Frontend Logic
 * Handles API communication, state transitions, and UI rendering.
 */

"use strict";

// ── Constants ────────────────────────────────────────────────────────

const PRESETS = [
  { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2 }, // setosa
  { sepal_length: 6.0, sepal_width: 2.9, petal_length: 4.5, petal_width: 1.5 }, // versicolor
  { sepal_length: 6.7, sepal_width: 3.0, petal_length: 5.2, petal_width: 2.3 }, // virginica
];

const SPECIES_COLORS = ["species-0", "species-1", "species-2"];

// ── State ────────────────────────────────────────────────────────────

let requestCount = 0;
let isLoading = false;

// ── DOM refs (resolved once) ─────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const dom = {
  healthBadge: $("healthBadge"),
  healthDot:   $("healthDot"),
  healthLabel: $("healthLabel"),
  predictBtn:  $("predictBtn"),
  reqCount:    $("requestCount"),
  footerModel: $("footerModel"),

  // inputs
  sepalLength: $("sepalLength"),
  sepalWidth:  $("sepalWidth"),
  petalLength: $("petalLength"),
  petalWidth:  $("petalWidth"),

  // result panels
  idle:    $("resultIdle"),
  loading: $("resultLoading"),
  output:  $("resultOutput"),
  error:   $("resultError"),

  // output fields
  species:    $("resultSpecies"),
  confBar:    $("confBar"),
  confPct:    $("confPct"),
  probTable:  $("probTable"),
  metaLatency:$("metaLatency"),
  metaModel:  $("metaModel"),
  errorMsg:   $("errorMsg"),
};

// ── Health Check ─────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();

    if (data.model_loaded) {
      setHealth("online", `Model Loaded · ${data.status.toUpperCase()}`);
      dom.footerModel.textContent = data.model_description;
    } else {
      setHealth("offline", "Model Not Loaded");
      dom.footerModel.textContent = "No model";
    }
  } catch {
    setHealth("offline", "API Unreachable");
    dom.footerModel.textContent = "—";
  }
}

function setHealth(state, label) {
  dom.healthBadge.className = `health-badge ${state}`;
  dom.healthLabel.textContent = label;
}

// Poll health every 15 seconds
checkHealth();
setInterval(checkHealth, 15_000);

// ── Prediction ───────────────────────────────────────────────────────

async function runPrediction() {
  if (isLoading) return;

  // Validate and collect inputs
  const fields = [
    { key: "sepal_length", el: dom.sepalLength, label: "Sepal Length" },
    { key: "sepal_width",  el: dom.sepalWidth,  label: "Sepal Width"  },
    { key: "petal_length", el: dom.petalLength, label: "Petal Length" },
    { key: "petal_width",  el: dom.petalWidth,  label: "Petal Width"  },
  ];

  clearFieldErrors();
  const body = {};
  let hasError = false;

  for (const { key, el, label } of fields) {
    const val = parseFloat(el.value);
    if (isNaN(val) || el.value.trim() === "") {
      markFieldError(el, `${label} is required`);
      hasError = true;
    } else {
      body[key] = val;
    }
  }

  if (hasError) return;

  // Update UI to loading state
  isLoading = true;
  setState("loading");
  dom.predictBtn.disabled = true;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();

    if (!res.ok) {
      const msg = Array.isArray(data.detail)
        ? data.detail.join("\n")
        : data.detail || `HTTP ${res.status}`;
      showError(msg);
      return;
    }

    requestCount++;
    dom.reqCount.textContent = `${requestCount} request${requestCount !== 1 ? "s" : ""}`;
    renderResult(data);

  } catch (err) {
    showError(`Network error: ${err.message}`);
  } finally {
    isLoading = false;
    dom.predictBtn.disabled = false;
  }
}

// ── Render Result ────────────────────────────────────────────────────

function renderResult(data) {
  const idx = data.prediction_index;

  // Species name
  dom.species.textContent = data.prediction;
  dom.species.className = `result-value ${SPECIES_COLORS[idx] ?? ""}`;

  // Confidence bar
  const pct = Math.round(data.confidence * 100);
  dom.confBar.style.width = `${pct}%`;
  dom.confBar.style.background = getSpeciesColor(idx);
  dom.confPct.textContent = `${pct}%`;

  // Probability breakdown
  dom.probTable.innerHTML = "";
  const entries = Object.entries(data.probabilities);
  entries.forEach(([species, prob], i) => {
    const pctVal = Math.round(prob * 100);
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name">${species}</span>
      <div class="prob-bar-wrap">
        <div class="prob-bar ${SPECIES_COLORS[i] ?? ""}" style="width: ${pctVal}%"></div>
      </div>
      <span class="prob-val">${pctVal}%</span>
    `;
    dom.probTable.appendChild(row);
  });

  // Meta
  dom.metaLatency.textContent = `${data.latency_ms.toFixed(2)} ms`;
  dom.metaModel.textContent = data.model_description;

  setState("output");
}

function getSpeciesColor(idx) {
  const colors = ["#2dd4a0", "#3d7fff", "#c97bff"];
  return colors[idx] ?? "#3d7fff";
}

// ── UI States ────────────────────────────────────────────────────────

function setState(state) {
  dom.idle.classList.add("hidden");
  dom.loading.classList.add("hidden");
  dom.output.classList.add("hidden");
  dom.error.classList.add("hidden");

  if (state === "loading") dom.loading.classList.remove("hidden");
  else if (state === "output") dom.output.classList.remove("hidden");
  else if (state === "error") dom.error.classList.remove("hidden");
  else dom.idle.classList.remove("hidden");
}

function showError(msg) {
  dom.errorMsg.textContent = msg;
  setState("error");
}

// ── Field Validation ─────────────────────────────────────────────────

function markFieldError(el, _msg) {
  el.classList.add("error");
}

function clearFieldErrors() {
  [dom.sepalLength, dom.sepalWidth, dom.petalLength, dom.petalWidth]
    .forEach((el) => el.classList.remove("error"));
}

// ── Presets ──────────────────────────────────────────────────────────

function fillPreset(idx) {
  const p = PRESETS[idx];
  if (!p) return;
  dom.sepalLength.value = p.sepal_length;
  dom.sepalWidth.value  = p.sepal_width;
  dom.petalLength.value = p.petal_length;
  dom.petalWidth.value  = p.petal_width;
  clearFieldErrors();
}

function fillSample() {
  // Pick a random preset
  fillPreset(Math.floor(Math.random() * PRESETS.length));
}

// ── Keyboard shortcut: Enter to predict ──────────────────────────────

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT") runPrediction();
  }
});
