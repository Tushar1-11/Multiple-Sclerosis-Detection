/**
 * MSNet Medical AI Diagnostic Dashboard — Client Logic
 */

let allPatients = [];
let currentPatientId = 0;
let volumetricChart = null;

document.addEventListener("DOMContentLoaded", () => {
  initApp();
  setupEventListeners();
});

async function initApp() {
  try {
    const res = await fetch("/api/patients?limit=100");
    if (!res.ok) throw new Error("API Offline");

    const data = await res.json();
    allPatients = data.patients;

    document.getElementById("status-text").innerText = "PyTorch 1D-CNN Engine Connected";
    document.getElementById("server-status").style.borderColor = "rgba(16, 185, 129, 0.4)";
    document.getElementById("patient-count-badge").innerText = `${data.total} Total Patients`;

    renderPatientList(allPatients);

    if (allPatients.length > 0) {
      selectPatient(0);
    }
  } catch (err) {
    console.error(err);
    document.getElementById("status-text").innerText = "API Server Disconnected";
    document.getElementById("server-status").style.borderColor = "rgba(244, 63, 94, 0.4)";
    document.getElementById("patient-list-container").innerHTML =
      '<div class="loading-spinner">Unable to connect to FastAPI backend server. Start server with `python api_server.py`.</div>';
  }
}

function setupEventListeners() {
  // Search input
  const searchInput = document.getElementById("patient-search-input");
  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase();
    const filtered = allPatients.filter((p) => p.hash.toLowerCase().includes(query));
    renderPatientList(filtered);
  });

  // Filter pills
  const pills = document.querySelectorAll(".filter-pills .pill");
  pills.forEach((pill) => {
    pill.addEventListener("click", () => {
      pills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      const filter = pill.dataset.filter;

      if (filter === "all") {
        renderPatientList(allPatients);
      } else {
        const filtered = allPatients.filter((p) => String(p.phenotype_code) === filter);
        renderPatientList(filtered);
      }
    });
  });

  // Modal open / close
  const simBtn = document.getElementById("tab-simulator-btn");
  const modal = document.getElementById("simulator-modal");
  const closeBtn = document.getElementById("close-modal-btn");
  const runSimBtn = document.getElementById("run-sim-btn");

  simBtn.addEventListener("click", () => modal.classList.add("active"));
  closeBtn.addEventListener("click", () => modal.classList.remove("active"));

  runSimBtn.addEventListener("click", runCustomSimulation);
}

function renderPatientList(patients) {
  const container = document.getElementById("patient-list-container");
  container.innerHTML = "";

  if (patients.length === 0) {
    container.innerHTML = '<div class="text-muted" style="padding: 1rem;">No matching patient records found.</div>';
    return;
  }

  patients.forEach((p) => {
    const card = document.createElement("div");
    card.className = `patient-item ${p.id === currentPatientId ? "selected" : ""}`;
    card.dataset.id = p.id;

    let badgeClass = "badge-hc";
    if (p.phenotype_code === 1) badgeClass = "badge-rrms";
    if (p.phenotype_code === 2) badgeClass = "badge-prog";

    const edssStr = p.edss !== null ? `EDSS ${p.edss.toFixed(1)}` : "No EDSS";

    card.innerHTML = `
      <div class="patient-item-header">
        <span class="patient-hash-short">${p.hash.substring(0, 10)}...</span>
        <span class="badge ${badgeClass}">${p.phenotype_label.split(" ")[0]}</span>
      </div>
      <div class="patient-meta-text">
        <span>Age: ${p.age}</span>
        <span>Sex: ${p.sex}</span>
        <span style="color: var(--accent-amber); font-weight:600;">${edssStr}</span>
      </div>
    `;

    card.addEventListener("click", () => {
      document.querySelectorAll(".patient-item").forEach((el) => el.classList.remove("selected"));
      card.classList.add("selected");
      selectPatient(p.id);
    });

    container.appendChild(card);
  });
}

async function selectPatient(patientId) {
  currentPatientId = patientId;

  try {
    const res = await fetch(`/api/patient/${patientId}`);
    if (!res.ok) throw new Error("Failed to fetch patient");
    const p = await res.json();

    // Update Header Card
    document.getElementById("display-patient-hash").innerText = p.hash;
    document.getElementById("display-patient-title").innerText = `Patient #${p.id + 1}`;
    document.getElementById("display-age").innerText = `${p.age} yrs`;
    document.getElementById("display-sex").innerText = p.sex;
    document.getElementById("display-duration").innerText = p.disease_duration ? `${p.disease_duration.toFixed(1)} yrs` : "N/A (Control)";
    document.getElementById("display-t2lv").innerText = p.t2_lv ? `${p.t2_lv.toFixed(0)} mm³` : "0 mm³";

    const subtypeBadge = document.getElementById("display-subtype-badge");
    subtypeBadge.innerText = p.phenotype_label;
    subtypeBadge.className = `badge badge-lg ${getBadgeClass(p.phenotype_code)}`;

    // Update EDSS Card
    const edssVal = p.edss !== null ? p.edss.toFixed(1) : "0.0";
    document.getElementById("display-edss-score").innerText = edssVal;
    document.getElementById("display-edss-tier").innerText = p.edss_status;

    const meterPct = p.edss !== null ? Math.min((p.edss / 10.0) * 100, 100) : 0;
    document.getElementById("edss-meter-fill").style.width = `${meterPct}%`;

    // Render Volumetric Chart
    renderVolumetricChart(p.volumetrics);

    // Run Live PyTorch Inference
    runModelPrediction({ patient_id: patientId });

  } catch (err) {
    console.error("Error selecting patient:", err);
  }
}

async function runModelPrediction(predictBody) {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(predictBody),
    });

    if (!res.ok) throw new Error("Prediction API Error");
    const pred = await res.json();

    // Update AI Prediction Card
    document.getElementById("ai-pred-result").innerText = pred.predicted_label;
    document.getElementById("ai-confidence").innerText = `${pred.confidence.toFixed(1)}% Confidence`;

    // Update Probability Bars
    document.getElementById("bar-hc").style.width = `${pred.probabilities["Healthy Control"]}%`;
    document.getElementById("bar-rrms").style.width = `${pred.probabilities["RRMS"]}%`;
    document.getElementById("bar-prog").style.width = `${pred.probabilities["Progressive MS"]}%`;

    document.getElementById("pct-hc").innerText = `${pred.probabilities["Healthy Control"]}%`;
    document.getElementById("pct-rrms").innerText = `${pred.probabilities["RRMS"]}%`;
    document.getElementById("pct-prog").innerText = `${pred.probabilities["Progressive MS"]}%`;

    // Render Feature Importances
    renderSaliency(pred.top_feature_importances);

  } catch (err) {
    console.error("Inference Error:", err);
  }
}

function renderVolumetricChart(v) {
  const ctx = document.getElementById("volumetricChart").getContext("2d");

  const labels = [
    "Cortical Gray",
    "White Matter",
    "Thalamus",
    "Hippocampus",
    "Other DGM",
    "Brainstem",
    "Cerebellum GM",
    "Cerebellum WM",
  ];

  const data = [
    v.ncorticalGREY,
    v.nWHITE,
    v.nThalamus,
    v.nHippocampus,
    v.nOtherDGM,
    v.nBrainstem,
    v.nGM_Cerebellum,
    v.nWM_Cerebellum,
  ];

  if (volumetricChart) {
    volumetricChart.destroy();
  }

  volumetricChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Brain Volume (ml)",
          data: data,
          backgroundColor: "rgba(6, 182, 212, 0.5)",
          borderColor: "#06b6d4",
          borderWidth: 1.5,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8", font: { size: 11 } },
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

function renderSaliency(importances) {
  const container = document.getElementById("saliency-container");
  container.innerHTML = "";

  if (!importances || importances.length === 0) return;

  const maxVal = importances[0].saliency || 1.0;

  importances.forEach((item) => {
    const pct = Math.min((item.saliency / maxVal) * 100, 100);

    const div = document.createElement("div");
    div.className = "saliency-item";
    div.innerHTML = `
      <div style="display:flex; justify-content:space-between;">
        <span class="saliency-name">${item.feature}</span>
        <span style="font-size:0.75rem; color:var(--accent-cyan);">${item.saliency.toFixed(4)}</span>
      </div>
      <div class="saliency-bar-fill" style="width: ${pct}%;"></div>
    `;
    container.appendChild(div);
  });
}

async function runCustomSimulation() {
  const age = parseFloat(document.getElementById("sim-age").value);
  const sex_code = parseInt(document.getElementById("sim-sex").value);
  const edss = parseFloat(document.getElementById("sim-edss").value);
  const disease_duration = parseFloat(document.getElementById("sim-duration").value);
  const t2_lv = parseFloat(document.getElementById("sim-t2lv").value);
  const ncorticalGREY = parseFloat(document.getElementById("sim-grey").value);
  const nWHITE = parseFloat(document.getElementById("sim-white").value);
  const nThalamus = parseFloat(document.getElementById("sim-thalamus").value);

  const reqBody = {
    age, sex_code, edss, disease_duration, t2_lv,
    ncorticalGREY, nWHITE, nThalamus
  };

  document.getElementById("simulator-modal").classList.remove("active");

  // Update header display for simulation
  document.getElementById("display-patient-hash").innerText = "SIMULATED_CUSTOM_PATIENT";
  document.getElementById("display-patient-title").innerText = "Simulated Patient";
  document.getElementById("display-age").innerText = `${age} yrs`;
  document.getElementById("display-sex").innerText = sex_code === 1 ? "Male" : "Female";
  document.getElementById("display-duration").innerText = `${disease_duration.toFixed(1)} yrs`;
  document.getElementById("display-t2lv").innerText = `${t2_lv.toFixed(0)} mm³`;

  document.getElementById("display-edss-score").innerText = edss.toFixed(1);
  document.getElementById("display-edss-tier").innerText = `Simulated EDSS ${edss.toFixed(1)}`;
  document.getElementById("edss-meter-fill").style.width = `${Math.min((edss / 10.0) * 100, 100)}%`;

  runModelPrediction(reqBody);
}

function getBadgeClass(code) {
  if (code === 1) return "badge-rrms";
  if (code === 2) return "badge-prog";
  return "badge-hc";
}
