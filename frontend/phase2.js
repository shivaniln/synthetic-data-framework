(() => {
  "use strict";

  const tabularFunctions = {
    startAudit: window.startAudit,
    renderResults: window.renderResults,
    renderSliders: window.renderSliders,
    renderModelToggles: window.renderModelToggles,
    renderRunModes: window.renderRunModes,
    updateRunDisplay: window.updateRunDisplay,
    updateModelCount: window.updateModelCount
  };

  const TIME_SERIES_MODES = {
    quick: {
      rows: 240,
      blockSize: 12,
      queries: 50,
      label: "Quick demo",
      desc: "240 rows - 12-step sequence blocks",
      timeLabel: "< 1 min",
      timeClass: "fast"
    },
    standard: {
      rows: 2000,
      blockSize: 24,
      queries: 100,
      label: "Standard",
      desc: "2k rows - 24-step sequence blocks",
      timeLabel: "~1 min",
      timeClass: "med"
    },
    full: {
      rows: 8000,
      blockSize: 48,
      queries: 200,
      label: "Full sequence",
      desc: "8k rows - 48-step sequence blocks",
      timeLabel: "~2 min",
      timeClass: "slow"
    }
  };

  const TIME_SERIES_ICONS = {
    quick: "⚡",
    standard: "〰",
    full: "📈"
  };

  state.modality = "tabular";
  state.timeSeriesRunMode = "standard";
  state.timeSeriesSliders = {
    "Sequence similarity risk limit": {
      key: "sequence_similarity",
      val: 10
    }
  };

  function isTimeSeries() {
    return state.modality === "time_series";
  }

  function addPhase2Styles() {
    const style = document.createElement("style");

    style.textContent = `
      .modality-picker {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 0 0 18px;
      }

      .modality-btn {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 150px;
        padding: 10px 14px;
        border: 1.5px solid rgba(121, 116, 126, 0.24);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.62);
        color: var(--md-on-bg);
        cursor: pointer;
        font: inherit;
        text-align: left;
        transition: 180ms var(--ease);
      }

      .modality-btn:hover:not(:disabled) {
        border-color: var(--md-primary);
        background: rgba(103, 80, 164, 0.05);
      }

      .modality-btn.active {
        border-color: var(--md-primary);
        background: rgba(103, 80, 164, 0.10);
        box-shadow: inset 0 0 0 1px var(--md-primary);
      }

      .modality-btn:disabled {
        cursor: not-allowed;
        opacity: 0.52;
      }

      .modality-btn-icon {
        font-size: 1.15rem;
      }

      .modality-btn-title {
        display: block;
        font-size: 0.84rem;
        font-weight: 700;
      }

      .modality-btn-sub {
        display: block;
        margin-top: 1px;
        font-size: 0. सातrem;
        color: var(--md-on-surf-var);
      }

      .time-series-note {
        margin: 0 0 16px;
        padding: 11px 14px;
        border-radius: 12px;
        background: rgba(103, 80, 164, 0.07);
        color: var(--md-on-surf-var);
        font-size: 0.8rem;
        line-height: 1.5;
      }
    `.replace("0. सातrem", "0.72rem");

    document.head.appendChild(style);
  }

  function addModalityPicker() {
    const uploadZone = document.getElementById("uploadZone");

    if (!uploadZone || document.getElementById("modalityPicker")) {
      return;
    }

    const picker = document.createElement("div");
    picker.id = "modalityPicker";
    picker.className = "modality-picker";
    picker.innerHTML = `
      <button class="modality-btn active" type="button" data-modality="tabular">
        <span class="modality-btn-icon">▦</span>
        <span>
          <span class="modality-btn-title">Tabular data</span>
          <span class="modality-btn-sub">CSV records and columns</span>
        </span>
      </button>

      <button class="modality-btn" type="button" data-modality="time_series">
        <span class="modality-btn-icon">〰</span>
        <span>
          <span class="modality-btn-title">Time-series data</span>
          <span class="modality-btn-sub">Timestamped numerical sequences</span>
        </span>
      </button>

      <button id="imageModalityButton" class="modality-btn" type="button">
        <span class="modality-btn-icon">▧</span>
        <span>
          <span class="modality-btn-title">Image data</span>
          <span class="modality-btn-sub">Local image audit</span>
        </span>
      </button>
    `;

    uploadZone.parentNode.insertBefore(picker, uploadZone);

    picker.querySelectorAll("[data-modality]").forEach(button => {
      button.addEventListener("click", () => {
        selectModality(button.dataset.modality);
      });
    });
  }

  function selectModality(modality) {
    state.modality = modality;

    document.querySelectorAll("[data-modality]").forEach(button => {
      button.classList.toggle(
        "active",
        button.dataset.modality === modality
      );
    });

    const secSubs = document.querySelectorAll(".sec-head .sec-sub");
    const uploadCta = document.querySelector(".upload-cta");
    const uploadHint = document.querySelector(".upload-hint");

    if (isTimeSeries()) {
      secSubs[0].textContent =
        "Upload a timestamped CSV with one time column and one or more numerical measurement columns.";

      secSubs[1].textContent =
        "MIDST will preserve time order, generate a synthetic sequence, and evaluate privacy risk, trends, autocorrelation, and distribution similarity.";

      uploadCta.textContent = "Click to choose a time-series CSV file";
      uploadHint.textContent =
        "Include a timestamp, time, date, or datetime column";

      document.getElementById("cfgModelCount").textContent =
        "1 model selected";

      renderTimeSeriesRunModes();
      renderTimeSeriesSliders();
      renderTimeSeriesModels();
      updateTimeSeriesRunDisplay();
    } else {
      secSubs[0].textContent =
        "Drop any CSV file here. After the run, you will see the original dataset shape, the generated dataset shape, and a few example rows with changed values highlighted.";

      secSubs[1].textContent =
        "Pick which models to test and set the privacy thresholds they must satisfy. Target-column prediction and TSTR gap are removed for now to keep the workflow focused.";

      uploadCta.textContent = "Click to choose a CSV file";
      uploadHint.textContent = "or drag and drop it here · max 50 MB";

      tabularFunctions.renderRunModes();
      tabularFunctions.renderSliders();
      tabularFunctions.renderModelToggles();
      tabularFunctions.updateModelCount();
    }
  }

  function renderTimeSeriesRunModes() {
    const container = document.getElementById("runModeGroup");
    container.innerHTML = "";

    Object.entries(TIME_SERIES_MODES).forEach(([key, mode]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className =
        "run-mode-btn" +
        (state.timeSeriesRunMode === key ? " active" : "");

      button.innerHTML = `
        <div class="rmb-icon ${key}">
          ${TIME_SERIES_ICONS[key]}
        </div>
        <div style="flex:1">
          <div class="rmb-name">${mode.label}</div>
          <div class="rmb-desc">${mode.desc}</div>
        </div>
        <span class="rmb-time ${mode.timeClass}">
          ${mode.timeLabel}
        </span>
      `;

      button.addEventListener("click", () => {
        state.timeSeriesRunMode = key;
        renderTimeSeriesRunModes();
        updateTimeSeriesRunDisplay();
      });

      container.appendChild(button);
    });
  }

  function renderTimeSeriesSliders() {
    const container = document.getElementById("sliderContainer");
    container.innerHTML = "";

    Object.entries(state.timeSeriesSliders).forEach(([label, slider]) => {
      const section = document.createElement("div");
      section.className = "slider-section";

      section.innerHTML = `
        <div class="slider-head">
          <span class="slider-name">${label}</span>
          <span class="slider-val" id="tsv_${slider.key}">
            ${slider.val}%
          </span>
        </div>
        <div class="slider-track" id="tst_${slider.key}">
          <div class="slider-fill" id="tsf_${slider.key}"
            style="width:${slider.val}%"></div>
          <div class="slider-thumb" id="tsth_${slider.key}"
            style="left:${slider.val}%"></div>
        </div>
      `;

      const track = section.querySelector(".slider-track");

      track.addEventListener("click", event => {
        const rect = track.getBoundingClientRect();
        const value = Math.max(
          1,
          Math.min(
            50,
            Math.round(
              ((event.clientX - rect.left) / rect.width) * 100
            )
          )
        );

        slider.val = value;

        document.getElementById(`tsf_${slider.key}`).style.width =
          `${value}%`;

        document.getElementById(`tsth_${slider.key}`).style.left =
          `${value}%`;

        document.getElementById(`tsv_${slider.key}`).textContent =
          `${value}%`;
      });

      container.appendChild(section);
    });
  }

  function renderTimeSeriesModels() {
    const container = document.getElementById("modelToggles");

    container.innerHTML = `
      <div class="toggle-row">
        <div>
          <div class="toggle-label-text">Block Bootstrap</div>
          <div class="toggle-sub-text">
            Sequence-aware baseline using short contiguous time blocks
          </div>
        </div>
        <div class="toggle-switch on disabled"></div>
      </div>

      <div class="time-series-note">
        The first time-series version uses block bootstrap sampling.
        It keeps local sequence behaviour while adding controlled noise
        to avoid returning the original series unchanged.
      </div>
    `;
  }

  function updateTimeSeriesRunDisplay() {
    const settings =
      TIME_SERIES_MODES[state.timeSeriesRunMode];

    const rowCap = state.rowCount
      ? Math.min(state.rowCount, settings.rows)
      : settings.rows;

    document.getElementById("cfgMaxRows").textContent =
      Number(rowCap).toLocaleString();

    document.getElementById("cfgEpochs").textContent =
      `${settings.blockSize} rows`;

    document.getElementById("cfgNAttacks").textContent =
      settings.queries;

    document.getElementById("estRuntime").textContent =
      settings.timeLabel;

    const labels = document.querySelectorAll(
      ".config-grid .cfg-card:first-child .cfg-key"
    );

    if (labels.length >= 3) {
      labels[0].textContent = "Rows used";
      labels[1].textContent = "Sequence block size";
      labels[2].textContent = "Similarity checks";
    }
  }

  async function startTimeSeriesAudit() {
    if (!state.filename) {
      alert("Please upload a timestamped CSV file first.");
      return;
    }

    if (state.running) {
      alert("An audit is already running.");
      return;
    }

    const settings =
      TIME_SERIES_MODES[state.timeSeriesRunMode];

    const maxRows = state.rowCount
      ? Math.min(state.rowCount, settings.rows)
      : settings.rows;

    const riskLimit =
      state.timeSeriesSliders[
        "Sequence similarity risk limit"
      ].val / 100;

    const body = {
      modality: "time_series",
      filename: state.filename,
      max_rows: maxRows,
      train_ratio: 0.5,
      models: {
        block_bootstrap: {
          block_size: settings.blockSize,
          noise_scale: 0.03,
          window_size: Math.max(6, Math.floor(settings.blockSize / 2)),
          max_queries: settings.queries,
          random_seed: 42
        }
      },
      privacy_thresholds: {
        sequence_similarity: riskLimit
      },
      utility_thresholds: {
        composite_utility: 0.70
      },
      score_weights: {
        privacy: 0.5,
        utility: 0.5
      }
    };

    state.running = true;

    document.getElementById("btnRun").disabled = true;
    document.getElementById("btnRun").textContent = "Running...";
    document.getElementById("progressBanner").classList.add("show");
    document.getElementById("resultsSection").classList.remove("show");
    document.getElementById("placeholder").style.display = "block";

    setStatus("Running...", "running");
    clearLogs();
    startProgressSim(1);
    openLogStream();

    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });

      const data = await response.json();

      if (data.error) {
        addLog("error", data.error);
        stopRun("error");
      }
    } catch (error) {
      addLog(
        "error",
        `Failed to start time-series audit: ${error.message || error}`
      );

      stopRun("error");
    }
  }

  function renderTimeSeriesCards(results, bestModel) {
    const grid = document.getElementById("modelsGrid");
    grid.innerHTML = "";

    results.forEach(result => {
      const isBest = result.model === bestModel;
      const pass = Boolean(result.overall_pass);
      const risk = Number(result.sequence_similarity_risk || 0);

      const metrics = [
        {
          label: "Sequence risk",
          value: risk,
          type: "risk",
          threshold: 0.1
        },
        {
          label: "Trend similarity",
          value: Number(result.trend_similarity || 0),
          type: "util",
          threshold: 0.7
        },
        {
          label: "Autocorrelation",
          value: Number(result.autocorrelation_similarity || 0),
          type: "util",
          threshold: 0.7
        },
        {
          label: "Distribution match",
          value: Number(result.distribution_similarity || 0),
          type: "util",
          threshold: 0.7
        }
      ];

      const metricsHtml = metrics.map(metric => {
        const percentage = Math.round(metric.value * 100);
        const good = metric.type === "risk"
          ? metric.value <= metric.threshold
          : metric.value >= metric.threshold;

        const width = metric.type === "risk"
          ? Math.min(percentage * 3, 100)
          : Math.min(percentage, 100);

        return `
          <div class="met-row">
            <span class="met-label">${metric.label}</span>
            <div class="met-bar-wrap">
              <div class="met-bar ${metric.type}"
                style="width:${width}%"></div>
            </div>
            <span class="met-num ${good ? "good" : "bad"}">
              ${percentage}%
            </span>
          </div>
        `;
      }).join("");

      const card = document.createElement("div");
      card.className =
        "model-card" + (isBest ? " winner" : "");

      card.innerHTML = `
        <div class="mc-head">
          <div>
            <div class="mc-name">Block Bootstrap</div>
            <div class="mc-type">
              Sequence-aware time-series baseline
            </div>
          </div>
          <span class="mc-badge ${isBest ? "best" : pass ? "pass" : "fail"}">
            ${isBest ? "Best" : pass ? "Passed" : "Failed"}
          </span>
        </div>
        <div class="mc-div"></div>
        <div class="mc-metrics">${metricsHtml}</div>
        <div class="mc-foot">
          <span class="mc-score-label">Composite safety score</span>
          <span class="mc-score-val">
            ${Number(result.composite_score || 0).toFixed(3)}
          </span>
        </div>
      `;

      grid.appendChild(card);
    });
  }

  function renderTimeSeriesChart(results, bestModel) {
    const group = document.getElementById("chartDots");
    group.innerHTML = "";

    results.forEach(result => {
      const isBest = result.model === bestModel;
      const risk = Number(result.sequence_similarity_risk || 0);
      const utility = Number(result.composite_utility || 0);

      const cx = 48 + (Math.min(risk, 0.3) / 0.3) * 292;
      const cy = 185 - utility * 175;
      const color = isBest ? "#6750A4" : "#7D5260";

      group.innerHTML += `
        <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}"
          r="${isBest ? 13 : 10}" fill="${color}" opacity="0.88"></circle>
        <text x="${(cx + 16).toFixed(1)}" y="${(cy + 4).toFixed(1)}"
          font-size="10" fill="${color}" font-weight="700">
          Block Bootstrap
        </text>
      `;
    });
  }

  function renderTimeSeriesPrivacy(results, bestModel) {
    const best = results.find(
      result => result.model === bestModel
    ) || results[0];

    const risk = Number(best.sequence_similarity_risk || 0);
    const color = risk <= 0.1
      ? "var(--md-success)"
      : risk <= 0.2
        ? "var(--md-warning)"
        : "var(--md-error)";

    document.getElementById("attackRows").innerHTML = `
      <div class="atk-item">
        <div class="atk-icon so">〰</div>
        <div class="atk-info">
          <div class="atk-name">Sequence similarity risk</div>
          <div class="atk-sub">
            Checks whether generated sequence windows are unusually
            close to unseen real control windows.
          </div>
        </div>
        <div style="text-align:right">
          <div class="atk-pct" style="color:${color}">
            ${(risk * 100).toFixed(1)}%
          </div>
          <div class="atk-ci">lower is safer</div>
        </div>
      </div>
    `;
  }

  async function renderTimeSeriesSample(summary) {
    const bestModel = summary.recommended_model;
    const sampleBadge = document.getElementById("sampleBadge");
    const sampleTable = document.getElementById("sampleTable");

    let synthetic = {
      columns: [],
      rows: []
    };

    try {
      const response = await fetch(
        `/api/download/${encodeURIComponent(
          `${bestModel}_time_series_synthetic.csv`
        )}`
      );

      if (response.ok) {
        synthetic = parseCsvText(await response.text(), 4);
      }
    } catch (error) {
      synthetic = { columns: [], rows: [] };
    }

    const original = state.originalPreview;
    const columns = original.columns.filter(
      column => synthetic.columns.includes(column)
    ).slice(0, 4);

    const count = Math.min(
      original.rows.length,
      synthetic.rows.length,
      4
    );

    if (!columns.length || !count) {
      sampleBadge.textContent = "Sample unavailable";
      sampleTable.innerHTML = `
        <thead><tr><th>Preview</th></tr></thead>
        <tbody>
          <tr>
            <td>Time-series sample preview could not be loaded.</td>
          </tr>
        </tbody>
      `;
      return;
    }

    const rows = [];

    for (let rowIndex = 0; rowIndex < count; rowIndex += 1) {
      columns.forEach(column => {
        rows.push({
          row: rowIndex + 1,
          column,
          original: original.rows[rowIndex][column],
          synthetic: synthetic.rows[rowIndex][column]
        });
      });
    }

    sampleBadge.textContent = `${rows.length} sample values`;

    sampleTable.innerHTML = `
      <thead>
        <tr>
          <th>Time step</th>
          <th>Column</th>
          <th>Original value</th>
          <th>Synthetic value</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(row => `
          <tr>
            <td>Step ${row.row}</td>
            <td>${escHtml(row.column)}</td>
            <td>
              <span class="sample-old">
                ${escHtml(String(row.original ?? "(empty)"))}
              </span>
            </td>
            <td>
              <span class="sample-new">
                ${escHtml(String(row.synthetic ?? "(empty)"))}
              </span>
            </td>
            <td>
              <span class="sample-chip changed">Generated</span>
            </td>
          </tr>
        `).join("")}
      </tbody>
    `;
  }

  function renderTimeSeriesTable(results, bestModel) {
    const table = document.getElementById("reportTable");

    table.innerHTML = `
      <thead>
        <tr>
          <th>Model</th>
          <th>Sequence risk</th>
          <th>Trend</th>
          <th>Autocorrelation</th>
          <th>Distribution</th>
          <th>Score</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
        ${results.map(result => {
          const isBest = result.model === bestModel;
          const pass = Boolean(result.overall_pass);

          return `
            <tr>
              <td class="c-model">Block Bootstrap</td>
              <td class="${
                result.sequence_similarity_risk <= 0.1
                  ? "c-good"
                  : "c-bad"
              }">
                ${(result.sequence_similarity_risk * 100).toFixed(1)}%
              </td>
              <td>${(result.trend_similarity * 100).toFixed(1)}%</td>
              <td>
                ${(result.autocorrelation_similarity * 100).toFixed(1)}%
              </td>
              <td>
                ${(result.distribution_similarity * 100).toFixed(1)}%
              </td>
              <td style="font-weight:700">
                ${Number(result.composite_score || 0).toFixed(3)}
              </td>
              <td>
                <span class="pass-chip ${
                  isBest ? "best" : pass ? "yes" : "no"
                }">
                  ${isBest ? "Recommended" : pass ? "Passed" : "Failed"}
                </span>
              </td>
            </tr>
          `;
        }).join("")}
      </tbody>
    `;
  }

  function renderTimeSeriesVerdict(summary) {
    const best = summary.results[0];
    const passed = Boolean(summary.threshold_passed);

    document.getElementById("vBadge").textContent = passed
      ? "Time-series audit passed"
      : "Time-series review required";

    document.getElementById("vTitle").textContent = passed
      ? "Block Bootstrap is cleared for review"
      : "Block Bootstrap needs threshold review";

    document.getElementById("vDesc").textContent = passed
      ? "The generated sequence stayed within the selected privacy limit and retained useful trend, autocorrelation, and distribution behaviour."
      : "The generated sequence did not pass every configured threshold. Review the sequence risk and utility scores before sharing it.";

    document.getElementById("vDownloadBtn").onclick = () => {
      downloadFile(
        `${summary.recommended_model}_time_series_synthetic.csv`
      );
    };

    document.getElementById("vMetrics").innerHTML = [
      {
        name: "Sequence privacy risk",
        value: `${(best.sequence_similarity_risk * 100).toFixed(1)}%`
      },
      {
        name: "Trend similarity",
        value: `${(best.trend_similarity * 100).toFixed(1)}%`
      },
      {
        name: "Autocorrelation",
        value: `${(best.autocorrelation_similarity * 100).toFixed(1)}%`
      },
      {
        name: "Distribution match",
        value: `${(best.distribution_similarity * 100).toFixed(1)}%`
      },
      {
        name: "Composite score",
        value: `${Number(best.composite_score).toFixed(3)} / 1.00`
      }
    ].map(metric => `
      <div class="v-metric">
        <span class="v-met-name">${metric.name}</span>
        <span class="v-met-val">${metric.value}</span>
      </div>
    `).join("");
  }

  window.startAudit = async function startAudit() {
    if (isTimeSeries()) {
      return startTimeSeriesAudit();
    }

    return tabularFunctions.startAudit();
  };

  window.renderResults = async function renderResults(summary) {
    if (summary?.modality !== "time_series") {
      return tabularFunctions.renderResults(summary);
    }

    state.results = summary;

    document.getElementById("resultsSection").classList.add("show");
    document.getElementById("placeholder").style.display = "none";
    document.getElementById("resultsBadge").textContent =
      `${summary.results.length} model selected`;

    renderTimeSeriesCards(
      summary.results,
      summary.recommended_model
    );

    renderTimeSeriesChart(
      summary.results,
      summary.recommended_model
    );

    renderTimeSeriesPrivacy(
      summary.results,
      summary.recommended_model
    );

    await renderTimeSeriesSample(summary);

    renderTimeSeriesTable(
      summary.results,
      summary.recommended_model
    );

    renderTimeSeriesVerdict(summary);
  };

  document.addEventListener("DOMContentLoaded", () => {
    addPhase2Styles();
    addModalityPicker();
    selectModality("tabular");
  });

  // Keep background tabular refreshes from overwriting time-series settings.
  window.renderSliders = function renderSliders() {
    if (isTimeSeries()) {
      return renderTimeSeriesSliders();
    }

    return tabularFunctions.renderSliders();
  };

  window.renderModelToggles = function renderModelToggles() {
    if (isTimeSeries()) {
      return renderTimeSeriesModels();
    }

    return tabularFunctions.renderModelToggles();
  };

  window.renderRunModes = function renderRunModes() {
    if (isTimeSeries()) {
      return renderTimeSeriesRunModes();
    }

    return tabularFunctions.renderRunModes();
  };

  window.updateRunDisplay = function updateRunDisplay() {
    if (isTimeSeries()) {
      return updateTimeSeriesRunDisplay();
    }

    return tabularFunctions.updateRunDisplay();
  };

  window.updateModelCount = function updateModelCount() {
    if (isTimeSeries()) {
      document.getElementById("cfgModelCount").textContent =
        "1 model selected";

      return updateTimeSeriesRunDisplay();
    }

    return tabularFunctions.updateModelCount();
  };

  window.setRunMode = function setRunMode(key) {
    if (isTimeSeries()) {
      state.timeSeriesRunMode = key;
      renderTimeSeriesRunModes();
      updateTimeSeriesRunDisplay();
      return;
    }

    state.runMode = key;
    tabularFunctions.renderRunModes();
  };
})();
