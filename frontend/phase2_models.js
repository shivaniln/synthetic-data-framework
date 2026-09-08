(() => {
  "use strict";

  const previousStartAudit = window.startAudit;
  const previousRenderResults = window.renderResults;

  const TIME_SERIES_SETTINGS = {
    quick: {
      rows: 240,
      blockSize: 12,
      queries: 50
    },
    standard: {
      rows: 2000,
      blockSize: 24,
      queries: 100
    },
    full: {
      rows: 8000,
      blockSize: 48,
      queries: 200
    }
  };

  function isTimeSeries() {
    return state.modality === "time_series";
  }

  function displayModelName(modelName) {
    const names = {
      block_bootstrap: "Block Bootstrap",
      fourier_surrogate: "Fourier Surrogate"
    };

    return names[modelName] || modelName;
  }

  function displayModelDescription(modelName) {
    const descriptions = {
      block_bootstrap:
        "Preserves short local patterns through contiguous sequence blocks",
      fourier_surrogate:
        "Preserves dominant periodic and frequency patterns"
    };

    return descriptions[modelName] || "";
  }

  function renderTwoTimeSeriesModels() {
    if (!isTimeSeries()) {
      return;
    }

    const container = document.getElementById("modelToggles");

    if (!container) {
      return;
    }

    container.innerHTML = `
      <div class="toggle-row">
        <div>
          <div class="toggle-label-text">Block Bootstrap</div>
          <div class="toggle-sub-text">
            Preserves short local patterns through contiguous sequence blocks
          </div>
        </div>
        <div class="toggle-switch on disabled"></div>
      </div>

      <div class="toggle-row">
        <div>
          <div class="toggle-label-text">Fourier Surrogate</div>
          <div class="toggle-sub-text">
            Preserves dominant periodic and frequency patterns
          </div>
        </div>
        <div class="toggle-switch on disabled"></div>
      </div>

      <div class="time-series-note">
        Both generators use the same chronological split and are compared
        using sequence privacy risk, trend similarity, autocorrelation,
        and distribution similarity.
      </div>
    `;

    document.getElementById("cfgModelCount").textContent =
      "2 models selected";
  }

  async function startTimeSeriesComparison() {
    if (!state.filename) {
      alert("Please upload a timestamped CSV file first.");
      return;
    }

    if (state.running) {
      alert("An audit is already running.");
      return;
    }

    const settings =
      TIME_SERIES_SETTINGS[state.timeSeriesRunMode] ||
      TIME_SERIES_SETTINGS.standard;

    const maxRows = state.rowCount
      ? Math.min(state.rowCount, settings.rows)
      : settings.rows;

    const sequenceRiskLimit =
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
          window_size: Math.max(
            6,
            Math.floor(settings.blockSize / 2)
          ),
          max_queries: settings.queries,
          random_seed: 42
        },
        fourier_surrogate: {
          harmonics: settings.blockSize,
          noise_scale: 0.03,
          window_size: Math.max(
            6,
            Math.floor(settings.blockSize / 2)
          ),
          max_queries: settings.queries,
          random_seed: 42
        }
      },
      privacy_thresholds: {
        sequence_similarity: sequenceRiskLimit
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
    startProgressSim(2);
    openLogStream();

    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
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

  function updateTimeSeriesLabels(summary) {
    const results = summary.results || [];
    const cards = document.querySelectorAll(".model-card");

    cards.forEach((card, index) => {
      const result = results[index];

      if (!result) {
        return;
      }

      const name = card.querySelector(".mc-name");
      const description = card.querySelector(".mc-type");

      if (name) {
        name.textContent = displayModelName(result.model);
      }

      if (description) {
        description.textContent = displayModelDescription(result.model);
      }
    });

    const tableRows = document.querySelectorAll(
      "#reportTable tbody tr"
    );

    tableRows.forEach((row, index) => {
      const result = results[index];

      if (!result) {
        return;
      }

      const modelCell = row.querySelector(".c-model");

      if (modelCell) {
        modelCell.textContent = displayModelName(result.model);
      }
    });

    const chartTexts = document.querySelectorAll("#chartDots text");

    chartTexts.forEach((text, index) => {
      const result = results[index];

      if (result) {
        text.textContent = displayModelName(result.model);
      }
    });

    const bestModel = displayModelName(
      summary.recommended_model
    );

    const verdictTitle = document.getElementById("vTitle");

    if (verdictTitle) {
      verdictTitle.textContent = summary.threshold_passed
        ? `${bestModel} is cleared for review`
        : `${bestModel} is the best available option`;
    }
  }

  window.startAudit = async function startAudit() {
    if (isTimeSeries()) {
      return startTimeSeriesComparison();
    }

    return previousStartAudit();
  };

  window.renderResults = async function renderResults(summary) {
    await previousRenderResults(summary);

    if (summary?.modality === "time_series") {
      updateTimeSeriesLabels(summary);
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    const picker = document.getElementById("modalityPicker");

    if (picker) {
      picker.addEventListener(
        "click",
        event => {
          const button = event.target.closest("[data-modality]");

          if (button?.dataset.modality === "time_series") {
            setTimeout(renderTwoTimeSeriesModels, 0);
          }
        },
        true
      );
    }

    const modelContainer = document.getElementById("modelToggles");

    if (modelContainer) {
      const observer = new MutationObserver(() => {
        if (
          isTimeSeries() &&
          !modelContainer.dataset.updatingTimeSeriesModels
        ) {
          modelContainer.dataset.updatingTimeSeriesModels = "true";

          setTimeout(() => {
            renderTwoTimeSeriesModels();
            delete modelContainer.dataset.updatingTimeSeriesModels;
          }, 0);
        }
      });

      observer.observe(modelContainer, {
        childList: true,
        subtree: true
      });
    }

    setTimeout(renderTwoTimeSeriesModels, 100);
  });
})();