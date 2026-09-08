(() => {
  "use strict";

  const previousHandleFileSelect = window.handleFileSelect;
  const previousRenderResults = window.renderResults;

  function parseFullCsv(text) {
    const parsed = parseCsvText(text, 10000);

    return {
      columns: parsed.columns || [],
      rows: parsed.rows || []
    };
  }

  window.handleFileSelect = async function handleFileSelect(file) {
    if (
      state.modality === "time_series" &&
      file &&
      file.name.toLowerCase().endsWith(".csv")
    ) {
      try {
        state.fullTimeSeriesOriginal = parseFullCsv(
          await file.text()
        );
      } catch (error) {
        state.fullTimeSeriesOriginal = {
          columns: [],
          rows: []
        };
      }
    }

    return previousHandleFileSelect(file);
  };

  async function loadSyntheticSeries(modelName) {
    const response = await fetch(
      `/api/download/${encodeURIComponent(
        `${modelName}_time_series_synthetic.csv`
      )}`
    );

    if (!response.ok) {
      throw new Error("Could not load synthetic time-series output.");
    }

    return parseFullCsv(await response.text());
  }

  function numericValues(rows, column) {
    return rows
      .map(row => Number(row[column]))
      .filter(value => Number.isFinite(value));
  }

  function downsample(values, limit = 100) {
    if (values.length <= limit) {
      return values;
    }

    const output = [];

    for (let index = 0; index < limit; index += 1) {
      const sourceIndex = Math.round(
        (index / (limit - 1)) * (values.length - 1)
      );

      output.push(values[sourceIndex]);
    }

    return output;
  }

  function buildPath(values, width, height, minValue, maxValue) {
    if (!values.length) {
      return "";
    }

    const range = Math.max(maxValue - minValue, 1e-8);

    return values.map((value, index) => {
      const x = 42 + (
        index / Math.max(values.length - 1, 1)
      ) * (width - 58);

      const y = 18 + (
        1 - ((value - minValue) / range)
      ) * (height - 54);

      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  }

  function renderSequenceChart(column) {
    const chartBody = document.getElementById(
      "timeSeriesChartBody"
    );

    const original = state.fullTimeSeriesOriginal;
    const synthetic = state.fullTimeSeriesSynthetic;

    if (!chartBody || !original || !synthetic) {
      return;
    }

    const originalValues = downsample(
      numericValues(original.rows, column)
    );

    const syntheticValues = downsample(
      numericValues(synthetic.rows, column)
    );

    const count = Math.min(
      originalValues.length,
      syntheticValues.length
    );

    if (count < 2) {
      chartBody.innerHTML =
        "<p style='font-size:.82rem;color:var(--md-on-surf-var)'>Not enough numerical values are available for this chart.</p>";
      return;
    }

    const real = originalValues.slice(0, count);
    const generated = syntheticValues.slice(0, count);

    const allValues = [...real, ...generated];
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);

    const width = 720;
    const height = 250;

    const realPath = buildPath(
      real,
      width,
      height,
      minValue,
      maxValue
    );

    const syntheticPath = buildPath(
      generated,
      width,
      height,
      minValue,
      maxValue
    );

    chartBody.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}"
        style="width:100%;height:260px;display:block">
        <line x1="42" y1="18" x2="42" y2="${height - 36}"
          stroke="#d7dfea"></line>
        <line x1="42" y1="${height - 36}"
          x2="${width - 16}" y2="${height - 36}"
          stroke="#d7dfea"></line>

        <line x1="42" y1="72" x2="${width - 16}" y2="72"
          stroke="#e7eef8" stroke-dasharray="4 4"></line>
        <line x1="42" y1="126" x2="${width - 16}" y2="126"
          stroke="#e7eef8" stroke-dasharray="4 4"></line>
        <line x1="42" y1="180" x2="${width - 16}" y2="180"
          stroke="#e7eef8" stroke-dasharray="4 4"></line>

        <path d="${realPath}"
          fill="none"
          stroke="#102346"
          stroke-width="2.4"
          stroke-linejoin="round"
          stroke-linecap="round"></path>

        <path d="${syntheticPath}"
          fill="none"
          stroke="#6750A4"
          stroke-width="2.4"
          stroke-linejoin="round"
          stroke-linecap="round"></path>

        <text x="42" y="${height - 14}"
          font-size="10" fill="#7b8190">
          Earlier observations
        </text>

        <text x="${width - 16}" y="${height - 14}"
          font-size="10" fill="#7b8190"
          text-anchor="end">
          Later observations
        </text>

        <text x="34" y="24"
          font-size="10" fill="#7b8190"
          text-anchor="end">
          ${maxValue.toFixed(1)}
        </text>

        <text x="34" y="${height - 40}"
          font-size="10" fill="#7b8190"
          text-anchor="end">
          ${minValue.toFixed(1)}
        </text>
      </svg>

      <div class="chart-legend">
        <div class="leg-item">
          <div class="leg-dot" style="background:#102346"></div>
          Original sequence
        </div>
        <div class="leg-item">
          <div class="leg-dot" style="background:#6750A4"></div>
          Recommended synthetic sequence
        </div>
      </div>
    `;
  }

  async function renderTimeSeriesChart(summary) {
    const chartCard = document.querySelector(".chart-card");

    if (!chartCard) {
      return;
    }

    try {
      state.fullTimeSeriesSynthetic =
        await loadSyntheticSeries(
          summary.recommended_model
        );
    } catch (error) {
      chartCard.innerHTML = `
        <div class="sec-title" style="font-size:1rem">
          Original vs synthetic sequence
        </div>
        <p style="margin-top:12px;font-size:.82rem;color:var(--md-on-surf-var)">
          The chart could not load the generated time-series CSV.
        </p>
      `;
      return;
    }

    const availableColumns = (
      summary.dataset_details?.value_columns || []
    ).filter(column =>
      state.fullTimeSeriesOriginal?.columns?.includes(column) &&
      state.fullTimeSeriesSynthetic?.columns?.includes(column)
    );

    if (!availableColumns.length) {
      return;
    }

    const defaultColumn = availableColumns.includes("Appliances")
      ? "Appliances"
      : availableColumns[0];

    chartCard.innerHTML = `
      <div class="sec-head" style="margin-bottom:0">
        <div>
          <div class="sec-title" style="font-size:1rem">
            Original vs synthetic sequence
          </div>
          <div class="sec-sub" style="font-size:.75rem">
            The recommended generator is compared with the uploaded
            time-series values. Similar overall movement is useful,
            while exact copying would increase privacy concern.
          </div>
        </div>

        <select id="timeSeriesColumnSelect"
          style="
            border:1px solid rgba(121,116,126,.28);
            border-radius:10px;
            background:#fff;
            color:var(--md-on-bg);
            padding:8px 10px;
            font:inherit;
            font-size:.78rem;
          ">
          ${availableColumns.map(column => `
            <option value="${column}">
              ${column}
            </option>
          `).join("")}
        </select>
      </div>

      <div id="timeSeriesChartBody"
        style="margin-top:14px"></div>
    `;

    const selector = document.getElementById(
      "timeSeriesColumnSelect"
    );

    selector.value = defaultColumn;

    selector.addEventListener("change", event => {
      renderSequenceChart(event.target.value);
    });

    renderSequenceChart(defaultColumn);
  }

  window.renderResults = async function renderResults(summary) {
    await previousRenderResults(summary);

    if (summary?.modality === "time_series") {
      await renderTimeSeriesChart(summary);
    }
  };
})();