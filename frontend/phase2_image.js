(() => {
  "use strict";

  const imageState = {
    originalFile: null,
    syntheticFile: null,
    uploaded: false,
    originalFolder: null,
    syntheticFolder: null,
    originalImages: 0,
    syntheticImages: 0,
    logSource: null,
  };

  const css = `
    #imageModeControls {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 0 0 18px;
      padding: 12px 14px;
      border-radius: 16px;
      background: var(--md-surf-cont);
      border: 1px solid rgba(121,116,126,.12);
    }

    #imageModeControls span {
      font-size: .84rem;
      font-weight: 500;
      color: var(--md-on-surf-var);
      margin-right: auto;
    }

    .image-mode-btn {
      border: 1px solid rgba(103,80,164,.3);
      background: transparent;
      color: var(--md-primary);
      border-radius: 999px;
      padding: 8px 14px;
      font: inherit;
      font-size: .8rem;
      font-weight: 500;
      cursor: pointer;
    }

    .image-mode-btn.active {
      background: var(--md-primary);
      color: white;
      border-color: var(--md-primary);
    }

    #imageUploadZone {
      display: none;
      margin-bottom: 28px;
    }

    .image-upload-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }

    .image-upload-card {
      padding: 22px;
      border-radius: 18px;
      border: 2px dashed rgba(103,80,164,.3);
      text-align: center;
      background: rgba(255,255,255,.46);
    }

    .image-upload-card.ready {
      border-color: var(--md-success);
      background: rgba(56,106,32,.06);
    }

    .image-upload-card input {
      display: none;
    }

    .image-upload-card label {
      display: block;
      cursor: pointer;
    }

    .image-upload-icon {
      font-size: 28px;
      margin-bottom: 8px;
    }

    .image-upload-title {
      color: var(--md-primary);
      font-size: .95rem;
      font-weight: 700;
    }

    .image-upload-help {
      color: var(--md-on-surf-var);
      font-size: .78rem;
      line-height: 1.5;
      margin-top: 6px;
    }

    .image-file-name {
      color: var(--md-success);
      font-size: .78rem;
      font-weight: 500;
      margin-top: 12px;
      overflow-wrap: anywhere;
    }

    #imageUploadAction {
      margin-top: 18px;
    }

    #imageResults {
      display: none;
      margin: 0 0 28px;
    }

    .image-result-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-top: 18px;
    }

    .image-result-stat {
      background: var(--md-surf-cont);
      border-radius: 16px;
      padding: 16px;
      border: 1px solid rgba(121,116,126,.1);
    }

    .image-result-label {
      color: var(--md-on-surf-var);
      font-size: .75rem;
      font-weight: 500;
    }

    .image-result-value {
      color: var(--md-primary);
      font-size: 1.45rem;
      font-weight: 700;
      margin-top: 6px;
    }

    .image-verdict {
      margin-top: 18px;
      padding: 20px;
      border-radius: 18px;
      color: white;
      background: var(--md-primary);
    }

    .image-verdict.failed {
      background: var(--md-error);
    }

    .image-verdict-title {
      font-size: 1.08rem;
      font-weight: 700;
    }

    .image-verdict-text {
      font-size: .86rem;
      line-height: 1.55;
      margin-top: 7px;
      opacity: .92;
    }

    @media (max-width: 700px) {
      .image-upload-grid,
      .image-result-grid {
        grid-template-columns: 1fr;
      }

      #imageModeControls {
        flex-wrap: wrap;
      }
    }
  `;

  function addStyles() {
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function percentage(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "N/A";
  }

  function numberValue(summary, keys, fallback = 0) {
    for (const key of keys) {
      if (summary[key] !== undefined && summary[key] !== null) {
        return Number(summary[key]);
      }
    }
    return fallback;
  }

  function setImageMode(enabled) {
    const csvUpload = byId("uploadZone");
    const imageUpload = byId("imageUploadZone");
    const imageButton = byId("enableImageMode");
    const csvButton = byId("enableCsvMode");

    if (enabled) {
      state.modality = "image";

      if (csvUpload) csvUpload.style.display = "none";
      if (imageUpload) imageUpload.style.display = "block";

      imageButton.classList.add("active");
      csvButton.classList.remove("active");

      const heading = document.querySelector(".sec-head .sec-title");
      const subtitle = document.querySelector(".sec-head .sec-sub");

      if (heading) heading.textContent = "Step 1 - Upload image datasets";
      if (subtitle) {
        subtitle.textContent =
          "Upload one ZIP of original images and one ZIP of synthetic images. Both datasets stay on your machine.";
      }

      if (byId("resultsSection")) {
        byId("resultsSection").classList.remove("show");
      }

      return;
    }

    state.modality = "tabular";

    if (csvUpload) csvUpload.style.display = "";
    if (imageUpload) imageUpload.style.display = "none";

    imageButton.classList.remove("active");
    csvButton.classList.add("active");

    const heading = document.querySelector(".sec-head .sec-title");
    const subtitle = document.querySelector(".sec-head .sec-sub");

    if (heading) heading.textContent = "Step 1 - Upload your dataset";
    if (subtitle) {
      subtitle.textContent =
        "Drop a CSV file here. The tool reads its columns and rows locally before running the audit.";
    }
  }

  function updateImageFileCard(kind, file) {
    const card = byId(`${kind}ImageCard`);
    const label = byId(`${kind}ImageFileName`);

    if (!file) {
      card.classList.remove("ready");
      label.textContent = "";
      return;
    }

    card.classList.add("ready");
    label.textContent = `Selected: ${file.name}`;
  }

  async function uploadImageArchives() {
    if (!imageState.originalFile || !imageState.syntheticFile) {
      alert("Choose both the original-image ZIP and synthetic-image ZIP first.");
      return;
    }

    const uploadButton = byId("imageUploadAction");
    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading ZIP files...";

    try {
      const formData = new FormData();
      formData.append("original_zip", imageState.originalFile);
      formData.append("synthetic_zip", imageState.syntheticFile);

      const response = await fetch("/api/upload-images", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || "Image upload failed.");
      }

      imageState.uploaded = true;
      imageState.originalFolder = data.original_folder;
      imageState.syntheticFolder = data.synthetic_folder;
      imageState.originalImages = data.original_images;
      imageState.syntheticImages = data.synthetic_images;

      uploadButton.textContent =
        `Ready: ${data.original_images} original and ${data.synthetic_images} synthetic images`;

      if (typeof setStatus === "function") {
        setStatus("Ready to run", "done");
      }

      if (typeof addLog === "function") {
        addLog(
          "info",
          `Image ZIP files uploaded locally: ${data.original_images} original and ${data.synthetic_images} synthetic images.`
        );
      }
    } catch (error) {
      uploadButton.disabled = false;
      uploadButton.textContent = "Upload image ZIP files";
      alert(error.message);

      if (typeof setStatus === "function") {
        setStatus("Upload failed", "error");
      }
    }
  }

  function renderImageResults(summary) {
    const results = byId("imageResults");
    results.style.display = "block";

    const risk = numberValue(
      summary,
      ["near_duplicate_risk", "privacy_risk", "risk_score"]
    );

    const utility = numberValue(
      summary,
      ["composite_utility", "utility_score"]
    );

    const score = numberValue(
      summary,
      ["composite_score", "score"]
    );

    const matches = numberValue(
      summary,
      ["near_duplicate_matches", "matches"]
    );

    const passed = Boolean(
      summary.threshold_passed ?? summary.overall_pass ?? false
    );

    byId("imageResultsBody").innerHTML = `
      <div class="image-result-grid">
        <div class="image-result-stat">
          <div class="image-result-label">Original images</div>
          <div class="image-result-value">${summary.original_images ?? imageState.originalImages}</div>
        </div>
        <div class="image-result-stat">
          <div class="image-result-label">Synthetic images</div>
          <div class="image-result-value">${summary.synthetic_images ?? imageState.syntheticImages}</div>
        </div>
        <div class="image-result-stat">
          <div class="image-result-label">Near-duplicate risk</div>
          <div class="image-result-value">${percentage(risk)}</div>
        </div>
        <div class="image-result-stat">
          <div class="image-result-label">Composite utility</div>
          <div class="image-result-value">${percentage(utility)}</div>
        </div>
      </div>

      <div class="image-verdict ${passed ? "" : "failed"}">
        <div class="image-verdict-title">
          ${passed ? "Image dataset passed the configured audit" : "Image dataset needs privacy review"}
        </div>
        <div class="image-verdict-text">
          ${
            passed
              ? "The synthetic image set stayed within the configured near-duplicate privacy limit while preserving useful visual properties."
              : `The audit found ${matches} near-duplicate image match${matches === 1 ? "" : "es"}. High visual similarity can be useful, but near-duplicate images may expose original training images.`
          }
        </div>
      </div>
    `;
  }

  function closeImageLogStream() {
    if (imageState.logSource) {
      imageState.logSource.close();
      imageState.logSource = null;
    }
  }

  function runImageAudit() {
    if (!imageState.uploaded) {
      alert("Upload both image ZIP files before starting the image audit.");
      return;
    }

    if (state.running) {
      alert("An audit is already running.");
      return;
    }

    state.running = true;

    const runButton = byId("btnRun");
    runButton.disabled = true;
    runButton.textContent = "Running image audit...";

    if (typeof clearLogs === "function") clearLogs();
    if (typeof setStatus === "function") setStatus("Running...", "running");

    closeImageLogStream();

    imageState.logSource = new EventSource("/api/logs");

    imageState.logSource.onmessage = (event) => {
      let item;

      try {
        item = JSON.parse(event.data);
      } catch {
        return;
      }

      if (!item || !item.level) return;

      if (typeof addLog === "function") {
        addLog(item.level, item.message, item.ts);
      }

      if (item.level === "done") {
        closeImageLogStream();

        try {
          renderImageResults(JSON.parse(item.message));
        } catch (error) {
          if (typeof addLog === "function") {
            addLog("error", `Could not display image results: ${error.message}`);
          }
        }

        state.running = false;
        runButton.disabled = false;
        runButton.innerHTML = "&#9654; &nbsp;Run Audit";

        if (typeof setStatus === "function") {
          setStatus("Audit complete", "done");
        }
      }
    };

    imageState.logSource.onerror = () => {
      if (state.running && typeof addLog === "function") {
        addLog("warn", "Image log stream disconnected.");
      }
    };

    fetch("/api/run-image-audit", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        modality: "image",
        original_folder: imageState.originalFolder,
        synthetic_folder: imageState.syntheticFolder,
        privacy_thresholds: {
          near_duplicate: 0.10,
        },
        utility_thresholds: {
          composite_utility: 0.70,
        },
      }),
    })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok || data.error) {
          throw new Error(data.error || "Could not start image audit.");
        }
      })
      .catch((error) => {
        closeImageLogStream();
        state.running = false;
        runButton.disabled = false;
        runButton.innerHTML = "&#9654; &nbsp;Run Audit";

        if (typeof setStatus === "function") {
          setStatus("Error", "error");
        }

        if (typeof addLog === "function") {
          addLog("error", error.message);
        }
      });
  }

  function createInterface() {
    const page = document.querySelector(".page");
    const csvUploadZone = byId("uploadZone");

    if (!page || !csvUploadZone) return;

    const controls = document.createElement("div");
    controls.id = "imageModeControls";
    controls.innerHTML = `
      <span>Audit type</span>
      <button class="image-mode-btn active" id="enableCsvMode" type="button">CSV / Time Series</button>
      <button class="image-mode-btn" id="enableImageMode" type="button">Image Audit</button>
    `;

    csvUploadZone.parentNode.insertBefore(controls, csvUploadZone);

    const imageUpload = document.createElement("section");
    imageUpload.id = "imageUploadZone";
    imageUpload.className = "upload-zone";
    imageUpload.innerHTML = `
      <div class="blur-orb orb-a" aria-hidden="true"></div>
      <div class="blur-orb orb-b" aria-hidden="true"></div>

      <div class="upload-inner" style="display:block;">
        <div class="image-upload-grid">
          <div class="image-upload-card" id="originalImageCard">
            <label for="originalImageZip">
              <div class="image-upload-icon">Original</div>
              <div class="image-upload-title">Upload original images ZIP</div>
              <div class="image-upload-help">
                ZIP the folder containing the real images you want to protect.
              </div>
            </label>
            <input id="originalImageZip" type="file" accept=".zip,application/zip">
            <div class="image-file-name" id="originalImageFileName"></div>
          </div>

          <div class="image-upload-card" id="syntheticImageCard">
            <label for="syntheticImageZip">
              <div class="image-upload-icon">Synthetic</div>
              <div class="image-upload-title">Upload synthetic images ZIP</div>
              <div class="image-upload-help">
                ZIP the synthetic images generated by another model or workflow.
              </div>
            </label>
            <input id="syntheticImageZip" type="file" accept=".zip,application/zip">
            <div class="image-file-name" id="syntheticImageFileName"></div>
          </div>
        </div>

        <button class="btn-run" id="imageUploadAction" type="button">
          Upload image ZIP files
        </button>
      </div>
    `;

    csvUploadZone.parentNode.insertBefore(imageUpload, csvUploadZone.nextSibling);

    const results = document.createElement("section");
    results.id = "imageResults";
    results.innerHTML = `
      <div class="sec-head">
        <div>
          <div class="sec-title">Image audit results</div>
          <div class="sec-sub">
            The audit compares overall visual properties and checks for images that are too close to originals.
          </div>
        </div>
        <span class="badge">Local-only audit</span>
      </div>
      <div id="imageResultsBody"></div>
    `;

    const logCard = document.querySelector(".log-card");
    if (logCard) {
      logCard.parentNode.insertBefore(results, logCard);
    } else {
      page.appendChild(results);
    }

    byId("enableImageMode").addEventListener("click", () => setImageMode(true));
    byId("enableCsvMode").addEventListener("click", () => setImageMode(false));

    byId("originalImageZip").addEventListener("change", (event) => {
      imageState.originalFile = event.target.files[0] || null;
      imageState.uploaded = false;
      updateImageFileCard("original", imageState.originalFile);
    });

    byId("syntheticImageZip").addEventListener("change", (event) => {
      imageState.syntheticFile = event.target.files[0] || null;
      imageState.uploaded = false;
      updateImageFileCard("synthetic", imageState.syntheticFile);
    });

    byId("imageUploadAction").addEventListener("click", uploadImageArchives);
  }

  function overrideRunButton() {
    const originalStartAudit = window.startAudit;

    window.startAudit = function () {
      if (state.modality === "image") {
        runImageAudit();
        return;
      }

      originalStartAudit();
    };
  }

  addStyles();
  createInterface();
  overrideRunButton();
})();