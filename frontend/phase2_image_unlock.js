(() => {
  "use strict";

  let imageModeActive = false;
  let imageAuditRunning = false;
  let imageLogSource = null;

  const style = document.createElement("style");
  style.textContent = `
    #imageAuditWorkspace {
      display: none;
      margin-bottom: 28px;
    }

    .midst-image-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }

    .midst-image-box {
      background: rgba(255,255,255,.55);
      border: 2px dashed rgba(103,80,164,.32);
      border-radius: 18px;
      padding: 25px 20px;
      text-align: center;
    }

    .midst-image-box.ready {
      border-color: var(--md-success);
      background: rgba(56,106,32,.06);
    }

    .midst-image-box input {
      display: none;
    }

    .midst-image-box label {
      cursor: pointer;
      display: block;
    }

    .midst-image-box h3 {
      color: var(--md-primary);
      font-size: .95rem;
      margin-bottom: 6px;
    }

    .midst-image-box p {
      color: var(--md-on-surf-var);
      font-size: .78rem;
      line-height: 1.5;
    }

    .midst-image-file {
      color: var(--md-success);
      font-size: .78rem;
      font-weight: 500;
      margin-top: 12px;
      overflow-wrap: anywhere;
    }

    #imageAuditUpload {
      margin-top: 18px;
    }

    #imageAuditResults {
      display: none;
      margin-bottom: 28px;
    }

    .midst-image-results-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-top: 16px;
    }

    .midst-image-stat {
      background: var(--md-surf-cont);
      border-radius: 16px;
      padding: 16px;
      border: 1px solid rgba(121,116,126,.12);
    }

    .midst-image-stat small {
      display: block;
      color: var(--md-on-surf-var);
      font-size: .74rem;
    }

    .midst-image-stat strong {
      display: block;
      color: var(--md-primary);
      font-size: 1.35rem;
      margin-top: 7px;
    }

    .midst-image-verdict {
      margin-top: 16px;
      border-radius: 18px;
      padding: 20px;
      color: white;
      background: var(--md-primary);
    }

    .midst-image-verdict.failed {
      background: var(--md-error);
    }

    .midst-image-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }

    .midst-image-actions a {
      display: inline-block;
      padding: 9px 15px;
      border-radius: 999px;
      background: rgba(103,80,164,.10);
      color: var(--md-primary);
      font-size: .8rem;
      font-weight: 500;
      text-decoration: none;
    }

    .midst-image-preview-title {
      margin-top: 24px;
      font-size: .95rem;
      font-weight: 700;
    }

    .midst-image-previews {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 10px;
      margin-top: 10px;
    }

    .midst-image-preview {
      overflow: hidden;
      border-radius: 10px;
      background: var(--md-surf-cont);
      border: 1px solid rgba(121,116,126,.14);
    }

    .midst-image-preview img {
      display: block;
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
    }

    .midst-image-preview small {
      display: block;
      padding: 5px;
      color: var(--md-on-surf-var);
      font-size: .65rem;
    }

    @media (max-width: 700px) {
      .midst-image-grid,
      .midst-image-results-grid {
        grid-template-columns: 1fr;
      }
    }
  `;
  document.head.appendChild(style);

  function setStatusSafe(text, status) {
    if (typeof window.setStatus === "function") {
      window.setStatus(text, status);
    }
  }

  function addLogSafe(level, message, timestamp) {
    if (typeof window.addLog === "function") {
      window.addLog(level, message, timestamp);
    }
  }

  function findImageTile() {
    const namedTile = document.getElementById("imageModalityButton");
    if (namedTile) return namedTile;

    const candidates = [...document.querySelectorAll("button, div")].filter((element) => {
      const text = element.textContent.replace(/\s+/g, " ").trim();
      return text.includes("Image data") && text.length < 80;
    });

    return candidates.sort(
      (first, second) => first.textContent.length - second.textContent.length
    )[0];
  }

  function replaceTextInside(element, oldText, newText) {
    for (const child of element.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        child.nodeValue = child.nodeValue.replace(oldText, newText);
      } else {
        replaceTextInside(child, oldText, newText);
      }
    }
  }

  function createImageWorkspace() {
    const uploadZone = document.getElementById("uploadZone");
    const page = document.querySelector(".page");

    if (!uploadZone || !page) return null;

    const workspace = document.createElement("section");
    workspace.id = "imageAuditWorkspace";
    workspace.className = "upload-zone";

    workspace.innerHTML = `
      <div class="blur-orb orb-a" aria-hidden="true"></div>
      <div class="blur-orb orb-b" aria-hidden="true"></div>

      <div class="upload-inner" style="display:block; position:relative; z-index:1;">
        <div class="midst-image-grid">
          <div class="midst-image-box" id="midstOriginalBox">
            <label for="midstOriginalZip">
              <div class="upload-icon">Original</div>
              <h3>Original images ZIP</h3>
              <p>Choose a ZIP file containing the real images that need privacy protection.</p>
            </label>
            <input id="midstOriginalZip" type="file" accept=".zip,application/zip">
            <div class="midst-image-file" id="midstOriginalName"></div>
          </div>

          <div class="midst-image-box" id="midstSyntheticBox">
            <label for="midstSyntheticZip">
              <div class="upload-icon">Synthetic</div>
              <h3>Synthetic images ZIP</h3>
              <p>Choose a ZIP of generated images, or leave this empty to generate PCA images locally.</p>
            </label>
            <input id="midstSyntheticZip" type="file" accept=".zip,application/zip">
            <div class="midst-image-file" id="midstSyntheticName"></div>
          </div>
        </div>

        <button class="btn-run" type="button" id="imageAuditUpload">
          Upload ZIPs / generate synthetic images
        </button>
      </div>
    `;

    uploadZone.parentNode.insertBefore(workspace, uploadZone.nextSibling);

    const results = document.createElement("section");
    results.id = "imageAuditResults";
    results.innerHTML = `
      <div class="sec-head">
        <div>
          <div class="sec-title">Image audit results</div>
          <div class="sec-sub">
            The audit checks visual similarity and near-duplicate privacy risk.
          </div>
        </div>
        <span class="badge">Local image audit</span>
      </div>
      <div id="imageAuditResultsBody"></div>
    `;

    const logCard = document.querySelector(".log-card");
    if (logCard) {
      logCard.parentNode.insertBefore(results, logCard);
    } else {
      page.appendChild(results);
    }

    return workspace;
  }

  const imageWorkspace = createImageWorkspace();
  if (!imageWorkspace) return;

  const originalInput = document.getElementById("midstOriginalZip");
  const syntheticInput = document.getElementById("midstSyntheticZip");
  const originalBox = document.getElementById("midstOriginalBox");
  const syntheticBox = document.getElementById("midstSyntheticBox");
  const originalName = document.getElementById("midstOriginalName");
  const syntheticName = document.getElementById("midstSyntheticName");

  originalInput.addEventListener("change", () => {
    const file = originalInput.files[0];
    originalBox.classList.toggle("ready", Boolean(file));
    originalName.textContent = file ? `Selected: ${file.name}` : "";
  });

  syntheticInput.addEventListener("change", () => {
    const file = syntheticInput.files[0];
    syntheticBox.classList.toggle("ready", Boolean(file));
    syntheticName.textContent = file ? `Selected: ${file.name}` : "";
  });

  function activateImageMode() {
    imageModeActive = true;

    document.getElementById("uploadZone").style.display = "none";
    imageWorkspace.style.display = "block";

    // Do not show results from a previous tabular or time-series run.
    const sharedResults = document.getElementById("resultsSection");
    if (sharedResults) sharedResults.classList.remove("show");

    const placeholder = document.getElementById("placeholder");
    if (placeholder) placeholder.style.display = "none";

    const stepTitle = document.querySelector(".sec-head .sec-title");
    const stepSubtitle = document.querySelector(".sec-head .sec-sub");

    if (stepTitle) stepTitle.textContent = "Step 1 - Upload image datasets";
    if (stepSubtitle) {
      stepSubtitle.textContent =
        "Upload original and synthetic image folders as ZIP files. MIDST processes both datasets locally.";
    }
  }

  async function uploadAndRunImageAudit() {
    const originalFile = originalInput.files[0];
    const syntheticFile = syntheticInput.files[0];
    const runButton = document.getElementById("btnRun");

    if (!originalFile) {
      alert("Choose the original-images ZIP first.");
      return;
    }

    if (imageAuditRunning) {
      alert("An image audit is already running.");
      return;
    }

    imageAuditRunning = true;
    runButton.disabled = true;
    runButton.textContent = "Uploading images...";
    setStatusSafe("Uploading...", "");

    try {
      const formData = new FormData();
      formData.append("original_zip", originalFile);
      if (syntheticFile) formData.append("synthetic_zip", syntheticFile);

      const endpoint = syntheticFile
        ? "/api/upload-images"
        : "/api/generate-image-audit";

      const uploadResponse = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const upload = await uploadResponse.json();

      if (!uploadResponse.ok || upload.error) {
          throw new Error(
            upload.error ||
            (syntheticFile
              ? "Image ZIP upload failed."
              : "Could not generate synthetic images.")
          );
      }

      runButton.textContent = "Running image audit...";
      setStatusSafe("Running...", "running");

      if (typeof window.clearLogs === "function") {
        window.clearLogs();
      }

      addLogSafe(
        "info",
        syntheticFile
          ? `Uploaded ${upload.original_images} original and ${upload.synthetic_images} synthetic images locally.`
          : `Generated ${upload.synthetic_images} synthetic images locally using PCA. Explained variance: ${(upload.explained_variance * 100).toFixed(1)}%.`
      );

      if (imageLogSource) imageLogSource.close();
      imageLogSource = new EventSource("/api/logs");

      imageLogSource.onmessage = (event) => {
        let item;

        try {
          item = JSON.parse(event.data);
        } catch {
          return;
        }

        if (!item?.level) return;

        addLogSafe(item.level, item.message, item.ts);

        if (item.level === "done") {
          imageLogSource.close();
          imageLogSource = null;
          imageAuditRunning = false;

          runButton.disabled = false;
          runButton.innerHTML = "&#9654; &nbsp;Run Audit";
          setStatusSafe("Audit complete", "done");

          try {
            renderImageResults(JSON.parse(item.message));
          } catch (error) {
            addLogSafe("error", `Could not show image results: ${error.message}`);
          }
        }
      };

      const auditResponse = await fetch("/api/run-image-audit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          modality: "image",
          original_folder: upload.original_folder,
          synthetic_folder: upload.synthetic_folder,
          generated_zip: upload.generated_zip || null,
          privacy_thresholds: {
            near_duplicate: 0.10,
          },
          utility_thresholds: {
            composite_utility: 0.70,
          },
        }),
      });

      const audit = await auditResponse.json();

      if (!auditResponse.ok || audit.error) {
        throw new Error(audit.error || "Could not start the image audit.");
      }
    } catch (error) {
      if (imageLogSource) {
        imageLogSource.close();
        imageLogSource = null;
      }

      imageAuditRunning = false;
      runButton.disabled = false;
      runButton.innerHTML = "&#9654; &nbsp;Run Audit";
      setStatusSafe("Error", "error");
      addLogSafe("error", error.message);
      alert(error.message);
    }
  }

  function renderImageResults(summary) {
    const results = document.getElementById("imageAuditResults");
    const body = document.getElementById("imageAuditResultsBody");

    const row = summary.results?.[0] || summary;
    const details = summary.dataset_details || {};
    const risk = Number(row.near_duplicate_risk ?? 0);
    const utility = Number(row.composite_utility ?? 0);
    const score = Number(row.composite_score ?? 0);
    const matches = Number(row.duplicate_count ?? row.near_duplicate_matches ?? 0);
    const passed = Boolean(summary.threshold_passed ?? summary.overall_pass);

    const sharedResults = document.getElementById("resultsSection");
    if (sharedResults) sharedResults.classList.remove("show");
    results.style.display = "block";

    body.innerHTML = `
      <div class="midst-image-results-grid">
        <div class="midst-image-stat">
          <small>Original images</small>
          <strong>${details.original_images ?? "N/A"}</strong>
        </div>
        <div class="midst-image-stat">
          <small>Synthetic images</small>
          <strong>${details.synthetic_images ?? "N/A"}</strong>
        </div>
        <div class="midst-image-stat">
          <small>Near-duplicate risk</small>
          <strong>${(risk * 100).toFixed(1)}%</strong>
        </div>
        <div class="midst-image-stat">
          <small>Composite utility</small>
          <strong>${(utility * 100).toFixed(1)}%</strong>
        </div>
      </div>

      <div class="midst-image-verdict ${passed ? "" : "failed"}">
        <strong>${passed ? "Image audit passed" : "Image audit requires privacy review"}</strong>
        <div style="margin-top:7px; font-size:.86rem; line-height:1.55;">
          Composite score: ${score.toFixed(3)}.
          ${
            passed
              ? " The generated images remained within the configured privacy limit."
              : ` The audit found ${matches} near-duplicate match${matches === 1 ? "" : "es"} with the original images.`
          }
        </div>
      </div>

      <div class="midst-image-actions">
        <a href="/api/download/image_audit_report.csv" download>Download CSV report</a>
        <a href="/api/download/image_audit_summary.json" download>Download JSON summary</a>
        ${summary.run_config?.generated_zip
          ? `<a href="/api/download/${encodeURIComponent(summary.run_config.generated_zip)}" download>Download synthetic ZIP</a>`
          : ""}
      </div>

      <div class="midst-image-preview-title">Sample images</div>
      <div class="midst-image-previews" id="midstImagePreviews">
        <span style="color:var(--md-on-surf-var);font-size:.78rem;">Loading previews...</span>
      </div>
    `;

    loadImagePreviews(summary);
  }

  async function loadImagePreviews(summary) {
    const container = document.getElementById("midstImagePreviews");
    const config = summary.run_config || {};
    if (!container || !config.original_folder || !config.synthetic_folder) return;

    try {
      const response = await fetch("/api/image-previews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          original_folder: config.original_folder,
          synthetic_folder: config.synthetic_folder,
        }),
      });
      const data = await response.json();
      container.innerHTML = (data.previews || []).map((item) => `
        <div class="midst-image-preview">
          <img src="${item.data_url}" alt="${item.kind} preview">
          <small>${item.kind}</small>
        </div>
      `).join("") || "<span>No previews available.</span>";
    } catch {
      container.textContent = "Preview unavailable.";
    }
  }

  document.getElementById("imageAuditUpload").addEventListener(
    "click",
    uploadAndRunImageAudit
  );

  const visibleImageTile = findImageTile();

  if (visibleImageTile) {
    const replacementTile = visibleImageTile.cloneNode(true);

    // Cloning preserves the disabled attribute, so explicitly remove it.
    replacementTile.disabled = false;
    replacementTile.removeAttribute("disabled");
    replacementTile.style.pointerEvents = "auto";
    replacementTile.style.opacity = "1";
    replacementTile.style.cursor = "pointer";

    replacementTile.classList.remove("disabled", "coming-soon");
    replacementTile.classList.add("active");

    replaceTextInside(
      replacementTile,
      "Coming in Phase 2",
      "Local image audit"
    );

    visibleImageTile.replaceWith(replacementTile);

    replacementTile.addEventListener("click", activateImageMode);
  }

  const previousStartAudit = window.startAudit;

  window.startAudit = function () {
    if (imageModeActive) {
      uploadAndRunImageAudit();
      return;
    }

    previousStartAudit();
  };
})();
