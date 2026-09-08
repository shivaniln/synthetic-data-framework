(() => {
  "use strict";

  let imageModeActive = false;
  let imageLogSource = null;

  const csvUploadZone = document.getElementById("uploadZone");
  const imageUploadZone = document.getElementById("imageUploadZone");
  const originalButton = document.getElementById("enableImageMode");
  const csvButton = document.getElementById("enableCsvMode");

  if (!csvUploadZone || !imageUploadZone || !originalButton || !csvButton) {
    console.warn("MIDST image controls were not found.");
    return;
  }

  const stepTitle = document.querySelector(".sec-head .sec-title");
  const stepSubtitle = document.querySelector(".sec-head .sec-sub");

  const originalTitle = stepTitle ? stepTitle.textContent : "Step 1 - Upload your dataset";
  const originalSubtitle = stepSubtitle
    ? stepSubtitle.textContent
    : "Drop a CSV file here.";

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

  function setImageMode(enabled) {
    imageModeActive = enabled;

    if (enabled) {
      csvUploadZone.style.display = "none";
      imageUploadZone.style.display = "block";

      originalButton.classList.add("active");
      csvButton.classList.remove("active");

      if (stepTitle) {
        stepTitle.textContent = "Step 1 - Upload image datasets";
      }

      if (stepSubtitle) {
        stepSubtitle.textContent =
          "Upload one ZIP containing original images and one ZIP containing synthetic images. The files are processed locally.";
      }

      return;
    }

    csvUploadZone.style.display = "";
    imageUploadZone.style.display = "none";

    originalButton.classList.remove("active");
    csvButton.classList.add("active");

    if (stepTitle) stepTitle.textContent = originalTitle;
    if (stepSubtitle) stepSubtitle.textContent = originalSubtitle;
  }

  // Replace the original buttons to remove conflicting click handlers.
  const imageButton = originalButton.cloneNode(true);
  const normalButton = csvButton.cloneNode(true);

  originalButton.replaceWith(imageButton);
  csvButton.replaceWith(normalButton);

  imageButton.addEventListener("click", () => setImageMode(true));
  normalButton.addEventListener("click", () => setImageMode(false));

  function renderFallbackImageResults(summary) {
    const resultArea = document.getElementById("imageResults");
    const resultBody = document.getElementById("imageResultsBody");

    if (!resultArea || !resultBody) return;

    const risk = Number(summary.near_duplicate_risk ?? 0);
    const utility = Number(summary.composite_utility ?? 0);
    const score = Number(summary.composite_score ?? 0);
    const passed = Boolean(summary.threshold_passed ?? summary.overall_pass);

    resultArea.style.display = "block";

    resultBody.innerHTML = `
      <div class="image-result-grid">
        <div class="image-result-stat">
          <div class="image-result-label">Original images</div>
          <div class="image-result-value">${summary.original_images ?? "N/A"}</div>
        </div>
        <div class="image-result-stat">
          <div class="image-result-label">Synthetic images</div>
          <div class="image-result-value">${summary.synthetic_images ?? "N/A"}</div>
        </div>
        <div class="image-result-stat">
          <div class="image-result-label">Near-duplicate risk</div>
          <div class="image-result-value">${(risk * 100).toFixed(1)}%</div>
        </div>
        <div class="image-result-stat">
          <div class="image-result-label">Composite utility</div>
          <div class="image-result-value">${(utility * 100).toFixed(1)}%</div>
        </div>
      </div>

      <div class="image-verdict ${passed ? "" : "failed"}">
        <div class="image-verdict-title">
          ${passed ? "Image audit passed" : "Image audit requires review"}
        </div>
        <div class="image-verdict-text">
          Composite score: ${score.toFixed(3)}.
          ${
            passed
              ? " The generated images stayed within the configured privacy limit."
              : " The generated images are still too similar to one or more original images."
          }
        </div>
      </div>
    `;
  }

  async function runImageAudit() {
    const originalFile = document.getElementById("originalImageZip")?.files?.[0];
    const syntheticFile = document.getElementById("syntheticImageZip")?.files?.[0];
    const runButton = document.getElementById("btnRun");

    if (!originalFile || !syntheticFile) {
      alert("Choose both the original-images ZIP and synthetic-images ZIP.");
      return;
    }

    runButton.disabled = true;
    runButton.textContent = "Uploading images...";
    setStatusSafe("Uploading...", "");

    try {
      const formData = new FormData();
      formData.append("original_zip", originalFile);
      formData.append("synthetic_zip", syntheticFile);

      const uploadResponse = await fetch("/api/upload-images", {
        method: "POST",
        body: formData,
      });

      const uploadData = await uploadResponse.json();

      if (!uploadResponse.ok || uploadData.error) {
        throw new Error(uploadData.error || "Could not upload image ZIP files.");
      }

      addLogSafe(
        "info",
        `Uploaded ${uploadData.original_images} original and ${uploadData.synthetic_images} synthetic images locally.`
      );

      runButton.textContent = "Running image audit...";
      setStatusSafe("Running...", "running");

      if (typeof window.clearLogs === "function") {
        window.clearLogs();
      }

      if (imageLogSource) {
        imageLogSource.close();
      }

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

          try {
            const summary = JSON.parse(item.message);
            renderFallbackImageResults(summary);
          } catch (error) {
            addLogSafe("error", `Could not display image results: ${error.message}`);
          }

          runButton.disabled = false;
          runButton.innerHTML = "&#9654; &nbsp;Run Audit";
          setStatusSafe("Audit complete", "done");
        }
      };

      const auditResponse = await fetch("/api/run-image-audit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          modality: "image",
          original_folder: uploadData.original_folder,
          synthetic_folder: uploadData.synthetic_folder,
          privacy_thresholds: {
            near_duplicate: 0.10,
          },
          utility_thresholds: {
            composite_utility: 0.70,
          },
        }),
      });

      const auditData = await auditResponse.json();

      if (!auditResponse.ok || auditData.error) {
        throw new Error(auditData.error || "Could not start image audit.");
      }
    } catch (error) {
      if (imageLogSource) {
        imageLogSource.close();
        imageLogSource = null;
      }

      runButton.disabled = false;
      runButton.innerHTML = "&#9654; &nbsp;Run Audit";
      setStatusSafe("Error", "error");
      addLogSafe("error", error.message);
      alert(error.message);
    }
  }

  const earlierStartAudit = window.startAudit;

  window.startAudit = function () {
    if (imageModeActive) {
      runImageAudit();
      return;
    }

    earlierStartAudit();
  };
})();