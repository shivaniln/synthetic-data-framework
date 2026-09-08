(() => {
  "use strict";

  function loadImageControls() {
    if (document.getElementById("imageAuditWorkspace")) {
      attachDirectImageClick();
      return;
    }

    const script = document.createElement("script");
    script.src = "/phase2_image_unlock.js?after-dom=1";
    script.onload = () => attachDirectImageClick();
    document.body.appendChild(script);
  }

  function attachDirectImageClick() {
    const tile = document.getElementById("imageModalityButton");
    const csvZone = document.getElementById("uploadZone");
    const imageZone = document.getElementById("imageAuditWorkspace");

    if (!tile || !csvZone || !imageZone || tile.dataset.imageClickBound) {
      return;
    }

    tile.dataset.imageClickBound = "true";
    tile.disabled = false;
    tile.removeAttribute("disabled");

    tile.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopImmediatePropagation();

      if (window.state) window.state.modality = "image";

      csvZone.style.display = "none";
      imageZone.style.display = "block";
      hideTabularControls(true);

      document.querySelectorAll(".modality-btn").forEach((button) => {
        button.classList.toggle("active", button === tile);
      });

      const title = document.querySelector(".sec-head .sec-title");
      const subtitle = document.querySelector(".sec-head .sec-sub");

      if (title) title.textContent = "Step 1 - Upload image datasets";
      if (subtitle) {
        subtitle.textContent =
          "Upload original and synthetic image folders as ZIP files. MIDST processes both datasets locally.";
      }

      imageZone.scrollIntoView({ behavior: "smooth", block: "center" });
    }, true);

    attachNonImageModeReset(tile, csvZone, imageZone);
  }

  function attachNonImageModeReset(imageTile, csvZone, imageZone) {
    document.querySelectorAll(".modality-btn[data-modality]").forEach((button) => {
      if (button.dataset.imageResetBound) return;
      button.dataset.imageResetBound = "true";

      button.addEventListener("click", () => {
        if (window.state) window.state.modality = button.dataset.modality;

        imageTile.classList.remove("active");
        csvZone.style.display = "";
        imageZone.style.display = "none";

        hideTabularControls(false);

        const imageResults = document.getElementById("imageAuditResults");
        if (imageResults) imageResults.style.display = "none";

        // A previous time-series/tabular audit must not remain visible
        // after switching modality.
        const sharedResults = document.getElementById("resultsSection");
        if (sharedResults) sharedResults.classList.remove("show");
      }, true);
    });
  }

  function hideTabularControls(hidden) {
    const stepTwoHeading = [...document.querySelectorAll(".sec-head")].find(
      (element) => element.textContent.includes("Step 2 - Configure the audit")
    );

    if (stepTwoHeading) {
      stepTwoHeading.style.display = hidden ? "none" : "";
      const configGrid = stepTwoHeading.nextElementSibling;
      if (configGrid && configGrid.classList.contains("config-grid")) {
        configGrid.style.display = hidden ? "none" : "";
      }
    }

    const placeholder = document.getElementById("placeholder");
    if (placeholder) placeholder.style.display = hidden ? "none" : "";
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadImageControls, { once: true });
  } else {
    loadImageControls();
  }
})();
