const uploadCard = document.querySelector(".upload-card");
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const previewBox = document.getElementById("previewBox");
const previewImg = document.getElementById("previewImg");
const fileMeta = document.getElementById("fileMeta");
const submitBtn = document.getElementById("submitBtn");
const errorMsg = document.getElementById("errorMsg");
const loading = document.getElementById("loading");
const resultCard = document.getElementById("resultCard");
const resultContent = document.getElementById("resultContent");

let selectedFile = null;
let previewUrl = "";

const binAssets = {
  green: "./assets/bin-green.png",
  yellow: "./assets/bin-yellow.png",
  blue: "./assets/bin-blue.png",
  red: "./assets/bin-red.png",
};

const accentMap = {
  green: "var(--green)",
  yellow: "var(--yellow)",
  blue: "var(--blue)",
  red: "var(--red)",
};

uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadCard.classList.add("is-dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadCard.classList.remove("is-dragover");
});

uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadCard.classList.remove("is-dragover");
  const file = event.dataTransfer?.files?.[0];
  if (file) setFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) setFile(file);
});

submitBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  setLoading(true);
  resultCard.classList.remove("show");
  submitBtn.disabled = true;

  try {
    const result = await classifyWithBackend(selectedFile);
    renderResult(result);
    resultCard.classList.add("show");
  } catch (error) {
    showError(error.message || "Classification failed. Please try another image.");
  } finally {
    setLoading(false);
    submitBtn.disabled = false;
  }
});

function setFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please choose a JPG, PNG, or WebP image.");
    return;
  }

  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showError("This file type is not supported. Please use JPG, PNG, or WebP.");
    return;
  }

  selectedFile = file;

  if (previewUrl) URL.revokeObjectURL(previewUrl);

  previewUrl = URL.createObjectURL(file);
  previewImg.src = previewUrl;
  previewBox.classList.add("show");
  fileMeta.textContent = `${file.name || "Selected image"} · ${formatBytes(file.size)}`;
  submitBtn.disabled = false;
  resultCard.classList.remove("show");
  hideError();
}

async function classifyWithBackend(file) {
  const image = await fileToDataUrl(file);
  const response = await fetch("/api/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image,
      fileName: file.name || "",
      mimeType: file.type,
    }),
  });

  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { ok: false, error: text || "Invalid backend response." };
  }

  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Backend classification failed.");
  }

  return data;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read image."));
    reader.readAsDataURL(file);
  });
}

function renderResult(result) {
  const binKey = result.bin || "blue";
  const asset = binAssets[binKey] || binAssets.blue;
  resultCard.style.setProperty("--accent", accentMap[binKey] || accentMap.blue);

  const evidence = Array.isArray(result.evidence) ? result.evidence : [];
  const tips = Array.isArray(result.thailandTips) ? result.thailandTips : [];
  const examples = Array.isArray(result.examples) ? result.examples : [];
  const source = result.source || {};

  resultContent.innerHTML = `
    <div class="bin-badge ${escapeHtml(binKey)}">${escapeHtml(result.label || "Blue bin · General waste")}</div>
    <img class="bin-illustration" src="${asset}" alt="${escapeHtml(result.label || "Waste bin")}" />
    <div class="result-section">
      <h3>Category</h3>
      <p>${escapeHtml(result.categoryDetail || "General waste")}</p>
      <p class="category-sub">${escapeHtml(result.thaiName || "")}${result.thaiName && result.chineseName ? " · " : ""}${escapeHtml(result.chineseName || "")}</p>
    </div>
    <div class="result-section">
      <h3>Why</h3>
      ${renderList(evidence.length ? evidence : ["Manual confirmation recommended."])}
    </div>
    <div class="result-section">
      <h3>Disposal</h3>
      <p>${escapeHtml(result.disposal || "")}</p>
    </div>
    <div class="result-section">
      <h3>Thailand Tips</h3>
      ${renderList(tips)}
    </div>
    <div class="result-section">
      <h3>Examples</h3>
      ${renderList(examples)}
    </div>
    <div class="result-section">
      <h3>Avoid</h3>
      <p>${escapeHtml(result.avoid || "")}</p>
    </div>
    <div class="result-section official-note">
      <h3>Basis</h3>
      <p>${escapeHtml(result.officialBasis || "")}</p>
      ${source.url ? `<a href="${escapeAttribute(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.title || "Official reference")}</a>` : ""}
    </div>
    <div class="confidence-row">
      <div class="confidence-track" aria-label="Confidence ${Number(result.confidence || 0)}%">
        <div class="confidence-fill" style="--confidence: ${Number(result.confidence || 0)}%"></div>
      </div>
      <span>${Number(result.confidence || 0)}% confidence</span>
    </div>
    ${result.needsManualCheck ? `<p class="manual-check">Please confirm manually if local bin signage differs.</p>` : ""}
  `;
}

function renderList(items) {
  if (!Array.isArray(items) || !items.length) return "<p>Not available.</p>";
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function setLoading(isLoading) {
  loading.classList.toggle("show", isLoading);
  loading.setAttribute("aria-hidden", String(!isLoading));
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.classList.add("show");
}

function hideError() {
  errorMsg.textContent = "";
  errorMsg.classList.remove("show");
}

function formatBytes(bytes) {
  if (!bytes) return "0 KB";

  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value ?? "");
  return div.innerHTML;
}

function escapeAttribute(value) {
  return String(value ?? "").replace(/"/g, "&quot;");
}

if (["localhost", "127.0.0.1"].includes(window.location.hostname)) {
  window.__trashSorterTest = {
    async loadDemo(name = "banana-peel-organic.png") {
      const canvas = document.createElement("canvas");
      canvas.width = 180;
      canvas.height = 120;
      const context = canvas.getContext("2d");
      context.fillStyle = "#55aa43";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#d8f0c5";
      context.beginPath();
      context.ellipse(76, 56, 42, 24, -0.4, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#f5d060";
      context.beginPath();
      context.arc(125, 50, 16, 0, Math.PI * 2);
      context.fill();
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      const file = new File([blob], name, { type: "image/png" });
      setFile(file);
      return {
        buttonDisabled: submitBtn.disabled,
        previewVisible: previewBox.classList.contains("show"),
        meta: fileMeta.textContent,
      };
    },
  };

  const params = new URLSearchParams(window.location.search);
  if (params.has("demo")) {
    window.addEventListener("load", async () => {
      const name = params.get("demo") || "banana-peel-organic.png";
      await window.__trashSorterTest.loadDemo(name);
      if (params.has("classify")) submitBtn.click();
    });
  }
}
