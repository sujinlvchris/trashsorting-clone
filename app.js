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
const uploadModeBtn = document.getElementById("uploadModeBtn");
const cameraModeBtn = document.getElementById("cameraModeBtn");
const uploadPanel = document.getElementById("uploadPanel");
const cameraPanel = document.getElementById("cameraPanel");
const cameraVideo = document.getElementById("cameraVideo");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraStatus = document.getElementById("cameraStatus");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const switchCameraBtn = document.getElementById("switchCameraBtn");
const classifyNowBtn = document.getElementById("classifyNowBtn");

let selectedFile = null;
let previewUrl = "";
let cameraStream = null;
let cameraFacingMode = "environment";
let scanTimer = null;
let scanInFlight = false;
let lastCameraCandidate = "";
let stableCameraHits = 0;

const SCAN_INTERVAL_MS = 1500;
const CAMERA_STABLE_HITS = 2;
const CAMERA_MAX_FRAME_SIZE = 2048;
const CAMERA_FRAME_MIME_TYPE = "image/jpeg";
const CAMERA_FRAME_QUALITY = 0.97;

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

uploadModeBtn.addEventListener("click", () => setActiveMode("upload"));
cameraModeBtn.addEventListener("click", () => setActiveMode("camera"));
startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", () => stopCamera("Camera stopped."));
switchCameraBtn.addEventListener("click", switchCamera);
classifyNowBtn.addEventListener("click", () => classifyCameraFrame(true));

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

document.addEventListener("visibilitychange", () => {
  if (document.hidden && cameraStream) stopCamera("Camera paused while the page is hidden.");
});

window.addEventListener("pagehide", () => {
  if (cameraStream) stopCamera("Camera stopped.");
});

function setActiveMode(mode) {
  const isCamera = mode === "camera";
  uploadModeBtn.classList.toggle("is-active", !isCamera);
  cameraModeBtn.classList.toggle("is-active", isCamera);
  uploadModeBtn.setAttribute("aria-pressed", String(!isCamera));
  cameraModeBtn.setAttribute("aria-pressed", String(isCamera));
  uploadPanel.hidden = isCamera;
  cameraPanel.hidden = !isCamera;

  hideError();
  if (!isCamera && cameraStream) stopCamera("Camera stopped.");
  if (isCamera) setCameraStatus("Point the item inside the frame, then start the camera.");
}

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
  return classifyImagePayload({
    image,
    fileName: file.name || "",
    mimeType: file.type,
    source: "upload",
  });
}

async function classifyImagePayload(payload) {
  const response = await fetch("/api/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
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

async function startCamera() {
  hideError();

  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus("This browser does not support camera access.", "error");
    return;
  }

  if (!window.isSecureContext && !["localhost", "127.0.0.1"].includes(window.location.hostname)) {
    setCameraStatus("Camera access requires HTTPS. Please use the deployed Vercel URL.", "error");
    return;
  }

  stopCamera("", { silent: true });
  setCameraStatus("Requesting camera permission...");
  startCameraBtn.disabled = true;

  try {
    cameraStream = await getCameraStream();
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    cameraPanel.classList.add("is-running");
    setCameraButtons(true);
    resetCameraStability();
    setCameraStatus("Scanning every 1.5 seconds. Keep the item inside the frame.", "good");
    startCameraScanner();
  } catch (error) {
    cameraStream = null;
    cameraPanel.classList.remove("is-running", "is-scanning");
    setCameraButtons(false);
    setCameraStatus(formatCameraError(error), "error");
  }
}

async function getCameraStream() {
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: cameraFacingMode },
      width: { ideal: 960 },
      height: { ideal: 720 },
    },
  };

  try {
    return await navigator.mediaDevices.getUserMedia(constraints);
  } catch (error) {
    if (cameraFacingMode === "environment") {
      return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    throw error;
  }
}

function stopCamera(message = "Camera stopped.", options = {}) {
  if (scanTimer) {
    clearInterval(scanTimer);
    scanTimer = null;
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  cameraVideo.pause();
  cameraVideo.srcObject = null;
  scanInFlight = false;
  resetCameraStability();
  cameraPanel.classList.remove("is-running", "is-scanning");
  setCameraButtons(false);
  if (!options.silent && message) setCameraStatus(message);
}

async function switchCamera() {
  cameraFacingMode = cameraFacingMode === "environment" ? "user" : "environment";
  const wasRunning = Boolean(cameraStream);
  stopCamera("Switching camera...");
  if (wasRunning) await startCamera();
}

function startCameraScanner() {
  if (scanTimer) clearInterval(scanTimer);
  window.setTimeout(() => classifyCameraFrame(false), 450);
  scanTimer = window.setInterval(() => classifyCameraFrame(false), SCAN_INTERVAL_MS);
}

async function classifyCameraFrame(forceRender = false) {
  if (!cameraStream || cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    setCameraStatus("Camera is not ready yet. Try again in a moment.");
    return null;
  }

  if (scanInFlight) {
    if (forceRender) setCameraStatus("Still analysing the previous frame...");
    return null;
  }

  scanInFlight = true;
  cameraPanel.classList.add("is-scanning");
  if (forceRender) setCameraStatus("Analysing current frame...");

  try {
    const image = captureCameraFrame();
    const result = await classifyImagePayload({
      image,
      mimeType: CAMERA_FRAME_MIME_TYPE,
      fileName: "camera-frame.jpg",
      source: "camera",
    });
    handleCameraResult(result, forceRender);
    return result;
  } catch (error) {
    setCameraStatus(error.message || "Camera classification failed. Please try again.", "error");
    return null;
  } finally {
    scanInFlight = false;
    cameraPanel.classList.remove("is-scanning");
  }
}

function captureCameraFrame() {
  const videoWidth = cameraVideo.videoWidth || cameraVideo.clientWidth;
  const videoHeight = cameraVideo.videoHeight || cameraVideo.clientHeight;

  if (!videoWidth || !videoHeight) {
    throw new Error("Camera frame is not available yet.");
  }

  const scale = Math.min(1, CAMERA_MAX_FRAME_SIZE / Math.max(videoWidth, videoHeight));
  cameraCanvas.width = Math.max(1, Math.round(videoWidth * scale));
  cameraCanvas.height = Math.max(1, Math.round(videoHeight * scale));

  const context = cameraCanvas.getContext("2d", { willReadFrequently: false });
  context.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
  return cameraCanvas.toDataURL(CAMERA_FRAME_MIME_TYPE, CAMERA_FRAME_QUALITY);
}

function handleCameraResult(result, forceRender) {
  const confidence = Number(result.confidence || 0);
  const bin = result.bin || "blue";

  if (bin === lastCameraCandidate) {
    stableCameraHits += 1;
  } else {
    lastCameraCandidate = bin;
    stableCameraHits = 1;
  }

  if (!forceRender && confidence < 60) {
    setCameraStatus("Need clearer view. Move closer, improve light, or use Classify now.");
    return;
  }

  if (!forceRender && stableCameraHits < CAMERA_STABLE_HITS) {
    setCameraStatus("Checking stability... keep the item steady in the frame.");
    return;
  }

  renderResult(result);
  resultCard.classList.add("show");
  setCameraStatus(
    result.needsManualCheck
      ? "Result shown with low confidence. Please confirm with local bin signage."
      : "Stable result updated.",
    result.needsManualCheck ? "" : "good"
  );
}

function resetCameraStability() {
  lastCameraCandidate = "";
  stableCameraHits = 0;
}

function setCameraButtons(isRunning) {
  startCameraBtn.disabled = isRunning;
  stopCameraBtn.disabled = !isRunning;
  switchCameraBtn.disabled = !isRunning;
  classifyNowBtn.disabled = !isRunning;
}

function setCameraStatus(message, type = "") {
  cameraStatus.textContent = message;
  cameraStatus.classList.toggle("is-error", type === "error");
  cameraStatus.classList.toggle("is-good", type === "good");
}

function formatCameraError(error) {
  if (error?.name === "NotAllowedError") {
    return "Camera permission was denied. Allow camera access and try again.";
  }
  if (error?.name === "NotFoundError") {
    return "No camera was found on this device.";
  }
  if (error?.name === "NotReadableError") {
    return "The camera is already in use by another app.";
  }
  if (error?.name === "OverconstrainedError") {
    return "This camera mode is not available on the device.";
  }
  return error?.message || "Camera could not be started.";
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
    async classifyDemoAsCamera(name = "banana-peel-organic.png") {
      const canvas = document.createElement("canvas");
      canvas.width = 240;
      canvas.height = 180;
      const context = canvas.getContext("2d");
      context.fillStyle = "#3fa65a";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#d5b04d";
      context.beginPath();
      context.ellipse(126, 92, 68, 34, -0.3, 0, Math.PI * 2);
      context.fill();
      const result = await classifyImagePayload({
        image: canvas.toDataURL("image/png"),
        mimeType: "image/png",
        fileName: name,
        source: "camera",
      });
      renderResult(result);
      resultCard.classList.add("show");
      return result;
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
