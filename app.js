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

const bins = {
  green: {
    label: "Green bin · Organic",
    detail: "Organic and food waste",
    color: "var(--green)",
    asset: "./assets/bin-green.png",
    disposal:
      "Drain liquids, keep it separate from dry recyclables, and place it in the green organic bin. Composting is the best option when available.",
    tips: [
      "Good matches: food scraps, fruit peels, vegetable trimmings, leaves, flowers, and other biodegradable waste.",
      "Avoid mixing plastic bags, foam boxes, or hazardous items into organic waste.",
    ],
  },
  yellow: {
    label: "Yellow bin · Recyclable",
    detail: "Recyclable waste",
    color: "var(--yellow)",
    asset: "./assets/bin-yellow.png",
    disposal:
      "Empty, rinse, and dry the item first. Flatten bulky packaging, then place it in the yellow recyclable bin.",
    tips: [
      "Good matches: plastic bottles, clean paper, cardboard, glass bottles, metal cans, and cartons.",
      "Food-stained packaging should be cleaned before recycling; if it cannot be cleaned, use general waste.",
    ],
  },
  blue: {
    label: "Blue bin · General waste",
    detail: "General non-recyclable waste",
    color: "var(--blue)",
    asset: "./assets/bin-blue.png",
    disposal:
      "Bag it securely and place it in the blue general waste bin. Separate anything recyclable or hazardous before disposal.",
    tips: [
      "Good matches: tissues, snack wrappers, foam containers, broken ceramics, and mixed-material packaging.",
      "Reduce volume where possible and keep sharp fragments wrapped for collection safety.",
    ],
  },
  red: {
    label: "Red bin · Hazardous",
    detail: "Hazardous waste",
    color: "var(--red)",
    asset: "./assets/bin-red.png",
    disposal:
      "Do not mix this with household trash. Keep the item sealed, label it if needed, and use the red hazardous waste bin or a municipal drop-off point.",
    tips: [
      "Good matches: batteries, lamps, chemical containers, paint, aerosols, medicines, electronics, and sharp medical waste.",
      "Keep leaking or broken items in a separate container before taking them to collection.",
    ],
  },
};

const keywordRules = [
  {
    bin: "red",
    words: [
      "battery",
      "batteries",
      "chemical",
      "paint",
      "medicine",
      "medical",
      "syringe",
      "needle",
      "lamp",
      "bulb",
      "aerosol",
      "spray",
      "e-waste",
      "ewaste",
      "phone",
      "circuit",
      "charger",
      "hazard",
    ],
  },
  {
    bin: "green",
    words: [
      "food",
      "banana",
      "apple",
      "peel",
      "leaf",
      "leaves",
      "flower",
      "vegetable",
      "compost",
      "organic",
      "leftover",
      "fruit",
      "rice",
    ],
  },
  {
    bin: "yellow",
    words: [
      "bottle",
      "plastic",
      "paper",
      "cardboard",
      "carton",
      "glass",
      "can",
      "tin",
      "aluminium",
      "aluminum",
      "recycle",
      "newspaper",
      "box",
    ],
  },
  {
    bin: "blue",
    words: [
      "wrapper",
      "tissue",
      "napkin",
      "foam",
      "styrofoam",
      "ceramic",
      "diaper",
      "mask",
      "mixed",
      "trash",
      "general",
    ],
  },
];

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
  if (file) {
    setFile(file);
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) {
    setFile(file);
  }
});

submitBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  hideError();
  setLoading(true);
  resultCard.classList.remove("show");
  submitBtn.disabled = true;

  try {
    await wait(620);
    const result = await classifyImage(selectedFile, previewImg);
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

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImg.src = previewUrl;
  previewBox.classList.add("show");
  fileMeta.textContent = `${file.name || "Selected image"} · ${formatBytes(file.size)}`;
  submitBtn.disabled = false;
  resultCard.classList.remove("show");
  hideError();
}

async function classifyImage(file, image) {
  const scores = {
    green: 0,
    yellow: 0,
    blue: 0,
    red: 0,
  };
  const reasons = [];
  const filename = `${file.name || ""}`.toLowerCase();

  for (const rule of keywordRules) {
    const matches = rule.words.filter((word) => filename.includes(word));
    if (matches.length) {
      scores[rule.bin] += matches.length * 8;
      reasons.push(`filename matched ${matches.slice(0, 3).join(", ")}`);
    }
  }

  const imageSignal = await analyseImagePixels(image);
  if (imageSignal.greenRatio > 0.28 || imageSignal.greenLead > 22) {
    scores.green += 3;
    reasons.push("green/natural tones detected");
  }
  if (imageSignal.brightRatio > 0.42 && imageSignal.saturation < 0.28) {
    scores.yellow += 2;
    reasons.push("clean bright packaging tones detected");
  }
  if (imageSignal.redRatio > 0.22 && scores.red > 0) {
    scores.red += 2;
    reasons.push("red hazard-like tones detected");
  }
  if (imageSignal.darkRatio > 0.38 && scores.red > 0) {
    scores.red += 1;
  }

  scores.blue += 1;

  const binKey = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
  const confidence = Math.min(94, Math.max(58, 58 + scores[binKey] * 5));

  return {
    bin: binKey,
    confidence,
    reason: reasons.length ? reasons.slice(0, 2).join("; ") : "general waste fallback",
  };
}

async function analyseImagePixels(image) {
  await waitForImage(image);

  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    return emptySignal();
  }

  context.drawImage(image, 0, 0, size, size);
  const { data } = context.getImageData(0, 0, size, size);
  let greenPixels = 0;
  let redPixels = 0;
  let brightPixels = 0;
  let darkPixels = 0;
  let saturationTotal = 0;
  let greenLeadTotal = 0;
  let counted = 0;

  for (let index = 0; index < data.length; index += 4) {
    const alpha = data[index + 3];
    if (alpha < 20) {
      continue;
    }

    const red = data[index];
    const green = data[index + 1];
    const blue = data[index + 2];
    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    const brightness = (red + green + blue) / 3;
    const saturation = max === 0 ? 0 : (max - min) / max;

    if (green > red + 18 && green > blue + 10) {
      greenPixels += 1;
    }
    if (red > green + 25 && red > blue + 18) {
      redPixels += 1;
    }
    if (brightness > 202) {
      brightPixels += 1;
    }
    if (brightness < 68) {
      darkPixels += 1;
    }

    greenLeadTotal += green - Math.max(red, blue);
    saturationTotal += saturation;
    counted += 1;
  }

  if (!counted) {
    return emptySignal();
  }

  return {
    greenRatio: greenPixels / counted,
    redRatio: redPixels / counted,
    brightRatio: brightPixels / counted,
    darkRatio: darkPixels / counted,
    saturation: saturationTotal / counted,
    greenLead: greenLeadTotal / counted,
  };
}

function emptySignal() {
  return {
    greenRatio: 0,
    redRatio: 0,
    brightRatio: 0,
    darkRatio: 0,
    saturation: 0,
    greenLead: 0,
  };
}

function waitForImage(image) {
  if (image.complete && image.naturalWidth > 0) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    image.addEventListener("load", resolve, { once: true });
    image.addEventListener("error", () => reject(new Error("Failed to read image.")), {
      once: true,
    });
  });
}

function renderResult(result) {
  const bin = bins[result.bin] || bins.blue;
  resultCard.style.setProperty("--accent", bin.color);
  resultContent.innerHTML = `
    <div class="bin-badge ${result.bin}">${escapeHtml(bin.label)}</div>
    <img class="bin-illustration" src="${bin.asset}" alt="${escapeHtml(bin.label)}" />
    <div class="result-section">
      <h3>Category</h3>
      <p>${escapeHtml(bin.detail)}</p>
    </div>
    <div class="result-section">
      <h3>Disposal</h3>
      <p>${escapeHtml(bin.disposal)}</p>
    </div>
    <div class="result-section">
      <h3>Thailand Tips</h3>
      <ul>${bin.tips.map((tip) => `<li>${escapeHtml(tip)}</li>`).join("")}</ul>
    </div>
    <div class="confidence-row">
      <div class="confidence-track" aria-label="Confidence ${result.confidence}%">
        <div class="confidence-fill" style="--confidence: ${result.confidence}%"></div>
      </div>
      <span>${result.confidence}% confidence</span>
    </div>
  `;
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
  if (!bytes) {
    return "0 KB";
  }

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

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
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
      if (params.has("classify")) {
        submitBtn.click();
      }
    });
  }
}
