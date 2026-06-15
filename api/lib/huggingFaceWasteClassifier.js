const { OFFICIAL_REFERENCE, THAI_WASTE_RULES } = require("./thaiWasteRules");
const { normalizeInput } = require("./imageInput");

const DEFAULT_HF_SPACE_API_URL = "https://chrissujinlv-bingo-thai-waste-api.hf.space";
const HF_SPACE_TIMEOUT_MS = Number(process.env.HF_SPACE_TIMEOUT_MS || 55000);
const BIN_KEYS = ["green", "yellow", "blue", "red"];

async function classifyWasteWithHuggingFace(input) {
  const normalized = normalizeInput(input);
  const spaceUrl = String(process.env.HF_SPACE_API_URL || DEFAULT_HF_SPACE_API_URL).replace(/\/+$/, "");
  const endpoint = `${spaceUrl}/classify`;
  const imageUrl = buildImageDataUrl(normalized.buffer, normalized.mimeType);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), HF_SPACE_TIMEOUT_MS);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: buildHeaders(),
      signal: controller.signal,
      body: JSON.stringify({
        image: imageUrl,
        mimeType: normalized.mimeType,
        fileName: normalized.fileName,
        source: normalized.source,
      }),
    });

    const responseText = await response.text();
    const parsed = parseJson(responseText, `Hugging Face Space returned HTTP ${response.status}.`);

    if (!response.ok || !parsed.ok) {
      throw createHttpError(response.status || 502, parsed.detail || parsed.error || "Hugging Face Space classification failed.");
    }

    return buildResultFromHuggingFace(parsed, {
      normalized,
      endpoint,
      modelRepoId: parsed.modelRepoId,
    });
  } catch (error) {
    if (error.statusCode) throw error;
    const statusCode = error.name === "AbortError" ? 504 : 502;
    throw createHttpError(
      statusCode,
      `Hugging Face model service is unavailable: ${error.message || "request failed"}.`
    );
  } finally {
    clearTimeout(timeout);
  }
}

function buildResultFromHuggingFace(parsed, context) {
  const bin = normalizeBin(parsed.bin);
  if (!bin) {
    throw createHttpError(502, "Hugging Face Space did not return a valid Thailand bin.");
  }

  const rule = THAI_WASTE_RULES[bin];
  const confidence = clampPercent(parsed.confidence);
  const top = Array.isArray(parsed.top) ? parsed.top : [];
  const scores = Object.fromEntries(
    top
      .map((item) => [normalizeBin(item.bin), clampPercent(item.confidence)])
      .filter(([key]) => key)
  );

  return {
    ok: true,
    algorithmVersion: "huggingface-space-vit-v1",
    classifier: "huggingface-space-vit",
    model: context.modelRepoId || "ChrisSujinlv/bingo-thai-four-bin-waste-vit",
    inputSource: context.normalized.source,
    source: OFFICIAL_REFERENCE,
    bin: rule.bin,
    label: rule.label,
    category: rule.category,
    categoryDetail: rule.categoryDetail,
    thaiName: rule.thaiName,
    chineseName: rule.chineseName,
    binColor: rule.binColor,
    examples: rule.examples,
    disposal: rule.disposal,
    avoid: rule.avoid,
    thailandTips: rule.thailandTips,
    officialBasis: rule.officialBasis,
    confidence,
    needsManualCheck: Boolean(parsed.needsManualCheck) || confidence < 70,
    scores: Object.keys(scores).length ? scores : { [bin]: confidence },
    evidence: [
      `Hugging Face model predicted: ${rule.label}.`,
      `Top model scores: ${formatTopScores(top) || `${bin} ${confidence}%`}.`,
      "The result is mapped to Thailand/Bangkok four-bin waste sorting guidance.",
    ],
    image: {
      source: context.normalized.source,
      mimeType: context.normalized.mimeType,
      bytes: context.normalized.buffer.length,
    },
    ai: {
      modelBin: bin,
      finalBin: bin,
      provider: "huggingface-space",
      modelRepoId: context.modelRepoId,
      endpoint: context.endpoint,
      rawTop: top,
    },
  };
}

function buildHeaders() {
  const headers = { "Content-Type": "application/json" };
  const token = process.env.HF_SPACE_TOKEN || process.env.HF_TOKEN;
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

function buildImageDataUrl(buffer, mimeType) {
  return `data:${mimeType};base64,${buffer.toString("base64")}`;
}

function normalizeBin(value) {
  const bin = String(value || "").trim().toLowerCase();
  return BIN_KEYS.includes(bin) ? bin : "";
}

function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 50;
  return Math.max(0, Math.min(100, Math.round(number)));
}

function formatTopScores(top) {
  if (!Array.isArray(top) || !top.length) return "";
  return top
    .slice(0, 4)
    .map((item) => {
      const bin = normalizeBin(item.bin);
      if (!bin) return "";
      return `${bin} ${clampPercent(item.confidence)}%`;
    })
    .filter(Boolean)
    .join(", ");
}

function parseJson(value, message) {
  try {
    return JSON.parse(value);
  } catch {
    throw new Error(message);
  }
}

function createHttpError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

module.exports = {
  classifyWasteWithHuggingFace,
};
