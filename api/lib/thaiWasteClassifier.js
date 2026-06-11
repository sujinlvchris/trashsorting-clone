const zlib = require("zlib");
const { CATEGORY_ORDER, OFFICIAL_REFERENCE, THAI_WASTE_RULES } = require("./thaiWasteRules");

const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
const SUPPORTED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

function classifyWaste(input) {
  const normalized = normalizeInput(input);
  const imageInfo = analyzeImage(normalized.buffer, normalized.mimeType);
  const text = normalizeText(
    [normalized.fileName, normalized.hint, normalized.userLabel].filter(Boolean).join(" ")
  );
  const scores = createInitialScores();
  const evidence = [];

  for (const key of CATEGORY_ORDER) {
    const matches = matchRuleText(text, THAI_WASTE_RULES[key].keywords);
    if (!matches.length) continue;

    const weight = key === "red" ? 24 : key === "blue" ? 12 : 18;
    const score = matches.length * weight;
    scores[key] += score;
    evidence.push({
      bin: key,
      weight: score,
      message: `Keyword match (${THAI_WASTE_RULES[key].categoryDetail}): ${matches
        .slice(0, 5)
        .join(", ")}`,
    });
  }

  if (imageInfo.signals) {
    applyImageSignalScores(scores, evidence, imageInfo.signals);
  }

  // Thai sorting UX: hazardous hints should override weak image-only guesses.
  if (scores.red >= 24) scores.red += 12;

  const winner = pickWinner(scores);
  const runnerUp = pickRunnerUp(scores, winner);
  const rule = THAI_WASTE_RULES[winner];
  const confidence = computeConfidence(scores[winner], scores[runnerUp], evidence, imageInfo);
  const isFallback = evidence.length === 0 && winner === "blue";

  const resultEvidence = evidence
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 6)
    .map((item) => item.message);

  if (isFallback) {
    resultEvidence.push("No strong Thailand waste-rule signal was found, so the item is treated as general waste for manual confirmation.");
  }

  return {
    ok: true,
    algorithmVersion: "thai-no-mix-rules-v1",
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
    needsManualCheck: confidence < 64 || isFallback,
    scores,
    evidence: resultEvidence,
    image: {
      mimeType: normalized.mimeType,
      bytes: normalized.buffer.length,
      width: imageInfo.width || null,
      height: imageInfo.height || null,
      analyzedPixels: imageInfo.analyzedPixels || 0,
      signals: imageInfo.signals || null,
      notes: imageInfo.notes,
    },
  };
}

function normalizeInput(input) {
  if (!input || typeof input !== "object") {
    throw createHttpError(400, "Expected JSON body.");
  }

  const image = String(input.image || "");
  const mimeType = String(input.mimeType || "").toLowerCase();
  const fileName = String(input.fileName || input.filename || "");
  const hint = String(input.hint || "");
  const userLabel = String(input.userLabel || "");

  if (!image) throw createHttpError(400, "Missing image data.");
  if (!SUPPORTED_MIME_TYPES.includes(mimeType)) {
    throw createHttpError(415, "Only JPG, PNG, and WebP images are supported.");
  }

  const base64 = image.includes(",") ? image.split(",").pop() : image;
  if (!/^[a-zA-Z0-9+/=\s]+$/.test(base64)) {
    throw createHttpError(400, "Image must be base64 encoded.");
  }

  const buffer = Buffer.from(base64.replace(/\s/g, ""), "base64");
  if (!buffer.length) throw createHttpError(400, "Image is empty.");
  if (buffer.length > MAX_IMAGE_BYTES) {
    throw createHttpError(413, "Image is too large. Please use an image under 4 MB.");
  }

  return { buffer, fileName, hint, mimeType, userLabel };
}

function createInitialScores() {
  // General waste is the safe fallback only when no food/recycling/hazardous
  // signal is found.
  return { green: 0, yellow: 0, blue: 8, red: 0 };
}

function matchRuleText(text, keywordsByLanguage) {
  if (!text) return [];
  const allKeywords = [
    ...keywordsByLanguage.en,
    ...keywordsByLanguage.th,
    ...keywordsByLanguage.zh,
  ];
  const matches = [];

  for (const keyword of allKeywords) {
    const normalizedKeyword = normalizeText(keyword);
    if (!normalizedKeyword) continue;
    if (containsCjkOrThai(normalizedKeyword)) {
      if (text.includes(normalizedKeyword)) matches.push(keyword);
      continue;
    }

    const escaped = normalizedKeyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = normalizedKeyword.includes(" ")
      ? new RegExp(escaped.replace(/\s+/g, "[-_\\s]+"), "i")
      : new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, "i");
    if (pattern.test(text)) matches.push(keyword);
  }

  return [...new Set(matches)];
}

function analyzeImage(buffer, mimeType) {
  const base = {
    mimeType,
    width: null,
    height: null,
    analyzedPixels: 0,
    signals: null,
    notes: [],
  };

  try {
    if (mimeType === "image/png") return analyzePng(buffer, base);
    if (mimeType === "image/jpeg") {
      return {
        ...base,
        ...readJpegSize(buffer),
        notes: ["JPEG dimensions read; color pixels are not decoded in the lightweight server runtime."],
      };
    }
    if (mimeType === "image/webp") {
      return {
        ...base,
        ...readWebpSize(buffer),
        notes: ["WebP dimensions read; color pixels are not decoded in the lightweight server runtime."],
      };
    }
  } catch (error) {
    base.notes.push(`Image analysis failed: ${error.message}`);
  }

  return base;
}

function applyImageSignalScores(scores, evidence, signals) {
  if (signals.greenRatio > 0.28 || signals.greenLead > 24) {
    scores.green += 6;
    evidence.push({
      bin: "green",
      weight: 6,
      message: "Image color signal suggests natural green/plant-like material.",
    });
  }

  if (signals.redRatio > 0.2) {
    scores.red += 4;
    evidence.push({
      bin: "red",
      weight: 4,
      message: "Image color signal includes noticeable red/orange warning-like tones.",
    });
  }

  if (signals.brightRatio > 0.42 && signals.saturation < 0.26) {
    scores.yellow += 4;
    evidence.push({
      bin: "yellow",
      weight: 4,
      message: "Image color signal suggests clean bright packaging.",
    });
  }

  if (signals.darkRatio > 0.36 && signals.saturation > 0.25) {
    scores.blue += 3;
    evidence.push({
      bin: "blue",
      weight: 3,
      message: "Image color signal suggests mixed or dirty general waste.",
    });
  }
}

function analyzePng(buffer, base) {
  if (buffer.toString("hex", 0, 8) !== "89504e470d0a1a0a") {
    return { ...base, notes: ["PNG signature not found; color analysis skipped."] };
  }

  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  let interlace = 0;
  const idat = [];

  while (offset + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("ascii", offset + 4, offset + 8);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd > buffer.length) break;

    if (type === "IHDR") {
      width = buffer.readUInt32BE(dataStart);
      height = buffer.readUInt32BE(dataStart + 4);
      bitDepth = buffer[dataStart + 8];
      colorType = buffer[dataStart + 9];
      interlace = buffer[dataStart + 12];
    } else if (type === "IDAT") {
      idat.push(buffer.subarray(dataStart, dataEnd));
    } else if (type === "IEND") {
      break;
    }

    offset = dataEnd + 4;
  }

  const result = { ...base, width, height };
  if (!width || !height) {
    result.notes.push("PNG dimensions unavailable.");
    return result;
  }

  if (bitDepth !== 8 || ![2, 6].includes(colorType) || interlace !== 0 || !idat.length) {
    result.notes.push("PNG color analysis supports only non-interlaced 8-bit RGB/RGBA images.");
    return result;
  }

  const channels = colorType === 6 ? 4 : 3;
  const rowBytes = width * channels;
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const pixels = Buffer.alloc(width * height * channels);
  let rawOffset = 0;
  let pixelOffset = 0;
  let previousRow = Buffer.alloc(rowBytes);

  for (let y = 0; y < height; y += 1) {
    const filter = raw[rawOffset];
    rawOffset += 1;
    const row = Buffer.from(raw.subarray(rawOffset, rawOffset + rowBytes));
    rawOffset += rowBytes;
    unfilterRow(row, previousRow, filter, channels);
    row.copy(pixels, pixelOffset);
    pixelOffset += rowBytes;
    previousRow = row;
  }

  const signals = samplePixels(pixels, width, height, channels);
  return {
    ...result,
    analyzedPixels: signals.analyzedPixels,
    signals: omitAnalyzedCount(signals),
    notes: ["PNG pixels sampled on the backend."],
  };
}

function unfilterRow(row, previousRow, filter, bytesPerPixel) {
  for (let i = 0; i < row.length; i += 1) {
    const left = i >= bytesPerPixel ? row[i - bytesPerPixel] : 0;
    const up = previousRow[i] || 0;
    const upLeft = i >= bytesPerPixel ? previousRow[i - bytesPerPixel] || 0 : 0;
    if (filter === 1) row[i] = (row[i] + left) & 255;
    else if (filter === 2) row[i] = (row[i] + up) & 255;
    else if (filter === 3) row[i] = (row[i] + Math.floor((left + up) / 2)) & 255;
    else if (filter === 4) row[i] = (row[i] + paeth(left, up, upLeft)) & 255;
  }
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function samplePixels(pixels, width, height, channels) {
  const strideX = Math.max(1, Math.floor(width / 80));
  const strideY = Math.max(1, Math.floor(height / 80));
  let counted = 0;
  let greenPixels = 0;
  let redPixels = 0;
  let brightPixels = 0;
  let darkPixels = 0;
  let saturationTotal = 0;
  let greenLeadTotal = 0;

  for (let y = 0; y < height; y += strideY) {
    for (let x = 0; x < width; x += strideX) {
      const index = (y * width + x) * channels;
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      const alpha = channels === 4 ? pixels[index + 3] : 255;
      if (alpha < 20) continue;

      const max = Math.max(red, green, blue);
      const min = Math.min(red, green, blue);
      const brightness = (red + green + blue) / 3;
      const saturation = max === 0 ? 0 : (max - min) / max;

      if (green > red + 18 && green > blue + 10) greenPixels += 1;
      if (red > green + 25 && red > blue + 18) redPixels += 1;
      if (brightness > 202) brightPixels += 1;
      if (brightness < 68) darkPixels += 1;
      greenLeadTotal += green - Math.max(red, blue);
      saturationTotal += saturation;
      counted += 1;
    }
  }

  if (!counted) {
    return {
      analyzedPixels: 0,
      greenRatio: 0,
      redRatio: 0,
      brightRatio: 0,
      darkRatio: 0,
      saturation: 0,
      greenLead: 0,
    };
  }

  return {
    analyzedPixels: counted,
    greenRatio: round(greenPixels / counted),
    redRatio: round(redPixels / counted),
    brightRatio: round(brightPixels / counted),
    darkRatio: round(darkPixels / counted),
    saturation: round(saturationTotal / counted),
    greenLead: round(greenLeadTotal / counted),
  };
}

function omitAnalyzedCount(signals) {
  const { analyzedPixels, ...rest } = signals;
  return rest;
}

function readJpegSize(buffer) {
  if (buffer[0] !== 0xff || buffer[1] !== 0xd8) return {};
  let offset = 2;

  while (offset + 9 < buffer.length) {
    if (buffer[offset] !== 0xff) {
      offset += 1;
      continue;
    }
    const marker = buffer[offset + 1];
    const length = buffer.readUInt16BE(offset + 2);
    if (marker >= 0xc0 && marker <= 0xc3) {
      return { height: buffer.readUInt16BE(offset + 5), width: buffer.readUInt16BE(offset + 7) };
    }
    offset += 2 + length;
  }

  return {};
}

function readWebpSize(buffer) {
  if (buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WEBP") {
    return {};
  }
  const chunk = buffer.toString("ascii", 12, 16);
  if (chunk === "VP8X" && buffer.length >= 30) {
    return {
      width: 1 + buffer.readUIntLE(24, 3),
      height: 1 + buffer.readUIntLE(27, 3),
    };
  }
  return {};
}

function pickWinner(scores) {
  return Object.entries(scores).sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return CATEGORY_ORDER.indexOf(a[0]) - CATEGORY_ORDER.indexOf(b[0]);
  })[0][0];
}

function pickRunnerUp(scores, winner) {
  return Object.entries(scores)
    .filter(([key]) => key !== winner)
    .sort((a, b) => b[1] - a[1])[0][0];
}

function computeConfidence(winnerScore, runnerScore, evidence, imageInfo) {
  const gap = Math.max(0, winnerScore - runnerScore);
  let confidence = 50 + gap * 2.5 + Math.min(14, evidence.length * 3);
  if (imageInfo.analyzedPixels) confidence += 4;
  return Math.max(50, Math.min(96, Math.round(confidence)));
}

function normalizeText(value) {
  return String(value || "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function containsCjkOrThai(value) {
  return /[\u0E00-\u0E7F\u3400-\u9FFF]/.test(value);
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}

function createHttpError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

module.exports = {
  classifyWaste,
  normalizeInput,
};
