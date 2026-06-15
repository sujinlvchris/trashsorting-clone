const { OFFICIAL_REFERENCE, THAI_WASTE_RULES } = require("./thaiWasteRules");
const { normalizeInput } = require("./imageInput");

const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_MODEL = "gpt-5.5";
const BIN_KEYS = ["green", "yellow", "blue", "red"];
const OPENAI_TIMEOUT_MS = Number(process.env.OPENAI_TIMEOUT_MS || 45000);

async function classifyWasteWithOpenAI(input) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return handleOpenAIFailure("OPENAI_API_KEY is not configured.", 503);
  }

  const normalized = normalizeInput(input);
  const model = process.env.OPENAI_WASTE_MODEL || DEFAULT_MODEL;
  const responsesUrl = buildResponsesUrl();
  const imageUrl = buildImageDataUrl(input.image, normalized.mimeType);
  const payload = buildOpenAIRequest({ input, imageUrl, model, normalized });

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), OPENAI_TIMEOUT_MS);

  try {
    const response = await fetch(responsesUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify(payload),
    });

    const responseText = await response.text();
    const contentType = response.headers.get("content-type") || "";

    if (!response.ok) {
      const message = extractErrorMessage(responseText) || `OpenAI API returned HTTP ${response.status}.`;
      throw new Error(message);
    }

    const output = extractResponseOutput(responseText, contentType);
    const parsed = parseJson(output.text, "OpenAI output was not valid JSON.");
    return buildResultFromOpenAI(parsed, {
      imageInfo: {},
      model,
      normalized,
      responseId: output.responseId,
    });
  } catch (error) {
    return handleOpenAIFailure(
      error.message || "OpenAI API classification failed.",
      error.name === "AbortError" ? 504 : 502
    );
  } finally {
    clearTimeout(timeout);
  }
}

function buildOpenAIRequest({ input, imageUrl, model, normalized }) {
  const hintText = [input.fileName || input.filename, input.hint, input.userLabel]
    .filter(Boolean)
    .join(" | ");

  const request = {
    model,
    input: [
      {
        role: "system",
        content:
          "You are the final OpenAI vision judge for Thailand/Bangkok waste disposal. " +
          "Use the image as the primary source, identify the visible item and material, then give one final answer. " +
          "Choose exactly one Thai municipal stream: green food/organic, yellow recyclable, blue general, or red hazardous. " +
          "Use Thai/Bangkok sorting context: food scraps and plant matter are green; clean recyclable containers, paper, cardboard, glass, and metal are yellow; non-recyclable non-hazardous mixed waste is blue; batteries, bulbs, chemicals, medicine, aerosols, needles, electronics, and toxic or hazardous containers are red. " +
          "The server will not override or correct your bin choice, so make the best final judgment from the visual evidence. " +
          "If visual evidence is weak, lower confidence and set needsManualCheck true.",
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text:
              `Source: ${normalized.source}. File/hint: ${hintText || "none"}. ` +
              "Use the image as the primary source. Identify the visible item and material as precisely as possible. " +
              "Return your final answer for where this item should be disposed in Thailand/Bangkok sorting. " +
              "Do not rely on filename or hint if it conflicts with the image.",
          },
          {
            type: "input_image",
            image_url: imageUrl,
            detail: process.env.OPENAI_IMAGE_DETAIL || "original",
          },
        ],
      },
    ],
    max_output_tokens: 700,
    stream: process.env.OPENAI_STREAM === "0" ? false : true,
    text: {
      format: {
        type: "json_schema",
        name: "thai_waste_classification",
        strict: true,
        schema: {
          type: "object",
          additionalProperties: false,
          required: [
            "bin",
            "itemName",
            "material",
            "recognizedItemType",
            "isRecyclableContainer",
            "isHazardous",
            "isFoodOrOrganic",
            "isGeneralOnly",
            "confidence",
            "needsManualCheck",
            "evidence",
            "disposalNote",
          ],
          properties: {
            bin: {
              type: "string",
              enum: BIN_KEYS,
              description: "green, yellow, blue, or red Thailand waste bin key.",
            },
            itemName: {
              type: "string",
              description: "Short visible item name in English.",
            },
            material: {
              type: "string",
              description: "Short material description, such as food waste, plastic bottle, tissue, battery, or mixed packaging.",
            },
            recognizedItemType: {
              type: "string",
              description:
                "Normalized visible item type, for example plastic_bottle, glass_bottle, metal_can, cardboard, food_scrap, wrapper, plastic_bag, tissue, battery, bulb, chemical, electronics, or unknown.",
            },
            isRecyclableContainer: {
              type: "boolean",
              description: "True for plastic bottles, PET bottles, water bottles, beverage bottles, glass bottles, jars, metal cans, clean paper, cardboard, and cartons.",
            },
            isHazardous: {
              type: "boolean",
              description: "True for batteries, bulbs, chemicals, medicine, aerosols, needles, electronics, and similar hazardous waste.",
            },
            isFoodOrOrganic: {
              type: "boolean",
              description: "True for food scraps, fruit peels, vegetable scraps, leaves, eggshells, and similar organic waste.",
            },
            isGeneralOnly: {
              type: "boolean",
              description: "True only for non-recyclable, non-hazardous general waste such as dirty wrappers, plastic bags, tissues, foam boxes, masks, diapers, ceramics, or mixed waste.",
            },
            confidence: {
              type: "integer",
              minimum: 0,
              maximum: 100,
            },
            needsManualCheck: {
              type: "boolean",
            },
            evidence: {
              type: "array",
              minItems: 1,
              maxItems: 4,
              items: { type: "string" },
            },
            disposalNote: {
              type: "string",
              description: "One concise Thailand-specific disposal instruction.",
            },
          },
        },
      },
    },
  };

  if (process.env.OPENAI_REASONING_EFFORT) {
    request.reasoning = { effort: process.env.OPENAI_REASONING_EFFORT };
  }

  if (process.env.OPENAI_TEXT_VERBOSITY) {
    request.text.verbosity = process.env.OPENAI_TEXT_VERBOSITY;
  }

  return request;
}

function buildImageDataUrl(image, mimeType) {
  const value = String(image || "");
  if (value.startsWith("data:")) return value;
  return `data:${mimeType};base64,${value}`;
}

function buildResponsesUrl() {
  const baseUrl = String(process.env.OPENAI_BASE_URL || DEFAULT_OPENAI_BASE_URL).replace(/\/+$/, "");
  return `${baseUrl}/responses`;
}

function buildResultFromOpenAI(parsed, context) {
  const itemName = String(parsed.itemName || "Unclear item").trim();
  const material = String(parsed.material || "Unknown material").trim();
  const recognizedItemType = String(parsed.recognizedItemType || "").trim();
  const evidence = sanitizeStringArray(parsed.evidence, 4);
  const bin = normalizeBin(parsed.bin);
  if (!bin) {
    throw createHttpError(502, "OpenAI output did not include a valid Thailand bin.");
  }

  const rule = THAI_WASTE_RULES[bin];
  const confidence = clampPercent(parsed.confidence);
  const disposalNote = String(parsed.disposalNote || rule.disposal).trim();

  return {
    ok: true,
    algorithmVersion: "openai-responses-vision-direct-v1",
    classifier: "openai-direct",
    model: context.model,
    openaiResponseId: context.responseId || null,
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
    disposal: disposalNote || rule.disposal,
    avoid: rule.avoid,
    thailandTips: rule.thailandTips,
    officialBasis: rule.officialBasis,
    confidence,
    needsManualCheck: Boolean(parsed.needsManualCheck) || confidence < 70,
    scores: { [bin]: confidence },
    evidence: [
      `OpenAI final answer: ${rule.label}.`,
      `OpenAI vision identified: ${itemName}${material ? ` (${material})` : ""}.`,
      ...evidence,
    ],
    image: {
      ...context.imageInfo,
      source: context.normalized.source,
      mimeType: context.normalized.mimeType,
      bytes: context.normalized.buffer.length,
    },
    ai: {
      itemName,
      material,
      modelBin: bin,
      finalBin: bin,
      recognizedItemType,
      isRecyclableContainer: Boolean(parsed.isRecyclableContainer),
      isHazardous: Boolean(parsed.isHazardous),
      isFoodOrOrganic: Boolean(parsed.isFoodOrOrganic),
      isGeneralOnly: Boolean(parsed.isGeneralOnly),
      provider: "openai",
    },
  };
}

function normalizeBin(value) {
  const bin = String(value || "").trim().toLowerCase();
  return BIN_KEYS.includes(bin) ? bin : "";
}

function extractResponseOutput(responseText, contentType) {
  if (contentType.includes("text/event-stream") || /^\s*(event:|data:)/.test(responseText)) {
    return extractOutputFromSse(responseText);
  }

  const data = parseJson(responseText, "OpenAI response was not valid JSON.");
  return {
    text: extractOutputText(data),
    responseId: data.id || null,
  };
}

function extractOutputFromSse(responseText) {
  let responseId = null;
  let deltaText = "";
  let doneText = "";
  let completedResponse = null;

  for (const event of parseSseEvents(responseText)) {
    if (!event.data || event.data === "[DONE]") continue;

    const payload = parseJson(event.data, "OpenAI stream event was not valid JSON.");
    const type = payload.type || event.event;

    if (payload.response?.id) responseId = payload.response.id;
    if (payload.id) responseId = payload.id;
    if (typeof payload.output_text === "string" && payload.output_text.trim()) doneText = payload.output_text;

    if (type === "response.failed") {
      throw new Error(payload.response?.error?.message || payload.error?.message || "OpenAI streaming response failed.");
    }

    if (type === "response.incomplete") {
      throw new Error(payload.response?.incomplete_details?.reason || "OpenAI streaming response was incomplete.");
    }

    if (type === "response.output_text.delta" && typeof payload.delta === "string") {
      deltaText += payload.delta;
    }

    if (type === "response.output_text.done" && typeof payload.text === "string") {
      doneText = payload.text;
    }

    if (type === "response.completed" && payload.response) {
      completedResponse = payload.response;
    }
  }

  const text = doneText || deltaText || (completedResponse ? extractOutputText(completedResponse) : "");
  if (!text.trim()) throw new Error("OpenAI streaming response did not include output text.");

  return {
    text,
    responseId: responseId || completedResponse?.id || null,
  };
}

function parseSseEvents(responseText) {
  return responseText
    .split(/\r?\n\r?\n/)
    .map((block) => {
      const event = { event: "", data: "" };
      const dataLines = [];

      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) event.event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }

      event.data = dataLines.join("\n").trim();
      return event;
    })
    .filter((event) => event.event || event.data);
}

function extractErrorMessage(responseText) {
  try {
    const data = parseJson(responseText, "Invalid error JSON.");
    return data?.error?.message || data?.message || "";
  } catch {
    try {
      for (const event of parseSseEvents(responseText)) {
        if (!event.data || event.data === "[DONE]") continue;
        const data = parseJson(event.data, "Invalid error event JSON.");
        if (data?.error?.message) return data.error.message;
        if (data?.response?.error?.message) return data.response.error.message;
      }
    } catch {
      // Fall through to a short plain-text error.
    }
  }

  return responseText.replace(/\s+/g, " ").trim().slice(0, 240);
}

function handleOpenAIFailure(reason, statusCode) {
  throw createHttpError(
    statusCode,
    `OpenAI direct answer is unavailable: ${reason}. Configure OPENAI_API_KEY and OPENAI_BASE_URL.`
  );
}

function createHttpError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

function extractOutputText(data) {
  if (typeof data.output_text === "string" && data.output_text.trim()) {
    return data.output_text;
  }

  for (const item of data.output || []) {
    if (item.type !== "message" || !Array.isArray(item.content)) continue;
    for (const content of item.content) {
      if (content.type === "refusal") {
        throw new Error(content.refusal || "OpenAI refused to classify the image.");
      }
      if (typeof content.text === "string" && content.text.trim()) return content.text;
    }
  }

  throw new Error("OpenAI response did not include output text.");
}

function parseJson(value, message) {
  try {
    return JSON.parse(value);
  } catch {
    throw new Error(message);
  }
}

function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 50;
  return Math.max(0, Math.min(100, Math.round(number)));
}

function sanitizeStringArray(items, maxItems) {
  if (!Array.isArray(items)) return [];
  return items
    .map((item) => String(item || "").trim())
    .filter(Boolean)
    .slice(0, maxItems);
}

module.exports = {
  classifyWasteWithOpenAI,
  buildResultFromOpenAI,
  buildResponsesUrl,
};
