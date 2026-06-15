const { classifyWasteWithHuggingFace } = require("./lib/huggingFaceWasteClassifier");
const { classifyWasteWithOpenAI } = require("./lib/openaiWasteClassifier");

module.exports = async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ ok: false, error: "Use POST /api/classify." });
    return;
  }

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body;
    const result = await classifyWaste(body);
    res.status(200).json(result);
  } catch (error) {
    res.status(error.statusCode || 500).json({
      ok: false,
      error: error.message || "Classification failed.",
    });
  }
};

async function classifyWaste(body) {
  const provider = String(
    process.env.WASTE_CLASSIFIER_PROVIDER || process.env.CLASSIFIER_PROVIDER || "huggingface-space"
  )
    .trim()
    .toLowerCase();

  if (provider === "openai") {
    return classifyWasteWithOpenAI(body);
  }

  if (provider === "auto") {
    try {
      return await classifyWasteWithHuggingFace(body);
    } catch (error) {
      if (!process.env.OPENAI_API_KEY) throw error;
      return classifyWasteWithOpenAI(body);
    }
  }

  return classifyWasteWithHuggingFace(body);
}

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}
