const { classifyWaste } = require("./lib/thaiWasteClassifier");

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
    const result = classifyWaste(body);
    res.status(200).json(result);
  } catch (error) {
    res.status(error.statusCode || 500).json({
      ok: false,
      error: error.message || "Classification failed.",
    });
  }
};

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}
