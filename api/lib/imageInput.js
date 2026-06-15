const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
const SUPPORTED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

function normalizeInput(input) {
  if (!input || typeof input !== "object") {
    throw createHttpError(400, "Expected JSON body.");
  }

  const image = String(input.image || "");
  const mimeType = String(input.mimeType || "").toLowerCase();
  const fileName = String(input.fileName || input.filename || "");
  const hint = String(input.hint || "");
  const userLabel = String(input.userLabel || "");
  const source = normalizeSource(input.source);

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

  return { buffer, fileName, hint, mimeType, source, userLabel };
}

function normalizeSource(source) {
  const value = String(source || "upload").toLowerCase();
  return ["camera", "upload", "test"].includes(value) ? value : "upload";
}

function createHttpError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

module.exports = {
  normalizeInput,
};
