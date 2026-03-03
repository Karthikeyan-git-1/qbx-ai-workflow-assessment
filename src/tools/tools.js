const { getDoc } = require("./docs");

/**
 * lookupDoc({ docId })
 * Returns document text if exists, else throws error.
 */
function lookupDoc({ docId }) {
  if (!docId) {
    throw new Error("lookupDoc requires docId");
  }

  const doc = getDoc(docId);
  if (!doc) {
    throw new Error(`Document not found: ${docId}`);
  }

  return {
    docId,
    content: doc
  };
}

/**
 * summarize({ text, bullets })
 * Deterministic "summary": returns first N sentences as bullet points.
 */
function summarize({ text, bullets }) {
  if (typeof text !== "string") {
    throw new Error("summarize requires text");
  }

  const count = typeof bullets === "number" && bullets > 0 ? bullets : 3;

  const sentences = text
    .split(".")
    .map((s) => s.trim())
    .filter(Boolean);

  const selected = sentences.slice(0, count);

  return {
    bullets: selected.map((s) => `- ${s}.`)
  };
}

/**
 * sendEmailDraft({ to, subject, body })
 * Returns deterministic draft object.
 */
function sendEmailDraft({ to, subject, body }) {
  if (!Array.isArray(to) || to.length === 0) {
    throw new Error("sendEmailDraft requires non-empty 'to' array");
  }

  return {
    draftId: "DRAFT-STATIC-ID",
    to,
    subject: subject || "",
    body: body || ""
  };
}

/**
 * Registry of available tools
 */
const TOOL_REGISTRY = {
  lookupDoc,
  summarize,
  sendEmailDraft
};

module.exports = {
  TOOL_REGISTRY
};