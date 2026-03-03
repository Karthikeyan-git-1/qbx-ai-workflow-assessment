/**
 * Expected LLM response shapes:
 *
 * Tool call:
 * {
 *   type: "tool_call",
 *   tool: string,
 *   args: object
 * }
 *
 * Final:
 * {
 *   type: "final",
 *   final: {
 *     action: "SEND_EMAIL_DRAFT" | "REQUEST_INFO" | "REFUSE",
 *     payload: object
 *   }
 * }
 */

/**
 * Validates tool_call response
 */
function isValidToolCall(obj) {
  return (
    obj &&
    obj.type === "tool_call" &&
    typeof obj.tool === "string" &&
    obj.tool.length > 0 &&
    typeof obj.args === "object" &&
    obj.args !== null
  );
}

/**
 * Validates final response
 */
function isValidFinal(obj) {
  if (!obj || obj.type !== "final") return false;
  if (!obj.final || typeof obj.final !== "object") return false;

  const allowedActions = ["SEND_EMAIL_DRAFT", "REQUEST_INFO", "REFUSE"];

  return (
    typeof obj.final.action === "string" &&
    allowedActions.includes(obj.final.action) &&
    typeof obj.final.payload === "object" &&
    obj.final.payload !== null
  );
}

/**
 * Safely parse JSON string
 */
function safeParse(jsonStr) {
  try {
    return { ok: true, value: JSON.parse(jsonStr) };
  } catch (err) {
    return { ok: false, error: err };
  }
}

module.exports = {
  isValidToolCall,
  isValidFinal,
  safeParse
};