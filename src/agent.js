const { mockLlm } = require("./llm/mockLlm");
const { safeParse } = require("./llm/schema");
const { TOOL_REGISTRY } = require("./tools/tools");
const {
  detectPromptInjection,
  enforceToolAllowlist,
  validateLlmResponse
} = require("./guardrails");

/**
 * runAgentForItem(ticket, config)
 *
 * config:
 *  - maxToolCalls
 *  - maxLlmAttempts
 *
 * Must return:
 * {
 *   id,
 *   status: "DONE" | "NEEDS_CLARIFICATION" | "REJECTED",
 *   plan: string[],
 *   tool_calls: { tool: string, args: object }[],
 *   final: { action: "SEND_EMAIL_DRAFT" | "REQUEST_INFO" | "REFUSE", payload: object },
 *   safety: { blocked: boolean, reasons: string[] }
 * }
 *
 * Behavior enforced by tests:
 * - Prompt injection in ticket.user_request => REJECTED, safety.blocked true, tool_calls []
 * - If mock LLM requests a tool not in allowed_tools => REJECTED
 * - For "latest report" requests => must execute lookupDoc at least once, then DONE with SEND_EMAIL_DRAFT
 * - For default ("Can you help me...") => DONE with REQUEST_INFO
 * - For MALFORMED ticket => retry parsing; ultimately REJECTED cleanly
 *
 * Bounded:
 * - max tool calls per ticket: config.maxToolCalls
 * - max LLM attempts per ticket: config.maxLlmAttempts
 */
async function runAgentForItem(ticket, config) {
  const maxToolCalls = config?.maxToolCalls ?? 3;
  const maxLlmAttempts = config?.maxLlmAttempts ?? 3;

  const plan = [];
  const tool_calls = [];
  const safety = { blocked: false, reasons: [] };

  // TODO 1: prompt injection detection
  // If detected: return REJECTED before calling LLM or tools.

  // TODO 2: build initial messages array
  // Must include system + user message
  const messages = [];

  // TODO 3: agent loop (attempts bounded)
  // - call mockLlm(messages)
  // - safeParse
  // - validateLlmResponse
  // - if tool_call:
  //    - enforce allowlist
  //    - execute tool
  //    - push TOOL_RESULT: ... into messages
  // - if final: return DONE with final
  // - if malformed JSON: retry with stricter system message once (within max attempts)

  return {
    id: ticket.id,
    status: "REJECTED",
    plan: ["Not implemented"],
    tool_calls: [],
    final: { action: "REFUSE", payload: { reason: "Not implemented" } },
    safety
  };
}

module.exports = {
  runAgentForItem
};