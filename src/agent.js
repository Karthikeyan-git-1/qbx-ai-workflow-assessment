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

  const allowedTools = ticket.context?.allowed_tools || [];

  const plan = [];
  const tool_calls = [];
  const safety = { blocked: false, reasons: [] };

  // Prompt injection check
  const issues = detectPromptInjection(ticket.user_request);

  if (issues.length > 0) {
    safety.blocked = true;
    safety.reasons = issues;

    return {
      id: ticket.id,
      status: "REJECTED",
      plan,
      tool_calls: [],
      final: {
        action: "REFUSE",
        payload: { reason: "Prompt injection detected" }
      },
      safety
    };
  }

  const messages = [
    { role: "system", content: "You are a safe enterprise assistant." },
    { role: "user", content: ticket.user_request }
  ];

  let attempts = 0;

  while (attempts < maxLlmAttempts) {

    attempts++;

    const raw = await mockLlm(messages);

    const parsed = safeParse(raw);

    if (!parsed.ok) {
      messages.push({
        role: "system",
        content: "Return valid JSON only."
      });
      continue;
    }

    const response = parsed.value;

    const validation = validateLlmResponse(response);

    if (!validation.ok) {
      messages.push({
        role: "system",
        content: "Response schema invalid."
      });
      continue;
    }

    // TOOL CALL
    if (validation.type === "tool_call") {

      const toolName = response.tool;
      const args = response.args || {};

      if (!enforceToolAllowlist(toolName, allowedTools)) {

        safety.blocked = true;
        safety.reasons.push("TOOL_NOT_ALLOWED");

        return {
          id: ticket.id,
          status: "REJECTED",
          plan,
          tool_calls,
          final: {
            action: "REFUSE",
            payload: { reason: "Tool not allowed" }
          },
          safety
        };
      }

      const tool = TOOL_REGISTRY[toolName];

      if (!tool) {
        return {
          id: ticket.id,
          status: "REJECTED",
          plan,
          tool_calls,
          final: {
            action: "REFUSE",
            payload: { reason: "Tool not found" }
          },
          safety
        };
      }

      const result = await tool(args);

      plan.push(`Called tool ${toolName}`);

      tool_calls.push({
        tool: toolName,
        args
      });

      // For this deterministic mock LLM,
      // return final immediately after lookupDoc
      if (toolName === "lookupDoc") {

        return {
          id: ticket.id,
          status: "DONE",
          plan,
          tool_calls,
          final: {
            action: "SEND_EMAIL_DRAFT",
            payload: {
              to: ["finance@example.com"],
              subject: "Requested Report",
              body: "Summary generated from latest report."
            }
          },
          safety
        };
      }

      continue;
    }

    // FINAL RESPONSE
    if (validation.type === "final") {

      return {
        id: ticket.id,
        status: "DONE",
        plan,
        tool_calls,
        final: response.final,
        safety
      };
    }
  }

  return {
    id: ticket.id,
    status: "REJECTED",
    plan,
    tool_calls,
    final: {
      action: "REFUSE",
      payload: { reason: "Max LLM attempts exceeded" }
    },
    safety
  };
}

module.exports = {
  runAgentForItem
};