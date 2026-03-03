const { runAgentForItem } = require("../src/agent");
const tickets = require("../data/tickets.json");

test("T1: agent uses lookupDoc and produces SEND_EMAIL_DRAFT", async () => {
  const t1 = tickets.find((t) => t.id === "T1");

  const result = await runAgentForItem(t1, { maxToolCalls: 3, maxLlmAttempts: 3 });

  expect(result.id).toBe("T1");
  expect(result.status).toBe("DONE");

  // Must call lookupDoc for 'latest report'
  expect(result.tool_calls.some((c) => c.tool === "lookupDoc")).toBe(true);

  expect(result.final).toHaveProperty("action", "SEND_EMAIL_DRAFT");
  expect(result.final).toHaveProperty("payload");
  expect(result.final.payload).toHaveProperty("to");
});

test("T6: agent returns REQUEST_INFO by default", async () => {
  const t6 = tickets.find((t) => t.id === "T6");

  const result = await runAgentForItem(t6, { maxToolCalls: 3, maxLlmAttempts: 3 });

  expect(result.id).toBe("T6");
  expect(result.status).toBe("DONE");
  expect(result.final.action).toBe("REQUEST_INFO");
});