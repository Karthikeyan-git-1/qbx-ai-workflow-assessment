const { runAgentForItem } = require("../src/agent");
const tickets = require("../data/tickets.json");

test("T5: malformed LLM output triggers retry and eventually rejects cleanly", async () => {
  const t5 = tickets.find((t) => t.id === "T5");

  const result = await runAgentForItem(t5, { maxToolCalls: 3, maxLlmAttempts: 3 });

  expect(result.id).toBe("T5");
  expect(result.status).toBe("REJECTED");
  expect(result.final.action).toBe("REFUSE");
  expect(result.final.payload).toHaveProperty("reason");
});