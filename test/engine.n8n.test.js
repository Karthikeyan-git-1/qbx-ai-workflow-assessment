const { runWorkflow } = require("../src/n8n/engine");

test("n8n-style engine runs Start -> Agent -> Done and returns one result per input item", async () => {
  const workflow = {
    nodes: [
      { id: "start", name: "Start", type: "Start" },
      { id: "agent", name: "Agent", type: "Agent", parameters: { maxToolCalls: 3, maxLlmAttempts: 3 } },
      { id: "done", name: "Done", type: "Done" }
    ],
    connections: {
      Start: ["Agent"],
      Agent: ["Done"]
    }
  };

  const items = [
    { json: { id: "X1", user_request: "Can you help me?", context: { allowed_tools: [], policy: "NONE" } } },
    { json: { id: "X2", user_request: "Can you help me?", context: { allowed_tools: [], policy: "NONE" } } }
  ];

  const out = await runWorkflow(workflow, items);
  expect(Array.isArray(out)).toBe(true);
  expect(out).toHaveLength(2);

  // Each output item should be { json: result }
  expect(out[0]).toHaveProperty("json");
  expect(out[0].json).toHaveProperty("id");
});