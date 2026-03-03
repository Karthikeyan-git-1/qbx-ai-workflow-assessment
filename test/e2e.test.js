const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

test("end-to-end: npm run start produces output/result.json with expected shape", () => {
  const outPath = path.resolve("output/result.json");
  if (fs.existsSync(outPath)) fs.unlinkSync(outPath);

  execFileSync(
    "node",
    ["src/index.js", "--workflow", "workflow/workflow.json", "--in", "data/tickets.json", "--out", "output/result.json"],
    { stdio: "inherit" }
  );

  const raw = fs.readFileSync(outPath, "utf8");
  const obj = JSON.parse(raw);

  expect(obj).toHaveProperty("results");
  expect(Array.isArray(obj.results)).toBe(true);
  expect(obj.results.length).toBeGreaterThanOrEqual(1);

  // Each result must have required fields
  for (const r of obj.results) {
    expect(r).toHaveProperty("id");
    expect(r).toHaveProperty("status");
    expect(r).toHaveProperty("plan");
    expect(r).toHaveProperty("tool_calls");
    expect(r).toHaveProperty("final");
    expect(r).toHaveProperty("safety");
  }
});