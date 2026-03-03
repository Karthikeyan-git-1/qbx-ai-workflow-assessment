const fs = require("fs");
const path = require("path");
const { runWorkflow } = require("./n8n/engine");

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--workflow") out.workflow = argv[++i];
    if (a === "--in") out.in = argv[++i];
    if (a === "--out") out.out = argv[++i];
  }
  if (!out.workflow || !out.in || !out.out) {
    throw new Error(
      "Usage: node src/index.js --workflow workflow/workflow.json --in data/tickets.json --out output/result.json"
    );
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv);

  const workflowPath = path.resolve(args.workflow);
  const inputPath = path.resolve(args.in);
  const outPath = path.resolve(args.out);

  const workflow = JSON.parse(fs.readFileSync(workflowPath, "utf8"));
  const tickets = JSON.parse(fs.readFileSync(inputPath, "utf8"));

  // n8n-style items
  const items = tickets.map((t) => ({ json: t }));

  const resultItems = await runWorkflow(workflow, items);

  // Convention: final node returns items where json is a result object
  // We aggregate into { results: [...] }
  const results = resultItems.map((it) => it.json);

  const output = { results };

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2), "utf8");
  console.log(`Wrote ${outPath}`);
}

main().catch((e) => {
  console.error(e.stack || String(e));
  process.exit(1);
});