/**
 * resolveExpression(expr, ctx)
 *
 * ctx = {
 *   currentItem: { json: {...} },
 *   nodeOutputs: { [nodeName]: [ { json: {...} }, ... ] }
 * }
 */
function resolveExpression(expr, ctx) {
  if (typeof expr !== "string") return expr;

  const trimmed = expr.trim();
  if (!trimmed.startsWith("{{") || !trimmed.endsWith("}}")) return expr;

  const inner = trimmed.slice(2, -2).trim();

  // {{$json.foo.bar}}
  if (inner.startsWith("$json.")) {
    const path = inner.slice("$json.".length);
    return getPath(ctx.currentItem?.json, path);
  }

  // {{$node["X"].json.foo.bar}}
  // Very small parser, assumes valid format.
  if (inner.startsWith('$node["')) {
    const after = inner.slice('$node["'.length);
    const endNameIdx = after.indexOf('"]');
    if (endNameIdx < 0) throw new Error(`Invalid node expression: ${expr}`);

    const nodeName = after.slice(0, endNameIdx);
    const rest = after.slice(endNameIdx + 2).trim(); // starts with .json....

    if (!rest.startsWith(".json.")) throw new Error(`Invalid node expression: ${expr}`);
    const path = rest.slice(".json.".length);

    const items = ctx.nodeOutputs?.[nodeName];
    if (!items || items.length === 0) return undefined;
    return getPath(items[0].json, path);
  }

  throw new Error(`Unsupported expression: ${expr}`);
}

function getPath(obj, path) {
  if (!obj) return undefined;
  const parts = path.split(".").filter(Boolean);
  let cur = obj;
  for (const p of parts) {
    if (cur == null) return undefined;
    cur = cur[p];
  }
  return cur;
}

module.exports = {
  resolveExpression
};