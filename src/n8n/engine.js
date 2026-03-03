const { runAgentForItem } = require("../agent");

/**
 * runWorkflow(workflow, items)
 *
 * workflow: parsed workflow JSON
 * items: array of n8n-style items [{ json: {...} }]
 *
 * Returns final items after executing nodes.
 */
async function runWorkflow(workflow, items) {
  const nodesByName = {};
  for (const node of workflow.nodes) {
    nodesByName[node.name] = node;
  }

  const connections = workflow.connections;

  // Simple linear traversal based on connections
  // Assumes single path Start -> ... -> Done
  let currentNodeName = "Start";
  let currentItems = items;

  // Track node outputs (for possible expression resolution later)
  const executionData = {};

  while (currentNodeName) {
    const node = nodesByName[currentNodeName];
    if (!node) {
      throw new Error(`Node not found: ${currentNodeName}`);
    }

    switch (node.type) {
      case "Start":
        executionData[node.name] = currentItems;
        break;

      case "Agent":
        currentItems = await executeAgentNode(node, currentItems);
        executionData[node.name] = currentItems;
        break;

      case "Done":
        executionData[node.name] = currentItems;
        return currentItems;

      default:
        throw new Error(`Unsupported node type: ${node.type}`);
    }

    const next = connections[node.name];
    if (!next || next.length === 0) {
      return currentItems;
    }

    // For simplicity, assume single next node
    currentNodeName = next[0];
  }

  return currentItems;
}

/**
 * Executes Agent node for each item independently
 */
async function executeAgentNode(node, items) {
  const out = [];

  for (const item of items) {
    const result = await runAgentForItem(item.json, {
      maxToolCalls: node.parameters?.maxToolCalls || 3,
      maxLlmAttempts: node.parameters?.maxLlmAttempts || 3
    });

    out.push({ json: result });
  }

  return out;
}

module.exports = {
  runWorkflow
};