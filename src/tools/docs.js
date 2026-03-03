/**
 * Deterministic document fixtures used by lookupDoc tool
 */

const DOCS = {
  "RPT-2026-02": `
Quarterly Finance Report - February 2026

Revenue increased by 12% compared to previous quarter.
Operating expenses reduced by 4%.
Net margin improved to 18%.
Key drivers included subscription growth and cost optimization.
  `.trim(),

  "PUBLIC-GUIDE-001": `
Public Operations Guide

All operational emails must follow compliance standards.
Do not include confidential information without approval.
  `.trim()
};

function getDoc(docId) {
  return DOCS[docId] || null;
}

module.exports = {
  getDoc
};