"use client";

import type { AnalysisResponse } from "../lib/api";

export default function ReportPanel({ result }: { result: AnalysisResponse | null }) {
  if (!result) {
    return <p className="placeholder">The draft report will appear here.</p>;
  }

  return (
    <div>
      <div className="report">{result.report}</div>
      <p className="meta-line">
        backbone: {result.backbone} · explain: {result.explain_method} · report
        backend: {result.report_backend}
      </p>
      <p className="disclaimer">
        Research prototype output. This draft is machine-generated and must be
        reviewed and signed by a licensed radiologist before any clinical use.
      </p>
    </div>
  );
}
