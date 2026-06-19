"use client";

import type { Finding } from "../lib/api";

export default function FindingsList({ findings }: { findings: Finding[] }) {
  if (findings.length === 0) {
    return <p className="placeholder">No predictions yet.</p>;
  }

  return (
    <div>
      {findings.map((f) => (
        <div className="finding" key={f.label}>
          <div className="finding-top">
            <span className={`finding-name ${f.present ? "present" : "absent"}`}>
              {f.label.replace(/_/g, " ")}
            </span>
            <span className="finding-prob">{(f.probability * 100).toFixed(1)}%</span>
          </div>
          {f.present && f.location !== "n/a" && (
            <div className="finding-loc">localised to {f.location}</div>
          )}
          <div className="bar">
            <span style={{ width: `${Math.round(f.probability * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
