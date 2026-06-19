"use client";

import { useState } from "react";
import UploadPanel from "../components/UploadPanel";
import FilmViewer from "../components/FilmViewer";
import FindingsList from "../components/FindingsList";
import ReportPanel from "../components/ReportPanel";
import { analyze, type AnalysisResponse } from "../lib/api";

export default function Home() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze(file: File, modality: string, indication: string) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await analyze(file, modality, indication);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <div className="wordmark">
            M<span className="axis">I</span>RROR
          </div>
          <div className="tagline">
            multimodal radiology reasoning &amp; observation reporter
          </div>
        </div>
        <div className="tagline">research prototype · not for clinical use</div>
      </header>

      <div className="grid">
        {/* Left column: input + the film viewer (signature). */}
        <div style={{ display: "grid", gap: 20 }}>
          <section className="card">
            <h2>Study input</h2>
            <UploadPanel
              onAnalyze={handleAnalyze}
              busy={busy}
              onFileSelected={(url) => {
                setImageUrl(url);
                setResult(null);
                setError(null);
              }}
            />
            {error && <div className="error">{error}</div>}
          </section>

          <section className="card">
            <h2>Film viewer · evidence localization</h2>
            <FilmViewer imageUrl={imageUrl} findings={result?.findings ?? []} />
          </section>
        </div>

        {/* Right column: predictions + report. */}
        <div style={{ display: "grid", gap: 20 }}>
          <section className="card">
            <h2>Predictions</h2>
            <FindingsList findings={result?.findings ?? []} />
          </section>

          <section className="card">
            <h2>Draft report · clinical reasoning</h2>
            <ReportPanel result={result} />
          </section>
        </div>
      </div>
    </div>
  );
}
