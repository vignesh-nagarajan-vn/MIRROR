"use client";

import { useRef, useState } from "react";

type Props = {
  onAnalyze: (file: File, modality: string, indication: string) => void;
  busy: boolean;
  onFileSelected: (url: string) => void;
};

const MODALITIES = ["chest X-ray", "brain MRI", "CT"];

export default function UploadPanel({ onAnalyze, busy, onFileSelected }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [modality, setModality] = useState(MODALITIES[0]);
  const [indication, setIndication] = useState("");
  const [drag, setDrag] = useState(false);

  function pick(f: File | null) {
    if (!f) return;
    setFile(f);
    onFileSelected(URL.createObjectURL(f));
  }

  return (
    <div>
      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pick(e.dataTransfer.files?.[0] ?? null);
        }}
      >
        <p style={{ color: "var(--ink)", margin: 0 }}>
          {file ? file.name : "Drop a radiograph or click to browse"}
        </p>
        <p className="hint">PNG · JPEG · up to 20 MB</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/bmp,image/webp"
          hidden
          onChange={(e) => pick(e.target.files?.[0] ?? null)}
        />
      </div>

      <div className="controls">
        <div className="field">
          <label htmlFor="modality">Modality</label>
          <select
            id="modality"
            value={modality}
            onChange={(e) => setModality(e.target.value)}
          >
            {MODALITIES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="indication">Clinical indication (optional)</label>
          <input
            id="indication"
            placeholder="e.g. productive cough, 3 days"
            value={indication}
            onChange={(e) => setIndication(e.target.value)}
          />
        </div>

        <button
          className="btn"
          disabled={!file || busy}
          onClick={() => file && onAnalyze(file, modality, indication)}
        >
          {busy ? (
            <>
              <span className="spinner" /> Analyzing
            </>
          ) : (
            "Run analysis"
          )}
        </button>
      </div>
    </div>
  );
}
