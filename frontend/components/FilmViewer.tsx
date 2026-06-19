"use client";

import { useState } from "react";
import type { Finding } from "../lib/api";

type Props = {
  imageUrl: string | null;
  findings: Finding[];
};

// The film viewer is the page's signature: a darkened lightbox where the
// original radiograph and the Grad-CAM evidence overlay register exactly, with
// a switch to fade between them and chips to pick which finding to inspect.
export default function FilmViewer({ imageUrl, findings }: Props) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [activeIdx, setActiveIdx] = useState(0);

  const explained = findings.filter((f) => f.overlay_png_b64);
  const active = explained[activeIdx];

  if (!imageUrl) {
    return (
      <div className="viewer">
        <span className="empty">awaiting study</span>
      </div>
    );
  }

  return (
    <>
      <div className={`viewer ${showOverlay && active ? "show-overlay" : ""}`}>
        <img src={imageUrl} alt="Uploaded radiograph" />
        {active && (
          <img
            className="overlay"
            src={`data:image/png;base64,${active.overlay_png_b64}`}
            alt={`Saliency overlay for ${active.label}`}
          />
        )}
      </div>

      <div className="viewer-bar">
        <div
          className="toggle"
          role="switch"
          aria-checked={showOverlay}
          tabIndex={0}
          onClick={() => setShowOverlay((v) => !v)}
          onKeyDown={(e) => e.key === "Enter" && setShowOverlay((v) => !v)}
        >
          <span className={`switch ${showOverlay ? "on" : ""}`} />
          evidence overlay
        </div>

        {explained.length > 1 && (
          <div className="chip-row">
            {explained.map((f, i) => (
              <button
                key={f.label}
                className={`chip ${i === activeIdx ? "active" : ""}`}
                onClick={() => setActiveIdx(i)}
              >
                {f.label.replace(/_/g, " ")}
              </button>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
