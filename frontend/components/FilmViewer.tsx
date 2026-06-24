"use client";

import { useState, type CSSProperties } from "react";
import type { Finding } from "../lib/api";

type Props = {
  imageUrl: string | null;
  findings: Finding[];
};

// A finding is "explainable" if it carries either a rendered Grad-CAM overlay
// (FastAPI / PyTorch path) or a normalized bounding box (Vercel / Claude-vision
// path). The viewer renders whichever the active finding provides.
function isExplained(f: Finding): boolean {
  return Boolean(f.overlay_png_b64) || Boolean(f.bbox);
}

// Map a normalized bbox (fractions of the *image*) onto the square viewer.
// The film is shown with object-fit: contain, which letterboxes non-square
// images, so we offset/scale by the image's displayed content rect rather than
// the container, keeping the box aligned for any aspect ratio.
function boxStyle(
  bbox: [number, number, number, number],
  natural: { w: number; h: number } | null,
): CSSProperties {
  const [bx, by, bw, bh] = bbox;
  // Until the image reports its natural size, fall back to container-relative.
  const aspect = natural && natural.h > 0 ? natural.w / natural.h : 1;
  // Fraction of the square container the image content actually covers.
  const contentW = aspect >= 1 ? 1 : aspect;
  const contentH = aspect >= 1 ? 1 / aspect : 1;
  const offX = (1 - contentW) / 2;
  const offY = (1 - contentH) / 2;
  return {
    left: `${(offX + bx * contentW) * 100}%`,
    top: `${(offY + by * contentH) * 100}%`,
    width: `${bw * contentW * 100}%`,
    height: `${bh * contentH * 100}%`,
  };
}

// The film viewer is the page's signature: a darkened lightbox where the
// original radiograph and the evidence overlay register exactly, with a switch
// to fade the evidence in and chips to pick which finding to inspect.
export default function FilmViewer({ imageUrl, findings }: Props) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [activeIdx, setActiveIdx] = useState(0);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);

  const explained = findings.filter(isExplained);
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
        <img
          src={imageUrl}
          alt="Uploaded radiograph"
          onLoad={(e) =>
            setNatural({
              w: e.currentTarget.naturalWidth,
              h: e.currentTarget.naturalHeight,
            })
          }
        />
        {active?.overlay_png_b64 && (
          <img
            className="overlay"
            src={`data:image/png;base64,${active.overlay_png_b64}`}
            alt={`Saliency overlay for ${active.label}`}
          />
        )}
        {active && !active.overlay_png_b64 && active.bbox && (
          <div className="evidence-box" style={boxStyle(active.bbox, natural)}>
            <span className="evidence-tag">{active.label.replace(/_/g, " ")}</span>
          </div>
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
