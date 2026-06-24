// Thin client for the MIRROR backend.

export type Finding = {
  label: string;
  probability: number;
  present: boolean;
  location: string;
  overlay_png_b64: string | null;
  // Normalized [x, y, width, height] in [0,1]. Returned by the Vercel
  // serverless path (Claude-vision localization); the FastAPI path ships a
  // rendered Grad-CAM PNG in overlay_png_b64 instead.
  bbox?: [number, number, number, number] | null;
};

export type AnalysisResponse = {
  modality: string;
  backbone: string;
  explain_method: string;
  report: string;
  report_backend: string;
  findings: Finding[];
  meta: Record<string, unknown>;
};

// Where the analyze call is sent.
//
// - Hosted on Vercel: leave NEXT_PUBLIC_API_URL unset. Requests go to the
//   same-origin Next.js serverless route at /api/analyze (Claude-vision engine).
// - Local full stack: set NEXT_PUBLIC_API_URL=http://localhost:8000 to target
//   the FastAPI backend running the real PyTorch pipeline.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

export async function analyze(
  file: File,
  modality: string,
  indication: string,
): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("image", file);
  form.append("modality", modality);
  if (indication) form.append("indication", indication);

  const res = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* ignore parse errors, keep the status message */
    }
    throw new Error(detail);
  }
  return res.json();
}
