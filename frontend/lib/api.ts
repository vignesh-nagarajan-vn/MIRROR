// Thin client for the MIRROR backend.

export type Finding = {
  label: string;
  probability: number;
  present: boolean;
  location: string;
  overlay_png_b64: string | null;
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

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
