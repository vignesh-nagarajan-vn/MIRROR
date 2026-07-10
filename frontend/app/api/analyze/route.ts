// Serverless MIRROR pipeline for the Vercel deployment.
//
// The local stack runs the real PyTorch pipeline (DenseNet/ViT classifier +
// Grad-CAM saliency + report generator) behind FastAPI. That cannot run on
// Vercel's serverless functions (no GPU, 250 MB bundle ceiling, cold-start model
// loads), so the hosted site uses Claude's vision model as a drop-in inference
// engine instead: it reads the uploaded study, scores the finding taxonomy for
// the selected modality (chest X-ray, brain MRI, or head CT), localizes each
// positive finding with a bounding box, and drafts the FINDINGS/IMPRESSION
// report, returning the exact same JSON contract the frontend already consumes
// from the FastAPI backend.
//
// Set ANTHROPIC_API_KEY in the Vercel project to enable this. With no key the
// route still responds with a deterministic, clearly-labelled demo result so the
// deployed site never hard-fails.

import Anthropic from "@anthropic-ai/sdk";
import {
  resolveModality,
  type ModalitySpec,
} from "../../../lib/modalities";

export const runtime = "nodejs";
export const maxDuration = 60;

const PRESENCE_THRESHOLD = 0.5;
const MODEL = process.env.ANTHROPIC_MODEL || "claude-haiku-4-5";

type Finding = {
  label: string;
  probability: number;
  present: boolean;
  location: string;
  overlay_png_b64: string | null;
  bbox: [number, number, number, number] | null; // normalized [x, y, w, h]
};

function systemPrompt(spec: ModalitySpec): string {
  return (
    `You are a radiology reporting assistant analysing a single ${spec.displayName}. ` +
    `You assess the ${spec.labels.length} finding categories for this modality, ` +
    "estimate a calibrated probability for each, localize every present finding " +
    "with a bounding box, and draft a concise clinician-style report. You never " +
    "invent findings you cannot see, and you always mark the report as an " +
    "AI-generated draft requiring review by a licensed radiologist. Use standard " +
    "headings: FINDINGS and IMPRESSION."
  );
}

// The tool schema is built per modality so the label enum and the count match
// the selected taxonomy.
function analysisTool(spec: ModalitySpec): Anthropic.Tool {
  return {
    name: "record_analysis",
    description:
      `Record the structured ${spec.displayName} analysis: a probability and ` +
      "(for present findings) a localized bounding box for each of the " +
      `${spec.labels.length} labels, plus a clinician-style draft report.`,
    input_schema: {
      type: "object",
      properties: {
        findings: {
          type: "array",
          description: `Exactly ${spec.labels.length} entries, one per label.`,
          items: {
            type: "object",
            properties: {
              label: { type: "string", enum: [...spec.labels] },
              probability: {
                type: "number",
                description: "Calibrated probability in [0, 1] for this finding.",
              },
              location: {
                type: "string",
                description:
                  "Plain-English anatomical region, or 'n/a' if the finding is " +
                  "below threshold.",
              },
              bbox: {
                type: "array",
                description:
                  "Normalized [x, y, width, height] in [0,1] over the image for a " +
                  "present finding, or null. Origin is the top-left corner.",
                items: { type: "number" },
              },
            },
            required: ["label", "probability", "location"],
          },
        },
        report: {
          type: "string",
          description:
            "Draft report with FINDINGS and IMPRESSION sections, ending with: " +
            "'AI-GENERATED DRAFT - requires verification by a licensed radiologist.'",
        },
      },
      required: ["findings", "report"],
    },
  };
}

function buildUserPrompt(spec: ModalitySpec, indication: string): string {
  const indicationLine = indication ? `Clinical indication: ${indication}\n` : "";
  return (
    `Modality: ${spec.displayName}\n` +
    indicationLine +
    `${spec.reportGuidance}\n\n` +
    "Analyse the attached image and call record_analysis. For every one of the " +
    `${spec.labels.length} labels return a probability; mark a finding present ` +
    "when probability >= 0.5 and give it an anatomical location and a normalized " +
    "bounding box. Write FINDINGS (each present finding with its location and " +
    "qualitative confidence, plus pertinent negatives for major absent findings) " +
    "and a brief prioritised IMPRESSION."
  );
}

// Stitch a partial model response into a complete, contract-shaped finding list:
// always the modality's labels in canonical order, defaults filled for anything
// the model omitted, values clamped to valid ranges.
function normalizeFindings(raw: unknown, spec: ModalitySpec): Finding[] {
  const byLabel = new Map<string, any>();
  if (Array.isArray(raw)) {
    for (const f of raw) {
      if (f && typeof f.label === "string") byLabel.set(f.label, f);
    }
  }
  return spec.labels.map((label) => {
    const f = byLabel.get(label) ?? {};
    const probability = Math.min(1, Math.max(0, Number(f.probability) || 0));
    const present = probability >= PRESENCE_THRESHOLD;
    let bbox: Finding["bbox"] = null;
    if (
      present &&
      Array.isArray(f.bbox) &&
      f.bbox.length === 4 &&
      f.bbox.every((n: unknown) => typeof n === "number")
    ) {
      const [x, y, w, h] = f.bbox.map((n: number) => Math.min(1, Math.max(0, n)));
      bbox = [x, y, w, h];
    }
    return {
      label,
      probability: Math.round(probability * 1e4) / 1e4,
      present,
      location: present ? String(f.location || "unspecified region") : "n/a",
      overlay_png_b64: null,
      bbox,
    };
  });
}

// A per-modality present/absent pattern for the deterministic demo, so each
// modality's stand-in shows plausible findings from its own taxonomy.
const DEMO_PRESENT: Record<string, { label: string; prob: number; loc: string; bbox: [number, number, number, number] }[]> = {
  chest_xray: [
    { label: "Effusion", prob: 0.81, loc: "right lower lung zone", bbox: [0.55, 0.62, 0.32, 0.3] },
    { label: "Cardiomegaly", prob: 0.66, loc: "cardiac silhouette", bbox: [0.34, 0.5, 0.36, 0.28] },
  ],
  brain_mri: [
    { label: "Glioma", prob: 0.79, loc: "left frontal region", bbox: [0.2, 0.28, 0.24, 0.22] },
    { label: "Edema", prob: 0.61, loc: "left frontal region", bbox: [0.16, 0.24, 0.34, 0.3] },
  ],
  head_ct: [
    { label: "Intracranial_Hemorrhage", prob: 0.83, loc: "right parietal region", bbox: [0.55, 0.35, 0.26, 0.24] },
    { label: "Subdural", prob: 0.6, loc: "right convexity", bbox: [0.6, 0.2, 0.28, 0.5] },
  ],
};

// Deterministic stand-in used when no API key is configured, so the deployed
// site still demonstrates the full UI flow end to end.
function demoResult(spec: ModalitySpec) {
  const present = DEMO_PRESENT[spec.key] ?? [];
  const presentByLabel = new Map(present.map((p) => [p.label, p]));
  const findings: Finding[] = spec.labels.map((label): Finding => {
    const p = presentByLabel.get(label);
    return {
      label,
      probability: p ? p.prob : 0.07,
      present: Boolean(p),
      location: p ? p.loc : "n/a",
      overlay_png_b64: null,
      bbox: p ? p.bbox : null,
    };
  });
  const findingLines = present
    .map((p) => `- ${p.label.replace(/_/g, " ")} identified in the ${p.loc} (high confidence).`)
    .join("\n");
  const impressionLines = present
    .map((p, i) => `${i + 1}. ${p.label.replace(/_/g, " ")}.`)
    .join("\n");
  return {
    modality: spec.displayName,
    backbone: "demo (no ANTHROPIC_API_KEY set)",
    explain_method: "bbox",
    report:
      `FINDINGS:\n${findingLines}\n\nIMPRESSION:\n${impressionLines}\n\n` +
      "AI-GENERATED DRAFT - requires verification by a licensed radiologist.",
    report_backend: "demo",
    findings,
    meta: {
      modality_key: spec.key,
      num_present: findings.filter((f) => f.present).length,
      num_labels: spec.labels.length,
      demo: true,
      note: "Set ANTHROPIC_API_KEY in your Vercel project to run live analysis.",
    },
  };
}

export async function POST(req: Request) {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return Response.json({ detail: "Expected multipart/form-data." }, { status: 400 });
  }

  const image = form.get("image");
  const modalityInput = (form.get("modality") as string) || "chest X-ray";
  const spec = resolveModality(modalityInput);
  const indication = (form.get("indication") as string) || "";

  if (!(image instanceof File)) {
    return Response.json({ detail: "No image uploaded." }, { status: 400 });
  }

  const mediaType = image.type || "image/png";
  if (!/^image\/(png|jpeg|jpg|webp)$/.test(mediaType)) {
    return Response.json(
      {
        detail:
          "The hosted demo accepts PNG, JPEG, or WEBP. Run the local stack for " +
          "BMP and native DICOM (.dcm) support.",
      },
      { status: 415 },
    );
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return Response.json(demoResult(spec));
  }

  const bytes = Buffer.from(await image.arrayBuffer());
  const base64 = bytes.toString("base64");
  const sdkMediaType = mediaType === "image/jpg" ? "image/jpeg" : mediaType;

  const t0 = Date.now();
  try {
    const client = new Anthropic({ apiKey });
    const message = await client.messages.create({
      model: MODEL,
      max_tokens: 2048,
      system: systemPrompt(spec),
      tools: [analysisTool(spec)],
      tool_choice: { type: "tool", name: "record_analysis" },
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: sdkMediaType as
                  | "image/png"
                  | "image/jpeg"
                  | "image/webp"
                  | "image/gif",
                data: base64,
              },
            },
            { type: "text", text: buildUserPrompt(spec, indication) },
          ],
        },
      ],
    });

    const toolUse = message.content.find((b) => b.type === "tool_use");
    if (!toolUse || toolUse.type !== "tool_use") {
      throw new Error("Model returned no structured analysis.");
    }
    const out = toolUse.input as { findings?: unknown; report?: string };

    const findings = normalizeFindings(out.findings, spec);
    const report =
      out.report?.trim() ||
      `FINDINGS:\n- No acute findings above the reporting threshold.\n\n` +
        `IMPRESSION:\n1. ${spec.normalImpression}\n\n` +
        "AI-GENERATED DRAFT - requires verification by a licensed radiologist.";

    return Response.json({
      modality: spec.displayName,
      backbone: `claude-vision (${MODEL})`,
      explain_method: "bbox",
      report,
      report_backend: "anthropic",
      findings,
      meta: {
        modality_key: spec.key,
        num_present: findings.filter((f) => f.present).length,
        num_labels: spec.labels.length,
        timings_ms: { total: Date.now() - t0 },
        engine: "vercel-serverless",
        model: MODEL,
        stages: { prediction: true, localization: true, report: true },
      },
    });
  } catch (err) {
    const detail = err instanceof Error ? err.message : "Analysis failed.";
    return Response.json({ detail: `Analysis failed: ${detail}` }, { status: 500 });
  }
}
