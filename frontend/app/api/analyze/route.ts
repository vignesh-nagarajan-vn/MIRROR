// Serverless MIRROR pipeline for the Vercel deployment.
//
// The local stack runs the real PyTorch pipeline (DenseNet/ViT classifier +
// Grad-CAM saliency + report generator) behind FastAPI. That cannot run on
// Vercel's serverless functions (no GPU, 250 MB bundle ceiling, cold-start model
// loads), so the hosted site uses Claude's vision model as a drop-in inference
// engine instead: it reads the uploaded radiograph, scores the 14 ChestX-ray14
// labels, localizes each positive finding with a bounding box, and drafts the
// FINDINGS/IMPRESSION report, returning the exact same JSON contract the
// frontend already consumes from the FastAPI backend.
//
// Set ANTHROPIC_API_KEY in the Vercel project to enable this. With no key the
// route still responds with a deterministic, clearly-labelled demo result so the
// deployed site never hard-fails.

import Anthropic from "@anthropic-ai/sdk";

export const runtime = "nodejs";
export const maxDuration = 60;

// NIH ChestX-ray14 taxonomy, kept in lockstep with
// models/common/constants.py so the hosted and local paths agree on label order.
const CHESTXRAY14_LABELS = [
  "Atelectasis",
  "Cardiomegaly",
  "Effusion",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pneumonia",
  "Pneumothorax",
  "Consolidation",
  "Edema",
  "Emphysema",
  "Fibrosis",
  "Pleural_Thickening",
  "Hernia",
] as const;

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

const SYSTEM_PROMPT =
  "You are a radiology reporting assistant analysing a single chest radiograph. " +
  "You assess the 14 NIH ChestX-ray14 pathology categories, estimate a calibrated " +
  "probability for each, localize every present finding with a bounding box, and " +
  "draft a concise clinician-style report. You never invent findings you cannot " +
  "see, and you always mark the report as an AI-generated draft requiring review " +
  "by a licensed radiologist. Use standard headings: FINDINGS and IMPRESSION.";

const ANALYSIS_TOOL: Anthropic.Tool = {
  name: "record_analysis",
  description:
    "Record the structured radiograph analysis: a probability and (for present " +
    "findings) a localized bounding box for each of the 14 ChestX-ray14 labels, " +
    "plus a clinician-style draft report.",
  input_schema: {
    type: "object",
    properties: {
      findings: {
        type: "array",
        description: "Exactly 14 entries, one per ChestX-ray14 label.",
        items: {
          type: "object",
          properties: {
            label: {
              type: "string",
              enum: [...CHESTXRAY14_LABELS],
            },
            probability: {
              type: "number",
              description: "Calibrated probability in [0, 1] for this finding.",
            },
            location: {
              type: "string",
              description:
                "Plain-English anatomical zone (e.g. 'right lower lung zone'), " +
                "or 'n/a' if the finding is below threshold.",
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
          "'AI-GENERATED DRAFT — requires verification by a licensed radiologist.'",
      },
    },
    required: ["findings", "report"],
  },
};

function buildUserPrompt(modality: string, indication: string): string {
  const indicationLine = indication ? `Clinical indication: ${indication}\n` : "";
  return (
    `Modality: ${modality}\n` +
    indicationLine +
    "\nAnalyse the attached radiograph and call record_analysis. For every one of " +
    "the 14 labels return a probability; mark a finding present when probability " +
    "≥ 0.5 and give it an anatomical location and a normalized bounding box. " +
    "Write FINDINGS (each present finding with its location and qualitative " +
    "confidence, plus pertinent negatives for major absent findings) and a brief " +
    "prioritised IMPRESSION."
  );
}

// Stitch a partial model response into a complete, contract-shaped finding list:
// always 14 labels in canonical order, defaults filled for anything the model
// omitted, values clamped to valid ranges.
function normalizeFindings(raw: unknown): Finding[] {
  const byLabel = new Map<string, any>();
  if (Array.isArray(raw)) {
    for (const f of raw) {
      if (f && typeof f.label === "string") byLabel.set(f.label, f);
    }
  }
  return CHESTXRAY14_LABELS.map((label) => {
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

// Deterministic stand-in used when no API key is configured, so the deployed
// site still demonstrates the full UI flow end to end.
function demoResult(modality: string) {
  const findings: Finding[] = CHESTXRAY14_LABELS.map((label): Finding => {
    const present = label === "Effusion" || label === "Cardiomegaly";
    const bbox: Finding["bbox"] = present
      ? label === "Effusion"
        ? [0.55, 0.62, 0.32, 0.3]
        : [0.34, 0.5, 0.36, 0.28]
      : null;
    return {
      label,
      probability: present ? (label === "Effusion" ? 0.81 : 0.66) : 0.07,
      present,
      location: present
        ? label === "Effusion"
          ? "right lower lung zone"
          : "cardiac silhouette"
        : "n/a",
      overlay_png_b64: null,
      bbox,
    };
  });
  return {
    modality,
    backbone: "demo (no ANTHROPIC_API_KEY set)",
    explain_method: "bbox",
    report:
      "FINDINGS:\n- Pleural effusion identified in the right lower lung zone " +
      "(high confidence).\n- Cardiomegaly identified in the cardiac silhouette " +
      "(moderate confidence).\n- No evidence of: Pneumothorax, Mass, Nodule.\n\n" +
      "IMPRESSION:\n1. Right pleural effusion is the dominant finding.\n2. " +
      "Possible cardiomegaly.\n\nAI-GENERATED DRAFT — requires verification by a " +
      "licensed radiologist.",
    report_backend: "demo",
    findings,
    meta: {
      num_present: findings.filter((f) => f.present).length,
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
  const modality = (form.get("modality") as string) || "chest X-ray";
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
    return Response.json(demoResult(modality));
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
      system: SYSTEM_PROMPT,
      tools: [ANALYSIS_TOOL],
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
            { type: "text", text: buildUserPrompt(modality, indication) },
          ],
        },
      ],
    });

    const toolUse = message.content.find((b) => b.type === "tool_use");
    if (!toolUse || toolUse.type !== "tool_use") {
      throw new Error("Model returned no structured analysis.");
    }
    const out = toolUse.input as { findings?: unknown; report?: string };

    const findings = normalizeFindings(out.findings);
    const report =
      out.report?.trim() ||
      "FINDINGS:\n- No acute findings above the reporting threshold.\n\n" +
        "IMPRESSION:\n1. No acute cardiopulmonary abnormality detected.\n\n" +
        "AI-GENERATED DRAFT — requires verification by a licensed radiologist.";

    return Response.json({
      modality,
      backbone: `claude-vision (${MODEL})`,
      explain_method: "bbox",
      report,
      report_backend: "anthropic",
      findings,
      meta: {
        num_present: findings.filter((f) => f.present).length,
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
