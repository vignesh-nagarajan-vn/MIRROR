// Modality registry for the hosted (Vercel / Claude-vision) engine.
//
// This is the TypeScript mirror of models/common/modalities.py. The local
// PyTorch stack and the hosted route must agree on the finding taxonomy and its
// ORDER for each modality (model output index i maps to labels[i]), so this file
// is kept in lockstep with the Python registry by hand. If you change a label
// set in Python, change it here too.

export type ModalityKey = "chest_xray" | "brain_mri" | "head_ct";

export type ModalitySpec = {
  key: ModalityKey;
  displayName: string;
  plane: "frontal" | "axial";
  labels: readonly string[];
  labelDescriptions: Record<string, string>;
  aliases: readonly string[];
  normalImpression: string;
  reportGuidance: string;
};

// --- Chest X-ray (NIH ChestX-ray14) ---------------------------------------
const CHEST_XRAY: ModalitySpec = {
  key: "chest_xray",
  displayName: "Chest X-ray",
  plane: "frontal",
  labels: [
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
  ],
  labelDescriptions: {
    Atelectasis: "partial or complete collapse of a lung or lobe",
    Cardiomegaly: "enlargement of the cardiac silhouette",
    Effusion: "fluid in the pleural space",
    Infiltration: "ill-defined opacity suggesting interstitial or alveolar filling",
    Mass: "a discrete lesion larger than 3 cm",
    Nodule: "a rounded opacity up to 3 cm",
    Pneumonia: "airspace opacity consistent with infection",
    Pneumothorax: "air in the pleural space",
    Consolidation: "region of lung filled with liquid instead of air",
    Edema: "fluid accumulation in the interstitial and alveolar spaces",
    Emphysema: "lung hyperinflation with destruction of alveolar walls",
    Fibrosis: "scarring and architectural distortion of lung tissue",
    Pleural_Thickening: "thickening of the pleural lining",
    Hernia: "protrusion of abdominal contents into the thorax",
  },
  aliases: ["chest x-ray", "chest xray", "chest", "cxr", "radiograph", "xray", "x-ray", "cr", "dx"],
  normalImpression: "No acute cardiopulmonary abnormality detected.",
  reportGuidance:
    "This is a frontal chest radiograph. Describe findings using standard " +
    "thoracic anatomy (lung zones, cardiomediastinal silhouette, pleura, " +
    "costophrenic angles).",
};

// --- Brain MRI ------------------------------------------------------------
const BRAIN_MRI: ModalitySpec = {
  key: "brain_mri",
  displayName: "Brain MRI",
  plane: "axial",
  labels: [
    "Glioma",
    "Meningioma",
    "Pituitary_Adenoma",
    "Metastasis",
    "Acute_Infarct",
    "Hemorrhage",
    "Edema",
    "Mass_Effect",
    "Hydrocephalus",
    "White_Matter_Hyperintensity",
    "Atrophy",
  ],
  labelDescriptions: {
    Glioma: "an intra-axial tumour arising from glial tissue",
    Meningioma: "an extra-axial dural-based tumour",
    Pituitary_Adenoma: "a sellar/suprasellar tumour of the pituitary gland",
    Metastasis: "one or more secondary intracranial tumour deposits",
    Acute_Infarct: "acute ischaemic injury with restricted diffusion",
    Hemorrhage: "intracranial blood products",
    Edema: "vasogenic or cytotoxic parenchymal swelling",
    Mass_Effect: "displacement of adjacent structures by a lesion",
    Hydrocephalus: "enlargement of the ventricular system",
    White_Matter_Hyperintensity: "T2/FLAIR-hyperintense white-matter signal",
    Atrophy: "volume loss with sulcal and ventricular prominence",
  },
  aliases: ["brain mri", "mri brain", "mri", "brain", "head mri", "neuro mri", "mr"],
  normalImpression: "No acute intracranial abnormality detected.",
  reportGuidance:
    "This is an axial brain MRI slice. Describe findings using neuroanatomy " +
    "(frontal/parietal/temporal/occipital lobes, cerebellum, ventricles, " +
    "midline) and note laterality. Do not use thoracic terminology.",
};

// --- Head CT (RSNA intracranial hemorrhage taxonomy) ----------------------
const HEAD_CT: ModalitySpec = {
  key: "head_ct",
  displayName: "Head CT",
  plane: "axial",
  labels: [
    "Intracranial_Hemorrhage",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
    "Epidural",
    "Acute_Infarct",
    "Mass_Effect",
    "Midline_Shift",
    "Hydrocephalus",
    "Skull_Fracture",
  ],
  labelDescriptions: {
    Intracranial_Hemorrhage: "any acute intracranial blood (parent finding)",
    Intraparenchymal: "haemorrhage within the brain parenchyma",
    Intraventricular: "haemorrhage within the ventricular system",
    Subarachnoid: "haemorrhage in the subarachnoid space",
    Subdural: "a crescentic collection between dura and arachnoid",
    Epidural: "a lenticular collection between skull and dura",
    Acute_Infarct: "an acute ischaemic territory of low attenuation",
    Mass_Effect: "effacement or displacement of adjacent structures",
    Midline_Shift: "displacement of midline structures across the falx",
    Hydrocephalus: "enlargement of the ventricular system",
    Skull_Fracture: "a lucent cortical break in the calvarium or skull base",
  },
  aliases: ["ct", "head ct", "brain ct", "ct head", "ct brain", "cct", "computed tomography"],
  normalImpression: "No acute intracranial abnormality detected.",
  reportGuidance:
    "This is an axial non-contrast head CT slice. Describe findings using " +
    "neuroanatomy and standard haemorrhage subtypes; note laterality and any " +
    "midline shift. Do not use thoracic terminology.",
};

export const DEFAULT_MODALITY: ModalityKey = "chest_xray";

export const MODALITIES: Record<ModalityKey, ModalitySpec> = {
  chest_xray: CHEST_XRAY,
  brain_mri: BRAIN_MRI,
  head_ct: HEAD_CT,
};

export const MODALITY_LIST: ModalitySpec[] = [CHEST_XRAY, BRAIN_MRI, HEAD_CT];

// Free-text -> spec, mirroring resolve_modality() in Python: match the key,
// display name, or any alias, case-insensitively; unknown/empty -> default.
export function resolveModality(name: string | null | undefined): ModalitySpec {
  if (!name) return MODALITIES[DEFAULT_MODALITY];
  const norm = name.trim().toLowerCase();
  if (norm === "" || norm === "auto") return MODALITIES[DEFAULT_MODALITY];
  for (const spec of MODALITY_LIST) {
    if (
      spec.key === norm ||
      spec.displayName.toLowerCase() === norm ||
      spec.aliases.includes(norm)
    ) {
      return spec;
    }
  }
  return MODALITIES[DEFAULT_MODALITY];
}
