# API Reference

Base URL (default): `http://localhost:8000`. Interactive docs at `/docs`.

## `GET /api/health`

Liveness and model-load status.

```json
{ "status": "ok", "version": "0.1.0", "model_loaded": false, "backbone": null }
```

`model_loaded` is `false` until the first `/api/analyze` call triggers lazy
loading.

## `GET /api/labels`

The label set the classifier predicts for a modality. Query `?modality=` to
switch (accepts a display name, alias, or key; default chest X-ray).

```json
{ "modality": "Brain MRI", "modality_key": "brain_mri",
  "labels": ["Glioma", "Meningioma", "..."], "count": 11 }
```

## `GET /api/modalities`

Every supported modality and its label set (for the UI selector).

```json
{ "modalities": [
  { "key": "chest_xray", "display_name": "Chest X-ray", "labels": ["Atelectasis", "..."], "count": 14, "plane": "frontal" },
  { "key": "brain_mri", "display_name": "Brain MRI", "labels": ["Glioma", "..."], "count": 11, "plane": "axial" },
  { "key": "head_ct", "display_name": "Head CT", "labels": ["Intracranial_Hemorrhage", "..."], "count": 11, "plane": "axial" }
] }
```

## `POST /api/analyze`

Multipart form upload. Runs the full pipeline.

**Form fields**

| field | type | required | notes |
| --- | --- | --- | --- |
| `image` | file | yes | PNG/JPEG/BMP/WEBP or DICOM (`.dcm`), ≤ 20 MB |
| `modality` | string | no | `"chest X-ray"` (default), `"brain MRI"`, `"CT"`, or `"auto"` to route a DICOM by its `Modality` tag |
| `indication` | string | no | free-text clinical context |

The `modality` picks the finding taxonomy and the anatomical vocabulary; `"auto"`
detects it from the DICOM `Modality` tag (`MR`→brain MRI, `CT`→head CT,
`CR`/`DX`→chest X-ray), falling back to chest X-ray for non-DICOM input.

**Example**

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@chest_xray.png" \
  -F "modality=chest X-ray" \
  -F "indication=productive cough, 3 days"

# brain MRI:
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@brain.png" -F "modality=brain MRI"

# DICOM, self-routing by its Modality tag:
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@head.dcm" -F "modality=auto"
```

**Response (`200`)**

```json
{
  "modality": "chest X-ray",
  "backbone": "densenet121",
  "explain_method": "gradcam",
  "report": "FINDINGS:\n- ...\n\nIMPRESSION:\n1. ...",
  "report_backend": "template",
  "findings": [
    {
      "label": "Effusion",
      "probability": 0.78,
      "present": true,
      "location": "the lower left zone",
      "overlay_png_b64": "iVBORw0KGgo..."
    }
  ],
  "meta": { "modality_key": "chest_xray", "num_present": 1, "num_labels": 14 }
}
```

For a brain MRI or head CT the `modality`/`meta.modality_key` change, `findings`
carry that modality's labels, and `location` uses axial brain-region vocabulary
(e.g. `"the right frontal region"`) instead of chest lung zones.

`overlay_png_b64` is a base64-encoded PNG of the Grad-CAM overlay (present only
for the top-k positive findings). Decode and display, or `data:image/png;base64,`
inline.

**DICOM uploads.** Native DICOM is accepted. Many clients send `.dcm` as
`application/dicom`; some send `application/octet-stream` — both are allowed and
the server confirms the DICOM magic bytes before decoding (applying the
modality/VOI LUT and MONOCHROME1 inversion). Compressed transfer syntaxes need a
handler installed server-side (`pylibjpeg` or `gdcm`).

```bash
curl -X POST http://localhost:8000/api/analyze -F "image=@study.dcm"
```

**Errors**

| status | meaning |
| --- | --- |
| `413` | image exceeds the size limit |
| `415` | unsupported content type |
| `500` | analysis failed (message in `detail`) |
