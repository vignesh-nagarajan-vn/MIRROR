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

The ChestX-ray14 label set the classifier predicts.

```json
{ "labels": ["Atelectasis", "Cardiomegaly", "..."], "count": 14 }
```

## `POST /api/analyze`

Multipart form upload. Runs the full pipeline.

**Form fields**

| field | type | required | notes |
| --- | --- | --- | --- |
| `image` | file | yes | PNG/JPEG/BMP/WEBP, ≤ 20 MB |
| `modality` | string | no | default `"chest X-ray"` |
| `indication` | string | no | free-text clinical context |

**Example**

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@chest_xray.png" \
  -F "modality=chest X-ray" \
  -F "indication=productive cough, 3 days"
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
  "meta": { "num_present": 1 }
}
```

`overlay_png_b64` is a base64-encoded PNG of the Grad-CAM overlay (present only
for the top-k positive findings). Decode and display, or `data:image/png;base64,`
inline.

**Errors**

| status | meaning |
| --- | --- |
| `413` | image exceeds the size limit |
| `415` | unsupported content type |
| `500` | analysis failed (message in `detail`) |
