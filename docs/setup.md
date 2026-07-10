# Setup

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+ (for the frontend)
- ~6 GB disk for pretrained weights and deps; a CUDA GPU is optional but speeds
  up training and Score-CAM substantially.

## 1. Python environment

```bash
git clone https://github.com/vignesh-nagarajan-vn/MIRROR.git mirror
cd mirror
python -m venv .venv && source .venv/bin/activate    # Windows / Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
```

The first pipeline run downloads ImageNet-pretrained backbone weights (handled by
torchvision).

## 2. Smoke test (no data, no servers)

```bash
python -m demo.run_demo path/to/any_chest_xray.png
# bundled synthetic samples work with zero downloads, across all three modalities:
python -m demo.run_demo datasets/samples/chestxray14/images/synth_0001.png
python -m demo.run_demo datasets/samples/brain_mri/images/mri_0001.png --modality "brain MRI"
python -m demo.run_demo datasets/samples/head_ct/images/ct_0000.dcm --modality auto
```

You should see predictions, an offline-template report, and saliency overlays
written to `demo/assets/`. `--modality` selects the finding taxonomy (chest X-ray
/ brain MRI / CT); `--modality auto` routes a DICOM by its `Modality` tag.

## 3. Backend API

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# Swagger UI at http://localhost:8000/docs
```

## 4. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local      # points at http://localhost:8000
npm run dev                            # http://localhost:3000
```

## 5. (Optional) Enable the Claude report backend

The offline template backend needs nothing. To generate richer prose reports
with Claude, set an API key and switch the provider:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # never commit this
```

In `configs/default.yaml`:

```yaml
report:
  provider: anthropic        # was: template
  model: claude-haiku-4-5
```

If the API call fails for any reason, the generator automatically falls back to
the template backend, so analysis never hard-fails on a missing key.

## 6. (Optional) Train your own checkpoint

Obtain ChestX-ray14 (see `datasets/README.md`), then:

```bash
python -m models.classification.train --config configs/default.yaml
python -m evaluation.evaluate --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt
```

Point the pipeline at your checkpoint via `model.checkpoint_path` in the config.

## Docker

```bash
docker compose up --build      # backend on :8000, frontend on :3000
```

## Deploy a public site (Vercel)

To put MIRROR online as a live, shareable website (no backend to host, powered
by Claude vision through a Next.js serverless route), see
[`deployment.md`](deployment.md). It covers the Vercel import flow for your repo
and the `ANTHROPIC_API_KEY` setup.
