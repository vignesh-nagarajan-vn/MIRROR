# Demo

Two ways to see MIRROR end-to-end.

## 1. Command line (no servers)

```bash
pip install -r requirements.txt
python -m demo.run_demo path/to/chest_xray.png
```

Prints predictions and a draft report, and writes Grad-CAM overlays into
`demo/assets/`. Runs with ImageNet-pretrained weights and the offline template
report backend, so no checkpoint or API key is required.

## 2. Full stack (API + web UI)

Terminal 1 — backend:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Terminal 2 — frontend:

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000, drop in a radiograph, and click **Run analysis**.

> Research prototype. Not a medical device. Drafts must be reviewed by a
> licensed radiologist.
