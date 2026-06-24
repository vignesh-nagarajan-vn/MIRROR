# Deployment (Vercel)

This guide stands up a **live, publicly shareable** MIRROR site on
[Vercel](https://vercel.com). The deployed site is fully functional on its own —
no separate backend to host — because the Next.js app ships a serverless
analysis route that uses **Claude's vision model** as the inference engine.

> **How the hosted path differs from local.** The local full stack runs the real
> PyTorch pipeline (DenseNet/ViT classifier → Grad-CAM saliency → report) behind
> FastAPI. Vercel's serverless functions can't host that (no GPU, a 250 MB bundle
> ceiling, cold-start model loads), so on Vercel the route
> [`frontend/app/api/analyze/route.ts`](../frontend/app/api/analyze/route.ts)
> asks Claude to read the radiograph, score the 14 ChestX-ray14 labels, localize
> each positive finding with a bounding box, and draft the report — returning the
> **same JSON contract** the UI already consumes. The frontend code is identical
> either way; only `NEXT_PUBLIC_API_URL` changes which engine answers.

## What you need

- A free [Vercel account](https://vercel.com/signup) (sign in with GitHub).
- An **Anthropic API key** from <https://console.anthropic.com/>. This is set as
  a server-side environment variable on Vercel; it is never exposed to browsers
  and never committed.

## One-click deploy (button)

The README's **Deploy to Vercel** button opens Vercel's import flow pre-filled to
ask for `ANTHROPIC_API_KEY`. It clones this repo into your own GitHub account and
deploys it.

When the configure screen appears:

1. **Root Directory** → set to **`frontend`** (the Next.js app lives in a
   subdirectory). This is the one setting Vercel can't infer from the URL.
2. **Environment Variables** → set `ANTHROPIC_API_KEY` to your key.
   Optionally set `ANTHROPIC_MODEL` (defaults to `claude-sonnet-4-6`).
3. Click **Deploy**. In ~1–2 minutes you get a public `*.vercel.app` URL.

## Deploy your existing repo (recommended for your own fork)

To deploy the repo you already have on GitHub (rather than a fresh clone):

1. Go to **<https://vercel.com/new>** → **Import Git Repository** → pick your
   `MIRROR` repo.
2. **Root Directory** → **`frontend`**.
3. Framework preset is auto-detected as **Next.js** (build `next build`, install
   `npm install`). Leave the defaults.
4. **Environment Variables**:
   | Name | Value | Notes |
   | --- | --- | --- |
   | `ANTHROPIC_API_KEY` | `sk-ant-...` | required for live analysis |
   | `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | optional override |
   | `NEXT_PUBLIC_API_URL` | *(leave empty)* | empty → use the built-in serverless route |
5. **Deploy.** Every push to your default branch redeploys automatically.

> If you deploy **without** `ANTHROPIC_API_KEY`, the site still loads and the UI
> works end to end, but the analyze route returns a clearly-labelled **demo
> result** instead of live analysis. Add the key in
> **Project → Settings → Environment Variables** and redeploy to go live.

## Deploy from the CLI (optional)

```bash
npm i -g vercel
cd frontend
vercel            # first run links the project; set Root Directory if asked
vercel env add ANTHROPIC_API_KEY     # paste your key
vercel --prod     # promote to the public production URL
```

## After deploying

- Open the production URL, drop in a PNG/JPEG/WEBP chest radiograph, toggle the
  evidence overlay, and read the draft report.
- The hosted route accepts **PNG / JPEG / WEBP**. Native **DICOM (`.dcm`)** and
  **BMP** ingest are local-stack only (they need the Python decode path); see
  [`datasets/README.md`](../datasets/README.md#dicom-ingest).
- Share the URL — it's a fully working public demo.

## Costs and limits

- Vercel's Hobby tier hosts the site for free; serverless functions have a
  per-request time budget (the analyze route sets `maxDuration = 60s`).
- Each live analysis is one Claude API call billed to **your** Anthropic account.
  The demo (no-key) path costs nothing.

## Safety note

The hosted demo is a research prototype, **not for clinical use** — the same
disclaimer the app surfaces and that every generated report carries. Don't upload
real patient images / PHI to a public deployment.
