# Contributing to MIRROR

Thanks for your interest in improving MIRROR. This is a research prototype, so
the bar is "clear, reproducible, and honest about limitations" rather than
production hardening.

## Ground rules

- **No PHI, no weights, no raw datasets in commits.** These are git-ignored for a
  reason. Share checkpoints via releases or external storage.
- **Keep the layers grounded.** The report generator must only describe evidence
  produced upstream. Do not let it infer findings the classifier didn't predict.
- **Safety language stays.** Every report path ends with the AI-generated /
  requires-radiologist-review disclaimer. Don't remove it.

## Development setup

See `docs/setup.md`. The CLI demo (`python -m demo.run_demo <image>`) runs with
no checkpoint and no API key, which makes it a fast sanity check after changes.

## Adding a backbone

1. Extend the factory in `models/classification/model.py` and add its default
   Grad-CAM target layer to `DEFAULT_TARGET_LAYERS`.
2. If it's a transformer, supply a reshape transform like `vit_reshape_transform`.
3. Add the option to `configs/default.yaml` comments.

## Adding a dataset / modality

1. Implement a `Dataset` subclass next to `ChestXray14Dataset`.
2. Register its label list and descriptions in `models/common/constants.py`.
3. Update `num_classes` handling so the head and report layer agree.

## Style

- Python: type hints, docstrings explaining *why* not just *what*, small modules.
- TypeScript/React: keep components focused; data fetching lives in `lib/`.

## Pull requests

Describe what you changed and how you verified it (commands + expected output).
For model changes, include before/after numbers from `evaluation/evaluate.py`.
