# MIRROR convenience targets.
.PHONY: help install demo backend frontend train eval eval-loc lint clean

help:
	@echo "MIRROR targets:"
	@echo "  make install    install Python deps"
	@echo "  make demo IMG=path/to.png   run the CLI pipeline demo"
	@echo "  make backend    run the FastAPI server on :8000"
	@echo "  make frontend   run the Next.js dev server on :3000"
	@echo "  make train      train the classifier (needs ChestX-ray14)"
	@echo "  make eval CKPT=path.pt   evaluate prediction quality (AUROC/F1)"
	@echo "  make eval-loc CKPT=path.pt   evaluate explanation quality (pointing game / IoU)"
	@echo "  make clean      remove caches and build artifacts"

install:
	pip install -r requirements.txt

demo:
	python -m demo.run_demo $(IMG)

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

train:
	python -m models.classification.train --config configs/default.yaml

eval:
	python -m evaluation.evaluate --config configs/default.yaml --checkpoint $(CKPT)

eval-loc:
	python -m evaluation.evaluate_localization --config configs/default.yaml --checkpoint $(CKPT)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/.next frontend/out
