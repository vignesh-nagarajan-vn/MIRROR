# MIRROR backend image.
# Build from the repository root so the `models` package is in the build context:
#   docker build -f backend/Dockerfile -t mirror-backend .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for Pillow / OpenCV-free image handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (root requirements covers the ML stack).
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install -r requirements.txt -r backend-requirements.txt

# Copy source: the models package and the backend app.
COPY models ./models
COPY configs ./configs
COPY backend ./backend

ENV MIRROR_CONFIG=configs/default.yaml \
    PYTHONPATH=/app:/app/backend

EXPOSE 8000
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
