# Serving image: FastAPI + NumPy user tower + FAISS + LightGBM (no torch).
# Model artifacts are mounted at runtime, never baked in: they are derived from
# MovieLens, which may not be redistributed.
#
#   docker build -t reelrank .
#   docker run -p 8000:8000 -v "$PWD/artifacts/ml-1m:/artifacts:ro" reelrank
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/ src/
ENV PYTHONPATH=/app/src REELRANK_ARTIFACTS=/artifacts PYTHONUNBUFFERED=1
RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "reelrank.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
