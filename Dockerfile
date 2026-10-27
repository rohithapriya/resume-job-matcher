FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
    API_URL=http://localhost:8000

WORKDIR /app

# CPU-only PyTorch is much smaller than the default CUDA build, and this model doesn't need a GPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model at build time so the container starts without fetching it
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['MODEL_NAME'])"

COPY app ./app
COPY ui ./ui
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000 8501
CMD ["./start.sh"]