FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download and initialize both local models during the image build.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')"

COPY app ./app
COPY frontend ./frontend
COPY ingest ./ingest
COPY data/index ./data/index

EXPOSE 8000 8501
