# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL maintainer="YouTube Transcripts Summarizer & Q&A Bot"
LABEL version="1.0.0"
LABEL description="AI-Powered YouTube Video Summarizer & RAG Q&A Bot (LangChain, FAISS, HuggingFace Embeddings, Google Gemini, Gradio)"

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
     PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    EMBEDDING_DEVICE=cpu \
    GEMINI_MODEL=gemini-3.5-flash-lite \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

WORKDIR /app

# Install minimal system dependencies required by faiss-cpu / torchgomp and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "PySocks>=1.7.1"

# Install lightweight userspace Cloudflare WARP tools (wgcf + wireproxy) for automatic Cloud IP bypass
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSL "https://github.com/ViRb3/wgcf/releases/download/v2.3.1/wgcf_2.3.1_linux_${ARCH}" -o /usr/local/bin/wgcf && \
    chmod +x /usr/local/bin/wgcf && \
    curl -fsSL "https://github.com/pufferffish/wireproxy/releases/download/v1.1.3/wireproxy_linux_${ARCH}.tar.gz" | tar -xz -C /usr/local/bin wireproxy && \
    chmod +x /usr/local/bin/wireproxy

LABEL release_version="1.1.1"

# Copy application source code and tests
COPY ytbot.py app.py .env.example ./
COPY tests/ ./tests/

# Create a non-root user and cache directory for HuggingFace models
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache/huggingface /app/flagged && \
    chown -R appuser:appuser /app

USER appuser

# Optional build argument to pre-download the HuggingFace embedding model during build
ARG PRELOAD_MODEL=false
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='${EMBEDDING_MODEL}', model_kwargs={'device': 'cpu'})"; \
    fi

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Default: Run the Gradio Web UI (ytbot.py)
# To run the FastAPI version instead, override command with: uvicorn app:app --host 0.0.0.0 --port 7860
CMD ["python", "ytbot.py"]
