FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Download model at build time so startup is fast
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-xsmall')"

EXPOSE 8082

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]
