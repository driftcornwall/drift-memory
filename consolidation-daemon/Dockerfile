FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY consolidation_engine.py .
COPY incremental_merkle.py .
COPY incremental_fingerprint.py .

# State directory for persistent incremental data
RUN mkdir -p /app/state

EXPOSE 8083

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8083"]
