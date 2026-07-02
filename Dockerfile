# ──────────────────────────────────────────────────────────────
# Dockerfile — CineScope Sentiment Analysis Dashboard
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="CineScope · AI Sentiment Analysis Dashboard"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train model if not already present
RUN python model.py

# Expose Streamlit port
EXPOSE 8501

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
