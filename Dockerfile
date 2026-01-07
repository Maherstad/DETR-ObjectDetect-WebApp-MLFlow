# Multi-stage build for efficient image size
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY . .

# Create MLflow directories
RUN mkdir -p mlruns && \
    mkdir -p assets

# Make Python packages available
ENV PATH=/root/.local/bin:$PATH

# Expose ports
EXPOSE 5050 7100 8501

# Default command (can be overridden in docker-compose)
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./mlruns", "--host", "0.0.0.0", "--port", "5050"]
