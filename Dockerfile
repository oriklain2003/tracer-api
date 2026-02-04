# =============================================================================
# Dockerfile for Anomaly Detection API Service
# Optimized for AWS App Runner
# =============================================================================

# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies with retry logic
RUN apt-get update --fix-missing || apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p realtime \
    && mkdir -p training_ops \
    && mkdir -p service \
    && mkdir -p ml_deep/output \
    && mkdir -p ml_deep_cnn/output \
    && mkdir -p ml_hybrid/output \
    && mkdir -p ml_transformer/output \
    && mkdir -p mlboost/output

# Expose port (AWS App Runner will set the PORT environment variable)
EXPOSE 8000


# Run the application
# AWS App Runner expects the service to listen on 0.0.0.0:8000
CMD ["python", "app.py"]
