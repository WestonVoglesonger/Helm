FROM python:3.11-slim

# Set up working directory
WORKDIR /app

ENV PYTHONPATH="/app:${PYTHONPATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Default command
CMD ["python", "-m", "src.main"]