FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/sklearn build (if needed) and openpyxl
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir openpyxl && \
    pip install --no-cache-dir -r requirements.txt

# Source code & data
COPY backend/ backend/
COPY data/ data/
COPY expectations/ expectations/
COPY scripts/ scripts/

# Create models/ dir for SVM cache
RUN mkdir -p models

# Run data preparation (generates expectations/suite.json + data/normal_samples.npy)
RUN python scripts/prepare_data.py

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
