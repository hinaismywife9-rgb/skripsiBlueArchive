FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific versions for stability
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p sentiment_models results logs data

# Expose port for Streamlit
EXPOSE 8501

# Run verify setup first, then start Streamlit app
CMD ["bash", "-c", "python verify_setup.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
