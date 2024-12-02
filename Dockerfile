FROM python:3.9.18-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages with specific version constraints
RUN pip install --no-cache-dir --upgrade pip && \
    # Install PyTorch separately with extended timeout
    pip install --no-cache-dir --timeout 1000 torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    # Install remaining requirements
    pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVER_URL=http://172.17.149.236/api


EXPOSE 8503

CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0"]
