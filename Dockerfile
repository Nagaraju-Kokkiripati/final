# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV CUDA_VISIBLE_DEVICES ""
ENV TF_ENABLE_ONEDNN_OPTS 0
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV PYTHONHASHSEED=0

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create uploads directory with proper permissions
RUN mkdir -p uploads && chmod 777 uploads

# Expose the port the app runs on
EXPOSE 5000

# Run the application with optimized settings
CMD ["gunicorn", \
     "--timeout", "600", \
     "--workers", "1", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--worker-tmp-dir", "/dev/shm", \
     "--bind", "0.0.0.0:5000", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "sample:app"]
