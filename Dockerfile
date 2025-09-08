# Use Python 3.8 as base image
FROM python:3.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install pip and upgrade it to the latest version
RUN python -m pip install --upgrade pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install fastapi uvicorn tritonclient[http]

# Copy the rest of the application code into the container
COPY . .

# Expose FastAPI port
EXPOSE 9000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
