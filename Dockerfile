# Use the official Python 3.6.5 image as a base
FROM python:3.6.5

# RUN apt-get update && apt-get install -y \
#     git \
#     build-essential \
#     cmake \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     wget \
#     && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install pip and upgrade it to the latest version
RUN python -m pip install --upgrade pip

# Install PyTorch 1.5.0 for CPU
RUN pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install the necessary Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
# COPY . .

# # Expose port (if you need to run a web service)
# EXPOSE 8080

# # Set the default command to run P2PNet or another entrypoint
# CMD ["python", "main.py"]
