# Use a PyTorch base image
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Install system dependencies (ffmpeg is needed for torchaudio and ffprobe for metadata)
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set the default command to run prediction
CMD ["python", "predict.py"]
