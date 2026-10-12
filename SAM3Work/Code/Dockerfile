FROM python:3.12-slim

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install einops
RUN pip install matplotlib
RUN pip install pillow
RUN pip install triton

RUN git clone https://github.com/facebookresearch/sam3.git /sam3
WORKDIR /sam3
RUN pip install --no-cache-dir -e .
RUN pip install einops pycocotools psutil

WORKDIR /app

# Copy the Python script into the container
COPY sam3_test.py /app/sam3_test.py 
# Run the script when the container starts
CMD ["python", "sam3_test.py"]
