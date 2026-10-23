FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    numpy \
    einops \
    matplotlib \
    pillow \
    triton \
    pycocotools \
    psutil \
    tqdm \
    opencv-python 

RUN git clone https://github.com/facebookresearch/sam3.git /sam3
WORKDIR /sam3
RUN pip install --no-cache-dir -e ".[train]"

WORKDIR /app

COPY sam3_test.py /app/sam3_test.py 

CMD ["python", "sam3_test.py"]