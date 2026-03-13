FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml .
RUN pip3 install --no-cache-dir "torch" "torchvision" "timm" "numpy" "scikit-learn" "pandas" "scipy" "pillow" "matplotlib" "tqdm" "scikit-image" "opencv-python" "openpyxl" "pyyaml" "lightgbm"

COPY . .
WORKDIR /workspace

CMD ["python3", "-c", "print('Promega classifier ready. Run: python split_data.py; python image_classifier/training/train_model_accuracy.py')"]
