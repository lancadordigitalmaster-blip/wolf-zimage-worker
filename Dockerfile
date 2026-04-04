FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install compatible versions
RUN pip install --no-cache-dir \
    diffusers==0.34.0 \
    transformers==4.51.0 \
    accelerate \
    safetensors \
    runpod \
    Pillow \
    sentencepiece \
    protobuf

# Download Z-Image Turbo model at build time
RUN python3 -c "\
from diffusers import FluxPipeline; \
import torch; \
print('Downloading Z-Image Turbo...'); \
pipe = FluxPipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo', torch_dtype=torch.bfloat16); \
pipe.save_pretrained('/models/z-image-turbo'); \
print('Model saved!')"

ENV MODEL_PATH=/models/z-image-turbo

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
