FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install compatible versions - diffusers 0.35+ has ZImageTransformer2DModel
RUN pip install --no-cache-dir \
    diffusers==0.35.2 \
    transformers==4.51.0 \
    accelerate \
    safetensors \
    runpod \
    Pillow \
    sentencepiece \
    protobuf

# Download Z-Image Turbo model files manually via huggingface_hub
# (avoids FluxPipeline import issues during build - no GPU available)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
print('Downloading Z-Image Turbo...'); \
snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir='/models/z-image-turbo'); \
print('Model saved!')"

ENV MODEL_PATH=/models/z-image-turbo

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
