FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Proven working versions (2026-04-05)
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    diffusers==0.36.0 \
    transformers==4.51.0 \
    accelerate \
    safetensors \
    Pillow \
    sentencepiece \
    protobuf

COPY handler.py /app/handler.py
COPY boot.sh /app/boot.sh
RUN chmod +x /app/boot.sh

ENV MODEL_PATH=/workspace/z-image-turbo

CMD ["bash", "-c", "bash /app/boot.sh && sleep infinity"]
