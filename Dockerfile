FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install deps only - model downloads at runtime (first cold start)
RUN pip install --no-cache-dir \
    diffusers==0.35.2 \
    transformers==4.51.0 \
    accelerate \
    safetensors \
    runpod \
    Pillow \
    sentencepiece \
    protobuf

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
