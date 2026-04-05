FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# PyTorch 2.5.1 + diffusers from source (ZImagePipeline)
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers \
    transformers==4.51.0 \
    accelerate \
    safetensors \
    Pillow \
    sentencepiece \
    protobuf \
    realesrgan \
    basicsr \
    gfpgan

COPY handler.py /app/handler.py
COPY boot.sh /app/boot.sh
RUN chmod +x /app/boot.sh

ENV MODEL_PATH=/workspace/z-image-turbo

CMD ["bash", "-c", "bash /app/boot.sh && sleep infinity"]
