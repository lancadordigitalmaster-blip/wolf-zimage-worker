#!/bin/bash
pip install -q diffusers==0.36.0 transformers==4.51.0 accelerate safetensors Pillow sentencepiece protobuf
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
cd /workspace && nohup python3 handler.py > /workspace/handler.log 2>&1 &
echo "Handler started"
