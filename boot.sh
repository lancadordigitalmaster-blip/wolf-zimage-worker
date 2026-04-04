#!/bin/bash
echo "[Boot] Starting Z-Image handler..."
cd /workspace
nohup python3 /app/handler.py > /workspace/handler.log 2>&1 &
echo "[Boot] Handler PID: $!"
