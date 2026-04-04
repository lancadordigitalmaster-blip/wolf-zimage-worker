"""
Z-Image Turbo — RunPod Serverless Handler (Light)
Model downloads on first cold start and caches in /tmp.
Subsequent warm starts load from cache instantly.
"""
import runpod
import torch
import base64
import io
import os
import time

pipe = None

def init():
    global pipe
    from diffusers import FluxPipeline
    from huggingface_hub import snapshot_download

    cache_dir = "/tmp/z-image-turbo"
    model_id = "Tongyi-MAI/Z-Image-Turbo"

    # Download model if not cached
    if not os.path.exists(os.path.join(cache_dir, "model_index.json")):
        print(f"[Z-Image] First run — downloading model (~32GB)...")
        start = time.time()
        snapshot_download(model_id, local_dir=cache_dir)
        elapsed = round(time.time() - start)
        print(f"[Z-Image] Model downloaded in {elapsed}s")
    else:
        print("[Z-Image] Loading from cache...")

    # Load pipeline
    print("[Z-Image] Loading pipeline to GPU...")
    pipe = FluxPipeline.from_pretrained(cache_dir, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Warmup
    pipe("warmup", num_inference_steps=1, width=64, height=64)
    print("[Z-Image] Ready!")


def handler(event):
    inp = event.get("input", {})
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt required"}

    seed = inp.get("seed", -1)
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    w = inp.get("width", 1024)
    h = inp.get("height", 1024)
    steps = inp.get("steps", 8)
    cfg = inp.get("guidance_scale", 3.5)

    start = time.time()
    image = pipe(
        prompt,
        width=w,
        height=h,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    elapsed = round(time.time() - start, 1)
    print(f"[Z-Image] {w}x{h} in {elapsed}s (seed={seed})")

    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "seed": seed,
        "width": w,
        "height": h,
        "elapsed": elapsed,
    }


init()
runpod.serverless.start({"handler": handler})
