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
    model_path = os.environ.get("MODEL_PATH", "/models/z-image-turbo")
    print(f"[Z-Image] Loading from {model_path}...")
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
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
    image = pipe(prompt, width=w, height=h, num_inference_steps=steps, guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    elapsed = round(time.time() - start, 1)
    return {"image": base64.b64encode(buf.getvalue()).decode(), "seed": seed, "width": w, "height": h, "elapsed": elapsed}

init()
runpod.serverless.start({"handler": handler})
