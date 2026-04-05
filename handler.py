"""
Z-Image Turbo — RunPod GPU Pod Handler
Generate 1024 base + PIL Lanczos upscale to target resolution
Key: guidance_scale=0.0, steps=9, DiffusionPipeline (auto-detects ZImagePipeline)
Proven versions: PyTorch 2.5.1, diffusers 0.36.0, transformers 4.51.0
"""
import torch
import base64
import io
import os
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image

pipe = None

def load_model():
    global pipe
    from huggingface_hub import snapshot_download
    from diffusers import DiffusionPipeline

    model_path = os.environ.get("MODEL_PATH", "/workspace/z-image-turbo")
    index_file = os.path.join(model_path, "model_index.json")

    if not os.path.exists(index_file):
        print(f"[Z-Image] Downloading model to {model_path}...")
        start = time.time()
        snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir=model_path)
        print(f"[Z-Image] Downloaded in {round(time.time()-start)}s")
    else:
        print(f"[Z-Image] Loading from cache: {model_path}")

    print("[Z-Image] Loading pipeline to GPU...")
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        text_encoder_2=None,
        tokenizer_2=None,
        image_encoder=None,
        feature_extractor=None,
    )
    pipe.to("cuda")
    print("[Z-Image] Warmup...")
    pipe("test", num_inference_steps=1, width=64, height=64, guidance_scale=0.0)
    print("[Z-Image] Ready on 0.0.0.0:8080!")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ready", "model": "Z-Image Turbo"}).encode())

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            inp = body.get("input", body)

            prompt = inp.get("prompt", "")
            if not prompt:
                self._respond(400, {"error": "prompt required"})
                return

            seed = inp.get("seed", -1)
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()

            target_w = inp.get("width", 1024)
            target_h = inp.get("height", 1024)
            steps = inp.get("steps", 9)
            cfg = inp.get("guidance_scale", 0.0)

            # Generate at 1024 max, keep aspect ratio
            if target_w <= 1024 and target_h <= 1024:
                gen_w, gen_h = target_w, target_h
            elif target_w >= target_h:
                gen_w = 1024
                gen_h = max(64, int(1024 * target_h / target_w) // 16 * 16)
            else:
                gen_h = 1024
                gen_w = max(64, int(1024 * target_w / target_h) // 16 * 16)

            start = time.time()
            image = pipe(
                prompt, width=gen_w, height=gen_h,
                num_inference_steps=steps, guidance_scale=cfg,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]

            # Upscale if target is larger than generation
            if target_w > gen_w or target_h > gen_h:
                image = image.resize((target_w, target_h), Image.LANCZOS)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            elapsed = round(time.time() - start, 1)
            print(f"[Z-Image] {gen_w}x{gen_h}->{target_w}x{target_h} in {elapsed}s")

            self._respond(200, {
                "image": base64.b64encode(buf.getvalue()).decode(),
                "seed": seed, "width": target_w, "height": target_h, "elapsed": elapsed,
            })
        except Exception as e:
            print(f"[Z-Image] Error: {e}")
            self._respond(500, {"error": str(e)})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    load_model()
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("[Z-Image] Serving on 0.0.0.0:8080")
    server.serve_forever()
