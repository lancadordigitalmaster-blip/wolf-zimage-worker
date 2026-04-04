"""
Z-Image Turbo — RunPod GPU Pod Handler
HTTP server on port 8080. Model saved to /workspace for persistence.
"""
import torch
import base64
import io
import os
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

pipe = None

def load_model():
    global pipe
    from diffusers import FluxPipeline
    from huggingface_hub import snapshot_download

    model_path = os.environ.get("MODEL_PATH", "/workspace/z-image-turbo")
    index_file = os.path.join(model_path, "model_index.json")

    if not os.path.exists(index_file):
        print(f"[Z-Image] First run — downloading model to {model_path}...")
        start = time.time()
        snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir=model_path)
        print(f"[Z-Image] Downloaded in {round(time.time()-start)}s")
    else:
        print(f"[Z-Image] Loading from cache: {model_path}")

    print("[Z-Image] Loading pipeline to GPU...")
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe("warmup", num_inference_steps=1, width=64, height=64)
    print("[Z-Image] Ready on port 8080!")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ready", "model": "Z-Image Turbo"}).encode())

    def do_POST(self):
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

        w = inp.get("width", 1024)
        h = inp.get("height", 1024)
        steps = inp.get("steps", 8)
        cfg = inp.get("guidance_scale", 3.5)

        start = time.time()
        image = pipe(
            prompt, width=w, height=h,
            num_inference_steps=steps, guidance_scale=cfg,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        elapsed = round(time.time() - start, 1)
        print(f"[Z-Image] {w}x{h} in {elapsed}s")

        self._respond(200, {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "seed": seed, "width": w, "height": h, "elapsed": elapsed,
        })

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress access logs


if __name__ == "__main__":
    load_model()
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("[Z-Image] Serving on port 8080")
    server.serve_forever()
