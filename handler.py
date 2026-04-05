"""
Z-Image Turbo — RunPod GPU Pod Handler
Generate 1024x1024 + Real-ESRGAN 2x upscale = 2048x2048 output
Key: guidance_scale=0.0, steps=9, ZImagePipeline, pipe.to("cuda")
"""
import torch
import base64
import io
import os
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

pipe = None
upsampler = None

def load_model():
    global pipe, upsampler
    from huggingface_hub import snapshot_download

    model_path = os.environ.get("MODEL_PATH", "/workspace/z-image-turbo")
    index_file = os.path.join(model_path, "model_index.json")

    if not os.path.exists(index_file):
        print(f"[Z-Image] Downloading model to {model_path}...")
        start = time.time()
        snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir=model_path)
        print(f"[Z-Image] Downloaded in {round(time.time()-start)}s")
    else:
        print(f"[Z-Image] Loading from cache: {model_path}")

    from diffusers import ZImagePipeline
    print("[Z-Image] Loading ZImagePipeline to GPU...")
    pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Load Real-ESRGAN upscaler
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_file = "/workspace/RealESRGAN_x4plus.pth"
        if not os.path.exists(model_file):
            import urllib.request
            print("[Upscale] Downloading Real-ESRGAN model...")
            urllib.request.urlretrieve(model_url, model_file)
        upsampler = RealESRGANer(scale=4, model_path=model_file, model=esrgan_model, half=True, device="cuda")
        print("[Upscale] Real-ESRGAN loaded!")
    except Exception as e:
        print(f"[Upscale] Not available: {e} — upscale disabled, will use PIL resize")
        upsampler = None

    print("[Z-Image] Warmup...")
    pipe("test", num_inference_steps=1, width=64, height=64, guidance_scale=0.0)
    print("[Z-Image] Ready on 0.0.0.0:8080!")


def upscale_image(pil_image, target_w, target_h):
    """Upscale PIL image to target resolution."""
    import numpy as np
    from PIL import Image

    if upsampler:
        # Real-ESRGAN: 4x upscale then resize to target
        img_np = np.array(pil_image)
        output, _ = upsampler.enhance(img_np, outscale=2)
        result = Image.fromarray(output)
        if result.size != (target_w, target_h):
            result = result.resize((target_w, target_h), Image.LANCZOS)
        return result
    else:
        # Fallback: PIL Lanczos resize
        return pil_image.resize((target_w, target_h), Image.LANCZOS)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        info = {"status": "ready", "model": "Z-Image Turbo", "upscaler": "Real-ESRGAN" if upsampler else "PIL"}
        self.wfile.write(json.dumps(info).encode())

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

            # Generate at 1024 base, upscale if target is larger
            gen_w = min(target_w, 1024)
            gen_h = min(target_h, 1024)
            # Keep aspect ratio at generation
            if target_w > target_h:
                gen_w = 1024
                gen_h = max(64, int(1024 * target_h / target_w) // 16 * 16)
            elif target_h > target_w:
                gen_h = 1024
                gen_w = max(64, int(1024 * target_w / target_h) // 16 * 16)

            start = time.time()
            image = pipe(
                prompt, width=gen_w, height=gen_h,
                num_inference_steps=steps, guidance_scale=cfg,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]

            # Upscale if needed
            if target_w > gen_w or target_h > gen_h:
                image = upscale_image(image, target_w, target_h)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            elapsed = round(time.time() - start, 1)
            print(f"[Z-Image] {gen_w}x{gen_h}→{target_w}x{target_h} in {elapsed}s")

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
