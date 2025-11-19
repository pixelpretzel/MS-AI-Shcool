# app/diffusion/sd_client.py
import os
from uuid import uuid4

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from app.config import SD_MODEL_ID

_device = "cuda" if torch.cuda.is_available() else "cpu"
_pipe: StableDiffusionXLPipeline | None = None

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# app/static/generated
_DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, "static", "generated")

def _get_pipeline() -> StableDiffusionXLPipeline:
    global _pipe

    if _pipe is None:

        if not SD_MODEL_ID:
            raise RuntimeError("SD_MODEL_ID is not set or empty")

        dtype = torch.float16 if _device == "cuda" else torch.float32

        print("[SDXL] Loading pipeline from:", SD_MODEL_ID)

        _pipe = StableDiffusionXLPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
        )

        _pipe = _pipe.to(_device)

        try:
            _pipe.enable_attention_slicing()
        except Exception:
            pass
        
        try:
            _pipe.enable_vae_tiling()
        except Exception:
            pass

    return _pipe


def generate_image_from_prompt(
    prompt: str,
    output_dir: str | None = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 10.0,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
) -> tuple[str, str]:
    """
    SDXL로 이미지를 생성하고 저장한 뒤,
    (url_path, file_path)를 반환
    """

    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR


    os.makedirs(output_dir, exist_ok=True)

    pipe = _get_pipeline()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(seed)

    # 기본 negative prompt
    #negative_prompt = "low quality, blurry, distorted, text, watermark, signature"
    negative_prompt = "deformed face, distorted face, asymmetrical face, extra eyes, extra limbs, extra fingers, long neck, disfigured, mutated, low quality, blurry, distorted, text, cropped, ugly, deformed, disfigured, poor anatomy, missing limbs, malformed hands, mutated, poorly drawn eyes, unsettling, monochrome, grayscale, realistic, photography, photo"

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    image: Image.Image = result.images[0]

    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)

    url_path = f"/static/generated/{filename}"
    return url_path, filepath

