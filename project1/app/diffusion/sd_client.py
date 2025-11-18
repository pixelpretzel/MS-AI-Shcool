# app/diffusion/sd_client.py
import os
from uuid import uuid4

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from app.config import SD_MODEL_ID

_device = "cuda" if torch.cuda.is_available() else "cpu"
_pipe: StableDiffusionPipeline | None = None


def _get_pipeline() -> StableDiffusionPipeline:
    """
    Stable Diffusion v1.5 파이프라인을 한 번만 로드해서 재사용.
    """
    global _pipe

    if _pipe is None:
        dtype = torch.float16 if _device == "cuda" else torch.float32

        _pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
        )
        _pipe = _pipe.to(_device)

        try:
            _pipe.enable_attention_slicing()
        except Exception:
            pass

    return _pipe


def generate_image_from_prompt(
    prompt: str,
    output_dir: str = "app/static/generated",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    width: int = 512,
    height: int = 512,
) -> str:
    """
    Stable Diffusion v1.5로 이미지를 생성하고 저장한 뒤,
    /static 경로로 접근 가능한 URL 반환.
    """
    os.makedirs(output_dir, exist_ok=True)

    pipe = _get_pipeline()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(seed)

    result = pipe(
        prompt,
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
    return url_path

