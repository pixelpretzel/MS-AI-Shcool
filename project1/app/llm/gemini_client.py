# app/llm/gemini_client.py
import google.generativeai as genai

from app.config import GEMINI_API_KEY


def _configure_client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Check your .env or environment variables.")
    genai.configure(api_key=GEMINI_API_KEY)


def build_sd_prompt_from_text(raw_text: str) -> str:
    """
    OCR로 추출된 텍스트(raw_text)를 바탕으로
    Stable Diffusion용 프롬프트를 만들어주는 함수.
    """
    _configure_client()

    system_prompt = """
You are a professional prompt engineer for text-to-image models.
Your job is to convert input text into a clean, detailed visual prompt.

Requirements:
- Output ONLY the final prompt, no explanations.
- Focus on clear visual description (who/what/where/when/mood).
- Style: warm, child-friendly illustrated book, soft colors.
- Do NOT mention 'Stable Diffusion' or 'prompt' in the output.
"""

    user_prompt = f"""
Input text:
{raw_text}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [
            system_prompt,
            user_prompt,
        ]
    )

    return (response.text or "").strip()

