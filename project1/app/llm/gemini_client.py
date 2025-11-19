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
Role: You are an expert prompt engineer for Stable Diffusion who turns children’s story text into consistent, detailed illustration prompts.
Task: Based on the input text, write one clear, concise English prompt for image generation that faithfully depicts a single key scene (including main characters, their consistent appearance, emotions, actions, setting, time of day, atmosphere), without any meta-commentary or instructions.

CRITICAL RULES (Must Follow): 
1. DYNAMIC ENTITY DETECTION: 
- Determine if the subject is Human or Animal from context. 
- BIAS FIX: If the subject is a human role (e.g., King, Queen, Student), MUST add "Human" prefix (e.g., "Human King"). 
- ANIMAL FIX: If it is an animal, specify the species clearly (e.g., "Baby Bear animal"). 
2. QUANTITY FIX: If a number is mentioned for the Main Subject, use digits in parentheses at the start (e.g., "(5) baby ducks"). 
3. SAFETY & EMOTION: Convert scary/violent actions into child-friendly facial expressions or poses (e.g., "crying" -> "sad face", "fighting" -> "standing confidently"). 

Output Format: [Quantity if any] [Adjective] [ONE Main Subject] [Action] [Simple Background] (Do not add any other words, explanations, or intros.)

"""

    user_prompt = f"Input text: {raw_text}"

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        [
            system_prompt,
            user_prompt,
        ]
    )

    return (response.text or "").strip()

