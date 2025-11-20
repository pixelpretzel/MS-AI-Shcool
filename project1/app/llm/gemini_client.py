# app/llm/gemini_client.py
import os
import google.generativeai as genai

from app.config import GEMINI_API_KEY


def build_sd_prompt_from_text(ocr_text):
    
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

    user_prompt = f"Input text: {ocr_text}"

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        [
            system_prompt,
            user_prompt,
        ]
    )

    return (response.text or "").strip()

def build_ai_question(ocr_text):
    question_prompt = """
**Role:** You are a highly specialized AI designed to serve as the core function of an **'AI-Powered Reading Companion'** for young children.

**Objective:** Your primary task is to analyze a given children's story text (fairytale/picture book content) and generate data that specifically aids the language comprehension and developmental needs of children aged **3 to 7 years old**, who require visual information for better understanding.

**Core Output Directives:**
You must produce a structured output focused on promoting active engagement and linguistic development.

---

### **Language Development Questions (5 Total)**

* Generate **exactly five (5) high-quality questions** to facilitate an engaging dialogue with the child.
* The questions must cover the following **five mandatory and distinct developmental areas** to ensure diversity and creativity:
    1.  **Text Comprehension & Recall:** A question focused on **recalling the main content, characters, or setting** (Who, What, Where, When).
    2.  **Inference & Emotional Literacy:** A question about **inferring a character's feelings, motivations, or intentions**, requiring the child to understand 'why' a character acted a certain way or 'how' they felt.
    3.  **Creative Prediction & Alternative Ending:** A question that encourages the child to **imagine what happens next** in the story or **propose a new, creative outcome or alternative ending.**
    4.  **Vocabulary & Sensory Detail:** A question that prompts the child to **use a specific, newly introduced vocabulary word** from the text, or describe the story using **sensory details** (e.g., "What colors did you see?" "What sound did X make?").
    5.  **Personal Connection & Role-Playing ('What if I were'):** A personalized question (e.g., **"If you were the character, what would you do differently?"** or **"What part of the story reminds you of your own experience?"**).
    * *Example Output Format (MUST BE IN INFORMAL KOREAN. 반말로 작성해줘.):*
        * Q1. [질문 텍스트(반말)]
        * Q2. [질문 텍스트(반말)]
        * Q3. [질문 텍스트(반말)]
        * Q4. [질문 텍스트(반말)]
        * Q5. [질문 텍스트(반말)]

---
"""

    user_prompt = f"Input text: {ocr_text}"

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(                                              [                                                                               question_prompt,                                                            user_prompt,                                                            ]                                                                       )

    return response.text.strip()
