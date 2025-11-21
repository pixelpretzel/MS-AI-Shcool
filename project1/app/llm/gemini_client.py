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


# --------------------------
# 💬 아이 답장에 리액션하는 채팅용 함수
# --------------------------

STORY_TEACHER_SYSTEM_PROMPT = """
너는 그림책을 함께 읽어주는 다정한 선생님이야.
3~7살 아이와 이야기 나누듯이 대화해.
항상 편안한 반말을 쓰고, 짧게 1~3문장 정도로 대답해.
아이의 대답을 잘 받아주고, 가끔은 다시 물어보면서 대화를 이어가.
AI나 모델이라는 말은 절대 하지 마.
"""


def build_chat_reaction(child_message: str, history: list[dict]) -> str:
    """
    아이가 보낸 최신 메시지 + 이전 history를 바탕으로
    '선생님' 역할의 반말 리액션을 생성.
    history 형식 예:
      [
        {"role": "assistant", "content": "늑대가 나타나서 아기 돼지는 기분이 어땠을까?"},
        {"role": "user", "content": "무서웠을 것 같아."}
      ]
    """

    model = genai.GenerativeModel("gemini-2.5-flash")

    # 대화 로그를 텍스트로 이어붙이기
    conv_lines = []

    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")

        if role == "user":
            conv_lines.append(f"아이: {content}")
        elif role == "assistant":
            conv_lines.append(f"선생님: {content}")
        else:
            conv_lines.append(content)

    # 최신 아이 메시지는 history 바깥에서 받은 것으로 처리
    conv_lines.append(f"아이: {child_message}")

    conversation_text = "\n".join(conv_lines)

    prompt = f"""
{STORY_TEACHER_SYSTEM_PROMPT}

아래는 지금까지의 대화야. 마지막 줄의 '아이' 말에 이어서,
'선생님' 입장에서 따뜻하게 반말로 1~3문장 정도로 대답해줘.

대화:
{conversation_text}

주의사항:
- 아이의 말을 먼저 공감해 주고, 필요하면 쉬운 질문을 한 번 더 해 줘.
- 이모지 쓰지 마.
"""

    response = model.generate_content(prompt)
    return (response.text or "").strip()

def summarize_chat_history(history: list[dict]) -> str:
    """
    아이와 선생님 사이의 대화 history를 받아
    아이가 어떤 생각/감정을 말했는지 중심으로 짧게 요약해준다.

    history 예:
    [
      {"role": "assistant", "content": "늑대가 나타났을 때 아기 돼지는 어떤 기분이었을까?"},
      {"role": "user", "content": "무서웠을 것 같아."},
      {"role": "assistant", "content": "그렇구나, 무서웠겠구나. 너라면 어떻게 했을 것 같아?"},
      {"role": "user", "content": "나는 도망갔을 것 같아."}
    ]
    """

    model = genai.GenerativeModel("gemini-2.5-flash")

    conv_lines = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            conv_lines.append(f"아이: {content}")
        elif role == "assistant":
            conv_lines.append(f"선생님: {content}")
        else:
            conv_lines.append(content)

    convo_text = "\n".join(conv_lines)

    prompt = f"""
너는 3~7살 아이와 그림책을 읽고 대화한 내용을 정리해 주는 선생님이야.

아래는 선생님(assistant)과 아이(user)의 대화 기록이야.
이 대화를 바탕으로,
1) 아이가 어떤 감정/생각/관점을 말했는지
2) 어떤 주제(상황)에 대해 이야기했는지
를 중심으로 짧게 요약해줘.

형식:
- 2~4문장 정도의 간단한 한국어 문단
- 아이가 한 말은 "아이의 말에 따르면 ~" 이런 식으로 정리해줘.
- 전체적인 분위기(재밌어함, 무서워함, 공감함 등)도 한 줄 언급해줘.
- 반말 말고, 교사용 기록처럼 정중한 서술체로 작성해.

대화 기록:
{convo_text}
"""

    response = model.generate_content(prompt)
    return (response.text or "").strip()
