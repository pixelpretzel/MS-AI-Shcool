# app/config.py
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로딩
load_dotenv()

AZURE_CV_ENDPOINT = os.getenv("AZURE_CV_ENDPOINT")
AZURE_CV_KEY = os.getenv("AZURE_CV_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# SD_API_KEY = os.getenv("SD_API_KEY")

# Stable Diffusion 기본 모델
# SD_MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")

SD_MODEL_ID = os.getenv("SD_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
