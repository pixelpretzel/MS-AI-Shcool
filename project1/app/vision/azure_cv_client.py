# app/vision/azure_cv_client.py

import os
import requests
from googletrans import Translator

PREDICTION_URL = os.getenv("AZURE_CV_PREDICTION_URL")
PREDICTION_KEY = os.getenv("AZURE_CV_PREDICTION_KEY")

_translation_cache: dict[str, str] = {}
_translator = Translator()

def _translate_name_en_to_ko(name: str) -> str:
    """
    Object Detection 결과의 name(영어)을 한국어로 번역.
    - googletrans 사용
    - 실패 시 원본 name 그대로 반환
    """
    if not name:
        return name

    # 1) 캐시 확인
    if name in _translation_cache:
        return _translation_cache[name]

    # 2) googletrans 호출
    try:
        result = _translator.translate(name, src="en", dest="ko")
        translated = result.text
    except Exception:
        # 번역 실패하면 그냥 원본 반환
        translated = name

    # 3) 캐시 저장
    _translation_cache[name] = translated
    return translated


def detect_objects_from_image_path(image_path: str) -> list[dict]:
    """
    Custom Vision의 Prediction URL을 이용해
    로컬 이미지 파일에 대해 Object Detection을 수행한다.

    Returns:
        [
          {
            "name": str,
            "confidence": float,
            "boundingBox": {
              "left": float, "top": float,
              "width": float, "height": float,
            },
          },
          ...
        ]
    """
    if not PREDICTION_URL or not PREDICTION_KEY:
        raise RuntimeError(
            "AZURE_CV_PREDICTION_URL or AZURE_CV_PREDICTION_KEY is not set"
        )

    # 1) 이미지 파일을 바이너리로 읽기
    with open(image_path, "rb") as f:
        image_data = f.read()

    # 2) 문서에서 알려준 대로 헤더 구성
    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream",
    }

    # 3) REST API 호출 (Body = 이미지 바이너리)
    response = requests.post(
        PREDICTION_URL,
        headers=headers,
        data=image_data,
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()

    # 4) 결과 파싱
    # 예상 응답 구조:
    # {
    #   "id": "...",
    #   "project": "...",
    #   "predictions": [
    #     {
    #       "probability": 0.95,
    #       "tagId": "...",
    #       "tagName": "bed",
    #       "boundingBox": {
    #         "left": 0.1, "top": 0.2,
    #         "width": 0.3, "height": 0.4
    #       }
    #     },
    #     ...
    #   ]
    # }
    detections: list[dict] = []

    for pred in result.get("predictions", []):
        box = pred.get("boundingBox", {}) or {}
        detections.append(
            {
                "name": pred.get("tagName"),
                "confidence": float(pred.get("probability", 0.0)),
                "boundingBox": {
                    "left": float(box.get("left", 0.0)),
                    "top": float(box.get("top", 0.0)),
                    "width": float(box.get("width", 0.0)),
                    "height": float(box.get("height", 0.0)),
                },
            }
        )

    return detections

def _resolve_local_path_from_url(image_url: str) -> str:
    # "/static/generated/abcd.png" -> "app/static/generated/abcd.png"
    rel_path = image_url.lstrip("/")  # "static/generated/abcd.png"

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # project1
    local_path = os.path.join(base_dir, "app", rel_path)  # project1/app/static/...
    return local_path


def detect_objects_from_image_url(image_url: str, top_k: int=3) -> list[dict]:
    """
    프론트와 주고받는 imageUrl("/static/generated/xxx.png")을 받아
    실제 로컬 경로로 매핑하고,
    Object Detection 후 confidence 상위 top_k 개만 반환한다.
    """
    image_path = _resolve_local_path_from_url(image_url)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found for detection: {image_path}")

    detections = detect_objects_from_image_path(image_path)

    # 🔥 confidence 기준 내림차순 정렬
    detections_sorted = sorted(
        detections,
        key=lambda x: x.get("confidence", 0),
        reverse=True
    )

    # 상위 top_k만 잘라서 번역 적용
    top_detections = detections_sorted[:top_k]

    translated_detections: list[dict] = []
    for det in top_detections:
        name_en = det.get("name")
        name_ko = _translate_name_en_to_ko(name_en)

        # 구조는 그대로, name만 한국어로 교체
        translated_detections.append(
            {
                **det,
                "name": name_ko,
            }
        )

    return translated_detections
