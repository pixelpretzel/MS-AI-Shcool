# app/main.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ocr.azure_ocr import extract_text_from_image
from app.llm.gemini_client import build_sd_prompt_from_text

app = FastAPI()

# static / templates 설정
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    메인 페이지: 이미지 업로드 폼
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "ocr_text": None, "error": None},
    )


@app.post("/ocr", response_class=HTMLResponse)
async def ocr_image(request: Request, file: UploadFile = File(...)):
    try:
        # 파일 바이트 읽기
        image_bytes = await file.read()

        # Azure OCR 실행
        ocr_text = extract_text_from_image(image_bytes)

        sd_prompt = build_sd_prompt_from_text(ocr_text)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ocr_text": ocr_text,
                "sd_prompt": sd_prompt,
                "error": None,
                "filename": file.filename,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ocr_text": None,
                "sd_prompt": None,
                "error": str(e),
                "filename": getattr(file, "filename", None),
            },
        )

