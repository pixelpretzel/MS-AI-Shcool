# app/main.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ocr.azure_ocr import extract_text_from_image

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
    """
    이미지 파일을 받아 Azure OCR을 수행하고 결과 텍스트를 보여준다.
    """
    try:
        # 파일 바이트 읽기
        image_bytes = await file.read()

        # Azure OCR 실행
        ocr_text = extract_text_from_image(image_bytes)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ocr_text": ocr_text,
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
                "error": str(e),
                "filename": getattr(file, "filename", None),
            },
        )

