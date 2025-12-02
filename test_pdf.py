import os
import re
import uuid
import json
import requests
import pytesseract
import pypdfium2 as pdfium
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from docling.document_converter import DocumentConverter

app = FastAPI()

# -------------------------------
# Config
# -------------------------------
MEILI_URL = "http://10.1.0.150:7700/indexes/documents/documents"
OLLAMA_API = "http://10.1.0.150:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

# Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def generate_keywords_from_filename(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    words = re.split(r"[\s._-]+", name)
    return [w for w in words if not w.isdigit() and len(w) > 1]


def ocr_pdf_tesseract(path):
    pdf = pdfium.PdfDocument(path)
    full_text = ""
    for i, page in enumerate(pdf):
        bitmap = page.render(scale=3)
        pil_image = bitmap.to_pil()
        page_text = pytesseract.image_to_string(pil_image, lang="tha+eng")
        full_text += page_text + "\n"
    pdf.close()
    return full_text


def check_filename_keywords(text, filename):
    keywords = generate_keywords_from_filename(filename)
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            return "เนื้อหาสอดคล้อง"
    return "เนื้อหาไม่สอดคล้อง"


def classify_document(markdown_text: str) -> str:
    prompt = f"""
จำแนกเอกสารนี้เป็นหนึ่งในประเภท:

1. ข้อมูลที่เกี่ยวข้องกับการประกอบธุรกิจ บัตรเครดิต  
2. ทะเบียนผู้ถือหุ้นของบริษัทฉบับล่าสุด  
3. เอกสารแสดงฐานะทางการเงิน  
4. มติคณะกรรมการบริษัท / เอกสารอนุมัติ  
5. โครงสร้างองค์กร  
6. โครงสร้างกลุ่มธุรกิจ  
7. นโยบายและคู่มือปฏิบัติงาน  

ตอบเฉพาะหมายเลข + ชื่อประเภท เช่น "3. เอกสารแสดงฐานะทางการเงิน"

เนื้อหา:
{markdown_text[:800]}
"""
    resp = requests.post(
        OLLAMA_API,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "max_tokens": 50},
        timeout=300
    )

    raw = resp.text
    combined = ""
    for line in raw.splitlines():
        try:
            js = json.loads(line)
            combined += js.get("response", "")
        except:
            combined += line

    combined = combined.strip()
    first_line = combined.splitlines()[0] if combined else ""

    match = re.search(r"^\d+\.\s*.+$", first_line)
    return match.group(0).strip() if match else first_line


def extract_share_ratio(markdown):
    pattern = r'(\d{1,3}(?:\.\d+)?\s*%)'
    matches = re.findall(pattern, markdown)
    return matches or "ไม่พบสัดส่วนผู้ถือหุ้น"


def extract_license(markdown):
    pattern = r'(ใบอนุญาต[^\n]+|License\s*No\.\s*\S+)'
    matches = re.findall(pattern, markdown)
    return matches or "ไม่พบ license"

def extract_contact_time(markdown):
    """
    ดึงช่วงเวลาในการติดต่อลูกค้า
    สมมติรูปแบบเช่น "เวลาโทร: 09:00-17:00" หรือ "Contact time: 09:00-17:00"
    """
    pattern = r'(เวลา\s*(?:โทร|ติดต่อ)[^\n:]*[:\s]*\d{1,2}:\d{2}-\d{1,2}:\d{2}|Contact time[^\n]*[:\s]*\d{1,2}:\d{2}-\d{1,2}:\d{2})'
    matches = re.findall(pattern, markdown)
    return matches or "ไม่พบช่วงเวลาติดต่อ"

# --------------------------------------------------
# MAIN PROCESS FUNCTION (ไม่ print แต่ return dict)
# --------------------------------------------------
def process_pdf_to_meili(pdf_path: str, filename: str):

    # 1) แปลง PDF → Markdown
    converter = DocumentConverter()
    try:
        result = converter.convert(pdf_path)
        markdown_content = result.document.export_to_markdown()

        if not markdown_content.strip():
            raise Exception("ไม่มีข้อความจาก Markdown")
        text_for_ai = markdown_content

    except:
        text_for_ai = ocr_pdf_tesseract(pdf_path)

    # 2) เช็คชื่อไฟล์
    filename_check = check_filename_keywords(text_for_ai, filename)

    # 3) AI จำแนกเอกสาร
    doc_type = classify_document(text_for_ai)

    # 4) Extract ตามประเภทใหม่
    if doc_type.startswith("1"):
        extracted_info = extract_contact_time(text_for_ai)
    elif doc_type.startswith("2"):
        extracted_info = extract_share_ratio(text_for_ai)
    elif doc_type.startswith("3"):
        extracted_info = "ตรวจเลขฐานการเงิน"
    elif doc_type.startswith("4"):
        extracted_info = extract_license(text_for_ai)
    elif doc_type.startswith("6"):
        extracted_info = extract_share_ratio(text_for_ai)  # ประเภท 6 ต้องการตัวเลขสัดส่วนผู้ถือหุ้น
    else:
        extracted_info = "ไม่มีข้อมูลที่ต้องดึง"



    # 5) เตรียม JSON 
    doc_id = "DL" + str(uuid.uuid4())
    now_str = datetime.now().isoformat()

    data = {
        "id": doc_id,
        "filename_only": filename,
        "file_path": pdf_path,
        "folder_path": os.path.dirname(pdf_path),
        "date": now_str,
        "filename_check": filename_check,
        "doc_type": doc_type,
        "extracted_info": extracted_info
    }
    return data

# --------------------------------------------------
# FastAPI Endpoints
# --------------------------------------------------

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):

    # เซฟไฟล์ลงเครื่องก่อน
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    result = process_pdf_to_meili(temp_path, file.filename)

    os.remove(temp_path)
    return result


@app.get("/")
def home():
    return {
        "message": "FastAPI PDF → AI → MeiliSearch is running",
        "usage": "POST /process-pdf (upload PDF)"
    }
