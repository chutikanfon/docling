import io
import json
import re
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from typhoon_ocr import ocr_document
import fitz  # PyMuPDF
import os

app = FastAPI(title="OrgChart OCR API (Typhoon 7B via Ollama)")

# ----------------------------
# PDF / Image → List of PIL Images
# ----------------------------
def pdf_or_image_to_images(file_bytes: bytes, filename):
    images = []
    filename_str = filename.decode() if isinstance(filename, bytes) else str(filename)

    # PDF
    if filename_str.lower().endswith(".pdf"):
        try:
            doc = fitz.open("pdf", file_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {e}")

        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        doc.close()
    # PNG/JPG
    elif filename_str.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    else:
        raise HTTPException(status_code=400, detail="File must be PDF, PNG, or JPG")
    return images

# ----------------------------
# Extract JSON from OCR Markdown
# ----------------------------
def extract_json(markdown: str):
    try:
        json_str = re.search(r"\{.*\}", markdown, re.S).group(0)
        return json.loads(json_str)
    except Exception:
        return {"natural_text": markdown}

# ----------------------------
# Text processing helpers
# ----------------------------
def classify_text_block(text: str):
    name_pattern = r"[A-Za-zก-๙]{2,50}\s+[A-Za-zก-๙]{2,50}"
    position_keywords = ["Manager", "Director", "หัวหน้า", "ผู้จัดการ", "CEO", "CTO", "COO"]
    if any(k.lower() in text.lower() for k in position_keywords):
        return "position"
    elif re.match(name_pattern, text.strip()):
        return "name"
    else:
        return "other"

def extract_blocks_from_text(text: str):
    blocks = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            blocks.append({"text": line, "type": classify_text_block(line)})
    return blocks

def build_name_position_pairs(blocks):
    pairs = []
    last_position = None
    for b in blocks:
        if b["type"] == "position":
            last_position = b["text"]
        elif b["type"] == "name" and last_position:
            pairs.append({"position": last_position, "name": b["text"]})
    return pairs

def build_hierarchy(blocks):
    hierarchy = {}
    stack = []
    for i, b in enumerate(blocks):
        if b["type"] != "position":
            continue
        level = i
        while len(stack) > level:
            stack.pop()
        node = {b["text"]: {}}
        if not stack:
            hierarchy.update(node)
            stack.append(hierarchy[b["text"]])
        else:
            parent = stack[-1]
            parent[b["text"]] = {}
            stack.append(parent[b["text"]])
    return hierarchy

# ----------------------------
# Process one page (async) with temp file
# ----------------------------
async def process_page(img: Image.Image, page_number: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        img.save(tmp_file, format="PNG")
        tmp_path = tmp_file.name

    try:
        markdown = ocr_document(
            tmp_path,  # ส่ง path ให้ Typhoon OCR
            base_url="http://10.1.0.150:11434/v1",
            api_key="ollama",
            model="scb10x/typhoon-ocr-7b"
        )
    finally:
        os.remove(tmp_path)  # ลบ temp file ทันที

    parsed = extract_json(markdown)
    raw_text = parsed.get("natural_text", markdown)

    blocks = extract_blocks_from_text(raw_text)
    pairs = build_name_position_pairs(blocks)
    hierarchy = build_hierarchy(blocks)

    return {
        "page": page_number,
        "raw_text": raw_text,
        "blocks": blocks,
        "pairs": pairs,
        "hierarchy": hierarchy
    }

# ----------------------------
# Main async processing (PDF/Image)
# ----------------------------
async def process_file_async(file_bytes: bytes, filename):
    images = pdf_or_image_to_images(file_bytes, filename)

    pages = []
    for idx, image in enumerate(images):
        page_data = await process_page(image, idx + 1)
        pages.append(page_data)

    all_pairs = []
    all_blocks = []
    all_hierarchy = {}
    for p in pages:
        all_pairs.extend(p["pairs"])
        all_blocks.extend(p["blocks"])
        all_hierarchy.update(p["hierarchy"])

    return {
        "total_pages": len(images),
        "combined_pairs": all_pairs,
        "combined_blocks": all_blocks,
        "combined_hierarchy": all_hierarchy,
        "pages": pages
    }

# ----------------------------
# API endpoints
# ----------------------------
@app.post("/process-org-chart")
async def process_org_chart_endpoint(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        result = await process_file_async(file_bytes, file.filename)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "ocr_model": "typhoon-ocr-7b via Ollama"}
