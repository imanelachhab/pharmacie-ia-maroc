"""API de reconnaissance de medicaments a partir d'une photo de boite (Challenge 2).

Lancer :  python -m uvicorn app.main:app --reload      puis http://127.0.0.1:8001  (ou --port 8001)
Docs   :  /docs
"""
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .matcher import Matcher
from .ocr import OCR

KB_DIR = Path(os.environ.get("KB_DIR", r"C:\Users\DELL\dataset_challenge\kb"))
STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_BYTES = 12 * 1024 * 1024

app = FastAPI(title="Reconnaissance de boîtes de médicaments", version="1.0.0",
              description="Photo de boîte -> OCR (français + arabe) -> correspondance avec la base de médicaments.")

matcher = Matcher(KB_DIR)
_ocr: Optional[OCR] = None


def get_ocr() -> OCR:
    """charge le moteur OCR au premier appel (EasyOCR telecharge ses modeles la premiere fois)"""
    global _ocr
    if _ocr is None:
        _ocr = OCR()
    return _ocr


def decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Fichier illisible : envoyez une image JPEG, PNG ou WebP.")
    return img


@app.post("/recognize", tags=["Reconnaissance"], summary="Reconnaît le médicament d'une photo de boîte")
async def recognize(image: UploadFile = File(...), top_k: int = Query(5, ge=1, le=10),
                    fast: bool = Query(False, description="lecture plus rapide, un peu moins précise"),
                    debug: bool = Query(False, description="renvoie aussi le texte lu par l'OCR")):
    data = await image.read()
    if not data:
        raise HTTPException(400, "Fichier vide")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image trop lourde (12 Mo maximum)")
    img = decode(data)
    t0 = time.perf_counter()
    try:
        ocr = get_ocr().read(img, fast=fast)
    except RuntimeError as e:
        raise HTTPException(503, f"Moteur OCR indisponible : {e}")
    t1 = time.perf_counter()
    res = matcher.match(ocr.lines, top_k)
    t2 = time.perf_counter()
    out = {
        "status": res["status"],                 # recognized | uncertain | not_found | no_text
        "best": res["candidates"][0] if res["candidates"] else None,
        "candidates": res["candidates"],
        "detected": {"dosages": res["ocr_dosages"], "pack_sizes": res["ocr_packs"],
                     "languages": sorted({l.script for l in ocr.lines})},
        "timing_ms": {"ocr": round((t1 - t0) * 1000), "matching": round((t2 - t1) * 1000), **ocr.timings},
        "backend": ocr.backend,
    }
    if debug:
        out["ocr_lines"] = [{"text": l.text, "confidence": round(l.conf, 2), "script": l.script} for l in ocr.lines]
    return out


@app.get("/health", tags=["Divers"])
def health():
    return {"status": "ok", "reference_rows": int(len(matcher.df)), "brand_names": len(matcher.by_base),
            "ocr_loaded": _ocr is not None}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")
