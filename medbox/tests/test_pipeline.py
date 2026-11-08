"""Tests du pipeline.  Lancer :  set KB_DIR=C:\\Users\\DELL\\dataset_challenge\\kb  &&  python -m pytest -q tests
(necessite un moteur OCR : Tesseract ou EasyOCR)"""
import os
import random
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
KB = os.environ.get("KB_DIR", r"C:\Users\DELL\dataset_challenge\kb")

from app.matcher import Matcher, dose_key  # noqa: E402
from app.ocr import Line, OCR  # noqa: E402


def test_dose_parsing_tolerates_ocr_confusions():
    assert dose_key("5OO", "mg") == (500.0, "mg")
    assert dose_key("1", "g") == (1000.0, "mg")


def test_matcher_from_text_only():
    m = Matcher(KB)
    lines = [Line("DOLIPRANE", 0.9, "latin"), Line("1000 mg", 0.9, "latin"), Line("Boite de 8", 0.9, "latin")]
    r = m.match(lines)
    assert r["status"] in ("recognized", "uncertain")
    assert r["candidates"][0]["nom_base"] == "doliprane"
    assert r["candidates"][0]["dosage"] == "1 G"                  # 1000 mg == 1 g
    assert any("boîte de 8" in e for e in r["candidates"][0]["evidence"])


def test_ocr_typo_and_longer_name_wins():
    m = Matcher(KB)
    r = m.match([Line("CLAMOXYL", 0.8, "latin"), Line("500 mg", 0.8, "latin")])
    assert r["candidates"][0]["nom_base"] == "clamoxyl"          # et non "amoxyl"


def test_empty_and_garbage_text():
    m = Matcher(KB)
    assert m.match([])["status"] == "no_text"
    assert m.match([Line("xqzv wkj", 0.4, "latin")])["status"] == "not_found"


def _ocr():
    try:
        return OCR("auto")
    except Exception as e:  # pas de moteur OCR installe
        pytest.skip(f"moteur OCR indisponible : {e}")


def test_end_to_end_on_synthetic_box():
    import make_synthetic_boxes as g
    rng = random.Random(1)
    prods = g.pick_products(KB, 1, rng)
    r = prods.iloc[0]
    face = g.draw_face(r, "A", rng)
    img = cv2.cvtColor(__import__("numpy").array(g.scene(face, rng)), cv2.COLOR_RGB2BGR)
    res = Matcher(KB).match(_ocr().read(img, fast=True).lines)
    assert res["candidates"] and res["candidates"][0]["nom_base"] == r["nom_base"]


def test_api_recognize():
    from fastapi.testclient import TestClient
    import make_synthetic_boxes as g
    _ocr()
    from app.main import app
    rng = random.Random(2)
    r = g.pick_products(KB, 1, rng).iloc[0]
    img = g.scene(g.draw_face(r, "C", rng), rng)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    c = TestClient(app)
    j = c.post("/recognize?fast=true", files={"image": ("box.jpg", buf.tobytes(), "image/jpeg")}).json()
    assert j["status"] in ("recognized", "uncertain") and j["best"]["nom_base"] == r["nom_base"]
    assert c.post("/recognize", files={"image": ("x.txt", b"not an image", "text/plain")}).status_code == 400
    assert c.get("/health").json()["status"] == "ok"
