"""Tests de bout en bout de l'API.  Lancer :  set KB_DIR=C:\\Users\\DELL\\dataset_challenge\\kb  &&  pytest -q"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

c = TestClient(app)


def chat(text, sid=None):
    r = c.post("/chat", json={"text": text, "session_id": sid})
    assert r.status_code == 200
    return r.json()


def ents(r):
    return {(e["label"], e["value"]) for e in r["entities"]}


def test_health():
    r = c.get("/health").json()
    assert r["medications"] > 1000 and r["pharmacies"] > 100


def test_price_darija_latin():
    r = chat("ch7al taman dyal doliprane 1000mg ?")
    assert r["intents"] == ["MEDICATION_PRICE"]
    assert ("MEDICATION", "Doliprane") in ents(r) and ("DOSAGE", "1000MG") in ents(r)
    assert "dh" in r["response"]


def test_reimbursement_french():
    r = chat("Est-ce que l'augmentin est remboursé ?")
    assert r["intents"] == ["MEDICATION_REIMBURSEMENT"] and r["language"] == "fr"


def test_availability_arabic_script():
    r = chat("واش كاين سبازفون عندكم ؟")
    assert r["intents"] == ["MEDICATION_AVAILABILITY"] and ("MEDICATION", "Spasfon") in ents(r)


def test_typo_is_confirmed_to_user():
    r = chat("dolipran composition")
    assert r["intents"] == ["MEDICATION_COMPOSITION"] and "dolipran" in r["response"]


def test_multi_intent():
    r = chat("ch7al taman amoxil w wach kayt3awd ?")
    assert set(r["intents"]) == {"MEDICATION_PRICE", "MEDICATION_REIMBURSEMENT"}


def test_pharmacy_and_duty():
    assert chat("fin nl9a pharmacie f Marrakech ?")["intents"] == ["PHARMACY_SEARCH"]
    d = chat("pharmacie de garde Casa")
    assert d["intents"] == ["PHARMACY_DUTY"] and ("CITY", "Casablanca") in ents(d)


def test_context_follow_up():
    sid = chat("ch7al taman amoxil 500mg")["session_id"]
    r = chat("w wach kayt3awd ?", sid)
    assert r["intents"] == ["MEDICATION_REIMBURSEMENT"] and r["context_used"]


def test_clarification_when_medication_missing():
    r = chat("ch7al taman ?")
    assert r["needs_clarification"]


def test_out_of_scope_and_greeting():
    assert chat("ch7al sa3a daba")["intents"] == ["OUT_OF_SCOPE"]
    assert chat("salam")["intents"] == ["GREETING"]


def test_structured_endpoints():
    s = c.get("/medications/search", params={"q": "dolipran"}).json()
    assert s["count"] > 0
    mid = s["results"][0]["id"]
    assert c.get(f"/medications/{mid}").status_code == 200
    assert "prix" in c.get(f"/medications/{mid}/price").json()
    assert "remboursement" in c.get(f"/medications/{mid}/reimbursement").json()
    assert c.get("/medications/99999999").status_code == 404
    assert c.get("/pharmacies/search", params={"city": "Rabat"}).json()["total"] > 0
    assert c.get("/pharmacies/search", params={"city": "Atlantide"}).status_code == 404
    assert "results" in c.get("/pharmacies/duty", params={"city": "Casablanca"}).json()


def test_speech_without_whisper_is_explicit():
    r = c.post("/speech", files={"audio": ("a.webm", b"123", "audio/webm")})
    assert r.status_code in (501, 422, 200)


def test_blocks_and_suggestions_for_ui():
    r = chat("ch7al taman doliprane")
    assert r["blocks"][0]["type"] == "prices" and r["reply_language"] == "dl"
    assert len(r["suggestions"]) >= 2
    p = chat("fin kayna pharmacie de garde f Casa ?")
    assert p["blocks"][0]["type"] == "pharmacies" and p["blocks"][0]["items"][0]["telephone"].startswith("0")


def test_unknown_medication_is_not_confused_with_previous_one():
    sid = chat("ch7al taman amoxil")["session_id"]
    r = chat("ch7al taman xyzabcdef ?", sid)
    assert "xyzabcdef" in r["response"] and r["blocks"][0]["type"] == "text"
