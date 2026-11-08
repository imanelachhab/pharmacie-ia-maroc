"""API FastAPI du chatbot pharmacie (Challenge 1).

Lancer :  uvicorn app.main:app --reload      puis ouvrir http://127.0.0.1:8000
Docs   :  http://127.0.0.1:8000/docs
"""
import os
import uuid
from functools import lru_cache
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .kb import KB
from .nlp import NLU
from .responder import Responder

KB_DIR = Path(os.environ.get("KB_DIR", r"C:\Users\DELL\dataset_challenge\kb"))
STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Assistant Pharmacie – Darija / Arabe / Français", version="1.0.0",
              description="Comprend des demandes écrites ou vocales sur les médicaments et les pharmacies "
                          "et renvoie une interprétation structurée (intent + entités) et une réponse.")

kb = KB(KB_DIR)
nlu = NLU(kb.med_names, kb.aliases, kb.cities)
responder = Responder(kb)

# memoire de conversation (en RAM, 500 sessions max)
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()


@lru_cache(maxsize=4096)
def analyze_cached(text: str):
    return nlu.analyze(text)


def get_ctx(session_id: str) -> dict:
    ctx = SESSIONS.get(session_id)
    if ctx is None:
        ctx = SESSIONS[session_id] = {}
        while len(SESSIONS) > 500:
            SESSIONS.popitem(last=False)
    SESSIONS.move_to_end(session_id)
    return ctx


# --------------------------------------------------------------------------- modeles
class ChatIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["ch7al taman dyal doliprane 1000mg ?"])
    session_id: Optional[str] = Field(None, description="identifiant de conversation (pour le contexte)")


def run_chat(text: str, session_id: Optional[str], extra: Optional[dict] = None) -> dict:
    sid = session_id or uuid.uuid4().hex[:12]
    ctx = get_ctx(sid)
    a = analyze_cached(text)
    r = responder.respond(a, ctx)
    out = {
        "session_id": sid,
        "input_text": text,
        "normalized_text": a.normalized_text,
        "language": a.language,
        "intent": a.intent,
        "intents": a.intents,
        "entities": [e.public() for e in a.entities if e.text != "" or e.label == "CITY"],
        "confidence": a.confidence,
        "response": r["text"],
        "blocks": r["blocks"],
        "suggestions": r["suggestions"],
        "reply_language": r["reply_language"],
        "data": r["data"],
        "needs_clarification": r["needs_clarification"],
        "context_used": r["context_used"],
    }
    if extra:
        out.update(extra)
    return out


# --------------------------------------------------------------------------- conversation
@app.post("/chat", tags=["Conversation"], summary="Traite une requête texte")
def chat(body: ChatIn):
    return run_chat(body.text, body.session_id)


@app.post("/analyze", tags=["Conversation"], summary="Interprétation seule (langue, intent, entités), sans réponse")
def analyze(body: ChatIn):
    a = nlu.analyze(body.text)
    return {"normalized_text": a.normalized_text, "language": a.language, "intent": a.intent, "intents": a.intents,
            "entities": [e.public() for e in a.entities if e.text != "" or e.label == "CITY"],
            "confidence": a.confidence}


@app.post("/speech", tags=["Conversation"], summary="Traite un message vocal (STT puis NLP)")
async def speech(audio: UploadFile = File(...), session_id: Optional[str] = Form(None),
                 language: Optional[str] = Form(None, description="ar, fr ... (laisser vide = détection auto)")):
    from . import stt
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Fichier audio vide")
    try:
        suffix = Path(audio.filename or "a.webm").suffix or ".webm"
        text, lang = stt.transcribe(data, suffix, language)
    except stt.STTUnavailable as e:
        raise HTTPException(501, f"Speech-to-Text indisponible : {e}")
    if not text:
        raise HTTPException(422, "Aucune parole détectée dans l'audio")
    return run_chat(text, session_id, {"input_type": "audio", "transcription": text, "stt_language": lang})


# --------------------------------------------------------------------------- medicaments
def _summary(r: dict) -> dict:
    keys = ("id", "nom", "dosage", "forme", "presentation", "fabricant", "statut", "prix", "source")
    return {k: r.get(k) for k in keys}


@app.get("/medications/search", tags=["Médicaments"], summary="Recherche d'un médicament (tolère les fautes)")
def medications_search(q: str = Query(..., min_length=2, examples=["dolipran"]), limit: int = Query(10, ge=1, le=50)):
    res = kb.search(q, limit)
    return {"query": q, "count": len(res), "results": [_summary(r) for r in res]}


def _med_or_404(med_id: int) -> dict:
    r = kb.get(med_id)
    if r is None:
        raise HTTPException(404, f"Médicament {med_id} introuvable")
    return r


@app.get("/medications/{med_id}", tags=["Médicaments"], summary="Informations détaillées")
def medication_detail(med_id: int):
    return _med_or_404(med_id)


@app.get("/medications/{med_id}/price", tags=["Médicaments"], summary="Prix (PPV)")
def medication_price(med_id: int):
    r = _med_or_404(med_id)
    return {"id": med_id, "nom": r["nom"], "presentation": r.get("presentation"), "prix": r.get("prix"),
            "devise": "MAD", "source_prix": r.get("prix_source"), "ph": r.get("ph"),
            "conflit_entre_sources": bool(r.get("conflit_prix"))}


@app.get("/medications/{med_id}/reimbursement", tags=["Médicaments"], summary="Remboursement")
def medication_reimbursement(med_id: int):
    r = _med_or_404(med_id)
    return {"id": med_id, "nom": r["nom"], "remboursement": r.get("remboursement"),
            "information_disponible": r.get("remboursement") is not None, "source": r.get("source")}


# --------------------------------------------------------------------------- pharmacies
@app.get("/pharmacies/search", tags=["Pharmacies"], summary="Recherche de pharmacies par ville / quartier / nom")
def pharmacies_search(city: Optional[str] = Query(None, examples=["Rabat"]), q: Optional[str] = None,
                      limit: int = Query(10, ge=1, le=50)):
    canon = None
    if city:
        canon = kb.canon_city(city)
        if canon is None:
            raise HTTPException(404, f"Ville inconnue : {city}. Villes disponibles : {', '.join(kb.cities)}")
    items, total, _ = kb.pharmacies(city=canon, q=q, limit=limit)
    return {"city": canon, "total": total, "results": items}


@app.get("/pharmacies/duty", tags=["Pharmacies"], summary="Pharmacies de garde (si l'info existe dans les données)")
def pharmacies_duty(city: str = Query(..., examples=["Casablanca"])):
    canon = kb.canon_city(city)
    if canon is None:
        raise HTTPException(404, f"Ville inconnue : {city}")
    items, total, _ = kb.pharmacies(city=canon, duty=True, limit=20)
    return {"city": canon, "total": total, "results": items,
            "note": None if items else "Aucune information de garde dans les données pour cette ville."}


# --------------------------------------------------------------------------- divers
@app.get("/health", tags=["Divers"])
def health():
    return {"status": "ok", "medications": len(kb.meds), "pharmacies": len(kb.pharm),
            "known_medication_names": len(nlu.med_vocab)}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")
