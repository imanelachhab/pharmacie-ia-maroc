#!/usr/bin/env python3
"""
evaluate_recognition.py - Evalue le pipeline (OCR + matching) sur un dossier d'images avec labels.csv.

  python evaluate_recognition.py --data data\\synthetic --kb C:\\Users\\DELL\\dataset_challenge\\kb --out reports

labels.csv attendu : file, id, nom_base, dosage, degradation (id = -1 pour un medicament absent de la base)
Metriques : top-1 / top-3 (nom du medicament), top-1 avec dosage, statuts, fausses reconnaissances, temps.
Ecrit : recognition_report.json, recognition_errors.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from app.matcher import Matcher, dose_key_from_kb  # noqa: E402
from app.ocr import OCR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--kb", default=r"C:\Users\DELL\dataset_challenge\kb")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--fast", action="store_true", help="une seule passe OCR rapide")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(data / "labels.csv")
    if lab["id"].isna().any():
        print(f"ATTENTION : {int(lab['id'].isna().sum())} photo(s) sans id ignorees (renseigne la colonne id de labels.csv)")
        lab = lab.dropna(subset=["id"])
    if a.limit:
        lab = lab.head(a.limit)
    ocr, matcher = OCR(a.backend), Matcher(a.kb)

    rows = []
    for i, r in enumerate(lab.itertuples(), 1):
        img = cv2.imread(str(data / r.file))
        t = time.perf_counter()
        o = ocr.read(img, fast=a.fast)
        m = matcher.match(o.lines)
        ms = (time.perf_counter() - t) * 1000
        cands = m["candidates"]
        known = int(r.id) >= 0
        bases = [c["nom_base"] for c in cands]
        gold_dose = dose_key_from_kb(r.dosage)
        top1 = bool(cands) and cands[0]["nom_base"] == r.nom_base
        top1_dose = top1 and dose_key_from_kb(cands[0]["dosage"]) == gold_dose
        rows.append({"file": r.file, "degradation": r.degradation, "known": known, "gold": r.nom_base, "status": m["status"],
                     "pred": cands[0]["nom"] if cands else "", "score": cands[0]["score"] if cands else 0,
                     "top1": top1, "top3": r.nom_base in bases[:3], "top1_dose": top1_dose, "ms": round(ms),
                     "ocr_text": " | ".join(l.text for l in o.lines[:8])})
        if i % 10 == 0:
            print(f"{i}/{len(lab)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "recognition_errors.csv", index=False, encoding="utf-8-sig")
    k = d[d.known]
    neg = d[~d.known]
    rec = k[k.status == "recognized"]
    rep = {
        "n_images": len(d), "n_known": len(k), "n_unknown": len(neg), "backend": o.backend, "fast": a.fast,
        "top1": round(k.top1.mean(), 3), "top3": round(k.top3.mean(), 3), "top1_with_dosage": round(k.top1_dose.mean(), 3),
        "status_known": k.status.value_counts().to_dict(),
        "precision_when_recognized": round(rec.top1.mean(), 3) if len(rec) else None,
        "coverage_recognized": round(len(rec) / max(1, len(k)), 3),
        "top1_by_degradation": k.groupby("degradation").top1.mean().round(3).to_dict(),
        "top3_by_degradation": k.groupby("degradation").top3.mean().round(3).to_dict(),
        "unknown_products": ({"false_recognized": int((neg.status == "recognized").sum()),
                              "status": neg.status.value_counts().to_dict()} if len(neg) else None),
        "avg_ms": int(d.ms.mean()), "p95_ms": int(d.ms.quantile(.95)),
    }
    (out / "recognition_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
