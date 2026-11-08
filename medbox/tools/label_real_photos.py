#!/usr/bin/env python3
"""
label_real_photos.py - Cree labels.csv pour tes VRAIES photos de boites, a partir du nom des fichiers.

Tu photographies les boites et tu nommes chaque photo avec le medicament, par exemple :
    doliprane 1000mg boite 8 - 1.jpg
    doliprane 1000mg boite 8 - flou.jpg          (le mot apres le tiret = condition de la photo)
    augmentin 1g 12 sachets - reflet.jpg
    id00123 - sombre.jpg                           (si tu connais l'id exact dans medications.csv)

Le script retrouve la ligne correspondante dans ta base et ecrit labels.csv. Les cas ambigus sont marques
to_check=True : OUVRE labels.csv et corrige la colonne id a la main avant d'evaluer.

Usage :
  python tools/label_real_photos.py --photos data/real --kb C:\\Users\\DELL\\dataset_challenge\\kb
  python evaluate_recognition.py --data data/real
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.matcher import Matcher, normalize  # noqa: E402
from app.ocr import Line  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONDITIONS = {"flou": "blur", "blur": "blur", "sombre": "dark", "dark": "dark", "reflet": "glare", "glare": "glare",
              "incline": "rotate", "rotate": "rotate", "coupe": "crop", "crop": "crop", "bruit": "noise"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="data/real")
    ap.add_argument("--kb", default=r"C:\Users\DELL\dataset_challenge\kb")
    ap.add_argument("--out", default=None, help="defaut : <photos>/labels.csv")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    folder = Path(a.photos)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXT)
    if not files:
        sys.exit(f"Aucune image dans {folder}")
    m = Matcher(a.kb)
    by_id = m.df.set_index("id")
    rows, to_check = [], 0
    for p in files:
        stem = p.stem
        name_part, _, cond_part = stem.partition(" - ")
        cond = next((v for k, v in CONDITIONS.items() if k in normalize(cond_part)), "real")
        mid = re.match(r"id(\d+)", name_part.strip(), re.I)
        status, alts = "by_id", ""
        if mid and int(mid.group(1)) in by_id.index:
            row = by_id.loc[int(mid.group(1))]; rid = int(mid.group(1))
        else:
            res = m.match([Line(name_part, 1.0, "latin")], top_k=3)
            if not res["candidates"]:
                rows.append({"file": p.name, "id": "", "nom_base": "", "nom": "", "dosage": "", "presentation": "",
                             "degradation": cond, "to_check": True, "alternatives": "AUCUN MEDICAMENT TROUVE : renseigne l'id"})
                to_check += 1
                continue
            c = res["candidates"][0]
            rid, status = c["id"], res["status"]
            row = by_id.loc[rid]
            alts = " | ".join(f"{x['id']}:{x['nom']} {x['presentation'] or ''}" for x in res["candidates"][1:])
        check = status not in ("by_id", "recognized")
        # le meme nom de produit peut avoir plusieurs presentations : on le signale
        n_same = int((m.df["nom_base"] == row["nom_base"]).sum())
        check = check or (status != "by_id" and n_same > 1 and not re.search(r"\d", name_part))
        to_check += bool(check)
        rows.append({"file": p.name, "id": rid, "nom_base": row["nom_base"], "nom": row["nom"], "dosage": row["dosage"],
                     "presentation": row["presentation"], "degradation": cond, "to_check": bool(check), "alternatives": alts})
    out = Path(a.out) if a.out else folder / "labels.csv"
    df = pd.DataFrame(rows)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(rows)} photos -> {out}\n{to_check} a verifier (colonne to_check = True) : corrige la colonne id si besoin.")


if __name__ == "__main__":
    main()
