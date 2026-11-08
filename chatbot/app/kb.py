"""Acces a la base de connaissances (fichiers produits par build_kb.py)."""
import json
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

from .nlp import normalize

DOSE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s?(mg|g|µg|ug|mcg|ml|ui|iu|%)", re.I)


def dose_to_key(d) -> Optional[tuple]:
    """'1 G' / '1000mg' -> (1000.0, 'mg') pour comparer des dosages equivalents"""
    if not isinstance(d, str):
        return None
    m = DOSE_RE.search(d.replace("µ", "u").replace("μ", "u"))
    if not m:
        return None
    v, u = float(m.group(1).replace(",", ".")), m.group(2).lower()
    if u == "g":
        v, u = v * 1000, "mg"
    elif u in ("ug", "mcg"):
        v, u = v / 1000, "mg"
    elif u == "iu":
        u = "ui"
    return (round(v, 4), u)


def _clean(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def rec(row: dict) -> dict:
    return {k: _clean(v) for k, v in row.items()}


class KB:
    def __init__(self, kb_dir):
        kb = Path(kb_dir)
        self.meds = pd.read_csv(kb / "medications.csv")
        self.pharm = pd.read_csv(kb / "pharmacies.csv", dtype={"telephone": "string"})
        # protection si l'ancienne version de build_kb.py a ete utilisee (texte d'interface scrape)
        for c in ("composition", "code_atc"):
            self.meds[c] = self.meds[c].astype("string").str.replace(
                r"\s*Nature\s*:.*$|\s*Ajouter à mon simulateur", "", regex=True, case=False).str.strip()
            self.meds[c] = self.meds[c].replace("", pd.NA)
        # lisibilite : "A , B" -> "A, B" ; "A | B" -> "A + B"
        for c in ("composition", "classe_therapeutique"):
            self.meds[c] = (self.meds[c].astype("string").str.replace(r"\s+,", ",", regex=True)
                            .str.replace(r"\s*\|\s*", " + ", regex=True))
        self.meds["_dose"] = self.meds["dosage"].map(dose_to_key)
        self.pharm["_adr"] = self.pharm["adresse"].fillna("").map(normalize)
        self.pharm["_garde"] = self.pharm["garde_info"].fillna("").str.contains("garde", case=False)
        self.by_id = {int(r["id"]): i for i, r in self.meds[["id"]].iterrows()}
        self.by_base = self.meds.groupby("nom_base").indices
        self.aliases = {}
        f = kb / "city_aliases.json"
        if f.exists():
            self.aliases = json.loads(f.read_text(encoding="utf-8"))
        self.cities = sorted(self.pharm["ville_canonique"].dropna().unique())

    # ------------------------------------------------------------------ medicaments
    @property
    def med_names(self):
        return self.meds["nom_base"].dropna().unique()

    def get(self, med_id: int) -> Optional[dict]:
        i = self.by_id.get(int(med_id))
        return None if i is None else self._row(i)

    def _row(self, i):
        r = rec(self.meds.iloc[i].drop(labels=["_dose"]).to_dict())
        return r

    def rows_for(self, base: str, dosage: Optional[str] = None):
        """toutes les presentations d'un medicament ; filtre par dosage si possible.
        retourne (lignes, dosage_filtre_applique)"""
        idx = self.by_base.get(base)
        if idx is None:
            return [], False
        sub = self.meds.iloc[idx]
        applied = False
        if dosage:
            key = dose_to_key(dosage)
            f = sub[sub["_dose"] == key]
            if key and len(f):
                sub, applied = f, True
        sub = sub.sort_values(["prix"], na_position="last")
        return [rec(r) for r in sub.drop(columns=["_dose"]).to_dict("records")], applied

    def search(self, q: str, limit: int = 10):
        qn = normalize(q)
        if not qn:
            return []
        names = self.meds["nom_normalise"]
        hit = self.meds[names.str.contains(re.escape(qn), na=False)]
        if hit.empty:
            keys = list(self.by_base.keys())
            res = process.extract(qn, keys, scorer=fuzz.WRatio, score_cutoff=75, limit=5)
            idx = [i for k, _, _ in res for i in self.by_base[k]]
            hit = self.meds.iloc[idx]
        hit = hit.assign(_s=hit["nom_normalise"].map(lambda x: fuzz.ratio(qn, x)))
        hit = hit.sort_values("_s", ascending=False).head(limit)
        return [rec(r) for r in hit.drop(columns=["_dose", "_s"]).to_dict("records")]

    # ------------------------------------------------------------------ pharmacies
    def canon_city(self, name: str) -> Optional[str]:
        n = normalize(name)
        if n in self.aliases:
            return self.aliases[n]
        for c in self.cities:
            if normalize(c) == n:
                return c
        return None

    def pharmacies(self, city: Optional[str] = None, quartier: Optional[str] = None, q: Optional[str] = None,
                   duty: bool = False, limit: int = 5):
        d = self.pharm
        if city:
            d = d[d["ville_canonique"] == city]
        note = None
        if quartier:
            qn = normalize(quartier)
            dq = d[d["_adr"].str.contains(qn, na=False) | d["nom_normalise"].str.contains(qn, na=False)]
            if len(dq):
                d = dq
            else:
                note = "quartier_not_found"
        if q:
            qn = normalize(q)
            d = d[d["nom_normalise"].str.contains(qn, na=False) | d["_adr"].str.contains(qn, na=False)]
        if duty:
            d = d[d["_garde"]]
        d = d.sort_values(["_garde"], ascending=False)
        out = [rec(r) for r in d.head(limit).drop(columns=["_adr", "_garde"]).to_dict("records")]
        return out, int(len(d)), note
