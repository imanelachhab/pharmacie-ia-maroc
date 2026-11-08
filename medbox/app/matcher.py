"""Mise en correspondance du texte lu sur la boite avec la base de medicaments (kb/medications.csv).

Principe : le NOM est l'indice principal (recherche floue tolerante aux erreurs d'OCR), puis le DOSAGE, la TAILLE
DE BOITE, le LABORATOIRE et la FORME departagent les presentations d'un meme medicament.
Le resultat contient toujours les indices qui ont servi (`evidence`) et un statut honnete :
recognized / uncertain / not_found / no_text.
"""
import re
import unicodedata
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

AR_DIAC = re.compile(r"[\u064B-\u0652\u0640]")


def normalize(text: str) -> str:
    t = AR_DIAC.sub("", str(text))
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي")
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = t.lower().replace("µ", "u")
    t = re.sub(r"[^\w\s%/.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# --- extraction de dosage / taille de boite dans un texte OCR (tolere O/0 et l/1) ---
DOSE_RE = re.compile(r"(?<![a-z0-9])([0-9oOlI]{1,5}(?:[.,][0-9oO]{1,3})?)\s?(mg|g|ml|ui|iu|ug|mcg|%)(?![a-z])", re.I)
PACK_RE = re.compile(r"(?:boite|bte|box|b/|bt)\s*(?:de|d)?\s*([0-9oOlI]{1,3})\b|\b([0-9oOlI]{1,3})\s*(?:comprimes?|cp|gelules?|sachets?|"
                     r"ampoules?|capsules?|suppositoires?|unites?|flacons?|tablets?|comp)\b", re.I)
FORMS = {"comprime": "compr", "gelule": "gelul", "sirop": "sirop", "solution": "solution", "suspension": "suspension",
         "sachet": "sachet", "injectable": "inject", "suppositoire": "suppo", "pommade": "pommade", "creme": "creme",
         "collyre": "collyre", "spray": "spray", "capsule": "capsul"}
GENERIC = {"pharma", "maroc", "boite", "comprime", "comprimes", "sachet", "sachets", "gelule", "gelules", "poudre",
           "solution", "suspension", "buvable", "injectable", "dispersible", "pellicule", "pelliculé", "pharmaceutical",
           "laboratoires", "laboratoire", "medicament", "usage", "adulte", "enfant", "enfants", "nourrissons",
           "voie", "orale", "verre", "flacon", "ampoule", "capsule", "tablets", "tablet", "sanofi", "aventis"}
AR_FORMS = {"اقراص": "compr", "قرص": "compr", "كبسولات": "gelul", "شراب": "sirop", "حقن": "inject", "مرهم": "pommade"}


def _digits(s: str) -> str:
    return s.translate(str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1"}))


def dose_key(value, unit):
    try:
        v = float(_digits(value).replace(",", "."))
    except ValueError:
        return None
    u = unit.lower()
    if u == "g":
        v, u = v * 1000, "mg"
    elif u in ("ug", "mcg"):
        v, u = v / 1000, "mg"
    elif u == "iu":
        u = "ui"
    return (round(v, 4), u)


def dose_key_from_kb(s) -> Optional[tuple]:
    if not isinstance(s, str):
        return None
    m = DOSE_RE.search(s.replace("µ", "u"))
    return dose_key(m.group(1), m.group(2)) if m else None


def pack_from_kb(s) -> Optional[int]:
    if not isinstance(s, str):
        return None
    m = re.search(r"\b(?:de|d['’]?)\s*(\d{1,3})\b", s, re.I)
    return int(m.group(1)) if m else None


class Matcher:
    def __init__(self, kb_dir):
        d = pd.read_csv(f"{kb_dir}/medications.csv")
        d = d[d["nature"].fillna("Médicament").isin(["Médicament", "Médicament homéopathique"])].copy()
        d = d.dropna(subset=["nom_base"])
        d["_dose"] = d["dosage"].map(dose_key_from_kb)
        d["_pack"] = d["presentation"].map(pack_from_kb)
        d["_lab"] = d["fabricant"].fillna("").map(normalize)
        d["_forme"] = d["forme"].fillna("").map(normalize)
        self.df = d.reset_index(drop=True)
        self.by_base = d.groupby("nom_base").indices
        keys = list(self.by_base)
        self.keys = [k for k in keys if len(k) >= 5]
        self.short = {k for k in keys if 3 <= len(k) < 5}
        self.arabic = {}
        try:  # noms arabes de medicaments courants (partages avec le chatbot si disponible)
            from .arabic_names import ARABIC_NAMES
            for b, lst in ARABIC_NAMES.items():
                if b in self.by_base:
                    for a in lst:
                        self.arabic[normalize(a)] = b
        except ImportError:
            pass

    # ------------------------------------------------------------------
    def _name_candidates(self, lines):
        text = normalize(" ".join(l.text for l in lines if l.script == "latin"))
        comp = text.replace(" ", "")
        cands = {}
        if comp:
            for k, sc, _ in process.extract(comp, self.keys, scorer=fuzz.partial_ratio, score_cutoff=82, limit=25):
                cands[k] = sc / 100
            tokens = set(text.split())
            for k in self.short:
                if k in tokens:
                    cands[k] = 1.0
            for k in list(cands):
                if k in tokens:
                    cands[k] = min(1.0, cands[k] + 0.03)
        # noms ecrits en arabe
        ar = normalize(" ".join(l.text for l in lines if l.script == "arabic")).replace(" ", "")
        for a, b in self.arabic.items():
            if a and a in ar:
                cands[b] = max(cands.get(b, 0), 0.95)
        # un nom plus long qui contient le nom trouve est prefere ("clamoxyl" gagne contre "amoxyl")
        for k in list(cands):
            if any(k != k2 and k in k2 and cands[k2] >= cands[k] - 0.05 for k2 in cands):
                del cands[k]
        # nom coupe (boite partiellement visible) : fragment d'au moins 5 lettres contenu dans un nom connu
        via_fragment = set()
        if not cands or max(cands.values()) < 0.9:
            for tok in set(text.split()):
                if len(tok) >= 5 and tok.isalpha() and tok not in GENERIC:
                    for k in self.keys:
                        if tok in k and len(tok) / len(k) >= 0.6 and k not in cands:
                            cands[k] = 0.5 + 0.35 * len(tok) / len(k)
                            via_fragment.add(k)
        self._fragment = via_fragment
        out = {}
        for k, s in cands.items():
            line_conf = [l.conf for l in lines if l.script == "latin" and k in normalize(l.text).replace(" ", "")]
            conf = max(line_conf) if line_conf else 0.6
            out[k] = (s, conf)
        return out, text

    @staticmethod
    def _doses_packs(text, lines):
        raw = " ".join(l.text for l in lines if l.script == "latin").replace("µ", "u")
        doses, packs = [], []
        for m in DOSE_RE.finditer(raw):
            k = dose_key(m.group(1), m.group(2))
            if k and k not in doses:
                doses.append(k)
        for m in PACK_RE.finditer(raw):
            v = m.group(1) or m.group(2)
            try:
                n = int(_digits(v))
            except ValueError:
                continue
            if 1 <= n <= 500 and n not in packs:
                packs.append(n)
        return doses, packs

    def match(self, lines, top_k: int = 5) -> dict:
        if not [l for l in lines if l.text.strip()]:
            return {"status": "no_text", "candidates": [], "ocr_dosages": [], "ocr_packs": []}
        names, text = self._name_candidates(lines)
        doses, packs = self._doses_packs(text, lines)
        ar_text = normalize(" ".join(l.text for l in lines if l.script == "arabic")).replace(" ", "")
        forms_seen = {v for k, v in FORMS.items() if k in text.replace(" ", "")}
        forms_seen |= {v for k, v in AR_FORMS.items() if normalize(k).replace(" ", "") in ar_text}
        comp = text.replace(" ", "")
        if not names:
            return {"status": "not_found", "candidates": [], "ocr_dosages": [dict(zip(("value", "unit"), d)) for d in doses],
                    "ocr_packs": packs}

        rows = []
        for base, (nscore, conf) in sorted(names.items(), key=lambda x: (-x[1][0], -len(x[0])))[:8]:
            name_part = nscore * (0.75 + 0.25 * min(1.0, conf + 0.2))
            for i in self.by_base[base]:
                r = self.df.iloc[i]
                ev = [f"nom : {base} ({round(nscore * 100)} %)"]
                bonus = 0.0
                if doses and r["_dose"] is not None and not (isinstance(r["_dose"], float)):
                    if r["_dose"] in doses:
                        bonus += 0.20; ev.append(f"dosage : {r['dosage']} ✓")
                    else:
                        bonus -= 0.05
                if packs and pd.notna(r["_pack"]):
                    if int(r["_pack"]) in packs:
                        bonus += 0.08; ev.append(f"boîte de {int(r['_pack'])} ✓")
                lab = r["_lab"].split(" ")[0] if r["_lab"] else ""
                if len(lab) >= 4 and fuzz.partial_ratio(lab, comp) >= 88:
                    bonus += 0.07; ev.append(f"laboratoire : {r['fabricant']} ✓")
                if forms_seen and any(f in r["_forme"] for f in forms_seen):
                    bonus += 0.04; ev.append(f"forme : {r['forme']} ✓")
                score = max(0.0, min(1.0, 0.72 * name_part / 0.97 + bonus))
                if base in self._fragment:           # nom seulement partiellement lu : jamais "reconnu"
                    score = min(score, 0.78)
                    ev[0] += " – fragment"
                rows.append((score, base, i, ev))
        rows.sort(key=lambda x: -x[0])
        seen, cands = set(), []
        for score, base, i, ev in rows:
            r = self.df.iloc[i]
            key = (base, r["dosage"], r["forme"], r["presentation"])
            if key in seen:
                continue
            seen.add(key)
            cands.append({
                "id": int(r["id"]), "nom": r["nom"], "nom_base": base, "dosage": _n(r["dosage"]), "forme": _n(r["forme"]),
                "presentation": _n(r["presentation"]), "fabricant": _n(r["fabricant"]), "prix": _n(r["prix"]),
                "remboursement": _n(r["remboursement"]), "score": round(float(score), 3), "evidence": ev})
            if len(cands) >= top_k:
                break
        best = cands[0]
        other = next((c for c in cands[1:] if c["nom_base"] != best["nom_base"]), None)
        margin = best["score"] - (other["score"] if other else 0.0)
        if best["score"] >= 0.80 and margin >= 0.06:
            status = "recognized"
        elif best["score"] >= 0.55:
            status = "uncertain"
        else:
            status = "not_found"
        return {"status": status, "candidates": cands, "margin": round(margin, 3),
                "ocr_dosages": [{"value": d[0], "unit": d[1]} for d in doses], "ocr_packs": packs}


def _n(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v
