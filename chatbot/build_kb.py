#!/usr/bin/env python3
"""
build_kb.py - Nettoyage et fusion des datasets (Challenge 1)

Entrees  : saydalia_medicaments.xlsx, medicament_ma_listing.csv, saydalia_pharmacies.xlsx
Sorties  : kb/medications.csv, kb/pharmacies.csv, kb/non_pharmacies.csv,
           kb/city_aliases.json, kb/kb.sqlite, kb/quality_report.json

Usage : python build_kb.py
        (par defaut : lit C:\\Users\\DELL\\dataset_challenge et ecrit dans ...\\dataset_challenge\\kb)
"""
import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Chemins par defaut (modifiables via --in / --out)
# --------------------------------------------------------------------------
DEFAULT_IN = Path(r"C:\Users\DELL\dataset_challenge")
DEFAULT_OUT = DEFAULT_IN / "kb"

FILE_SAYDALIA_MEDS = "saydalia_medicaments.xlsx"
FILE_MEDICAMENT_MA = "medicament_ma_listing.csv"
FILE_SAYDALIA_PHARMA = "saydalia_pharmacies.xlsx"


# --------------------------------------------------------------------------
# Utilitaires de normalisation
# --------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s) -> str:
    """minuscules, sans accents, sans ponctuation, espaces simplifies"""
    if pd.isna(s):
        return ""
    s = strip_accents(str(s)).lower().replace("µ", "u")
    s = re.sub(r"[^a-z0-9\u0600-\u06FF/%.\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


DOSAGE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?\s?(?:mg|g|µg|ug|mcg|ui|iu|ml|%|mg/ml|mg/g|meq)(?:\s?/\s?\d+(?:[.,]\d+)?\s?(?:mg|g|ml|h|l))?)",
    re.I,
)
PACK_RE = re.compile(r"\b(?:de|d'|d’)\s*(\d+)\b", re.I)
PRESENTATION_RE = re.compile(
    r"\s(Bo[iî]te|Flacon|Tube|Sachet|Ampoule|Seringue|Stylo|Pot|Plaquette|Blister|Coffret|Poche)\b", re.I
)
NATURE_RE = re.compile(r"Nature\s*:\s*([^\n]*?)(?:\s*Ajouter|$)", re.I)


def extract_dosage(name: str):
    m = DOSAGE_RE.search(name) if isinstance(name, str) else None
    return re.sub(r"\s+", " ", m.group(1)).upper().replace("UG", "µG") if m else None


def extract_pack(text: str):
    m = PACK_RE.search(text) if isinstance(text, str) else None
    return int(m.group(1)) if m else None


def base_name(name: str) -> str:
    """nom sans dosage : 'DOLIPRANE 1000 MG' -> 'doliprane'"""
    n = DOSAGE_RE.sub(" ", name) if isinstance(name, str) else ""
    return norm(n)


def clean_nature(*fields):
    for f in fields:
        if isinstance(f, str):
            m = NATURE_RE.search(f)
            if m:
                return m.group(1).strip()
    return None


RESIDUE_RE = re.compile(r"\s*Nature\s*:.*$|\s*Ajouter à mon simulateur", re.S | re.I)


def clean_residue(v):
    """retire le texte d'interface scrape ('Nature: ... Ajouter a mon simulateur')"""
    if not isinstance(v, str):
        return None
    v = RESIDUE_RE.sub("", v).strip(" ,;|")
    return v or None


def yes_no(v):
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    if v.startswith("oui"):
        return True
    if v.startswith("non"):
        return False
    return None


# --------------------------------------------------------------------------
# Medicaments
# --------------------------------------------------------------------------
def load_saydalia_meds(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path)
    parts = d["nom"].astype(str).str.split(",", n=1, expand=True)
    out = pd.DataFrame()
    out["nom"] = parts[0].str.strip()
    out["forme"] = parts[1].str.strip() if 1 in parts else None
    out["dosage"] = d["Dosage"].where(d["Dosage"].notna(), out["nom"].map(extract_dosage))
    out["presentation"] = d["Présentation"]
    out["pack"] = d["Présentation"].map(extract_pack)
    out["composition"] = d["Composition"].map(clean_residue)
    out["fabricant"] = d["Distributeur ou fabriquant"]
    out["classe_therapeutique"] = d["Classe thérapeutique"]
    out["code_atc"] = d["Code ATC"].map(clean_residue)
    out["prix_saydalia"] = d["prix_mad"]
    out["princeps"] = d["princeps"].map(lambda v: True if isinstance(v, str) and v.lower().startswith("oui") else None)
    out["remboursement"] = d["remboursement"].map(yes_no)
    out["nature"] = [
        clean_nature(a, b, c, e, f) or "Médicament"
        for a, b, c, e, f in zip(d["Statut"], d["princeps"], d["remboursement"], d["Composition"], d["Code ATC"])
    ]
    out["statut"] = d["Statut"].fillna("").astype(str).str.extract(r"(Commercialis\w+|Non commercialis\w+|Suspendu|Retir\w+)", expand=False)
    out["url_saydalia"] = d["lien"]
    return out


def load_medicament_ma(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    out = pd.DataFrame()
    out["nom"] = d["nom"].fillna("").astype(str).str.strip()
    fp = d["forme_presentation"].fillna("").astype(str)
    split = fp.map(lambda s: PRESENTATION_RE.search(s))
    out["forme"] = [s[: m.start()].strip() if m else s.strip() for s, m in zip(fp, split)]
    out["forme"] = out["forme"].str.replace(r"\s*\[[A-Z]\]\s*", " ", regex=True).str.strip()
    out["presentation"] = [s[m.start():].strip() if m else None for s, m in zip(fp, split)]
    out["pack"] = out["presentation"].map(extract_pack)
    out["dosage"] = out["nom"].map(extract_dosage)
    out["fabricant"] = d["laboratoire"]
    out["ppv"] = d["ppv"]
    out["ph"] = d["ph"]
    out["url_medicament_ma"] = d["lien"]
    return out


def merge_meds(s: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    for df in (s, m):
        df["key"] = df["nom"].map(norm)
        df["_i"] = range(len(df))
    # Passe 1 : cle exacte (nom normalise + taille de boite)
    s["k1"] = s["key"] + "|" + s["pack"].fillna(-1).astype(int).astype(str)
    m["k1"] = m["key"] + "|" + m["pack"].fillna(-1).astype(int).astype(str)
    s1 = s.drop_duplicates("k1", keep="first")
    m1 = m.drop_duplicates("k1", keep="first")
    j1 = s1.merge(m1, on="k1", suffixes=("", "_m"))
    used_s, used_m = set(j1["_i"]), set(j1["_i_m"])

    # Passe 2 : nom seul, uniquement si 1 seule ligne restante de chaque cote
    sr, mr = s[~s["_i"].isin(used_s)], m[~m["_i"].isin(used_m)]
    sc, mc = sr["key"].value_counts(), mr["key"].value_counts()
    ok = set(sc[sc == 1].index) & set(mc[mc == 1].index)
    j2 = sr[sr["key"].isin(ok)].merge(mr[mr["key"].isin(ok)], on="key", suffixes=("", "_m"))
    used_s |= set(j2["_i"])
    used_m |= set(j2["_i_m"])

    joined = pd.concat([j1, j2], ignore_index=True)
    joined["source"] = "saydalia+medicament.ma"
    s_only = s[~s["_i"].isin(used_s)].copy()
    s_only["source"] = "saydalia"
    m_only = m[~m["_i"].isin(used_m)].copy()
    m_only["source"] = "medicament.ma"
    df = pd.concat([joined, s_only, m_only], ignore_index=True)

    # Resolution des champs : medicament.ma prioritaire pour le prix, saydalia pour le reste
    def pick(a, b):
        return df[a].where(df[a].notna(), df[b]) if b in df else df[a]

    df["nom_final"] = df["nom"].where(df["nom"].notna(), df.get("nom_m"))
    df["forme_final"] = pick("forme", "forme_m")
    df["presentation_final"] = pick("presentation", "presentation_m")
    df["dosage_final"] = pick("dosage", "dosage_m")
    df["fabricant_final"] = pick("fabricant", "fabricant_m")
    df["prix"] = df["ppv"].where(df["ppv"].notna(), df["prix_saydalia"])
    df["prix_source"] = df["ppv"].notna().map({True: "ppv_medicament.ma", False: None})
    df.loc[df["prix_source"].isna() & df["prix_saydalia"].notna(), "prix_source"] = "saydalia"
    both = df["ppv"].notna() & df["prix_saydalia"].notna()
    df["conflit_prix"] = both & ((df["ppv"] - df["prix_saydalia"]).abs() / df["ppv"] > 0.01)

    final = pd.DataFrame({
        "nom": df["nom_final"],
        "nom_normalise": df["nom_final"].map(norm),
        "nom_base": df["nom_final"].map(base_name),
        "dosage": df["dosage_final"],
        "forme": df["forme_final"],
        "presentation": df["presentation_final"],
        "composition": df["composition"],
        "fabricant": df["fabricant_final"],
        "classe_therapeutique": df["classe_therapeutique"],
        "code_atc": df["code_atc"],
        "statut": df["statut"],
        "nature": df["nature"].fillna("Médicament"),
        "princeps": df["princeps"],
        "remboursement": df["remboursement"],
        "prix": df["prix"],
        "prix_source": df["prix_source"],
        "ph": df["ph"],
        "conflit_prix": df["conflit_prix"],
        "source": df["source"],
        "url_saydalia": df["url_saydalia"],
        "url_medicament_ma": df["url_medicament_ma"],
    })
    final.insert(0, "id", range(1, len(final) + 1))
    return final


# --------------------------------------------------------------------------
# Pharmacies
# --------------------------------------------------------------------------
CITY_ALIASES = {
    "Casablanca": ["casa", "casablanca", "dar el beida", "dar lbayda", "الدار البيضاء", "كازا", "الدارالبيضاء"],
    "Rabat": ["rabat", "ribat", "الرباط", "rbat"],
    "Salé": ["sale", "slа", "salé", "سلا"],
    "Marrakech": ["marrakech", "marrakesh", "marrakch", "مراكش"],
    "Fès": ["fes", "fès", "fez", "فاس"],
    "Tanger": ["tanger", "tangier", "tanja", "طنجة"],
    "Agadir": ["agadir", "اكادير", "أكادير"],
    "Meknès": ["meknes", "meknès", "مكناس"],
    "Oujda": ["oujda", "wjda", "وجدة"],
    "Tétouan": ["tetouan", "tétouan", "titwan", "تطوان"],
    "Kénitra": ["kenitra", "kénitra", "القنيطرة"],
    "Nador": ["nador", "الناظور"],
    "Safi": ["safi", "asfi", "آسفي", "اسفي"],
    "El Jadida": ["el jadida", "eljadida", "jadida", "الجديدة"],
    "Settat": ["settat", "سطات"],
    "Mohammedia": ["mohammedia", "mohammadia", "المحمدية"],
    "Béni Mellal": ["beni mellal", "béni mellal", "بني ملال"],
    "Khouribga": ["khouribga", "خريبكة"],
    "Dakhla": ["dakhla", "الداخلة"],
    "Laâyoune": ["laayoune", "laâyoune", "العيون"],
    "Dar Bouazza": ["dar bouazza", "دار بوعزة"],
}


def build_alias_map(cities):
    """alias normalise -> ville canonique (inclut toutes les villes du dataset)"""
    amap = {}
    for canon, aliases in CITY_ALIASES.items():
        for a in aliases + [canon]:
            amap[norm(a)] = canon
    for c in cities:
        amap.setdefault(norm(c), c)
    return amap


def clean_phone(v):
    if pd.isna(v):
        return None
    digits = re.sub(r"\D", "", str(v))
    if len(digits) == 9:
        digits = "0" + digits
    elif digits.startswith("212") and len(digits) == 12:
        digits = "0" + digits[3:]
    return digits if len(digits) == 10 else None


def clean_pharmacies(path: Path):
    d = pd.read_excel(path)
    n0 = len(d)
    d = d.drop_duplicates("id")
    d["_k"] = d["title"].map(norm) + "|" + d["address"].map(norm)
    d = d.drop_duplicates("_k")
    n_dedup = len(d)

    out = pd.DataFrame()
    out["id"] = d["id"]
    out["nom"] = d["title"].str.strip()
    out["nom_normalise"] = d["title"].map(norm)
    out["adresse"] = d["address"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["telephone"] = d["Phone"].map(clean_phone)
    out["ville"] = d["Ville"].fillna(d["ville_recherchee"]).str.strip()
    out["ville_normalisee"] = out["ville"].map(norm)
    out["lat"], out["lng"] = d["lat"], d["lng"]
    # Les GPS partages par > 5 etablissements sont consideres comme non fiables (centre-ville)
    shared = d.groupby(["lat", "lng"])["id"].transform("count")
    out["gps_fiable"] = shared <= 5
    out["garde_info"] = d["garde"]
    out["en_garde"] = d["garde"].notna()
    adr2 = d["Adresse 2"].fillna("").astype(str)
    is_para = adr2.str.match(r"^\s*Parapharmacie", case=False) | out["nom"].str.contains(
        r"laborato|distribution|negoce|négoce|grossiste|repartition|répartition", case=False, regex=True)
    out["type"] = is_para.map({True: "parapharmacie_ou_labo", False: "pharmacie"})
    out = out.reset_index(drop=True)
    return out, {"lignes_brutes": n0, "apres_dedoublonnage": n_dedup}


# --------------------------------------------------------------------------
# Rapport qualite + export
# --------------------------------------------------------------------------
def completeness(df, cols):
    return {c: round(float(df[c].notna().mean()) * 100, 1) for c in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # evite les erreurs d'accents sous Windows
    inp, out = Path(a.inp), Path(a.out)
    for f in (FILE_SAYDALIA_MEDS, FILE_MEDICAMENT_MA, FILE_SAYDALIA_PHARMA):
        if not (inp / f).exists():
            sys.exit(f"Fichier introuvable : {inp / f}")
    out.mkdir(parents=True, exist_ok=True)

    s = load_saydalia_meds(inp / FILE_SAYDALIA_MEDS)
    m = load_medicament_ma(inp / FILE_MEDICAMENT_MA)
    n_s, n_m = len(s), len(m)
    meds = merge_meds(s, m)

    ph, ph_stats = clean_pharmacies(inp / FILE_SAYDALIA_PHARMA)
    pharmacies = ph[ph["type"] == "pharmacie"].reset_index(drop=True)
    non_ph = ph[ph["type"] != "pharmacie"].reset_index(drop=True)

    aliases = build_alias_map(pharmacies["ville"].dropna().unique())
    pharmacies["ville_canonique"] = pharmacies["ville_normalisee"].map(aliases)

    meds.to_csv(out / "medications.csv", index=False, encoding="utf-8-sig")
    pharmacies.to_csv(out / "pharmacies.csv", index=False, encoding="utf-8-sig")
    non_ph.to_csv(out / "non_pharmacies.csv", index=False, encoding="utf-8-sig")
    (out / "city_aliases.json").write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")

    with sqlite3.connect(out / "kb.sqlite") as cx:
        meds.to_sql("medications", cx, if_exists="replace", index=False)
        pharmacies.to_sql("pharmacies", cx, if_exists="replace", index=False)
        cx.execute("CREATE INDEX IF NOT EXISTS ix_med_nom ON medications(nom_normalise)")
        cx.execute("CREATE INDEX IF NOT EXISTS ix_med_base ON medications(nom_base)")
        cx.execute("CREATE INDEX IF NOT EXISTS ix_ph_ville ON pharmacies(ville_canonique)")

    report = {
        "medicaments": {
            "saydalia_brut": n_s, "medicament_ma_brut": n_m, "fusionne": len(meds),
            "par_source": meds["source"].value_counts().to_dict(),
            "par_nature": meds["nature"].value_counts().to_dict(),
            "conflits_prix": int(meds["conflit_prix"].sum()),
            "completude_pct": completeness(meds, ["dosage", "forme", "presentation", "composition", "fabricant",
                                                  "classe_therapeutique", "code_atc", "remboursement", "prix"]),
        },
        "pharmacies": {
            **ph_stats, "pharmacies": len(pharmacies), "parapharmacies_labos_exclus": len(non_ph),
            "telephone_valide_pct": completeness(pharmacies, ["telephone"])["telephone"],
            "gps_fiable_pct": round(float(pharmacies["gps_fiable"].mean()) * 100, 1),
            "en_garde": int(pharmacies["en_garde"].sum()),
            "villes": pharmacies["ville_canonique"].nunique(),
        },
    }
    (out / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
