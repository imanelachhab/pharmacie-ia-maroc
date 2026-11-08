#!/usr/bin/env python3
"""
evaluate.py - Evalue le module NLP sur le dataset de requetes.

Usage :
  python evaluate.py                      # evalue le split dev puis test
  python evaluate.py --split test
  python evaluate.py --handwritten ..\\queries\\queries_handwritten_template.csv

Sorties (dossier --out) : evaluation_report.json, errors_<split>.csv
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from app.nlp import NLU, normalize  # noqa: E402

DEFAULT_KB = Path(os.environ.get("KB_DIR", r"C:\Users\DELL\dataset_challenge\kb"))
DEFAULT_Q = Path(os.environ.get("QUERIES_DIR", r"C:\Users\DELL\dataset_challenge\queries"))


def build_nlu(kb: Path) -> NLU:
    meds = pd.read_csv(kb / "medications.csv", usecols=["nom_base"])["nom_base"].dropna().unique()
    aliases = json.loads((kb / "city_aliases.json").read_text(encoding="utf-8")) if (kb / "city_aliases.json").exists() else {}
    cities = pd.read_csv(kb / "pharmacies.csv", usecols=["ville_canonique"])["ville_canonique"].dropna().unique()
    return NLU(meds, aliases, cities)


def vkey(label: str, value: str) -> str:
    v = str(value).lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    return f"{label}:{normalize(v) if label != 'DOSAGE' else v}"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": tp + fn}


def evaluate(nlu: NLU, df: pd.DataFrame, name: str, out: Path):
    rows, errors = [], []
    intent_c = defaultdict(lambda: [0, 0, 0])   # tp, fp, fn par label
    ent_c = defaultdict(lambda: [0, 0, 0])
    by = {"language": defaultdict(list), "query_style": defaultdict(list)}
    lang_ok = lang_relaxed = 0
    relax = {"mixed": "latin_darija", "darija_latin": "latin_darija", "ar": "arabic", "darija_ar": "arabic", "fr": "fr"}

    for r in df.itertuples():
        a = nlu.analyze(r.raw_text)
        gold_i = set(x.strip() for x in str(r.intent).split("+"))
        pred_i = set(a.intents)
        ok = gold_i == pred_i
        for l in gold_i | pred_i:
            if l in gold_i and l in pred_i: intent_c[l][0] += 1
            elif l in pred_i: intent_c[l][1] += 1
            else: intent_c[l][2] += 1
        gold_e = {vkey(e["label"], e["value"]) for e in json.loads(r.entities)}
        pred_e = {vkey(e.label, e.value) for e in a.entities if e.text != ""}
        for k in gold_e | pred_e:
            lab = k.split(":")[0]
            if k in gold_e and k in pred_e: ent_c[lab][0] += 1
            elif k in pred_e: ent_c[lab][1] += 1
            else: ent_c[lab][2] += 1
        ent_ok = gold_e == pred_e
        for g in ("language", "query_style"):
            by[g][getattr(r, g)].append(ok)
        lang_ok += a.language == r.language
        lang_relaxed += relax.get(a.language) == relax.get(r.language)
        rows.append(ok)
        if not ok or not ent_ok:
            errors.append({"id": r.id, "raw_text": r.raw_text, "language": r.language, "style": r.query_style,
                           "gold_intent": r.intent, "pred_intent": a.intent, "gold_entities": sorted(gold_e),
                           "pred_entities": sorted(pred_e), "pred_language": a.language})
    n = len(df)
    rep = {
        "split": name, "n": n,
        "intent_exact_match_accuracy": round(sum(rows) / n, 4),
        "intent_per_label": {k: prf(*v) for k, v in sorted(intent_c.items())},
        "intent_macro_f1": round(sum(prf(*v)["f1"] for v in intent_c.values()) / len(intent_c), 4),
        "accuracy_by_language": {k: round(sum(v) / len(v), 3) for k, v in by["language"].items()},
        "accuracy_by_style": {k: round(sum(v) / len(v), 3) for k, v in by["query_style"].items()},
        "entities_per_label": {k: prf(*v) for k, v in sorted(ent_c.items())},
        "entities_micro_f1": prf(*[sum(v[i] for v in ent_c.values()) for i in range(3)])["f1"],
        "language_id_accuracy_strict": round(lang_ok / n, 4),
        "language_id_accuracy_relaxed": round(lang_relaxed / n, 4),
    }
    pd.DataFrame(errors).to_csv(out / f"errors_{name}.csv", index=False, encoding="utf-8-sig")
    rep["n_errors_logged"] = len(errors)
    return rep


def evaluate_handwritten(nlu: NLU, path: Path):
    d = pd.read_csv(path).dropna(subset=["raw_text", "intent"])
    d = d[d.raw_text.astype(str).str.strip() != ""]
    if d.empty:
        return None
    ok = med_ok = med_n = city_ok = city_n = 0
    fails = []
    for r in d.itertuples():
        a = nlu.analyze(str(r.raw_text))
        good = set(a.intents) == {x.strip() for x in str(r.intent).split("+")}
        ok += good
        if not good:
            fails.append((r.raw_text, r.intent, a.intent))
        if isinstance(r.city, str) and r.city.strip():
            city_n += 1
            city_ok += any(e.value == r.city.strip() for e in a.get("CITY"))
        if isinstance(r.medication, str) and r.medication.strip():
            med_n += 1
            med_ok += any(e.value.lower() == r.medication.strip().lower() for e in a.get("MEDICATION"))
    return {"n": len(d), "intent_exact_match_accuracy": round(ok / len(d), 4),
            "medication_accuracy": round(med_ok / med_n, 4) if med_n else None,
            "city_accuracy": round(city_ok / city_n, 4) if city_n else None,
            "intent_failures": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=str(DEFAULT_KB))
    ap.add_argument("--queries", default=str(DEFAULT_Q))
    ap.add_argument("--split", choices=["dev", "test", "train", "all"], default="all")
    ap.add_argument("--handwritten", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nlu = build_nlu(Path(a.kb))
    df = pd.read_csv(Path(a.queries) / "queries.csv")
    splits = ["dev", "test"] if a.split == "all" else [a.split]
    report = {s: evaluate(nlu, df[df.split == s], s, out) for s in splits}
    if a.handwritten:
        hw = evaluate_handwritten(nlu, Path(a.handwritten))
        if hw:
            report["handwritten"] = hw
    (out / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for s, r in report.items():
        print(f"\n=== {s} (n={r['n']}) ===")
        for k in ("intent_exact_match_accuracy", "intent_macro_f1", "entities_micro_f1",
                  "language_id_accuracy_relaxed"):
            if k in r:
                print(f"{k}: {r[k]}")
        if "accuracy_by_language" in r:
            print("par langue :", r["accuracy_by_language"])
            print("par style  :", r["accuracy_by_style"])
            print("entites    :", {k: v["f1"] for k, v in r["entities_per_label"].items()})


if __name__ == "__main__":
    main()
