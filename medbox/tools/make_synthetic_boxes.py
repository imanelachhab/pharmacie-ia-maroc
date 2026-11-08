#!/usr/bin/env python3
"""
make_synthetic_boxes.py - Genere des images de boites de medicaments a partir de ta base (kb/medications.csv).

But : disposer tout de suite d'un jeu de test AVEC verite terrain pour developper et comparer le pipeline
(OCR + matching) sous flou, faible lumiere, reflets, rotation, boite partiellement visible, bruit.

ATTENTION : ces images sont synthetiques (texte propre + degradations simulees). Les scores obtenus sont un
MAJORANT : ils seront plus bas sur de vraies photos. Voir README : il faut photographier de vraies boites.

Usage :
  python tools/make_synthetic_boxes.py --kb C:\\Users\\DELL\\dataset_challenge\\kb --out data\\synthetic --n 40
"""
import argparse
import os
import random
import re
import sys
from pathlib import Path

import arabic_reshaper
import cv2
import numpy as np
import pandas as pd
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

BOLD = ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
REG = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
       "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/Library/Fonts/Arial.ttf"]
PALETTE = [(0, 122, 94), (200, 40, 50), (30, 90, 170), (240, 150, 20), (110, 50, 150), (20, 140, 170), (60, 60, 70), (180, 30, 110)]
AR_FORME = {"compr": "أقراص", "gelul": "كبسولات", "sirop": "شراب", "solution": "محلول", "inject": "حقن",
            "sachet": "أكياس", "suppo": "تحاميل", "pommade": "مرهم", "creme": "كريم", "collyre": "قطرة العين"}
VARIANTS = ["clean", "blur", "dark", "glare", "rotate", "crop", "noise"]

ARABIC_NAMES = {}
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.arabic_names import ARABIC_NAMES  # noqa
except Exception:
    pass


def font(paths, size):
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fit(draw, text, paths, max_w, max_size, min_size=18):
    for s in range(max_size, min_size - 1, -2):
        f = font(paths, s)
        if draw.textlength(text, font=f) <= max_w:
            return f
    return font(paths, min_size)


def ar(text):
    """Pillow avec libraqm gere seul la liaison des lettres et le sens droite->gauche ; sinon on le fait ici"""
    from PIL import features
    return text if features.check("raqm") else get_display(arabic_reshaper.reshape(text))


def barcode(draw, x, y, w, h, rng):
    draw.rectangle([x, y, x + w, y + h], fill="white")
    cx = x + 8
    while cx < x + w - 8:
        bw = rng.choice([2, 2, 3, 4])
        draw.rectangle([cx, y + 6, cx + bw - 1, y + h - 22], fill="black")
        cx += bw + rng.choice([2, 3, 4])
    draw.text((x + 12, y + h - 20), "".join(rng.choice("0123456789") for _ in range(13)), fill="black", font=font(REG, 14))


def draw_face(r, layout, rng):
    W, H = 700, 920
    base = r["nom_base"].upper()
    color = rng.choice(PALETTE)
    if layout == "B":
        bg, fg, name_c, sub_c = (22, 38, 52), "white", (255, 214, 10), (230, 230, 230)
    elif layout == "C":
        bg, fg, name_c, sub_c = (250, 250, 247), (35, 35, 35), color, (70, 70, 70)
    else:
        bg, fg, name_c, sub_c = (252, 252, 252), (35, 35, 35), "white", (60, 60, 60)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    if layout == "A":
        d.rectangle([0, 0, W, 260], fill=color)
        d.ellipse([W - 190, -60, W + 60, 190], fill=tuple(min(255, c + 35) for c in color))
    if layout == "C":
        d.rectangle([0, H - 150, W, H], fill=color)
        d.rectangle([0, 0, 28, H - 150], fill=color)
    # pictogramme
    d.rectangle([W - 120, H - 270, W - 40, H - 190], outline=fg if layout != "C" else color, width=6)
    # nom de marque
    f = fit(d, base, BOLD, W - 90, 120, 40)
    ny = 70 if layout == "A" else 60
    d.text((50, ny), base, font=f, fill=name_c)
    # dosage / forme
    y = 300 if layout == "A" else 230
    dosage = str(r["dosage"]) if pd.notna(r["dosage"]) else ""
    if dosage:
        d.text((50, y), dosage.lower(), font=fit(d, dosage, BOLD, W - 100, 84, 36), fill=color if layout != "B" else (120, 220, 190))
        y += 105
    forme = str(r["forme"]) if pd.notna(r["forme"]) else ""
    if forme:
        d.text((50, y), forme, font=fit(d, forme, REG, W - 100, 40, 22), fill=sub_c)
        y += 58
    pres = str(r["presentation"]) if pd.notna(r["presentation"]) else ""
    if pres:
        pres = pres.replace("Boite", "Boîte")
        d.text((50, y), pres, font=fit(d, pres, REG, W - 100, 38, 22), fill=sub_c)
        y += 56
    # ligne arabe : nom arabe si connu, sinon type de forme
    fl = forme.lower()
    ar_form = next((v for k, v in AR_FORME.items() if k in fl), "دواء")
    ar_text = (rng.choice(ARABIC_NAMES[r["nom_base"]]) + " · " if r["nom_base"] in ARABIC_NAMES and rng.random() < .5 else "") + ar_form
    f_ar = fit(d, ar(ar_text), REG, W - 100, 46, 24)
    d.text((W - 50 - d.textlength(ar(ar_text), font=f_ar), y + 8), ar(ar_text), font=f_ar, fill=sub_c)
    y += 80
    # laboratoire
    lab = str(r["fabricant"]) if pd.notna(r["fabricant"]) else ""
    if lab:
        d.text((50, H - 120 if layout != "C" else H - 215), lab, font=fit(d, lab, REG, W - 280, 30, 18),
               fill=sub_c)
    barcode(d, 50, H - 205 if layout != "C" else H - 330, 230, 80, rng) if layout != "C" else None
    return im


def scene(face, rng, angle=0.0, persp=0.04, size=(1000, 1250)):
    W, H = size
    bg = np.full((H, W, 3), rng.choice([(140, 140, 140), (200, 195, 185), (95, 90, 85), (225, 225, 225)]), np.uint8)
    noise = np.random.default_rng(rng.randrange(10**6)).normal(0, 9, bg.shape)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    fw, fh = face.size
    s = rng.uniform(0.82, 0.95) * min(W / fw, H / fh)
    w, h = fw * s, fh * s
    cx, cy = W / 2 + rng.uniform(-30, 30), H / 2 + rng.uniform(-30, 30)
    src = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
    pts = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = pts @ R.T + [cx, cy]
    pts += np.array([[rng.uniform(-1, 1) * persp * w, rng.uniform(-1, 1) * persp * h] for _ in range(4)], np.float32)
    M = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    img = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full((fh, fw), 255, np.uint8), M, (W, H))
    # ombre portee
    shadow = cv2.GaussianBlur(cv2.warpPerspective(np.full((fh, fw), 90, np.uint8), M, (W, H)), (0, 0), 14)
    shadow = np.roll(shadow, (12, 12), (0, 1))
    out = (bg * (1 - shadow[..., None] / 255.0)).astype(np.uint8)
    m = (mask[..., None] / 255.0)
    return (out * (1 - m) + warped * m).astype(np.uint8)


def degrade(img, kind, rng):
    h, w = img.shape[:2]
    if kind == "blur":
        if rng.random() < .5:
            k = rng.choice([9, 13, 17])
            ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1 / k
            M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1)
            ker = cv2.warpAffine(ker, M, (k, k)); ker /= max(ker.sum(), 1e-6)
            return cv2.filter2D(img, -1, ker)
        return cv2.GaussianBlur(img, (0, 0), rng.uniform(2.0, 3.6))
    if kind == "dark":
        g = rng.uniform(1.9, 2.6)
        out = (255 * (img / 255.0) ** g * rng.uniform(0.65, 0.9)).astype(np.uint8)
        return out
    if kind == "glare":
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(.2, .8) * w, rng.uniform(.15, .6) * h
        g = np.exp(-(((xx - cx) / (w * .22)) ** 2 + ((yy - cy) / (h * .12)) ** 2))
        return np.clip(img + (g[..., None] * rng.uniform(190, 255)), 0, 255).astype(np.uint8)
    if kind == "noise":
        n = np.random.default_rng(rng.randrange(10**6)).normal(0, 20, img.shape)
        out = np.clip(img + n, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 22])
        return cv2.imdecode(enc, 1)
    if kind == "crop":
        side = rng.choice(["right", "bottom", "left"])
        f = rng.uniform(0.25, 0.4)
        if side == "right":
            return img[:, : int(w * (1 - f))]
        if side == "left":
            return img[:, int(w * f):]
        return img[: int(h * (1 - f)), :]
    return img


def pick_products(kb, n, rng):
    d = pd.read_csv(Path(kb) / "medications.csv")
    d = d[(d["nature"] == "Médicament") & d["dosage"].notna() & d["presentation"].notna() & d["fabricant"].notna()]
    d = d[d["nom_base"].astype(str).str.fullmatch(r"[a-z]{5,14}")]
    d = d[d["dosage"].astype(str).str.fullmatch(r"\d+(?:[.,]\d+)? ?(?:MG|G|ML|UI|%)", case=False)]
    d = d.drop_duplicates("nom_base")
    pop = d[d["nom_base"].isin(["doliprane", "efferalgan", "augmentin", "spasfon", "voltarene", "aspegic", "smecta",
                                "gaviscon", "ventoline", "clamoxyl", "panadol", "rhumagrip", "oroken", "humex"])]
    rest = d[~d.index.isin(pop.index)].sample(frac=1, random_state=rng.randrange(10**6))
    return pd.concat([pop, rest]).head(n)


def fake_products(kb, n, rng):
    """boites de medicaments INVENTES (absents de la base) : le systeme doit repondre not_found / uncertain"""
    known = set(pd.read_csv(Path(kb) / "medications.csv", usecols=["nom_base"])["nom_base"].dropna())
    syl = ["zor", "vex", "ta", "mir", "lo", "quen", "dra", "pho", "nel", "bex", "ri", "ul", "kar", "sy", "tor", "min"]
    rows = []
    while len(rows) < n:
        name = "".join(rng.choice(syl) for _ in range(rng.choice([2, 3])))
        if name in known or len(name) < 6:
            continue
        rows.append({"id": -1, "nom_base": name, "nom": name.upper(), "dosage": rng.choice(["250 MG", "500 MG", "20 MG", "1 G"]),
                     "forme": rng.choice(["Comprimé", "Gélule", "Sirop"]), "presentation": f"Boîte de {rng.choice([10, 14, 20, 30])}",
                     "fabricant": rng.choice(["PHARMA 5", "SOTHEMA", "COOPER PHARMA", "AFRIC-PHAR"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", type=int, default=0, help="nombre de boites de medicaments inventes (negatifs)")
    ap.add_argument("--kb", default=r"C:\Users\DELL\dataset_challenge\kb")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=40, help="nombre de medicaments")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prods = pick_products(a.kb, a.n, rng)
    if a.fake:
        prods = pd.concat([prods, fake_products(a.kb, a.fake, rng)], ignore_index=True)
    rows = []
    for _, r in prods.iterrows():
        layout = rng.choice(["A", "B", "C"])
        face = draw_face(r, layout, rng)
        for v in a.variants.split(","):
            ang, persp = 0.0, 0.015
            if v == "rotate":
                ang, persp = rng.choice([-1, 1]) * rng.uniform(12, 25), 0.07
            img = scene(face, rng, ang, persp)
            if v == "rotate" and rng.random() < 0.3:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = degrade(img, v, rng)
            fn = f"{'fake' if int(r['id']) < 0 else format(int(r['id']), '05d')}_{r['nom_base']}_{v}.jpg"
            cv2.imwrite(str(out / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"file": fn, "id": int(r["id"]), "nom_base": r["nom_base"], "nom": r["nom"], "dosage": r["dosage"],
                         "forme": r["forme"], "presentation": r["presentation"], "fabricant": r["fabricant"],
                         "layout": layout, "degradation": v})
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False, encoding="utf-8-sig")
    print(f"{len(rows)} images ({len(prods)} medicaments x {len(a.variants.split(','))} variantes) -> {out}")


if __name__ == "__main__":
    main()
