"""Lecture du texte d'une boite de medicament (francais + arabe).

Deux moteurs interchangeables :
  - "tesseract" : leger, a installer separement (https://github.com/UB-Mannheim/tesseract/wiki) + langues fra, ara
  - "easyocr"   : pip install easyocr (telecharge ses modeles au 1er lancement, plus lourd, souvent meilleur)
  - "auto"      : easyocr si installe, sinon tesseract

Le texte des boites est difficile : texte clair sur fond sombre, texte incline, flou, reflets.
On lit donc plusieurs versions de l'image (normale, inversee, contrastee) et on fusionne les lignes.
"""
import os
import re
import shutil
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

AR_RE = re.compile(r"[\u0600-\u06FF]")


@dataclass
class Line:
    text: str
    conf: float          # 0..1
    script: str          # "latin" | "arabic"


@dataclass
class OCRResult:
    lines: list = field(default_factory=list)
    rotation: int = 0
    timings: dict = field(default_factory=dict)
    backend: str = ""

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines)


# --------------------------------------------------------------------------
# Pretraitement
# --------------------------------------------------------------------------
def resize_for_ocr(img, target=1800):
    h, w = img.shape[:2]
    m = max(h, w)
    if m < 1200:
        s = min(2.0, target / m)
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if m > 2400:
        s = 2200 / m
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def variants(img, fast=False):
    """liste de (nom, image grise) a lire. Le canal de couleur le plus contraste ("ch") sauve le texte blanc sur
    fond colore, que la conversion en gris rend illisible (meme luminosite)."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        chans = cv2.split(img)
        ch = max(chans, key=lambda c: float(cv2.meanStdDev(c)[1][0][0]))
        ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX)
    else:
        g, ch = img, img
    gc, chc = clahe.apply(g), ch          # canal etire mais sans CLAHE (qui degrade le texte blanc sur fond colore)
    out = [("gray", gc), ("ch", chc)]
    if not fast:
        blur = cv2.GaussianBlur(g, (0, 0), 1.0)
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12)
        out += [("inv", 255 - gc), ("chinv", 255 - chc), ("thr", thr)]
    return out


def rotate(img, angle):
    if angle == 0:
        return img
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]
    return cv2.rotate(img, code)


def deskew(img):
    """corrige l'inclinaison de la boite a partir de ses longs bords (Hough) ; retourne (image, angle) ou (None, 0)"""
    h, w = img.shape[:2]
    sc = 900 / max(h, w)
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else img
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(g, 50, 150)
    mn = 0.22 * max(small.shape[:2])
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(mn), maxLineGap=12)
    if lines is None or len(lines) == 0:
        return None, 0.0
    lines = np.asarray(lines).reshape(-1, 4)       # (N,1,4) ou (N,4) selon la version d'OpenCV
    hist = np.zeros(90)
    for x1, y1, x2, y2 in lines:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 90          # 0..90, modulo 90 : horizontal ~ vertical
        hist[int(ang) % 90] += np.hypot(x2 - x1, y2 - y1)
    hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, "valid")
    ang = float(np.argmax(hist))
    if ang > 45:
        ang -= 90                                                     # -45..45
    if abs(ang) < 2.0 or abs(ang) > 40:
        return None, 0.0
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)             # cv2 : angle > 0 = sens anti-horaire
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), ang


def shrink(img, m=900):
    mx = max(img.shape[:2])
    return cv2.resize(img, None, fx=m / mx, fy=m / mx, interpolation=cv2.INTER_AREA) if mx > m else img


# --------------------------------------------------------------------------
# Moteurs
# --------------------------------------------------------------------------
class TesseractBackend:
    name = "tesseract"

    def __init__(self):
        import pytesseract
        cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if not cmd and os.path.exists(win):
            cmd = win
        if not cmd:
            raise RuntimeError("Tesseract introuvable : installez-le ou definissez TESSERACT_CMD")
        pytesseract.pytesseract.tesseract_cmd = cmd
        self.pt = pytesseract
        langs = self.pt.get_languages(config="")
        self.has_ar = "ara" in langs
        self.latin_lang = "fra+eng" if "fra" in langs else "eng"

    def _lines(self, gray, lang, psm, script):
        d = self.pt.image_to_data(gray, lang=lang, config=f"--psm {psm} --oem 1", output_type=self.pt.Output.DICT)
        groups = {}
        for i, t in enumerate(d["text"]):
            t = (t or "").strip()
            try:
                c = float(d["conf"][i])
            except ValueError:
                continue
            if not t or c < 15:
                continue
            groups.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append((t, c))
        out = []
        for ws in groups.values():
            txt = " ".join(w for w, _ in ws)
            conf = sum(c for _, c in ws) / len(ws) / 100
            out.append(Line(txt, conf, script))
        return out

    def read(self, gray, script="latin", fast=False):
        if script == "arabic":
            if not self.has_ar:
                return []
            return self._lines(gray, "ara", 11, "arabic")
        out = self._lines(gray, self.latin_lang, 11, "latin")
        return out if fast else out + self._lines(gray, self.latin_lang, 6, "latin")

    def quick_score(self, img_bgr):
        """pour choisir la rotation : somme des mots alphabetiques surs"""
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        d = self.pt.image_to_data(g, lang="eng", config="--psm 11 --oem 1", output_type=self.pt.Output.DICT)
        s = 0.0
        for t, c in zip(d["text"], d["conf"]):
            try:
                c = float(c)
            except ValueError:
                continue
            t = (t or "").strip()
            if c >= 60 and len(t) >= 3 and t.isalpha():
                s += len(t) * c / 100
        return s


class EasyOCRBackend:
    name = "easyocr"

    def __init__(self):
        import easyocr
        self.lat = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
        self.ara = easyocr.Reader(["ar", "en"], gpu=False, verbose=False)   # l'arabe ne se combine pas avec le francais

    def read(self, gray, script="latin", fast=False):
        rd = self.ara if script == "arabic" else self.lat
        out = []
        for _bbox, text, conf in rd.readtext(gray, detail=1, paragraph=False):
            text = text.strip()
            if text and conf >= 0.15:
                out.append(Line(text, float(conf), script))
        return out

    def quick_score(self, img_bgr):
        s = 0.0
        for _b, t, c in self.lat.readtext(img_bgr, detail=1, paragraph=False):
            t = t.strip()
            if c >= 0.6 and len(t) >= 3 and t.isalpha():
                s += len(t) * c
        return s


def make_backend(name="auto"):
    if name in ("auto", "easyocr"):
        try:
            return EasyOCRBackend()
        except ImportError:
            if name == "easyocr":
                raise
    return TesseractBackend()


# --------------------------------------------------------------------------
class OCR:
    def __init__(self, backend="auto"):
        self.be = make_backend(os.environ.get("OCR_BACKEND", backend))

    GOOD = 14.0   # score au-dessus duquel l'orientation est consideree comme bonne

    def _orient(self, img):
        """(image, rotation, redressee) : image brute ; sinon correction d'inclinaison ; sinon rotations 90/180/270"""
        s0 = self.be.quick_score(shrink(img))
        dsk, ang = deskew(img)
        if s0 >= self.GOOD and dsk is None:
            return img, 0, False
        best = (s0, img, 0, False)
        if dsk is not None:
            sc = self.be.quick_score(shrink(dsk))
            if sc > best[0] * 1.15 + 1:
                best = (sc, dsk, 0, True)
        if best[0] < self.GOOD:
            for im, tag in ((img, False), (dsk, True)):
                if im is None:
                    continue
                for a in (90, 270, 180):
                    cand = rotate(im, a)
                    sc = self.be.quick_score(shrink(cand))
                    if sc > best[0] * 1.3 + 1:
                        best = (sc, cand, a, tag)
        return best[1], best[2], best[3]

    @staticmethod
    def _merge(lines):
        """garde la meilleure confiance pour chaque texte distinct ; ecarte le bruit"""
        best = {}
        for l in lines:
            t = re.sub(r"\s+", " ", l.text).strip()
            if l.script == "arabic":
                letters = len(AR_RE.findall(t))
                if letters < 2 or letters / max(1, len(t.replace(" ", ""))) < 0.6:
                    continue
            else:
                alnum = sum(c.isalnum() for c in t)
                if alnum < 2 or alnum / max(1, len(t)) < 0.5:
                    continue
            k = (l.script, t.lower())
            if k not in best or l.conf > best[k].conf:
                best[k] = Line(t, l.conf, l.script)
        return sorted(best.values(), key=lambda x: -x.conf)

    def read(self, img_bgr, fast=False, auto_rotate=True) -> OCRResult:
        t0 = time.perf_counter()
        img = resize_for_ocr(img_bgr)
        rot, rect = 0, False
        if auto_rotate:
            img, rot, rect = self._orient(img)
        t1 = time.perf_counter()
        lines = []
        for name, g in variants(img, fast):
            lines += self.be.read(g, "latin", fast)
            if name == "gray" or (name == "ch" and not fast):
                lines += self.be.read(g, "arabic")
        t2 = time.perf_counter()
        return OCRResult(self._merge(lines), rot, {"rectified": rect, "rotation_ms": round((t1 - t0) * 1000),
                                                   "ocr_ms": round((t2 - t1) * 1000)}, self.be.name)
