#!/usr/bin/env python3
"""
scrape_product_images.py - Telecharge les photos de boites a partir des pages produit deja listees dans ta base
(colonnes url_saydalia / url_medicament_ma de kb/medications.csv). Chaque image est reliee a son medicament :
pas d'etiquetage manuel.

Regles de bonne conduite (integrees) :
  * lit robots.txt et saute les pages interdites ;
  * attend entre deux requetes (--delay) ;
  * S'ARRETE si le site repond 403 / 429 (protection anti-robot) : il ne cherche JAMAIS a la contourner ;
  * garde la page source de chaque image (manifest.csv) ;
  * usage academique : verifie les conditions d'utilisation du site avant d'utiliser ou publier les images.

Etape 1 - SONDE (pour savoir si le site contient des photos exploitables), ~10 pages :
  python tools/scrape_product_images.py --source saydalia --probe 10
Etape 2 - telechargement :
  python tools/scrape_product_images.py --source saydalia --limit 300 --delay 3
"""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

SOURCES = {"saydalia": "url_saydalia", "medicament_ma": "url_medicament_ma"}
BAD_URL = re.compile(r"logo|icon|sprite|placeholder|no[-_]?image|default|avatar|banner|favicon|loader|flag", re.I)
PRODUCT_HINT = re.compile(r"product|produit|wp-post-image|zoom|gallery|medic|pack|packshot|main", re.I)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:40]


def image_candidates(html, base):
    """URLs d'images plausibles pour le produit, par ordre de priorite"""
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    def add(u):
        if u:
            u = urljoin(base, u.strip())
            if u.startswith("http") and u not in urls and not BAD_URL.search(urlparse(u).path):
                urls.append(u)

    for m in soup.find_all("meta", attrs={"property": "og:image"}):
        add(m.get("content"))
    for l in soup.find_all("link", rel="image_src"):
        add(l.get("href"))
    for sc in soup.find_all("script", type="application/ld+json"):
        try:
            j = json.loads(sc.string or "")
        except Exception:
            continue
        for item in (j if isinstance(j, list) else [j]):
            img = item.get("image") if isinstance(item, dict) else None
            for u in (img if isinstance(img, list) else [img]):
                add(u.get("url") if isinstance(u, dict) else u)
    for im in soup.find_all("img"):
        cls = " ".join(im.get("class", [])) + " " + (im.get("id") or "") + " " + " ".join(
            p for p in [(im.parent.get("class") and " ".join(im.parent.get("class"))) or ""])
        src = im.get("data-src") or im.get("data-large_image") or im.get("src")
        if src and PRODUCT_HINT.search(cls + " " + src):
            add(src)
    return urls


class Polite:
    def __init__(self, delay, ua, ignore_robots=False):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Language": "fr,ar;q=0.8,en;q=0.5"})
        self.delay, self.robots, self.ignore = delay, {}, ignore_robots
        self.last = 0.0
        self.blocked = 0

    def allowed(self, url):
        if self.ignore:
            return True
        host = "{0.scheme}://{0.netloc}".format(urlparse(url))
        rp = self.robots.get(host)
        if rp is None:
            rp = robotparser.RobotFileParser()
            try:
                r = self.s.get(host + "/robots.txt", timeout=15)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])
            self.robots[host] = rp
        return rp.can_fetch(self.s.headers["User-Agent"], url)

    def get(self, url, **kw):
        wait = self.delay + random.uniform(0, self.delay * 0.4) - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        r = self.s.get(url, timeout=25, **kw)
        if r.status_code in (403, 429, 503) and "text/html" in r.headers.get("content-type", ""):
            self.blocked += 1
        else:
            self.blocked = 0
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=r"C:\Users\DELL\dataset_challenge\kb")
    ap.add_argument("--source", choices=list(SOURCES), default="saydalia")
    ap.add_argument("--out", default="data/scraped")
    ap.add_argument("--probe", type=int, default=0, help="n pages au hasard : rapport seulement, aucun fichier garde")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--delay", type=float, default=3.0, help="secondes entre deux requetes")
    ap.add_argument("--max-per-page", type=int, default=2)
    ap.add_argument("--min-side", type=int, default=300)
    ap.add_argument("--contact", default="etudiant@exemple.ma", help="affiche dans le User-Agent")
    ap.add_argument("--kb-file", default=None, help="CSV a la place de <kb>/medications.csv (tests)")
    ap.add_argument("--ignore-robots", action="store_true", help="NE PAS utiliser sur un site public")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    col = SOURCES[a.source]
    d = pd.read_csv(a.kb_file or Path(a.kb) / "medications.csv")
    d = d[d[col].notna()].drop_duplicates(col)
    if d.empty:
        sys.exit(f"Aucune URL dans la colonne {col}")
    d = d.sample(frac=1, random_state=1).head(a.probe or a.limit)
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    web = Polite(a.delay, f"Mozilla/5.0 (academic project; contact: {a.contact})", a.ignore_robots)

    manifest, seen = [], {}
    stats = {"pages": 0, "robots_disallowed": 0, "http_errors": 0, "pages_with_candidates": 0, "images_ok": 0,
             "too_small": 0, "duplicates": 0}
    for _, r in d.iterrows():
        url = r[col]
        if not web.allowed(url):
            stats["robots_disallowed"] += 1
            continue
        try:
            resp = web.get(url)
        except requests.RequestException:
            stats["http_errors"] += 1
            continue
        if web.blocked >= 3:
            print("\n!! Le site bloque les requetes automatiques (403/429 repetes). Arret : je ne contourne pas la "
                  "protection. Reduisez le rythme (--delay 8), reessayez plus tard, ou changez de source.")
            break
        if resp.status_code != 200:
            stats["http_errors"] += 1
            continue
        stats["pages"] += 1
        cands = image_candidates(resp.text, url)
        if cands:
            stats["pages_with_candidates"] += 1
        got = 0
        for iu in cands:
            if got >= a.max_per_page:
                break
            try:
                ir = web.get(iu)
                im = Image.open(io.BytesIO(ir.content)); im.load()
            except Exception:
                continue
            if min(im.size) < a.min_side:
                stats["too_small"] += 1
                continue
            h = hashlib.sha1(ir.content).hexdigest()
            seen[h] = seen.get(h, 0) + 1
            if seen[h] > 1:           # meme image sur plusieurs pages : logo / image par defaut
                stats["duplicates"] += 1
                continue
            fn = f"{int(r['id']):05d}_{slug(r['nom_base'])}_{got}.jpg"
            if not a.probe:
                im.convert("RGB").save(out / "images" / fn, quality=93)
            manifest.append({"file": fn, "id": int(r["id"]), "nom_base": r["nom_base"], "nom": r["nom"],
                             "dosage": r.get("dosage"), "forme": r.get("forme"), "presentation": r.get("presentation"),
                             "fabricant": r.get("fabricant"), "source_page": url, "image_url": iu, "sha1": h,
                             "width": im.size[0], "height": im.size[1], "scraped_at": datetime.now().isoformat(timespec="seconds"),
                             "degradation": "real"})
            got += 1
            stats["images_ok"] += 1
        print(f"[{stats['pages']}] {r['nom'][:40]:40s} -> {got} image(s)", flush=True)

    if manifest and not a.probe:
        pd.DataFrame(manifest).to_csv(out / "labels.csv", index=False, encoding="utf-8-sig")
    print("\n=== Bilan ===\n" + json.dumps(stats, ensure_ascii=False, indent=2))
    if a.probe:
        rate = stats["pages_with_candidates"] / max(1, stats["pages"])
        print(f"\nPages avec image candidate : {rate:.0%}.",
              "Le site semble contenir des photos exploitables." if stats["images_ok"] else
              "Aucune photo exploitable trouvee : ce site n'est probablement pas une source d'images.")
    else:
        print(f"\nImages enregistrees dans {out / 'images'} ; etiquettes : {out / 'labels.csv'}")


if __name__ == "__main__":
    main()
