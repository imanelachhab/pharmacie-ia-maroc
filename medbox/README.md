# Reconnaissance de boîtes de médicaments (Challenge 2)

Photo de boîte → lecture du texte (OCR français + arabe) → correspondance avec ta base `kb/medications.csv`
→ médicament, dosage, présentation, prix, remboursement, avec **les indices qui justifient la réponse**.

## Pipeline

```
photo ─► orientation (inclinaison + rotation 90/180/270) ─► OCR (plusieurs versions de l'image :
         gris, meilleur canal de couleur, ...) ─► lignes latines + arabes
      ─► nom (recherche floue tolérante aux erreurs d'OCR, nom coupé = fragment, nom le plus long préféré)
      ─► départage des présentations : dosage, taille de boîte, laboratoire, forme
      ─► statut : recognized | uncertain | not_found | no_text  (+ top-k candidats avec score)
```
Le système préfère dire **« je ne suis pas sûr »** plutôt que d'affirmer à tort : un nom lu partiellement
n'est jamais « reconnu », et un médicament absent de la base donne `not_found`.

| Fichier | Rôle |
|---|---|
| `app/ocr.py` | OCR (Tesseract ou EasyOCR), orientation, variantes d'image |
| `app/matcher.py` | correspondance texte lu ↔ base de médicaments |
| `app/main.py` | API FastAPI (`POST /recognize`, `GET /health`) + page web |
| `tools/make_synthetic_boxes.py` | génère des images de boîtes avec flou, faible lumière, reflets, rotation, boîte coupée, bruit |
| `tools/scrape_product_images.py` | télécharge des photos depuis les pages produit de ta base (poli, avec sonde) |
| `evaluate_recognition.py` | top-1 / top-3, précision, faux « reconnu », temps, par type de dégradation |

## Installation (Windows)

1. `pip install -r requirements.txt`
2. **Un moteur OCR** :
   - *Tesseract* : installer https://github.com/UB-Mannheim/tesseract/wiki en cochant les langues **French** et
     **Arabic**, puis `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` si besoin. **Testé.**
   - *EasyOCR* : `pip install easyocr` (modèles téléchargés au premier lancement). **Non testé** dans mon environnement :
     le code est écrit mais je n'ai pas pu l'exécuter ; compare-le à Tesseract avec `evaluate_recognition.py --backend easyocr`.
3. `set KB_DIR=C:\Users\DELL\dataset_challenge\kb` (inutile si c'est le dossier par défaut)
4. `python -m uvicorn app.main:app --reload --port 8001` → http://127.0.0.1:8001 (page de photo) et `/docs`

Exemple : `curl -X POST "http://127.0.0.1:8001/recognize?debug=true" -F "image=@boite.jpg"`

## Constituer le jeu d'images

**Jeu synthétique (immédiat, avec vérité terrain)**
`python tools/make_synthetic_boxes.py --n 40 --fake 10 --out data/synthetic`
→ 40 médicaments × 7 variantes (nette, floue, sombre, reflet, tournée, coupée, bruitée) + 10 boîtes de médicaments
inventés pour mesurer les fausses reconnaissances. **Ces images sont propres et simplifiées : les scores sont un
majorant.** Elles servent à développer et à comparer des réglages, pas à conclure.

**Photos réelles (indispensable pour un résultat crédible)**
Photographie 30 à 50 boîtes (pharmacie, domicile) sous plusieurs angles, éclairages et distances, puis crée
`data/real/labels.csv` avec les colonnes `file,id,nom_base,dosage,degradation` (l'`id` est celui de `medications.csv`)
et évalue : `python evaluate_recognition.py --data data/real`.

Pour éviter de remplir `labels.csv` à la main : **nomme chaque photo avec le médicament** (`doliprane 1000mg boite 8 - flou.jpg` ;
le mot après le tiret = condition : flou, sombre, reflet, incline, coupe) et lance
`python tools/label_real_photos.py --photos data/real` : il retrouve l'`id` dans ta base. Ouvre ensuite `labels.csv`
et corrige les lignes `to_check = True` (nom ambigu ou introuvable).

**Images de sites web** : `tools/scrape_product_images.py` lit les colonnes `url_saydalia` / `url_medicament_ma` de ta
base. Lance d'abord `--probe 10` : il dit si le site contient de vraies photos avant de tout télécharger. Il respecte
`robots.txt`, attend entre les requêtes et **s'arrête si le site bloque** (jamais de contournement).
Vérifie les conditions d'utilisation : les photos de produits sont protégées, usage académique uniquement.

> **Asafar (asafar.net)** : c'est un guide de médicaments (nom, composition, prix, codes). Il utilise une détection
> anti-robot et les données structurées de ses fiches ne comportent pas d'image : ce n'est pas une source de photos
> de boîtes. Je ne l'ai pas scrapé et je ne recommande pas de contourner sa protection.

## Limites connues
- OCR sur de **vraies** photos : reflets, courbure des boîtes, polices décoratives et texte blanc sur fond coloré sont
  nettement plus difficiles que sur les images synthétiques.
- Les noms de marque sont en lettres latines ; le texte arabe n'aide que pour les médicaments courants dont l'écriture
  arabe est listée dans `app/arabic_names.py` (à compléter) et pour la forme (أقراص, شراب...).
- Un nom de marque coupé reste « incertain » par conception. Le code-barres (EAN-13) n'est pas lu : ce serait le
  moyen le plus fiable d'identifier un produit, mais ta base ne contient pas ces codes.
- Temps : quelques secondes par image sur CPU (plus en mode non rapide).
