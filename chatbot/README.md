# Assistant pharmacie Darija / Arabe / Français (Challenge 1)

Chatbot texte + voix qui transforme une demande en **interprétation structurée** (langue, intent, entités)
puis répond à partir de ta base (Saydalia + Medicament.ma + pharmacies). Aucune réponse n'est inventée :
prix, remboursement, composition et pharmacies viennent toujours de la base.

## Installation (Windows)

```bash
cd chatbot
pip install -r requirements.txt
set KB_DIR=C:\Users\DELL\dataset_challenge\kb
uvicorn app.main:app --reload
```
Puis ouvre http://127.0.0.1:8000 (interface de chat) ou http://127.0.0.1:8000/docs (documentation de l'API).

`KB_DIR` = dossier `kb` produit par `build_kb.py`. **Relance `build_kb.py` (version fournie) avant** :
elle nettoie le texte parasite « Nature: ... Ajouter à mon simulateur » dans la composition.

## Architecture

```
texte ─────────────┐
voix → Whisper (STT)┴→ normalisation → entités → intents → langue → réponse (base de connaissances)
```
| Fichier | Rôle |
|---|---|
| `app/nlp.py` | normalisation, entités (médicament avec fautes, dosage, quantité, ville, quartier), intents, langue |
| `app/lexicon.py` | lexiques FR / AR / darija arabe / darija latin, villes, quartiers |
| `app/kb.py` | accès aux médicaments et pharmacies |
| `app/responder.py` | réponses en français, arabe, darija latin |
| `app/stt.py` | Whisper (optionnel) |
| `app/main.py` | API FastAPI |
| `static/index.html` | interface de chat : cartes de résultats, appel / itinéraire, suggestions, mode sombre, micro |
| `evaluate.py` | évaluation (accuracy, F1, erreurs détaillées) |

## Interface

L'écran de chat n'affiche plus l'interprétation (langue, intent, entités) : l'utilisateur voit seulement la réponse,
sous forme de cartes (prix par dosage, badge de remboursement, fiche composition, pharmacies avec boutons
**Appeler** et **Itinéraire**) et de suggestions cliquables. L'interprétation reste disponible dans la réponse JSON
de `/chat` et via `/analyze` (utile pour le rapport et le débogage).

Champs de la réponse `/chat` utilisés par l'interface : `blocks` (contenu structuré), `suggestions`, `reply_language`.

## Endpoints

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/chat` | `{"text": "...", "session_id": "..."}` → langue, intent(s), entités, réponse |
| POST | `/speech` | fichier audio → transcription → même résultat que `/chat` |
| POST | `/analyze` | interprétation seule, sans réponse |
| GET | `/medications/search?q=` | recherche tolérante aux fautes |
| GET | `/medications/{id}` · `/price` · `/reimbursement` | détail, prix, remboursement |
| GET | `/pharmacies/search?city=` · `/pharmacies/duty?city=` | pharmacies, pharmacies de garde |

Exemple :
```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"text\": \"ch7al taman dyal dolipran 1000mg ?\"}"
```

## Voix

```bash
pip install faster-whisper
set WHISPER_MODEL=small        # ou medium / large-v3 (plus précis, plus lent)
```
Sans cette installation, `/speech` répond 501 avec un message explicite. Whisper reconnaît bien l'arabe et le
français, moins bien la darija : mesure le WER avec les phrases de `queries/audio_to_record.csv`.
Cette partie n'a **pas pu être testée** dans l'environnement de développement (pas de modèle téléchargeable).

## Évaluation

```bash
python evaluate.py --kb C:\Users\DELL\dataset_challenge\kb --queries C:\Users\DELL\dataset_challenge\queries ^
       --handwritten tests\realistic_queries.csv --out reports
pytest -q tests
```
Résultats : voir `reports/evaluation_report.json` et `reports/errors_*.csv` (analyse d'erreurs).

## Limites connues

- Les données de **disponibilité en stock** n'existent pas : le bot le dit et renvoie vers la pharmacie.
- Pharmacies de garde : seules 2 entrées de la base indiquent réellement une garde (les autres valeurs sont des horaires).
- Pharmacies : 39 à 40 par ville maximum (limite du scraping) ; des quartiers comme Agdal n'y apparaissent pas.
- Les réponses en darija arabe utilisent un arabe simple ; la darija latin est écrite de façon approximative.
- Le NLP est à règles + recherche floue : il ne comprend pas les formulations très éloignées des lexiques.
