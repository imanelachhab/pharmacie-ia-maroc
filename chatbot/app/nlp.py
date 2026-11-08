"""Moteur NLP (baseline a regles + recherche floue) : langue, entites, intents.

Pipeline : texte -> normalisation -> tokens -> entites (dosage, quantite, ville/quartier, medicament)
           -> intents (lexiques multilingues, tokens d'entites masques) -> langue
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

from . import lexicon as L

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    t = L.ar_fold(text)
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = t.lower().replace("؟", " ").replace("،", " ")
    t = re.sub(r"[^\w\s]", " ", t)  # garde chiffres darija (7, 3, 9) et lettres arabes
    return re.sub(r"\s+", " ", t).strip()


def tokenize(text: str):
    """tokens normalises + empan (debut, fin) du mot d'origine, pour retrouver la graphie exacte"""
    toks, spans = [], []
    for m in re.finditer(r"\S+", text):
        piece = re.sub(r"(?<=\d)[.,](?=\d)", "_", m.group())      # 10.8mg -> 10_8mg
        piece = re.sub(r"(?<=\d)\s?%", "pct", piece)                 # 10% -> 10pct
        for t in normalize(piece).split():
            toks.append(t.replace("_", "."))
            spans.append((m.start(), m.end()))
    return toks, spans


UNITS = {"mg", "g", "ug", "µg", "mcg", "ml", "ui", "iu", "pct"}
DOSE_RE = re.compile(r"(\d+(?:[.,]\d+)?)(mg|g|ug|µg|mcg|ml|ui|iu|pct)")
QTY_UNITS = {"boite", "boites", "comprime", "comprimes", "sachet", "sachets", "flacon", "flacons", "tube", "tubes",
             "علبة", "علب", "boit", "boites"}

AR_PREFIXES = "فبلوك"

DARIJA_AR_MARKERS = {"واش", "شحال", "بشحال", "كاين", "كاينة", "فين", "شنو", "بغيت", "ديال", "علاش", "كيتعوض",
                     "كيتباع", "كيصلح", "نلقى", "كيسوا", "شكون", "مازال", "ولا", "فاش", "شمن", "دابا", "بزاف",
                     "ثمنو", "كيتاخد", "كيدخل", "لقيتو", "عافاك", "د"}
DARIJA_LAT_MARKERS = {"ch7al", "bch7al", "taman", "wach", "kayn", "kayna", "dyal", "3ndkom", "bghit", "chno", "nl9a",
                      "kayt3awd", "t3wid", "7da", "9riba", "lyoum", "lgarde", "mn", "wla", "3afak", "dyali", "chkoun",
                      "kaysalah", "kaytakhd", "kaysswa", "fiha", "fih", "lmadda", "fa3ila", "chmn", "3ndo", "tarkibat",
                      "ma3lumat", "kaytst3mel", "lil", "had", "salam", "sba7"}
FR_FUNC = {"est", "quel", "quelle", "quels", "le", "la", "les", "du", "des", "un", "une", "pour", "avez", "vous", "je",
           "ce", "ca", "que", "qu", "il", "elle", "au", "aux", "dans", "sur", "par", "et", "ou", "y", "en", "de", "mon"}
FR_CONTENT = {"prix", "pharmacie", "composition", "remboursable", "rembourse", "remboursement", "disponible", "stock",
              "garde", "dispo", "principe", "actif", "combien", "coute", "notice", "info", "infos", "tarif", "cherche",
              "trouver", "informations", "pharmacies", "bonjour", "bonsoir", "salut", "cnss", "cnops", "infos"}


@dataclass
class Entity:
    label: str
    text: str
    value: str
    score: float = 100.0
    start: int = 0
    end: int = 0
    idx: tuple = field(default=(), repr=False)

    def public(self):
        return {"label": self.label, "text": self.text, "value": self.value,
                "score": round(self.score, 1), "start": self.start, "end": self.end}


@dataclass
class Analysis:
    text: str
    normalized_text: str
    language: str
    intents: list
    entities: list
    confidence: float
    med_alternatives: list = field(default_factory=list)
    unknown_terms: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def intent(self):
        return " + ".join(self.intents)

    def get(self, label):
        return [e for e in self.entities if e.label == label]


# --------------------------------------------------------------------------
class NLU:
    def __init__(self, med_names, city_aliases=None, kb_cities=()):
        """med_names : iterable de nom_base (deja en minuscules/sans accents)"""
        self._stop_cache = {}
        self._intent_patterns = [L.PRICE, L.REIMB, L.AVAIL_STRONG, L.AVAIL_WEAK, L.COMP, L.INFO_STRONG,
                                 L.DUTY, L.PHARMACY, L.GREETING, L.OOS]
        # ---- villes / quartiers
        self.city_alias = {}
        for a, canon in (city_aliases or {}).items():
            self.city_alias[normalize(a)] = canon
        for canon, (lat, ara) in L.CITY_FORMS.items():
            for a in lat + ara + [canon]:
                self.city_alias[normalize(a)] = canon
        for c in kb_cities:
            self.city_alias.setdefault(normalize(c), c)
        self.quartier_alias = {}
        for city, qs in L.QUARTIERS.items():
            for lat, ara in qs:
                for a in (lat, ara):
                    if a:
                        self.quartier_alias[normalize(a)] = (lat, city)
        self._city_keys = [k for k in self.city_alias if len(k) >= 6 and " " not in k]
        # ---- medicaments
        self.med_vocab = {}
        for n in med_names:
            k = normalize(n)
            if len(k) < 3 or self._is_stop(k.split()[0]) or k in self.city_alias or k in self.quartier_alias:
                continue
            self.med_vocab[k] = k
        self.ar_alias = {}
        for base, lst in L.ARABIC_NAMES.items():
            if base in self.med_vocab:
                for a in lst:
                    self.ar_alias[normalize(a)] = base
        self.med_keys = list(self.med_vocab)

    # ----------------------------------------------------------------------
    def _is_stop(self, tok: str) -> bool:
        r = self._stop_cache.get(tok)
        if r is None:
            r = tok in L.STOP or tok in L.CANON or any(p.search(tok) for p in self._intent_patterns) or tok.isdigit()
            self._stop_cache[tok] = r
        return r

    # ---------------------------------------------------------------- dosage
    def _find_dosage_qty(self, toks, spans, text, used):
        ents = []
        i = 0
        while i < len(toks):
            t = toks[i]
            m = DOSE_RE.fullmatch(t)
            if m:
                ents.append(self._mk("DOSAGE", (i,), toks, spans, text, self._dose_value(m.group(1), m.group(2))))
                used.add(i); i += 1; continue
            if re.fullmatch(r"\d+(?:[.,]\d+)?", t) and i + 1 < len(toks) and toks[i + 1] in UNITS:
                ents.append(self._mk("DOSAGE", (i, i + 1), toks, spans, text, self._dose_value(t, toks[i + 1])))
                used.update((i, i + 1)); i += 2; continue
            if re.fullmatch(r"\d{1,2}", t) and i + 1 < len(toks) and toks[i + 1] in QTY_UNITS:
                ents.append(self._mk("QUANTITY", (i, i + 1), toks, spans, text, t))
                used.update((i, i + 1)); i += 2; continue
            i += 1
        return ents

    @staticmethod
    def _dose_value(num, unit):
        return num.replace(",", ".") + unit.upper().replace("IU", "UI").replace("PCT", "%")

    def _mk(self, label, idxs, toks, spans, text, value, score=100.0):
        s, e = spans[idxs[0]][0], spans[idxs[-1]][1]
        surface = text[s:e].strip(" ?!.,;:؟،")
        return Entity(label, surface, value, score, s, s + len(surface), tuple(idxs))

    # ------------------------------------------------------------ city / quartier
    def _city_lookup(self, cand: str):
        """retourne (type, valeur, ville_canonique, score) ou None ; gere les prefixes arabes collés (فالرباط)"""
        variants = [cand]
        if cand and cand[0] in AR_PREFIXES and len(cand) > 3:
            variants.append(cand[1:])
        for v in variants:
            if v in self.quartier_alias:
                lat, city = self.quartier_alias[v]
                return ("QUARTIER", lat, city, 100.0)
            if v in self.city_alias:
                return ("CITY", self.city_alias[v], self.city_alias[v], 100.0)
        if " " not in cand and len(cand) >= 6 and not self._is_stop(cand):
            r = process.extractOne(cand, self._city_keys, scorer=fuzz.ratio, score_cutoff=88)
            if r and r[0][0] == cand[0]:
                return ("CITY", self.city_alias[r[0]], self.city_alias[r[0]], r[1])
        return None

    def _find_places(self, toks, spans, text, used):
        ents, found_city, quartier_city = [], None, None
        for n in (3, 2, 1):
            for i in range(len(toks) - n + 1):
                idxs = tuple(range(i, i + n))
                if any(j in used for j in idxs):
                    continue
                cand = " ".join(toks[i:i + n])
                if n == 1 and len(cand) < 3:
                    continue
                hit = self._city_lookup(cand)
                if hit:
                    kind, val, city, sc = hit
                    if kind == "CITY" and (found_city is not None):
                        continue
                    ents.append(self._mk(kind, idxs, toks, spans, text, val, sc))
                    used.update(idxs)
                    if kind == "CITY":
                        found_city = city
                    else:
                        quartier_city = city
        if found_city is None and quartier_city is not None:
            # ville deduite du quartier
            ents.append(Entity("CITY", "", quartier_city, 80.0, 0, 0, ()))
        return ents

    # ------------------------------------------------------------ medicament
    @staticmethod
    def _ar_variants(cand: str):
        """retire l'article/prefixe arabe colle : فالدافلون -> الدافلون -> دافلون"""
        out = [cand]
        c = cand
        if c and c[0] in AR_PREFIXES and len(c) > 4:
            c = c[1:]; out.append(c)
        if c.startswith("ال") and len(c) > 4:
            out.append(c[2:])
        return out

    def _match_med(self, cand: str):
        if re.search(r"[\u0600-\u06FF]", cand) and " " not in cand:
            for v in self._ar_variants(cand):
                if v in self.ar_alias:
                    return [(self.ar_alias[v], 100.0)]
        if cand in self.med_vocab:
            return [(cand, 100.0)]
        if cand in self.ar_alias:
            return [(self.ar_alias[cand], 100.0)]
        if len(cand) < 5:
            return []
        # une seule faute tolere des 5 lettres ; deux fautes seulement pour les noms longs
        cutoff = 80 if len(cand) <= 5 else 83 if len(cand) <= 7 else 80
        is_ar = bool(re.search(r"[\u0600-\u06FF]", cand))
        keys = list(self.ar_alias) if is_ar else self.med_keys
        res = process.extract(cand, keys, scorer=fuzz.ratio, score_cutoff=cutoff, limit=4)
        out = []
        for k, sc, _ in res:
            # la 1re lettre doit correspondre, sauf pour les noms longs tres proches
            if k[0] != cand[0] and not (len(cand) >= 8 and sc >= 88):
                continue
            out.append((self.ar_alias[k] if is_ar else k, float(sc)))
        return out

    def _find_med(self, toks, spans, text, used):
        best, alts = None, {}
        n_tok = len(toks)
        for n in (3, 2, 1):
            for i in range(n_tok - n + 1):
                idxs = tuple(range(i, i + n))
                if any(j in used for j in idxs):
                    continue
                if self._is_stop(toks[i]) or self._is_stop(toks[idxs[-1]]):
                    continue
                cand = " ".join(toks[i:i + n])
                if n == 1 and len(cand) < 3:
                    continue
                matches = self._match_med(cand)
                if n > 1 and not matches:
                    matches = self._match_med(cand.replace(" ", ""))
                for base, sc in matches:
                    key = (sc, n)
                    if best is None or key > best[0]:
                        best = (key, base, idxs)
                    alts[base] = max(alts.get(base, 0), sc)
        if not best:
            return None, []
        (sc, _), base, idxs = best
        ent = self._mk("MEDICATION", idxs, toks, spans, text, base.capitalize(), sc)
        others = [b for b, s2 in sorted(alts.items(), key=lambda x: -x[1]) if b != base and s2 >= sc - 8][:3]
        return ent, others

    # --------------------------------------------------------------- intents
    def _detect_intents(self, ctoks, ents):
        masked = list(ctoks)
        for e in ents:
            for j in e.idx:
                masked[j] = "_"
        t = " ".join(masked)
        t_av = t
        for ph in L.ABSORB_FOR_AVAIL:
            t_av = re.sub(ph, " ", t_av)
        has_med = any(e.label == "MEDICATION" for e in ents)
        has_loc = any(e.label in ("CITY", "QUARTIER") for e in ents)
        phar = bool(L.PHARMACY.search(t))
        duty = bool(L.DUTY.search(t))
        where = bool(L.WHERE.search(t))

        # hors perimetre
        if L.OOS.search(t) and not has_med and not phar and not duty:
            return ["OUT_OF_SCOPE"], 0.8

        out, conf = [], 0.92
        if L.PRICE.search(t):
            out.append("MEDICATION_PRICE")
        if L.REIMB.search(t):
            out.append("MEDICATION_REIMBURSEMENT")
        if L.AVAIL_STRONG.search(t_av) or (has_med and L.AVAIL_WEAK.search(t_av)):
            out.append("MEDICATION_AVAILABILITY")
        if L.COMP.search(t):
            out.append("MEDICATION_COMPOSITION")
        if L.INFO_STRONG.search(t):
            out.append("MEDICATION_INFO")
        elif L.INFO_WEAK.search(t) and not out:
            out.append("MEDICATION_INFO"); conf = 0.7

        if duty and (not has_med or has_loc):
            out.append("PHARMACY_DUTY")
        elif phar or where:
            if not has_med:
                if phar:
                    out.append("PHARMACY_SEARCH")
            elif has_loc:
                if "MEDICATION_AVAILABILITY" not in out:
                    out.append("MEDICATION_AVAILABILITY")
                out.append("PHARMACY_SEARCH")

        if where and has_med and has_loc and "PHARMACY_SEARCH" not in out and not (set(out) - {"MEDICATION_AVAILABILITY"}):
            out = ["MEDICATION_AVAILABILITY", "PHARMACY_SEARCH"]

        if not out:
            if has_med:
                out, conf = ["MEDICATION_INFO"], 0.5
            elif has_loc and len(ctoks) <= 3:
                out, conf = ["PHARMACY_SEARCH"], 0.5
            elif L.GREETING.search(t):
                out, conf = ["GREETING"], 0.9
            else:
                out, conf = ["OUT_OF_SCOPE"], 0.4
        order = ["MEDICATION_PRICE", "MEDICATION_REIMBURSEMENT", "MEDICATION_AVAILABILITY", "MEDICATION_COMPOSITION",
                 "MEDICATION_INFO", "PHARMACY_SEARCH", "PHARMACY_DUTY", "GREETING", "OUT_OF_SCOPE"]
        out = sorted(set(out), key=order.index)
        return out, conf

    # -------------------------------------------------------------- langue
    def _language(self, text, toks, ents):
        claimed = {j for e in ents for j in e.idx}
        free = [t for i, t in enumerate(toks) if i not in claimed]
        n_ar = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        n_lat = sum(1 for c in text if c.isascii() and c.isalpha())
        free_lat = [t for t in free if re.fullmatch(r"[a-z0-9]+", t) and not t.isdigit()]
        d_lat = sum(1 for t in free_lat if t in DARIJA_LAT_MARKERS or re.fullmatch(r"[a-z]*[2-9][a-z0-9]*[a-z][a-z0-9]*|[a-z]+[2-9]", t))
        fr_f = sum(1 for t in free_lat if t in FR_FUNC)
        fr_c = sum(1 for t in free_lat if t in FR_CONTENT)
        if n_ar > 0:
            has_d = any(t in DARIJA_AR_MARKERS for t in free)
            lat_other = [t for t in free_lat if t in FR_CONTENT or t in FR_FUNC or t in DARIJA_LAT_MARKERS]
            if lat_other:
                return "mixed"
            return "darija_ar" if has_d else "ar"
        if n_lat == 0:
            return "fr"
        if d_lat == 0:
            return "fr"
        return "mixed" if (fr_f > 0 or fr_c >= 1) else "darija_latin"

    # --------------------------------------------------------------- API
    def analyze(self, text: str) -> Analysis:
        toks, spans = tokenize(text)
        ctoks = [L.CANON.get(t, t) for t in toks]
        used = set()
        ents = self._find_dosage_qty(toks, spans, text, used)
        ents += self._find_places(toks, spans, text, used)
        med, alts = self._find_med(toks, spans, text, used)
        if med:
            ents.append(med)
        ents.sort(key=lambda e: (e.start, e.end))
        intents, conf = self._detect_intents(ctoks, ents)
        if med and med.score < 100:
            conf = min(conf, 0.5 + med.score / 200)
        lang = self._language(text, toks, ents)
        claimed = set(used) | set(med.idx if med else ())
        unknown = [t for i, t in enumerate(toks) if i not in claimed and len(t) >= 5 and not self._is_stop(t)
                   and not re.search(r"\d", t)][:2]
        return Analysis(text=text, normalized_text=" ".join(toks), language=lang, intents=intents, entities=ents,
                        confidence=round(conf, 2), med_alternatives=alts,
                        unknown_terms=unknown)
