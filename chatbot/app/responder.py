"""Generation des reponses a partir des donnees de la base (jamais inventees).

Langues de reponse : fr, ar (arabe simple, utilise aussi pour la darija ecrite en arabe),
dl (darija en lettres latines, utilisee aussi pour le code-mixing latin).
"""
import re
from typing import Optional

from .nlp import Analysis
from .kb import KB, dose_to_key

S = {
    "fr": {
        "greet": "Bonjour ! Je peux vous renseigner sur le prix, le remboursement, la composition ou la disponibilité d'un médicament, et vous aider à trouver une pharmacie. Que souhaitez-vous savoir ?",
        "ask_med": "De quel médicament parlez-vous ? Indiquez son nom (et son dosage si possible).",
        "ask_city": "Dans quelle ville cherchez-vous ?",
        "not_found": "Je n'ai pas trouvé de médicament correspondant dans ma base. Vérifiez l'orthographe ou envoyez le nom inscrit sur la boîte.",
        "understood": "J'ai compris « {name} » (vous avez écrit « {raw} »).",
        "alts": "Autres possibilités : {alts}.",
        "dose_missing": "Je n'ai pas ce dosage ; dosages connus : {doses}.",
        "price_h": "Prix public indicatif (PPV) de {name} :",
        "price_none": "Le prix de {name} n'est pas renseigné dans ma base.",
        "price_diff": "Les sources consultées affichent de légères différences de prix pour certaines présentations.",
        "reimb_yes": "{name} est indiqué comme remboursable.",
        "reimb_no": "{name} n'est pas indiqué comme remboursable dans ma base.",
        "reimb_mix": "Le remboursement de {name} dépend de la présentation :",
        "reimb_unk": "L'information de remboursement n'est pas renseignée pour {name}. Voyez avec votre pharmacien ou votre organisme (CNSS, CNOPS...).",
        "comp": "Composition de {name} : {comp}.",
        "comp_none": "La composition de {name} n'est pas renseignée dans ma base.",
        "cls": "Classe thérapeutique : {v}.", "atc": "Code ATC : {v}.", "maker": "Fabricant : {v}.",
        "forms": "Formes : {v}.", "status": "Statut : {v}.", "name_line": "{name}",
        "avail": "Je n'ai pas accès au stock des pharmacies en temps réel. {name} figure dans ma base comme {st}. Appelez votre pharmacie pour confirmer sa disponibilité.",
        "st_ok": "commercialisé", "st_unk": "médicament référencé",
        "ph_h": "Pharmacies à {city} :", "ph_h_q": "Pharmacies à {q} ({city}) :",
        "ph_q_none": "Je n'ai pas trouvé de pharmacie dans ce quartier ; voici d'autres pharmacies de {city} :",
        "ph_none": "Je n'ai aucune pharmacie enregistrée pour cette ville.",
        "ph_line": "• {nom} – {adr}{tel}", "tel": " – Tél : {t}",
        "duty_h": "Pharmacies de garde à {city} (d'après mes données) :",
        "duty_none": "Je n'ai pas d'information fiable sur les pharmacies de garde à {city} : elles changent chaque semaine. La pharmacie la plus proche affiche généralement la garde sur sa vitrine. Voici quelques pharmacies de la ville à contacter :",
        "oos": "Je suis un assistant pharmacie : je peux vous aider pour le prix, le remboursement, la composition ou la disponibilité d'un médicament, et pour trouver une pharmacie.",
        "disc": "Ces informations sont indicatives ; demandez conseil à votre pharmacien.",
        "lbl_comp": "Composition", "lbl_cls": "Classe thérapeutique", "lbl_atc": "Code ATC", "lbl_maker": "Fabricant",
        "lbl_forms": "Formes", "lbl_ppv": "Prix (PPV)",
        "not_found_term": "Je n'ai pas trouvé « {term} » dans ma base de médicaments. Vérifiez l'orthographe ou envoyez le nom inscrit sur la boîte.",
        "s_price": "Combien ça coûte ?", "s_reimb": "Est-il remboursé ?", "s_comp": "Quelle est sa composition ?",
        "s_pharm": "Où trouver une pharmacie ?", "s_duty": "Pharmacie de garde ?", "s_price_dose": "Prix {name} {dose}",
        "starters": ["Prix du Doliprane", "L'Augmentin est-il remboursé ?", "Pharmacie de garde à Rabat"],
        "b_yes": "Remboursable", "b_no": "Non remboursable", "b_unk": "Information indisponible", "b_mix": "Selon la boîte",
        "cur": "MAD", "more": "… et {n} autre(s) option(s).",
    },
    "ar": {
        "greet": "مرحبا! يمكنني مساعدتك في معرفة ثمن دواء أو تعويضه أو مكوناته أو توفره، وفي البحث عن صيدلية. كيف أساعدك؟",
        "ask_med": "ما هو اسم الدواء؟ المرجو كتابة الاسم (والجرعة إن أمكن).",
        "ask_city": "في أي مدينة تبحث؟",
        "not_found": "لم أجد دواء مطابقا في قاعدة بياناتي. تأكد من الكتابة أو أرسل الاسم المكتوب على العلبة.",
        "understood": "فهمت أنك تقصد « {name} » (كتبت « {raw} »).",
        "alts": "احتمالات أخرى: {alts}.",
        "dose_missing": "هذه الجرعة غير متوفرة عندي؛ الجرعات المعروفة: {doses}.",
        "price_h": "الثمن العمومي الإرشادي (PPV) لـ {name}:",
        "price_none": "ثمن {name} غير مسجل في قاعدة بياناتي.",
        "price_diff": "المصادر التي اعتمدتها تعطي فروقا بسيطة في الثمن لبعض العلب.",
        "reimb_yes": "{name} مسجل كدواء قابل للتعويض.",
        "reimb_no": "{name} غير مسجل كدواء قابل للتعويض في قاعدة بياناتي.",
        "reimb_mix": "تعويض {name} يختلف حسب العلبة:",
        "reimb_unk": "معلومة التعويض غير متوفرة لـ {name}. اسأل الصيدلي أو هيئة التأمين (CNSS، CNOPS...).",
        "comp": "تركيبة {name}: {comp}.",
        "comp_none": "تركيبة {name} غير مسجلة في قاعدة بياناتي.",
        "cls": "الفئة العلاجية: {v}.", "atc": "رمز ATC: {v}.", "maker": "المصنع: {v}.",
        "forms": "الأشكال: {v}.", "status": "الحالة: {v}.", "name_line": "{name}",
        "avail": "ليست لدي معلومات عن مخزون الصيدليات في الوقت الحقيقي. {name} مسجل في قاعدة بياناتي كدواء {st}. اتصل بالصيدلية للتأكد من توفره.",
        "st_ok": "مسوّق", "st_unk": "معروف",
        "ph_h": "صيدليات في {city}:", "ph_h_q": "صيدليات في {q} ({city}):",
        "ph_q_none": "لم أجد صيدلية في هذا الحي؛ إليك صيدليات أخرى في {city}:",
        "ph_none": "لا توجد صيدليات مسجلة لهذه المدينة.",
        "ph_line": "• {nom} – {adr}{tel}", "tel": " – الهاتف: {t}",
        "duty_h": "صيدليات الحراسة في {city} (حسب بياناتي):",
        "duty_none": "ليست لدي معلومات موثوقة عن صيدليات الحراسة في {city} لأنها تتغير كل أسبوع. أقرب صيدلية تعلق عادة لائحة الحراسة على واجهتها. إليك بعض صيدليات المدينة للاتصال بها:",
        "oos": "أنا مساعد خاص بالصيدلية: أساعدك في ثمن الدواء وتعويضه وتركيبته وتوفره، وفي البحث عن صيدلية.",
        "disc": "هذه المعلومات إرشادية، استشر الصيدلي.",
        "lbl_comp": "التركيبة", "lbl_cls": "الفئة العلاجية", "lbl_atc": "رمز ATC", "lbl_maker": "المصنع",
        "lbl_forms": "الأشكال", "lbl_ppv": "الثمن (PPV)",
        "not_found_term": "لم أجد « {term} » في قاعدة بيانات الأدوية. تأكد من الكتابة أو أرسل الاسم المكتوب على العلبة.",
        "s_price": "كم ثمنه ؟", "s_reimb": "هل هو مشمول بالتعويض ؟", "s_comp": "ما هي مكوناته ؟",
        "s_pharm": "أين أجد صيدلية ؟", "s_duty": "صيدلية الحراسة ؟", "s_price_dose": "ثمن {name} {dose}",
        "starters": ["ثمن دوليبران", "هل الأوغمانتين مشمول بالتعويض ؟", "صيدلية الحراسة في الرباط"],
        "b_yes": "قابل للتعويض", "b_no": "غير قابل للتعويض", "b_unk": "المعلومة غير متوفرة", "b_mix": "حسب العلبة",
        "cur": "درهم", "more": "… و{n} خيارات أخرى.",
    },
    "dl": {
        "greet": "Salam! Imken lia n3awnek f taman, t3wid, tarkib wla tawafor dyal chi dwa, wla n9eleb 3la pharmacie. Chno bghiti t3ref?",
        "ask_med": "Chno smiyt dwa? Kteb smito (w dosage ila 3ndek).",
        "ask_city": "F achmen mdina kat9eleb?",
        "not_found": "Ma lqit hta dwa b had smiya f la base dyali. Chouf l'orthographe wla sift lia smiya li mktouba 3la la boîte.",
        "understood": "Fhemt blli bghiti « {name} » (ktebti « {raw} »).",
        "alts": "Ihtimalat khrin: {alts}.",
        "dose_missing": "Had dosage ma 3ndich; li 3ndi: {doses}.",
        "price_h": "Taman l'public (PPV) dyal {name}:",
        "price_none": "Taman dyal {name} ma mseggelch f la base dyali.",
        "price_diff": "L'sources kaybayno ikhtilaf sghir f taman dyal chi boîtat.",
        "reimb_yes": "{name} mseggel bli kayt3awd.",
        "reimb_no": "{name} ma mseggelch bli kayt3awd f la base dyali.",
        "reimb_mix": "T3wid dyal {name} kaykhtalef 7sab la boîte:",
        "reimb_unk": "L'information dyal t3wid ma 3ndich 3la {name}. Sewwel pharmacien dyalk wla CNSS/CNOPS.",
        "comp": "Tarkib dyal {name}: {comp}.",
        "comp_none": "Tarkib dyal {name} ma mseggelch f la base dyali.",
        "cls": "Classe thérapeutique: {v}.", "atc": "Code ATC: {v}.", "maker": "Fabricant: {v}.",
        "forms": "Les formes: {v}.", "status": "Statut: {v}.", "name_line": "{name}",
        "avail": "Ma 3ndich l'stock dyal pharmacies f nafs lwa9t. {name} kayn f la base dyali ({st}). Tssel b la pharmacie dyalk bach t2eked.",
        "st_ok": "commercialisé", "st_unk": "mseggel",
        "ph_h": "Pharmacies f {city}:", "ph_h_q": "Pharmacies f {q} ({city}):",
        "ph_q_none": "Ma lqit hta pharmacie f had l7ay; hna chi pharmacies khrin f {city}:",
        "ph_none": "Ma 3ndi hta pharmacie mseggla l had lmdina.",
        "ph_line": "• {nom} – {adr}{tel}", "tel": " – Tél: {t}",
        "duty_h": "Pharmacies de garde f {city} (7sab la base dyali):",
        "duty_none": "Ma 3ndich information mowtouqa 3la pharmacies de garde f {city}, 7it kaybdlo kol jem3a. Aqrab pharmacie ghaliban katsele9 l'garde 3la lbab. Hna chi pharmacies f lmdina t9dr tssel bihom:",
        "oos": "Ana assistant dyal pharmacie: n9dr n3awnek f taman, t3wid, tarkib wla tawafor dyal dwa, w f l9ya3 3la pharmacie.",
        "disc": "Had l'ma3lumat ghir lil istirchad, sewwel pharmacien dyalk.",
        "lbl_comp": "Tarkib", "lbl_cls": "Classe", "lbl_atc": "Code ATC", "lbl_maker": "Fabricant",
        "lbl_forms": "Les formes", "lbl_ppv": "Taman (PPV)",
        "not_found_term": "Ma lqit « {term} » f la base dyal les médicaments. Chouf l'orthographe wla sift lia smiya li mktouba 3la la boîte.",
        "s_price": "ch7al taman ?", "s_reimb": "wach kayt3awd ?", "s_comp": "chno fih ?",
        "s_pharm": "fin nl9a pharmacie ?", "s_duty": "pharmacie de garde ?", "s_price_dose": "taman {name} {dose}",
        "starters": ["ch7al taman doliprane ?", "wach augmentin kayt3awd ?", "fin kayna pharmacie de garde f Rabat ?"],
        "b_yes": "Kayt3awd", "b_no": "Ma kayt3awdch", "b_unk": "L'info ma kaynach", "b_mix": "7sab la boîte",
        "cur": "dh", "more": "… w {n} khiyarat khrin.",
    },
}


def resp_lang(language: str, text: str) -> str:
    if language == "fr":
        return "fr"
    if language in ("ar", "darija_ar"):
        return "ar"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    return "dl"


def _num(v, lang):
    s = f"{v:g}"
    return s.replace(".", ",") if lang == "fr" else s



UNIT_FIX = re.compile(r"(?<=\d)\s?(mg|g|ml|ui|µg|mcg)\b", re.I)


def pretty(name: str) -> str:
    """'DOLIPRANE 1 G' -> 'Doliprane 1 G'"""
    if not name:
        return ""
    if name.isupper():
        name = name.title()
    return UNIT_FIX.sub(lambda m: " " + m.group(1).upper() if m.group(0)[0] == " " else m.group(1).upper(), name)


def blocks_to_text(blocks) -> str:
    out = []
    for b in blocks:
        t = b["type"]
        if t in ("text", "notice"):
            out.append(b["text"])
        elif t == "prices":
            lines = [b["title"]] + [f"• {i['name']}" + (f" ({i['sub']})" if i.get("sub") else "") +
                                    f" : {_num(i['price'], b['lang'])}" + (f" – {_num(i['price_max'], b['lang'])}" if i.get("price_max") else "") + f" {i['cur']}" for i in b["items"]]
            if b.get("more"):
                lines.append(b["more"])
            if b.get("note"):
                lines.append(b["note"])
            out.append("\n".join(lines))
        elif t == "status":
            lines = [b["text"]] + [f"• {i['name']}: {'✓' if i['value'] else '✗'}" for i in b.get("items", [])]
            out.append("\n".join(lines))
        elif t == "facts":
            out.append("\n".join([b["title"]] + [f"{k} : {v}" for k, v in b["rows"]]))
        elif t == "pharmacies":
            lines = [b["title"]]
            for p in b["items"]:
                tel = f" – {p['telephone']}" if p.get("telephone") else ""
                lines.append(f"• {p['nom']} – {p.get('adresse') or ''}{tel}")
            out.append("\n".join(lines))
    return "\n\n".join(out)


class Responder:
    def __init__(self, kb: KB):
        self.kb = kb

    def _t(self, lang, key, **kw):
        return S[lang][key].format(**kw)

    # ------------------------------------------------------------------ intents -> blocks
    SIMPLE_DOSE = re.compile(r"\d+(?:[.,]\d+)?\s?(?:mg|g|µg|ml|ui|%)", re.I)

    def _price(self, lang, base, rows, dosage):
        name = base.title()
        seen, items = set(), []
        for r in rows:
            if r.get("prix") is None:
                continue
            k = (r["nom"], r.get("forme"), r.get("presentation"), r["prix"])
            if k in seen:
                continue
            seen.add(k)
            items.append(r)
        if not items:
            return [{"type": "text", "text": self._t(lang, "price_none", name=name)}], {}, []
        cur = S[lang]["cur"]
        dkey = lambda d: dose_to_key(d) or (1e12, "")
        sub = lambda r: " · ".join(x for x in (r.get("forme"), r.get("presentation")) if x)
        chips = []
        if len(items) > 6 and not dosage:
            # trop de presentations : un resume par dosage (fourchette de prix), dosages courants en premier
            groups = {}
            for r in items:
                groups.setdefault((r.get("dosage") or "").strip(), []).append(r)
            order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), dkey(kv[0])))
            top = sorted(order[:6], key=lambda kv: dkey(kv[0]))
            out = []
            for d, rs in top:
                lo, hi = min(r["prix"] for r in rs), max(r["prix"] for r in rs)
                forms = sorted({r["forme"] for r in rs if r.get("forme")})[:2]
                out.append({"name": pretty(f"{name} {d}".strip()), "sub": " · ".join(forms), "price": lo,
                            "price_max": hi if hi > lo else None, "cur": cur})
            more = self._t(lang, "more", n=len(order) - len(top)) if len(order) > len(top) else None
            chips = [self._t(lang, "s_price_dose", name=name, dose=d.lower()) for d, _ in order
                     if self.SIMPLE_DOSE.fullmatch(d)][:3]
            shown = [r for d, rs in top for r in rs]
        else:
            items.sort(key=lambda r: (dkey(r.get("dosage")), r["prix"]))
            shown = items[:6]
            out = [{"name": pretty(r["nom"]), "sub": sub(r), "price": r["prix"], "price_max": None, "cur": cur}
                   for r in shown]
            more = self._t(lang, "more", n=len(items) - 6) if len(items) > 6 else None
        blk = {"type": "prices", "lang": lang, "title": self._t(lang, "price_h", name=name), "cur": cur,
               "items": out, "more": more,
               "note": self._t(lang, "price_diff") if any(r.get("conflit_prix") for r in shown) else None}
        data = {"prices": [{"id": r["id"], "nom": r["nom"], "presentation": r.get("presentation"),
                            "prix": r["prix"], "source": r.get("prix_source")} for r in shown[:20]]}
        return [blk], data, chips

    def _reimb(self, lang, base, rows):
        name = base.title()
        vals = {r["remboursement"] for r in rows if r.get("remboursement") is not None}
        if not vals:
            return [{"type": "status", "value": None, "badge": S[lang]["b_unk"],
                     "text": self._t(lang, "reimb_unk", name=name)}], {"reimbursement": None}
        if vals == {True}:
            return [{"type": "status", "value": True, "badge": S[lang]["b_yes"],
                     "text": self._t(lang, "reimb_yes", name=name)}], {"reimbursement": True}
        if vals == {False}:
            return [{"type": "status", "value": False, "badge": S[lang]["b_no"],
                     "text": self._t(lang, "reimb_no", name=name)}], {"reimbursement": False}
        items = [{"name": pretty(f"{r['nom']} {r.get('presentation') or ''}".strip()), "value": r["remboursement"]}
                 for r in rows[:5] if r.get("remboursement") is not None]
        return [{"type": "status", "value": "mixed", "badge": S[lang]["b_mix"],
                 "text": self._t(lang, "reimb_mix", name=name), "items": items}], {"reimbursement": "mixed"}

    def _comp(self, lang, base, rows):
        name = base.title()
        r = next((x for x in rows if x.get("composition")), rows[0])
        if not r.get("composition"):
            return [{"type": "text", "text": self._t(lang, "comp_none", name=name)}], {}
        fr = [(S[lang]["lbl_comp"], r["composition"])]
        for key, col in (("lbl_cls", "classe_therapeutique"), ("lbl_atc", "code_atc")):
            if r.get(col):
                fr.append((S[lang][key], r[col]))
        return [{"type": "facts", "title": name, "rows": fr}], {"composition": r["composition"],
                                                                "atc": r.get("code_atc"),
                                                                "classe": r.get("classe_therapeutique")}

    def _info(self, lang, base, rows):
        name = base.title()
        r = next((x for x in rows if x.get("composition")), rows[0])
        fr = []
        forms = sorted({x["forme"] for x in rows if x.get("forme")})
        if forms:
            fr.append((S[lang]["lbl_forms"], ", ".join(forms[:4])))
        for key, col in (("lbl_comp", "composition"), ("lbl_cls", "classe_therapeutique"), ("lbl_atc", "code_atc"),
                         ("lbl_maker", "fabricant")):
            if r.get(col):
                fr.append((S[lang][key], r[col]))
        prices = [x["prix"] for x in rows if x.get("prix") is not None]
        if prices:
            lo, hi = min(prices), max(prices)
            rng = _num(lo, lang) if lo == hi else f"{_num(lo, lang)} – {_num(hi, lang)}"
            fr.append((S[lang]["lbl_ppv"], f"{rng} {S[lang]['cur']}"))
        return [{"type": "facts", "title": name, "rows": fr}], {"id": r["id"], "forms": forms}

    def _avail(self, lang, base, rows):
        ok = any("commercialis" in str(x.get("statut") or "").lower() for x in rows)
        st = S[lang]["st_ok"] if ok else S[lang]["st_unk"]
        return [{"type": "text", "text": self._t(lang, "avail", name=base.title(), st=st)}]

    def _pharm_block(self, lang, title, items, duty=False):
        return {"type": "pharmacies", "title": title, "duty": duty,
                "items": [{"nom": pretty(p["nom"]), "adresse": p.get("adresse"), "telephone": p.get("telephone"),
                           "ville": p.get("ville")} for p in items]}

    def _pharmacy_search(self, lang, city, quartier):
        if not city:
            return [{"type": "text", "text": self._t(lang, "ask_city")}], {}, True
        items, total, note = self.kb.pharmacies(city=city, quartier=quartier, limit=5)
        if not items:
            return [{"type": "text", "text": self._t(lang, "ph_none")}], {}, False
        if quartier and note == "quartier_not_found":
            head = self._t(lang, "ph_q_none", city=city)
        elif quartier:
            head = self._t(lang, "ph_h_q", q=quartier, city=city)
        else:
            head = self._t(lang, "ph_h", city=city)
        return [self._pharm_block(lang, head, items)], {"pharmacies": items, "total": total}, False

    def _pharmacy_duty(self, lang, city):
        if not city:
            return [{"type": "text", "text": self._t(lang, "ask_city")}], {}, True
        items, total, _ = self.kb.pharmacies(city=city, duty=True, limit=5)
        if items:
            return [self._pharm_block(lang, self._t(lang, "duty_h", city=city), items, True)], \
                {"pharmacies": items, "duty": True}, False
        items, _, _ = self.kb.pharmacies(city=city, limit=3)
        return [self._pharm_block(lang, self._t(lang, "duty_none", city=city), items)], \
            {"pharmacies": items, "duty": False}, False

    # ------------------------------------------------------------------ suggestions
    def _suggest(self, lang, intents, dose_chips):
        s = S[lang]
        out = list(dose_chips)
        if "MEDICATION_PRICE" in intents:
            order = ["s_reimb", "s_comp", "s_pharm"]
        elif "MEDICATION_REIMBURSEMENT" in intents:
            order = ["s_price", "s_comp", "s_pharm"]
        elif "MEDICATION_COMPOSITION" in intents or "MEDICATION_INFO" in intents:
            order = ["s_price", "s_reimb", "s_pharm"]
        elif "MEDICATION_AVAILABILITY" in intents:
            order = ["s_price", "s_pharm"]
        elif "PHARMACY_SEARCH" in intents:
            order = ["s_duty"]
        else:
            order = []
        out += [s[k] for k in order]
        return out[:4]

    # ------------------------------------------------------------------ entree
    def _wrap(self, lang, blocks, data=None, clarify=False, ctx_used=False, suggestions=None):
        return {"text": blocks_to_text(blocks), "blocks": blocks, "data": data or {}, "needs_clarification": clarify,
                "context_used": ctx_used, "suggestions": suggestions or [], "reply_language": lang}

    def respond(self, a: Analysis, ctx: Optional[dict] = None) -> dict:
        ctx = ctx if ctx is not None else {}
        lang = resp_lang(a.language, a.text)
        intents = a.intents
        if intents == ["GREETING"]:
            return self._wrap(lang, [{"type": "text", "text": self._t(lang, "greet")}], suggestions=S[lang]["starters"])
        if intents == ["OUT_OF_SCOPE"]:
            return self._wrap(lang, [{"type": "text", "text": self._t(lang, "oos")}], suggestions=S[lang]["starters"])

        med_e = next(iter(a.get("MEDICATION")), None)
        dose_e = next(iter(a.get("DOSAGE")), None)
        city_e = next(iter(a.get("CITY")), None)
        quart_e = next(iter(a.get("QUARTIER")), None)
        city = city_e.value if city_e else None
        quartier = quart_e.value if quart_e else None
        used_ctx = False

        needs_med = any(i.startswith("MEDICATION") for i in intents)
        base = med_e.value.lower() if med_e else None
        dosage = dose_e.value if dose_e else None
        if needs_med and not base:
            term = next((t for t in a.unknown_terms if len(t) >= 5), None)
            if term:  # un nom inconnu est cite : ne jamais repondre sur le medicament precedent
                return self._wrap(lang, [{"type": "text", "text": self._t(lang, "not_found_term", term=term)}],
                                  clarify=True)
            if ctx.get("med"):
                base, dosage, used_ctx = ctx["med"], dosage or ctx.get("dosage"), True
        if not city and any(i.startswith("PHARMACY") for i in intents) and ctx.get("city"):
            city, used_ctx = ctx["city"], True

        blocks, data, clarify, rows, chips = [], {}, False, [], []
        if needs_med:
            if not base:
                return self._wrap(lang, [{"type": "text", "text": self._t(lang, "ask_med")}], clarify=True,
                                  suggestions=S[lang]["starters"])
            rows, dose_applied = self.kb.rows_for(base, dosage)
            if not rows:
                return self._wrap(lang, [{"type": "text", "text": self._t(lang, "not_found")}], clarify=True,
                                  ctx_used=used_ctx)
            if med_e and med_e.score < 100:
                note = self._t(lang, "understood", name=base.title(), raw=med_e.text)
                if a.med_alternatives:
                    note += " " + self._t(lang, "alts", alts=", ".join(x.title() for x in a.med_alternatives))
                blocks.append({"type": "text", "text": note})
            if dosage and not dose_applied:
                doses = sorted({r["dosage"] for r in rows if r.get("dosage")})[:6]
                if doses:
                    blocks.append({"type": "text", "text": self._t(lang, "dose_missing", doses=", ".join(doses))})
            ctx["med"], ctx["dosage"] = base, dosage

        for it in intents:
            if it == "MEDICATION_PRICE":
                b, d, chips = self._price(lang, base, rows, dosage if dose_applied_ok(rows, dosage) else None)
                blocks += b; data.update(d)
            elif it == "MEDICATION_REIMBURSEMENT":
                b, d = self._reimb(lang, base, rows); blocks += b; data.update(d)
            elif it == "MEDICATION_COMPOSITION":
                b, d = self._comp(lang, base, rows); blocks += b; data.update(d)
            elif it == "MEDICATION_INFO":
                b, d = self._info(lang, base, rows); blocks += b; data.update(d)
            elif it == "MEDICATION_AVAILABILITY":
                blocks += self._avail(lang, base, rows)
            elif it == "PHARMACY_SEARCH":
                b, d, c = self._pharmacy_search(lang, city, quartier); blocks += b; data.update(d); clarify |= c
            elif it == "PHARMACY_DUTY":
                b, d, c = self._pharmacy_duty(lang, city); blocks += b; data.update(d); clarify |= c
        if city:
            ctx["city"] = city
        if any(i in ("MEDICATION_PRICE", "MEDICATION_REIMBURSEMENT", "MEDICATION_COMPOSITION", "MEDICATION_INFO")
               for i in intents):
            blocks.append({"type": "notice", "text": S[lang]["disc"]})
        sug = self._suggest(lang, intents, chips) if not clarify else []
        return self._wrap(lang, blocks, data, clarify, used_ctx, sug)


def dose_applied_ok(rows, dosage):
    """vrai si un dosage a ete demande et correspond a au moins une ligne"""
    if not dosage:
        return False
    key = dose_to_key(dosage)
    return bool(key) and any(dose_to_key(r.get("dosage")) == key for r in rows)
