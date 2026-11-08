"""Ressources linguistiques : lexiques d'intents (FR / AR / darija arabe / darija latin), villes, quartiers.

Les motifs sont des expressions regulieres ecrites sur du texte DEJA normalise
(minuscules, sans accents, alef normalises, sans ponctuation).
  w("prix")   -> mot exact
  s("rembours") -> tout mot contenant ce radical
"""
import re

AR_DIAC = re.compile(r"[\u064B-\u0652\u0640]")


def ar_fold(t: str) -> str:
    """normalisation des lettres arabes (utilisee aussi sur les lexiques)"""
    t = AR_DIAC.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t)
    return (t.replace("ى", "ي").replace("ڤ", "ف").replace("گ", "ك")
             .replace("چ", "ش").replace("پ", "ب"))


def w(word: str) -> str:
    return r"(?<!\S)" + ar_fold(word) + r"(?!\S)"


def s(stem: str) -> str:
    return r"\S*" + ar_fold(stem) + r"\S*"


def p(stem: str) -> str:
    """mot commencant par ce radical"""
    return r"(?<!\S)" + ar_fold(stem) + r"\S*"


def P(*pats) -> re.Pattern:
    return re.compile("|".join(ar_fold(x) for x in pats))


# --- variantes d'ecriture du darija latin -> forme canonique (utilise pour les intents) ---
CANON = {
    "chhal": "ch7al", "sh7al": "ch7al", "chal": "ch7al", "bchhal": "bch7al", "bsh7al": "bch7al",
    "tamn": "taman", "thaman": "taman", "tmn": "taman",
    "wash": "wach", "wesh": "wach", "wa": "wach",
    "fine": "fin", "fen": "fin",
    "shno": "chno", "chnou": "chno", "chnu": "chno",
    "kayen": "kayn", "kayyn": "kayn", "kayne": "kayn",
    "bghyt": "bghit", "bgit": "bghit",
    "3andkom": "3ndkom", "3ndkum": "3ndkom", "3andkum": "3ndkom",
    "dial": "dyal", "diyal": "dyal",
    "9riba": "9riba", "9rib": "9riba",
    "lyoum": "lyoum", "lioum": "lyoum",
}

# --------------------------------------------------------------------------
# INTENTS MEDICAMENT
# --------------------------------------------------------------------------
PRICE = P(
    w("prix"), w("taman"), w("tarif"), w("price"), p("combien"), p("cout"), w("sa3r"),
    r"(?<!\S)b?ch7al(?!\S)", w("kaysswa"), w("kaysswi"), w("kayswa"), w("kaytbi3"),
    s("سعر"), s("ثمن"), w("شحال"), w("بشحال"), w("بكم"), w("كيسوا"), w("كيسوي"), w("كيتباع"),
    s("بشحال"),
)
REIMB = P(
    s("rembours"), s("t3awd"), s("t3awed"), s("t3wid"), w("cnss"), w("cnops"), w("assurance"), w("mutuelle"),
    r"prise en charge", s("تعويض"), s("مشمول"), s("يعوض"), s("كيتعوض"), w("الضمان"), s("الضمان"), s("تامين"),
)
AVAIL_STRONG = P(
    s("disponib"), w("dispo"), w("stock"), s("rupture"), r"avez vous", r"vous avez", w("3ndkom"),
    s("متوفر"), s("توفر"), w("عندكم"), w("عندها"), w("عنكم"),
)
AVAIL_WEAK = P(w("kayn"), w("كاين"), w("موجود"), w("موجودة"), r"mazal kayn", w("مازال"))
COMP = P(
    w("composition"), r"principe actif", s("substance"), w("contient"), s("tarkib"), r"lmadda l fa3ila",
    r"madda fa3ila", w("fa3ila"), s("مكونات"), s("المادة الفعالة"), s("الفعالة"), s("تركيب"), s("يحتوي"),
    s("تركيبة"), r"fih\S* chno", r"chno fih\S*", r"chno kayn", r"(?:شنو|ماذا) (?:كاين|فيه|فيها)", r"(?:فيه|فيها) شنو",
    r"شنو كاين", r"chno 3ndo", s("composant"),
)
INFO_STRONG = P(
    w("notice"), p("info"), s("indication"), w("sert"), r"pour quoi", s("ma3lumat"), s("معلومات"),
    r"فيما يستعمل", r"فاش كيصلح", s("kaysalah"), s("kaysalh"), s("kaytakhd"), r"علاش كيتاخد", w("كيتاخد"),
    s("kaytst3mel"), s("دواعي"), s("يستعمل"), w("parlez"), r"fach kayssalh", s("kaysalh"),
)
INFO_WEAK = P(r"chno howa", r"شنو هو", r"ما هو", r"c est quoi", r"qu est ce que")

# phrases qui se confondent avec 'kayn' (disponibilite) -> retirees avant le scoring de la disponibilite
_ABS = [r"kayn t3wid", r"kayn remboursement", r"كاين تعويض", r"chno kayn", r"شنو كاين",
                    r"kayn \S*t3wid\S*", r"fiha chno", r"fih chno", r"chno fih\S*", r"(?:فيه|فيها) شنو"]
ABSORB_FOR_AVAIL = [ar_fold(x) for x in _ABS]

# demande de localisation d'un produit : "fin nl9a X", "où trouver X"
WHERE = P(r"fin nl9a", r"fin nlqa", r"ou trouver", r"فين نلقى", r"اين اجد", r"اين توجد", r"فين كاين",
          r"fin kayn", r"ou acheter", r"fin n9dr nchri", r"فين نجيب")

# --------------------------------------------------------------------------
# INTENTS PHARMACIE
# --------------------------------------------------------------------------
DUTY = P(r"(?<!\S)(?:l|d|la|de|le)?garde(?!\S)", s("حراسة"), s("مداومة"), r"pharmacie de nuit", w("l7irassa"),
         r"pharmacie\S* nuit", s("مناوبة"), w("mdawma"), w("mdawem"), w("mdawm"), w("nuit"), w("night"), w("24h"),
         r"24 24", w("مفتوحة"))
PHARMACY = P(s("pharmacie"), s("صيدلي"), s("صيدليات"), s("pharmacy"))

# --------------------------------------------------------------------------
# SALUTATIONS / HORS PERIMETRE
# --------------------------------------------------------------------------
GREETING = P(w("salam"), w("bonjour"), w("bonsoir"), w("salut"), w("hello"), w("hi"), w("مرحبا"), w("اهلا"),
             r"السلام عليكم", r"صباح الخير", r"مساء الخير", r"sba7 lkhir", r"sba7 lkhair", w("سلام"), w("صباح"),
             w("مساء"), r"salam 3likom")
OOS = P(s("sa3a"), s("ساعة"), p("heure"), w("taxi"), w("طاكسي"), w("meteo"), w("jaw"), w("الجو"), w("الطقس"),
        s("temps fait"), w("capitale"), w("عاصمة"), p("recette"), w("tajine"), w("restaurant"), w("gare"),
        w("قطار"), w("تذكرة"), w("smitek"), w("سميتك"), r"comment aller", s("حجز"), r"n7jz", r"n réserver",
        r"reserver", r"akhssan", w("kifach l jaw"), r"كيفاش الجو", r"chno smitek")

# --------------------------------------------------------------------------
# MOTS VIDES (jamais pris pour un medicament)
# --------------------------------------------------------------------------
STOP = set("""
a au aux avez avoir c ca ce ces cet cette combien comme comment contient d dans de des du elle en est et
etes il je j la le les leur lui ma me mes mon mais ne ni non nos notre nous on ou par pas pour pouvez
qu que quel quelle quelles quels qui sa se ses si son sur ta te tes ton tu un une vos votre vous y
svp please plz merci bonjour salut sil
wach wash ch7al chhal taman dyal d f w l ou wla wlla la lyoum lil lila had hadi hada dak dik chi chno chnou
fin fine kayn kayna kayen 3ndkom 3andkom 3afak afak bghit bghyt bgit nl9a nl9ah n9leb kayt 3la 3lach mn m3a
mazal mazalo howa hiya fach 9riba 9rib 7da 7dya dyali dyalk dyalo ana nta nti l3fo bzaf chwiya ghir
kayn fih fiha fihom tkoun tkon momkin imken kifach kifash chkoun chmn chhal 3ndi 3andi
في من على الى عن هل ما ماذا ما هو هي هو هذا هذه الذي التي واش شنو شحال فين فاين كاين كاينة بغيت ديال د ف و ب ل
عندكم لدي لي لك لنا مع او ام ثم ان انا انت نحن هناك اين كيف لماذا علاش شكون شمن اي
ص م ك
dwa dawa dawaa dwaya medicament medicaments medoc produit boite boites comprime comprimes sirop gelule gelules
pharmacien pharmacienne docteur ordonnance aussi meme pareil celui celle ceci cela encore alors donc bien tres plus
moins quand ok voila daba db hta kolchi hadchi hadak hadik dakchi tani tanya
دواء الدواء دوا الدوا دوايا علبة حبوب سيروب صيدلي الصيدلي طبيب دابا
""".split())

# --------------------------------------------------------------------------
# NOMS DE MEDICAMENTS ECRITS EN ARABE (medicaments courants)
# --------------------------------------------------------------------------
ARABIC_NAMES = {
    "doliprane": ["دوليبران", "دوليپران", "دوليبرين"], "efferalgan": ["إيفيرالغان", "ايفيرالغان", "افيرالجان"],
    "augmentin": ["أوغمانتين", "اوغمنتين", "اوجمنتين"], "spasfon": ["سبازفون", "سباسفون"],
    "voltarene": ["فولتارين", "فولتارن"], "aspegic": ["أسبيجيك", "اسبيجيك"], "smecta": ["سميكتا", "سمكتا"],
    "gaviscon": ["غافيسكون", "جافيسكون"], "ventoline": ["فونتولين", "فينتولين"], "clamoxyl": ["كلاموكسيل"],
    "panadol": ["بنادول", "بانادول"], "rhumagrip": ["روماغريب"], "amoxil": ["أموكسيل", "اموكسيل"],
    "betadine": ["بيتادين", "بيتادين"], "oroken": ["اوروكين", "أوروكين"], "humex": ["هيمكس", "يوميكس"],
    "toplexil": ["توبليكسيل"], "exomuc": ["إكزوموك", "اكزوموك"], "daflon": ["دافلون"],
}

# --------------------------------------------------------------------------
# VILLES : canonique -> (formes latines, formes arabes)
# --------------------------------------------------------------------------
CITY_FORMS = {
    "Casablanca": (["Casablanca", "casa", "Dar El Beida", "Dar Lbayda"], ["الدار البيضاء", "كازا", "الدارالبيضاء"]),
    "Rabat": (["Rabat", "Rbat"], ["الرباط"]),
    "Salé": (["Salé", "Sale", "Sla"], ["سلا"]),
    "Marrakech": (["Marrakech", "Marrakesh", "Marrakch"], ["مراكش"]),
    "Fès": (["Fès", "Fes", "Fez"], ["فاس"]),
    "Tanger": (["Tanger", "Tangier", "Tanja"], ["طنجة"]),
    "Agadir": (["Agadir"], ["أكادير", "اكادير"]),
    "Meknès": (["Meknès", "Meknes"], ["مكناس"]),
    "Oujda": (["Oujda", "Wjda"], ["وجدة"]),
    "Tétouan": (["Tétouan", "Tetouan", "Titwan"], ["تطوان"]),
    "Kénitra": (["Kénitra", "Kenitra"], ["القنيطرة"]),
    "Nador": (["Nador"], ["الناظور"]),
    "Safi": (["Safi", "Asfi"], ["آسفي", "اسفي"]),
    "El Jadida": (["El Jadida", "Jadida"], ["الجديدة"]),
    "Settat": (["Settat"], ["سطات"]),
    "Mohammedia": (["Mohammedia"], ["المحمدية"]),
    "Béni Mellal": (["Béni Mellal", "Beni Mellal"], ["بني ملال"]),
    "Khouribga": (["Khouribga"], ["خريبكة"]),
    "Dakhla": (["Dakhla"], ["الداخلة"]),
    "Laâyoune": (["Laâyoune", "Laayoune"], ["العيون"]),
    "Témara": (["Témara", "Temara"], ["تمارة"]),
    "Essaouira": (["Essaouira"], ["الصويرة"]),
    "Ouarzazate": (["Ouarzazate"], ["ورزازات"]),
    "Berkane": (["Berkane"], ["بركان"]),
    "Larache": (["Larache"], ["العرائش"]),
    "Al hoceima": (["Al Hoceima", "Hoceima"], ["الحسيمة"]),
    "Dar Bouazza": (["Dar Bouazza"], ["دار بوعزة"]),
    "Inezgane": (["Inezgane"], ["إنزكان", "انزكان"]),
    "Séfrou": (["Séfrou", "Sefrou"], ["صفرو"]),
}

# quartiers : ville -> [(latin, arabe)]  (le mot generique "المدينة" est volontairement exclu de l'arabe)
QUARTIERS = {
    "Rabat": [("Agdal", "أكدال"), ("Hay Riad", "حي الرياض"), ("Hassan", "حسان"), ("Océan", "الأوداية")],
    "Casablanca": [("Maarif", "المعاريف"), ("Gauthier", "غوتيي"), ("Ain Diab", "عين الذئاب"), ("Hay Hassani", "الحي الحسني")],
    "Marrakech": [("Guéliz", "گيليز"), ("Hivernage", "هيفرناج"), ("Médina", None)],
    "Fès": [("Atlas", "أطلس"), ("Narjiss", "نرجس")],
    "Tanger": [("Malabata", "مالاباطا"), ("Iberia", "إيبيريا")],
}
