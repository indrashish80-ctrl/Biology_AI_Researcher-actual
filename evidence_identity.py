"""
evidence_identity.py
--------------------
A quote can match a paper word for word and still be evidence about a DIFFERENT biological
system (arterial plaques, myocardial ischaemia, a related-but-distinct disease). Textual
grounding measures correspondence, not scientific relevance. This module gives every claim a
full EVIDENCE IDENTITY and applies one hard rule before anything enters the evidence pool:

    Claim
    ├── Source paper              ├── Intervention / exposure
    ├── Original experiment       ├── Comparator
    ├── Evidence type             ├── Dose
    ├── Species                   ├── Duration
    ├── Disease                   ├── Outcome
    ├── Disease subtype           ├── Direction
    ├── Cell / tissue             ├── Statistical evidence
    │                             ├── Causal language
    │                             └── Exact source passage

    HARD RULE: if the claim's disease, model (cell / tissue system), intervention/exposure or
    outcome does not match the research question, it cannot enter the evidence pool.

It also provides
  * question normalisation  - spelling ("leukimia" -> "leukemia") before anything is searched
  * the question frame      - intent (causes / treatment / cure / mechanism ...), disease, subtypes,
                              named exposures, endpoints, organism restriction
  * the evidence ROLE of each admitted claim. A mechanism is not a cause:
        disease_risk     exposure -> occurrence / risk / incidence of the disease (can address causes)
        intermediate     exposure -> intermediate endpoint (chromosome damage, mutation ...) - the
                         link from that endpoint to the disease is NOT shown by this claim
        mechanistic      a biological process inside disease cells/models (apoptosis, ROS, expression)
        direct           measures the outcome the question asks about (treatment / effect questions)
        secondary_report the result is reported by a review, not observed by this paper
  * subtype tracking for heterogeneous diseases (leukemia = AML / ALL / CML / CLL / ...)
"""

from __future__ import annotations

import difflib
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

_I = re.I

# --------------------------------------------------------------------------- disease lexicon
# words: case-insensitive pattern; acronyms: CASE-SENSITIVE pattern (so 'ALL' is not 'all')
DISEASES: Dict[str, Dict[str, Any]] = {
    "leukemia": {
        "words": r"leuk(?:a)?emi\w*|leuk(?:a)?emogen\w*|myelogenous|lymphoblastic|lymphocytic leuk\w*|blast crisis|"
                 r"acute (?:myeloid|lymphoid|promyelocytic|non-?lymphocytic)",
        "acronyms": r"\b(?:AML|B-ALL|T-ALL|ALL|CML|CLL|APL|JMML|CMML)\b",
        "organ": "blood",
        "heterogeneous": True,
        "subtypes": [
            ("AML", r"acute myeloid leuk\w*|acute myelogenous leuk\w*|acute non-?lymphocytic leuk\w*", r"\bAML\b"),
            ("ALL", r"acute lymphoblastic leuk\w*|acute lymphocytic leuk\w*|precursor B-?cell leuk\w*", r"\b(?:B-|T-)?ALL\b"),
            ("CML", r"chronic myeloid leuk\w*|chronic myelogenous leuk\w*", r"\bCML\b"),
            ("CLL", r"chronic lymphocytic leuk\w*", r"\bCLL\b"),
            ("APL", r"acute promyelocytic leuk\w*", r"\bAPL\b"),
            ("JMML", r"juvenile myelomonocytic leuk\w*", r"\bJMML\b"),
            ("CMML", r"chronic myelomonocytic leuk\w*", r"\bCMML\b"),
            ("hairy cell leukemia", r"hairy[- ]cell leuk\w*", r"$^"),
        ],
        "molecular": [
            ("BCR-ABL1 / t(9;22)", r"BCR[-:/ ]?ABL1?|t\(9;22\)|philadelphia chromosome"),
            ("PML-RARA / t(15;17)", r"PML[-:/ ]?RARA?|t\(15;17\)"),
            ("RUNX1-RUNX1T1 / t(8;21)", r"RUNX1[-:/ ]RUNX1T1|AML1[-:/ ]ETO|t\(8;21\)"),
            ("FLT3-ITD", r"FLT3[- ]?ITD"),
            ("NPM1-mutated", r"NPM1"),
            ("KMT2A (MLL)-rearranged", r"KMT2A|MLL[- ]?r\b|MLL-rearranged|t\(4;11\)"),
            ("ETV6-RUNX1 / t(12;21)", r"ETV6[-:/ ]RUNX1|TEL[-:/ ]AML1|t\(12;21\)"),
        ],
        "related": [("myelodysplastic syndrome", r"myelodysplastic|\bMDS\b"),
                    ("myeloproliferative neoplasm", r"myeloproliferative|\bMPNs?\b"),
                    ("lymphoma", r"lymphoma"), ("multiple myeloma", r"myeloma"),
                    ("anaemia / other blood disorder", r"an(?:a)?emi\w*|reticulocyt\w*|erythropoie\w*")],
    },
    "vitiligo / leukoderma": {
        "words": r"vitiligo|leu[ck]oderma|depigment\w*|achromotrichia|hypopigment\w*|loss of (?:skin |hair )?pigment\w*|"
                 r"white (?:patches|spots)",
        "acronyms": r"$^", "organ": "skin", "heterogeneous": True,
        "subtypes": [("non-segmental", r"non-?segmental|generali[sz]ed vitiligo", r"$^"), ("segmental", r"(?<!non-)(?<!non)segmental", r"$^"),
                     ("acral / acrofacial", r"acr(?:al|ofacial)", r"$^"), ("chemical / occupational", r"chemical|occupational", r"$^")],
        "molecular": [],
        "related": [("albinism", r"albinism"), ("piebaldism", r"piebald\w*"), ("melasma", r"melasma")],
    },
    "breast cancer": {
        "words": r"breast (?:cancer|carcinoma|tumou?r|neoplasm)s?|mammary (?:gland )?(?:carcinoma|tumou?r|cancer|neoplasm|adenocarcinoma)s?|"
                 r"MCF-?7|MDA-MB-\d+|T-?47D|SK-BR-3|BT-?474|4T1",
        "acronyms": r"\b(?:TNBC|DCIS)\b", "organ": "breast", "heterogeneous": True,
        "subtypes": [("ER-positive", r"ER[- ]?positive|ER\+|oestrogen receptor[- ]positive|estrogen receptor[- ]positive|luminal", r"$^"),
                     ("HER2-positive", r"HER2[- ]?positive|HER2\+", r"$^"),
                     ("triple-negative", r"triple[- ]negative", r"\bTNBC\b")],
        "molecular": [("BRCA1/2-mutated", r"BRCA[12]")],
        "related": [("ovarian cancer", r"ovarian cancer")],
    },
    "diabetes": {
        "words": r"diabet\w*|hyperglyc(?:a)?emi\w*", "acronyms": r"\bT[12]DM\b", "organ": None, "heterogeneous": True,
        "subtypes": [("type 1", r"type 1 diabet\w*|type I diabet\w*", r"\bT1DM\b"), ("type 2", r"type 2 diabet\w*|type II diabet\w*", r"\bT2DM\b"),
                     ("gestational", r"gestational diabet\w*", r"$^")],
        "molecular": [], "related": [("prediabetes / insulin resistance", r"prediabet\w*|insulin resistance")],
    },
    "lymphoma": {"words": r"lymphoma\w*", "acronyms": r"\b(?:DLBCL|NHL)\b", "organ": "blood", "heterogeneous": True,
                 "subtypes": [("Hodgkin", r"hodgkin", r"$^"), ("non-Hodgkin", r"non-?hodgkin", r"\bNHL\b")], "molecular": [],
                 "related": [("leukemia", r"leuk(?:a)?emi\w*")]},
    "alzheimer's disease": {"words": r"alzheimer\w*", "acronyms": r"$^", "organ": "brain", "heterogeneous": False,
                            "subtypes": [], "molecular": [], "related": [("other dementia", r"dementia")]},
    "parkinson's disease": {"words": r"parkinson\w*", "acronyms": r"$^", "organ": "brain", "heterogeneous": False,
                            "subtypes": [], "molecular": [], "related": []},
    "asthma": {"words": r"asthma\w*", "acronyms": r"$^", "organ": "lung", "heterogeneous": True, "subtypes": [], "molecular": [],
               "related": [("COPD", r"\bCOPD\b|chronic obstructive")]},
    "psoriasis": {"words": r"psoria\w*", "acronyms": r"$^", "organ": "skin", "heterogeneous": False, "subtypes": [],
                  "molecular": [], "related": []},
    "malaria": {"words": r"malaria\w*|plasmodium", "acronyms": r"$^", "organ": None, "heterogeneous": False, "subtypes": [],
                "molecular": [], "related": []},
    "tuberculosis": {"words": r"tubercul\w*|mycobacterium tuberculosis", "acronyms": r"\bTB\b", "organ": "lung",
                     "heterogeneous": False, "subtypes": [], "molecular": [], "related": []},
    "colorectal cancer": {"words": r"colorectal (?:cancer|carcinoma|tumou?r)s?|colon (?:cancer|carcinoma)|HCT-?116|HT-?29|SW480",
                          "acronyms": r"\bCRC\b", "organ": "gut", "heterogeneous": True, "subtypes": [], "molecular": [], "related": []},
    "lung cancer": {"words": r"lung (?:cancer|carcinoma|adenocarcinoma)s?|A549", "acronyms": r"\b(?:NSCLC|SCLC)\b", "organ": "lung",
                    "heterogeneous": True, "subtypes": [("NSCLC", r"non-?small[- ]cell", r"\bNSCLC\b"), ("SCLC", r"(?<!non-)small[- ]cell", r"\bSCLC\b")],
                    "molecular": [], "related": []},
    "prostate cancer": {"words": r"prostate (?:cancer|carcinoma)s?|LNCaP|PC-?3|DU-?145", "acronyms": r"$^", "organ": "prostate",
                        "heterogeneous": True, "subtypes": [], "molecular": [], "related": []},
    "melanoma": {"words": r"melanoma\w*", "acronyms": r"$^", "organ": "skin", "heterogeneous": True, "subtypes": [], "molecular": [],
                 "related": []},
    "hypertension": {"words": r"hypertensi\w*|high blood pressure", "acronyms": r"$^", "organ": "cardiovascular",
                     "heterogeneous": False, "subtypes": [], "molecular": [], "related": []},
    "obesity": {"words": r"obes\w*", "acronyms": r"$^", "organ": None, "heterogeneous": False, "subtypes": [], "molecular": [],
                "related": []},
}
for _d in DISEASES.values():
    _d["rx_words"] = re.compile(r"\b(?:" + _d["words"] + r")", _I)
    _d["rx_acr"] = re.compile(_d["acronyms"])

# other diseases / organ systems that signal a claim is about a DIFFERENT biological system
OTHER_CONDITIONS = [
    ("cardiovascular disease", r"cardiovascular|atheroscler\w*|arterial plaques?|plaques? of (?:the )?arter\w*|coronary|"
                               r"heart failure|cardiomyopath\w*|myocardial (?:infarction|isch(?:a)?emi\w*)|isch(?:a)?emi\w*[/ -]reperfusion|"
                               r"vascular (?:smooth muscle|injury|remodel\w*)|restenosis|aortic"),
    ("neurological disease", r"neurodegenerat\w*|alzheimer\w*|parkinson\w*|stroke|epilep\w*|multiple sclerosis"),
    ("liver disease", r"steatohepatitis|fatty liver|\bNAFLD\b|\bNASH\b|cirrho\w*|hepatitis"),
    ("kidney disease", r"nephropath\w*|chronic kidney|renal (?:failure|injury|fibrosis)"),
    ("metabolic disease", r"(?<!non-)(?<!non )diabet\w*|obes\w*|insulin resistance"),
    ("inflammatory / autoimmune disease", r"arthritis|colitis|crohn\w*|lupus|psoria\w*"),
    ("infectious disease", r"sepsis|covid|sars-cov-2|\bHIV\b|tubercul\w*|malaria"),
    ("other cancer", r"(?:lung|colorectal|colon|prostate|ovarian|pancreatic|gastric|liver|hepatocellular|cervical|bladder|"
                     r"renal cell|head and neck|thyroid) (?:cancer|carcinoma)|glioblastoma|glioma|osteosarcoma|neuroblastoma"),
]
OTHER_RX = [(n, re.compile(r"\b(?:" + p + r")", _I)) for n, p in OTHER_CONDITIONS]

# organ / cell systems (bone marrow is BLOOD, not bone)
ORGANS = [
    ("blood", r"bone marrow|h(?:a)?ematopoie\w*|blood(?!\s+(?:concentration|level|glucose|pressure|sugar|lipid|test|draw))|myeloid|lymphoid|lymphocyt\w*|leukocyt\w*|leucocyt\w*|monocyt\w*|"
              r"granulocyt\w*|neutrophil\w*|blasts?\b|erythro\w*|platelet\w*|spleen|thym\w*|PBMCs?|K562|HL-?60|Jurkat|"
              r"MOLM-?13|MV4-?11|NB4|U-?937|KG-?1|Kasumi-?1|OCI-AML\d|NALM-?6|REH|RS4;11|MEC-?1|CCRF-CEM|THP-1|Ba/F3"),
    ("cardiovascular", r"cardiomyocyte\w*|myocardi\w*|cardiac|\bheart\b|vascular smooth muscle|VSMCs?|arter\w*|aort\w*|"
                       r"endotheli\w*|HUVECs?|plaques?\b"),
    ("liver", r"hepatocyte\w*|hepatic|\bliver\b|HepG2|Huh-?7"),
    ("brain", r"neurons?\b|neuronal|neural|\bbrain\b|astrocyt\w*|microglia\w*|hippocamp\w*|cortex|SH-SY5Y"),
    ("kidney", r"\bkidney\w*|\brenal\b|podocyt\w*|HEK-?293"),
    ("skin", r"keratinocyt\w*|melanocyt\w*|\bskin\b|epiderm\w*|derm(?:al|is)|HaCaT"),
    ("lung", r"\blungs?\b|alveol\w*|bronch\w*|airway\w*|A549"),
    ("gut", r"intestin\w*|\bcolon(?:ic|s|ocytes?)?\b(?!y)|\bgut\b|enterocyt\w*|Caco-?2"),
    ("bone", r"osteoblast\w*|osteoclast\w*|osteocyt\w*|cartilage|chondrocyt\w*|(?<!marrow )\bbone\b(?! marrow)"),
    ("breast", r"mammary|\bbreast\b|MCF-?7|MDA-MB"),
    ("muscle", r"skeletal muscle|myotube\w*|myoblast\w*|C2C12"),
    ("prostate", r"prostat\w*|LNCaP"),
]
ORGAN_RX = [(n, re.compile(r"\b(?:" + p + r")", _I)) for n, p in ORGANS]

# --------------------------------------------------------------------------- question normalisation
_LEXICON = sorted({
    "leukemia", "leukaemia", "lymphoma", "myeloma", "carcinoma", "melanoma", "sarcoma", "glioblastoma", "neuroblastoma",
    "diabetes", "alzheimer", "parkinson", "schizophrenia", "tuberculosis", "malaria", "psoriasis", "eczema", "vitiligo",
    "leukoderma", "arthritis", "osteoporosis", "hypertension", "asthma", "pneumonia", "hepatitis", "cirrhosis", "anemia",
    "anaemia", "dementia", "epilepsy", "obesity", "atherosclerosis", "myelodysplastic", "lymphoblastic", "myeloid",
    "myelogenous", "lymphocytic", "promyelocytic", "metformin", "aspirin", "apoptosis", "proliferation", "inflammation",
    "mitochondria", "mitochondrial", "chromosome", "chromosomal", "mutation", "carcinogen", "carcinogenic", "benzene",
    "radiation", "exposure", "prevalence", "incidence", "treatment", "therapy", "efficacy", "bacteria", "bacterial",
    "antibiotic", "resistance", "mechanism", "pathway", "protein", "expression", "cancer", "tumour", "tumor", "disease",
    "syndrome", "infection", "virus", "causes", "curable", "remission", "pigmentation", "depigmentation", "immunotherapy",
    "chemotherapy", "genetic", "hereditary", "environmental", "pesticides", "smoking", "drought", "photosynthesis",
    "arabidopsis", "escherichia", "staphylococcus", "pseudomonas", "zebrafish", "microbiome", "inheritance",
    # organs / tissues
    "breast", "prostate", "colorectal", "colon", "rectal", "pancreatic", "pancreas", "ovarian", "ovary", "cervical",
    "endometrial", "gastric", "stomach", "esophageal", "oesophageal", "hepatic", "hepatocellular", "liver", "kidney",
    "renal", "bladder", "thyroid", "brain", "glioma", "lung", "pulmonary", "cardiac", "heart", "skin", "bone", "muscle",
    "intestinal", "neuronal", "neurons", "cardiovascular", "vascular", "endothelial", "epithelial", "adipose", "blood",
    # diseases
    "cardiomyopathy", "fibrosis", "nephropathy", "neuropathy", "retinopathy", "colitis", "crohn", "lupus", "sclerosis",
    "huntington", "autism", "depression", "anxiety", "insomnia", "migraine", "stroke", "ischemia", "ischaemia",
    "infarction", "thrombosis", "sepsis", "influenza", "coronavirus", "covid", "malignancy", "metastasis", "metastatic",
    "adenocarcinoma", "mesothelioma", "osteosarcoma", "retinoblastoma", "hyperglycemia", "hypoglycemia", "insulin",
    "glucose", "cholesterol", "dyslipidemia", "steatosis", "fatty", "alopecia", "dermatitis", "acne", "rosacea",
    # drugs / agents
    "tamoxifen", "letrozole", "trastuzumab", "cisplatin", "doxorubicin", "paclitaxel", "docetaxel", "gemcitabine",
    "imatinib", "rapamycin", "everolimus", "statin", "statins", "atorvastatin", "simvastatin", "ibuprofen", "curcumin",
    "resveratrol", "berberine", "quercetin", "melatonin", "vitamin", "caffeine", "nicotine", "alcohol", "ethanol",
    "probiotics", "antioxidant", "dexamethasone", "prednisone", "hydroxychloroquine", "ivermectin", "penicillin",
    "vancomycin", "methotrexate", "pembrolizumab", "nivolumab", "bevacizumab", "olaparib", "fluorouracil",
    # endpoints / biology
    "survival", "mortality", "recurrence", "relapse", "growth", "invasion", "migration", "angiogenesis", "autophagy",
    "senescence", "differentiation", "oxidative", "stress", "metabolism", "metabolic", "glycolysis", "signaling",
    "signalling", "receptor", "kinase", "enzyme", "hormone", "estrogen", "oestrogen", "androgen", "testosterone",
    "cytokine", "immune", "immunity", "antibody", "vaccine", "neurodegeneration", "cognition", "memory", "obesity",
    "weight", "fertility", "pregnancy", "longevity", "aging", "ageing", "lifespan", "toxicity", "cytotoxicity",
    "viability", "tumorigenesis", "carcinogenesis", "microbiota", "biofilm", "virulence", "transcription", "methylation",
    "epigenetic", "genome", "sequencing",
    # question verbs (so 'reduse' -> 'reduce', while 'induce' stays 'induce')
    "reduce", "increase", "decrease", "prevent", "inhibit", "promote", "induce", "affect", "improve", "suppress",
    "trigger", "regulate", "enhance", "protect", "contribute", "influence", "worsen", "delay", "reverse", "treat",
    "patients", "humans", "people", "women", "children", "adults", "elderly", "mice",
})
_NO_CORRECT = set("there their which where whose would could should about these those other another because cause causes "
                  "humans human people women children adults patients mice rats cells".split())


_SUFFIX = re.compile(r"(?:ations?|ances?|ences?|ally|ing|ity|ive|ant|ent|ed|es|al|ic|ly|s)$")


def _stem(w: str) -> str:
    return _SUFFIX.sub("", w)


_EXACT = threading.local()


class exact_spelling:
    """`with exact_spelling():` - the user chose 'search exactly as typed', so no spelling pass changes the question
    (local or PubMed) for the work done in this thread."""
    def __enter__(self):
        self._prev = getattr(_EXACT, "on", False)
        _EXACT.on = True
        return self

    def __exit__(self, *exc):
        _EXACT.on = self._prev
        return False


def spelling_is_exact() -> bool:
    return bool(getattr(_EXACT, "on", False))


_CHAT_SHORT = {"r": "are", "u": "you", "ur": "your", "wat": "what", "wht": "what", "abt": "about", "bcoz": "because",
               "bcz": "because", "coz": "because", "hw": "how", "w/": "with", "w/o": "without",
               "pls": "please", "plz": "please", "thx": "thanks", "becoz": "because"}


_OPPOSITE_PREFIXES = (("re", "de"), ("hyper", "hypo"), ("up", "down"), ("pro", "anti"), ("in", "ex"), ("over", "under"),
                      ("hyper", "hypo"), ("intra", "extra"), ("pre", "post"), ("inter", "intra"), ("en", "dis"),
                      ("a", "an"), ("im", "ex"), ("endo", "exo"), ("homo", "hetero"), ("macro", "micro"), ("poly", "mono"))


def _opposite_prefix(a: str, b: str) -> bool:
    for x, y in _OPPOSITE_PREFIXES:
        for p, q in ((x, y), (y, x)):
            if a.startswith(p) and b.startswith(q) and a[len(p):] == b[len(q):]:
                return True
    return False


def normalize_question(question: str) -> Dict[str, Any]:
    """Correct clear misspellings of biomedical terms ('leukimia' -> 'leukemia'). Conservative: a word is
    corrected only if it is not itself a known term, is at least 5 letters, and is very close to ONE term."""
    q = question or ""
    if spelling_is_exact():
        return {"question": q.strip(), "original": q, "corrections": [], "note": "", "exact": True}
    corrections: List[Tuple[str, str]] = []

    def fix(m: re.Match) -> str:
        w = m.group(0)
        low = w.lower()
        if len(low) < 5 or low in _LEXICON or low in _NO_CORRECT or any(low.startswith(t) or t.startswith(low) for t in _LEXICON):
            return w
        # v25: a correctly spelled English word is never 'corrected' ('benzene-exposed workers' became 'benzene-exposure')
        try:
            from spelling import is_known as _known
            if _known(low):
                return w
        except Exception:
            pass
        # one typo in a 5-7 letter word scores ~0.83 ('reduse' vs 'reduce'); the first letter must then match so
        # 'induce' never becomes 'reduce'
        cand = difflib.get_close_matches(low, _LEXICON, n=2, cutoff=0.84 if len(low) > 7 else 0.8)
        if cand and len(low) <= 7:
            cand = [c for c in cand if c[0] == low[0]]
        if not cand or (len(cand) > 1 and difflib.SequenceMatcher(None, low, cand[0]).ratio()
                        == difflib.SequenceMatcher(None, low, cand[1]).ratio()):
            return w
        new = cand[0]
        if _stem(low) == _stem(new):
            return w                       # an inflection ('resistant' vs 'resistance'), not a misspelling
        # v25: a different PREFIX is a different word, often the opposite one ('repigmentation' was "corrected" to
        # 'depigmentation', 'hyperglycaemia' could become 'hypoglycaemia'). A typo fix keeps the first two letters
        # unless it is a swap of those two letters.
        if low[:2] != new[:2] and not (low[:2] == new[1::-1][:2]) or _opposite_prefix(low, new):
            return w
        new = new.capitalize() if w[0].isupper() else new
        corrections.append((w, new))
        return new

    fixed = re.sub(r"[A-Za-z]+", fix, q)
    # chat shorthand ('what r its symptoms', 'how does it affect u') - expanded so it is never searched as a word
    for short, full in _CHAT_SHORT.items():
        new_fixed = re.sub(rf"(?<![\w'-]){short}(?![\w'-])", full, fixed, flags=re.I)
        if new_fixed != fixed:
            corrections.append((short, full))
            fixed = new_fixed
    fixed = re.sub(r"\s+", " ", fixed).strip()
    try:                                   # second pass: domain-ranked speller (sound-alikes, run-together words)
        from spelling import correct_question
        more = correct_question(fixed)
        fixed = more["question"]
        corrections.extend(more["corrections"])
    except Exception:
        pass
    return {"question": fixed, "original": q, "corrections": corrections,
            "note": ("Question normalised before searching: " + ", ".join(f"'{a}' -> '{b}'" for a, b in corrections))
            if corrections else ""}


def _known_word(low: str) -> bool:
    return (len(low) < 5 or low in _LEXICON or low in _NO_CORRECT or low in _Q_STOP
            or any(low.startswith(t) or t.startswith(low) for t in _LEXICON))


def online_spelling(question: str, http: Any = None, timeout: float = 6.0) -> Dict[str, Any]:
    """Second spelling pass with NCBI's PubMed spell checker (ESpell) for words the local vocabulary does not know
    ('metformn', 'glioblastomma', a rare drug name). Runs on the user's machine; offline or on any error the question
    is returned unchanged. A word is replaced only by a close correction (same first letter, similarity >= 0.75)."""
    q = question or ""
    if spelling_is_exact():
        return {"question": q, "corrections": [], "note": ""}
    unknown = [w for w in re.findall(r"[A-Za-z]{5,}", q) if not _known_word(w.lower())]
    if not unknown:
        return {"question": q, "corrections": [], "note": ""}
    try:
        if http is None:
            from http_client import HttpClient
            http = HttpClient(cache_dir="cache/retrieval/ncbi")
        from ncbi_throttle import api_key, ncbi_wait
        ncbi_wait(False)
        _p = {"db": "pubmed", "term": " ".join(unknown), "tool": "BiologyAIResearcher"}
        if api_key():
            _p["api_key"] = api_key()
        xml = http.get_text("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi", params=_p)
        m = re.search(r"<CorrectedQuery>(.*?)</CorrectedQuery>", xml or "", re.S)
        fixed_words = (m.group(1).split() if m else [])
    except Exception:
        return {"question": q, "corrections": [], "note": ""}
    corrections: List[Tuple[str, str]] = []
    if len(fixed_words) == len(unknown):
        for w, new in zip(unknown, fixed_words):
            if new.lower() != w.lower() and new[:1].lower() == w[:1].lower() and \
                    difflib.SequenceMatcher(None, w.lower(), new.lower()).ratio() >= 0.75:
                q = re.sub(rf"\b{re.escape(w)}\b", new.capitalize() if w[0].isupper() else new, q, count=1)
                corrections.append((w, new))
    return {"question": q, "corrections": corrections,
            "note": ("Spelling checked with PubMed: " + ", ".join(f"'{a}' -> '{b}'" for a, b in corrections)) if corrections else ""}


# --------------------------------------------------------------------------- question frame
INTENTS = [
    ("etiology", r"\b(?:causes?|caused by|causing|cause of|causal|etiolog\w*|aetiolog\w*|risk factors?|"
                 r"what (?:leads|contributes) to|why do(?:es)? (?:people|someone|patients)|origin of|triggers? (?:of|for)|"
                 r"predispos\w*)\b"),
    ("cure", r"\b(?:cur(?:e|es|ed|able|ability|ative)|incurable|eradicat\w*|reversible|remission)\b"),
    ("treatment", r"\b(?:treat\w*|therap\w*|effective|efficacy|effect of|effects of|benefit\w*|improv\w*|reduc\w*|"
                  r"prevent\w*|inhibit\w*|increas\w*|decreas\w*|affect\w*)\b"),
    ("mechanism", r"\b(?:how does|how do|mechanism\w*|pathways?|why does)\b"),
    ("prognosis", r"\b(?:prognos\w*|survival|outcome)\b"),
    ("epidemiology", r"\b(?:prevalence|incidence|how common)\b"),
    ("clinical_features", r"\b(?:symptoms?|signs?(?: and symptoms?)?|clinical (?:features?|presentations?|picture|manifestations?|"
                          r"course)|presentations?|manifestations?|look like|diagnos\w*|differential)\b"),
]
_Q_STOP = set("""its it it's they them their his her you your please also symptoms symptom signs sign features feature
presentation presentations manifestation manifestations diagnosis diagnose diagnosed differential look like
what which who whom whose why when where how is are was were be been being do does did can could would should
may might will shall the a an of in on for to and or with by from at as into about than then there their this that these
those any some all known main major common most best current role roles effect effects cause causes causal caused causing
risk factor factors develop development reason reasons etiology aetiology origin triggers trigger predisposing exist
cure cured curable treat treated treatment treatments therapy therapies effective efficacy benefit benefits work works help
reduce reduces increase increases decrease decreases improve improves prevent prevents affect affects inhibit inhibits
people person humans human patients patient women men children adults individuals subjects mice mouse rats rat cells
cell model models vivo vitro study studies evidence exposure exposures agent agents mechanism mechanisms pathway pathways
lead leads contribute contributes associated association levels level outcome outcomes prognosis survival
drives drive driven driving common role underlie underlies explain explains determine determines regulate regulates
control controls influence influences mediate mediates coli spp species strain strains
slow slows slowed lower lowers lowered raise raises boost boosts extend extends shorten shortens delay delays worsen worsens
protect protects promote promotes induce induces enhance enhances suppress suppresses reverse reverses alter alters
change changes modulate modulates impact impacts fight fights kill kills stop stops block blocks really actually much
good bad safe safely better worse useful harmful have has had get gets taking take takes using use used eat eating
drink drinking
attenuate attenuates attenuated ameliorate ameliorates ameliorated alleviate alleviates alleviated mitigate mitigates
mitigated abolish abolishes abolished abrogate abrogates diminish diminishes dampen dampens curb curbs counteract
counteracts restore restores restored rescue rescues rescued sensitise sensitises sensitize sensitizes potentiate
potentiates augment augments elevate elevates upregulate upregulates downregulate downregulates activate activates
strengthen strengthens confer confers exacerbate exacerbates aggravate aggravates accelerate accelerates facilitate
facilitates impair impairs eradicate eradicates clear clears resolve resolves relieve relieves minimise minimises
minimize minimizes maximise maximises amplify amplifies knockout knockdown silencing overexpression tolerance
susceptibility sensitivity resistance under during within following after before versus
idiopathic chronic acute severe mild moderate advanced early late primary secondary systemic localised localized
pulmonary hepatic renal cardiac cerebral gastric intestinal cutaneous formation production accumulation
content index score activity capacity ability rate rates"""
.split())
# cues that a question is about an organism in the laboratory, not about treating people
_LAB_GENETICS = re.compile(r"\b(?:CRISPR\w*|CRISPRi|Cas9|knock-?outs?|knock-?downs?|knock-?ins?|mutants?|deletions?|"
                           r"\u0394\w+|transposons?|plasmids?|operons?|promoters?|guide RNAs?|sgRNAs?|gRNAs?|silencing|"
                           r"overexpress\w*|strains?|isolates?|in vitro|cultures?|culturing|biofilms?|planktonic|"
                           r"minimum inhibitory|\bMIC\b|\bMBC\b|\bCFU\b|time[- ]kill|broth|agar|auxotroph\w*)\b", _I)
CLASS_WORDS = {"drug", "drugs", "chemotherapy", "antibiotic", "antibiotics", "pollutant", "pollutants", "toxin", "toxins",
               "chemicals", "chemical", "diet", "lifestyle", "environment", "environmental", "genetic", "genetics", "genes",
               "gene", "infection", "infections", "virus", "viruses", "radiation", "supplement", "supplements", "herbal"}


def match_disease(question: str) -> Optional[str]:
    for name, d in DISEASES.items():
        if d["rx_words"].search(question or "") or d["rx_acr"].search(question or ""):
            return name
    return None


def question_frame(question: str) -> Dict[str, Any]:
    from clinical_context import parse_question
    from semantic_audit import FAMILY_RX
    norm = normalize_question(question)
    q = norm["question"]
    intents = [name for name, rx in INTENTS if re.search(rx, q, _I)]
    intent = intents[0] if intents else "general"
    disease = match_disease(q)
    info = parse_question(q)
    terms = info["disease_terms"]
    spec = DISEASES.get(disease) if disease else None
    if spec is None and terms:                       # generic disease not in the lexicon: build a tolerant pattern
        # EVERY disease word of the question counts ('silicosis-like lung fibrosis' -> silicosis OR fibrosis), so a
        # paper on 'pulmonary fibrosis' in an obsidian-dust model is not excluded for lacking the word 'silicosis'
        stems = [re.escape(t[:max(5, len(t) - 2)]).replace("ae", "a?e").replace("oe", "o?e") for t in dict.fromkeys(terms)]
        spec = {"words": "|".join(st + r"\w*" for st in stems), "acronyms": r"$^", "organ": None, "heterogeneous": False,
                "subtypes": [], "molecular": [], "related": []}
        spec["rx_words"] = re.compile(r"\b(?:" + spec["words"] + r")", _I)
        spec["rx_acr"] = re.compile(spec["acronyms"])
        disease = " / ".join(dict.fromkeys(terms))
    q_subtypes = [n for n, w, a in (spec or {}).get("subtypes", []) if re.search(w, q, _I) or re.search(a, q)]
    endpoints = [f for f, rx, _ in FAMILY_RX if rx.search(q)]
    if "microbial_growth" in endpoints:
        # 'Mycobacterium tuberculosis survival' is the microbe's survival, not a patient's
        from semantic_audit import MICROBE_WORDS
        if re.search(rf"\b(?:{MICROBE_WORDS})", q, _I):
            endpoints = [e for e in endpoints if e not in ("patient_survival", "viability", "proliferation", "tumour_growth")]
    # 'Does <X> cause / reduce / improve <Y> (in <organism>)?' - Y is the OUTCOME, X the exposure. Outside clinical
    # medicine Y is often not a built-in family ('pectoral fin defects', 'drought tolerance'), and without this it
    # would be mistaken for a second exposure and the question could never be answered directly.
    endpoint_phrase = ""
    _vm = re.search(r"\b(?:caus\w*|lead(?:s)? to|result(?:s)? in|trigger\w*|induc\w*|produc\w*|affect\w*|influenc\w*|"
                    r"reduc\w*|decreas\w*|increas\w*|lower\w*|rais\w*|improv\w*|enhanc\w*|attenuat\w*|ameliorat\w*|"
                    r"alleviat\w*|mitigat\w*|suppress\w*|inhibit\w*|prevent\w*|promot\w*|restor\w*|rescue\w*|"
                    r"sensiti[sz]\w*|abolish\w*|impair\w*|accelerat\w*|delay\w*|change\w*|alter\w*|modulat\w*)\b", q, _I)
    if _vm:
        tail = q[_vm.end():]
        tail = re.split(r"\b(?:in|among|within|across|of)\s+(?:\w+\s+){0,2}"
                        r"(?:humans?|patients?|people|women|men|children|adults|mice|mouse|rats?|animals?|cells?|"
                        r"cell lines?|culture|vitro|vivo|plants?|seedlings?|[A-Z]\w+\s+\w+)\b", tail, maxsplit=1, flags=_I)[0]
        endpoint_phrase = re.sub(r"[?.!]+\s*$", "", tail).strip()
    # named exposure / intervention: content words that are not question words, disease words, endpoints or organisms
    removed = q
    if spec:
        removed = spec["rx_words"].sub(" ", removed)
        removed = spec["rx_acr"].sub(" ", removed)
    for _, rx, _ in FAMILY_RX:
        removed = rx.sub(" ", removed)
    removed = re.sub(r"\b(?:breast|cancer|carcinoma|tumou?rs?|disease|disorder|syndrome|in humans|in patients|in mice|in rats)\b",
                     " ", removed, flags=_I)
    # a Latin binomial is the organism studied, both words of it ('Saccharomyces cerevisiae', 'Danio rerio'):
    # without this the species epithet alone looks like a drug name
    removed = re.sub(r"\b[A-Z][a-z]{2,}\s+[a-z]{3,}(?:\s+(?:subsp\.|var\.)\s+[a-z]{3,})?\b(?=\W|$)",
                     lambda m: " " if _BINOMIAL_OK.search(m.group(0)) else m.group(0), removed)
    from study_classifier import ANIMAL, HUMAN_PEOPLE, MICROBE, PLANT
    for rx in (ANIMAL, HUMAN_PEOPLE, MICROBE, PLANT):      # organisms are the population, not the exposure
        removed = rx.sub(" ", removed)
    # adjacent content words form ONE name ('vitamin D', 'green tea', 'drought stress', 'vitamin B12'); a short token
    # (D, B12, C) is kept only as part of a name, never alone
    exposures, run, last_end = [], [], -2
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-]*", removed):
        w = m.group(0)
        content = w.lower() not in _Q_STOP and (len(w) >= 3 and re.match(r"[A-Za-z]", w) or (run and len(w) <= 3))
        if content and (not run or removed[last_end:m.start()] == " "):
            run.append(w)
        else:
            if run:
                exposures.append(" ".join(run))
            run = [w] if content and len(w) >= 3 and re.match(r"[A-Za-z]", w) else []
        last_end = m.end()
    if run:
        exposures.append(" ".join(run))
    organism = None
    if re.search(r"\b(?:in|among) (?:humans?|patients|people|women|men|children|adults|participants)\b", q, _I):
        organism = "human"
    elif re.search(r"\bin (?:mice|mouse|rats?|animals?|dogs?|cattle|buffal\w+)\b", q, _I):
        organism = "animal"
    from study_classifier import ANIMAL as _ANI, MICROBE as _MIC, PLANT as _PLA
    from study_classifier import HUMAN_PEOPLE
    # which kind of biology the question is about: clinical screening rules (treatment / risk vocabulary, the human
    # evidence ladder) apply to human medicine only - a plant or microbe question is answered by papers that never
    # say 'therapy' or 'risk factor'
    if _PLA.search(q) or re.search(r"\b(?:auxins?|gibberellins?|cytokinins?|phytohormones?|abscisic|brassinosteroids?|"
                                   r"roots?(?! canal| causes?| mean)|shoots?|leaf|leaves|flowers?|flowering|pollen|seeds?|photosynth\w*|stomat\w*|"
                                   r"chloroplasts?|xylem|phloem|\w*tropism)\b", q, _I) and not disease:
        domain = "plant_biology"
    elif _MIC.search(q) and _LAB_GENETICS.search(q) and not HUMAN_PEOPLE.search(q):
        # 'CRISPRi knockdown of inhA in Mycobacterium tuberculosis' is microbiology, not tuberculosis medicine: the
        # disease lexicon matches the organism's name, but the question is about the microbe in culture
        domain = "microbiology"
    elif disease or organism == "human":
        domain = "human_medicine"
    elif _MIC.search(q):
        domain = "microbiology"
    elif _ANI.search(q):
        domain = "animal_biology"
    else:
        domain = "general_biology"
    return {"question": q, "normalisation": norm, "domain": domain, "intent": intent, "intents": intents, "disease": disease,
            "disease_spec": spec, "heterogeneous": bool(spec and spec.get("heterogeneous")),
            "question_subtypes": q_subtypes, "endpoints": endpoints,
            "endpoint_phrase": endpoint_phrase,
            "exposures": _drop_endpoint_words([e for e in (_clean_exposure(x) for x in exposures)
                                               if e and e.lower() not in CLASS_WORDS], endpoint_phrase)[:3],
            "exposure_classes": [e for e in exposures if e.lower() in CLASS_WORDS][:3],
            "organism_restriction": organism, "asks_cure": info["asks_cure"], "disease_terms": terms}


# v25: 'riociguat compared' (with placebo) and 'placebo' were returned as exposures, 'lifespan' as an exposure of
# 'how does rapamycin extend lifespan'. A comparator word ends the exposure, a comparator is never an exposure, and a
# single outcome word is the endpoint.
_COMPARATOR_WORD = re.compile(r"\s+(?:compared|comparison|versus|vs|relative|than|against)\b.*$", re.I)
_NOT_EXPOSURE = {"placebo", "sham", "vehicle", "control", "controls", "usual care", "standard care", "baseline",
                 "lifespan", "longevity", "survival", "mortality", "healthspan", "aging", "ageing", "growth", "yield"}


def _clean_exposure(e: str) -> str:
    e = _COMPARATOR_WORD.sub("", e or "").strip()
    return "" if e.lower() in _NOT_EXPOSURE else e


# --------------------------------------------------------------------------- disease / organ detection
def disease_mentions(text: str, frame: Dict[str, Any]) -> Dict[str, Any]:
    """Does this text concern the question's disease (or a subtype), a related-but-distinct condition,
    or a different disease/organ system?"""
    t = text or ""
    spec = frame.get("disease_spec")
    out = {"match": False, "subtypes": [], "molecular": [], "related": [], "other": []}
    if spec:
        out["match"] = bool(spec["rx_words"].search(t) or spec["rx_acr"].search(t))
        out["subtypes"] = [n for n, w, a in spec.get("subtypes", []) if re.search(w, t, _I) or re.search(a, t)]
        out["molecular"] = [n for n, rx in spec.get("molecular", []) if re.search(rx, t, _I)]
        out["related"] = [n for n, rx in spec.get("related", []) if re.search(rx, t, _I)]
        if out["subtypes"] or out["molecular"] and out["match"]:
            out["match"] = True
    # mentions of the question's own disease are removed first, so 'diabetic' is never 'other (metabolic) disease'
    rest = spec["rx_words"].sub(" ", t) if spec else t
    for n, rx in OTHER_RX:
        if rx.search(rest):
            out["other"].append(n)
    return out


def organs_in(text: str) -> List[str]:
    return [n for n, rx in ORGAN_RX if rx.search(text or "")]


# --------------------------------------------------------------------------- identity record
_QUESTION_WORDS = re.compile(r"^(?:what|which|who|why|how|when|where|is|are|does|do|can|the|a|an|of|causes?|"
                             r"experimental|treatment|intervention|factor|unknown|unspecified|none|n/?a)$", _I)
DISEASE_OUTCOME = (r"(?:risk|incidence|odds|hazard|development|onset|occurrence|diagnos\w*|cases?|rates?|excess|mortality)\s+"
                   r"(?:of\s+|for\s+)?(?:\w+\s+){0,3}(?:{d})|(?:{d})\s+(?:\w+\s+){0,1}(?:risk|incidence|development|onset|cases|mortality)|"
                   r"(?:developed|develop|diagnosed with|progress\w* to|transform\w* (?:to|into)|induc\w*|caus\w*|cause of|"
                   r"associated with (?:an? )?(?:increased |higher |elevated )?(?:risk of )?)\s+(?:\w+\s+){0,2}(?:{d})|"
                   r"leuk(?:a)?emogen\w*|carcinogen\w*|tumou?rigen(?:esis|icity)\b")   # 'tumorigenic genes' is a gene label
# a new clinical event in a followed population ('incident heart failure', 'developed cirrhosis', 'new cases of ...')
_INCIDENT_CLINICAL = re.compile(r"\b(?:incident|incidence of|new cases of|newly diagnosed|developed|onset of|"
                                r"progression to)\s+(?:\w+\s+){0,3}?"
                                r"(?:\w*(?:pathy|opathy|itis|osis|aemia|emia|oma|plegia|trophy|iasis)|disease|syndrome|"
                                r"failure|infarction|stroke|cancer|carcinoma|fibrosis|cirrhosis|diabetes|dementia|"
                                r"blindness|deafness|death|mortality)\b", _I)
INTERMEDIATE = re.compile(r"\b(?:chromosom\w* (?:aberration|damage|abnormalit|break|loss|gain)\w*|aneuploid\w*|"
                          r"DNA (?:damage|adducts?|strand breaks?|lesions?)|genotoxic\w*|micronucle\w*|mutation\w*|mutagen\w*|"
                          r"sister chromatid|translocation\w*|clonal h(?:a)?ematopoiesis|cytogenetic\w*|oxidative DNA|8-oxo\w*|"
                          r"telomer\w*|epigenetic|methylation)\b", _I)
CAUSAL_WORDS = re.compile(r"(?<![\w-])(?:caus\w*|led to|leads to|lead to|resulted in|results in|responsible for|due to|"
                          r"attributable to|drives?|driven by|triggers?|triggered|induc\w*)\b", _I)   # 'HMGB1-induced' is a label, not causal wording
ASSOC_WORDS = re.compile(r"(?<![\w-])(?:associat\w*|correlat\w*|linked (?:to|with)|odds ratio|hazard ratio|relative risk|"
                         r"heightened risk|increased risk|higher risk|elevated risk)\b", _I)
_SPECIES = [("human", r"\b(?:patients?|humans?|workers?|participants?|women|men|children|adults?|individuals|people|subjects|"
                      r"donors?|cohorts?|homo sapiens)\b"),
            ("mouse", r"\b(?:mice|mouse|murine)\b"), ("rat", r"\brats?\b"), ("zebrafish", r"\bzebrafish\b"),
            ("dog", r"\b(?:dogs?|canine)\b"), ("cattle/buffalo", r"\b(?:cattle|cows?|bovine|buffal\w+|bubalus)\b"),
            ("non-human primate", r"\b(?:primates?|macaques?|monkeys?)\b")]
_CELL_NAMES = re.compile(r"\b(?:K562|HL-?60|Jurkat|MOLM-?13|MV4-?11|NB4|U-?937|KG-?1|Kasumi-?1|OCI-AML\d|NALM-?6|REH|RS4;11|"
                         r"MEC-?1|CCRF-CEM|THP-1|Ba/F3|MCF-?7|MDA-MB-\d+|T-?47D|HeLa|HEK-?293\w*|A549|HepG2|HCT-?116|"
                         r"SK-?BR-?3|BT-?474|BT-?549|ZR-?75-?1|Hs ?578T|LNCaP|PC-?3|DU-?145|SH-?SY5Y|U-?87(?: ?MG)?|"
                         r"Caco-?2|HT-?29|SW-?480|SW-?620|Huh-?7|U-?2 ?OS|A-?375|HaCaT|HUVECs?|H1299|PANC-?1|"
                         r"PBMCs?|mesenchymal (?:stromal|stem) cells|MSCs?|lymphocytes|blasts|bone marrow|cardiomyocytes|"
                         r"keratinocytes|melanocytes|hepatocytes|fibroblasts|endothelial cells|vascular smooth muscle cells|"
                         r"reticulocytes|neutrophils|macrophages|T cells|B cells|stem cells|progenitor cells)\b", _I)


def is_question_word_subject(subject: Optional[str]) -> bool:
    """'What Are', 'What are the', 'Causes' - an extraction failure, not an intervention."""
    words = re.findall(r"[A-Za-z]+", subject or "")
    return not words or all(_QUESTION_WORDS.match(w) for w in words)


def expand_abbreviations(text: str, abbr: Optional[Dict[str, str]]) -> str:
    """'reduced BC incidence' -> 'reduced breast cancer (BC) incidence' when the paper defines BC as breast cancer."""
    for a, full in (abbr or {}).items():
        if len(a) >= 2 and full.lower() != a.lower():
            text = re.sub(rf"(?<![\w-]){re.escape(a)}(?![\w-])", f"{full} ({a})", text or "")
    return text or ""


_XENOGRAFT = re.compile(r"\b(?:xenograft\w*|(?:tumou?r|cancer|cell)s? (?:were |was )?(?:trans)?(?:planted|implanted|inoculated|injected)|"
                        r"transplant(?:ed|ation)? tumou?r|implanted|nude mice|athymic|SCID|NSG mice|NOD[/-]SCID|"
                        r"tumou?r-bearing|PDX|patient-derived)\b", _I)
_MOUSE_LINES = re.compile(r"\b(?:4T1|B16(?:-?F10)?|RAW ?264\.7|NIH[- ]?3T3|LLC|CT26|MC38|E0771|EMT6)\b", _I)
_ANIMAL_SPECIES = [s for s in _SPECIES if s[0] != "human"]


def claim_species(model: Optional[str], quote: str, context: str, title: str, fallback: str) -> str:
    """Species of THIS result. A tumour grown in mice from human cells is a mouse experiment ('mouse xenograft
    (human tumour cells)'), never 'human'; a cell-line result is 'human cells (in vitro)'."""
    local = f"{quote} {context}"
    if model == "animal":
        sp = next((n for n, rx in _ANIMAL_SPECIES if re.search(rx, quote, _I)), None) or \
            next((n for n, rx in _ANIMAL_SPECIES if re.search(rx, context, _I)), None) or \
            next((n for n, rx in _ANIMAL_SPECIES if re.search(rx, title or "", _I)), None)
        xeno = _XENOGRAFT.search(local) or _XENOGRAFT.search(title or "")
        human_cells = bool(_CELL_NAMES.search(local) or re.search(r"\bhuman\b[^.;]{0,40}\b(?:cells?|tumou?rs?|cancer)", local, _I))
        if xeno or (human_cells and sp in (None, "mouse", "rat")):
            return f"{sp or 'mouse'} xenograft (human tumour cells)" if human_cells or xeno else (sp or "mouse")
        return sp or "animal (species not stated)"
    if model in ("review_meta", None, "unclear", "unknown"):
        # a review's report: the system named in the quote ('co-treatment of BC cells with ...' = cells)
        if _CELL_NAMES.search(quote) or re.search(r"\bcells?\b|\bin vitro\b|cell lines?", quote, _I):
            model = "cell_culture"
        elif any(re.search(rx, quote, _I) for _, rx in _ANIMAL_SPECIES):
            model = "animal"
        elif re.search(_SPECIES[0][1], quote, _I):
            return "human"
        else:
            return fallback if not re.search(r"\bcells?\b", fallback or "", _I) else "cells (in vitro)"
    if model in ("cell_culture", "organoid_ex_vivo"):
        mouse = _MOUSE_LINES.search(local) or re.search(r"\b(?:murine|mouse|rat)\b[^.;]{0,30}\bcells?\b", local, _I)
        human = any(re.search(r"[A-Z0-9]", x.group(0)) and not _MOUSE_LINES.fullmatch(x.group(0))
                    for x in _CELL_NAMES.finditer(local)) or \
            re.search(r"\bhuman\b[^.;]{0,40}\bcells?\b", local, _I)
        if mouse and human:
            return "human and mouse cells (in vitro)"
        if mouse:
            return "mouse / rat cells (in vitro)"
        return "human cells (in vitro)" if (_CELL_NAMES.search(local) or re.search(r"\bhuman\b", local, _I)) \
            else "cells (in vitro; species not stated)"
    if model in ("human_clinical", "human_observational"):
        return "human"
    return fallback


def claim_type_label(model: Optional[str], paper_type: str, paper_label: Optional[str]) -> str:
    from study_classifier import EVIDENCE_TYPES
    if paper_type == "review_meta" or not model or model not in EVIDENCE_TYPES:
        return paper_label or paper_type
    return EVIDENCE_TYPES[model]


def _cell_line_subtype(model: Optional[str], text: str, paper_level: List[str]) -> List[str]:
    """v24.2: a cell / animal result carries its cell line's own subtype (MDA-MB-231 = triple-negative)."""
    try:
        from metadata_authority import constrain_subtype
        return constrain_subtype(model, text, paper_level)
    except Exception:
        return paper_level


def build_identity(claim: Dict[str, Any], audit: Dict[str, Any], cls: Dict[str, Any], title: str,
                   frame: Dict[str, Any], abstract: str = "", abbreviations: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    from clinical_context import comparator, follow_up_days
    from evidence_quality import parse_exposures
    from stats_parsing import parse_p_values
    quote = claim.get("quote") or ""
    context = audit.get("source_context") or ""
    local = f"{quote} {context[:700]}"
    # disease mentions read the quote with the paper's own abbreviations expanded ('BC' = breast cancer)
    q_dm = disease_mentions(expand_abbreviations(quote, abbreviations), frame)
    t_dm = disease_mentions(title, frame)
    c_dm = disease_mentions(expand_abbreviations(context, abbreviations), frame)
    a_dm = disease_mentions(abstract, frame)
    species = next((n for n, rx in _SPECIES if re.search(rx, quote, _I)), None) or \
        next((n for n, rx in _SPECIES if re.search(rx, context[:700], _I)), None) or (cls.get("organism") or "not stated")
    species = claim_species(audit.get("claim_model"), quote, context[:900], title, species)
    stats = parse_p_values(quote)
    ratio = re.findall(r"\b(?:(?-i:OR|HR|RR|SIR)|odds ratio|hazard ratio|relative risk)\s*[=:,]?\s*(\d+(?:\.\d+)?)", quote, _I)
    ci = re.search(r"9[05]\s*%\s*(?:CI|confidence interval)[^;)]*", quote, _I)
    dur = follow_up_days(local)
    hours = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:h|hours?|hrs?)\b", local, _I)
    causal_hits = sorted({m.group(0).lower() for m in CAUSAL_WORDS.finditer(f"{claim.get('statement') or ''} {quote}")})
    assoc_hits = sorted({m.group(0).lower() for m in ASSOC_WORDS.finditer(f"{claim.get('statement') or ''} {quote}")})
    etype = cls.get("evidence_type") or "unclear"
    origin = ("reported by a review (secondary literature) - not observed by this paper" if etype == "review_meta"
              else "this study's own result" if (audit.get("checks", {}).get("this_study", {}).get("ok") is not False)
              else "prior work cited by the authors")
    subj = claim.get("intervention") or claim.get("subject")
    return {
        "source_paper": {"paper_id": claim.get("paper_id"), "title": title},
        "original_experiment": origin,
        # the type of THIS result (a paper can report cell AND mouse experiments); the paper's own design kept beside it
        "evidence_type": claim_type_label(audit.get("claim_model"), etype, cls.get("label")),
        "paper_design": cls.get("label") or etype,
        "claim_model": audit.get("claim_model"),
        "species": species,
        "disease": (frame.get("disease") if (q_dm["match"] or t_dm["match"] or (c_dm["match"] and a_dm["match"])) else
                    ("; ".join(q_dm["related"] + q_dm["other"]) or "; ".join(t_dm["related"] + t_dm["other"]) or "not stated")),
        "disease_subtype": _cell_line_subtype(audit.get("claim_model"), f"{quote} {context[:700]} {title}",
                                              sorted(set(q_dm["subtypes"] + t_dm["subtypes"] + c_dm["subtypes"]))) or ["not specified"],
        "molecular_subtype": sorted(set(q_dm["molecular"] + c_dm["molecular"])),
        "cell_tissue": sorted({m.group(0) for m in _CELL_NAMES.finditer(local)}) or organs_in(quote) or ["not stated"],
        "organ_systems_in_quote": organs_in(quote),
        "intervention_exposure": subj or "not stated",
        "intervention_is_question_words": is_question_word_subject(subj),
        "comparator": comparator(local, etype),
        "dose": ", ".join(_own_doses(parse_exposures(local), local, subj)[:2]) or "not reported",
        "duration": (f"{dur:.0f} days" if dur else (f"{hours.group(1)} h" if hours else "not reported")),
        "outcome": claim.get("measurement") or claim.get("outcome") or "not stated",
        "outcome_family": audit.get("canonical_outcome"),
        "direction": audit.get("canonical_direction") or claim.get("effect_direction"),
        "statistical_evidence": ({"p_values": [p["raw"] if "raw" in p else p["value"] for p in stats],
                                  "ratios": ratio, "ci": ci.group(0) if ci else None}
                                 if (stats or ratio or ci) else "none reported in the quote"),
        "causal_language": {"causal": causal_hits, "associational": assoc_hits},
        "exact_source_passage": {"quote": quote, "section": audit.get("source_section")},
        # three separate roles (a diabetic breast cancer cohort is breast cancer evidence in a diabetic subgroup):
        "target_disease": frame.get("disease") or "not stated",
        # a subgroup / comorbidity describes PEOPLE: named in the quote, otherwise the paper's population; never for a
        # cell or animal result ('anti-diabetic medication' in a cell paper is not a diabetic subgroup)
        "comorbidity": ((comorbidities(quote, frame) or comorbidities(f"{title}. {abstract}", frame))
                        if (audit.get("claim_model") in ("human_clinical", "human_observational") or
                            (etype == "review_meta" and species == "human")) else []),
        "readout": audit.get("canonical_outcome") or claim.get("measurement") or "not stated",
        "_mentions": {"quote": q_dm, "title": t_dm, "context": c_dm, "abstract": a_dm,
                      "comorbid": comorbidities(f"{title}. {abstract}", frame)},
    }


def _own_doses(exposures: List[Dict[str, Any]], text: str, agent: Optional[str]) -> List[str]:
    """Doses of THIS claim's agent: 'IC50 ... 11.86 mM for APAP; 21.11 mM for MET' keeps 21.11 mM for metformin only."""
    from clinical_context import _DRUG_ABBR
    agent_low = (agent or "").lower()
    out = []
    for e in exposures:
        raw = e.get("raw") or ""
        m = re.search(re.escape(raw) + r"\s+(?:of\s+|for\s+)([A-Za-z][\w\-]*)", text)
        if m:
            who = m.group(1).lower()
            name = _DRUG_ABBR.get(who, who)
            if agent_low and not (name[:5] in agent_low or agent_low[:5] in name):
                continue                      # the dose of another agent named in the same sentence
        out.append(raw)
    return out


# biomarker / physiological states that describe a SUBGROUP of patients, never the disease studied
_STATE_RX = re.compile(r"(?<![\w-])(?:non-?diabet\w*|without (?:type 2 )?diabetes|(?:type [12] |pre-?)?diabet\w*|insulin[- ]resist\w*|"
                       r"hyperinsulin\w*|hyperglyc\w*|obes\w*|overweight|metabolic syndrome|hypertensi\w*|HIV[- ]positive|"
                       r"post-?menopausal|pre-?menopausal|smokers?)\b", _I)


def comorbidities(text: str, frame: Dict[str, Any]) -> List[str]:
    """Patient subgroup / comorbidity named TOGETHER with the question's disease in one sentence ('non-diabetic
    breast cancer patients with insulin resistance' -> ['non-diabetic', 'insulin resistance']). These qualify the
    population; they never replace the target disease."""
    spec = frame.get("disease_spec")
    if not spec:
        return []
    out: List[str] = []
    for sent in re.split(r"(?<=[.;!?])\s+", text or ""):
        if spec["rx_words"].search(sent) or spec["rx_acr"].search(sent):
            for m in _STATE_RX.finditer(sent):
                w = m.group(0).lower().replace("insulin resistant", "insulin resistance")
                if w not in out:
                    out.append(w)
    return out


# --------------------------------------------------------------------------- roles + the hard gate
ROLE_LABEL = {
    "disease_risk": "Disease occurrence / risk evidence",
    "intermediate": "Intermediate endpoint (link to the disease not shown)",
    "mechanistic": "Mechanistic context (not a demonstrated cause / not the question's outcome)",
    "direct": "Directly measures the question's outcome",
    "secondary_report": "Secondary report (review) - not independent evidence",
    "combination": "Combination regimen - the question's agent is not tested alone (context, not an answer)",
}
ANSWERING_ROLES = ("disease_risk", "direct")
DISEASE_LEVEL_FAMILIES = {"clinical_response", "patient_survival", "recurrence_incidence", "tumour_growth"}


def _disease_alt(frame: Dict[str, Any]) -> str:
    spec = frame.get("disease_spec") or {}
    words = spec.get("words") or re.escape(frame.get("disease") or "$^")
    return words


def claim_role(identity: Dict[str, Any], frame: Dict[str, Any], statement: str, quote: str, etype: str) -> str:
    if etype == "review_meta":
        return "secondary_report"
    role = _claim_role(identity, frame, statement, quote, etype)
    comps = identity.get("combination_components") or []
    exposures = [e.lower() for e in frame.get("exposures") or [] if e]
    if role in ANSWERING_ROLES and len(comps) >= 2 and exposures:
        # 'metformin and palbociclib synergistically inhibited proliferation' answers a question about the COMBINATION,
        # not about metformin: when the question asks about ONE agent of the regimen and does not name every component,
        # it is context only (a question naming no agent - 'Can leukemia be cured?' - is answered by any regimen)
        from clinical_context import _DRUG_ABBR
        names = [_DRUG_ABBR.get(c.lower(), c.lower()) for c in comps]
        qtext = " ".join([frame.get("question") or "", " ".join(exposures)]).lower()
        asked = any(e[:6] in n for e in exposures for n in names)
        if asked and not all(any(w[:6] in qtext for w in n.split() if len(w) >= 3) for n in names):
            return "combination"
    return role


# a two-word name is a species when its first word is a genus-shaped Latin word and the pair follows 'in' / is a
# known model organism: 'Saccharomyces cerevisiae', 'Danio rerio', 'Zea mays', 'Mycobacterium tuberculosis'
_BINOMIAL_OK = re.compile(r"^(?:[A-Z][a-z]+(?:us|um|a|is|ia|ii|yces|bacter|monas|coccus|bacillus|phila|ella|ium)"
                          r"|Danio|Zea|Mus|Rattus|Homo|Apis|Bos|Sus|Canis|Felis|Ovis|Capra|Equus|Gallus|Oryza|"
                          r"Xenopus|Drosophila|Caenorhabditis|Arabidopsis|Nicotiana|Solanum|Triticum|Hordeum|Glycine)\s",
                          re.UNICODE)
_ENDPOINT_STOP = set("""the a an of in on for to and or with by from at as into about than then is are was were be
does do did can could would should may might will shall what which who why when how no not its it their this that
these those any some all more less higher lower increase increased increases decrease decreased decreases reduce
reduced reduces improve improved improves change changes effect effects level levels rate rates amount amounts
measure measured measurement study studies cells cell patients patient mice mouse rats rat plants plant strains
strain humans human animals animal""".split())


def _drop_endpoint_words(exposures: List[str], endpoint_phrase: str) -> List[str]:
    """Words of the asked OUTCOME are not an exposure ('drought tolerance' in 'does silicon improve drought
    tolerance'). Never empties the list: a question with no other candidate keeps what it had."""
    if not endpoint_phrase or not exposures:
        return exposures
    out_toks = _endpoint_tokens(endpoint_phrase)
    kept = [e for e in exposures if not (_endpoint_tokens(e) and _endpoint_tokens(e) <= out_toks)]
    return kept or exposures


def _endpoint_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z\-]{2,}", (text or "").lower()) if w not in _ENDPOINT_STOP}


def _asks_this_endpoint(frame: Dict[str, Any], fam: str, identity: Dict[str, Any], text: str) -> bool:
    """Does the claim measure the endpoint the QUESTION names, in the question's own words?

    Used when the asked endpoint is not one of the built-in families. The question's endpoint words are whatever is
    left after removing the exposure, the organism and the question words; a claim is direct when its outcome shares
    a content word with them (two words for a multi-word endpoint, so 'water content' does not match 'cell content')."""
    q = frame.get("question") or ""
    if not q:
        return False
    q_toks = _endpoint_tokens(frame.get("endpoint_phrase") or q)
    for e in (frame.get("exposures") or []) + (frame.get("exposure_classes") or []):
        q_toks -= _endpoint_tokens(e)
    spec = frame.get("disease_spec") or {}
    if spec.get("rx_words"):
        q_toks -= _endpoint_tokens(spec["rx_words"].pattern.replace("\\w*", " ").replace("|", " "))
    for key in ("readout", "outcome_family"):
        pass
    out_toks = _endpoint_tokens(f"{fam.replace('other:', '')} {identity.get('readout') or ''}")
    shared = q_toks & out_toks
    if not shared:
        return False
    # one shared word is enough only when it is specific (>= 5 letters); otherwise two must match
    return len(shared) >= 2 or any(len(w) >= 5 for w in shared)


def _claim_role(identity: Dict[str, Any], frame: Dict[str, Any], statement: str, quote: str, etype: str) -> str:
    text = f"{statement} {quote}"
    intent = frame.get("intent")
    d_alt = _disease_alt(frame)
    disease_outcome = bool(frame.get("disease")) and bool(re.search(DISEASE_OUTCOME.replace("{d}", d_alt), text, _I))
    if intent == "etiology":
        if disease_outcome:
            return "disease_risk"
        # 'What causes <X>?' outside medicine ('what causes pectoral fin defects') names no disease: the question's own
        # endpoint words decide, exactly as for a treatment question
        if not frame.get("disease") and _asks_this_endpoint(frame, identity.get("outcome_family") or "", identity, text):
            return "disease_risk"
        if INTERMEDIATE.search(text):
            return "intermediate"
        if _INCIDENT_CLINICAL.search(text):
            # an incident clinical outcome in a cohort ('associated with incident cardiomyopathy') is disease-level
            # evidence, but the quote does not name the question's disease - a related outcome, not a demonstrated cause
            return "intermediate"
        return "mechanistic"
    if intent in ("treatment", "cure", "prognosis", "epidemiology"):
        fam = identity.get("outcome_family") or ""
        q_end = set(frame.get("endpoints") or [])
        related = {"proliferation", "tumour_growth"} if q_end & {"proliferation", "tumour_growth"} else set()
        if fam in q_end or fam in related or fam in DISEASE_LEVEL_FAMILIES or disease_outcome:
            return "direct"
        if not q_end and not fam.startswith("other") and fam not in ("expression", "signalling", "apoptosis", "viability",
                                                                         "inflammation"):
            return "direct"
        # the question's OWN endpoint words: outside clinical medicine the asked endpoint is usually not one of the
        # built-in families ('fluconazole sensitivity', 'pectoral fin outgrowth', 'relative water content'), and such a
        # question could never be answered directly. It is direct when the claim measures what the question names.
        if _asks_this_endpoint(frame, fam, identity, text):
            return "direct"
        return "mechanistic"
    return "direct"


def relevance_gate(identity: Dict[str, Any], frame: Dict[str, Any], statement: str = "", quote: str = "",
                   etype: str = "unclear") -> Dict[str, Any]:
    """HARD RULE: disease, model (cell/tissue system), intervention/exposure and outcome must match the question.
    Returns {'admitted', 'checks', 'reasons', 'role', 'role_label', 'answers_question'}."""
    m = identity["_mentions"]
    checks: Dict[str, Dict[str, Any]] = {}
    reasons: List[str] = []

    def mark(name, ok, detail):
        checks[name] = {"ok": ok, "detail": detail}
        if ok is False:
            reasons.append(f"{name}: {detail}")

    # 0. a real subject (the "What Are" extraction failure)
    if identity.get("intervention_is_question_words"):
        mark("subject", False, f"the extracted intervention/exposure '{identity.get('intervention_exposure')}' is made of "
                               f"question words - extraction failure, no subject identified")
    else:
        mark("subject", True, f"'{identity.get('intervention_exposure')}'")
    disease = frame.get("disease")
    # 1. disease
    if disease:
        qd, td, cd, ad = m["quote"], m["title"], m["context"], m["abstract"]
        if qd["match"]:
            mark("disease", True, f"{disease} named in the quote" + (f" (subtype: {', '.join(qd['subtypes'])})" if qd["subtypes"] else ""))
        elif (qd["other"] or qd["related"]) and (td["match"] or ad["match"]) and \
                (set(qd["other"]) <= set(td["other"]) or (set(qd["other"]) <= {"metabolic disease"} and m.get("comorbid"))) \
                and set(qd["related"]) <= {"prediabetes / insulin resistance"}:
            # 'breast cancer patients WITH type 2 diabetes' / 'non-diabetic women with insulin resistance': the
            # paper names the disease and the condition together, so the condition describes the study population
            # (a subgroup / comorbidity) - it is not a different disease
            mark("disease", True, f"{disease} is the study's disease; the quote's {', '.join(qd['other'] + qd['related'])} "
                                  f"describes the patients (subgroup / comorbidity: {', '.join(m.get('comorbid') or ['named in the title'])})")
        elif qd["other"] or qd["related"]:
            what = ", ".join(qd["related"] + qd["other"])
            mark("disease", False, f"the quote is about {what}, not {disease}"
                                   + (" (a related but distinct condition)" if qd["related"] and not qd["other"] else ""))
        elif td["match"]:
            mark("disease", True, f"{disease} named in the paper title")
        elif cd["match"] and ad["match"]:
            mark("disease", True, f"{disease} named in the quoted passage and the abstract")
        else:
            what = ", ".join(td["related"] + td["other"] + cd["related"] + cd["other"])
            mark("disease", False, (f"the study concerns {what}, not {disease}" if what
                                    else f"neither the quote, its passage nor the title concerns {disease}"))
    else:
        checks["disease"] = {"ok": None, "detail": "the question names no disease"}
    # 2. model / tissue system
    organ = (frame.get("disease_spec") or {}).get("organ")
    quote_organs = identity.get("organ_systems_in_quote") or []
    if organ and quote_organs and organ not in quote_organs and not m["quote"]["match"] and m["title"]["match"] \
            and identity.get("claim_model") in ("animal", "experimental_disease_model", "human_clinical", "human_observational"):
        # immune / blood / host read-outs measured in a living model OF the question's disease (e.g. splenic NKT
        # cells in 4T1 breast-tumour-bearing mice) are host-response context - not a different disease
        checks["model"] = {"ok": True, "detail": f"{'/'.join(quote_organs)} read-out in a {disease} model (host response - context)"}
    elif organ and quote_organs and organ not in quote_organs and not m["quote"]["match"]:
        mark("model", False, f"the quote describes a {'/'.join(quote_organs)} system, which is not a model of {disease} ({organ})")
    else:
        checks["model"] = {"ok": True, "detail": (identity.get("claim_model") or "?").replace("_", " ")
                           + (f"; cell/tissue: {', '.join(identity.get('cell_tissue') or [])}" if identity.get("cell_tissue") else "")}
    # 3. intervention / exposure named in the question
    exposures = (frame.get("exposures") or []) if disease else []   # open biology questions are not gated on exposure
    if exposures and not identity.get("intervention_is_question_words"):
        hay = f"{identity.get('intervention_exposure') or ''} {quote}".lower()
        hit = [e for e in exposures if e.lower()[:max(4, min(len(e), 6))] in hay]
        iv_low = (identity.get("intervention_exposure") or "").lower()
        from clinical_context import _DRUG_ABBR
        iv_full = " ".join(_DRUG_ABBR.get(w, w) for w in re.findall(r"[\w\-]+", iv_low))
        background = [e for e in hit if e.lower()[:6] not in iv_full and re.search(
            rf"\b(?:in the presence of|presence of|pre-?treated with|on top of|in addition to)\s+(?:the\s+)?{re.escape(e.lower()[:6])}\w*"
            rf"|\bin\s+{re.escape(e.lower()[:6])}\w*[- ]treated\b", quote.lower())]
        formulation = re.search(r"\bnano\w*|liposom\w*|conjugat\w*|[\w-]+-loaded\b|encapsulat\w*|hydrogel|micelle\w*|"
                                r"derivatives?\b|analogu?es?\b|prodrug|-met\b|\bmet-", iv_low)
        if hit and formulation and not re.search(r"\bnano\w*|liposom\w*|conjugat\w*|derivativ\w*|analog\w*|prodrug",
                                                 " ".join(frame.get("exposures") or []) + " " + (frame.get("question") or ""), re.I):
            # 'HA-GO-Met nanoparticles' / 'metformin derivatives' are other agents built from the drug, not the drug itself
            mark("exposure", False, f"studies '{identity.get('intervention_exposure')}' - a formulation / derivative, not "
                                    f"{', '.join(hit)} itself")
        elif hit and background and len(background) == len(hit) and iv_low and \
                not identity.get("intervention_is_question_words") and len(iv_low) >= 3:
            # 'In the presence of metformin, treating with SOD and catalase improved cell viability': SOD is the agent
            # tested (a rescue experiment); metformin is the background condition, not the cause of this result
            mark("exposure", False, f"studies '{identity.get('intervention_exposure')}' with {', '.join(hit)} as the background "
                                    f"condition (rescue / modifier experiment) - not the effect of {', '.join(hit)} itself")
        elif hit:
            mark("exposure", True, f"matches the question's {', '.join(hit)}")
        else:
            mark("exposure", False, f"studies '{identity.get('intervention_exposure')}', not the question's {', '.join(exposures)}")
    else:
        checks["exposure"] = {"ok": None, "detail": "the question does not name one exposure/intervention (open question)"}
    # 4. outcome -> role (a mechanism is not a cause; a mechanistic outcome cannot answer the question)
    role = claim_role(identity, frame, statement, quote, etype)
    checks["outcome"] = {"ok": True if role in ANSWERING_ROLES else None,
                         "detail": ROLE_LABEL[role] + ("" if role in ANSWERING_ROLES else " - kept as context, cannot answer the question")}
    admitted = not reasons
    return {"admitted": admitted, "checks": checks, "reasons": reasons, "role": role, "role_label": ROLE_LABEL[role],
            "answers_question": admitted and role in ANSWERING_ROLES}


def paper_on_question(title: str, abstract: str, frame: Dict[str, Any]) -> Dict[str, Any]:
    """Paper-level relevance: off-question papers are not counted in the evidence buckets."""
    if not frame.get("disease"):
        return {"on_question": True, "detail": "no disease in the question"}
    t, a = disease_mentions(title, frame), disease_mentions(abstract, frame)
    if t["match"] or a["match"]:
        return {"on_question": True, "detail": f"{frame['disease']} named in the " + ("title" if t["match"] else "abstract"),
                "subtypes": sorted(set(t["subtypes"] + a["subtypes"]))}
    what = ", ".join(sorted(set(t["related"] + t["other"] + a["related"] + a["other"])))
    return {"on_question": False, "detail": (f"about {what}, not {frame['disease']}" if what
                                             else f"title and abstract do not concern {frame['disease']}")}


def subtype_coverage(rows: List[Dict[str, Any]], frame: Dict[str, Any]) -> Dict[str, Any]:
    if not frame.get("disease"):
        return {}
    cov: Dict[str, List[str]] = {}
    for r in rows:
        for s in r.get("disease_subtypes") or ["not specified"]:
            cov.setdefault(s, []).append(r["ref"])
    het = frame.get("heterogeneous")
    spec = frame.get("disease_spec") or {}
    return {"disease": frame["disease"], "heterogeneous": het, "coverage": cov,
            "known_subtypes": [n for n, _, _ in spec.get("subtypes", [])],
            "statement": (f"{frame['disease'].capitalize()} is a heterogeneous group of diseases "
                          f"({', '.join(n for n, _, _ in spec.get('subtypes', []))}); each finding applies only to the subtype "
                          f"studied and findings are never pooled across subtypes." if het else "")}


def question_outcome(question_or_frame: Any) -> Optional[str]:
    """The outcome a question asks about, as text: the first canonical endpoint family ('proliferation'), else the
    endpoint phrase ('breast cancer'), else the disease. (v26: one accessor, so no caller reads a missing key.)"""
    fr = question_or_frame if isinstance(question_or_frame, dict) else question_frame(question_or_frame or "")
    eps = fr.get("endpoints") or []
    if eps:
        return str(eps[0]).replace("_", " ")
    return fr.get("endpoint_phrase") or fr.get("disease") or None
