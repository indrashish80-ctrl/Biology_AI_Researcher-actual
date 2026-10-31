"""
mesh_synonyms.py  (v25)
=======================
Biomedical synonym expansion for keyword search, so that a paper that says "dimethylbiguanide" or "breast neoplasms"
is found by a question that says "metformin" and "breast cancer".

Two sources, used together:
  1. A curated table of MeSH-style concept groups (built in; covers common diseases, drugs, outcomes, designs and
     model systems) plus British / American spelling variants (tumour/tumor, oedema/edema, -ise/-ize, haem/hem ...).
  2. OPTIONAL: the full NLM MeSH descriptor file. Download desc<YEAR>.xml (or the ASCII d<YEAR>.bin) from
     https://www.nlm.nih.gov/databases/download/mesh.html and set BIOSYN_MESH_FILE=/path/to/it. Every descriptor's
     entry terms then become synonyms of its heading (about 30,000 headings, 300,000 entry terms). The parsed table is
     cached next to the file the first time (a few seconds), so later runs load instantly.

    expand("metformin")           -> {"metformin", "dimethylbiguanide", "glucophage", ...}
    expand_query("does metformin reduce tumour growth") -> [concept groups] for an OR-of-ANDs query
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

MESH_VERSION = "1.0"

# ------------------------------------------------------------------------------------------- curated concept groups
_GROUPS: List[List[str]] = [
    # --- drugs / interventions
    ["metformin", "dimethylbiguanide", "glucophage", "biguanide"],
    ["aspirin", "acetylsalicylic acid", "asa"],
    ["paracetamol", "acetaminophen"],
    ["ibuprofen", "nsaid", "non-steroidal anti-inflammatory"],
    ["statin", "statins", "hmg-coa reductase inhibitor", "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin"],
    ["rapamycin", "sirolimus", "mtor inhibitor"],
    ["tamoxifen", "selective estrogen receptor modulator", "serm"],
    ["sglt2 inhibitor", "sodium-glucose cotransporter 2 inhibitor", "gliflozin", "dapagliflozin", "empagliflozin", "canagliflozin"],
    ["glp-1 receptor agonist", "glucagon-like peptide-1", "liraglutide", "semaglutide", "exenatide"],
    ["insulin", "insulin analogue", "insulin glargine"],
    ["vitamin d", "cholecalciferol", "calcitriol", "25-hydroxyvitamin d", "ergocalciferol"],
    ["omega-3", "fish oil", "n-3 fatty acids", "eicosapentaenoic acid", "docosahexaenoic acid"],
    ["corticosteroid", "glucocorticoid", "steroid", "prednisolone", "prednisone", "dexamethasone", "betamethasone"],
    ["antibiotic", "antibacterial", "antimicrobial"],
    ["probiotic", "lactobacillus", "bifidobacterium"],
    ["chemotherapy", "antineoplastic", "cytotoxic therapy"],
    ["immunotherapy", "immune checkpoint inhibitor", "pd-1 inhibitor", "pd-l1 inhibitor", "anti-pd-1"],
    ["radiotherapy", "radiation therapy", "irradiation"],
    ["phototherapy", "narrowband uvb", "narrow band uvb", "narrow-band uvb", "nb-uvb", "nbuvb", "ultraviolet therapy", "puva"],
    ["jak inhibitor", "janus kinase inhibitor", "tofacitinib", "ruxolitinib", "baricitinib", "upadacitinib"],
    ["tnf inhibitor", "anti-tnf", "tumor necrosis factor inhibitor", "infliximab", "adalimumab", "etanercept"],
    ["proton pump inhibitor", "ppi", "omeprazole", "esomeprazole", "pantoprazole"],
    ["ace inhibitor", "angiotensin-converting enzyme inhibitor", "enalapril", "captopril", "lisinopril", "ramipril"],
    ["angiotensin receptor blocker", "arb", "sartan", "losartan", "olmesartan", "valsartan"],
    ["beta blocker", "beta-adrenergic blocker", "metoprolol", "propranolol", "bisoprolol", "carvedilol"],
    ["anticoagulant", "warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran"],
    ["exercise", "physical activity", "exercise training", "aerobic training", "resistance training"],
    ["diet", "dietary intervention", "nutrition", "caloric restriction", "calorie restriction"],
    ["placebo", "sham", "vehicle", "dummy"],
    ["vaccine", "vaccination", "immunization", "immunisation"],
    ["crispr", "gene editing", "cas9"],
    ["knockdown", "sirna", "shrna", "rna interference", "gene silencing"],
    ["knockout", "gene deletion", "null mutant", "-/-"],
    ["overexpression", "over-expression", "transgenic expression"],
    # --- diseases
    ["breast cancer", "breast neoplasms", "breast carcinoma", "mammary carcinoma", "breast tumour", "breast tumor"],
    ["lung cancer", "lung neoplasms", "non-small cell lung cancer", "nsclc", "lung carcinoma"],
    ["colorectal cancer", "colorectal neoplasms", "colon cancer", "rectal cancer", "crc"],
    ["prostate cancer", "prostatic neoplasms", "prostate carcinoma"],
    ["pancreatic cancer", "pancreatic neoplasms", "pancreatic ductal adenocarcinoma", "pdac"],
    ["hepatocellular carcinoma", "liver cancer", "hcc", "liver neoplasms"],
    ["leukemia", "leukaemia", "aml", "acute myeloid leukemia", "all", "acute lymphoblastic leukemia", "cll"],
    ["lymphoma", "hodgkin", "non-hodgkin lymphoma"],
    ["melanoma", "cutaneous melanoma"],
    ["glioblastoma", "glioma", "gbm", "brain tumour", "brain tumor"],
    ["cancer", "neoplasm", "neoplasms", "tumour", "tumor", "carcinoma", "malignancy", "oncology"],
    ["type 2 diabetes", "type 2 diabetes mellitus", "t2dm", "non-insulin-dependent diabetes", "niddm"],
    ["type 1 diabetes", "type 1 diabetes mellitus", "t1dm", "insulin-dependent diabetes"],
    ["diabetes", "diabetes mellitus", "diabetic", "hyperglycaemia", "hyperglycemia"],
    ["obesity", "obese", "overweight", "adiposity", "high body mass index"],
    ["hypertension", "high blood pressure", "elevated blood pressure"],
    ["myocardial infarction", "heart attack", "acute coronary syndrome", "ami"],
    ["heart failure", "cardiac failure", "congestive heart failure", "hfref", "hfpef"],
    ["stroke", "cerebrovascular accident", "cerebral infarction", "ischaemic stroke", "ischemic stroke"],
    ["atherosclerosis", "arteriosclerosis", "atherosclerotic plaque", "carotid intima-media thickness", "carotid imt"],
    ["alzheimer disease", "alzheimer's disease", "alzheimers", "dementia", "ad"],
    ["parkinson disease", "parkinson's disease", "parkinsonism"],
    ["multiple sclerosis", "ms", "relapsing multiple sclerosis", "demyelinating disease"],
    ["depression", "depressive disorder", "major depressive disorder", "mdd", "depressive symptoms"],
    ["anxiety", "anxiety disorder", "generalized anxiety disorder", "gad"],
    ["schizophrenia", "psychosis", "psychotic disorder"],
    ["bipolar disorder", "bipolar depression", "manic-depressive"],
    ["asthma", "bronchial asthma", "reactive airway disease"],
    ["copd", "chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"],
    ["pneumonia", "lower respiratory tract infection", "lung infection"],
    ["tuberculosis", "mycobacterium tuberculosis", "tb"],
    ["hiv", "hiv-1", "human immunodeficiency virus", "aids"],
    ["covid-19", "sars-cov-2", "coronavirus disease 2019", "2019-ncov"],
    ["influenza", "flu", "influenza virus"],
    ["malaria", "plasmodium falciparum", "plasmodium"],
    ["sepsis", "septic shock", "bacteraemia", "bacteremia"],
    ["rheumatoid arthritis", "ra", "inflammatory arthritis"],
    ["osteoarthritis", "degenerative joint disease", "knee osteoarthritis"],
    ["osteoporosis", "bone loss", "low bone mineral density", "osteopenia"],
    ["psoriasis", "psoriatic"],
    ["vitiligo", "leukoderma", "depigmentation"],
    ["atopic dermatitis", "eczema", "atopic eczema"],
    ["inflammatory bowel disease", "ibd", "crohn disease", "crohn's disease", "ulcerative colitis"],
    ["chronic kidney disease", "ckd", "renal insufficiency", "kidney failure", "end-stage renal disease"],
    ["nonalcoholic fatty liver disease", "nafld", "masld", "fatty liver", "nash", "steatohepatitis"],
    ["polycystic ovary syndrome", "pcos", "polycystic ovarian syndrome"],
    ["migraine", "migraine headache"],
    ["pain", "analgesia", "nociception", "painful"],
    ["smoking cessation", "quit smoking", "tobacco cessation", "smoking abstinence"],
    ["pregnancy", "pregnant", "gestation", "antenatal", "prenatal"],
    ["preterm birth", "premature birth", "preterm delivery"],
    # --- outcomes / measurements
    ["proliferation", "cell proliferation", "cell growth", "brdu incorporation", "edu incorporation"],
    ["ki-67", "ki67", "mib-1", "ki-67 index", "ki-67 labelling index", "ki-67 labeling index"],
    ["apoptosis", "programmed cell death", "caspase-3", "tunel", "cell death"],
    ["cell viability", "viability", "mtt assay", "cck-8", "cytotoxicity"],
    ["tumour growth", "tumor growth", "tumour volume", "tumor volume", "tumour burden", "tumor burden", "xenograft growth"],
    ["metastasis", "metastases", "metastatic spread", "invasion", "migration"],
    ["survival", "overall survival", "mortality", "death", "os"],
    ["progression-free survival", "pfs", "disease-free survival", "dfs", "event-free survival"],
    ["recurrence", "relapse", "recurrent disease"],
    ["inflammation", "inflammatory", "cytokine", "il-6", "interleukin-6", "tnf-alpha", "crp", "c-reactive protein"],
    ["oxidative stress", "reactive oxygen species", "ros", "lipid peroxidation", "malondialdehyde"],
    ["insulin resistance", "homa-ir", "insulin sensitivity"],
    ["hba1c", "glycated haemoglobin", "glycated hemoglobin", "haemoglobin a1c", "hemoglobin a1c", "a1c"],
    ["blood glucose", "fasting glucose", "plasma glucose", "glycaemia", "glycemia", "fasting plasma glucose"],
    ["body weight", "weight", "weight gain", "weight loss", "bmi", "body mass index"],
    ["blood pressure", "systolic blood pressure", "diastolic blood pressure", "sbp", "dbp"],
    ["cholesterol", "ldl cholesterol", "hdl cholesterol", "lipids", "lipid profile", "triglycerides"],
    ["quality of life", "qol", "hrqol", "health-related quality of life"],
    ["adverse events", "adverse effects", "side effects", "toxicity", "adverse reactions", "safety"],
    ["cognition", "cognitive function", "cognitive performance", "memory", "mmse"],
    ["sleep", "sleep quality", "insomnia", "sleep efficiency"],
    ["bone mineral density", "bmd", "bone density"],
    ["gene expression", "mrna expression", "transcription", "expression"],
    ["angiogenesis", "neovascularization", "neovascularisation", "vegf"],
    ["autophagy", "lc3", "autophagic flux"],
    ["fibrosis", "collagen deposition", "fibrotic"],
    ["biofilm", "biofilm formation"],
    ["drought tolerance", "drought stress", "water deficit", "drought"],
    ["yield", "grain yield", "crop yield"],
    # --- designs / models
    ["randomized controlled trial", "randomised controlled trial", "rct", "randomized trial", "randomised trial",
     "randomly assigned", "randomly allocated"],
    ["meta-analysis", "meta analysis", "systematic review", "pooled analysis"],
    ["cohort study", "prospective cohort", "retrospective cohort", "longitudinal study"],
    ["case-control study", "case control"],
    ["mice", "mouse", "murine", "mus musculus"],
    ["rats", "rat", "rattus"],
    ["zebrafish", "danio rerio"],
    ["drosophila", "fruit fly", "drosophila melanogaster"],
    ["yeast", "saccharomyces cerevisiae", "s. cerevisiae"],
    ["arabidopsis", "arabidopsis thaliana"],
    ["e. coli", "escherichia coli"],
    ["xenograft", "patient-derived xenograft", "pdx", "tumour-bearing mice"],
    ["cell line", "cell lines", "cultured cells", "in vitro"],
    ["organoid", "organoids", "3d culture", "spheroid"],
]

# British / American spelling pairs, applied to every word
_SPELL = [("tumour", "tumor"), ("oedema", "edema"), ("oestrogen", "estrogen"), ("oesophag", "esophag"),
          ("haem", "hem"), ("anaem", "anem"), ("leukaem", "leukem"), ("paediatric", "pediatric"), ("foetal", "fetal"),
          ("diarrhoea", "diarrhea"), ("randomis", "randomiz"), ("organis", "organiz"), ("characteris", "characteriz"),
          ("behaviour", "behavior"), ("colour", "color"), ("fibre", "fiber"), ("centre", "center"),
          ("ischaem", "ischem"), ("glycaem", "glycem"), ("lipaem", "lipem"), ("orthopaed", "orthoped"),
          ("anaesth", "anesth"), ("haematolog", "hematolog"), ("gynaecolog", "gynecolog"), ("immunis", "immuniz"),
          ("neovascularis", "neovasculariz"), ("ageing", "aging")]

_LOCK = threading.Lock()
_INDEX: Optional[Dict[str, Set[str]]] = None


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower().replace("’", "'").strip())


def spelling_variants(term: str) -> Set[str]:
    t = _norm(term)
    out = {t}
    for gb, us in _SPELL:
        for x in list(out):
            if gb in x:
                out.add(x.replace(gb, us))
            # American -> British only for stems that cannot occur inside another word
            # ('hem' is inside 'chemotherapy', 'labor' inside 'laboratory')
            elif us in _SAFE_US and us in x:
                out.add(x.replace(us, gb))
    return out


_SAFE_US = {"tumor", "edema", "estrogen", "esophag", "pediatric", "fetal", "diarrhea", "randomiz", "ischem", "glycem",
            "leukem", "anem", "anesth", "hematolog", "gynecolog", "behavior", "fiber", "orthoped", "immuniz",
            "neovasculariz", "lipem"}


def _load_mesh_file(path: Path) -> Dict[str, Set[str]]:
    """{entry term: {heading, entry terms...}} from desc*.xml or d*.bin; cached as JSON next to the file."""
    cache = path.with_suffix(path.suffix + ".biosyn.json")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        try:
            raw = json.loads(cache.read_text(encoding="utf-8"))
            return {k: set(v) for k, v in raw.items()}
        except Exception:
            pass
    groups: List[List[str]] = []
    if path.suffix.lower() == ".xml":
        import xml.etree.ElementTree as ET
        for _, el in ET.iterparse(str(path), events=("end",)):
            if el.tag == "DescriptorRecord":
                name = el.findtext("DescriptorName/String") or ""
                terms = [t.text or "" for t in el.iter("String")]
                groups.append([name] + terms)
                el.clear()
    else:   # ASCII MeSH: '*NEWRECORD', 'MH = heading', 'ENTRY = term|...', 'PRINT ENTRY = term|...'
        cur: List[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("*NEWRECORD"):
                if cur:
                    groups.append(cur)
                cur = []
            elif line.startswith("MH = "):
                cur.insert(0, line[5:].strip())
            elif line.startswith(("ENTRY = ", "PRINT ENTRY = ")):
                cur.append(line.split("=", 1)[1].split("|")[0].strip())
        if cur:
            groups.append(cur)
    idx: Dict[str, Set[str]] = {}
    for g in groups:
        terms = {_norm(t) for t in g if t and len(t) <= 80}
        if len(terms) < 2 or len(terms) > 60:
            continue
        for t in terms:
            idx.setdefault(t, set()).update(terms)
    try:
        cache.write_text(json.dumps({k: sorted(v) for k, v in idx.items()}), encoding="utf-8")
    except OSError:
        pass
    return idx


def _index() -> Dict[str, Set[str]]:
    global _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        idx: Dict[str, Set[str]] = {}
        for g in _GROUPS:
            terms = {_norm(t) for t in g}
            for t in terms:
                idx.setdefault(t, set()).update(terms)
        mf = os.getenv("BIOSYN_MESH_FILE")
        if mf and Path(mf).is_file():
            try:
                for k, v in _load_mesh_file(Path(mf)).items():
                    idx.setdefault(k, set()).update(v)
            except Exception as exc:
                print(f"  [MeSH] could not read {mf}: {str(exc)[:120]}")
        _INDEX = idx
        return idx


def reset() -> None:
    global _INDEX
    with _LOCK:
        _INDEX = None


def expand(term: str, max_terms: int = 12) -> Set[str]:
    """The term, its spelling variants and its synonyms (ambiguous 2-letter acronyms are never ADDED as synonyms -
    'ms', 'ad', 'ra' would match everything; they still expand when they are what the user typed)."""
    t = _norm(term)
    out: Set[str] = set()
    for v in spelling_variants(t):
        out.add(v)
        for s in _index().get(v, set()):
            if len(s) <= 3 and s != v:
                continue
            out.add(s)
    out = {o for o in out if o}
    if len(out) > max_terms:
        keep = sorted(out, key=lambda s: (s != t, len(s)))[:max_terms]
        out = set(keep)
    return out


_STOP = set("""a an the of and or in on at to for with by from as is are was were be been being does do did can could
should would will may might what which who whom whose how why when where whether than then there this that these those
reduce reduces reduced increase increases increased decrease decreases decreased improve improves improved lower lowers
raise raises cause causes caused lead leads help helps work works treat treats prevent prevents extend extends
it its their they them his her our your my me we you i not no effect effects affect affects change changes changed
compared versus vs between among during after before into over under about against without within via per""".split())


def concepts(text: str, max_ngram: int = 4) -> List[Set[str]]:
    """Greedy longest-match of known terms in the text (n-grams up to 4 words), each expanded; remaining content words
    become their own (spelling-expanded) concepts."""
    words = re.findall(r"[a-z0-9][a-z0-9\-']*", _norm(text))
    idx = _index()
    out: List[Set[str]] = []
    i = 0
    while i < len(words):
        hit = None
        for n in range(min(max_ngram, len(words) - i), 0, -1):
            cand = " ".join(words[i:i + n])
            if cand in idx or any(v in idx for v in spelling_variants(cand)):
                hit = (cand, n)
                break
        if hit:
            out.append(expand(hit[0]))
            i += hit[1]
            continue
        # adjacent content words that are not a known term stay ONE phrase ('pulmonary vascular resistance')
        run = []
        while i < len(words) and words[i] not in _STOP and len(words[i]) > 1:
            nxt = None
            for n in range(min(max_ngram, len(words) - i), 0, -1):
                cand = " ".join(words[i:i + n])
                if cand in idx or any(v in idx for v in spelling_variants(cand)):
                    nxt = cand
                    break
            if nxt:
                break
            run.append(words[i])
            i += 1
        if run:
            phrase = " ".join(run)
            if len(phrase) > 2 and phrase not in _STOP:
                out.append(spelling_variants(phrase))
            continue
        i += 1
    # drop concepts that are a subset of another (e.g. 'cancer' when 'breast cancer' is present)
    uniq: List[Set[str]] = []
    for c in out:
        if not any(c < d for d in out if d is not c) and c not in uniq:
            uniq.append(c)
    return uniq


def info() -> Dict[str, object]:
    idx = _index()
    return {"terms": len(idx), "curated_groups": len(_GROUPS), "mesh_file": os.getenv("BIOSYN_MESH_FILE") or None,
            "version": MESH_VERSION}
