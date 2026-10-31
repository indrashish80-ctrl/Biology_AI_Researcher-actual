from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from stats_parsing import _norm_for_match, number_in_text, parse_p_values, quote_grounding
from study_classifier import claim_model, get_or_classify

AUDIT_VERSION = "3.27"
GROUNDING_THRESHOLD = 0.70

CLAIM_STATUS = {
    "rejected": "Rejected extraction",
    "causal_supported": "Causal claim supported",
    "causal_unsupported": "Causal claim unsupported (kept as an observation)",
    "association": "Source-supported association",
    "observation": "Source-supported observation",
    "supported": "Source-supported",
}

MICROBE_WORDS = (r"bacteri\w*|microb\w*|fung\w*|fungi|yeasts?|mycobacteri\w*|"
                 r"escherichia|salmonella|staphylococc\w*|streptococc\w*|pseudomonas|candida|aspergillus|"
                 r"saccharomyces|lactobacill\w*|enterococc\w*|klebsiella|acinetobacter|helicobacter|listeria|"
                 r"(?:bacterial|microbial|fungal|yeast|viral|clinical)\s+(?:strains?|isolates?|cultures?)|"
                 r"clostrid\w*|parasites?|plasmodium|trypanosoma|leishmania|e\. ?coli|m\. ?tuberculosis|s\. ?aureus|"
                 r"p\. ?aeruginosa|c\. ?albicans|s\. ?cerevisiae|pathogens?|inocul\w*|planktonic|"
                 r"viruse?s?|viral|phages?|bacteriophages?")
_MG_WORD = r"surviv\w*|viabilit\w*|growth|proliferat\w*|killing|clearance|burden|counts?|densit\w*|titers?|titres?|load"

ENDPOINT_FAMILIES: List[Tuple[str, str, str]] = [
    ("clinical_response", r"recover\w*|repigment\w*|remission|improvement|improved|heal(?:ed|ing)|clinical response|"
                          r"response rate|resolution|clearance|\bcured?\b|\bcures\b|curable|\bpigmentation\b|"
                          r"lesion (?:size|area|count|number|extent)|size of (?:the )?lesions?|affected (?:body )?(?:surface )?area|"
                          r"depigment\w*|disease (?:extent|activity|severity)|\bvasi\b|\bpasi\b|\bscorad\b|severity scores?",
     r"lesion|size of|affected|depigment|disease (?:extent|activity|severity)|vasi|pasi|scorad|severity"),
    ("plant_growth_yield", r"\byield\b|biomass|plant growth|root (?:growth|length)|shoot (?:growth|length)|germination|"
                          r"stomatal conductance|photosynthe\w*|chlorophyll fluorescence|fv/fm", r"$^"),
    ("tumour_growth", r"(?:tumou?r )?latency (?:in|of|to) (?:\w+[- ]){0,2}tumou?r\w*(?: growth| onset| formation| development)?|"
                      r"tumou?r(?:s)? (?:growth|volume|weight|size|burden|progression|latency|onset|appearance)|xenograft growth|"
                      r"tumou?r load|growth of (?:the )?(?:\w+[- ]){0,4}xenografts?|xenografts? (?:growth|volume|size)|"
                      r"(?:appearance|growth|development) (?:and (?:growth|progression) )?of (?:the )?(?:\w+[- ]){0,2}"
                      r"(?:carcinomas?|tumou?rs?|neoplasms?)",
     r"latency"),
    ("immune_function", r"immune (?:function|response|activation|cell (?:function|activation|proliferation|infiltration))|"
                        r"(?:t|nk|nkt|b)[- ]cells? (?:activation|function|proliferation|expansion|cytotoxicity|exhaustion|"
                        r"infiltration|response)|lymphocyte (?:proliferation|activation|infiltration)|immune (?:cell )?exhaustion|"
                        r"exhaustion markers?|cytokine production|immune[- ]cell function|"
                        r"(?:proliferat\w*|cytotoxic\w*|expansion|activation)(?:,? (?:and )?\w+)* (?:of|by) (?:[\w\-]+ ){0,2}"
                        r"(?:t|nk|nkt|b)[- ]cells?", r"exhaustion"),
    ("viability", r"cell viabilit\w*|viabilit\w*|cell densit\w*|cell survival|cytotoxic\w*|\bmtt\b|cck-?8", r"cytotoxic"),
    ("proliferation", r"proliferat\w*|ki-?67|\bpcna\b|mitotic index|brdu|edu incorporation|cell growth|cell numbers?|"
                      r"colony formation|clonogenic\w*|cell[- ]cycle progression|growth of (?:\w+ ){0,3}cells|cell division|"
                      r"\bgrowth\b", r"$^"),
    ("apoptosis", r"apopto\w*|caspase(?:-?\d+)?(?: activity| activation| cleavage)?|tunel|annexin|cleaved parp|cell death|necrosis",
     r"$^"),
    ("patient_survival", r"overall survival|progression-free survival|disease-free survival|recurrence-free survival|"
                         r"survival|mortality|death", r"mortality|death"),
    ("adverse_events", r"adverse (?:events?|effects?|reactions?)|side[- ]effects?|toxicit\w*|(?:gastrointestinal|gi) "
                       r"(?:complications?|symptoms?|intolerance|disturbances?)|complications?|diarrh\w*|nausea|vomiting|"
                       r"lactic acidosis|tolerab\w*|intolerance", r"tolerab"),
    ("recurrence_incidence", r"recurrence|relapse|incidence|(?:risk|odds)(?: of| for) (?:\w+[ -]){0,3}(?:cancer|disease|leuk\w*|lymphoma|myeloma|"
                             r"tumou?r\w*|carcinoma|diabet\w*|dementia|stroke|infection)|leuk(?:a)?emogen\w*",
     r"$^"),
    ("metastasis_invasion", r"metasta\w*|invasi\w*|migrat\w*|\bemt\b|epithelial[- ]mesenchymal|e-cadherin|motility|"
                            r"wound[- ]heal\w*|scratch(?: wound)? assay|wound closure", r"e-cadherin"),
    ("glycaemic", r"hba1c|glucose|insulin|homa-?ir|glyc(?:a)?emi\w*", r"$^"),
    ("inflammation", r"inflammat\w*|\bil-?6\b|\bil-?1(?:β|b|beta|α|a)?\b|\btnf(?:-?α|-?alpha|-a)?(?![\w-])|c-reactive|\bcrp\b|nf-?κb|nf-?kb|nf-?kappa", r"$^"),
    ("body_weight", r"body weight|weight gain|weight loss|\bbmi\b|adiposity|fat mass", r"weight loss"),
    ("stress_tolerance", r"tolerance|stomatal conductance|water content|photosynthe\w*|chlorophyll", r"$^"),
    ("antimicrobial_resistance", r"(?:antibiotic|antimicrobial|multidrug|antifungal|antiviral|bacterial|fungal|\w+cillin|"
                                 r"\w+mycin|\w+floxacin|\w+cycline|carbapenem|colistin|methicillin|vancomycin)[- ]resistan\w*|"
                                 r"resistan\w* (?:to|against) (?:\w+ ){0,2}(?:antibiotic\w*|antimicrobial\w*|\w+cillin|\w+mycin|"
                                 r"\w+floxacin|\w+cycline|carbapenems?|colistin|azoles?)|\bMRSA\b|\bAMR\b|\bMICs?\b|"
                                 r"minimum inhibitory|antimicrobial susceptibilit\w*|antibiotic susceptibilit\w*|"
                                 r"(?:isolates?|strains?|bacteri\w*|pathogens?|microb\w*|mutants?)\s+(?:\w+\s+){0,3}resistan\w*|"
                                 r"resistan\w*\s+(?:\w+\s+){0,2}(?:isolates?|strains?|bacteri\w*|pathogens?)", r"$^"),
    ("biofilm", r"biofilm\w*", r"$^"),
    ("virulence", r"virulen\w*|colonization|infectivity|pathogenicity", r"$^"),
    ("microbial_growth", rf"(?:{MICROBE_WORDS})(?:\s+\w+){{0,2}}\s+(?:{_MG_WORD})|"
                         rf"(?:{_MG_WORD})\s+of\s+(?:the\s+)?(?:\w+\s+){{0,2}}(?:{MICROBE_WORDS})|"
                         r"colony[- ]forming units?|\bCFU\b|viable counts?|plate counts?|optical density|\bOD600\b|"
                         r"log10 (?:reduction|kill)|time[- ]kill|zone of inhibition",
                         r"killing|clearance|log10 reduction|time[- ]kill|zone of inhibition"),
    ("cognition_behaviour", r"memory|cognit\w*|learning|behaviou?r\w*", r"$^"),
    ("genotoxicity", r"chromosom\w* (?:aberration|damage|abnormalit|breaks?|translocation)\w*|chromosomal instability|dna damage|"
                     r"dna (?:strand )?breaks?|micronucle\w*|sister chromatid exchange\w*|genotoxic\w*|comet assay|"
                     r"\b(?:somatic )?mutation (?:rate|frequency|burden)|\baneuploid\w*|(?:reciprocal|balanced|robertsonian|"
                     r"chromosom\w*|gene) translocations?|\bt\(\d+;\d+\)", r"$^"),
    ("protein_localization", r"\b(?:nuclear|membrane|cytoplasmic|cytosolic|mitochondrial|receptor|surface)\s+(?:translocation|"
                             r"locali[sz]ation|accumulation)|\btranslocat\w*\b|\blocali[sz]ation\b|internali[sz]ation", r"$^"),
    ("expression", r"expression|mrna|transcript\w*|protein levels?|abundance", r"$^"),
    ("signalling", r"phosphorylat\w*|activation|activity|signall?ing", r"$^"),
]
FAMILY_RX = [(f, re.compile(rx, re.I), re.compile(neg, re.I)) for f, rx, neg in ENDPOINT_FAMILIES]
FAMILY_LABEL = {f: f.replace("_", " ") for f, _, _ in ENDPOINT_FAMILIES}
RELATED_OPPOSED = {frozenset(("viability", "apoptosis")), frozenset(("proliferation", "apoptosis"))}

INVERSE_NOUN = r"(?:inhibition|suppression|reduction|decrease|loss|arrest|blockade|attenuation|impairment|abrogation|prevention|delay)"
INC = re.compile(r"\b(increas\w*|enhanc\w*|improv\w*|prolong\w*|elevat\w*|promot\w*|stimulat\w*|induc\w*|upregulat\w*|up-regulat\w*|augment\w*|"
                 r"higher|greater|larger|bigger|steeper|more(?!\s+(?:significant\w*|pronounced|evident|marked|potent|effective|sensitive|resistant|important|likely|"
                 r"frequent\w*|common|than|so|or|detail\w*|specific\w*|precise\w*|robust))|rais\w*|accelerat\w*|activat\w*|potentiat\w*|boost\w*|doubl\w*|gain\w*|rose|rise[sn]?|"
                 r"heighten\w*|excess|longer|extended|faster|earlier|sooner|re-?express\w*|restor\w*|"
                 r"better|superior|enrich\w*|more effective|more common|more frequent\w*|more likely)\b", re.I)
DEC = re.compile(r"\b(reduc\w*|decreas\w*|inhibit\w*|suppress\w*|lower\w*|attenuat\w*|diminish\w*|impair\w*|block\w*|"
                 r"abrogat\w*|abolish\w*|downregulat\w*|down-regulat\w*|declin\w*|less|fewer|smaller|shrank|shrunk|"
                 r"repress\w*|prevent\w*|delay\w*|slow\w*|lost|fell|drop\w*|halved|shorter|slower|hinder\w*|depress\w*|abrogat\w*|curb\w*|impede\w*|"
                 r"absent|missing|lacking|lacked|failed to (?:form|develop|appear|emerge)|loss|truncat\w*|"
                 r"worse|inferior|less effective|less common|less frequent\w*|less likely|"
                 r"ablat\w*|eliminat\w*|disrupt\w*|deplet\w*|sensiti[sz]\w*)\b", re.I)
NOEFFECT = re.compile(r"((?:little|minimal|negligible|marginal|no direct|no apparent|no obvious)(?: or no)? (?:effect|change|influence|"
                      r"differences?|alteration)|"
                      r"remain(?:ed|s)? (?:roughly |largely |essentially |relatively )?(?:constant|unchanged|stable|similar)|"
                      r"without (?:significantly )?(?:chang|affect|alter|influenc)\w*|"
                      r"no (?:significant |statistically significant |detectable |measurable )?(?:effect|change|difference|"
                      r"association|alteration|impact|influence)|not (?:significantly |statistically )?(?:affect|alter|change|"
                      r"differ|reduc|increas|decreas|improv|associat|modif|influenc)\w*|did not (?!use\b|receive\b|take\b|undergo\b|get\b|have\b|only\b|just\b|merely\b)\w+|does not (?!use\b|receive\b|take\b|have\b|only\b|just\b|merely\b)\w+|do not (?!use\b|receive\b|take\b|have\b|only\b|just\b|merely\b)\w+|"
                      r"failed to \w+|unchanged|unaltered|non-?significant|not (?:statistically )?significant(?!ly)|(?:was|were) not (?:statistically )?significant|"
                      r"similar between|comparable between|no longer|"
                      # v25 (gold benchmark): 'was similar in the three groups', 'similarly high in both groups',
                      # 'did not differ', 'equal values', 'comparable in both arms'
                      r"(?:was|were|is|are|remained|being|appeared)\s+(?:also\s+|very\s+|broadly\s+|statistically\s+)?"
                      r"(?:similar|comparable|equal|equivalent|identical|indistinguishable)(?!\s+to\b)|"
                      r"similarly\s+\w+\s+in\s+(?:both|all|the two|the three)|did not differ|do not differ|does not differ|"
                      r"no (?:statistically )?significant differences?|equal values|"
                      r"similar (?:\w+\s+){0,4}?to (?:that|those) (?:of|in|seen with|with|observed)|"
                      r"(?:a |an )?similar (?:\w+\s+){0,5}?to (?:the )?(?:placebo|control|sham|vehicle|comparator)|"
                      r"none of the (?:patients|participants|subjects|animals) in (?:any|either|both|all) (?:of the )?(?:\w+ )?groups?)", re.I)
OCCUR = re.compile(r"\b(observed|achieved|attained|occurred|noted|seen|showed|recorded|obtained|reported|documented|maintained|"
                   r"cured|remained|reached|sustained|was \d+(?:\.\d+)?\s*%|were \d+(?:\.\d+)?\s*%|in \d+(?:\.\d+)?\s*% of|"
                   r"was complete|were complete|resulted)\b", re.I)
CITATION_MARK = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|\(\s*[A-Z][A-Za-z\-]+ et al\.?,? \d{4}[a-z]?\s*\)|\bet al\.?\s*\(?\d{4}|"
                           r"\b(?:previous(?:ly)? (?:studies|reports|work)|has been (?:shown|reported|demonstrated)|"
                           r"have (?:been )?(?:shown|reported|demonstrated)|it is (?:well )?known|earlier studies)\b")
PRIOR_WORK = re.compile(r"\b(?:in (?:our|a|an) (?:previous|prior|earlier) (?:study|work|report|paper|publication)|"
                        r"we (?:have )?previously (?:shown|reported|found|observed|demonstrated|described)|"
                        r"as (?:we )?(?:previously|earlier) (?:reported|described|shown)|as in the [\w\-]+ (?:trial|study|cohort)|as (?:reported|observed|seen|shown) in (?:the )?[\w\-]+ "
                        r"(?:trial|study)|similar to (?:that of |those of )?(?:the )?[\w\-]+ (?:trial|study)|in the [A-Z]+-?\d+ trial|"
                        r"not all studies|some studies|several studies|most studies|many studies|other studies|"
                        r"(?:epidemiological|observational|previous|prior|existing|available|accumulating|clinical|preclinical) "
                        r"(?:evidence|studies|data|reports) (?:indicate|suggest|show|support)\w*|remains? inconsistent|"
                        r"[A-Z][a-z]+ et al\.?\s*\[?\(?\d{0,4}\)?\]?,?\s+(?:found|showed|reported|demonstrated|observed|suggested|noted)|"
                        r"previous(?:ly)? (?:studies|reports|work)|earlier studies|has been (?:shown|reported|linked|demonstrated)|"
                        r"have been (?:shown|reported|linked|demonstrated)|in (?:two|three|four|five|several|many|multiple|other|"
                        r"earlier|previous|prior|\d+) (?:\w+ ){0,2}(?:studies|cohorts|trials|reports|investigations)|"
                        r"(?:studies|reports|cohorts) have (?:shown|reported|found|linked))\b", re.I)
AIM_OR_SPECULATION = re.compile(
    r"\b(?:we )?(?:investigated|examined|assessed|evaluated|explored|tested|determined|asked) (?:whether|if|how)\b|"
    r"\bto (?:investigate|examine|determine|assess|evaluate|explore|test) (?:whether|if|the effect)\b|\bwe hypothesi[sz]ed\b|"
    r"\b(?:aim|objective|purpose|goal) (?:of|was|is)\b|"
    r"\b(?:may|might|could) (?:have|be|potentially|possibly|serve|represent|offer|provide|act)\b|\bis thought to\b|"
    r"\bit is (?:possible|conceivable|plausible) that\b|"
    # a CALL FOR FUTURE RESEARCH is never a result of this study ('these results merit further investigation of the
    # effects of increased BCAA on ... cancer recurrence' became 'metformin increased recurrence')
    r"\b(?:merits?|warrants?|deserves?|requires?|needs?|justifies)\s+(?:further|additional|future|more|continued|"
    r"dedicated|prospective)\s+(?:investigation|research|study|studies|work|evaluation|assessment|validation|"
    r"confirmation|trials?|attention|exploration)\b|"
    r"\b(?:further|additional|future|more|larger|prospective|randomi[sz]ed)\s+(?:investigation|research|study|studies|"
    r"work|trials?|evaluation|validation|confirmation|analys[ei]s)\s+(?:is|are|will be|would be|should be|remain)\s+"
    r"(?:needed|warranted|required|necessary|merited|justified)\b|"
    r"\b(?:future|further)\s+(?:work|studies|research|trials?)\s+(?:should|will|must|could|may|ought to)\b|"
    r"\b(?:should|needs? to|ought to|remains? to)\s+be\s+(?:investigated|examined|explored|determined|established|"
    r"confirmed|validated|tested|clarified|elucidated)\b|"
    r"\bit would be (?:interesting|important|valuable|useful) to\b|"
    r"\b(?:we|authors?) (?:call for|recommend|propose|suggest) (?:further|additional|future|that)\b", re.I)
INCONSISTENT = re.compile(r"\b(?:not all studies|remains? (?:inconsistent|unclear|controversial|conflicting)|inconsistent (?:results|findings|evidence)|"
                          r"conflicting (?:results|findings|evidence|data)|mixed (?:results|findings|evidence)|results (?:are|were) (?:inconsistent|mixed|conflicting))\b", re.I)
SIGNIF_WORD = re.compile(r"(?<!not )(?<!non-)(?<!non)(?<!no )(?<!without )(?<!lack of )(?<!not statistically )(?<!no statistically )"
                         r"\bsignificant(?:ly)?\b(?!\s+(?:different|differ)\w*\s+from\s+zero)", re.I)
_NEG_SIG = re.compile(r"\b(?:no|not|non-?|without|lack of|did not|does not|do not|failed to|was not|were not)\s+(?:\w+\s+){0,2}"
                      r"significant\w*", re.I)


def _claims_significance(text: str) -> bool:
    t = _NEG_SIG.sub(" ", text or "")
    return bool(SIGNIF_WORD.search(t))


_BETWEEN_NULL = re.compile(r"\bno (?:significant |statistically significant )?differences? (?:\w+ ){0,3}?between (?:the )?(?:two |treatment |"
                           r"study )?(?:arms|groups)\b|\b(?:did not|does not|no) (?:significantly )?differ\w* between (?:the )?(?:two )?"
                           r"(?:arms|groups)\b|\bsimilar (?:in|between) (?:both|the two) (?:arms|groups)\b", re.I)
_WITHIN_CHANGE = re.compile(r"\bthan (?:that )?(?:at|in|on) (?:the )?(?:baseline|core biopsy|pre-?treatment)|\bfrom baseline\b|"
                            r"\b(?:compared|relative) (?:with|to) baseline\b|\bover time\b|\bpre-? (?:to|vs\.?|versus) post|"
                            r"\bbefore and after\b|\bsurgery-biopsy\b|\bbaseline (?:core )?biopsy\b", re.I)
_WITHIN_ONLY = re.compile(r"\bfrom\s+[\d.]+\s*%?(?:[^.;]|\.(?=\d)){0,60}?\b(?:at\s+)?baseline\b(?:[^.;]|\.(?=\d)){0,50}?\bto\s+[\d.]+|\bfrom baseline to\b|"
                          r"\bbaseline to (?:surgery|post-?treatment|follow-?up)\b|\bpre- (?:to|and) post-?treatment\b", re.I)
_BETWEEN_TEST = re.compile(r"\b(?:compared (?:with|to)|versus|vs\.?|relative to|than (?:in |that in |with )?(?:the )?(?:control|placebo|"
                           r"standard|other|untreated)|between (?:the )?(?:two )?(?:arms|groups)|placebo|control (?:arm|group)|"
                           r"arm b\b|difference between)\b", re.I)
ALPHA = 0.05
_CI = re.compile(r"(?:(\d{2})\s*%\s*)?(?:CI|confidence interval)s?\s*[:=,]?\s*[\[(]?\s*(-?\d+(?:\.\d+)?)\s*%?\s*"
                 r"(?:to|–|—|,|;|−(?=\d)|-(?=\s*-?\d))\s*(-?\d+(?:\.\d+)?)\s*%?", re.I)
_RATIO_CUE = re.compile(r"\b(?:HR|OR|RR|IRR|aHR|aOR|SHR|hazard ratios?|odds ratios?|risk ratios?|rate ratios?|"
                        r"relative risks?|ratio)\b", re.I)


def statistical_reading(text: str) -> Dict[str, Any]:
    from stats_parsing import normalize_numeric_text
    t = normalize_numeric_text(text or "")     # v25: Unicode minus / mid-dot decimals in confidence intervals
    ps = parse_p_values(t)
    cis = []
    for m in _CI.finditer(t):
        lo, hi = float(m.group(2)), float(m.group(3))
        if lo > hi:
            lo, hi = hi, lo
        ratio = bool(_RATIO_CUE.search(t[max(0, m.start() - 60):m.start()]))
        null = 1.0 if ratio else 0.0
        cis.append({"lo": lo, "hi": hi, "null": null, "contains_null": lo <= null <= hi, "text": m.group(0)})
    says_sig = _claims_significance(t)
    says_nonsig = bool(_NEG_SIG.search(t)) or bool(re.search(r"\b(?:non-?significant|n\.?s\.?(?=[\s,;).]|$)|did not reach "
                                                               r"(?:statistical )?significance)\b", t, re.I))
    sig_p = [p for p in ps if p["value"] < ALPHA and p["comparator"] in ("<", "=", "<=")]
    nonsig_p = [p for p in ps if p["value"] >= ALPHA and p["comparator"] in ("=", ">", ">=")]
    out = {"p": [p["value"] for p in ps], "ci": cis, "author_says_significant": says_sig}
    if sig_p and not nonsig_p:
        return {**out, "significant": True, "basis": f"P = {sig_p[0]['value']:g} < {ALPHA}"}
    if nonsig_p and not sig_p:
        return {**out, "significant": False, "basis": f"P = {nonsig_p[0]['value']:g} (not below {ALPHA})"}
    if cis and not ps:
        if all(c["contains_null"] for c in cis):
            c = cis[0]
            return {**out, "significant": False,
                    "basis": f"95% CI {c['lo']:g} to {c['hi']:g} includes {c['null']:g} (no difference)"}
        if not any(c["contains_null"] for c in cis):
            return {**out, "significant": True, "basis": "confidence interval excludes no-difference"}
    if says_nonsig and not sig_p:
        return {**out, "significant": False, "basis": "the source states the difference was not significant"}
    if says_sig:
        return {**out, "significant": True, "basis": "the source states significance (no p-value / CI quoted)"}
    return {**out, "significant": None, "basis": "no significance test reported in the quote"}


ENDPOINT_ACRONYMS = {"RFS": "recurrence-free survival", "OS": "overall survival", "DFS": "disease-free survival",
                     "PFS": "progression-free survival", "EFS": "event-free survival", "DMFS": "distant metastasis-free survival",
                     "MFS": "metastasis-free survival", "BCSS": "breast cancer-specific survival", "CSS": "cancer-specific survival",
                     "TTP": "time to progression", "ORR": "overall response rate", "CR": "complete remission",
                     "pCR": "pathological complete response", "CBC": "contralateral breast cancer", "LFS": "leukemia-free survival",
                     "TFR": "treatment-free remission", "MMR": "major molecular response", "DCR": "disease control rate"}


def expand_endpoint(text: str, abbr: Optional[Dict[str, str]] = None) -> str:
    t = text or ""
    table = dict(ENDPOINT_ACRONYMS)
    table.update({k: v for k, v in (abbr or {}).items() if k not in ("MET",)})
    for a, full in table.items():
        if re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", t):
            t = re.sub(rf"(?<![\w-]){re.escape(a)}(?![\w-])", f"{full} ({a})", t)
    return t


GENERIC_INTERVENTION = {"experimental", "treatment", "treatments", "intervention", "exposure", "factor", "drug", "agent",
                        "compound", "therapy", "the", "and", "with", "use", "users", "effect", "effects", "biological",
                        "reduce", "reduces", "increase", "affect", "affects", "does", "how", "what", "levels", "level"}
_SYMBOL = re.compile(r"\b([A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{1,3})?)\b")


def flip(d: Optional[str]) -> Optional[str]:
    return {"increase": "decrease", "decrease": "increase"}.get(d or "", d)


def canonical_outcome(text: Optional[str]) -> Dict[str, Any]:
    t = (text or "").strip()
    low = t.lower()
    cell_wound = bool(re.search(r"wound[- ]heal|scratch|wound closure", low)) and \
        not re.search(r"\b(?:ulcers?|patients?|diabetic foot|burns?|surgical|participants|skin graft)\b", low)
    microbe_surv = bool(re.search(rf"(?:{MICROBE_WORDS}|viral|virus(?:es)?)"
                                  r"(?:\s+\w+){0,2}\s+(?:surviv\w*|viabilit\w*|growth|proliferat\w*)|"
                                  r"(?:surviv\w*|viabilit\w*|growth|proliferat\w*)\s+of\s+(?:the\s+)?(?:\w+\s+){0,2}"
                                  r"(?:bacteri\w*|microb\w*|fung\w*|yeasts?|strains?|isolates?|cultures?|mycobacteri\w*|"
                                  rf"{MICROBE_WORDS})", low))
    for fam, rx, neg in FAMILY_RX:
        m = rx.search(low)
        if not m:
            continue
        if fam == "clinical_response" and cell_wound and m.group(0).startswith("heal"):
            continue
        if fam in ("patient_survival", "viability", "proliferation", "tumour_growth") and microbe_surv:
            continue
        if fam == "protein_localization" and re.search(r"chromosom|\bt\(\d+;\d+\)|reciprocal|robertsonian|fusion gene", low):
            continue
        if fam == "clinical_response" and m.group(0) in ("improvement", "improved") and \
                any(r2.search(low) for f2, r2, _ in FAMILY_RX if f2 not in ("clinical_response", "expression", "signalling")):
            continue
        polarity = -1 if neg.search(m.group(0)) else 1
        span = re.escape(m.group(0))
        if re.search(rf"{span}\W+(?:\w+\W+){{0,1}}{INVERSE_NOUN}\b|{INVERSE_NOUN}\W+(?:of|in)\W+(?:the\W+)?(?:\w+\W+){{0,2}}{span}", low) \
                or re.search(rf"\banti-?{span}", low) or re.search(r"\banti-?(?:proliferat|tumou?r|metasta|apopto|inflammat)", low):
            polarity = -polarity
        targets = [s for s in _SYMBOL.findall(t) if s not in ("MTT", "CCK", "EMT", "MIC", "BMI", "CRP", "PCNA", "HOMA", "IR")]
        return {"family": fam, "target": targets[0] if targets and fam in ("expression", "signalling") else None,
                "polarity": polarity, "matched": m.group(0), "label": FAMILY_LABEL[fam]}
    kt = _key_tokens(low)
    toks = [w for w in kt if len(w) > 3 and w not in GENERIC_INTERVENTION] or [w for w in kt if w not in _ENDPOINT_FILLER]
    return {"family": "other:" + (" ".join(toks[:3]) or "unspecified"), "target": None, "polarity": 1,
            "matched": None, "label": " ".join(toks[:3]) or "unspecified"}


_IMMUNE_CELL = (r"(?:γδ|gamma[- ]?delta|gd)?\s*(?:t|nk|nkt|b|car-?t|cd8\+?|cd4\+?|cytotoxic t|helper t|regulatory t)[- ]?"
                r"(?:cells?|lymphocytes?)|lymphocytes?|pbmcs?|peripheral blood mononuclear cells?|macrophages?|"
                r"dendritic cells?|neutrophils?|natural killer cells?|immune cells?|tils?|tumou?r-infiltrating lymphocytes?")
_EMT_MARKER = re.compile(r"\b(?:e-?cadherin|n-?cadherin|vimentin|snail|slug|twist|zeb1|zeb2|claudin\w*|β-?catenin|beta-?catenin|"
                         r"fibronectin|mmp-?\d+)\b", re.I)
_FUNCTIONAL_INVASION = re.compile(r"\b(?:transwell|boyden|matrigel|scratch|wound[- ]heal\w*|wound closure|invaded|migrated|"
                                  r"migratory (?:capacity|ability)|invasive (?:capacity|ability)|cell (?:invasion|migration|motility)|"
                                  r"metasta(?:tic )?(?:nodules?|foci|lesions?|burden)|number of (?:lung |liver )?metastas\w*|"
                                  r"lung metastas\w*|liver metastas\w*|distant metastas\w*)\b", re.I)


def refine_endpoint(oinfo: Dict[str, Any], outcome_text: str, quote: str) -> Tuple[Dict[str, Any], str, Optional[str]]:
    q = quote or ""
    low = q.lower()
    fam = oinfo.get("family")
    if fam in ("proliferation", "viability", "apoptosis", "tumour_growth") and re.search(
            rf"(?:proliferat\w*|cytotoxic\w*|viabilit\w*|expansion|cytokine production|activation|killing)\s+(?:\w+[,\s]+){{0,6}}?"
            rf"(?:of|by|in)\s+(?:the\s+)?(?:[\w\-]+\s+){{0,2}}?(?:{_IMMUNE_CELL})\b|(?:{_IMMUNE_CELL})\s*(?:[\w\-]+\s+){{0,1}}"
            rf"(?:proliferat\w*|cytotoxic\w*|expansion|killing|activation)", low):
        info = canonical_outcome("immune cell function")
        return info, "immune-cell function (" + (outcome_text or fam) + " of immune cells)", None
    if fam in ("glycaemic", "body_weight", "inflammation") or fam == "expression":
        m = oinfo.get("matched") or ""
        if m and re.search(rf"\b(?:at|for|in|among|with)\s+(?:those|patients|women|men|participants|subjects|people)?\s*"
                           rf"(?:with\s+)?(?:lower|higher|low|high|elevated|reduced|normal|healthier|baseline|increasing|"
                           rf"decreasing)\s+(?:\([\w\s]+\)\s+)?(?:baseline\s+)?(?:levels?\s+of\s+|(?:[\w\-]+\s+){{0,1}})?{re.escape(m)}",
                           low):
            return oinfo, outcome_text, (f"'{m}' is a baseline SUBGROUP in the quote ('at lower / higher {m} levels') - an "
                                         f"effect modifier, not an outcome the treatment changed")
        rx = next((r for f, r, _ in FAMILY_RX if f == fam), None)
        hits = list(rx.finditer(low)) if rx else []
        adj = r"[- ](?:dependent|independent|mediated|sensitive|related|responsive|driven|lowering|signall?ing|pathway)"
        if hits and all(re.match(adj, low[h.end():]) for h in hits):
            w = hits[0].group(0)
            return oinfo, outcome_text, (f"'{w}' appears only as an adjective ('{w}-dependent ...') - the quote names a "
                                         f"pathway type, not a measured {w} level")
    if fam == "cognition_behaviour" and re.search(r"memory (?:t|b|cd8\+?|cd4\+?|nk)\b|(?:effector|central|stem[- ]cell|tissue[- ]resident) "
                                                 r"memory|memory (?:phenotype|subsets?|cells?)|immunolog\w* memory", low):
        return canonical_outcome("immune cell function"), "immune-cell phenotype (memory T cells)", None
    if fam == "recurrence_incidence" and re.search(r"\b(?:incidence|frequency|rates?|occurrence) of (?:\w+[ -]){0,2}(?:adverse|side|"
                                                  r"gastrointestinal|gi|complications?|toxicit|diarrh|nausea|vomiting|lactic)", low):
        return canonical_outcome("adverse events"), "adverse events (tolerability)", None
    if fam == "recurrence_incidence" and re.search(r"\b(?:likelihood|proportion|rates?|frequency|odds|prevalence|distribution) of "
                                                  r"(?:\w+[ -]){0,2}(?:tnbc|triple[- ]negative|er[- ]positive|her2[- ]positive|luminal|"
                                                  r"basal|subtypes?)|\b(?:er|pr|her2)[- ]positive rates?\b", low):
        info = {**canonical_outcome("tumour subtype distribution"), "family": "other:subtype distribution",
                "label": "tumour subtype distribution"}
        return info, "tumour subtype distribution (which subtype patients had - not disease incidence or recurrence)", None
    if fam == "glycaemic" and re.search(r"glucose[- ](?:6-phosphate )?transporters?|\bglut-?\d+|\bslc2a\d+", low):
        info = canonical_outcome("glucose transporter expression")
        info = {**info, "family": "expression", "target": "GLUT (glucose transporters)", "label": "expression"}
        return info, "glucose transporter (GLUT) expression", None
    if fam == "metastasis_invasion" and _EMT_MARKER.search(q) and not _FUNCTIONAL_INVASION.search(q):
        info = {**canonical_outcome("expression"), "target": "EMT markers"}
        return info, "EMT marker expression (" + ", ".join(sorted({x.group(0) for x in _EMT_MARKER.finditer(q)})[:4]) + ")", None
    return oinfo, outcome_text, None


def _endpoint_regex(outcome_info: Dict[str, Any], outcome_text: str) -> re.Pattern:
    fam = outcome_info["family"]
    for f, rx, _ in FAMILY_RX:
        if f == fam:
            return rx
    words = [re.escape(w) for w in re.findall(r"[a-z][a-z0-9\-]{3,}", (outcome_text or "").lower())
             if w not in GENERIC_INTERVENTION][:4]
    return re.compile("|".join(words) if words else r"$^", re.I)


_ENDPOINT_FILLER = {"expression", "level", "levels", "protein", "proteins", "gene", "genes", "rate", "rates", "change",
                    "changes", "activity", "measured", "total", "relative", "cells", "cell", "mrna"}
_GROUP_RATE = re.compile(r"\b(?:for|in|of)\s+(?:the\s+)?([\w\-]+(?:\s+[\w\-]+)?)\s+(?:group|arm|users|cohort)\s+"
                         r"(?:was|were|is|are|of)\s+(\d+(?:\.\d+)?)\s*%", re.I)

_RATIO_VALUE = re.compile(r"\b(?:HR|OR|RR|IRR|aHR|aOR|SHR|hazard ratios?|odds ratios?|risk ratios?|rate ratios?|"
                          r"incidence rate ratios?|relative risks?)\b[^.;,]{0,20}?(\d+(?:\.\d+)?)", re.I)
_ASSOC_WITH = re.compile(r"\b(?:associated with|association with|linked to|predicted|predictor of|risk of|odds of|"
                         r"hazard of|incidence of|conferred?|carried)\b", re.I)


def ratio_direction(text: str, outcome_info: Dict[str, Any], outcome_text: str = "") -> Optional[str]:
    t = text or ""
    if not _ASSOC_WITH.search(t):
        return None
    erx = _endpoint_regex(outcome_info, outcome_text)
    if not (erx.search(t.lower()) or any(tok in t.lower() for tok in _key_tokens(outcome_text))):
        return None
    m = _RATIO_VALUE.search(t)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    if val == 1.0:
        return "no_effect"
    stat = statistical_reading(t)
    cis = [c for c in stat.get("ci") or [] if c["null"] == 1.0]
    if stat.get("significant") is False or (cis and all(c["contains_null"] for c in cis)):
        return "no_effect"
    if stat.get("significant") is None and not cis:
        return None
    return "increase" if val > 1.0 else "decrease"


def group_rate_direction(text: str, intervention: str) -> Optional[str]:
    toks = _key_tokens(intervention)
    treated, control = None, None
    for m in _GROUP_RATE.finditer(text or ""):
        name, val = m.group(1).lower(), float(m.group(2))
        if re.match(r"(?:non|no|without|control|placebo|un)", name):
            control = val if control is None else control
        elif any(t in name for t in toks):
            treated = val if treated is None else treated
    if treated is None or control is None or treated == control:
        return None
    return "increase" if treated > control else "decrease"


def endpoint_directions(text: str, outcome_info: Dict[str, Any], outcome_text: str = "") -> List[str]:
    return direction_in_text(text, outcome_info, outcome_text, collect=True).get("all") or []


_ADJUSTED = re.compile(r"\b(?:after\s+|when\s+)?(?:adjust(?:ed|ing|ment)\s+(?:for|by)|controlling\s+for|stratified\s+by)\s+"
                       r"[^.;]{0,90}?(?=,\s|;|\.|$)|\b[\w\-]+-adjusted\b|\b(?:adjusted|stratified)\s+(?:by|for)\s+[\w\-]+", re.I)


def strip_adjusters(text: str) -> str:
    return _ADJUSTED.sub(" ", text or "")


_ANTI_ADJ = re.compile(r"\banti-?(proliferat\w*|tumou?r\w*|metasta\w*|apopto\w*|inflammat\w*|cancer|neoplastic|angiogen\w*)", re.I)


_RESULT_VERB = (r"(?:markedly |significantly |strongly |also |further |dramatically |robustly |significantly and robustly )?"
                r"(?:increas|decreas|reduc|elevat|enhanc|induc|promot|stimulat|upregulat|up-regulat|downregulat|down-regulat|"
                r"suppress|inhibit|lower|rais|diminish|attenuat|accelerat|impair|restor|abolish|boost|potentiat|activat|"
                r"caus|led to|lead to|leads to|resulted in|results in|result in|produc|trigger|drove|drive|rescu|prevent|"
                r"decreas|declin|ameliorat|worsen|exacerbat|aggravat|alleviat|mitigat|arrest|halt|abrogat|disrupt|block|revers|"
                r"counteract|normali[sz]|blunt|abolish)\w*")
_MANIPULATION_SUBJECT = re.compile(
    r"(^|[,;:]\s+|\b(?:that|while|whereas|although|though|and|but)\s+|\b(?:in|of|from)\s+(?:[\w\-/]+\s+){0,3}?(?:cells|mice|rats|patients|animals|group)\s*,\s*)"
    r"(?:the\s+)?(?:loss|knock-?down|knock-?out|depletion|deletion|ablation|silencing|inhibition|blockade|deficiency|"
    r"downregulation|down-regulation|suppression|disruption|elimination|inactivation|absence|lack|overexpression|"
    r"over-expression|upregulation|up-regulation|activation|stimulation|treatment|administration|supplementation)\s+"
    r"(?:of|with)\s+[\w\-/+]+(?:\s+(?!(?:did|does|do|not|failed|fails|fail|was|were|is|are|had|has|have|could|would|can|"
    r"will|unable|neither|nor|significantly|the|a|an|this|that|its|their)\b)(?![\w\-]+ed\b)[\w\-/+]+){0,2}?(?=\s+" + _RESULT_VERB + r"\b)", re.I)
_NEGATED_CUE = re.compile(r"(?:\b(?:did|does|do|was|were|is|are|has|have|had|could|would|can|will)\s*(?:not|n't)|\bfail(?:ed|s)?\s+to|"
                         r"\bunable\s+to|\bnot(?!\s+only)|\bneither|\bnor|\bwithout)\s+(?:been\s+|be\s+)?(?:significantly\s+|substantially\s+|"
                         r"further\s+|markedly\s+|appreciably\s+|meaningfully\s+)?$", re.I)
_MANIPULATION_AFTER = re.compile(
    r"(\b(?:after|following|upon|by|with|via|through|in response to|due to)\s+(?:the\s+)?)"
    r"(loss|knock-?down|knock-?out|depletion|deletion|ablation|silencing|inhibition|blockade|deficiency|downregulation|"
    r"down-regulation|suppression|disruption|inactivation|absence|lack|overexpression|over-expression|upregulation|"
    r"up-regulation|activation|stimulation)\s+((?:of|with)\s+[\w\-/+]+)", re.I)


_LOWER_IS_BETTER = re.compile(r"\b(?:blood pressure|systolic|diastolic|hypertension|cholesterol|ldl|triglyceride\w*|"
                              r"glucose|hba1c|a1c|glyc(?:a)?emi\w*|insulin resistance|homa|pain|vas\b|symptom\w*|"
                              r"severity|mortality|death|recurrence|relapse|exacerbation\w*|hospitali[sz]ation|"
                              r"adverse|toxicit\w*|nausea|vomiting|fatigue|anxiety|depression|ham-?d|madrs|bdi|"
                              r"disability|haq|wound infection|infection rate|bleeding|blood loss|readmission\w*|"
                              r"tumou?r (?:size|volume|burden)|lesion\w*|inflammation|crp|il-6|viral load|"
                              r"intraocular pressure|iop|bmi|body weight|waist)\b", re.I)
_GENERIC_MEASURE = set("""cholesterol glucose level levels score scores rate rates count counts time times volume volumes
concentration concentrations activity index total mean change changes number proportion incidence risk pressure function
density size value values amount amounts percentage ratio frequency duration""".split())


def _base_polarity(outcome_info: Dict[str, Any], m: "re.Match") -> int:
    """-1 when the matched words are the inverse form of the endpoint family (e.g. 'lesion area' for clinical response)."""
    return -1 if any(f == outcome_info.get("family") and neg.search(m.group(0)) for f, _, neg in FAMILY_RX) else 1


def _in_agent_name(clause: str, m: "re.Match") -> bool:
    """True when the endpoint mention is part of the name of an agent: 'anti-TNF therapy', 'anti-IL-6 antibody',
    'an antagonist of IL-6' (masked to 'agentx of il-6'), 'IL-6 inhibitor'."""
    before, after = clause[max(0, m.start() - 30):m.start()], clause[m.end():m.end() + 40]
    if re.search(r"\banti-?$", before) and re.match(r"\s*(?:-\w+\s*)?(?:monoclonal\s+)?(?:therapy|therapies|drugs?|treatment|"
                                                    r"medications?|agents?|antibod\w*|biologics?|blockers?|blockade|"
                                                    r"inhibitors?|mab|compounds?|regimens?)\b", after):
        return True
    if re.search(r"\b(?:agentx|inhibitors?|antagonists?|blockers?|neutrali[sz]ing antibod\w*)\s+(?:of|against|to)\s+(?:the\s+)?$", before):
        return True
    return bool(re.match(r"\s*-?(?:inhibitors?|antagonists?|blockers?|agentx|targeting)\b", after))


_AGENTIVE_NOUN = re.compile(r"\b(?:inhibitors?|blockers?|suppressors?|repressors?|inducers?|activators?|enhancers?|promoters?|"
                            r"stimulators?|potentiators?|attenuators?|antagonists?|agonists?|modulators?|depleters?)\b", re.I)


def direction_in_text(text: str, outcome_info: Dict[str, Any], outcome_text: str = "", collect: bool = False) -> Dict[str, Any]:
    found: List[str] = []
    low = _squash_symbols(strip_adjusters((text or "").lower()))
    # v24.2: 'JAK inhibitor', 'SGLT2 inhibitor', 'the blocker' NAME the agent; they are not a change in the endpoint. Left in,
    # 'The inhibitor caused a decrease in X' read as inhibition TWICE and came out as an INCREASE.
    if not _AGENTIVE_NOUN.search(outcome_text or ""):
        low = _AGENTIVE_NOUN.sub("agentx", low)
    # v24.2: 'Loss of p53 increased proliferation', 'Knockdown of STAT3 reduced Ki-67': a manipulation phrase in SUBJECT position
    # names what was done; the verb after it is the result. Masked only when a result verb follows it directly, so
    # 'inhibition of proliferation was observed' (the endpoint itself) is untouched.
    low = _MANIPULATION_SUBJECT.sub(lambda m: m.group(1) + "agentx", low)
    # v24.3: 'A decrease in Ki-67 was observed AFTER LOSS OF p53' - a manipulation after a time/cause preposition names what
    # was done, never a second change of the endpoint (its 'loss' was read as a decrease cue and flipped the answer)
    low = _MANIPULATION_AFTER.sub(lambda m: m.group(1) + "agentx " + m.group(3), low)
    # 'Although X decreased A, B was raised': the concessive clause is not about B
    # The fix: remove the concessive part entirely before processing, so only the main clause is analyzed
    # Match the whole concessive phrase including nested commas like "Although rapamycin, an mtor blocker, decreased"
    low = re.sub(r"\b(?:although|though|even though)\b[^,]*(?:,[^,]*)*,\s*", "", low, flags=re.I)
    low = re.sub(r"\b(?:increasing|increased|higher|high|rising|escalating|decreasing|lower|low|graded)\s+"
                 r"(?:concentrations?|doses?|dosages?|amounts?)\b", " dose ", low)
    erx = _endpoint_regex(outcome_info, outcome_text)
    clauses = [c for c in re.split(r";|\bwhereas\b|\bwhile\b|,\s*but\b|\bbut\b|,\s*yet\b|\.\s|,\s*and\s+(?=the\b|a\b|an\b|its\b|their\b)", low) if c.strip()]
    hits = [(c, m) for c in clauses for m in erx.finditer(c)]
    # v24.3: an endpoint word inside an AGENT NAME ('anti-TNF therapy', 'an antagonist of IL-6') names what was given, not what
    # changed - dropped while another mention of the endpoint remains
    named = [(c, m) for c, m in hits if _in_agent_name(c, m)]
    if named and len(named) < len(hits):
        hits = [h for h in hits if h not in named]
    # v25: 'HDL cholesterol' is matched by every 'cholesterol'; the mention whose clause carries the endpoint's
    # distinctive word ('hdl') is read first ('Blood pressure was lower ..., as was LDL cholesterol, while HDL was higher')
    _dist = sorted({t for t in re.findall(r"[a-z0-9][a-z0-9\-]+", (outcome_text or "").lower())
                    if t not in _ENDPOINT_FILLER and t not in _GENERIC_MEASURE and len(t) >= 2 and not t.isdigit()
                    and t not in ("and", "the", "of", "in", "on", "at", "to", "for", "with")})
    # a short distinctive acronym of the endpoint ('HDL' of 'HDL cholesterol') is a mention of it even where the
    # endpoint lexicon only knows the generic word
    for _t in [x for x in _dist if len(x) <= 5 and re.search(r"[a-z]", x)]:
        for c in clauses:
            for m2 in re.finditer(rf"(?<![a-z0-9\-]){re.escape(_t)}(?![a-z0-9\-])", c):
                if not any(h[0] is c and h[1].start() <= m2.start() < h[1].end() for h in hits):
                    hits.append((c, m2))
    if len(hits) > 1 and _dist:
        def _carries(h):
            c, m = h
            win = c[max(0, m.start() - 25):m.end() + 5]
            return any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", win) for t in _dist)
        def _wrong_specific(h):
            c, m = h
            win = c[max(0, m.start() - 6):m.start()]
            return bool(re.search(r"\b(?:ldl|hdl|vldl|total|non-hdl)\s*$", win)) and not _carries(h)
        hits = sorted(hits, key=lambda h: (not _carries(h), _wrong_specific(h)))
        if any(_carries(h) for h in hits):
            hits = [h for h in hits if not _wrong_specific(h)] or hits
    if not hits:
        toks = [t for t in _key_tokens(outcome_text) if t not in _ENDPOINT_FILLER]
        if toks:
            trx = re.compile("|".join(rf"\b{re.escape(t)}" for t in toks), re.I)
            hits = [(c, m) for c in clauses for m in trx.finditer(c)]
    _ANAPHOR = (r"\b(?:this|the|that|such|these)\s+(?:\w+\s+)?(?:increase|elevation|upregulation|up-regulation|induction|rise|"
                r"enhancement)s?\b[^.;]{0,50}?\b(?:abolish|block|prevent|reduc|attenuat|revers|inhibit|suppress|decreas|"
                r"diminish|abrogat|blunt|counteract)\w*")
    if not hits:
        if re.search(_ANAPHOR, low):
            return {"direction": "decrease", "endpoint_found": False, "all": ["decrease"]} if collect else \
                {"direction": "decrease", "endpoint_found": False}
        return {"direction": None, "endpoint_found": False}
    own_words = set(_key_tokens(outcome_text))
    for clause, m in hits:
        pre_txt = clause[max(0, m.start() - 90):m.start()]
        am = re.search(r"\b(?:assess|measur|evaluat|examin|analy[sz]|investigat|monitor|quantif|determin)\w*\s+"
                       r"(?:(?:the|of|cell|cellular|and|or|its|their)\s+|[\w\-]+\s*,\s*)*$", pre_txt)
        if am and not (INC.search(pre_txt[am.start():]) or DEC.search(pre_txt[am.start():])):
            continue
        if re.search(r"\bdifferences? (?:in|of|between)\b[^.;]{0,50}\b(?:was|were|is|are)\s+(?:\w+\s+)?(?:greater|larger|bigger|"
                     r"smaller|higher|lower|more|less)\b", clause):
            continue
        ep = re.escape(m.group(0))
        # v24.3: 'decreased NEITHER proliferation NOR apoptosis' - the endpoint itself is negated
        if re.search(r"\b(?:neither|nor)\s+(?:the\s+|a\s+|an\s+)?(?:[\w\-]+\s+){0,2}$", clause[max(0, m.start() - 40):m.start()]):
            if collect:
                found.append("no_effect")
                continue
            return {"direction": "no_effect", "endpoint_found": True, "clause": clause.strip()[:240], "negated": True}
        # v24.3: 'Without metformin, proliferation increased' / 'In the absence of X, ...' describes the UNTREATED condition
        if re.match(r"\s*(?:without|in the absence of|absent)\s+[^,;]{1,60},", clause) and m.start() > clause.find(","):
            continue
        # v24.3: 'reversed / rescued / prevented the DECREASE (loss) in X' is a RISE relative to the untreated course
        if re.search(rf"\b(?:revers|rescu|restor|prevent|counteract|abolish|abrogat|block|mitigat|alleviat|offset|ameliorat|"
                     rf"overc[oa]m|normali[sz])\w*\s+(?:the\s+|a\s+|an\s+|this\s+|any\s+)?(?:[\w\-]+\s+){{0,3}}?"
                     rf"(?:decrease|reduction|loss|decline|fall|drop|suppression|inhibition|impairment|depletion|downregulation)s?"
                     rf"\s+(?:in|of)\s+(?:the\s+)?(?:[\w\-]+\s+){{0,3}}?{ep}", clause):
            d = "increase" if _base_polarity(outcome_info, m) > 0 else "decrease"
            if collect:
                found.append(d)
                continue
            return {"direction": d, "endpoint_found": True, "clause": clause.strip()[:240], "relative_to_control": True}
        if re.search(rf"\b(?:blunt|attenuat|abolish|abrogat|prevent|counteract|limit|curb|reduc|suppress|inhibit|revers|block|"
                     rf"mitigat|dampen|lessen|diminish|offset|halt)\w*\s+(?:the\s+|a\s+|an\s+|this\s+|any\s+)?(?:[\w\-]+\s+){{0,4}}?"
                     rf"(?:rise|increase|elevation|upregulation|up-regulation|induction|enhancement|gain|surge)s?\b(?:\s+(?:in|of)\s+(?:the\s+)?"
                     rf"(?:[\w\-]+\s+){{0,3}}?{ep})?", clause) and \
                re.search(rf"{ep}[^.;]{{0,40}}\b(?:rise|increase|elevation|upregulation|induction|enhancement|gain|surge)|"
                          rf"(?:rise|increase|elevation|upregulation|induction|enhancement|gain|surge)s?\s+(?:in|of)\s+(?:the\s+)?(?:[\w\-]+\s+){{0,3}}?{ep}",
                          clause) or \
                re.search(rf"\b(?:smaller|lesser|less|lower|reduced|attenuated|blunted|weaker)\s+(?:[\w\-]+\s+){{0,2}}?"
                          rf"(?:rise|increase)s?\s+(?:in|of)\s+(?:the\s+)?(?:[\w\-]+\s+){{0,3}}?{ep}", clause):
            d = "decrease" if _base_polarity(outcome_info, m) > 0 else "increase"
            if collect:
                found.append(d)
                continue
            return {"direction": d, "endpoint_found": True, "clause": clause.strip()[:240], "relative_to_control": True}
        chg = re.search(rf"\b(changes?|reductions?|decreases?|declines?|falls?|drops?|increases?|rises?)\s+(?:in|of)\s+(?:the\s+)?"
                        rf"(?:[\w\-]+\s+){{0,2}}?{ep}[^.;]{{0,90}}?\b(higher|greater|larger|bigger|more pronounced|lower|smaller|"
                        rf"less pronounced|less)\b", clause)
        if chg:
            noun, comp = chg.group(1).lower(), chg.group(2).lower()
            bigger = comp in ("higher", "greater", "larger", "bigger", "more pronounced")
            if noun.startswith(("reduc", "decreas", "declin", "fall", "drop")):
                # v25: a SMALLER fall than the comparator is a relative rise of the quantity ('the decrease in platelets
                # was lower in the intervention group'), the mirror of 'blunted the rise' - unless the endpoint IS the
                # fall itself ('FVC decline'), when the decline decreased
                if re.search(r"\b(?:reduc|decreas|declin|fall|drop|loss)\w*", outcome_text or "", re.I):
                    d = "increase" if bigger else "decrease"
                else:
                    d = "decrease" if bigger else "increase"
            elif noun.startswith(("increas", "rise")):
                d = "increase" if bigger else "decrease"
            else:
                rest = low.replace(chg.group(0), " ")
                down = re.search(r"\b(?:decreas\w*|reduc\w*|fell|fall\w*|drop\w*|declin\w*|moving into the low|into the low|"
                                 r"to (?:the )?low\w*|lower (?:ki|prolif))", rest)
                up = re.search(r"\b(?:increas\w*|rose|rise|rising|into the high|to (?:the )?high\w*)", rest)
                d = ("decrease" if down and not up else "increase" if up and not down else None) if bigger else None
            if d is None:
                continue
            if collect:
                found.append(d)
                continue
            return {"direction": d, "endpoint_found": True, "clause": clause.strip()[:240], "change_magnitude": True}
        if re.search(_ANAPHOR, clause):
            d = "decrease"
            if collect:
                found.append(d)
                continue
            return {"direction": d, "endpoint_found": True, "clause": clause.strip()[:240]}
        anti_adj = next((a for a in _ANTI_ADJ.finditer(clause) if a.start() <= m.start() < a.end() + 1), None)
        if anti_adj:
            before = clause[max(0, anti_adj.start() - 60):anti_adj.start()]
            lost = re.search(r"\b(?:no|lack\w*|without|lost|loss of|abolish\w*|abrogat\w*|block\w*|revers\w*|attenuat\w*|"
                             r"reduc\w*|diminish\w*|impair\w*|blunt\w*|prevent\w*|did not|failed to)\b[^.;]{0,40}$", before)
            d = "increase" if lost else "decrease"
            if outcome_info.get("family") == "apoptosis":
                d = flip(d)
            if collect:
                found.append(d)
                continue
            return {"direction": d, "endpoint_found": True, "clause": clause.strip()[:240]}
        span = re.escape(m.group(0))
        base = -1 if any(f == outcome_info.get("family") and neg.search(m.group(0)) for f, _, neg in FAMILY_RX) else 1
        inv = re.search(rf"{span}[^\w,;:()]+(?:\w+[^\w,;:()]+){{0,1}}({INVERSE_NOUN})\b|\b({INVERSE_NOUN})\W+(?:of|in)\W+(?:the\W+)?(?:\w+\W+){{0,2}}{span}", clause)
        # v24.2: 'anti-X therapy' where X is the agent (not the endpoint effect) should not flip direction
        # "Anti-TNF therapy" in subject position is naming an agent, not describing an effect
        anti_pattern = rf"\banti-?{span}|\banti-?(?:proliferat|tumou?r|metasta|apopto|inflammat)\w*"
        anti_match = re.search(anti_pattern, clause)
        # Exclude if the match is "anti-" + non-endpoint word followed by agent keywords (therapy, drug, etc.)
        # E.g., "anti-tnf therapy reduced X" or "compared with vehicle, anti-tnf therapy reduced"
        # This means it's an agent name, not an effect description
        anti = None
        if anti_match:
            # Check what comes after the match
            after_match = clause[anti_match.end():anti_match.end() + 50]
            # If there's no agent keyword after it, then the "anti-" might be a real effect modifier
            has_agent_keyword = bool(re.match(r"\s+(?:therapy|drug|treatment|medication|agent|intervention|compound)\b", after_match))
            # If "anti-X therapy/drug/..." format, it's naming an agent, so don't apply anti flip
            if not has_agent_keyword:
                anti = anti_match
        polarity = base * (-1 if (inv or anti) else 1)
        work = clause
        if inv:
            work = clause[:inv.start()] + " " + m.group(0) + " " + clause[inv.end():]
        if anti:
            work = work.replace(anti.group(0), " ")
        occ = [o.start() for o in re.finditer(re.escape(m.group(0)), work)]
        pos = min(occ, key=lambda o: abs(o - m.start())) if occ else work.find(m.group(0))

        def own_null(ne):
            after = work[ne.end():ne.end() + 50]
            obj = re.match(r"\s+(?:in|of|on|for)\s+([^,;.]{1,45})", after)
            return not obj or m.group(0) in obj.group(1) or pos >= ne.start()

        _verbal = [x for x in list(INC.finditer(work)) + list(DEC.finditer(work))
                   if not re.search(r"(?:tion|ment|ance|ence)s?$", x.group(0)) and x.group(0) not in own_words]
        if any(abs(ne.start() - pos) <= 90 and own_null(ne) for ne in NOEFFECT.finditer(work)) or \
                (not _verbal and any(own_null(ne) for ne in NOEFFECT.finditer(work))):
            if collect:
                found.append("no_effect")
                continue
            return {"direction": "no_effect", "endpoint_found": True, "clause": clause.strip()[:240]}
        allc = [(abs(x.start() - pos), "increase", x) for x in INC.finditer(work)] + \
               [(abs(x.start() - pos), "decrease", x) for x in DEC.finditer(work)]

        def weak(x):
            pre = work[max(0, x.start() - 1):x.start()]
            # v24.3: 'thereby upregulated tumour volume' / 'and downregulated X' - after a connective it is the verb, not an adjective
            if re.search(r"\b(?:thereby|thus|hence|and|consequently|subsequently)\s+$", work[max(0, x.start() - 16):x.start()]):
                return False
            return pre == "-" or (re.match(r"(?:down|up)-?regulated$", x.group(0)) and 0 <= pos - x.end() <= 2)

        allc = [c for c in allc if c[2].group(0).lower() not in own_words]

        def own_object(x):
            if x.start() <= pos:
                return False
            nxt = re.match(r"\s+([\w\-]+)", work[x.end():])
            return bool(nxt) and not re.fullmatch(r"(?:by|in|after|with|at|upon|following|significantly|markedly|dramatically|"
                                                   r"slightly|strongly|greatly|compared|than|when|from|to|during|over|"
                                                   r"relative|versus|vs|and|or|but|while|whereas|\d[\w.%]*|was|were|is|"
                                                   r"are|only|further|also|substantially|considerably|significantly)", nxt.group(1), re.I)

        if len(allc) > 1:
            allc = [c for c in allc if not own_object(c[2])] or allc

        def listed(x):
            w = x.group(0).lower()
            if not re.search(r"(?:ion|ment|ance)s?$", w):
                return False
            between = work[x.end():pos] if x.end() <= pos else work[pos + len(m.group(0)):x.start()]
            return bool(re.fullmatch(r"\s*(?:,|and|or|&)?\s*(?:[a-z\-]+\s*,\s*)*(?:and\s+|or\s+)?", between))

        if len(allc) > 1:
            allc = [c for c in allc if not listed(c[2])] or allc
        strong = [c for c in allc if not weak(c[2])]
        allc = strong or allc
        cues = [(d, k) for d, k, _ in allc if d <= 120]
        if not cues and len({k for d, k, _ in allc if d <= 300}) == 1:
            cues = [(d, k) for d, k, _ in allc if d <= 300]
        if not cues and outcome_info.get("family") == "clinical_response" and OCCUR.search(work):
            cues = [(0, "increase")]
        if not cues and outcome_info.get("family") == "glycaemic" and re.search(
                r"\banti-?hyperglyc\w*|\bantidiabetic\b|glucose-lowering|\bhypoglyc(?:a)?emic (?:effect|action|agent|drug)", work):
            cues = [(0, "decrease")]
        if cues and inv:
            ipos = inv.start()
            cues = [c for c in allc if abs(c[2].start() - ipos) <= 30 and c[2].start() != ipos]
            cues = [(abs(c[2].start() - ipos), c[1]) for c in cues]
        if cues:
            raw = sorted(cues)[0][1]
            # v24.3: 'did not diminish X', 'failed to lower X', 'was not significantly increased' - the change is DENIED.
            # Read before as the denied direction (196 of 1,200 negated fuzzer sentences); now a null result.
            _best = min((c for c in allc if c[0] == sorted(cues)[0][0] and c[1] == raw), key=lambda c: c[0], default=None)
            # v25: 'improved / better / worse' on an endpoint where LOWER is better ('blood pressure improved',
            # 'HbA1c was better controlled', 'pain was worse') describes a fall / rise of the endpoint the other way
            if _best is not None and re.match(r"(?:improv|better|superior|worse|inferior|worsen|deteriorat|ameliorat)",
                                              _best[2].group(0).lower()) and _LOWER_IS_BETTER.search(outcome_text or m.group(0)) \
                    and not re.search(r"-free\b|\bfree of\b|\bsurvival\b|\bremission\b|\bresponse\b|\bcontrol\b",
                                      f"{outcome_text} {m.group(0)}", re.I):
                raw = flip(raw)
            if _best is not None and _NEGATED_CUE.search(work[max(0, _best[2].start() - 40):_best[2].start()]):
                if collect:
                    found.append("no_effect")
                    continue
                return {"direction": "no_effect", "endpoint_found": True, "clause": clause.strip()[:240], "negated": True}
        elif inv or anti:
            raw = "increase"
        else:
            continue
        d = raw if polarity > 0 else flip(raw)
        if re.search(r"\b(?:non-?treated|untreated|control|placebo|vehicle|non-?users?)\b[^.;]{0,60}?\b(?:compared (?:to|with)|than|"
                     r"versus|vs\.?)\s[^.;]{0,50}?\b(?:[\w-]*treated|users?|treatment)\b", clause):
            d = flip(d)
        else:
            # v24.2: 'X was HIGHER in controls than in the <agent> group' states a FALL under treatment. The control
            # comes first and the other term is anything that is not itself a control
            _m = re.search(r"\b(?:non-?treated|untreated|control|placebo|vehicle|non-?users?)s?\b[^.;]{0,60}?\bthan\b\s+"
                           r"(?:in |that in |with |those in )?(?:the )?(\w[\w\-]*)", clause)
            if _m and not re.match(r"(?:non-?treated|untreated|control|placebo|vehicle|baseline|non-?users?|normal|healthy|"
                                   r"wild|sham)", _m.group(1), re.I) and re.search(r"\b(?:higher|lower|greater|less|more|"
                                   r"larger|smaller|fewer|elevated|reduced|increased|decreased)\b[^.;]{0,50}?\b(?:non-?treated|"
                                   r"untreated|control|placebo|vehicle|non-?users?)", clause):
                d = flip(d)
        if collect:
            found.append(d)
            continue
        return {"direction": d, "endpoint_found": True, "clause": clause.strip()[:240]}
    if collect:
        return {"direction": found[0] if found else None, "endpoint_found": True, "all": found}
    return {"direction": None, "endpoint_found": True}


def sentence_direction(text: str) -> Optional[str]:
    """v25: the direction of a sentence when its endpoint cannot be located in it (the extractor's or annotator's
    wording differs from the paper's). Only an UNAMBIGUOUS sentence gets an answer: a no-difference statement, or
    change words that all point the same way. Callers label this reading as a fallback."""
    low = _squash_symbols(strip_adjusters((text or "").lower()))
    low = re.sub(r"\b(?:although|though|even though)\b[^,]*,\s*", "", low)
    if NOEFFECT.search(low):
        return "no_effect"
    low = _AGENTIVE_NOUN.sub("agentx", low)
    ups = [m.group(0) for m in INC.finditer(low)]
    downs = [m.group(0) for m in DEC.finditer(low)]
    if ups and not downs:
        return "increase"
    if downs and not ups:
        return "decrease"
    return None


def locate_quote(quote: str, sections: List[Dict[str, Any]], fallback_text: str) -> Dict[str, Any]:
    best = {"section_type": "unknown", "section_title": None, "score": 0.0, "context": "", "text": ""}
    pool = list(sections or []) or [{"section_type": "text", "title": "Text", "text": fallback_text or ""}]
    for s in pool:
        g = quote_grounding(quote, s.get("text") or "")
        if g > best["score"]:
            best = {"section_type": s.get("section_type") or "other", "section_title": s.get("title"), "score": g,
                    "text": s.get("text") or ""}
    if best["score"] == 0.0 and fallback_text:
        best.update({"text": fallback_text, "score": quote_grounding(quote, fallback_text)})
    ns, nq = _norm_for_match(best["text"]), _norm_for_match(quote)
    anchor = ns.find(nq[:40]) if nq else -1
    if anchor < 0 and len(nq) > 60:
        anchor = ns.find(nq[-40:])
    best["context"] = ns[max(0, anchor - 600): anchor + len(nq) + 600] if anchor >= 0 else ns[:1200]
    best.pop("text", None)
    return best


def _squash_symbols(s: Optional[str]) -> str:
    return re.sub(r"\b([a-z])\s+(\d+)\s+([a-z])\b", r"\1\2\3", (s or "").lower())


def _key_tokens(s: Optional[str]) -> List[str]:
    low = _squash_symbols(s)
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", low) if w not in GENERIC_INTERVENTION]
    words += [w for w in re.findall(r"\b[a-z]\d+[a-z]?\b", low) if w not in words]
    return words


def _magnitudes(statement: str) -> List[str]:
    s = statement or ""
    s = re.sub(r"\bp\s*[<=>]+\s*[\d.eE\-x^ ]+", " ", s, flags=re.I)
    s = re.sub(r"\bn\s*=\s*\d+", " ", s, flags=re.I)
    s = re.sub(r"\b(19|20)\d{2}\b", " ", s)
    out = re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|-?fold|x\b|times)", s, flags=re.I)
    out += re.findall(r"(?:by|to|of)\s+(\d+(?:\.\d+)?)\s*(?:mg|µg|ug|ng|mm|µm|um|nm|g|kg|ml|l)\b", s, flags=re.I)
    return out


def _resolve_intervention(claim: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    iv = (claim.get("intervention") or claim.get("subject") or "").strip()
    if not iv or len(iv) > 6 or " " in iv:
        return claim
    src = ctx.get("source_text") or ""
    full = next((f for a, f in (ctx.get("abbreviations") or {}).items() if a.lower() == iv.lower()), None)
    if not full:
        m = re.search(rf"\b([A-Za-z][\w\-]{{4,}})\s*\(\s*{re.escape(iv)}\s*\)", src, re.I)
        if m and m.group(1)[0].lower() == iv[0].lower():
            full = m.group(1)
    if not full:
        cand = {w.lower() for w in re.findall(rf"\b({re.escape(iv[:3])}[a-z]{{4,}})\b", f"{ctx.get('title') or ''} {src[:4000]}", re.I)}
        cand = {w for w in cand if w.startswith(iv.lower())}
        if len(cand) == 1:
            full = cand.pop()
    return {**claim, "intervention": full, "intervention_as_written": iv} if full else claim


_NOT_AN_AGENT = {"this", "that", "which", "it", "and", "or", "the", "treatment", "cells", "also", "further", "significantly",
                 "markedly", "strongly", "partially", "completely", "both", "not", "to", "may", "might", "could", "can"}
_COUNTER = (r"(?:disrupt|inhibit|block|reduc|suppress|attenuat|abolish|prevent|revers|counteract|antagoni[sz]|abrogat|"
            r"impair|blunt|diminish|decreas|interfer|rescu|mitigat|alleviat)\w*")


def _resolve_stimulus(claim: Dict[str, Any]) -> Dict[str, Any]:
    iv = (claim.get("intervention") or claim.get("subject") or "").strip()
    quote = claim.get("quote") or ""
    if not iv or len(iv) > 40:
        return claim
    stim = rf"{re.escape(iv)}[- ](?:induced|stimulated|mediated|triggered|evoked|dependent)"
    if not re.search(stim, quote, re.I):
        return claim
    m = re.search(rf"\b([A-Za-z][\w\-]{{2,}})\s+(?:(?:significantly|markedly|strongly|partially|completely|further|also|"
                  rf"effectively|potently|dose-dependently|substantially)\s+)?{_COUNTER}\s+(?:the\s+|this\s+)?{stim}", quote, re.I)
    if not m or m.group(1).lower() in _NOT_AN_AGENT or m.group(1).lower() == iv.lower():
        return claim
    return {**claim, "intervention": m.group(1), "stimulus": iv, "intervention_as_written": iv}


_CUE_ANIMAL = re.compile(r"\bin vivo\b|xenograft\w*|\bnude mice\b|\bmice\b|\bmouse\b|\brats?\b|tumou?r-bearing|\banimals?\b|"
                         r"\bzebrafish\b|orthotopic|syngeneic", re.I)
_CUE_CELL = re.compile(r"\bin vitro\b|\bcell lines?\b|\bcultured?\b|\b(?:MCF-?7|MDA-MB-\d+|T-?47D|4T1|SK-?BR-?3|HeLa|A549|HepG2)\b|"
                       r"\bcells\b", re.I)
_CUE_HUMAN = re.compile(r"\bpatients?\b|\bwomen\b|\bparticipants?\b|\bsubjects\b|\bcohort\b", re.I)
_CELLULAR_ENDPOINT = re.compile(r"\bcell (?:migration|proliferation|growth|viability|invasion|death|cycle)|\bcolon(?:y|ies)|"
                                r"\bclonogenic|\bviability\b|\bcck-?8\b|\bmtt\b|\bwound[- ]heal|\btranswell|\bspheroid", re.I)
# v24 fix 3/4 - exposures that damage, endpoints that mean damage, agents that rescue, and cancer words
_STRESSOR_RX = re.compile(r"\b(?:uv-?[abc]?|ultraviolet(?:[- ]?[abc])?|h2o2|hydrogen peroxide|oxidative stress|"
                          r"\blps\b|lipopolysaccharide|hypoxia|ischemi\w*|irradiation|ionizing radiation|x-?ray|"
                          r"tbhp|tert-butyl hydroperoxide|paraquat|rotenone|\bmnng\b|\bros\b|"
                          r"heat (?:stress|shock)|cold stress|salt stress|drought(?: stress)?|"
                          r"high glucose|palmitate|\bapap\b|acetaminophen|\bccl4\b)\b", re.I)
_DAMAGE_RX = re.compile(r"\b(?:dna damage|damage|apopto\w*|cell death|necrosis|\bros\b|reactive oxygen|"
                        r"oxidative stress|lipid peroxidation|\bmda\b|malondialdehyde|caspase|cytotoxic\w*|"
                        r"inflammation|senescence|\bgamma-?h2ax\b|γh2ax|comet (?:tail|assay))\b", re.I)
_RESCUER_RX = re.compile(r"\b(?:[\w\-]*mimic|antagomi\w*|\bsi[Rr][Nn][Aa]\b|\bsh[Rr][Nn][Aa]\b|overexpress\w*|"
                         r"[\w\-]*inhibitor|[\w\-]*agonist|[\w\-]*antagonist|[\w\-]*supplementation|"
                         r"pretreat\w*|pre-?incubat\w*|\bN-?acetylcystein\w*|\bNAC\b|vitamin [A-E]\b|"
                         r"[\w\-]{3,}\s+treatment)\b")
_CANCER_RX = re.compile(r"\b(?:cancers?|carcinomas?|tumou?rs?|neoplas\w*|melanomas?|sarcomas?|lymphomas?|leuk(?:a)?emi\w*|"
                        r"myelomas?|glioma\w*|metasta\w*|oncolog\w*|malignan\w*|xenografts?)\b", re.I)
_IMMUNE_INFILTRATE_RX = re.compile(r"\b(?:infiltrat\w*|recruit\w*|homing|migration of (?:[\w\-]+ ){0,2}"
                                   r"(?:t|cd8\+?|cd4\+?|immune) cells?|cd8\+? t cells?|cytotoxic t|lymphocyte\w*|"
                                   r"immune cells?|whole-?mount staining)\b", re.I)

# v24 fix 1b - 'melanocyte death', 'keratinocyte survival', 'cell death': survival of CELLS, not of patients
_CELL_DEATH_RX = re.compile(r"\b(melanocytes?|keratinocytes?|fibroblasts?|hepatocytes?|cardiomyocytes?|neurons?|"
                            r"lymphocytes?|macrophages?|monocytes?|myocytes?|osteoblasts?|chondrocytes?|"
                            r"endothelial cells?|epithelial cells?|tumou?r cells?|cancer cells?|cells?)\s+"
                            r"(?:death|survival|viability|loss|apoptosis)", re.I)

_CLAUSE_SPLIT = re.compile(r"\s*(?:;|,?\s+(?:and|while|whereas|but)\s+(?=[a-z]+(?:ed|es|s)?\b))\s*", re.I)


def _endpoint_clause_model(quote: str, outcome_text: str) -> Optional[Dict[str, str]]:
    q = quote or ""
    kinds = sum(bool(rx.search(q)) for rx in (_CUE_ANIMAL, _CUE_HUMAN)) + bool(re.search(r"\bin vitro\b", q, re.I))
    if kinds < 2:
        return None
    fam = canonical_outcome(outcome_text)
    rx = _endpoint_regex(fam, outcome_text)
    clauses = [c for c in _CLAUSE_SPLIT.split(q) if c and c.strip()]
    for c in clauses:
        if rx.search(c):
            if _CUE_ANIMAL.search(c):
                return {"model_type": "animal", "source": "the clause of the quote that reports this endpoint (in vivo)"}
            if re.search(r"\bin vitro\b", c, re.I) or (_CUE_CELL.search(c) and not _CUE_HUMAN.search(c)):
                return {"model_type": "cell_culture", "source": "the clause of the quote that reports this endpoint (in vitro)"}
            if _CUE_HUMAN.search(c):
                return {"model_type": "human_observational", "source": "the clause of the quote that reports this endpoint (people)"}
    return None


_GENETIC_BG = r"(?:[- ]?(?:knock-?down|knock-?out|KO|KD|deficient|depleted|silenced|null|overexpressing|-/-))"


def _resolve_genetic_background(claim: Dict[str, Any]) -> Dict[str, Any]:
    iv = (claim.get("intervention") or claim.get("subject") or "").strip()
    quote = claim.get("quote") or ""
    if not iv or len(iv) > 30 or not re.search(rf"\b{re.escape(iv)}\s*{_GENETIC_BG}", quote, re.I):
        return claim
    rest = re.sub(rf"\b{re.escape(iv)}\s*{_GENETIC_BG}", " ", quote, flags=re.I)
    if re.search(rf"\b{re.escape(iv)}\b", rest, re.I):
        return claim
    m = re.search(r"\b([A-Za-z][\w\-]{3,})(?:\s+(?:treatment|exposure|administration))?\s+(?:significantly\s+|markedly\s+|"
                  r"strongly\s+|further\s+|also\s+)?(?:reduc|inhibit|decreas|suppress|impair|block|increas|induc|promot|enhanc)\w*",
                  quote, re.I)
    if not m or m.group(1).lower() in _NOT_AN_AGENT or m.group(1).lower() == iv.lower() or \
            m.group(1).lower() in ("concentrations", "doses", "levels", "control", "cells", "analysis", "staining"):
        return claim
    return {**claim, "intervention": m.group(1), "genetic_background": iv, "intervention_as_written": iv}


_GENERIC_INTERVENTION = re.compile(r"^\s*(?:the |its |this |their )?(?:anti-?\w+ )?(?:efficacy|effects?|effectiveness|activity|"
                                   r"actions?|treatment|therapy|drug|compound|agent|response|impact|potency|intervention|"
                                   r"exposure|use|administration|it|they|this|these)\s*$", re.I)
_MODIFIED_EFFECT = r"(?:abrogat|abolish|block|revers|attenuat|prevent|blunt|diminish|repress|impair|negat|counteract|antagoni[sz])\w*" \
                   r"\s+(?:the\s+)?(?:\w+[- ]){0,2}(?:effects?|efficacy|actions?|ability|benefit\w*)\s+of\s+"
_BASELINE_CMP = re.compile(r"\b(?:lower|higher|greater|less|more|weaker|stronger)\s+than\s+(?:that|those)\s+(?:in|of)\b", re.I)
_OWN_WORK = re.compile(r"\b(?:we|our|us|here|in this study|the present study|this study|the current study|in this cohort|"
                       r"in our (?:study|cohort|trial|analysis)|these (?:results|data|findings)|in the present)\b", re.I)
_GENERIC_ASSOC = re.compile(r"\b(?:is|are) (?:associated|linked|correlated)\b|\b(?:has|have) been (?:associated|linked)\b",
                            re.I)
# v26.2: a reference-list entry read from a PDF ('... cancer stem cell growth, Proc' / 'Smith AB, Jones CD.' /
# '2013;110:972-7' / 'doi:10.') is a cited paper's title, never a finding
_BIBLIO = re.compile(r",\s*(?:Proc|Nature|Science|Cell|Lancet|JAMA|BMJ|PLoS|J|Ann|Int|Br|Am|Eur|Clin|Cancer Res|Oncotarget|"
                     r"Oncogene|Breast Cancer Res)\b[\w .]{0,25}$|\b\d{4};\s*\d+(?:\(\d+\))?:\s*\d+|\bdoi:\s*10\.|"
                     r"^(?:[A-Z][a-z]+ [A-Z]{1,3}, ){2,}")
_GENERIC_PRESENT = re.compile(r"^\s*(?:\w+[\s,-]+){0,4}?(?:reduces|increases|inhibits|suppresses|decreases|lowers|improves|"
                              r"promotes|induces|activates|enhances|impairs|blocks|prevents|is known|are known|can|may|might|could)\b",
                              re.I)
_SPECULATIVE_EXTRA = re.compile(
    r"\b(?:may|might|could|can)\s+(?:\w+ly\s+)?(?:negatively\s+|positively\s+|also\s+)?(?:impact|affect|influence|contribute|"
    r"show|exert|act|lead|play|explain|underlie|enhance|reduce|increase|improve|inhibit|promote|modulate)\w*\b|"
    r"(?<!our )(?<!these )(?<!present )(?<!this )\b(?:reports|studies|evidence|literature) (?:suggest|indicate|show)s?\b", re.I)
_RETRACTED = re.compile(r"^\s*(?:\[?\s*RETRACTED\b|Retracted\s*(?:article|:)|Retraction(?:\s+note)?\s*(?::|to|of)|"
                        r"WITHDRAWN\b|Withdrawn\s*:|Expression of concern)", re.I)


def is_retracted(title: Optional[str], abstract: Optional[str] = None) -> bool:
    if _RETRACTED.search(title or ""):
        return True
    return bool(re.search(r"\bthis article has been retracted\b|\bretraction notice\b", (abstract or "")[:600], re.I))


_READOUT_IV = re.compile(r"\b(?:phosphorylation|expression|levels?|activity|activation|signal(?:l)?ing|pathway|translocation|"
                         r"methylation|acetylation|ubiquitination|secretion|production|abundance|transcription)\b", re.I)
_GENE_IV = re.compile(r"^\s*(?:the\s+)?([A-Z][A-Za-z0-9\-]{1,9})(?:\s+(?:gene|protein|mRNA|transcript))?\s*$")
_MANIPULATION = re.compile(r"\b(?:knock-?down|knock-?out|si-?RNA|sh-?RNA|CRISPR|over-?express\w*|inhibitor|antagonist|agonist|"
                           r"deletion|deficien\w*|silenc\w*|mutant|transfect\w*|treatment|recombinant|exogenous)\b", re.I)
_ROUTE = re.compile(r"\b(?:via|through|by (?:inhibiting|suppressing|blocking|activating|targeting|reducing|increasing|"
                    r"downregulating|upregulating)|mediated by|dependent on|in a ([A-Za-z0-9\-]+)-dependent manner|"
                    r"(?:inhibition|suppression|activation|blockade|downregulation|upregulation) of)\s+(?:the\s+)?"
                    r"([A-Z][A-Za-z0-9\-]{1,9}(?:/[A-Z][A-Za-z0-9\-]{1,9})*)", re.I)


def _resolve_mediator(claim: Dict[str, Any], exposures: List[str]) -> Dict[str, Any]:
    iv = (claim.get("intervention") or claim.get("subject") or "").strip()
    quote = claim.get("quote") or ""
    if not iv or not exposures or _MANIPULATION.search(iv) or \
            any(w in iv.lower() for e in exposures for w in e.lower().split() if len(w) >= 4):
        return claim
    is_readout = bool(_READOUT_IV.search(iv))
    gm = _GENE_IV.match(iv)
    is_gene = bool(gm) and bool(re.search(r"\d|[A-Z]{3,}", gm.group(1))) and iv.lower() not in [e.lower() for e in exposures]
    if not (is_readout or is_gene):
        return claim
    agent = exposures[0]
    stem = re.escape(agent.lower().split()[0][:max(4, len(agent.split()[0]) - 1)])
    acts = (r"(?:inhibit|reduc|decreas|suppress|block|attenuat|lower|activat|increas|induc|enhanc|stimulat|promot|abolish|"
            r"impair|abrogat|diminish|prevent)\w*")
    if not re.search(rf"\b{stem}\w*(?:[- ]treated)?[^.;]{{0,80}}?\b{acts}|(?:treated|treatment|incubated|exposed) with "
                     rf"(?:\w+ ){{0,3}}{stem}|\b{stem}\w*[- ](?:treated|fed|exposed)|\b(?:by|after|upon|following|with) "
                     rf"(?:\w+ ){{0,2}}{stem}", quote, re.I):
        return claim
    return {**claim, "intervention": agent, "mediator": iv, "intervention_as_written": iv}


def mechanism_evidence(claim: Dict[str, Any]) -> Optional[Dict[str, str]]:
    q = claim.get("quote") or ""
    if claim.get("genetic_background"):
        return {"mediator": claim["genetic_background"], "level": "M3",
                "kind": f"required: the effect was lost when {claim['genetic_background']} was knocked down / out"}
    m = _ROUTE.search(q)
    if m:
        med = m.group(2)
        rest = q[m.start(2):]
        am = re.match(r"(?:inhibition|suppression|activation|blockade|down-?regulation|up-?regulation|regulation|targeting|"
                      r"modulation|induction|inactivation)\s+of\s+(?:the\s+)?([A-Za-z][\w\-/]*)", rest, re.I)
        if am:
            med, rest = am.group(1), rest[am.start(1):]
        if med.islower():
            nm = re.match(r"[a-z][\w\-]*(?:\s+(?!and\b|or\b|as\b|which\b|in\b|of\b|to\b|that\b|with\b|was\b|were\b|is\b|"
                          r"are\b)[a-z][\w\-]*){0,2}", rest)
            med = nm.group(0) if nm else med
            if " " not in med and re.fullmatch(r"[a-z]+(?:ive|al|ic|ous|ar|ary|ed|ing)", med):
                med = ""
        if med and not re.fullmatch(r"(?:the|a|an|this|these|its|their|cell|cells)", med, re.I):
            route = q[m.start():m.start(2)].strip() or "via"
            return {"mediator": med, "level": "M2", "kind": f"stated route in the source ('{route} {med}') - association, "
                                                            f"not shown to be required"}
    if claim.get("mediator"):
        return {"mediator": claim["mediator"], "level": "M2",
                "kind": f"{claim['mediator']} changed with the agent (read-out) - association, not shown to be required"}
    return None


_CHANGE_WORDS = re.compile(r"\b(?:increas\w*|decreas\w*|reduc\w*|lower\w*|rais\w*|improv\w*|worsen\w*|inhibit\w*|suppress\w*|"
                           r"associated|affect\w*|did not|was|were|versus|vs\.?|compared|significant\w*)\b", re.I)


def _c40(s: Optional[str]) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= 40 else s[:39] + "…"


_MICROBE_SUBJECT = re.compile(r"\b(?:bacteri\w*|microb\w*|fung\w*|yeasts?|strains?|isolates?|cultures?|colon(?:y|ies)|"
                              r"cfu|cell[- ]forming|mycobacteri\w*|escherichia|salmonella|staphylococc\w*|streptococc\w*|"
                              r"pseudomonas|candida|aspergillus|saccharomyces|pathogens?|inocul\w*|viral|virus(?:es)?|"
                              r"parasites?|biofilms?|planktonic|intracellular bacteri\w*)\b", re.I)


# --------------------------------------------------------------------------- bibliography entries are not findings
# A reference list contains other papers' TITLES, which read exactly like findings ('Long-term metformin use is
# associated with decreased risk of breast cancer, Diabetes Care, 2010, Vol. 33, pp. 1304-1308'). One of those was
# published as a study's own verified result and reached the bottom line. A citation has a bibliographic tail that a
# result never has: a journal name with a year, a volume, page numbers, a DOI or a PMID.
_BIB_TAIL = re.compile(
    r"\bvol\.?\s*\d+|\bpp?\.\s*\d+|\bdoi:\s*10\.\d{4,}|\bPMID:?\s*\d{6,}|"
    r"(?:19|20)\d\d\s*;\s*\d+\s*(?:\(\s*\d+[\w-]*\s*\))?\s*:\s*\d+|"          # 2010;33(6):1304-8
    r",\s*(?:19|20)\d\d\s*,\s*(?:vol|no|pp|p)\b|"                                      # , 2010, Vol. 33
    r"\bet al\.?\s*,?\s*(?:19|20)\d\d\s*[;,.]|"                                        # Smith et al., 2010;
    r"\b(?:available (?:from|at)|accessed(?: on)?|in press|epub ahead)\b", re.I)
# a journal-looking name immediately before a year ('..., Diabetes Care, 2010')
_BIB_JOURNAL = re.compile(r",\s*(?:[A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,4})\s*,\s*(?:19|20)\d\d\b")
_BIB_NUMBERED = re.compile(r"^\s*\[?\d{1,3}[\].]\s+[A-Z]")
_BIB_SECTION = re.compile(r"\b(?:references?|bibliograph\w*|literature cited|works cited|further reading)\b", re.I)


def is_bibliography_quote(quote: str, section_title: str = "", section_type: str = "") -> Optional[str]:
    """Why this quote is a reference-list entry rather than a result, or None when it is not one.

    Two independent signals, because a PDF's section labels are unreliable: the quote's own bibliographic shape, and
    the section it was found in.
    """
    q = (quote or "").strip()
    if not q:
        return None
    if _BIB_SECTION.search(section_title or "") or _BIB_SECTION.search(section_type or ""):
        return "the quote comes from the paper's reference list, so it is another paper's title, not a result"
    tail = _BIB_TAIL.search(q)
    if tail:
        return (f"the quote is a reference-list entry (it ends in bibliographic detail: '{tail.group(0).strip()}'), "
                f"so it is another paper's title, not this study's result")
    if _BIB_JOURNAL.search(q) and _BIB_NUMBERED.match(q):
        return "the quote is a numbered reference-list entry, not this study's result"
    return None


def _repair_endpoint(claim: Dict[str, Any], quote: str) -> Dict[str, Any]:
    out = (claim.get("measurement") or claim.get("outcome") or "").strip()
    if not out or not quote:
        return claim
    ot = _key_tokens(out)
    it = _key_tokens(claim.get("intervention") or claim.get("subject") or "")
    exposure_only = bool(ot and it and all(t in it or t in ("use", "usage", "users", "user", "intake", "exposure",
                                                              "administration", "treatment", "therapy", "group") for t in ot))
    sentence_like = len(out.split()) >= 6 and bool(_CHANGE_WORDS.search(out))
    if not (exposure_only or sentence_like):
        return claim
    low = quote.lower()
    fams = {}
    for fam, rx, _neg in FAMILY_RX:
        m = rx.search(low)
        if m and fam not in fams:
            fams[fam] = m
    if len(fams) != 1:
        return claim
    fam, m = next(iter(fams.items()))
    endpoint = quote[m.start():m.end()]
    key = "measurement" if claim.get("measurement") else "outcome"
    return {**claim, key: endpoint, "outcome_as_written": out, "endpoint_repaired": True}


def audit_claim(claim: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    quote = (claim.get("quote") or "").strip()
    original_iv = (claim.get("intervention") or "").strip()
    if original_iv:
        from claim_extraction import clean_subject as _clean
        trimmed = _clean(original_iv) or original_iv
        change_word = re.match(r"\s*(?:a |the )?(?:decrease|increase|reduction|elevation|loss|gain|change|rise|drop|fall)s?\s+"
                               r"(?:in|of)\b", original_iv, re.I)
        if trimmed != original_iv and original_iv.lower().startswith(trimmed.lower()) and not change_word:
            claim = {**claim, "intervention": trimmed}
    claim = _resolve_intervention(claim, ctx)
    claim = _resolve_stimulus(claim)
    claim = _resolve_genetic_background(claim)
    try:
        from evidence_identity import question_frame as _qf
        _expo = [e for e in ((ctx.get("frame") or _qf(ctx.get("question") or "")).get("exposures") or []) if e]
    except Exception:
        _expo = []
    claim = _resolve_mediator(claim, _expo)
    claim = _repair_endpoint(claim, quote)
    statement = claim.get("statement") or ""
    outcome_text = expand_endpoint(claim.get("measurement") or claim.get("outcome") or "", ctx.get("abbreviations"))
    direction = (claim.get("effect_direction") or claim.get("direction") or "unclear").lower()
    checks: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    reasons: List[str] = []

    def mark(name: str, ok: Optional[bool], detail: str, level: str = "quote") -> None:
        checks[name] = {"ok": ok, "level": level if ok else None, "detail": detail}
        if ok is False:
            reasons.append(f"{name}: {detail}")

    # 0. textual grounding
    loc = locate_quote(quote, ctx.get("sections") or [], ctx.get("source_text") or "")
    g = max(float(claim.get("grounding_score") or 0.0), quote_grounding(quote, ctx.get("source_text") or "")) if quote else 0.0
    mark("textual_grounding", g >= GROUNDING_THRESHOLD, f"quote found in paper (score {g:.2f})" if g >= GROUNDING_THRESHOLD
         else f"quote not found in the paper text (score {g:.2f} < {GROUNDING_THRESHOLD})")
    if re.match(r"^\s*\d{1,3}\s*\)", quote) or re.search(r"[pP]\s*[<=>]\s*0?\.\s*$", quote.rstrip(" .\"'”")) \
            or re.search(r"\(\s*[pP]\s*[<=>]\s*0\.?\s*$", quote):
        mark("textual_grounding", False, "the quote is a broken sentence fragment (split at a decimal point) - re-extract it")
    # v25: a retracted paper / one under an expression of concern cannot support a claim (publication_status.py);
    # a corrected paper keeps its claims, with the notice shown
    _ps = next((f for f in (claim.get("flags") or []) if isinstance(f, str) and f.startswith("publication_status:")), None)
    if _ps and re.match(r"publication_status:(?:retracted|expression_of_concern)", _ps):
        mark("publication_status", False, _ps.split(":", 1)[1].strip())
    elif _ps:
        checks["publication_status"] = {"ok": None, "level": None, "detail": _ps.split(":", 1)[1].strip()}
    # v25: the two independent readers disagreed on this finding -> a person decides (it stays visible, quarantined)
    _dis = next((f for f in (claim.get("flags") or []) if isinstance(f, str) and f.startswith("reader_disagreement")), None)
    if _dis:
        mark("independent_reader", False, _dis.split(":", 1)[1].strip() + " - sent to the review queue")
    # a reference-list entry is never a finding of this paper
    _bib = is_bibliography_quote(quote, loc.get("section_title") or "", loc.get("section_type") or "")
    if _bib:
        mark("this_study", False, _bib)
    if re.fullmatch(r"\s*(?:improved|improvement|reduced|increased|decreased|enhanced|better|worse|higher|lower|elevated|"
                    r"anti-?[\w\-]+|significant|positive|negative|beneficial|favou?rable|effect|effects|outcomes?|response)\s*",
                    claim.get("measurement") or claim.get("outcome") or "", re.I):
        mark("endpoint", False, f"'{claim.get('measurement') or claim.get('outcome')}' is a describing word, not a measured "
                                f"quantity - the claim has no endpoint")
    _ot = _key_tokens(claim.get("measurement") or claim.get("outcome") or "")
    _it = _key_tokens(claim.get("intervention") or claim.get("subject") or "")
    if _ot and _it and all(t in _it or t in ("use", "usage", "users", "user", "intake", "exposure", "administration",
                                             "treatment", "therapy", "group") for t in _ot):
        mark("endpoint", False, f"the outcome '{claim.get('measurement') or claim.get('outcome')}' names the exposure itself, "
                                f"not a measured endpoint")
    context = loc["context"]
    nquote = _squash_symbols(_norm_for_match(quote))

    # 1. this study's own result
    cited = bool(CITATION_MARK.search(quote)) and loc["section_type"] in ("introduction", "discussion", "other", "unknown", "text")
    cited = cited or bool(PRIOR_WORK.search(quote))
    aim = AIM_OR_SPECULATION.search(quote) or _SPECULATIVE_EXTRA.search(quote)
    background = loc["section_type"] == "introduction"
    try:      # v24.2: a BACKGROUND/OBJECTIVE line of a structured abstract, or an 'it has been reported' sentence, is prior work
        from section_roles import abstract_role, is_prior_work_phrase
        _role = abstract_role(quote, ctx.get("source_text") or "")
        if _role in ("background", "aim") or (loc["section_type"] in ("abstract", "unknown", "text", "other")
                                              and is_prior_work_phrase(quote)):
            cited = True
    except Exception:
        pass
    discussion_generic = loc["section_type"] == "discussion" and not _OWN_WORK.search(quote) and \
        bool(_GENERIC_PRESENT.search(quote))
    # v26.2: a general present-tense association ('Metformin treatment is associated with reduced ... mortality')
    # outside the abstract / results, with no number and no 'we / this study', is background, not this study's result
    discussion_generic = discussion_generic or (
        loc["section_type"] in ("discussion", "other", "unknown", "text", "introduction") and not _OWN_WORK.search(quote)
        and not re.search(r"\d", quote) and bool(_GENERIC_ASSOC.search(quote)))
    biblio = bool(_BIBLIO.search(quote))
    background = background or discussion_generic
    if is_retracted(ctx.get("title"), ctx.get("abstract")):
        mark("this_study", False, "the paper is RETRACTED - its results are not evidence")
    elif (ctx.get("classification") or {}).get("evidence_type") == "review_meta":
        checks["this_study"] = {"ok": None, "level": None, "detail": "review - the result is reported from other studies (secondary)"}
    else:
        mark("this_study", not (cited or background or aim or biblio),
             "quote is this study's result" if not (cited or background or aim or biblio)
             else "quote is a bibliography entry (the title of a cited paper), not a result" if biblio
             else ("quote reports prior work (other studies), not this study's result" if cited
                   else f"quote states an aim, question or speculation ('{aim.group(0)}'), not a result" if aim
                   else "quote is a general statement in the Discussion (no 'we' / 'our' / 'this study') - background, "
                        "not this study's result" if discussion_generic
                   else "quote comes from the Introduction (background)"))

    # 2. intervention
    itoks = _key_tokens(claim.get("intervention") or claim.get("subject"))
    title_low = (ctx.get("title") or "").lower()
    from claim_extraction import abbreviations, clean_subject, subject_from_sentence, implausible_subject
    from evidence_identity import is_question_word_subject
    # before asking whether the quote supports the intervention, ask whether the text IS an intervention:
    # a sentence fragment, an instrument, a pathway or the research question can never be one (v24 fix 2)
    _bad_iv = implausible_subject(claim.get("intervention") or claim.get("subject"),
                                  (ctx.get("question") or ((ctx.get("frame") or {}).get("question"))))
    if _bad_iv:
        mark("intervention", False, _bad_iv)
    abbr = ctx.get("abbreviations")
    if abbr is None:
        abbr = abbreviations(ctx.get("source_text") or "")
    named_abbr = [a for a, full in abbr.items() if itoks and any(t in full.lower() for t in itoks)
                  and re.search(rf"\b{re.escape(a)}\b", quote)]
    if _bad_iv:
        pass                                            # already rejected: do not re-mark (v24 fix 2)
    elif claim.get("intervention_as_written") and re.search(rf"\b{re.escape(claim['intervention_as_written'])}\b", quote):
        named_abbr.append(claim["intervention_as_written"])
    other = clean_subject(subject_from_sentence(quote), abbr)
    other_agent = other and not any(t in other.lower() for t in itoks) and not is_question_word_subject(other) \
        and canonical_outcome(other)["family"].startswith("other:") and not _READOUT_IV.search(other)
    implicit = re.search(r"\b(?:treat\w*|administ\w*|exposed|exposure|incubat\w*|dose\w*|concentration\w*|the drug|the compound|"
                         r"this (?:drug|agent|compound|treatment|effect)|these effects|it (?:also )?(?:inhibit|reduc|increas|decreas|promot|suppress|induc)\w*|"
                         r"(?:intervention|treatment|treated|study|active|drug) (?:group|arm)|(?:between|in) (?:the )?(?:two|both) "
                         r"(?:groups|arms|cohorts))\b",
                         quote, re.I)
    src = ctx.get("source_text") or ""
    qpos = src.find(quote[:60]) if quote else -1
    prev_sent = ""
    if qpos > 0:
        before = re.split(r"(?<=[.!?])\s+", src[max(0, qpos - 400):qpos].strip())
        prev_sent = before[-1] if before else ""
    from clinical_context import _DRUG_ABBR
    aliases = [a for a, full in abbr.items() if any(t in full.lower() for t in itoks)] + \
        [a for a, full in _DRUG_ABBR.items() if any(t in full for t in itoks)]
    prev_names = bool(prev_sent) and (any(t in _norm_for_match(prev_sent) for t in itoks)
                                      or any(re.search(rf"\b{re.escape(a)}\b", prev_sent, re.I) for a in aliases))
    iv_text = claim.get("intervention") or claim.get("subject") or ""
    if _bad_iv:
        pass                                            # rejected above; its reason is the one to keep (v24 fix 2)
    elif not itoks:
        mark("intervention", False, "claim does not name a specific intervention/factor")
    elif _GENERIC_INTERVENTION.match(iv_text):
        mark("intervention", False, f"'{iv_text}' is not an agent - the claim does not say what was given or manipulated")
    elif re.search(_MODIFIED_EFFECT + r"(?:the\s+)?" + re.escape(iv_text.split()[0]), quote, re.I):
        mark("intervention", False, f"the quote reports that another factor changed the effect of '{iv_text}' "
                                    f"(e.g. 'abrogated the effect of {iv_text}'), not the effect of '{iv_text}' itself")
    elif _BASELINE_CMP.search(quote) and not any(t in _norm_for_match(re.split(r",\s*(?:suggesting|indicating|which|implying)|;",
                                                                               quote)[0]) for t in itoks):
        mark("intervention", False, "the quote compares two systems at baseline ('lower than that in ...'); "
                                    f"'{iv_text}' is not what differs between them")
    elif re.match(r"\s*(?:a |the )?(?:decrease|increase|reduction|elevation|loss|gain|change|rise|drop|fall)s?\s+"
                  r"(?:in|of)\b", claim.get("intervention") or "", re.I):
        mark("intervention", False, f"'{claim.get('intervention')}' is a change in a measurement, not an intervention - "
                                    f"the claim does not say what was given or manipulated")
    elif any(t in nquote for t in itoks) or named_abbr:
        mark("intervention", True, "named in the quote" + (f" (as {named_abbr[0]})" if named_abbr and not any(t in nquote for t in itoks) else ""))
    elif other_agent:
        mark("intervention", False, f"the quote attributes the result to '{other}', not to '{claim.get('intervention')}'")
    elif other and re.search(r"\s(?:and|plus)\s|\bcombin", other, re.I) and \
            not re.search(r"\s(?:and|plus|\+)\s|\+|combin|regimen", claim.get("intervention") or "", re.I) and \
            len([w for w in re.split(r"\s+(?:and|plus)\s+", other) if w.strip()]) >= 2 and \
            not all(any(t in w.lower() for t in itoks) for w in re.split(r"\s+(?:and|plus)\s+", other)):
        mark("intervention", False, f"the quote attributes the result to the combination '{other}', not to "
                                    f"'{claim.get('intervention')}' alone")
    elif prev_names:
        mark("intervention", True, "named in the sentence immediately before the quote", level="passage")
    elif implicit and any(t in context for t in itoks):
        mark("intervention", True, "named in the quoted passage (the quote refers back to the treatment)", level="passage")
    elif implicit and any(t in title_low for t in itoks):
        mark("intervention", True, "named in the paper title and the quote refers to the treatment", level="title")
    else:
        mark("intervention", False, f"the quote does not mention '{claim.get('intervention')}' or refer to the treatment "
                                    f"(a paper about it can still describe other agents or background biology)")

    # 3. population / model
    cls = ctx.get("classification") or {}
    model = _endpoint_clause_model(quote, outcome_text or statement)
    if model is None:
        model = claim_model(quote, cls.get("evidence_type") or "unclear", cls.get("secondary_designs") or [])
    if model["source"] == "paper classification" and cls.get("evidence_type") not in ("review_meta",):
        from study_classifier import section_model
        sm = section_model(loc.get("section_title") or claim.get("source_section"))
        present = {cls.get("evidence_type")} | set(cls.get("secondary_designs") or [])
        lab = {"animal", "cell_culture", "organoid_ex_vivo", "experimental_disease_model", "unclear"}
        if sm and (sm["model_type"] in present or (cls.get("evidence_type") in lab and sm["model_type"] in lab)):
            model = sm
    if model["source"] == "paper classification":
        # the stated intervention is part of the evidence for which system this is: an agent named together with
        # a live-cell imager or a culture reagent is a cell experiment, however the paper was classified (v24 fix 1)
        _iv_ctx = (claim.get("intervention") or claim.get("subject") or "")
        model = claim_model(f"{quote} {_iv_ctx} {context[:900]}", cls.get("evidence_type") or "unclear",
                            cls.get("secondary_designs") or [])
    if model["source"] == "paper classification" and model["model_type"] in ("animal", "experimental_disease_model") and \
            _CELLULAR_ENDPOINT.search(f"{outcome_text} {quote}") and not _CUE_ANIMAL.search(f"{quote} {context[:400]}") and \
            ("cell_culture" in (cls.get("secondary_designs") or []) or _CUE_CELL.search(quote)):
        model = {"model_type": "cell_culture", "source": "cell-level endpoint with no animal named (paper also has cell experiments)"}
    st_low = re.sub(r"\bhuman[- ](?:blood|plasma|serum|therapeutic|equivalent|relevant|clinical)\b(?:[- ](?:concentrations?|levels?|doses?))?|"
                    r"\b(?:clinically|physiologically) relevant\b", " ", statement.lower())
    says_human = bool(re.search(r"\b(patients?|humans?|women|men|participants?|people|clinical(?:ly)?)\b", st_low))
    says_animal = bool(re.search(r"\b(mice|mouse|rats?|in vivo|animals?)\b", st_low))
    mt = model["model_type"]
    if mt in ("human_clinical", "human_observational") and cls.get("evidence_type") in ("human_clinical", "human_observational") \
            and mt != cls.get("evidence_type"):
        mt = cls["evidence_type"]
        model = {"model_type": mt, "source": "the paper's design (one design per human study)"}
    if says_human and mt in ("cell_culture", "organoid_ex_vivo", "animal", "in_silico") and \
            not re.search(r"\b(patients?|women|men|participants?|human subjects)\b", nquote):
        mark("population_model", False, f"claim is stated for humans but the quote describes {mt.replace('_', ' ')}")
    elif says_animal and mt in ("cell_culture", "in_silico") and not re.search(r"\b(mice|rats?|in vivo|animals?)\b", nquote):
        mark("population_model", False, f"claim is stated in vivo/animals but the quote describes {mt.replace('_', ' ')}")
    else:
        mark("population_model", True, f"{mt.replace('_', ' ')} ({model['source']})")

    # 4. endpoint
    oinfo = canonical_outcome(outcome_text or statement)
    if oinfo["family"] == "tumour_growth" and mt in ("cell_culture", "organoid_ex_vivo") and \
            re.search(r"\b(?:tumou?r|cancer) cells? (?:growth|proliferation)|growth of (?:\w+[- ]){0,3}cells", quote, re.I) and \
            not re.search(r"\btumou?rs? (?:growth|volume|weight|size|burden)|xenograft", quote, re.I):
        oinfo = canonical_outcome("cell proliferation")
        outcome_text = "tumour cell growth (proliferation)"
    oinfo, outcome_text, ep_reject = refine_endpoint(oinfo, outcome_text, quote)
    if ep_reject:
        mark("endpoint", False, ep_reject)
    # v24 fix 1b - 'survival'/'death' attached to a CELL TYPE is never a patient outcome, whatever design the
    # paper was given. A real report turned "increase in melanocyte death" into "patient survival decreased".
    if oinfo["family"] == "patient_survival" and _CELL_DEATH_RX.search(f"{outcome_text} {quote}") and \
            not re.search(r"\b(?:patients?|participants?|subjects?|cohort|mortality|overall survival|"
                          r"progression-free|disease-free|recurrence-free|follow(?:ed)?[- ]up|"
                          r"died|deaths? (?:of|among|in) (?:patients?|participants?|subjects?))\b",
                          quote, re.I):
        _cell_word = _CELL_DEATH_RX.search(f"{outcome_text} {quote}").group(1)
        oinfo = canonical_outcome("cell viability")
        outcome_text = f"{_cell_word} survival (cell viability)"
        if mt in ("human_clinical", "human_observational"):
            mt = "cell_culture"
            model = {"model_type": mt, "source": "the quote measures survival of cells, not of people"}
    if oinfo["family"] in ("patient_survival", "recurrence_incidence", "clinical_response") and \
            mt in ("cell_culture", "organoid_ex_vivo", "in_silico"):
        if oinfo["family"] == "patient_survival" and re.search(r"\bsurviv", quote, re.I):
            oinfo, outcome_text = canonical_outcome("cell viability"), "cell survival (viability)"
        else:
            mark("endpoint", False, f"a {mt.replace('_', ' ')} experiment cannot measure {oinfo['label']} (a patient outcome)")
    elif oinfo["family"] in ("patient_survival", "recurrence_incidence", "clinical_response") and mt == "microbial" and \
            _MICROBE_SUBJECT.search(f"{outcome_text} {quote}"):
        oinfo, outcome_text = canonical_outcome("microbial growth"), "microbial survival (growth / viability)"
    elif oinfo["family"] in ("patient_survival", "recurrence_incidence", "clinical_response") and mt == "plant" and \
            re.search(r"\b(?:plants?|seedlings?|leaf|leaves|shoots?|roots?|seeds?|crops?)\b", f"{outcome_text} {quote}", re.I):
        oinfo, outcome_text = canonical_outcome("plant growth"), "plant survival (growth)"
    if mt in ("cell_culture", "organoid_ex_vivo"):
        from methodology_flags import cell_line_flags
        _frame = ctx.get("frame") or {}
        for _cf in cell_line_flags(f"{quote} {context} {ctx.get('title') or ''}", _frame.get("disease"), mt):
            if _cf["severity"] == "critical":
                mark("cell_line_identity", False, _cf["statement"])
                break
    erx = _endpoint_regex(oinfo, outcome_text)
    if checks.get("endpoint", {}).get("ok") is False:
        pass
    elif not (erx.search(strip_adjusters(nquote)) or any(t in strip_adjusters(nquote) for t in _key_tokens(outcome_text))) and \
            (erx.search(nquote) or any(t in nquote for t in _key_tokens(outcome_text))):
        mark("endpoint", False, f"'{outcome_text or oinfo['label']}' appears in the quote only as an adjustment variable "
                                f"(e.g. 'BMI-adjusted'), not as the outcome that was measured")
    elif erx.search(nquote) or any(t in nquote for t in _key_tokens(outcome_text)):
        mark("endpoint", True, f"{oinfo['label']} measured in the quote")
    elif erx.search(context):
        mark("endpoint", True, f"{oinfo['label']} named in the quoted passage", level="passage")
    else:
        mark("endpoint", False, f"the quote does not mention the endpoint '{outcome_text or oinfo['label']}'")
    if oinfo["family"] in ("expression", "signalling") and checks.get("endpoint", {}).get("ok") is not False and \
            not oinfo.get("target") and not finding_target({"canonical_outcome": oinfo["family"], "measurement": outcome_text, "quote": quote}):
        mark("endpoint", False, f"'{outcome_text or oinfo['label']}' names no gene, protein or pathway - a bare "
                                f"'{oinfo['label']}' is not a measurable endpoint")

    # 5. direction
    claim_dir = direction if direction in ("increase", "decrease", "no_effect") else None
    canon_claim = claim_dir if (claim_dir == "no_effect" or oinfo["polarity"] > 0) else flip(claim_dir)
    qd = direction_in_text(quote, oinfo, outcome_text)
    if qd["direction"] is None:
        qx = direction_in_text(expand_endpoint(quote, ctx.get("abbreviations")), oinfo, outcome_text)
        if qx["direction"]:
            qd = qx
    if qd["direction"] is None and claim_dir in ("increase", "decrease"):
        gr = group_rate_direction(quote, claim.get("intervention") or "")
        if gr:
            qd = {"direction": gr if oinfo["polarity"] > 0 else flip(gr), "endpoint_found": True}
    if qd["direction"] is None:
        rd = ratio_direction(quote, oinfo, outcome_text)
        if rd:
            qd = {"direction": rd if (rd == "no_effect" or oinfo["polarity"] > 0) else flip(rd), "endpoint_found": True}
    if qd["direction"] is None and claim_dir == "no_effect" and NOEFFECT.search(quote):
        toks = _key_tokens(outcome_text)
        blanket = re.search(r"\b(?:any|all|none of the|each)\s+(?:\w+\s+)?(?:biomarkers?|markers?|endpoints?|outcomes?|"
                            r"parameters?|measures?)\b", quote, re.I)
        excepted = re.search(r"\bexcept(?: for)?\b[^.]*", quote, re.I)
        if toks and all(t in nquote for t in toks):
            qd = {"direction": "no_effect", "endpoint_found": True}
        elif blanket and erx.search(context) and not (excepted and erx.search(excepted.group(0))):
            qd = {"direction": "no_effect", "endpoint_found": True}
    if claim_dir is None:
        mark("direction", False, "claim has no determinable direction (unclear/mixed)")
    elif qd["direction"] is None:
        nonsig = any(p["value"] >= 0.05 and p["comparator"] in ("=", ">", ">=") for p in parse_p_values(quote))
        if claim_dir == "no_effect" and nonsig:
            mark("direction", True, "no effect supported by a non-significant p-value in the quote")
        else:
            mark("direction", False, "the quote does not state a direction for this endpoint")
    elif claim_dir in ("increase", "decrease") and INCONSISTENT.search(quote):
        mark("direction", False, "the quote reports inconsistent / conflicting evidence ('not all studies have found ...'), "
                                 "not a direction")
    elif len({x for x in endpoint_directions(quote, oinfo, outcome_text) if x in ("increase", "decrease")}) > 1:
        mark("direction", False, f"the quote states both an increase and a decrease for {oinfo['label']} "
                                 f"(internally inconsistent sentence - cannot be used)")
    elif canon_claim == "no_effect" and qd["direction"] in ("increase", "decrease") and \
            statistical_reading(quote)["significant"] is False:
        mark("direction", True, f"{oinfo['label']}: no statistically significant difference "
                                f"({statistical_reading(quote)['basis']}; numerical trend: {qd['direction']})")
    elif qd["direction"] != canon_claim:
        arrow = {"increase": "UP", "decrease": "DOWN", "no_effect": "no effect"}
        mark("direction", False, f"quote supports {oinfo['label']} {arrow.get(qd['direction'])}, "
                                 f"claim says {oinfo['label']} {arrow.get(canon_claim)}")
    else:
        mark("direction", True, f"{oinfo['label']} {qd['direction'].replace('_', ' ')} (matches quote)")
    sd = direction_in_text(statement, oinfo, outcome_text)
    if checks["direction"]["ok"] and sd["direction"] and canon_claim and sd["direction"] != canon_claim:
        mark("statement_consistency", False, f"generated statement says {oinfo['label']} {sd['direction']} "
                                             f"but the extracted direction is {canon_claim}")

    # 5b. statistical significance
    stat = statistical_reading(quote)
    trend, stat_note = None, None
    within_arm = None
    if checks["direction"]["ok"] and canon_claim in ("increase", "decrease") and _BETWEEN_NULL.search(quote) \
            and _WITHIN_CHANGE.search(quote):
        within_arm = canon_claim
        canon_claim, direction = "no_effect", "no_effect"
        stat_note = (f"the {within_arm} from baseline occurred in both arms; metformin vs comparator: no difference between "
                     f"arms - the treatment effect is null")
        checks["direction"] = {"ok": True, "level": "quote", "detail": f"{oinfo['label']}: {stat_note}"}
        stat = {**stat, "significant": None, "basis": "between-arm comparison: no difference"}
    if within_arm:
        pass
    elif checks["direction"]["ok"] and canon_claim in ("increase", "decrease") and stat["significant"] is False:
        trend, canon_claim = canon_claim, "no_effect"
        stat_note = (f"not statistically significant ({stat['basis']}); numerical trend: {trend} only"
                     + (" - the source calls it 'significant', but the test does not reach 0.05"
                        if stat["author_says_significant"] else ""))
        checks["direction"] = {"ok": True, "level": "quote",
                               "detail": f"{oinfo['label']}: no statistically significant difference - {stat_note}"}
        direction = "no_effect"
    elif checks["direction"]["ok"] and canon_claim == "no_effect" and qd.get("direction") in ("increase", "decrease") \
            and stat["significant"] is False:
        trend = qd["direction"]
        stat_note = f"not statistically significant ({stat['basis']}); numerical trend: {trend} only"
    checks["statistics"] = {"ok": None if stat["significant"] is None else True, "level": None,
                            "detail": stat_note or stat["basis"], "significant": stat["significant"]}

    # 6. magnitude
    mags = _magnitudes(statement)
    if mags:
        missing = [m for m in mags if not number_in_text(m, quote) and not number_in_text(m, context)]
        mark("magnitude", not missing, "magnitudes appear in the quote/passage" if not missing
             else f"magnitude(s) {', '.join(missing)} not found in the quote or its passage")
    else:
        checks["magnitude"] = {"ok": None, "level": None, "detail": "no magnitude claimed"}

    # 7. statistical significance
    p = claim.get("p_value")
    claims_sig = (_claims_significance(statement) or (p is not None and float(p) < 0.05)) and claim_dir != "no_effect" \
        and not trend
    p_dropped = False
    if claims_sig:
        quote_sig = _claims_significance(quote) or any(x["value"] < 0.05 for x in parse_p_values(quote))
        ctx_sig = any(x["value"] < 0.05 for x in parse_p_values(context)) or _claims_significance(context)
        if quote_sig:
            mark("significance", True, "significance stated in the quote")
        elif ctx_sig and not _claims_significance(statement):
            mark("significance", True, "p-value reported in the quoted passage", level="passage")
        else:
            mark("significance", False, "statistical significance is claimed but not supported by the quote")
        if p is not None and not any(abs(x["value"] - float(p)) <= max(0.01 * float(p), 1e-12)
                                     for x in parse_p_values(quote + " " + context)):
            p_dropped = True
    else:
        checks["significance"] = {"ok": None, "level": None, "detail": "no significance claimed"}

    # 8. attribution
    from clinical_context import (attribution, causal_supported, disease_context, inference_type, outcome_level,
                                  follow_up_days, parse_question, INFERENCE_LABEL)
    att = attribution(claim.get("intervention") or claim.get("subject") or "", quote, context[:700], abbr)
    if att["combination"] and not att["isolated"]:
        mark("attribution", False, f"the quote describes a combination regimen ({', '.join(att['components'])}); "
                                   f"it does not isolate '{claim.get('intervention')}' as the cause")
    elif att["combination"]:
        checks["attribution"] = {"ok": True, "level": "quote", "detail": "the whole regimen is the intervention"}

    # 9. outcome inflation
    q_out, s_out = outcome_level(quote), outcome_level(statement)
    if s_out["level"] is not None and (q_out["level"] is None or s_out["level"] > q_out["level"]) and s_out["level"] >= 3:
        c_out = outcome_level(context)
        if c_out["level"] is None or s_out["level"] > c_out["level"]:
            mark("outcome_strength", False, f"claim says '{s_out['label']}' but the source only reports "
                                            f"'{q_out['label'] if q_out['level'] is not None else 'no outcome grade'}'")

    etype = cls.get("evidence_type")
    inf = inference_type(quote, context, mt if mt in ("cell_culture", "organoid_ex_vivo", "animal",
                                                        "experimental_disease_model") else etype, outcome_rx=erx)
    cs = causal_supported(statement, inf)
    q_info = parse_question(ctx.get("question") or "")
    terms = q_info["disease_terms"]
    relevant = (not terms) or any(t in nquote or t in context for t in terms) or bool(erx.search(nquote))
    dctx = disease_context(f"{quote} {context[:900]} {ctx.get('title') or ''}", terms)
    st_low = statement.lower()
    narrower = [c for c in dctx["cause_or_induction"] if c.lower() not in st_low]
    context_limited = bool(narrower) or etype in ("experimental_disease_model", "case_report") or \
        (cls.get("organism_group") not in (None, "unknown", "human") and q_info["disease_terms"] and mt != "human_clinical")
    quantitative = bool(parse_p_values(quote)) or bool(re.search(r"\d+(?:\.\d+)?\s*(?:%|-?fold|mg|µm|um|mm|nm|patients|mice)", quote, re.I))
    abstract_only = (ctx.get("text_basis") in (None, "abstract", "none")) or loc.get("section_type") == "abstract"
    fu = follow_up_days(f"{quote} {context[:600]}")

    # --- v24 fix 3: a damaging exposure credited with reducing its own damage -------------------
    # A real report published "ultraviolet B -> DNA damage: decrease". UV-B causes DNA damage; the quote
    # credited a miR-31-5p mimic. When the named intervention is the INSULT and the direction is protective,
    # the direction has been attached to the wrong agent.
    _iv_now = (claim.get("intervention") or claim.get("subject") or "")
    _THERAPY_RX = re.compile(r"\b(?:therap\w*|phototherap\w*|NB-?UV-?B|narrow-?band|broad-?band|PUVA|excimer|laser|"
                             r"radiotherap\w*|radiation therapy|treatment)\b", re.I)
    # v24.2: UV-B / radiation GIVEN AS A TREATMENT (phototherapy for vitiligo or psoriasis) is not an insult
    _is_stressor = bool(_STRESSOR_RX.search(_iv_now)) and not _THERAPY_RX.search(_iv_now) and \
        not re.search(r"\b(?:phototherap\w*|NB-?UV-?B|PUVA|excimer|radiotherap\w*|radiation therapy)\b", quote, re.I)
    _dmg = bool(_DAMAGE_RX.search(f"{outcome_text} {oinfo['label']}"))
    _protective = (canon_claim == "decrease" and _dmg) or \
                  (canon_claim == "increase" and oinfo["family"] in ("viability", "proliferation") and
                   not _STRESSOR_RX.search(quote[:0] or ""))
    if _is_stressor and _protective and checks.get("direction", {}).get("ok") is not False:
        _rescuer = _RESCUER_RX.search(quote)
        if _rescuer:
            mark("direction", False,
                 f"'{_iv_now}' is the damaging exposure in this experiment; the quote credits "
                 f"'{_rescuer.group(0).strip()}' with the protective effect, not '{_iv_now}'")
        elif not re.search(rf"{re.escape(_iv_now.split()[0])}[^.]{{0,80}}(?:reduc|decreas|lower|attenuat|prevent|protect)",
                           quote, re.I):
            mark("direction", False,
                 f"'{_iv_now}' is a damaging exposure, and the quote does not say that it reduced "
                 f"{outcome_text or oinfo['label']} - the direction belongs to another agent")

    # --- v24 fix 4: a cancer endpoint on a paper that is not about cancer ------------------------
    # "T-96 -> metastasis invasion" came from a vitiligo paper counting CD8+ T cells by whole-mount staining.
    # That is immune-cell infiltration, not metastasis.
    if oinfo["family"] in ("metastasis_invasion", "tumour_growth"):
        _qd = f"{' '.join(terms)} {ctx.get('title') or ''} {quote}".lower()
        if not _CANCER_RX.search(_qd):
            if _IMMUNE_INFILTRATE_RX.search(f"{outcome_text} {quote}"):
                oinfo = canonical_outcome("immune cell infiltration")
                outcome_text = "immune-cell infiltration"
            else:
                mark("endpoint", False,
                     f"'{outcome_text or oinfo['label']}' is a cancer endpoint, and neither the question nor "
                     f"the quote is about cancer")

    # 10. EVIDENCE IDENTITY + RELEVANCE GATE
    from evidence_identity import ROLE_LABEL, build_identity, question_frame, relevance_gate
    frame = ctx.get("frame") or question_frame(ctx.get("question") or "")
    identity = build_identity(claim, {"source_context": context, "source_section": loc.get("section_title") or loc.get("section_type"),
                                      "claim_model": mt, "canonical_outcome": oinfo["family"], "canonical_direction": canon_claim,
                                      "checks": checks}, cls, ctx.get("title") or "", frame, ctx.get("abstract") or "",
                              abbreviations=abbr)
    identity["combination_components"] = att.get("components") if att["combination"] and att.get("isolated") else []
    gate = relevance_gate(identity, frame, statement, quote, cls.get("evidence_type") or "unclear")
    if gate["admitted"]:
        checks["relevance_gate"] = {"ok": True, "level": "quote", "detail": "disease, model, exposure and outcome match the question"}
    else:
        checks["relevance_gate"] = {"ok": False, "level": None, "detail": "; ".join(gate["reasons"])}
        reasons.extend(f"relevance_gate: {r}" for r in gate["reasons"])
    relevant = gate["admitted"]
    identity.pop("_mentions", None)

    within_only = bool(canon_claim in ("increase", "decrease") and _WITHIN_ONLY.search(quote) and not _BETWEEN_TEST.search(quote)
                       and mt in ("human_clinical", "animal", "experimental_disease_model") and not within_arm)
    if within_only and relevant:
        gate = {**gate, "answers_question": False}

    status = "verified" if not reasons else "quarantined"
    if status == "quarantined":
        primary = "rejected"
    elif cs["causal_language"] and not cs["supported"]:
        primary = "causal_unsupported"
    elif inf in ("randomised_controlled", "controlled_experiment"):
        primary = "causal_supported" if cs["causal_language"] else "supported"
    elif inf == "statistical_association":
        primary = "association"
    else:
        primary = "observation"
    tags = (["Abstract-only verified"] if abstract_only else ["Full-text verified"]) + \
        (["Context-limited"] if context_limited else []) + (["Quantitatively supported"] if quantitative else []) + \
        ([] if relevant else ["Off-question (excluded by the relevance gate)"]) + \
        ([ROLE_LABEL[gate["role"]]] if relevant else []) + \
        (["Secondary literature - not independent verification"] if gate["role"] == "secondary_report" else []) + \
        (["Within-arm change only (no comparison with the control arm) - context, not a treatment effect"] if within_only else []) + \
        ([f"Endpoint read from the quote ('{claim.get('measurement') or claim.get('outcome')}'; the extractor wrote "
          f"'{_c40(claim.get('outcome_as_written'))}')"] if claim.get("endpoint_repaired") else [])
    names = {True: "supported", False: "not supported", None: "not applicable"}
    profile = {
        "question_match": {"status": names[relevant], "detail": ("disease, model, exposure and outcome match the question"
                                                                  if relevant else "; ".join(gate["reasons"]))},
        "intervention_match": {"status": names[checks.get("intervention", {}).get("ok")], "detail": checks.get("intervention", {}).get("detail")},
        "population_model_match": {"status": names[checks.get("population_model", {}).get("ok")], "detail": checks.get("population_model", {}).get("detail")},
        "outcome_match": {"status": names[checks.get("endpoint", {}).get("ok")], "detail": checks.get("endpoint", {}).get("detail")},
        "direction_match": {"status": names[checks.get("direction", {}).get("ok")], "detail": checks.get("direction", {}).get("detail")},
        "causal_support": {"status": "supported" if cs["supported"] and inf in ("randomised_controlled", "controlled_experiment")
                           else ("not supported" if cs["causal_language"] else "observation only"),
                           "detail": INFERENCE_LABEL.get(inf, inf)},
        "quantitative_support": {"status": "supported" if quantitative else "not supported",
                                 "detail": "numbers / p-value in the quote" if quantitative else "no numbers in the quote"},
        "context_match": {"status": "partial" if context_limited else "supported",
                          "detail": ("narrower context: " + ", ".join(narrower + ([etype.replace('_', ' ')] if etype in ("experimental_disease_model", "case_report") else [])))
                          if context_limited else "same context as the question"},
    }
    inter = claim.get("intervention") or "the intervention"
    if outcome_text and (len(outcome_text.split()) > 6 or any(t in outcome_text.lower() for t in _key_tokens(inter))) \
            and not oinfo["family"].startswith("other:"):
        outcome_text = oinfo["label"]
    if primary in ("causal_unsupported", "observation"):
        evidence_statement = (f"{inter} was administered and {q_out['label'] if q_out['level'] else (outcome_text or 'the outcome')} "
                              f"was observed{' after ' + str(round(fu / 30.44, 1)) + ' months' if fu else ''} "
                              f"({INFERENCE_LABEL.get(inf, inf)}; causation not established).")
    elif primary == "association":
        polar = {"increase": "higher", "decrease": "lower", "no_effect": "no difference in"}.get(canon_claim or "", "")
        if oinfo["family"] == "patient_survival":
            polar = {"increase": "longer / better", "decrease": "shorter / worse", "no_effect": "no difference in"}.get(canon_claim or "", "")
        kind = {"human_observational": "observational", "human_clinical": "clinical", "animal": "animal",
                "cell_culture": "cell culture"}.get(mt, (mt or "").replace("_", " "))
        evidence_statement = (f"{inter} was associated with {polar + ' ' if polar else ''}{outcome_text or 'the outcome'}"
                              f" ({kind + '; ' if kind else ''}association - causation not established).")
    else:
        evidence_statement = statement
    disease_name = frame.get("disease") or "the disease"
    if relevant and gate["role"] == "mechanistic" and frame.get("intent") == "etiology":
        evidence_statement = (f"In {(mt or 'the model').replace('_', ' ')}, {inter} was followed by "
                              f"{outcome_text or oinfo['label']} {canon_claim or ''} - a mechanistic observation, not evidence "
                              f"that {inter} causes {disease_name}.").replace("  ", " ")
    elif relevant and gate["role"] == "intermediate":
        evidence_statement = (f"{inter} was associated with {outcome_text or oinfo['label']} (an intermediate endpoint); "
                              f"this claim does not show the link from that endpoint to {disease_name}.")
    elif relevant and gate["role"] == "disease_risk" and not (cs["supported"] and inf in ("randomised_controlled", "controlled_experiment")):
        evidence_statement = (f"{inter} was associated with {disease_name} occurrence/risk "
                              f"({INFERENCE_LABEL.get(inf, inf)}; association - causation not established).")
    elif relevant and gate["role"] == "secondary_report":
        evidence_statement = f"A review reports: {statement} (secondary literature - the original experiment was not verified here)."
    if within_arm:
        evidence_statement = (f"{inter}: no difference in {outcome_text or oinfo['label']} between arms (the "
                              f"{'rise' if within_arm == 'increase' else 'fall'} from baseline occurred in both arms - not an "
                              f"effect of {inter}).")
    if trend:
        evidence_statement = (f"{inter}: no statistically significant difference in {outcome_text or oinfo['label']} "
                              f"({stat['basis']}; numerical trend: {'lower' if trend == 'decrease' else 'higher'}"
                              f"{' - the authors describe it as significant' if stat['author_says_significant'] else ''}).")
    return {
        "status": status, "checks": dict(checks), "reasons": reasons, "audit_version": AUDIT_VERSION,
        "claim_status": CLAIM_STATUS[primary], "claim_status_key": primary, "tags": tags,
        "grounding_profile": profile, "inference_type": inf, "evidence_statement": evidence_statement,
        "outcome_level": q_out["key"], "outcome_label": q_out["label"], "follow_up_days": fu,
        "combination_components": att["components"] if att["combination"] else [],
        "disease_context": dctx, "verification_basis": "abstract" if abstract_only else "full text",
        "canonical_outcome": oinfo["family"], "canonical_target": oinfo["target"],
        "canonical_direction": canon_claim, "outcome_polarity": oinfo["polarity"],
        "nonsignificant_trend": trend, "within_arm_change": within_arm, "within_arm_only": within_only, "statistics": {k: stat[k] for k in ("significant", "basis", "p")},
        "claim_model": mt, "model_source": model["source"],
        "source_section": loc.get("section_title") or loc.get("section_type"), "source_section_type": loc.get("section_type"),
        "source_context": context[:1300], "p_value_dropped": p_dropped,
        "identity": identity, "relevance_gate": gate, "evidence_role": gate["role"],
        "evidence_role_label": gate["role_label"], "answers_question": gate["answers_question"],
        "resolved_intervention": (claim.get("intervention") if (claim.get("intervention") or "").strip() != original_iv
                                  and claim.get("intervention") else None),
        "original_intervention": original_iv or None,
        "repaired_outcome": (claim.get("measurement") or claim.get("outcome")) if claim.get("endpoint_repaired") else None,
        "outcome_as_written": claim.get("outcome_as_written"),
        "mechanism_evidence": mechanism_evidence(claim),
    }


_COLUMNS = (("claim_status", "TEXT"), ("inference_type", "TEXT"), ("evidence_statement", "TEXT"),
            ("verification_status", "TEXT"), ("semantic_audit_json", "TEXT"), ("canonical_outcome", "TEXT"),
            ("canonical_direction", "TEXT"), ("claim_model", "TEXT"), ("source_section", "TEXT"),
            ("source_context", "TEXT"), ("finding_key", "TEXT"))


def ensure_columns(db: Any) -> None:
    import sqlite3
    with db._lock:
        for col, typ in _COLUMNS:
            try:
                db._conn.execute(f"ALTER TABLE claims ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass
        db._conn.commit()


def paper_context(db: Any, paper_id: int, question: Optional[str] = None) -> Dict[str, Any]:
    from paper_extraction import gather_parts
    paper = db.get_paper(paper_id) or {"id": paper_id}
    parts = gather_parts(db, paper)
    sections = [{"section_type": s["section_type"], "title": s.get("title"), "text": s["text"]} for s in db.get_sections(paper_id)
                if s["section_type"] != "article_meta"]
    if paper.get("abstract") and not any(s["section_type"] == "abstract" for s in sections):
        sections.insert(0, {"section_type": "abstract", "title": "Abstract", "text": paper["abstract"]})
    source = "\n\n".join(x for x in (parts.get("abstract"), parts.get("results"), parts.get("results_discussion"),
                                     parts.get("other"), parts.get("tables"), parts.get("discussion"),
                                     parts.get("methods"), parts.get("introduction")) if x)
    return {"title": paper.get("title") or "", "abstract": paper.get("abstract") or "", "source_text": source,
            "sections": sections, "classification": get_or_classify(db, paper_id), "question": question,
            "text_basis": parts.get("basis"), "abbreviations": __import__("claim_extraction").abbreviations(source)}


def save_audit(db: Any, claim_id: int, audit: Dict[str, Any]) -> None:
    sets = {"verification_status": audit["status"], "semantic_audit_json": json.dumps(audit, ensure_ascii=False),
            "claim_status": audit.get("claim_status"), "inference_type": audit.get("inference_type"),
            "evidence_statement": audit.get("evidence_statement"),
            "canonical_outcome": audit["canonical_outcome"], "canonical_direction": audit["canonical_direction"],
            "claim_model": audit["claim_model"], "source_section": audit["source_section"],
            "source_context": audit["source_context"]}
    if audit.get("p_value_dropped"):
        sets["p_value"] = None
    if (audit.get("nonsignificant_trend") or audit.get("within_arm_change")) and audit["status"] == "verified":
        sets["statement"], sets["effect_direction"] = audit.get("evidence_statement"), "no_effect"
    if audit.get("repaired_outcome"):
        sets["measurement"] = audit["repaired_outcome"]
    new_iv, old_iv = audit.get("resolved_intervention"), audit.get("original_intervention")
    if new_iv and old_iv and new_iv != old_iv:
        sets["intervention"] = new_iv
        row = db._conn.execute("SELECT statement FROM claims WHERE id=?", (claim_id,)).fetchone()
        st = (row[0] if row else "") or ""
        nxt = st[len(old_iv):len(old_iv) + 1]
        if st.lower().startswith(old_iv.lower()) and not (nxt.isalnum() or (nxt and nxt in "-_")):
            sets["statement"] = new_iv + st[len(old_iv):]
    with db._lock:
        db._conn.execute(f"UPDATE claims SET {', '.join(k + '=?' for k in sets)} WHERE id=?", (*sets.values(), claim_id))
        db._conn.commit()


def ensure_audited(db: Any, research_question_id: Optional[int], force: bool = False) -> Dict[str, int]:
    ensure_columns(db)
    where = "research_question_id IS ?" if research_question_id is None else "research_question_id = ?"
    sql = f"SELECT * FROM claims WHERE {where}"
    rows = [dict(r) for r in db._conn.execute(sql, (research_question_id,)).fetchall()]
    if not force:
        rows = [r for r in rows if r.get("verification_status") is None
                or f'"audit_version": "{AUDIT_VERSION}"' not in (r.get("semantic_audit_json") or "")]
    ctx_cache: Dict[int, Dict[str, Any]] = {}
    stats = {"audited": 0, "verified": 0, "quarantined": 0}
    question = None
    if research_question_id is not None:
        rq = db._conn.execute("SELECT question FROM research_questions WHERE id=?", (research_question_id,)).fetchone()
        question = rq[0] if rq else None
    from evidence_identity import question_frame
    frame = question_frame(question or "")
    for c in rows:
        pid = c.get("paper_id")
        if pid is None:
            audit = {"status": "quarantined", "checks": {}, "reasons": ["claim has no source paper (AI inference)"],
                     "canonical_outcome": None, "canonical_direction": None, "claim_model": None,
                     "source_section": None, "source_context": "", "audit_version": AUDIT_VERSION,
                     "claim_status": CLAIM_STATUS["rejected"], "claim_status_key": "rejected", "tags": ["AI inference"]}
        else:
            if pid not in ctx_cache:
                ctx_cache[pid] = paper_context(db, pid, question)
                ctx_cache[pid]["frame"] = frame
            audit = audit_claim(c, ctx_cache[pid])
        save_audit(db, c["id"], audit)
        stats["audited"] += 1
        stats[audit["status"]] += 1
    return stats


def _model_detail(c: Dict[str, Any]) -> str:
    text = f"{c.get('quote') or ''} {c.get('source_context') or ''}"[:900]
    m = re.findall(r"\b(MCF-?7|MDA-MB-\d+|T-?47D|HeLa|HEK-?293\w*|A549|HepG2|HCT-?116|SK-BR-3|BT-?474|4T1|LNCaP|PC-?3|"
                   r"SH-SY5Y|U-?87|Caco-?2|THP-1|Jurkat)\b", text, re.I)
    if m:
        return ",".join(sorted({x.upper().replace("-", "") for x in m}))
    org = (c.get("organism") or "").strip()
    if org.lower() in ("homo sapiens", "human", "humans"):
        return "humans" if c.get("claim_model") in ("human_clinical", "human_observational") else "human-derived"
    if c.get("claim_model") in ("cell_culture", "organoid_ex_vivo"):
        cells = re.findall(r"\b(?:EMT6|E0771|CT26|MC38|B16(?:-?F10)?|PBCCs?|[A-Z]{2,}[\w\-]*\d[\w\-]*)\b(?=[^.;]{0,50}\bcells?\b)", org)
        if re.search(r"\b(?:mice|mouse|rats?|animals?)\b", org, re.I):
            return ",".join(dict.fromkeys(x.upper() for x in cells)) or "cells"
    if c.get("claim_model") == "animal" and re.fullmatch(r"(?:human\s+)?(?:primary\s+)?[\w\s\-()]*cells?(?:\s*\([^)]*\))?", org, re.I):
        return ""
    from study_classifier import _species_from_text
    return _species_from_text(org, text) or org


_SUBGROUP = re.compile(
    r"\b(?:in|among|for)\s+(?:the\s+)?((?:\w+[- ]){0,2}?(?:women|men|patients|participants|subjects|individuals|those|people|"
    r"tumou?rs|cases|users|non-?users)\s+(?:with|who|having|whose)\s+(?:[^.;()]|\.(?=\d)){2,60}?)(?=\s*(?:\(|,|;|\.(?!\d)|$|\bbut\b|\bwhereas\b|"
    r"\bwhile\b|\bcompared\b|\band\b\s+(?:not|no)\b))|"
    r"\bin\s+(?:the\s+)?((?:[\w+\-]+\s+){0,2}?(?:subgroup|subset|stratum|strata))\b|"
    r"\bin\s+((?:ER|PR|HER2|HR|triple)[\w\-]*[- ](?:positive|negative|\+|-)\s+(?:\w+\s+)?(?:tumou?rs|patients|disease|cancers?))|"
    r"\bin\s+the\s+(overall|whole|entire|total)\s+(?:population|cohort|sample|group|trial)", re.I)


def subgroup_of(quote: Optional[str]) -> Optional[str]:
    m = _SUBGROUP.search(quote or "")
    if not m:
        return None
    if m.group(4):
        return "overall"
    g = next(x for x in m.groups()[:3] if x)
    return re.sub(r"\s+", " ", g).strip(" ,")


_SG_ITEM = re.compile(r"(?:,\s*(?:and\s+)?|\s+and\s+)(?=(?:those|women|men|patients|participants|subjects|in)\b)", re.I)


def separate_subgroups(quote: Optional[str]) -> List[str]:
    q = quote or ""
    m = re.search(r"\b(?:in|among|for)\s+(?:the\s+)?(?:\w+[- ]){0,2}?(?:women|men|patients|participants|subjects|individuals|"
                  r"those|people|cases)\s+(?:with|who|in|having|whose)\b", q, re.I)
    if not m:
        return []
    tail = re.split(r"[.;](?!\d)|\s+\((?![^)]*[<>=])|\bbut\b|\bwhereas\b|\bcompared\b", q[m.start():], maxsplit=1)[0]
    tail = re.sub(r"^(?:in|among|for)\s+(?:the\s+)?", "", tail.strip(), flags=re.I)
    items = [re.sub(r"^(?:in|among|for)\s+", "", x.strip(" ,"), flags=re.I) for x in _SG_ITEM.split(tail)]
    items = [re.sub(r"\s+", " ", x) for x in items if len(x) >= 6 and re.search(r"\b(?:with|who|in|having|whose)\b", x, re.I)]
    return items if len(items) >= 2 else []


def finding_target(c: Dict[str, Any]) -> Optional[str]:
    if (c.get("canonical_outcome") or "") not in ("expression", "signalling", "protein_localization"):
        return None
    _NOT_TARGET = {"PBCCS", "PBCC", "MCF7", "MCF-7", "MFI", "RNA", "DNA", "MRNA", "QPCR", "EMT", "MET", "BC", "TNBC", "MDA", "SKBR3",
                   "T47D", "HR", "ER", "PR"}
    gene = r"((?:hsa-)?miR-\d+[a-z]?(?:-[35]p)?|cyclin [A-E]\d?|p\d{2}|(?:p-)?[a-z]?[A-Z][\w]{1,7}(?:-[\w]{1,4})?)"
    read = r"(?:expression|mRNA|protein|levels?|phosphorylation|activity|activation|abundance|transcription|signall?ing|pathway)"
    for src in (c.get("measurement") or c.get("outcome") or "", c.get("quote") or ""):
        for m in list(re.finditer(rf"\b{gene}\s+(?:[\w\-]+\s+){{0,2}}?{read}\b", src)) + \
                list(re.finditer(rf"\b{read}\s+(?:levels?\s+)?of\s+(?:the\s+)?{gene}", src)) + \
                list(re.finditer(rf"\b(?:genes?|proteins?|markers?)\s+{gene}", src)):
            tok = m.group(1)
            if tok.upper().replace("-", "") not in {x.replace("-", "") for x in _NOT_TARGET} and not tok.isdigit() and \
                    not re.fullmatch(r"(?:The|This|These|Metformin|Protein|Gene|Cell|Cells|mRNA|Total|Relative)", tok):
                return tok
        if src and src == (c.get("measurement") or c.get("outcome") or ""):
            t = canonical_outcome(src).get("target")
            if t and not re.search(rf"cyclin\s+{re.escape(t)}", src, re.I):
                return t
    return None


def build_findings(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    experiments: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for c in claims:
        comps = (c.get("semantic_audit") or {}).get("combination_components") or []
        inter = ("regimen:" + "+".join(sorted(x.lower() for x in comps))) if comps else \
            (" ".join(sorted(_key_tokens(c.get("intervention")))[:2]) or "unspecified")
        ekey = f"{c.get('claim_model') or '?'}|{_model_detail(c)}|{inter}"
        exp = experiments.setdefault(ekey, {"model_type": c.get("claim_model"), "model_detail": _model_detail(c),
                                            "intervention": (" + ".join(comps) + " (combination)") if comps else c.get("intervention"),
                                            "results": OrderedDict()})
        sg = subgroup_of(c.get("quote"))
        tgt = finding_target(c)
        rkey = f"{c.get('canonical_outcome')}|{tgt or ''}|{c.get('canonical_direction')}|{sg or ''}"
        res = exp["results"].setdefault(rkey, {"endpoint": c.get("canonical_outcome"), "measurement": c.get("measurement"),
                                               "target": tgt, "direction": c.get("canonical_direction"), "subgroup": sg,
                                               "claims": []})
        res["claims"].append(c)
    out = []
    for ei, exp in enumerate(experiments.values(), 1):
        for fi, res in enumerate(exp["results"].values(), 1):
            ps = [x.get("p_value") for x in res["claims"] if x.get("p_value") is not None]
            out.append({"experiment": f"E{ei}", "finding": f"E{ei}.F{fi}", "model_type": exp["model_type"],
                        "model_detail": exp["model_detail"], "intervention": exp["intervention"],
                        "endpoint": res["endpoint"], "measurement": res["measurement"], "direction": res["direction"],
                        "target": res.get("target"),
                        "subgroup": res.get("subgroup"),
                        "subgroups_separate": separate_subgroups(res["claims"][0].get("quote")),
                        "trend": next(((x.get("semantic_audit") or {}).get("nonsignificant_trend") for x in res["claims"]
                                       if (x.get("semantic_audit") or {}).get("nonsignificant_trend")), None),
                        "p_value": min(ps) if ps else None, "claim_ids": [x["id"] for x in res["claims"]],
                        "n_claims": len(res["claims"]),
                        "statement": res["claims"][0].get("evidence_statement") or res["claims"][0].get("statement"),
                        "claim_status": res["claims"][0].get("claim_status"),
                        "inference_type": res["claims"][0].get("inference_type"),
                        "outcome_level": (res["claims"][0].get("semantic_audit") or {}).get("outcome_label"),
                        "follow_up_days": (res["claims"][0].get("semantic_audit") or {}).get("follow_up_days"),
                        "tags": sorted({t for x in res["claims"] for t in ((x.get("semantic_audit") or {}).get("tags") or [])}),
                        "quote": res["claims"][0].get("quote"), "source_section": res["claims"][0].get("source_section"),
                        "evidence_role": (res["claims"][0].get("semantic_audit") or {}).get("evidence_role") or "direct",
                        "answers_question": (res["claims"][0].get("semantic_audit") or {}).get("answers_question", True),
                        "subtypes": sorted({s for x in res["claims"] for s in (((x.get("semantic_audit") or {}).get("identity") or {})
                                                                                .get("disease_subtype") or []) if s != "not specified"})})
    return out