"""
study_classifier.py
-------------------
Immutable study classification + strict evidence hierarchy.

1. Every paper gets ONE evidence type, decided once by an ordered, documented
   rule list and stored in the database (table `study_classifications`).
   Later steps read the stored value - nothing downstream may re-score or move a
   paper between categories on keyword counts.

       human_clinical       Human clinical (trials, interventional studies in people)
       human_observational  Human observational (cohort, case-control, registry, patient samples)
       animal               Animal in vivo (lab, farm or wild animals; PDX models are ANIMAL)
       organoid_ex_vivo     Organoid / ex vivo tissue (no live animal)
       cell_culture         Cell culture / in vitro
       in_silico            In silico / computational
       review_meta          Review / meta-analysis (secondary literature)
       case_report          Case report / case series (human or veterinary; uncontrolled)
       experimental_disease_model  Deliberately induced disease model (e.g. laser-, chemically-,
                            DSS-, STZ-induced) - kept separate from naturally occurring disease
       plant, microbial     whole-organism plant or microbial experiments (all-biology extension)
       unclear              design could not be determined (never guessed)

2. Evidence levels (relative to the organism the question is about):

       L1  Direct evidence in the target organism   (human: clinical)
       L2  Indirect evidence in the target organism (human: observational)
       L3  Animal / other whole organism
       L4  Organoid / ex vivo
       L5  Cell culture
       L6  Computational / mechanistic
       L7  AI hypothesis / speculation   (never a study - used for AI-generated content)
       S   Secondary literature (review / meta-analysis): summarises other studies,
           so it is not counted as an independent finding.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from core_utils import now_iso

CLASSIFIER_VERSION = "2.2"   # 1.1: occupational/exposed-worker cohorts, co-cultures, leukemia lines, review-style titles
                             # 1.2: sentences about PRIOR or FUTURE work no longer decide the design ("previously shown in mice")
                             # 1.8: no bare 'population' / 'donors' / 'individuals' as people; the extraction's
                             #      organism (a plant / microbe) overrides a weak regex 'human' call
                             # 1.9: reagents named after animals (fetal bovine serum, goat anti-rabbit) are not animal
                             #      experiments; an animal word that only names the cells' origin is cell culture
                             # 2.0: editorials / commentaries / perspectives (question titles, 'an evolving mystery',
                             #      'is it time to ...') with no primary results are secondary literature
                             # 1.3: publisher article-type (review-article), full text without Methods/Results, review-only abstracts

EVIDENCE_TYPES: Dict[str, str] = {
    "human_clinical": "Human clinical",
    "human_observational": "Human observational",
    "animal": "Animal (in vivo)",
    "organoid_ex_vivo": "Organoid / ex vivo",
    "cell_culture": "Cell culture",
    "in_silico": "In silico",
    "review_meta": "Review / meta-analysis",
    "case_report": "Case report / series",
    "experimental_disease_model": "Experimental disease model",
    "plant": "Plant (whole organism)",
    "microbial": "Microbial",
    "unclear": "Design unclear",
}

# legacy display buckets used by the website / matrix (derived, never stored)
TYPE_TO_BUCKET = {
    "human_clinical": "human", "human_observational": "human", "animal": "animal",
    "organoid_ex_vivo": "in_vitro", "cell_culture": "in_vitro", "plant": "plant", "microbial": "microbial",
    "in_silico": "other", "review_meta": "other", "unclear": "other",
    "case_report": "other", "experimental_disease_model": "other",     # refined by organism group (bucket_for)
}


def bucket_for(etype: str, organism_group: Optional[str]) -> str:
    """Display bucket. Case reports and induced-disease models follow the organism actually studied."""
    if etype in ("case_report", "experimental_disease_model"):
        return {"human": "human", "animal": "animal", "plant": "plant", "microbial": "microbial"}.get(organism_group or "", "other")
    return TYPE_TO_BUCKET.get(etype, "other")

LEVEL_NAMES = {
    1: "L1 Direct evidence in the target organism",
    2: "L2 Indirect evidence in the target organism",
    3: "L3 Animal / other whole organism",
    4: "L4 Organoid / ex vivo",
    5: "L5 Cell culture",
    6: "L6 Computational / mechanistic",
    7: "L7 AI hypothesis / speculation",
}
HUMAN_LEVEL_NAMES = {
    1: "L1 Direct human evidence (clinical)",
    2: "L2 Human indirect evidence (observational)",
    3: "L3 Animal evidence",
    4: "L4 Organoid / ex vivo",
    5: "L5 Cell culture",
    6: "L6 Computational / mechanistic",
    7: "L7 AI hypothesis / speculation",
}

# --------------------------------------------------------------------------- cue lexicons
_R = lambda p: re.compile(p, re.I)  # noqa: E731

REVIEW_TITLE = _R(r"\b(systematic review|meta-?analys[ie]s|umbrella review|scoping review|narrative review|"
                  r"literature review|mini-?review|a review|review of|reviews? on|overview of|state of the art|"
                  r"current (?:evidence|perspectives?|understanding)|recent advances|perspectives? on)\b|:\s*a review\b|\breview\s*$")
REVIEW_ABSTRACT = _R(r"\b(we (?:systematically )?(?:searched|reviewed?)|in this review|this review|we summari[sz]e|"
                     r"studies were included|pooled (?:analysis|estimate|hazard|odds|relative)|random-effects|"
                     r"fixed-effects? model|PRISMA guidelines|(?:PubMed|MEDLINE|Embase|Cochrane)[^.]{0,40}(?:searched|databases))\b")
# words that name PEOPLE. Generic biology words are left out on purpose: 'population' (plant / wild populations),
# bare 'donors' (conjugation donor strains), 'cases', 'individuals' (ecology) - they only count with a human qualifier
HUMAN_PEOPLE = _R(r"\b(patients?|participants?|volunteers?|women|men|children|adolescents|adults|"
                  r"people|persons|cases and controls|healthy controls|pregnant|elderly|postmenopausal|"
                  r"premenopausal|outpatients?|inpatients?|users and non-?users|woman|boy|girl|child|infant|neonates?|"
                  r"workers?|employees|residents|miners|smokers|survivors|twins|newborns|childhood|"
                  r"(?:human|study|general|adult|patient|paediatric|pediatric|elderly|urban|rural) population|"
                  r"(?:blood|healthy|human|organ|kidney|liver|living|deceased|bone marrow|stem cell) donors?|"
                  r"healthy (?:individuals|subjects)|human subjects|patient cohorts?|(?:individuals|subjects) (?:with|who|aged)|"
                  r"\d+ (?:incident )?cases|cases? and (?:\d+ )?(?:matched )?controls|"
                  r"\d+[- ]year[- ]old)\b"
                  # 'pregnant mice', 'elderly rats', 'neonatal piglets', 'worker bees': the word describes an ANIMAL
                  r"(?![- ](?:\w+[- ]){0,1}(?:mice|mouse|rats?|rodents?|animals?|pigs?|piglets?|dogs?|cats?|cows?|cattle|"
                  r"sheep|ewes?|lambs?|goats?|calves|monkeys?|macaques?|primates?|zebrafish|fish|bees?|honeybees?|ants?|"
                  r"larvae|flies|birds?|chickens?|hens?))")
TRIAL = _R(r"\b(randomi[sz]ed(?: controlled)? trial|clinical trial|phase\s?(?:i{1,3}|[1-3])\b|placebo|"
           r"window[- ]of[- ]opportunity|pre-?surgical|pre-?operative (?:trial|study|window)|neoadjuvant (?:trial|study)|"
           r"single-arm|open-label|double-blind|crossover (?:trial|study)|interventional study|randomi[sz]ed to|"
           r"were randomi[sz]ed|randomly assigned)\b")
OBSERVATIONAL = _R(r"\b(cohort|case[- ]control|cross[- ]sectional|registry|retrospective\w*|prospective(?: study| analysis| cohort)?|"
                   r"population[- ]based|epidemiolog\w*|mendelian randomi[sz]ation|nested|claims (?:data|database)|"
                   r"electronic (?:health|medical) records?|national (?:database|register)|surveys?|TCGA|"
                   r"the cancer genome atlas|tissue microarrays?|patient (?:samples|specimens|tissues|tumou?rs|biopsies)|"
                   r"biops(?:y|ies) from|clinical (?:samples|specimens|data)|occupational(?:ly)?(?: exposed| exposure)?|"
                   r"exposed (?:workers|individuals|subjects|population|people|children|residents)|workers exposed|"
                   r"biomonitoring|incidence|hazard ratios?|odds ratios?|relative risks?|standardi[sz]ed incidence|"
                   r"exposure assessment|atomic[- ]bomb survivors)\b")
ANIMAL = _R(r"\b(mice|mouse|murine|rats?|rodents?|xenografts?|zebrafish|drosophila|c\. ?elegans|caenorhabditis|"
            r"pigs?|piglets?|porcine|dogs?|canine|primates?|macaques?|rabbits?|sheep|ovine|cattle|bovine|cows?|"
            r"hamsters?|guinea pigs?|chickens?|poultry|birds?|avian|fish(?:es)?|frogs?|xenopus|insects?|bees?|"
            r"honeybees?|mosquito(?:es)?|transgenic (?:mice|animals|rats)|knockout mice|orthotopic|syngeneic|"
            r"patient-derived xenografts?|PDX|wild populations?|wildlife|livestock|animal models?|"
            # farm, companion and veterinary species (a buffalo case is ANIMAL evidence, never 'in silico/other')
            r"buffalo(?:es)?|bubalus|goats?|caprine|horses?|equine|mares?|ponies|donkeys?|camels?|cats?|feline|"
            r"calf|calves|heifers?|bulls?|bullocks?|ewes?|lambs?|yaks?|deer|monkeys?|veterinary|dairy (?:cattle|cows|herd)|"
            r"broilers?|ducks?|turkeys?|rodent models?|apis mellifera|bumblebees?|ants?|sticklebacks?|salmon|trout|"
            r"medaka|killifish|tilapia|carps?|cichlids?|lizards?|snakes?|turtles?|amphibians?|tadpoles?|salamanders?|axolotls?|"
            # Latin binomials of the standard model animals (a 'Danio rerio' paper never says 'zebrafish')
            r"danio rerio|mus musculus|rattus norvegicus|drosophila melanogaster|xenopus (?:laevis|tropicalis)|"
            r"oryzias latipes|gallus gallus|macaca (?:mulatta|fascicularis)|sus scrofa|bos taurus|canis (?:lupus )?familiaris|"
            r"felis catus|oryctolagus cuniculus|ovis aries|capra hircus|equus caballus|cavia porcellus|"
            r"mesocricetus auratus|apis mellifera|daphnia\w*|galleria mellonella|ambystoma mexicanum)\b")
CASE_REPORT = _R(r"\b(case reports?|case series|a case of|report of (?:a|an|one|two|three) (?:cases?|patients?|animals?)|"
                 r"we (?:report|describe|present) (?:a|an|the|two|three) (?:case|patient|\w+ (?:with|case))|"
                 r"successful(?:ly)? treat\w* (?:of )?(?:a|an) |single case|n-of-1|"
                 r"in (?:a|an) (?:\d+[- ](?:year|month)[- ]old )?(?:\w+ )?(?:buffalo|cow|bull|heifer|calf|dog|cat|horse|mare|"
                 r"goat|sheep|ewe|camel|pony|donkey|patient|woman|man|boy|girl|child|infant|neonate))\b")
EXP_MODEL = _R(r"\b(experimental (?:\w+ ){0,2}model|experimentally induced|(?:\w+-)?induced (?:\w+ ){0,2}model|"
               r"model of (?:\w+[- ])?induced|(?:laser|chemically|chemical|drug|dss|stz|streptozotocin|bleomycin|"
               r"monobenzone|hydroquinone|cfa|lps|uv|uvb|carrageenan|cisplatin|ovalbumin)[- ]induced)\b")
IN_VIVO = _R(r"\bin vivo\b")
ORGANOID = _R(r"\b(organoids?|ex vivo|explants?|tissue slices?|precision-cut|organotypic|organ cultures?|"
              r"patient-derived (?:tumou?r )?(?:fragments|explants|slices))\b")
CELL = _R(r"\b(in vitro|cell lines?|cultured|cell cultures?|primary (?:cells|cultures)|cells were (?:treated|incubated|"
          r"transfected|exposed|cultured)|transfect\w*|spheroids?|MCF-?7|MDA-MB-\d+|T-?47D|HeLa|HEK-?293\w*|A549|"
          r"HepG2|Caco-?2|SH-SY5Y|U-?87|iPSC\w*|THP-1|Jurkat|HCT-?116|PC-?3|LNCaP|SK-BR-3|BT-?474|3T3|RAW ?264\.7|"
          r"fibroblasts|keratinocytes|hepatocytes|cardiomyocytes|co-?cultur\w*|mesenchymal (?:stromal|stem) cells|MSCs?|"
          r"(?:bone marrow )?stromal cells|leuk(?:a)?emi[ac] cells|leuk(?:a)?emic blasts|PBMCs?|peripheral blood mononuclear cells|"
          r"K562|HL-?60|MOLM-?13|MV4-?11|NB4|U-?937|KG-?1|Kasumi-?1|OCI-AML\d|NALM-?6|RS4;11|MEC-?1|CCRF-CEM|Ba/F3|"
          r"incubated with|in culture)\b")
# exposure that happened IN PEOPLE (biomonitoring of exposed workers) - human observational even if their
# lymphocytes were later cultured for a chromosome assay
HUMAN_EXPOSED = _R(r"\b(?:(?:occupationally |environmentally )?exposed (?:workers|individuals|subjects|population|people|children|"
                   r"residents)|workers (?:exposed|occupationally)|occupational(?:ly)? exposure|exposure of workers)\b")
# material taken from people (patient samples) - human observational when no culture/in vitro step is described
HUMAN_SAMPLE = _R(r"\b(?:lymphocytes|blood|plasma|serum|bone marrow|samples?|specimens?|biops\w+|tissues?|urine)\s+"
                  r"(?:\w+\s+){0,3}(?:of|from|in)\s+(?:\w+\s+){0,2}(?:workers|patients|subjects|individuals|participants|"
                  r"donors|children|adults|cases|people|smokers|residents|survivors)\b")
# the treatment was given to / measured in live animals ('treatment of mice with MET', 'metformin-treated mice', 'in mice')
IN_ANIMAL = _R(r"\b(?:(?:treatment|administration|injection|gavage) of (?:the )?(?:mice|rats|animals)|(?:mice|rats|animals) "
               r"(?:were |was )?(?:treated|given|received|injected|fed|bearing)|(?:\w+-)?treated (?:mice|rats|animals)|"
               r"in (?:the )?(?:[\w/\-]+ ){0,2}(?:mice|rats|animals)|tumou?r[- ]bearing|tumou?r volume|xenografts?|of (?:the )?mice)\b")
# a group of PEOPLE that received a treatment / was followed ('Children with ALL (n = 498) were treated with ...',
# 'patients in deep molecular response stopped imatinib') - an uncontrolled treated cohort unless a trial is described
TREATED_PEOPLE = _R(r"\b(?:patients|children|adults|participants|women|men|survivors|recipients)\b[^.;]{0,90}?\b(?:were (?:treated|followed|"
                    r"managed|enrolled)|received|underwent|stopped|discontinued|were given)\b")
ANIMAL_TREATED = _R(r"\b(?:(?:mice|rats|animals) (?:were |was )?(?:treated|given|injected|fed|randomi[sz]ed|inoculated)|"
                    r"(?:\w+-)?treated (?:mice|rats|animals)|tumou?r[- ]bearing|(?:gavage|injection) of (?:the )?(?:mice|rats))\b")
HUMAN_RESULT = _R(r"\b(?:placebo|randomi[sz]ed|randomly assigned)\b[^.]{0,200}?\b(?:women|men|patients|participants|subjects|"
                  r"volunteers|children|adults)\b|\b(?:women|men|patients|participants|subjects|volunteers)\b[^.]{0,200}?"
                  r"\b(?:placebo|randomi[sz]ed|randomly assigned)\b")
CULTURE_STEP = _R(r"\b(?:in vitro|cell lines?|co-?cultur\w*|incubated|treated in culture|transfect\w*)\b")
# instruments and assays that exist only with cultured cells: a live-cell imager, an impedance reader, a
# transwell/scratch plate, a colorimetric viability kit. A paper on patients can still report one of these,
# and the result is then a CELL result, whatever the paper's own design is (v24 fix 1).
IN_VITRO_ASSAY = _R(r"\b(?:incucyte|xcelligence|seahorse|operetta|opera phenix|cytation|clariostar|"
                    r"transwell|boyden chamber|scratch assay|wound[- ]healing assay|clonogenic|colony[- ]formation|"
                    r"soft agar|spheroid|organoid culture|mtt assay|\bmtt\b|cck-?8|alamarblue|resazurin|"
                    r"crystal violet|edu incorporation|brdu incorporation|annexin v(?:/| |-)?(?:pi|propidium)?|"
                    r"luciferase reporter|dual-?luciferase|sirna|shrna|plasmid transfection|lipofectamine|"
                    r"culture (?:medium|media|supernatant)|growth medium|fbs|fetal (?:bovine|calf) serum|"
                    r"passage \d+|seeded (?:at|in|into)|serum[- ]starv\w+)\b")
# review-like titles ("Role of X in Y", "Insights into ...") whose abstract reports no primary methods/results
REVIEW_LIKE_TITLE = _R(r"^(?:the )?(?:emerging |multifaceted |dual |critical |potential |novel )?roles? of\b|\binsights? into\b|"
                       r"\bimplications? (?:of|for)\b|\bupdate on\b|\badvances in\b|\bhallmarks? of\b|\bfrom bench to bedside\b|"
                       r"\bnew horizons?\b|\bwhere (?:are|do) we\b|\bconcepts? (?:of|in)\b|\bthe (?:link|interplay|crosstalk) between\b")
COMMENTARY_TITLE = _R(r"^(?:is it|is there|should (?:we|it)|can we|could|are we|what (?:is|are|do|does)|why|how (?:can|should|far)|"
                      r"time to)\b|\bis it time\b|\bevolving (?:mystery|story|role|paradigm)\b|\bmystery\b|\b(?:editorial|"
                      r"commentary|perspectives?|viewpoint|opinion|point of view|counterpoint|a word of caution|lessons? (?:from|learned)|"
                      r"the case for|the case against|myth\w*|hope or hype|promise and peril)\b|\?\s*$")
REVIEW_LIKE_ABSTRACT = _R(r"\b(?:we discuss|is discussed|are discussed|(?:is|are) (?:summari[sz]ed|outlined|highlighted|reviewed|"
                          r"considered|explored)|we highlight|we summari[sz]e|here,? we review|this article|"
                          r"we provide an overview|recent (?:studies|findings|evidence|advances|research)|accumulating evidence|"
                          r"growing body of evidence|has emerged|have emerged|is reviewed|are reviewed)\b")
PRIMARY_ABSTRACT = _R(r"\b(?:we (?:performed|measured|found|treated|enrolled|recruited|analy[sz]ed|investigated|observed|assessed|"
                      r"examined|generated|used|conducted|show|demonstrate)|were (?:treated|randomi[sz]ed|enrolled|recruited|exposed|"
                      r"analy[sz]ed|incubated|cultured|followed)|n\s*=\s*\d+|p\s*[<=]\s*0?\.\d+|\d+ (?:patients|mice|rats|participants|"
                      r"workers|subjects|cases|controls))\b")
IN_SILICO = _R(r"\b(in silico|molecular docking|docking (?:study|analysis|simulation)|molecular dynamics|"
               r"network pharmacology|bioinformatic\w*|computational (?:model\w*|analysis|study|approach)|"
               r"simulations?|virtual screening|QSAR|machine learning model|mathematical model\w*)\b")
PLANT = _R(r"\b(plants?|arabidopsis|maize|zea mays|rice|oryza|wheat|barley|soybeans?|tomato(?:es)?|tobacco|"
           r"nicotiana|seedlings?|crops?|cultivars?|algae|chlamydomonas|moss|physcomitrella|potato(?:es)?|"
           r"grapevine|poplar)\b")
MICROBE = _R(r"\b(bacteri(?:a|um|al)|escherichia|e\. ?coli|bacillus|pseudomonas|staphylococc\w+|streptococc\w+|"
             r"mycobacteri\w+|salmonella|lactobacill\w+|archaea\w*|yeasts?|saccharomyces|candida|fung(?:i|us|al)|"
             r"aspergillus|viruse?s?|bacteriophages?|phages?|cyanobacteri\w+|biofilms?|microb(?:es|ial|iome|iota))\b")

LLM_TYPE_MAP = {
    "clinical_trial": "human_clinical", "randomized_controlled_experiment": None, "cohort": "human_observational",
    "case_control": "human_observational", "cross_sectional": "human_observational",
    "in_vivo_animal_study": "animal", "in_vitro_cell_assay": "cell_culture", "computational": "in_silico",
    "meta_analysis": "review_meta", "systematic_review": "review_meta",
}

# design rank when several primary designs are present (higher = stronger design)
DESIGN_RANK = ["in_silico", "cell_culture", "organoid_ex_vivo", "microbial", "plant", "animal",
               "human_observational", "human_clinical"]


# sentences describing OTHER work (background / previous studies / future plans) - they name systems this study did not use
BACKGROUND_SENT = _R(r"\b(?:previous(?:ly)?|earlier (?:studies|work|reports)|prior (?:studies|work|reports)|has been (?:shown|reported|"
                     r"demonstrated|described|tested|linked|evaluated|studied|investigated|proposed|suggested|associated)|"
                     r"have been (?:shown|reported|demonstrated|described|tested|linked|evaluated|studied|investigated|proposed|suggested)|"
                     r"(?:was|were) (?:previously|formerly)|(?:studies|reports|others) have|it is (?:well )?known|is known to|"
                     r"known to|future (?:studies|work|research|testing|experiments)|warrant\w* (?:further|future)|"
                     r"should be (?:tested|evaluated|examined|investigated)|remains? to be|will be (?:needed|required|tested)|"
                     r"further (?:studies|investigation|research) (?:are|is) (?:needed|required|warranted)|"
                     r"support (?:future|further) (?:testing|studies|work)|"
                     # 'Metformin has antiproliferative effects in mice' / 'preclinical studies suggest ...' (rationale, not this study)
                     r"ha(?:s|ve) (?:\w+[- ]){0,3}(?:effects?|activity|activities|properties) in (?:mice|rats|animals|vitro|"
                     r"preclinical|experimental|rodent)\w*|preclinical (?:studies|data|evidence|models|work) (?:have |has )?"
                     r"(?:show|suggest|indicate|demonstrat|report)\w*)\b")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z(])")       # also lower-cased passages (the audit's normalised context)


def own_work_text(text: str) -> str:
    """The text without sentences about other / previous / future work, so 'metformin was previously shown to
    reduce tumour growth in mice' cannot turn a cell-line study into an animal study."""
    kept = [s for s in _SENT.split(text or "") if not BACKGROUND_SENT.search(s)]
    return " ".join(kept)


# reagents named after an animal are not animal experiments: 'fetal bovine serum', 'goat anti-rabbit IgG',
# 'mouse monoclonal antibody', 'horseradish peroxidase', 'bovine serum albumin'
_REAGENT = _R(r"\b(?:f(?:o)?etal (?:bovine|calf) serum|(?:new[- ]born |newborn )?calf serum|bovine serum albumin|horse serum|goat serum|"
              r"donkey serum|chicken serum|normal (?:goat|donkey|horse|rabbit|mouse) serum|horseradish|"
              r"(?:mouse|rabbit|goat|donkey|rat|horse|sheep|chicken|guinea pig|hamster|camel|llama)\s+(?:anti-?[\w\-]+|"
              r"monoclonal|polyclonal|igg\w*|igm|secondary|primary antibod\w*|antibod\w*|antisera?|serum)|"
              r"anti-(?:mouse|rabbit|goat|rat|donkey|sheep|chicken|human) \w+|(?:bovine|porcine|ovine) (?:insulin|trypsin|collagen|"
              r"gelatin|albumin|pituitary extract)|\bFBS\b|\bBSA\b)")


def _strip_reagents(text: str) -> str:
    return _REAGENT.sub(" ", text or "")


def _n(rx: re.Pattern, text: str) -> int:
    return len(rx.findall(text or ""))


def _designs_in(text: str, strict_animal: bool = True) -> List[str]:
    """Primary designs with direct cues in a piece of text (title, or a quote's neighbourhood)."""
    t = _strip_reagents(text or "")
    found = []
    people = _n(HUMAN_PEOPLE, t)
    animal = _n(ANIMAL, t) + (_n(IN_VIVO, t) if not strict_animal else 0)
    if animal and _n(CELL, t) and not (IN_ANIMAL.search(t) or ANIMAL_TREATED.search(t) or IN_VIVO.search(t) or
                                       re.search(r"\b(?:xenograft\w*|nude mice|athymic|orthotopic\w*|syngeneic|tumou?r volume|"
                                                 r"in vivo|transgenic|knockout mice|animal models?|(?:mouse|mice|murine|rat|rodent) "
                                                 r"(?:\w+ )?models?|tumou?r[- ]bearing|mg/kg|(?:mice|rats|animals)\b[^.;]{0,50}?\b"
                                                 r"(?:treated|given|received|injected|fed|bearing|administered|gavaged|randomi[sz]ed|"
                                                 r"inoculated|implanted))\b", t, re.I)):
        # 'murine 4T1 cells', 'mouse embryonic fibroblasts': an animal word that only names where the CELLS come from
        animal = 0
    exposed_people = bool(HUMAN_EXPOSED.search(t))
    human_sample = bool(HUMAN_SAMPLE.search(t)) and not CULTURE_STEP.search(t)
    if (exposed_people or human_sample or TREATED_PEOPLE.search(t)) and not animal:
        found.append("human_observational")       # people treated / followed without a stated control = treated cohort
    if _n(TRIAL, t) and people and (not animal or people >= animal):
        found.append("human_clinical")            # a randomized trial in women that mentions mice in passing is still human
    human_dominant = people >= animal and not ANIMAL_TREATED.search(t)     # patients' records that mention mice in passing
    if (_n(OBSERVATIONAL, t) and (people or re.search(r"\b(TCGA|cancer genome atlas|patient|clinical)", t, re.I))) \
            and (not animal or human_dominant):
        found.append("human_observational")
    if animal:
        found.append("animal")
    if _n(ORGANOID, t) and not animal:
        found.append("organoid_ex_vivo")
    if _n(CELL, t) and not (exposed_people and not CULTURE_STEP.search(t) and not re.search(r"cell lines?", t, re.I)):
        found.append("cell_culture")
    if _n(PLANT, t) and not people and not animal:
        found.append("plant")
    if _n(MICROBE, t) and not people and not animal and not _n(CELL, t):
        found.append("microbial")
    if _n(IN_SILICO, t) and not (people or animal or _n(CELL, t)):
        found.append("in_silico")
    return found


def _strongest(designs: List[str]) -> Optional[str]:
    ranked = [d for d in DESIGN_RANK if d in designs]
    return ranked[-1] if ranked else None


REVIEW_ARTICLE_TYPES = {"review-article", "systematic-review", "meta-analysis", "review", "editorial", "letter", "commentary",
                        "book-review", "article-commentary"}


_TEXT_SPECIES = [("Rattus norvegicus (rat)", r"\brats?\b|\brattus\b"), ("Mus musculus (mouse)", r"\bmice\b|\bmouse\b|\bmurine\b|mus musculus"),
                 ("Danio rerio (zebrafish)", r"\bzebrafish\b"), ("Canis familiaris (dog)", r"\bdogs?\b|\bcanine\b"),
                 ("Sus scrofa (pig)", r"\bpigs?\b|\bporcine\b|\bswine\b")]


def _species_from_text(organism: Optional[str], text: str) -> Optional[str]:
    """The AI's organism, unless the paper's own title / abstract names exactly one OTHER animal species ('obese rats'
    extracted as 'Mus musculus'): the paper's words win."""
    in_text = [name for name, rx in _TEXT_SPECIES if re.search(rx, text or "", re.I)]
    if len(in_text) != 1:
        return organism
    ai = [name for name, rx in _TEXT_SPECIES if re.search(rx, organism or "", re.I)]
    if ai and ai[0] != in_text[0]:
        return in_text[0]
    return organism


def classify_study(paper: Dict[str, Any], extraction: Optional[Dict[str, Any]] = None,
                   methods_text: str = "", article_type: Optional[str] = None,
                   full_text_sections: Optional[List[str]] = None) -> Dict[str, Any]:
    """Pure function: decide the evidence type of ONE paper by an ordered rule list.

    Rule order (first that fires wins):
      R1 review/meta-analysis   title says review/meta-analysis, or the abstract describes a literature search
      R2 title design           strongest primary design named in the TITLE (title is the most reliable signal)
      R3 abstract + methods     strongest design with direct cues in abstract + methods
      R4 structured extraction  the study_type from the paper-extraction step (last resort only)
      R5 unclear                nothing reliable -> 'unclear' (never guessed)
    """
    extraction = extraction or {}
    study = extraction.get("study") if isinstance(extraction.get("study"), dict) else {}
    title = (paper.get("title") or "").replace("[SYNTHETIC]", "").strip()
    abstract = paper.get("abstract") or ""
    body = f"{own_work_text(abstract)}\n{own_work_text((methods_text or '')[:4000])}"   # this study's own work only
    llm_type = str(study.get("study_type") or extraction.get("study_type") or "").strip()
    organism = _species_from_text(study.get("organism") or extraction.get("organism") or None, f"{title} {abstract}")

    signals = {k: _n(rx, f"{title} {body}") for k, rx in (
        ("review", REVIEW_ABSTRACT), ("trial", TRIAL), ("observational", OBSERVATIONAL), ("people", HUMAN_PEOPLE),
        ("animal", ANIMAL), ("organoid", ORGANOID), ("cell", CELL), ("in_silico", IN_SILICO),
        ("plant", PLANT), ("microbe", MICROBE))}
    secondary = sorted(set(_designs_in(body)) | set(_designs_in(title)))

    def done(etype: str, rule: str) -> Dict[str, Any]:
        return {
            "evidence_type": etype, "label": EVIDENCE_TYPES[etype], "rule": rule,
            "organism_group": organism_group(etype, organism, f"{title} {abstract}"),
            "organism": organism, "signals": {k: v for k, v in signals.items() if v},
            "secondary_designs": [d for d in secondary if d != etype],
            "classifier_version": CLASSIFIER_VERSION,
        }

    # R0 the publisher's own article type (JATS <article article-type="review-article">) - the most reliable signal
    if (article_type or "").strip().lower() in REVIEW_ARTICLE_TYPES:
        return done("review_meta", f"R0: the publisher marks it as '{article_type}' (secondary literature)")
    # R0b full text with sections but no Methods / Results at all: a narrative review, not a primary study
    secs = [x for x in (full_text_sections or []) if x != "article_meta"]
    if len(secs) >= 3 and not ({"methods", "results", "results_discussion"} & set(secs)) and not PRIMARY_ABSTRACT.search(abstract):
        return done("review_meta", "R0b: the full text has no Methods or Results sections and the abstract reports no primary results")

    # R1 secondary literature
    if REVIEW_TITLE.search(title) and not CASE_REPORT.search(title):
        return done("review_meta", "R1: title identifies a review / meta-analysis")
    if _n(REVIEW_ABSTRACT, abstract) >= 1 and (llm_type in ("meta_analysis", "systematic_review")
                                               or _n(REVIEW_ABSTRACT, abstract) >= 2
                                               or re.match(r"\s*(this|in this|we) review", abstract, re.I)):
        return done("review_meta", "R1: abstract describes a literature search / pooled analysis")

    if REVIEW_LIKE_TITLE.search(title) and not PRIMARY_ABSTRACT.search(abstract) and \
            (REVIEW_LIKE_ABSTRACT.search(abstract) or not abstract.strip()):
        return done("review_meta", "R1: review-style title and the abstract reports no primary methods or results")
    if len(REVIEW_LIKE_ABSTRACT.findall(abstract)) >= 2 and not PRIMARY_ABSTRACT.search(abstract):
        return done("review_meta", "R1: abstract summarises other work and reports no primary methods or results")

    # R1 commentary / editorial / perspective: an opinion piece about a question has no data of its own
    if COMMENTARY_TITLE.search(title) and not PRIMARY_ABSTRACT.search(abstract) and not _n(TRIAL, title):
        return done("review_meta", "R1: editorial / commentary / perspective title and no primary methods or results")

    # R1b case report / series (uncontrolled; human or veterinary) - checked before animal/human designs
    if CASE_REPORT.search(title) or (CASE_REPORT.search(abstract[:600]) and not _n(TRIAL, body) and not _n(OBSERVATIONAL, title)):
        return done("case_report", "R1b: case report / case series (uncontrolled observation)")

    # R1c deliberately induced disease model (kept apart from naturally occurring disease)
    if EXP_MODEL.search(title) or (EXP_MODEL.search(body) and re.search(r"\bmodel\b|experimental", body, re.I)):
        return done("experimental_disease_model", "R1c: experimentally induced disease model")

    def organism_says(design: str) -> Optional[str]:
        """The extraction READ the paper and named its organism. When that organism is a plant or a microbe and the
        regex call is 'human' without a trial or a result in people, the regex was fooled by a generic word."""
        o = str(organism or "")
        if design not in ("human_clinical", "human_observational") or not o:
            return None
        if _n(TRIAL, f"{title} {body}") or HUMAN_RESULT.search(body) or re.search(r"homo sapiens|human", o, re.I):
            return None
        if PLANT.search(o):
            return "plant"
        if MICROBE.search(o):
            return "microbial"
        if ANIMAL.search(o):
            return "animal"
        return None

    # R2 design named in the title
    t_design = _strongest(_designs_in(title))
    if t_design:
        o = organism_says(t_design)
        if o:
            return done(o, f"R2+R4: the title reads as {EVIDENCE_TYPES[t_design]}, but the paper's organism is '{organism}'")
        return done(t_design, f"R2: design named in the title ({EVIDENCE_TYPES[t_design]})")

    # R3 abstract + methods (animal needs a concrete animal word; 'in vivo' alone is not enough)
    b_design = _strongest(_designs_in(body))
    if not b_design and OBSERVATIONAL.search(title) and HUMAN_PEOPLE.search(body) and not ANIMAL.search(body):
        b_design = "human_observational"     # 'a cohort study' in the title + patients in the abstract
    if b_design:
        o = organism_says(b_design)
        if o:
            return done(o, f"R3+R4: abstract cues read as {EVIDENCE_TYPES[b_design]}, but the paper's organism is '{organism}'")
        return done(b_design, f"R3: design described in abstract/methods ({EVIDENCE_TYPES[b_design]})")

    # R3b an abstract that only summarises / discusses other work, with no design of its own
    if (REVIEW_LIKE_ABSTRACT.search(abstract) or BACKGROUND_SENT.search(abstract)) and not PRIMARY_ABSTRACT.search(abstract):
        return done("review_meta", "R3b: the abstract summarises or discusses other work and describes no study of its own")

    # R4 structured extraction as last resort
    mapped = LLM_TYPE_MAP.get(llm_type)
    if mapped and (mapped not in ("human_clinical", "human_observational") or signals["people"]):
        return done(mapped, f"R4: structured extraction reported study_type={llm_type}")

    return done("unclear", "R5: no reliable design cue - not guessed")


def organism_group(etype: str, organism: Optional[str], text: str = "") -> str:
    if etype in ("human_clinical", "human_observational"):
        return "human"
    if etype == "animal":
        return "animal"
    if etype in ("case_report", "experimental_disease_model"):
        if ANIMAL.search(text) or re.search(r"mus|rattus|bos|bubalus|equus|capra|ovis|felis|canis", (organism or "").lower()):
            return "animal"
        if HUMAN_PEOPLE.search(text) or "homo sapiens" in (organism or "").lower():
            return "human"
        if _n(CELL, text):
            return "unknown"
    if etype in ("plant", "microbial"):
        return etype
    o = (organism or "").lower()
    if etype in ("cell_culture", "organoid_ex_vivo"):
        if "homo sapiens" in o or re.search(r"\bhuman\b|MCF-?7|MDA-MB|HeLa|HEK|A549|HepG2|HCT-?116|patient-derived", text, re.I):
            return "human"
        if re.search(r"mus|mouse|murine|rat", o) or re.search(r"\b(murine|mouse|rat) cells?\b|3T3|RAW ?264", text, re.I):
            return "animal"
        if PLANT.search(text):
            return "plant"
        if MICROBE.search(text):
            return "microbial"
    return "unknown"


# --------------------------------------------------------------------------- storage (immutable)
_SCHEMA = """CREATE TABLE IF NOT EXISTS study_classifications (
    paper_id INTEGER PRIMARY KEY, evidence_type TEXT NOT NULL, organism_group TEXT, organism TEXT,
    rule TEXT, signals_json TEXT, secondary_json TEXT, classifier_version TEXT, classified_at TEXT NOT NULL)"""


def _ensure_table(db: Any) -> None:
    with db._lock:
        db._conn.execute(_SCHEMA)
        db._conn.commit()


def get_classification(db: Any, paper_id: int) -> Optional[Dict[str, Any]]:
    _ensure_table(db)
    row = db._conn.execute("SELECT * FROM study_classifications WHERE paper_id=?", (paper_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    return {"evidence_type": d["evidence_type"], "label": EVIDENCE_TYPES.get(d["evidence_type"], d["evidence_type"]),
            "rule": d["rule"], "organism_group": d["organism_group"], "organism": d["organism"],
            "signals": json.loads(d["signals_json"] or "{}"), "secondary_designs": json.loads(d["secondary_json"] or "[]"),
            "classifier_version": d["classifier_version"], "classified_at": d["classified_at"], "frozen": True}


def get_or_classify(db: Any, paper_id: int) -> Dict[str, Any]:
    """Return the stored classification; classify and freeze it if this is the first time.
    INSERT OR IGNORE + re-read guarantees the first decision wins even under concurrency."""
    stored = get_classification(db, paper_id)
    if stored and stored.get("classifier_version") == CLASSIFIER_VERSION:
        return stored          # frozen: decided once per classifier version, never re-scored downstream
    paper = db.get_paper(paper_id) or {}
    extraction = db.get_extraction_raw(paper_id) or {}
    secs = db.get_sections(paper_id, include_meta=True)
    methods = "\n".join(s["text"] for s in secs if s.get("section_type") == "methods")
    atype = next((s["text"] for s in secs if s.get("section_type") == "article_meta" and s.get("title") == "article-type"), None)
    c = classify_study(paper, extraction, methods, article_type=atype, full_text_sections=[s.get("section_type") for s in secs])
    try:          # v24.2: indexed publication types and the all-cells rule override a weaker text reading
        from metadata_authority import lookup, reconcile_type
        c = reconcile_type(c, paper, lookup(db, paper))
    except Exception:
        pass
    with db._lock:
        db._conn.execute(
            ("INSERT OR REPLACE" if stored else "INSERT OR IGNORE") + " INTO study_classifications (paper_id, evidence_type, organism_group, organism, rule, "
            "signals_json, secondary_json, classifier_version, classified_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (paper_id, c["evidence_type"], c["organism_group"], c["organism"], c["rule"], json.dumps(c["signals"]),
             json.dumps(c["secondary_designs"]), c["classifier_version"], now_iso()))
        db._conn.commit()
    return get_classification(db, paper_id)


# --------------------------------------------------------------------------- evidence levels
def evidence_level(etype: str, focus: str = "human", organism_grp: Optional[str] = None) -> Optional[int]:
    """Level 1-6 for a study design relative to the question focus; None for secondary/unclear.
    Case reports and induced-disease models take the level of the organism they were done in
    (human -> L2 indirect, never L1; animal -> L3)."""
    if etype in ("review_meta", "unclear"):
        return None
    if etype in ("case_report", "experimental_disease_model"):
        if organism_grp == "human":
            return 2 if focus in ("human", "general") else 3
        if organism_grp in ("plant", "microbial", "animal") and organism_grp == focus:
            return 2
        return 3
    if etype == "in_silico":
        return 6
    if etype == "cell_culture":
        return 5
    if etype == "organoid_ex_vivo":
        return 4
    if focus in ("plant", "microbial", "animal"):
        if etype == focus:
            return 1
        return 3 if etype in ("human_clinical", "human_observational", "animal", "plant", "microbial") else 6
    # human-health and general questions: the human hierarchy
    return {"human_clinical": 1, "human_observational": 2, "animal": 3}.get(etype, 5)


def level_label(level: Optional[int], focus: str = "human", etype: Optional[str] = None) -> str:
    if level is None:
        return "S Secondary literature (not counted)" if etype == "review_meta" else "Unclassified (not counted)"
    names = HUMAN_LEVEL_NAMES if focus in ("human", "general") else LEVEL_NAMES
    return names[level]


# --------------------------------------------------------------------------- model normalisation
SYSTEM_OF = {"human_clinical": "in vivo (human)", "human_observational": "in vivo (human, observational)",
             "animal": "in vivo (animal)", "plant": "in vivo (whole plant)", "microbial": "microbial culture / organism",
             "organoid_ex_vivo": "ex vivo / organoid", "cell_culture": "in vitro (cells)", "in_silico": "in silico",
             "case_report": "in vivo (single case / series)", "experimental_disease_model": "in vivo (induced disease)",
             "review_meta": "secondary literature", "unclear": "unclear"}


def model_profile(etype: str, organism_grp: Optional[str], organism: Optional[str], focus: str = "human") -> Dict[str, Any]:
    """Explicit translational relationship of a model to the organism/setting the question is about.
    distance 0 = target organism, natural disease, controlled; each gap (uncontrolled, induced disease,
    other species, outside a living organism, computational) adds one step."""
    target = "human" if focus in ("human", "general") else focus
    gaps = []
    if organism_grp and organism_grp not in ("unknown", target):
        gaps.append(f"different species/organism group ({organism or organism_grp} vs {target})")
    if etype in ("organoid_ex_vivo", "cell_culture"):
        gaps.append("outside a living organism")
    if etype == "in_silico":
        gaps += ["computational only", "no biological material"]
    if etype == "experimental_disease_model":
        gaps.append("disease deliberately induced (may not behave like the naturally occurring disease)")
    if etype == "case_report":
        gaps.append("single uncontrolled case(s)")
    if etype == "human_observational":
        gaps.append("observational (no assigned intervention)")
    if etype in ("review_meta", "unclear"):
        gaps.append("not a primary study")
    return {"system": SYSTEM_OF.get(etype, etype), "species": organism or organism_grp or "unknown",
            "target": target, "translational_distance": len(gaps),
            "relation": "; ".join(gaps) if gaps else f"direct: {target}, natural disease, controlled design"}


# --------------------------------------------------------------------------- claim-level model
_SEC_ANIMAL = re.compile(r"\bin vivo\b|xenograft\w*|\b(?:nude |SCID |transgenic )?mice\b|\bmouse\b|\bmurine\b|\brats?\b|"
                         r"\banimals?\b|orthotopic|syngeneic|zebrafish|tumou?r-bearing|\bPDX\b", re.I)
_SEC_CELL = re.compile(r"\bin vitro\b|\bcells?\b|cell lines?|cultur\w*|cytotox\w*|viability|colony|clonogenic|"
                       r"wound[- ]heal\w*|transwell|spheroid\w*|\bMTT\b|\bCCK-?8\b|organoids?", re.I)
_SEC_HUMAN = re.compile(r"\bpatients?\b|participants?|\bcohort\b|\btrial\b|\bclinical\b|\bwomen\b|\bsubjects\b|"
                        r"volunteers|baseline characteristics|survival analysis", re.I)
_SEC_GENERIC = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:abstract|results?|discussion|introduction|background|"
                          r"conclusions?|methods?|materials and methods|text|body|other)\s*$", re.I)


def section_model(section_title: Optional[str]) -> Optional[Dict[str, Any]]:
    """The experimental system named by the heading of the section a result sits in ('3.1 In vitro cytotoxicity
    assays' -> cell culture; 'Metformin inhibits the growth of 4T1 cell xenografts in nude mice' -> animal). Used when
    the quote itself names no system, instead of falling back to the paper's overall type. None if the heading is
    generic ('Results') or names no system / several systems at the same level."""
    t = (section_title or "").strip()
    if not t or _SEC_GENERIC.match(t):
        return None
    animal, cell, human = bool(_SEC_ANIMAL.search(t)), bool(_SEC_CELL.search(t)), bool(_SEC_HUMAN.search(t))
    if animal:                              # cells injected INTO animals ('4T1 cell xenografts in nude mice') = animal
        return {"model_type": "animal", "source": f"section heading '{t[:80]}' (in vivo)"}
    if cell and not re.search(r"patient-derived|primary (?:tumou?r )?(?:tissue|samples)", t, re.I):
        return {"model_type": "cell_culture", "source": f"section heading '{t[:80]}' (in vitro)"}
    if re.search(r"organoids?|patient-derived|ex vivo|explants?", t, re.I):
        return {"model_type": "organoid_ex_vivo", "source": f"section heading '{t[:80]}' (ex vivo)"}
    if human and not cell:
        return {"model_type": "human_clinical" if re.search(r"\btrial|randomi[sz]", t, re.I) else "human_observational",
                "source": f"section heading '{t[:80]}' (people)"}
    return None


_CELL_WEAK = _R(r"\b(?:cells?|PBCCs?|incubat\w*|\d+(?:\.\d+)?\s*(?:mM|µM|uM|μM)|cell (?:migration|invasion|density|viability|"
                r"proliferation|cycle|growth|number)|colon(?:y|ies)|in culture)\b")


def claim_model(local_text: str, paper_type: str, paper_secondary: List[str]) -> Dict[str, Any]:
    """Which experimental system a specific claim comes from (a clinical paper can contain a
    cell-line experiment). Uses the quote and its surrounding passage; falls back to the
    paper's frozen type. Returns {'model_type', 'source'}."""
    present = set([paper_type] + list(paper_secondary or []))
    # the passage around a quote often carries the rationale ('Metformin has antiproliferative effects in mice') -
    # background sentences never decide which system THIS result comes from
    local_text = own_work_text(local_text or "") or local_text
    # an in-vitro-only instrument or assay next to the result settles the system before anything else:
    # the paper may be a patient study, but THIS measurement was made on cultured cells (v24 fix 1)
    if IN_VITRO_ASSAY.search(local_text or "") and not IN_ANIMAL.search(local_text or "") and \
            not HUMAN_EXPOSED.search(local_text or ""):
        return {"model_type": "cell_culture",
                "source": "quote context (an assay that is only done on cultured cells)"}
    if HUMAN_EXPOSED.search(local_text or "") or (HUMAN_SAMPLE.search(local_text or "") and not CULTURE_STEP.search(local_text or "")):
        if paper_type == "human_clinical" and not HUMAN_EXPOSED.search(local_text or ""):
            # tissue / blood taken from the participants OF A TRIAL is still the trial's (randomised) evidence
            return {"model_type": "human_clinical", "source": "quote context (samples from the trial's participants)"}
        return {"model_type": "human_observational", "source": "quote context (material from exposed people / patients)"}
    lt = local_text or ""
    if paper_type == "human_clinical" and re.search(r"\b(?:breast|tumou?r|blood|plasma|serum|tissue|biops\w+)\b", lt, re.I) and \
            not (CULTURE_STEP.search(lt) or ANIMAL.search(_strip_reagents(lt))):
        if not re.search(r"\b(?:cohort|registry|retrospective|observational|case[- ]control)\b", lt, re.I):
            return {"model_type": "human_clinical", "source": "quote context (measured in the trial's participants)"}
    # explicit people in a controlled comparison next to the result ('Compared with placebo, metformin decreased Ki-67
    # in women ...') - human evidence, whatever the paper was (mis)classified as
    if HUMAN_RESULT.search(lt) and not CULTURE_STEP.search(lt) and not IN_ANIMAL.search(lt):
        return {"model_type": "human_clinical", "source": "quote context (people in a placebo / randomized comparison)"}
    local = _designs_in(local_text or "", strict_animal=False)
    if "animal" in local and IN_ANIMAL.search(local_text or "") and not CULTURE_STEP.search(local_text or ""):
        return {"model_type": "animal", "source": "quote context (treatment given to animals)"}
    # 'Twenty-four-hour incubation with 25 mM metformin reduced the proliferation of PBCCs', 'suppress the m6A level in
    # breast cancer cells': cells / an incubation / a millimolar dose with NO animal word next to the result is a
    # cell experiment, even in a paper that also has a mouse experiment
    if paper_type in ("animal", "experimental_disease_model", "unclear") and "cell_culture" not in local and \
            _CELL_WEAK.search(lt) and not (ANIMAL.search(_strip_reagents(lt)) or IN_ANIMAL.search(lt) or IN_VIVO.search(lt)):
        return {"model_type": "cell_culture", "source": "quote context (cells / incubation, no animal named)"}
    # the most specific system named next to the result wins (cells > animal > human)
    for d in ("cell_culture", "organoid_ex_vivo", "animal", "plant", "microbial", "human_clinical",
              "human_observational", "in_silico"):
        if d in local and (d in present or paper_type in ("unclear",)):
            return {"model_type": d, "source": "quote context"}
    for d in ("cell_culture", "organoid_ex_vivo", "animal"):
        if d in local:
            return {"model_type": d, "source": "quote context (design not listed for the paper)"}
    # people named next to the result, nothing about animals or cells: never an animal / cell finding
    if paper_type in ("animal", "cell_culture", "organoid_ex_vivo", "unclear") and \
            re.search(r"\b(?:women|men|patients|participants|subjects|volunteers|children|adults)\b", lt, re.I) \
            and not (ANIMAL.search(lt) or CELL.search(lt) or CULTURE_STEP.search(lt)):
        return {"model_type": "human_clinical" if TRIAL.search(lt) else "human_observational",
                "source": "quote context (people named next to the result)"}
    return {"model_type": paper_type, "source": "paper classification"}
