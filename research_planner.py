import json
import re
import time

from ai_router import AIRouter


# ==========================================================
# CONFIGURATION
# ==========================================================

MAX_PLANNER_RETRIES = 3


# ==========================================================
# PROMPT
# ==========================================================

PLANNER_PROMPT = """
You are one component of a general-purpose biology research AI.

The user has supplied this research question:

{question}

Your job is to design a literature-search strategy for this
question.

Generate exactly 5 distinct scientific literature search
queries.

The queries should collectively cover different relevant
dimensions of the question, such as:

- mechanisms
- molecular or cellular causes
- biological pathways
- environmental or physiological factors
- experimental evidence
- conflicting findings
- relevant organisms or disease models
- established terminology and synonyms
- measurable surrogate endpoints and biomarkers

Do NOT assume that the question belongs to one particular
subfield of biology.

When the question names a biological endpoint (for example
proliferation, apoptosis, metastasis, inflammation, insulin
sensitivity, growth, stress tolerance or virulence), at least one
query MUST use the measurable surrogate markers or assays that
studies actually report instead of the abstract endpoint word.
Examples: proliferation -> Ki-67, PCNA, mitotic index, BrdU;
apoptosis -> cleaved caspase-3, TUNEL; metastasis -> circulating
tumor cells; inflammation -> IL-6, C-reactive protein; plant
drought tolerance -> relative water content, stomatal conductance;
antibiotic resistance -> minimum inhibitory concentration (MIC).
Also include common synonyms and abbreviations of key terms.

Use terminology appropriate for scientific literature
databases such as Europe PMC.

Important:
- Do not answer the research question.
- Do not write explanations.
- Do not invent papers.
- Do not cite sources.
- Do not make unsupported scientific claims.
- Return ONLY valid JSON.

Return exactly this structure:

{{
    "research_question": "{question_json}",
    "search_queries": [
        "query 1",
        "query 2",
        "query 3",
        "query 4",
        "query 5"
    ]
}}
"""


# ==========================================================
# JSON PARSER
# ==========================================================

def parse_json_response(text):
    """
    Extract JSON from a model response.

    Models sometimes return JSON inside markdown code fences.
    """

    if not text:
        raise ValueError(
            "Model returned an empty response."
        )

    text = text.strip()

    # Remove markdown code fences.
    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

    # Find JSON object if the model added surrounding text.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "No JSON object found in model response."
        )

    text = text[start:end + 1]

    return json.loads(text)


# ==========================================================
# VALIDATION
# ==========================================================

def validate_plan(plan):
    """
    Validate the structure returned by a model.
    """

    if not isinstance(plan, dict):

        raise ValueError(
            "Research plan is not a JSON object."
        )

    if "research_question" not in plan:

        raise ValueError(
            "Research plan is missing 'research_question'."
        )

    if "search_queries" not in plan:

        raise ValueError(
            "Research plan is missing 'search_queries'."
        )

    queries = plan["search_queries"]

    if not isinstance(queries, list):

        raise ValueError(
            "'search_queries' must be a list."
        )

    cleaned_queries = []

    for query in queries:

        if not isinstance(query, str):
            continue

        query = query.strip()

        if query:
            cleaned_queries.append(query)

    if not cleaned_queries:

        raise ValueError(
            "Research plan contains no usable search queries."
        )

    plan["search_queries"] = cleaned_queries

    return plan


# ==========================================================
# MODEL PLANNER
# ==========================================================

def generate_model_plan(
    router,
    question,
    provider
):
    """
    Ask one AI provider for an independent research plan.

    Failed providers are retried because temporary API/network
    failures should not immediately remove a model from the
    research process.
    """

    question_json = json.dumps(
        question,
        ensure_ascii=False
    )

    prompt = PLANNER_PROMPT.format(
        question=question,
        question_json=question_json
    )

    print()
    print("=" * 80)
    print(
        f"GENERATING PLAN WITH {provider.upper()}"
    )
    print("=" * 80)

    last_error = None

    for attempt in range(1, MAX_PLANNER_RETRIES + 1):

        try:

            response = router.ask(
                prompt,
                provider=provider
            )

            plan = parse_json_response(
                response
            )

            plan = validate_plan(
                plan
            )

            print(
                f"{provider.upper()} planner succeeded "
                f"on attempt {attempt}."
            )

            return plan

        except Exception as error:

            last_error = error

            print()
            print(
                f"{provider.upper()} attempt "
                f"{attempt}/{MAX_PLANNER_RETRIES} failed:"
            )

            print(error)

            if attempt < MAX_PLANNER_RETRIES:

                wait_time = 3 * attempt

                print(
                    f"Retrying {provider.upper()} "
                    f"in {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

    print()
    print(
        f"{provider.upper()} planner failed after "
        f"{MAX_PLANNER_RETRIES} attempts."
    )

    print(
        "The research pipeline will continue using "
        "the other available models."
    )

    print(
        f"Last error: {last_error}"
    )

    return None


# ==========================================================
# QUERY NORMALIZATION
# ==========================================================

def normalize_query(query):
    """
    Normalize a query only for duplicate detection.

    The original query remains unchanged.
    """

    query = query.lower().strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    query = query.strip(
        " .,!?:;\"'"
    )

    return query


# ==========================================================
# MERGE QUERIES
# ==========================================================

def merge_queries(model_plans):
    """
    Merge all successful model plans.

    Different models are NOT ranked against one another.

    Their queries are treated as independent search strategies.
    """

    merged = []

    seen = set()

    for provider, plan in model_plans.items():

        if not plan:
            continue

        for query in plan["search_queries"]:

            normalized = normalize_query(
                query
            )

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            merged.append(
                {
                    "provider": provider,
                    "query": query
                }
            )

    return merged


# ==========================================================
# PRINT RESULTS
# ==========================================================

def print_plan(
    question,
    model_plans,
    merged_queries
):

    print()
    print("=" * 80)
    print("BIOLOGY RESEARCH PLAN")
    print("=" * 80)

    print()
    print("RESEARCH QUESTION:")
    print("-" * 80)
    print(question)

    # ------------------------------------------------------
    # Individual plans
    # ------------------------------------------------------

    for provider, plan in model_plans.items():

        print()
        print("=" * 80)
        print(
            f"{provider.upper()} SEARCH STRATEGY"
        )
        print("=" * 80)

        if not plan:

            print(
                "No plan returned."
            )

            continue

        for number, query in enumerate(
            plan["search_queries"],
            start=1
        ):

            print(
                f"{number}. {query}"
            )

    # ------------------------------------------------------
    # Combined strategy
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("COMBINED SEARCH STRATEGY")
    print("=" * 80)

    if not merged_queries:

        print(
            "No search queries were generated."
        )

    else:

        for number, item in enumerate(
            merged_queries,
            start=1
        ):

            print()
            print(
                f"{number}. "
                f"[{item['provider'].upper()}]"
            )

            print(
                f"   {item['query']}"
            )

    print()
    print(
        f"TOTAL UNIQUE SEARCH QUERIES: "
        f"{len(merged_queries)}"
    )


# ==========================================================
# SAVE PLAN
# ==========================================================

def save_plan(
    question,
    model_plans,
    merged_queries,
    filename="research_plan.json"
):

    data = {
        "research_question": question,
        "model_plans": model_plans,
        "merged_queries": merged_queries
    }

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(
        f"Research plan saved to: {filename}"
    )


# ==========================================================
# MAIN
# ==========================================================

import sys  # noqa: E402

if __name__ == "__main__":
    # --help and closed / interrupted input exit cleanly instead of with a traceback
    import builtins as _bi
    if any(_a in ("-h", "--help") for _a in sys.argv[1:]):
        print("usage: python research_planner.py\n\nInteractive research planner: asks for a biology research question and prints search queries / a plan.\nPress Ctrl+C at any prompt to quit.")
        raise SystemExit(0)
    _real_input = _bi.input

    def _safe_input(prompt=""):
        try:
            return _real_input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nNo input - exiting.")
            raise SystemExit(0)
    _bi.input = _safe_input

    print("=" * 80)
    print("MULTI-MODEL BIOLOGY RESEARCH PLANNER")
    print("=" * 80)

    question = input(
        "\nEnter your biology research question:\n> "
    ).strip()

    if not question:

        print()
        print(
            "ERROR: Research question cannot be empty."
        )

        raise SystemExit(1)

    # ------------------------------------------------------
    # Create router
    # ------------------------------------------------------

    router = AIRouter()

    # ------------------------------------------------------
    # Detect providers
    # ------------------------------------------------------

    providers = []

    if router.gemini:
        providers.append("gemini")

    if router.groq:
        providers.append("groq")

    if router.nvidia:
        providers.append("nvidia")

    print()
    print(
        "Available AI providers:",
        ", ".join(
            provider.upper()
            for provider in providers
        )
    )

    if not providers:

        print()
        print(
            "ERROR: No AI providers are configured."
        )

        raise SystemExit(1)

    # ------------------------------------------------------
    # Generate independent plans
    # ------------------------------------------------------

    model_plans = {}

    for provider in providers:

        model_plans[provider] = (
            generate_model_plan(
                router,
                question,
                provider
            )
        )

    # ------------------------------------------------------
    # Merge queries
    # ------------------------------------------------------

    merged_queries = merge_queries(
        model_plans
    )

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print_plan(
        question,
        model_plans,
        merged_queries
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    save_plan(
        question,
        model_plans,
        merged_queries
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    successful = sum(
        1
        for plan in model_plans.values()
        if plan is not None
    )

    total = len(
        model_plans
    )

    print()
    print("=" * 80)
    print("PLANNER COMPLETE")
    print("=" * 80)

    print(
        f"Successful model planners: "
        f"{successful}/{total}"
    )

    print(
        f"Unique search queries: "
        f"{len(merged_queries)}"
    )

    if successful < total:

        print()
        print(
            "WARNING: One or more AI providers failed."
        )

        print(
            "The successful providers were still used."
        )

    else:

        print()
        print(
            "All configured AI providers "
            "completed successfully."
        )

# ==========================================================
# DETERMINISTIC ENDPOINT -> BIOMARKER EXPANSION (no AI needed)
# ==========================================================
# Studies rarely report an abstract endpoint word ("proliferation"); they report
# the surrogate they measured (Ki-67, PCNA ...). Adding one query that names those
# surrogates finds trials a plain keyword search misses. Works without an AI key.

ENDPOINT_SURROGATES = [
    (r"proliferat|cell growth|tumou?r growth|cell division", ["Ki-67", "PCNA", "mitotic index", "BrdU"]),
    (r"apoptos|cell death|programmed death", ["cleaved caspase-3", "TUNEL", "Bcl-2", "PARP cleavage"]),
    (r"metasta|invasion|invasive|cell migration", ["circulating tumor cells", "E-cadherin", "epithelial-mesenchymal transition", "MMP-9"]),
    (r"inflammat", ["IL-6", "TNF-alpha", "C-reactive protein", "NF-kB"]),
    (r"insulin sensitiv|insulin resist|glycemic|glycaemic|blood glucose|glucose control", ["HbA1c", "HOMA-IR", "fasting glucose", "clamp"]),
    (r"oxidative stress|redox", ["malondialdehyde", "glutathione", "reactive oxygen species", "superoxide dismutase"]),
    (r"angiogen|vasculari[sz]ation", ["VEGF", "microvessel density", "CD31"]),
    (r"fibros", ["collagen deposition", "alpha-SMA", "TGF-beta", "hydroxyproline"]),
    (r"senescen|ageing|aging|lifespan|longevity", ["p16INK4a", "SA-beta-gal", "telomere length", "survival curve"]),
    (r"autophag", ["LC3-II", "p62", "Beclin-1"]),
    (r"memory|cognit|learning", ["Morris water maze", "novel object recognition", "MMSE", "cognitive score"]),
    (r"neurodegenerat|alzheimer|parkinson", ["amyloid-beta", "tau", "alpha-synuclein", "neuronal loss"]),
    (r"drought|water stress|water deficit", ["relative water content", "stomatal conductance", "abscisic acid", "proline"]),
    (r"salt|salinity", ["Na+/K+ ratio", "ion homeostasis", "proline", "salt tolerance index"]),
    (r"photosynthe", ["chlorophyll fluorescence", "Fv/Fm", "net photosynthetic rate", "chlorophyll content"]),
    (r"\byield\b|crop productivity", ["grain yield", "biomass", "harvest index"]),
    (r"antibiotic resist|antimicrobial resist|drug resist", ["minimum inhibitory concentration", "MIC", "efflux pump", "resistance genes"]),
    (r"biofilm", ["crystal violet assay", "biofilm biomass", "quorum sensing", "extracellular polymeric substance"]),
    (r"virulen|pathogenicity|infectivity", ["LD50", "colonization", "bacterial load", "infection model"]),
    (r"gene expression|transcription", ["qPCR", "RNA-seq", "transcriptome"]),
    (r"body weight|obesity|adiposity|fat mass", ["BMI", "fat mass", "adipocyte size", "waist circumference"]),
]

_STOP = set("""a an the of in on at to for with and or by from as into about is are was were be been being
does do did how what which why when where who whom whether can could may might will would should shall
effect effects affect affects affecting impact impacts influence influences role roles mechanism mechanisms
reduce reduces reducing increase increases increasing decrease decreases improve improves inhibit inhibits
lower lowers raise raises regulate regulates cause causes via through between among during under over
humans human people patients subjects vs versus its their this that these those there any some
drive drives driving work works happen happens develop develops lead leads
slow slows boost boosts extend extends delay delays worsen worsens protect protects promote promotes induce induces
enhance enhances suppress suppresses reverse reverses help helps really actually good bad safe""".split())


def endpoint_expansion_queries(question, existing=None, max_queries=2):
    """Return up to `max_queries` extra search queries that pair the question's key
    terms with the surrogate biomarkers / assays of the endpoints it mentions."""
    q = question or ""
    existing_text = " ".join(existing or []).lower()
    out = []
    for pattern, terms in ENDPOINT_SURROGATES:
        if not re.search(pattern, q, re.I):
            continue
        if any(t.lower() in existing_text for t in terms):
            continue   # the planner already covered this endpoint
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-+/]*", q)
        core = [w for w in words if w.lower() not in _STOP and not re.search(pattern, w, re.I)][:6]
        if not core:
            continue
        ors = " OR ".join(f'"{t}"' if " " in t else t for t in terms)
        out.append(f"{' '.join(core)} ({ors})")
        if len(out) >= max_queries:
            break
    return out


# ----------------------------------------------------------------------------------------------------
# KEYWORD QUERY BUILDER (no AI key needed)
# A natural-language sentence ("Can leukimia be cured?") is a poor database query: PubMed returns nothing and
# fuzzy sources (Crossref, OpenAlex) return word matches such as image-classification papers. The question is
# turned into concept queries instead: disease (+ spelling variants, subtypes) x what is asked (cure / cause /
# effect / prognosis) x named exposure and endpoint biomarkers.
# ----------------------------------------------------------------------------------------------------

INTENT_TERMS = {
    "cure": ["cure", "curative", '"complete remission"', '"complete response"', "relapse", '"long-term follow-up"'],
    "treatment": ["treatment", "therapy", "trial", "efficacy"],
    "etiology": ['"risk factor"', "etiology", "exposure", "cohort", '"case-control"'],
    "prognosis": ["survival", "prognosis", "outcome"],
    "epidemiology": ["incidence", "prevalence"],
    "mechanism": ["mechanism", "pathway", "signaling"],
    "clinical_features": ["symptoms", "signs", '"clinical features"', '"clinical presentation"', "manifestations", "diagnosis"],
}
CANCER_CURE_TERMS = ['"cure rate"', '"event-free survival"', '"long-term survival"', '"treatment-free remission"']
_CANCER = re.compile(r"cancer|leuk(?:a)?emi|lymphoma|myeloma|carcinoma|tumou?r|sarcoma|glioma|melanoma|neoplas", re.I)
# readable subtype names for heterogeneous diseases (the DISEASES lexicon holds regexes)
SUBTYPE_QUERY_NAMES = {
    "leukemia": ['"acute lymphoblastic leukemia"', '"acute myeloid leukemia"', '"chronic myeloid leukemia"',
                 '"acute promyelocytic leukemia"', '"chronic lymphocytic leukemia"'],
    "lymphoma": ['"Hodgkin lymphoma"', '"diffuse large B-cell lymphoma"', '"follicular lymphoma"'],
    "breast cancer": ['"triple-negative breast cancer"', '"HER2-positive breast cancer"', '"hormone receptor-positive breast cancer"'],
    "diabetes": ['"type 1 diabetes"', '"type 2 diabetes"'],
}


def _spelling_variants(term):
    """leukemia -> leukemia OR leukaemia; tumor -> tumor OR tumour; anemia -> anemia OR anaemia."""
    out = [term]
    for us, uk in (("emia", "aemia"), ("tumor", "tumour"), ("esophag", "oesophag"), ("hemat", "haemat"), ("edema", "oedema"),
                   ("estrogen", "oestrogen"), ("pediatric", "paediatric"), ("hemo", "haemo")):
        if us in term and uk not in term:
            out.append(term.replace(us, uk))
    return out


def _or(terms):
    terms = [t for t in dict.fromkeys(terms) if t]
    if not terms:
        return ""
    q = lambda t: t if t.startswith('"') or " " not in t else f'"{t}"'  # noqa: E731
    return q(terms[0]) if len(terms) == 1 else "(" + " OR ".join(q(t) for t in terms) + ")"


def build_keyword_queries(question, max_queries=6):
    """Concept queries for a biology / medicine question. Works without an AI key."""
    from evidence_identity import question_frame
    fr = question_frame(question or "")
    q_text = fr.get("question") or question or ""
    disease = fr.get("disease")
    d_terms = [t.strip() for t in re.split(r"\s*/\s*", disease)] if disease else []
    disease_q = _or([v for t in d_terms for v in _spelling_variants(t)]) if disease else ""
    exposures = [e for e in (fr.get("exposures") or []) if not re.fullmatch(r"(?i)can|could|does|do|is|are|what|how|why|which", e)]
    expo_q = " ".join(_or([e]) for e in exposures[:2])
    intent = fr.get("intent") or "general"
    terms = list(INTENT_TERMS.get(intent, []))
    if intent == "cure" and disease and _CANCER.search(disease):
        terms = terms[:4] + CANCER_CURE_TERMS
    if fr.get("organism_restriction") == "animal" and intent == "etiology":
        terms = ["exposure", "induced", "model", "toxicity"]       # 'in rats': no cohort / case-control vocabulary
    intent_q = _or([t.strip('"') for t in terms])
    out = []
    if disease:
        core = " ".join(x for x in (expo_q, disease_q) if x)
        out.append(f"{core} {intent_q}".strip() if intent_q else core)
        # a question asking two things ('can it be cured AND what causes it') gets a query for each
        for extra in [i for i in (fr.get("intents") or []) if i != intent][:1]:
            xt = list(INTENT_TERMS.get(extra, []))
            if xt:
                out.append(f"{core} {_or([t.strip(chr(34)) for t in xt])}")
        if intent_q and exposures:
            out.append(core)                                    # exposure x disease without the intent filter
        if exposures and fr.get("organism_restriction") == "human":
            # 'in humans': trials first (presurgical window trials, phase III adjuvant trials such as null results)
            out.append(f'{core} (randomized OR "phase III" OR presurgical OR "window of opportunity" OR placebo)')
            # trials in people measure the endpoint through its biomarker ('proliferation' -> Ki-67) in short presurgical /
            # neoadjuvant windows: one query names both, so those trials are not missed
            for pattern, sur in ENDPOINT_SURROGATES[:3]:
                if re.search(pattern, q_text, re.I):
                    out.insert(1, f'{core} {_or(sur[:3])} (presurgical OR preoperative OR "window of opportunity" OR neoadjuvant)')
                    break
        # heterogeneous disease: one query per major subtype (cure rates differ completely between ALL, APL, CML ...)
        if fr.get("heterogeneous") and intent in ("cure", "treatment", "prognosis", "etiology"):
            short = _or([t.strip('"') for t in terms[:3]])
            for sub in SUBTYPE_QUERY_NAMES.get(d_terms[0] if d_terms else disease, []):
                if len(out) >= max_queries - 1:
                    break
                out.append(" ".join(x for x in (expo_q, sub, "" if exposures else short) if x))
    else:
        # no disease (plant, microbial, ecology ...): the question's own content words, exposure phrase kept whole
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", q_text) if w.lower() not in _STOP]
        expo_words = {w.lower() for e in exposures for w in e.split()}
        rest = [w for w in words if w.lower() not in expo_words]
        if expo_q or rest:
            out.append(" ".join(x for x in ([expo_q] if expo_q else []) + rest[:4] if x))
    out += endpoint_expansion_queries(q_text, existing=out)
    # broad fallbacks so an unusual question never rests on one exact phrase: every content word unquoted, and
    # the exposure with the question's first outcome word
    from evidence_identity import _Q_STOP as QSTOP
    content = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", q_text) if w.lower() not in QSTOP and w.lower() not in _STOP]
    fallbacks = []
    if content:
        fallbacks.append(" ".join(dict.fromkeys(content))[:120])
    if exposures and not disease:
        expo_words = {w.lower() for e in exposures for w in e.split()}
        other = [w for w in content if w.lower() not in expo_words]
        fallbacks.append(" ".join([exposures[0]] + other[:1]))
        # the broadest query of all - the exposure on its own - is always kept, so a question whose outcome words are
        # unusual ('ashwagandha lower cortisol') still retrieves the literature on the exposure itself
        fallbacks.append(exposures[0])
    out = out[:max_queries] + fallbacks
    seen, final = set(), []
    for q in out:
        k = normalize_query(q)
        if q and k not in seen:
            seen.add(k)
            final.append(q)
    return final[:max_queries + 2]
