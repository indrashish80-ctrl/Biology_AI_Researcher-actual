# BioSynthetica v27: its own laboratory, a readable report, no lost hours

## v27: what changed

| Your request | What v27 does |
|---|---|
| The simulator must run its own simulations | New own laboratory (`ode_engine.py`, `biolab.py`, `ai_lab.py`) with 17 tools: custom model (any ODE system the AI or you write; safe expressions, LSODA solver, stochastic Gillespie option), model library, bacteria vs antibiotic, virus vs antiviral, epidemic, drug exposure (PK), dose-response, DNA analysis, DNA mutation, CRISPR guides, primers, restriction sites, protein analysis, protein mutation, protein 3D structure, pathway, study power. No external simulator is needed (Tellurium stays optional). |
| Check-a-claim was useless | The AI (or, with no key, a rule-based planner) designs experiments for the claim, writes its predictions down **before** running them, runs them, repeats them across plausible biology (Monte Carlo, ±30%), runs controls, and asks a second AI for an independent reading. Verdicts: supported / contradicted / ambiguous / not testable, with the reason. |
| Show genes, proteins, DNA, pathogens, not only graphs | Result views: annotated DNA sequence (mutations highlighted, GC profile), protein sequence (hydropathy, mutations), 3D protein structure (3Dmol), pathogen population panel, pathway diagram, tables and plots. |
| Old results must clear; Enter must work; no need to press Run | Every new run clears the previous result first. Enter submits (Shift+Enter = new line). Lab tools re-run on their own 0.5 s after you stop typing; the custom model after 1.2 s or Ctrl+Enter. Example chips run on click. |
| Place for the NCBI key | NCBI key field on the keys page, sent with each run, used by PubMed, the bio databases, spell-check and dataset discovery (10 req/s instead of 3). "Test keys" checks it. Never stored on the server. |
| Readable report without technical noise | The report now opens with **Short answer → Explanation → Key studies → How certain → Tested in the virtual lab → What it means → Sources** in plain language, author–year citations. Rate-limit messages, paper indexes, audit tables and contexts move to a collapsed **Technical appendix (optional reading)**. Any AI polish is rejected if it adds a number or citation not in the evidence. |
| Hypotheses lacked confidence | Every hypothesis (yours plus up to 5 generated) is tested in the laboratory **before** it is shown, whatever the prompt; each gets a lab verdict and a confidence, and the list is ranked by it. |
| Timer ran while the computer slept | A heartbeat detects sleep; the clock pauses and shows "paused while asleep"; AI cool-downs are reset on wake. While a run is going the app asks the OS to stay awake (Windows / macOS / Linux), released when it ends. |
| Groq rate limits killed the run | Each Groq model has its own cool-down; the router waits (up to 180 s, `BIOSYN_RATE_WAIT_MAX_S`) instead of failing; models available on your key are discovered automatically; requests too large for a model are shrunk; daily-exhausted models are skipped with a clear message; Gemini timeouts/504s move to the next model. Short rate-limit waits no longer count toward the "AI outage" early stop. |

New endpoints: `GET /api/biolab/tools`, `POST /api/biolab/run` (no key), `POST /api/lab/verify` (now the own laboratory).

Tests: `tests/test_v27.py` (32 checks). The full suite has 32 files and 1,354 checks, all passing.

Honest limits: the 3D structures and NCBI lookups need internet; free Groq keys still have daily token caps (adding a Gemini or NVIDIA key makes long runs much safer); without an AI key the rule-based planner covers common claim types (genes/DNA, proteins, pathogens, drugs, epidemics, pathways) but cannot design a custom model; simulations are models, not wet-lab proof.

# Earlier: BioSynthetica v26: nothing partial

## v26.2: faults from your 55-minute report, Auto without restarts, the simulator in research

**Why the run took 55 minutes and said "only 28% of AI calls completed":**
- **Rate limits misread as dead keys.** Groq's rate-limit message ends with "Upgrade ... /settings/billing". The word
  "billing" made every rate limit look like a dead key: a 5-minute lock-out each time, and "0 rate-limited" in the
  report while 138 limits were hit. Rate limits are now recognised first.
- **Limits are per model.** When one model is rate-limited, it rests for the time Groq asks and the next model answers
  at once. Before, the whole key was given up.
- **A used-up daily allowance** (tokens per day) skips that model for the rest of the run instead of being retried on
  every paper. When every model is used up, the report says so.
- **The key check at the start of a run** (one short request per key, a few seconds) switches off a rejected or used-up
  key with the reason.
- **Requests too large for the free tier** are retried with a smaller answer allowance. The deep-reasoning call
  (8,192 answer tokens) never fit Groq's 8,000 tokens-per-minute limit, which is why the summary fell back to a plain
  tally.
- **Timeouts:** every AI request times out after 2 minutes (`BIOSYN_AI_TIMEOUT_S`). The library default could wait
  far longer.
- **Early stop:** when 70% of the last 30 requests fail on every provider, the run stops with the most common error.
  An hour of failures is no longer possible.
- **Counted per request:** completeness now counts requests, not attempts. A request that failed on one provider and
  succeeded on the next is complete.
- **A new report section, "AI provider problems in this run",** lists each error by provider and kind, with a count
  and an example (keys redacted).

**Database lock:**
- When a paper was read again (Auto, repair rounds), its old claims were deleted while contradictions still pointed at
  them. The foreign-key failure skipped the paper and left the database locked, so the next stage failed. Dependent
  contradictions are now deleted first.
- Each reading thread commits its work, and a failed paper rolls back.

**Auto mode no longer restarts.** It is one continuous run. After the papers are read, the evidence is checked. If it
is thin or conflicting, the search widens inside the same run (more abstracts, more full texts, citation chasing). Only
the new papers are read, and the later stages run once. The page shows "Widening the search in this run: ...", and the
decisions appear in the report under "Analysis tiers".

**Report quality (false alarms from your report):**
- **"Misquoted"** now uses the extraction guard's own rule (exact or near-verbatim).
- **"Abstract vs table"** compares a number only within its own clause, ignores words used throughout the paper, and
  requires values within a factor of 2. The 11 false list-item clashes are gone; real ones are still caught.
- **"Model confusion"** no longer flags the report's own dose cautions.
- **Exclusion reasons** are never cut mid-word ("... not of breast Their findings").
- **Bibliography lines** in PDFs ("... stem cell growth, Proc") and general background statements ("X is associated
  with reduced mortality", with no number and no "we") are not counted as this study's results. More
  reference-list headings are recognised and dropped.

**The simulator in research (`sim_verifier.py`):**
1. One AI turns the claim into an experiment on a documented model, and the validator checks it.
2. A second AI reads the claim again on its own. It uses another provider when you have one, and doesn't see the
   first reading.
3. The code, not the AI, runs the model and judges:
   - the base run at the stated conditions;
   - robustness: Monte Carlo over the uncertain biology only, never the dose or the drug constants;
   - every alternative model structure;
   - the quick checks (dose, mechanism signs, power).
4. The verdict is one of: supported (robust) / depends on assumptions / contradicted / ambiguous (both readings shown) /
   not testable.

It runs when the question says "verify", "simulate" or "in silico", or when you tick *Check hypotheses in the simulator*
(you can add your own hypothesis). The report gains a section, "Simulation checks of the hypotheses". On the Simulator
page it is the *Check a claim* tab (`POST /api/lab/verify`).

**Website:**
- **Research page:** four large speed cards replace the hard-to-read drop-down.
- **Simulator page:** three plain-language tabs:
  - **Check a claim:** examples and colour-coded verdicts.
  - **Run a model yourself:** examples, a parameter grid and a reset button.
  - **Plan experiments:** a guided form that replaces the JSON; the JSON is still under Advanced.
- **Virtual lab:** a filter box, plus shortcuts to *Check a claim* and *Plan experiments*.
- **Version check:** a version badge in the header turns red if the page and the server are different versions (an
  old copy still running). Assets are versioned so the browser loads the new ones.
- **Lab runs** left "running" by a stopped server are marked "interrupted".

Tests: `tests/test_v26_1.py` (62 checks). The full suite has 31 files and 1,322 checks.

## v26.1: final error audit and speed

**Speed, measured on a 200,000-record library** (2-core cloud machine; run `python bench_library.py` to measure on
yours):

| Step | v26.0 | v26.1 |
|---|---|---|
| Bulk loading | ~1,300 records/s, slowing as the library grew (duplicate checks scanned the whole table) | ~7,000 records/s, steady |
| Keyword search (BM25, top 100) | 115 ms | 42 ms |
| Semantic search, exact | 407 ms | 26 ms |
| Semantic search, hnswlib | n/a | 1.6 ms |
| Full local search (keyword + semantic + PICO rerank) | 594 ms | 97 ms |
| Same, with 4 planned queries | 1,365 ms | 262 ms |
| Lookup by PMID / DOI | <0.1 ms | <0.1 ms |

What changed:
- **Indexes:** the DOI, PMCID, trial and title duplicate checks have their own indexes, so they no longer scan the
  whole table.
- **Keyword search** ranks inside the full-text index first, then fetches only the top rows.
- **Vectors** are written in place into preallocated shards; a shard is never rewritten.
- **Vectors in RAM:** a float32 copy is kept in memory within `BIOSYN_VECTOR_RAM_MB` (default 1024, about
  650,000 vectors). The row map is saved to disk. The index is loaded once at server start.
- **Planned queries** run side by side on separate read connections.
- **The schema script** runs once per file, not on every open.

The search results are unchanged; only the time differs.

**How this compares with Google:** a Google search is a network round trip, typically a few hundred milliseconds. The
local library answers a full search in about 0.1 s with no network at all, and returns records already parsed,
screened by PICO and versioned, which a web search does not. But a complete research run is dominated by the AI calls
(reading and checking papers), not by search. Those are cut by reading each paper once (`preread`), by the answer
cache and by prompt caching, but they still take seconds per paper with a live provider.

**Errors fixed in the audit:**
- **Loading:** a truncated or garbage PubMed file is rejected whole and retried later, never half-applied. A
  citation with no title gets a placeholder. Odd years and PMIDs are coerced. Bad registry entries are counted, never
  silently skipped.
- **Uploads:** a broken PDF or XML upload gives a clear refusal (400), not a server error, and leaves no file behind.
- **Lab agent and uncertainty:** every malformed goal (type, metric, noise, names, design, observable) is refused
  with reasons before anything runs. The uncertainty endpoint refuses an unknown model or observable at once (400).
- **Adversarial review:** it checks quotes against the paper's whole text (all sections and tables). It checks
  causal wording in the report's own prose, bullets included, but not in tables, quotes or code.

- **Runs that seemed to reset:** a run sometimes went back to the first step. It wasn't crashing; it was
  rerunning on purpose:
  - In **Auto**, Quick moves up to Research and then Deep while more evidence could change the answer.
  - In **Research** and **Deep**, one automatic repair round runs if the report check finds a fixable problem.

  The results page now shows *"Pass 2: moving up from the quick level to the research level, because ..."* or
  *"Pass 2: repair round - ..."* instead of silently restarting the step list. For a single pass, choose **Quick
  scan** (never reruns) or **Research dossier** (at most one repair round).

These fixes are covered by `tests/test_v26_1.py`.

Every roadmap item that v25 had only partly implemented is now complete. Each one has tests in
`tests/test_v26.py` (the full suite: 31 test files, 1,322 checks, no internet or keys needed).

## 1. The local literature library (searched before any external database)

**Where it is stored:** on your computer, in `BioSynthetica/library/`. Move it with `BIOSYN_LIBRARY_DIR=D:\BioLibrary`
(for example to a second drive). Nothing is uploaded anywhere. It is separate from `research_knowledge.db`, which holds
your questions and dossiers, so rebuilding the library never touches a run.

```
library/
  library.db       records, passages, findings, version history, processing log, sync state (SQLite + FTS5)
  vectors/         semantic index (float16 shards; optional hnswlib index for millions of records)
  fulltext/        open-access JATS XML, one copy each, with its licence
  user_files/      your own uploaded papers, by collection
```

**Filling it** (permitted bulk files, not one API call per search):

    python library_loader.py scope --mesh "Breast Neoplasms; Diabetes Mellitus" --keywords "metformin" --year-from 2000
    python library_loader.py download pubmed-baseline --max-files 5     # NCBI baseline files, md5-checked, in scope only
    python library_loader.py download pmc-oa --subset oa_comm --max-files 2
    python library_loader.py trials --query "metformin breast cancer"   # ClinicalTrials.gov registry entries
    python library_loader.py embed --hnsw                               # semantic index (+ approximate index)
    python library_loader.py preread --limit 200 --designs rct,meta     # AI reads each paper ONCE (needs a key)
    python library_loader.py update                                     # the daily job (the app also runs it)
    python library_loader.py status

You can also load files you already downloaded (`pubmed FILE.xml.gz`, `pmc PACKAGE.tar.gz | FOLDER`,
`trials FILE.json`), or use the Library page.

**What is recorded for each paper:**
- persistent identifiers (PMID, DOI, PMCID, NCT);
- three dates kept apart: published, indexed and last revised;
- MeSH terms and publication types;
- retractions, errata and expressions of concern;
- the licence and the reuse allowed under it;
- funding and conflict-of-interest statements;
- the source link and the time it was retrieved.

**How changes are handled:**
- **Versioned:** a change makes a new version, and the old values are kept in its history. Nothing is replaced
  silently.
- **Deletions:** a record PubMed deletes is hidden and logged, never erased.
- **Duplicates:** matched first by identifier, then by title + year + first author.
- **Conflicting sources:** a weaker source never overwrites a stronger one (PubMed's title is not replaced by a
  Crossref spelling). A retraction is never undone by a source that hasn't heard of it.
- **Disclosures and licences** come from the full text itself.

**Local-first search.** The library is searched first, combining keyword search, semantic search and synonyms.
The external databases are skipped only when all three hold:
- the library was updated in the last 8 days;
- at least 20 records match the question's intervention and outcome;
- the question falls inside the library's scope.

Otherwise they run as before, and the local hits join them. `BIOSYN_LOCAL_FIRST=only` works fully offline;
`=off` ignores the library. The decision, its reasons and the library's snapshot date go into the dossier.
A full text already in the library is never downloaded again.

**Registered studies** (ClinicalTrials.gov, and registry records in the library) are listed separately and never
screened as evidence. They feed the missing-evidence analysis (section 3).

**Your own papers** (PDF, JATS/XML, TXT, MD) are added on the Library page or via `POST /api/papers`:
- They are parsed and classified, and their funding and conflict-of-interest statements are read.
- Each is compared with the closest published records.
- They stay in your own collection, which a research run searches only when you tick it.
- *Share* merges a paper into the literature (logged); *delete* removes your copy.

## 2. Faster without loss of accuracy
- **Each paper is read once.** The AI tabulates every result of a library paper a single time. Later questions select
  the relevant findings by code, with no AI call, and they still pass the verbatim-quote guard. The stored
  second-reader agreement is reused too.
- **Better passage selection.** Every results, discussion and table paragraph is scored against the question.
  Whole paragraphs are kept in their original order, the abstract is always included, duplicates are dropped, and the
  limitations paragraph is added as context. v25 read only the first four paragraphs.
- **Prompt caching.** The long, unchanging instructions come first in every prompt and the question and text last,
  so providers that cache a shared prefix can reuse it. Cached tokens are counted in the run metrics.
- **More run metrics:** CPU time per stage, peak memory, disk used by each store, GPU (if any), queue wait and server
  load.
- **Cancel.** A running research job can be stopped (*Cancel run* on the results page, or
  `POST /api/jobs/{id}/cancel`).
- **Escalation only when it can help.** A run moves to a deeper tier only if more evidence could change the answer
  (section 4).

## 3. Accuracy checks completed
| Module | Checks |
|---|---|
| `study_funding.py`, `quality_profile.py` | funding, industry funding, conflicts of interest, study duration; a quality profile per study (risk of bias, design fit, controls, precision, missing data, selective reporting, reproducibility, applicability, funding / COI); relevance shown separately |
| `risk_numbers.py` | relative vs absolute risk, NNT / NNH with intervals, recomputed percentages and changes, "%" vs percentage points, statistically significant but trivial effects; minimal important differences come from `mcid.json` (empty by design, none invented) |
| `causal_checks.py` | per study: how strongly the design supports causation, confounders adjusted for, reverse causation, selection / immortal-time / healthy-user bias, adjustment for mediators; a causal diagram drawn from the studies' own adjustment sets; causal wording the evidence cannot carry |
| `dose_plausibility.py` | dose (the existing human-equivalent-dose and plasma checks) plus route, exposure duration, species, and whether the measured outcome matches the one asked about; unusual results are flagged, not rejected |
| `confidence_calibration.py` | each finding's direction gets a probability calibrated on 444 doctor-labelled sentences never used for tuning, then checked on the separate held-out set |
| `evidence_reviewer.py` | the adversarial review, a separate stage of every dossier (see below) |
| `meta_analysis.subgroup_analysis` | pre-specified subgroups (risk of bias, dose relevance, follow-up length, subgroup-only reports) with a test for differences between them; labelled exploratory |
| `publication_bias.registry_gap` | completed registered trials with no posted or included result, and ongoing ones; reported as an uncertainty, never as proof of negative results |

**The adversarial review** looks for reasons the conclusion could be wrong. It works from the source evidence, not
from the first model's reasoning, and checks for:
- contradicting studies that were not included;
- causal wording the study designs can't support;
- misquoted or misread results;
- arithmetic errors and relative-only effects;
- claims about people that rest only on cell or animal studies;
- overstated certainty;
- limitations the report leaves out.

With an AI key, an AI critic also runs; it sees only the source quotes, and an objection without a real quote is
discarded. The review's findings appear near the top of the report, with the verdict's status after review.

**Measured calibration:**
- **Held-out calibration error:** 0.050 over 300 readings, with a Brier score of 0.138 against 0.152 for a constant
  guess.
- **Fitting on the tuned development set** made it over-confident (it said 0.92 when 0.84 was true), so that set is
  not used for fitting. This is recorded in `gold/calibration.json`.

## 4. Research planner
`research_value.py` asks whether more evidence would change the answer before searching further:
- **What's supported now:** the category, the number of independent human studies and of randomised trials.
- **The gaps:** quality, missing population, contradictory outcomes, lack of evidence or imprecision, each with the
  search that would address it.
- **How robust a pooled effect is:** how many null trials of average size would remove its significance.
- **The value of more research:** high, moderate or low. The analysis tiers escalate only while the value is not low.

## 5. Virtual laboratory
- **Model library: 13 models.**
  - Pharmacokinetics: oral 1- and 2-compartment, and PK/PD.
  - Tumour growth: Gompertz and logistic.
  - Gene regulation: negative autoregulation and the toggle switch.
  - Cell states: proliferating, quiescent and dead.
  - Metabolism: a pathway with end-product feedback.
  - Plus the existing SIR, viral, glucose and enzyme models.
  - Alternative structures of the same system share a `family`.
- **`model_uncertainty.py`:**
  - solver error, checked by re-running with tighter and looser tolerances and a finer output grid;
  - parameter and measurement uncertainty, as a Monte Carlo 95% band;
  - structural uncertainty: the same experiment on every alternative structure, compared on held-out data when you
    supply it;
  - sensitivity to each parameter.

  It always says which kinds of uncertainty were not assessed.
- **`lab_agent.py`, the autonomous loop:**
  - It chooses the next experiment by **expected information gain** between competing hypotheses (parameter sets or
    model structures), updates their probabilities with Bayes' rule, and repeats.
  - It stops when one hypothesis reaches 0.95, when no experiment is informative, when the budget is spent, or when
    the model is invalid for the question.
  - With your measured data, it tells you which real experiment to run next and never invents an outcome.
  - Explore mode runs the base simulation, Morris sensitivity, the hypothesis check, the expected-result test, and a
    threshold search near the decision boundary.
  - Every run is logged, including rejected ones, inside the same resource-capped process.
- **Windows** now has CPU and memory caps too, through a Job Object (no pywin32 needed).

## New API endpoints
- **Literature library:** `/api/literature/status`, `/scope`, `/update`, `/load`, `/search`, `/record/{id}`.
- **Your papers:** `/api/papers` (upload / list / delete / share).
- **Virtual lab:** `/api/lab/agent`, `/api/lab/agent/propose`, `/api/lab/uncertainty`.
- **Calibration:** `/api/calibration`.
- **Jobs:** `/api/jobs/{id}/cancel`.

## Optional installs
- `pip install hnswlib`: an approximate vector index for libraries of more than about a million records.
- `pip install psutil`: memory metrics on Windows.
- Tellurium: the Simulator (as before).

## Honest limits
- **Bulk downloads** (`download pubmed-baseline`, `pmc-oa`) and the ClinicalTrials.gov search need internet access on
  your computer. They are tested here on fixture files, not on the live NCBI servers, which this build environment
  cannot reach.
- **Pre-reading needs an AI key** and costs AI calls once per paper. Prioritise trials and meta-analyses.
- **The Windows Job Object caps** are implemented for 64-bit Windows but were not run on Windows here.
- **Tier 5 (Postgres, vLLM, multi-user) is not done.** At the scale of all of PubMed (about 37 million records), the
  library should move to Postgres + pgvector.
