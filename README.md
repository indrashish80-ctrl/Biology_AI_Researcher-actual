# BioSynthetica — Autonomous Biology Research Engine (web edition)

A web front-end (HTML + CSS + TypeScript, served by FastAPI) on top of the existing
Python research engine. The original architecture is unchanged: `ResearchEngine` →
`SourceManager` → `PaperExtractor` / `ClaimExtractor` → contradiction detection →
GEO loop closure → pathways → hypotheses → audit → report. The new capabilities are
**additive modules** switched on by optional flags.

## v26 (2026-10-28): nothing partial

Full notes: **README_v26.md**. Every roadmap item that v25 had only partly done is complete:
* **Local literature library** in `BioSynthetica/library/` (move it with `BIOSYN_LIBRARY_DIR`). It is filled from
  PubMed baseline / daily files, PMC open-access packages and ClinicalTrials.gov (`python library_loader.py ...`) and
  searched BEFORE any external database. Records are versioned, with published / indexed / revised dates, licences,
  retractions, funding and conflict-of-interest statements. You can add your own papers in a separate collection.
* **Speed:** each paper is read by the AI once and reused for every question; question-aware passage selection;
  prompt caching; job cancellation; CPU / memory / disk / GPU metrics.
* **Accuracy:** a quality profile per study (relevance kept apart), relative vs absolute risk and NNT, causal-inference
  checks with a causal diagram, dose / route / duration plausibility, confidence calibrated on doctor-labelled data
  (held-out calibration error 0.05), and an adversarial review stage in every dossier. Also subgroup analyses,
  registered-but-missing studies, and a "would more research change the answer?" planner.
* **Virtual lab:** 13 models, including gene regulation, cell states and a metabolic pathway; uncertainty at every level
  (solver, parameters, model structure); an autonomous loop that picks experiments by expected information gain; CPU and
  memory caps on Windows too.

## v25 (2026-10-27): measured accuracy, Tiers 1-4

Full notes: **README_v25.md**. In short:
* Fixed 14 errors found in v24.2. These include `p=0·19` being read as p = 0, a Unicode minus turning a
  non-significant interval into a significant one, opposite-direction readings ("did not lower" read as a decrease,
  "reversed the decrease" read as a decrease), a spell-checker that changed real words, duplicate papers from
  different sources, and setup checks that silently never ran.
* Every AI finding must quote the paper verbatim, and code checks the quote and its numbers. A second, independent
  reader re-reads the paper in full runs. Weak findings are labelled and go to the Review page.
* Measured on 323 held-out doctor-annotated sentences (`python gold_bench.py`): direction read correctly in
  **75.5%** (v24.2 reader 58.5%). Gold papers are human RCTs only; see the caveats in README_v25.md.
* New: hybrid BM25 + vector search, PICO query parser, retraction tracking, calibrated verdicts, caches, a
  full-text archive, daily ingestion, quick/research/deep/auto tiers, a dependency graph, RCT meta-analysis with
  publication-bias tests, risk of bias, reproducible dossiers, and a resource-limited virtual lab
  (`pip install -r requirements-lab.txt`).
* Not done: Tier 5 (Postgres, vLLM, multi-user).

## Quick start (runs on your own computer, ₹0)

**Windows:** unzip into a NEW empty folder and double-click `run.bat`.
It installs everything, runs `check_setup.py` (which lists any missing, old or misplaced file
in plain words), then opens http://127.0.0.1:8000 in your browser.


```bash
./run.sh            # macOS / Linux   (Windows: run.bat)
# or manually:
pip install -r requirements.txt
python server.py    # → http://127.0.0.1:8000
```

Open the page, go to **API keys** and paste **your own** free keys (Groq, Gemini and/or NVIDIA NIM),
then **New research** and ask a question. **At least one AI key is required:** without one the engine does not
start a research run and tells you why (there is no key-less "deterministic mode" any more - its reports were too
weak to trust). The **Virtual lab** and the **Library** work without a key.

### The website (separate pages)

| Page | What it is for |
|---|---|
| **Home** | an introduction: what BioSynthetica does, how a run works, your recent runs (runs start on New research) |
| **New research** | the question and your own data file; every run is the full deep evidence dossier (150 abstracts screened, 25 papers read in full by default; adjustable under Advanced options) |
| **Results** | live progress, then the answer; tabs: Summary, Evidence, Hypotheses, Your experiment, Audit, Method & traceability, PRISMA, Screening log, Proteins, Full report |
| **Virtual lab** | **Simple**: eight plain-language tasks (see a protein in 3D, what a DNA change does, can a lab dose be reached in people, how many samples, does an antiviral work, are two groups different, which genes changed, find cell types). **Advanced (technical)**: every tool with every parameter |
| **Library** | offline search of every abstract the engine has screened |
| **History** | runs in this session and every question stored in your knowledge base |
| **API keys** | your keys, a key test, and which AI does which job |

### Search that forgives typos (better than a general search engine for biology)

| Feature | What you see |
|---|---|
| **Did you mean** | while you type, a corrected question appears at the top of the suggestions (and under the box); one click or Enter accepts it |
| **Corrected automatically** | a misspelled question is searched with the corrected words; the results page says *"Showing results for ..."* with a **Search instead for** link that re-runs it exactly as typed |
| **Biology-first ranking** | "metfromin" -> metformin (a general speller says *merbromin*), "nuemonia" -> pneumonia, "fosfate" -> phosphate, "sorisis" -> psoriasis, "colestrol" -> cholesterol, "breastcancer" -> breast cancer, "Alzhiemer's" -> Alzheimer's |
| **Never "fixes" what is right** | gene symbols and abbreviations (TP53, HER2, mTOR, IL-6, MCF-7), words under 4 letters, and rare scientific words (ketamine, organoid, klotho, pyroptosis, sirtuin, CRISPR) are left as typed; a tie between two answers is left as typed |
| **Learns your field** | the vocabulary includes every word of the papers you have already read (refreshed in the background) |
| **Autocomplete** | your earlier questions (typo-tolerant) and the word you are typing; arrow keys + Enter, Esc to close |
| **"/" shortcut** | press `/` on any page to jump to its search box |
| **Library & History** | the library searches the corrected words, highlights them, and offers the original; History has an instant, typo-tolerant filter |

All of this runs on your computer (`spelling.py`, `/api/spell`, `/api/suggest`) - no key, no network, a few milliseconds per keystroke.

### Several keys = faster runs (each AI gets the job it does best)

| Task | Led by | Why |
|---|---|---|
| claim & study-design extraction, screening, agent loop | Groq | fastest for bulk JSON |
| search planning, hypotheses & research gaps | Gemini | long context |
| reasoned synthesis | Groq (gpt-oss-120b) | reasoning model |
| adversarial critique | NVIDIA | large second-opinion model |

With two or three keys the papers are read **in parallel** (one provider per paper) and claim prompts are spread
over all providers (separate rate limits). If a provider fails, the next one takes over, so nothing is lost.

### Fixes from the 2026-10-19 25-paper review ("consistency across sections 4/10")

| Problem | Fix |
|---|---|
| S3 (one neoadjuvant trial) shown as increase, decrease AND no effect, as clinical AND observational | "Changes in Ki67 were significantly HIGHER in the metformin arm ... moving into the low-Ki67 category" is a larger DECREASE (the direction of a change is read from the change, never from 'higher'); a human study has ONE design (a results section of a trial is never 'observational'); a before/after change inside one arm with no test against the control arm is context, not a treatment effect |
| Validation passed while S3 disagreed with itself | new evidence checks: one design per study; no study with increase AND decrease on one endpoint without a subgroup / analysis separating them; the Layer 1 tally must equal a recount of the evidence matrix; every human / animal / cell summary must be backed by the findings it describes |
| "2 of 2 preclinical findings show a reduction" vs "animal studies reported both increased and decreased proliferation" | an uncited study-type summary is proved against the verified findings (citations added) or replaced by a summary generated from them ("Verified animal findings: S23 tumour growth decrease · S21 immune function increase") |
| A cell experiment's "patient survival" compared with trial survival (gap 2, D29-D30, H2) | a cell experiment cannot measure a patient outcome: cell 'survival' is viability; gaps never compare cell results with trial survival / recurrence |
| H3 "metformin increases recurrence" from "incidence of gastrointestinal complications" | side effects / complications are adverse events (tolerability), never recurrence |
| "likelihood of TNBC and decreased ER-positive rates" as incidence; "effector memory T cells" as cognition | subtype distribution; immune function |
| S13 "metformin 2.5 mg/day" (letrozole's dose) | a dose written next to another drug belongs to that drug |
| An empty sentence listed as "withheld" | empty fragments are dropped |
| 150 / 117 / 25 / 272 hard to reconcile | the registry is a recomputable flow: identified - duplicates = unique = screened + not screened; eligible = deep-read + not deep-read (budget) |
| A friend's copy ran without the v11 fixes | the setup check flags outdated semantic_audit.py, reasoning_guards.py, report_validator.py, ai_router.py, evidence_quality.py |

### Fixes from the 2026-10-19 review + "prove statements instead of censoring them"

| Problem | Fix |
|---|---|
| "metformin reduced Ki-67 in HER2-positive women with insulin resistance (HOMA > 2.8)" - the paper lists HOMA > 2.8 and HER2-positive as SEPARATE subgroups | every finding stores its separate subgroups; a sentence or hypothesis that merges them keeps a note ("reported as separate subgroups - not in patients who meet all of them"); hypotheses say "in separately reported subgroups"; the validator fails a merged subgroup |
| "Identified 0 research gaps" next to five key uncertainties | the key uncertainties become gaps linked to the claims that touch them (or are marked "no verified finding addresses it"); the detailed report's count is corrected |
| Findings from two publications of one trial counted as independent in Layer 1 | Layer 1 says "S5, S8 are publications of one trial (T2) - not independent confirmations" |
| S7 "TUNEL higher than at baseline (P<0.0001), although no difference between arms" stored as "apoptosis increased" | a change from baseline in BOTH arms with no between-arm difference is a null treatment effect; the validator fails the old reading |
| "5 claims excluded by the relevance gate" when 1 was | only gate rejections are counted |
| Correct sentences withheld (6 in the report) | the engine now PROVES before it withholds: an uncited sentence is checked against the verified findings and gets its citations when they support it (a wrong "most" becomes the exact count); "did not assess X" is an absence statement; "leading to low certainty" is about the evidence, not biology; "non-diabetic women" is a population, not the topic; "invasive vs premalignant" is a lesion stage; a short-term marker read as patient benefit gets a note. Only statements nothing supports are withheld |
| 8 AI retries in 9 calls | a Groq model that refuses a request size (or is unavailable on the key) is skipped for later requests of that size |
| Screening 77 s for 25 requests | the timing table names the slowest website per stage |
| Abstract-only evidence | in a full run, papers with a free full text are read first within each design tier |
| The report came from an older research_engine.py | the setup check flags an outdated research_engine.py |

### Fixes from the 2026-10-16 reviews (two dossiers + your runtime notes)

| Problem in the report | Fix |
|---|---|
| "significantly lower ... P = .06" counted as a decrease (S8/S12); "mean increase of 4.0% (95% CI -5.6 to 14.4)" counted as "Ki-67 increased" (S11) | the audit reads the quote's own statistics: P ≥ 0.05, or a confidence interval containing no-difference (0 for a difference, 1 for HR / OR / RR), is published as **"no statistically significant difference"** with the numerical trend named - never as a decrease / increase, even when the authors say "significantly"; the validator FAILS a report that still counts one |
| "no effect" for non-significant results | written "no statistically significant difference" everywhere (a non-significant test is not proof of zero effect) |
| Layer 1: "3 show a reduction and 3 show no effect" with only 4 claims listed (and a different 4 in the detailed report) | every counted finding is listed under its own group; both places use the same function |
| γδ T-cell proliferation counted as tumour proliferation "increase" (S21); T-cell cytotoxicity read as "viability decreased" | new endpoint **immune-cell function** (mechanistic context, never the asked tumour endpoint) |
| "outcomes were worse ... for lower (healthier) levels of insulin" → "insulin decreased" (C16-C18); "insulin-dependent pathway" → "glycaemic decreased" (C26) | a baseline level used as a subgroup is an effect modifier, and "X-dependent" is an adjective - neither is an outcome the treatment changed (quarantined with the reason) |
| GLUT transporters → "glucose decreased" (C50); E-/N-cadherin changes → "metastasis increased" (C39); "decreased at 26 min ... increase at 24 h" kept as a decrease (C37) | transporter / EMT-marker expression is expression; a sentence with both directions is unusable (a bug read every mention from the first one) |
| "biopsy + post-treatment surgical specimen" as the intervention; "March 2013 + follow-up to October 2020"; "metformin + show significant" | measurement procedures, dates and verbs are never regimen components |
| 1,700 mg read as 700 mg, 1500 mg as 500 mg; 700 vs 1500 mg called the "same" dose; trial "arm" as an anatomical site; "anti-diabetic drug" making a non-diabetic trial "diabetic" | thousands separators, dose ratio > 1.25 = different, trial arms ignored, drug descriptions are not populations; premalignant vs invasive lesions are compared as a separate dimension |
| Cell-line paper filed as "human observational" (S20); mouse paper whose only findings are cell assays (S21); editorials ("Is it time to ...?", "an evolving mystery") filed as human trials; headline "In vitro 0" while cell findings exist | rule R6 counts a paper where its verified evidence was measured (reason shown); editorials / commentaries are secondary literature; every count comes from the reconciled table |
| Several publications of one trial counted as independent (MA.32 x2; one presurgical trial x5); two analyses of one trial called a "direct contradiction" | trial grouping by registry ID / trial name, identical statistics, or same n + design; "N publications = K independent trials"; same-trial pairs are labelled as such; the hypothesis score counts a trial once and is labelled a ranking, not confidence |
| "deep-read 26 papers" (detailed report) vs 25 (registry); "deep-read" when only abstracts were read | one registry feeds every count; "deep-read (abstracts only - no full text)" |
| Correct sentences withheld ("no deep-read animal studies were available", "an unstratified cohort [S7]", in-vitro limitation sentences); "[note: plant growth yield not measured]", "metastasis invasion not measured" on "invasive disease-free survival", "glycaemic not measured" on "women with HOMA > 2.8" | absence statements are allowed when true; a trial's cohort is not observational evidence; limitation wording is recognised; population descriptors and the verb "yield" are not endpoints |
| "No research gaps" next to a list of open questions | gaps are derived from the evidence when the AI gives none: inconsistent human findings, short-term surrogate only, no dose reported, unread evidence types (coverage, kept apart from biology) |
| "18 of 19 checks passed" read as proof | checks are labelled **technical consistency** vs **evidence vs source**; the summary states that scientific validity is not assessed by software; new critical checks: the verdict must agree with the human findings on the asked endpoint, no non-significant result as an effect, every finding measures what its endpoint says; the uncertainty is split into biological / statistical / coverage / extraction |
| Vitiligo: "increased pigmentation and reduced lesion size" supporting "holistic treatment decreased clinical response" | lesion size, affected area, depigmentation and severity scores are inverse response measures - the mismatch is caught; cure questions lead with "No sustained cure was demonstrated in the papers read (cure = complete response, no recurrence ≥12 months)" |
| AI-proposed assumptions shown as "05" or "t non-diabetic ..."; "Ki-67 → HER2 reported together" as a mechanism arrow | readable parameters ("at alpha = 0.05"); read-out / subtype markers named together are not an arrow, and co-mention is not evidence |

| Runtime | Fix |
|---|---|
| Leaving the page stopped the research until someone came back | the cause was the Windows console window: a click inside it (QuickEdit) or a sleeping PC freezes every program writing to it. Console output now goes through a queue that can never block the research; QuickEdit is switched off; the PC is kept awake while a job runs; the browser remembers the running job after the tab is closed |
| "about a minute" label | removed - the time depends on your keys' rate limits and the literature servers |
| Where do 42 minutes go? | every report ends with **Where the time went**: each stage's start / end, seconds and share, AI calls, AI answers from cache, failures, retries (rate-limited), tokens in / out, web requests, cache hits, web retries - and the slowest stage named |
| Repeated rate-limited requests | any site that answers 429 three times in a row is paused for 5 minutes for the whole run (Retry-After honoured up to 10 s) |
| Work repeated on a re-run | AI answers to reading prompts (claim extraction) are cached for 7 days - a re-run, a repair round or quick → full reuses them; synthesis is always re-thought |
| STRING / 3D structures for an outcome question | run only for mechanism questions (they only produced a "background biology" appendix) |

### Fixes from the 2026-10-15 review ("safeguards still allowing contradictions through")

| Problem | Fix |
|---|---|
| "No human study measured the asked endpoint" while S10 / S12 report human Ki-67 | the sentence is written only when no human finding on the asked endpoint exists; the validator now reads the engine's own answer (verdict, level-by-level answer, discordance) as well as the AI text, and a denial of evidence the table holds is a critical failure |
| "metformin significantly blunted the Ki-67 rise observed in the placebo arm" stored as "Ki-67 increased" | "blunted / prevented / reduced the rise", "a smaller increase than placebo" = decrease relative to the comparator; the result keeps its comparator ("relative to placebo - a smaller rise, not a fall from baseline"); a stored direction that contradicts its own quote fails validation |
| "All critical pre-publication checks passed" | every check is listed as passed / FAILED / noted / not applicable with its detail (Audit layer and website); the summary gives the tally; audit flags are explained as rejected claims |
| H2 "does not change survival" falsified by "no difference in survival" | every hypothesis (AI-written ones too) is checked: a no-effect hypothesis is falsified only by a significant change; "no difference" falsifies an effect only in a study powered for a pre-specified minimal important effect |
| 25 deep-read, 24 in the matrix, S1-S24, 121 eligible | one study registry (screened -> eligible -> deep-read -> excluded with the reason -> included = primary + reviews) feeds the scope line, the reliability block and the validator; a paper is never dropped silently; an unexplained difference fails validation |
| Semantic Scholar HTTP 429 again and again | after the first refusal the source is paused for the run (one retry, not three) |
| 148 s warm-up | the embedding model, once downloaded into the program folder, is loaded from disk without an online check |
| No per-stage evidence of where time goes | every stage logs its time, AI calls (and failures), web requests, cache hits and retries |
| AI hypothesis step answering empty three times | at most two attempts, then the evidence-derived hypotheses |
| 3D structures for a clinical-outcome question | 3D models run only for mechanism questions (naming a protein, or asking how / through what); the log says when they are skipped |

### Speed: Quick scan, Full dossier and saved results

| Choice on the research page | What it does | Time |
|---|---|---|
| **Quick scan** (default) | abstracts of the top 8 of 40 screened papers; fast sources (Europe PMC, PubMed, OpenAlex); no full texts, no repair round, no 3D structures, no citation chasing; evidence-derived hypotheses; the report is marked "Quick scan" | faster than a full dossier; the time depends on your API keys' rate limits and the literature servers (each stage's time is shown in the report) |
| **Full dossier** | full texts of up to 25 papers, citation chasing, 3D structures (mechanism questions only), at most one automatic repair round | slower than a quick scan; the report's 'Where the time went' table shows each stage's time |
| **Reuse saved result** (on by default) | the same question with the same settings opens the saved result instantly - no search, no AI calls (kept 7 days; untick to run again) | under 1 s |

Every run now prints and stores how long each stage took (`[Time] screening: 6.2 s ...`, `timings` in the dossier).
Faster in every mode: a paper's abstract and results paragraphs go to the AI in 1-2 calls instead of one call per
paragraph; with one AI key 3 papers are read side by side; entity look-ups run in parallel; one repair round at most.
Exact times depend on your keys' rate limits and on the literature servers.

### What every report now answers first

Every report (website dossier and the agent's report) opens with the **answer to the question, level by level** -
people, animals, cells - from verified findings only. A null clinical trial is never hidden behind positive cell or
mouse results ("Preclinical evidence is predominantly supportive, whereas the strongest clinical evidence does not
show the effect"); findings that measure a different endpoint are named as context, not used as the answer; a run
where nothing could be verified says so and why, and is never an empty report. Hypotheses are headed **proposed for
testing (not conclusions)**, with their verified support stated.

Fixed from the sample reports: a mistyped question word ("Csn leukoderma be cured") is no longer read as a drug;
each claim's type and species come from its own experiment (a cell result in a paper that also has mice is "Cell
culture, human cells (in vitro)"; a tumour grown in mice from human cells is "mouse xenograft (human tumour cells)");
"RAGE translocation" is protein localization, not genotoxicity; a wound-healing (scratch) assay is cell migration;
"metformin disrupts HMGB1-induced ..." is metformin's result with HMGB1 as the stimulus.

### Any biology question, not only medicine

The question gets a **domain** (human medicine, plant, microbe, animal, general biology). Clinical screening rules
(treatment / risk-factor vocabulary) apply to medical questions only; normal-biology words (gravitropism, apoptosis,
symbiosis) are not diseases; "population", "donors", "individuals" no longer make a plant or bacterial paper human,
and the organism the AI read in the paper corrects a weak regex call; a two-part question ("can it be cured and what
causes it") is searched for both parts; the best-ranked eligible records are never dropped by a cut-off relative to
one near-perfect hit.

### The agent spends a budget by objective, not a step count

`autonomous_agent.py`: tokens, tool calls, time, papers read and reasoning turns each have an allowance; every action
is charged what it really uses; the agent sees what is left and decides how to spend it; the run ends when the
objective is met (at least 10 papers read, competing explanations checked, every proof pillar backed by verified
claims) or when one resource is spent - and says which. A model that repeats itself is moved on automatically.

### 3D structures (`structure.py`)

Experimental structures from the RCSB PDB, predicted models from the AlphaFold database (by gene name or UniProt ID),
new predictions from a sequence (ESMFold, up to 400 residues) or your own PDB / mmCIF file. You get: a 3D view
(3Dmol.js, coloured by model confidence), chains and sequence, secondary structure, size, model confidence (pLDDT)
or experimental resolution, **pockets** where a molecule could fit (buriedness grid search; checked on deposited
drug complexes - the bound ligand's site is the top pocket), **contacts of bound ligands**, **flexibility** (Gaussian
network model, checked against crystal B-factors) and the structural context of any residue (for a mutation).
Research runs fetch AlphaFold models of the proteins named in the evidence and show them on the Proteins tab.
Not computed (and said so): binding free energy (dG), docking poses, all-atom molecular dynamics - these need
force-field software (AutoDock Vina, GROMACS, AMBER) that is not shipped.

### Gene-expression and single-cell data (`omics.py`)

Upload a count or expression table: median-of-ratios normalisation (DESeq2 method), low-count filtering, a linear
model per gene with **empirical-Bayes moderated t** (limma method), **Benjamini-Hochberg FDR**, **batch modelled in
the test**, PCA with a warning when batch drives the variation, volcano plot. Single-cell counts: normalisation,
highly variable genes, PCA above the random-noise limit, Ward clustering, marker genes - and "no population
structure" when there is none. GEO re-analysis in research runs now decides by FDR across the whole array.
Not done: FASTQ alignment (STAR / HISAT2) and negative-binomial GLMs - start from a matrix.

### Virtual lab (in silico, runs on your computer)

`virtual_lab.py` - DNA analysis (GC %, ORFs in 6 frames, translation), point-mutation effects, sequence alignment,
protein properties (mass, pI, charge, hydropathy, transmembrane segments), viral dynamics with an antiviral,
SIR epidemics, dose-response and IC50 fitting, one-compartment pharmacokinetics, exposure vs potency, tumour /
cell growth with drug kill, signed pathway perturbation (A -> B -| C), power and two-group statistics. The engine
also runs every hypothesis through it (are the mechanism's signs consistent with the claimed effect? is the
proposed dose anywhere near human exposure? is the proposed experiment powered?). Every result is labelled
"in silico - not experimental evidence".

### Your own data

Upload a CSV / TSV / Excel file (one row per sample) on **New research**. Bad files are rejected at once with the
reason (wrong type, empty, one column, missing / non-numeric outcome, fewer than two groups). Your experiment is
analysed in its own section (**Your experiment** tab and report section): its statistics, how it compares with the
published findings on the same agent and endpoint, and the hypothesis test. It never changes the evidence matrix,
the bottom line or the certainty rating, because it is not peer-reviewed literature.

### Report healer (`auto_healer.py`)

```bash
python auto_healer.py --report runs/<job>/RESEARCH_REPORT.md          # check a report
python auto_healer.py --report runs/<job>/ --fix                      # + repair the knowledge base (backup first)
python auto_healer.py --report RESEARCH_REPORT.md --fix --rerun       # + re-run the question with a repair plan
```

It finds the known failures (finding-id citations withheld, question words used as a drug, over-censoring,
duplicated gaps, key-less reports, cell counts shown as n, human studies shown as animal, mixed old/new files,
and every check of the built-in run auditor), lists every withheld sentence for review, and repairs DATA only:
stale claims are re-extracted, verdicts and classifications re-checked, gaps rebuilt. It never edits program code
(a program that rewrites itself from its own reports cannot check its own edits). `--purge-db` deletes the whole
knowledge base after a backup (your original option).

### Report censoring fixed

Correct sentences were being withheld because the AI cited `(S8)` or `[S1.E1.F1]` instead of `[S8]`, because
"animal models using human breast cancer cells" was read as a claim about people, because honest limitation
sentences ("current evidence does not demonstrate...") were treated as uncited conclusions, and because a sentence
listing several endpoints was withheld when one of them was not measured. All four are fixed (23 of 24 withheld
sentences in your six AI reports would now be published, the 24th was cut off in the report); real leaps (a cell
result stated as a human mechanism, one endpoint inferred from another) are still withheld.

The first run downloads the local embedding model `BAAI/bge-small-en-v1.5`
(~130 MB, once). After that, embeddings are computed on your CPU.

## API keys — who pays, who sees them

| | Old Streamlit `app.py` | New web app |
|---|---|---|
| Sidebar pre-filled from server `.env` | **Yes — every visitor got the owner's keys** | No |
| Keys copied into `os.environ` | Yes (shared by all visitors) | Never |
| Where keys live | Server process | User's browser (session storage; local storage only if "Remember" is ticked) |
| How they reach the engine | Global env vars | `AIRouter(keys=…)` built per request, discarded after the job |
| Logs / errors | Unredacted | Keys redacted from every log line and error |

`app.py` has also been patched so it no longer pre-fills or exports keys.
`.env` keys are used **only** by the CLI tools, or by the web app if you deliberately set
`ALLOW_SERVER_KEYS=1` on a private single-user machine.

## Evidence-audit layer (how BioSynthetica keeps the science honest)

Pipeline: search → de-duplication → **study classifier** → evidence extraction → **semantic auditor**
→ VERIFIED / QUARANTINE → **context comparator** → contradiction engine → hypothesis generator →
falsification → final synthesis. Every conclusion traces back: conclusion → claim → experiment → paper → exact passage.

| # | Safeguard | Where |
|---|---|---|
| 2 | Every paper gets ONE frozen evidence type (human clinical, human observational, animal, organoid/ex vivo, cell culture, in silico, review/meta-analysis; + plant, microbial). Decided once by an ordered rule list and stored; nothing downstream can re-classify it. PDX = animal, TCGA = human observational, reviews = review. | `study_classifier.py` |
| 3 | Evidence levels L1 direct human … L6 computational, L7 AI hypothesis. Sentences that turn cell/animal data into a human conclusion, or one endpoint into another (survival → proliferation), are flagged in place. | `study_classifier.py`, `reasoning_guards.py` |
| 4 | Contradictions compared on species, model, cell line, subtype, metabolic state, dose, duration, comparator, co-treatment, endpoint, conditions → TRUE_CONTRADICTION / CONTEXT_DIFFERENCE / UNRESOLVED_DISAGREEMENT, each with the comparison table and the reason. | `contradiction_detection.py` |
| 5 | Canonical endpoint + direction: "increased growth inhibition" = growth ↓; "increased cytotoxicity" = viability ↓; apoptosis ≠ viability. | `semantic_audit.py`, `claim_extraction.py` |
| 6 | One gate: `db.verified_claims()` is the only claim accessor for the matrix, contradictions, pathways, hypotheses, critic and summary. Quarantined claims have no back door. | `database.py` |
| 7 | Semantic grounding: the quote must support intervention, population/model, endpoint, direction, magnitude and significance, and be this study's result (not cited prior work). | `semantic_audit.py` |
| 8 | PRISMA ledger: every record has an ID (R0001…) and a tracked path; counts are derived from the ledger and 14 identities are checked. Failure → "PRISMA CONSISTENCY ERROR" instead of a diagram. | `prisma_screening.py` |
| 9 | paper → experiment → result → claim: re-worded claims about one result count as ONE independent finding. | `semantic_audit.py`, `evidence_matrix.py` |
| 10 | Six separate ratings: paper quality, evidence relevance, claim grounding, experimental quality, risk of bias, translational relevance. | `evidence_quality.py` |
| 11 | Dose/PK: concentrations vs human plasma exposure; animal mg/kg → human-equivalent dose (FDA BSA). Physiological / borderline / supra-physiological — "biological activity at this dose; relevance to human exposure uncertain", never "invalid". | `evidence_quality.py` |
| 12 | Mechanism steps labelled M1 candidate (STRING) → M2 experimental association → M3 causal (perturbation) in a model → M4 causal in the target organism; L7 AI hypothesis. | `reasoning_guards.py` |
| 13 | Overlapping hypotheses merged (H1 — insulin/IGF-1 mechanism …) and ranked by verified support weighted by evidence level. | `reasoning_guards.py`, `research_engine.py` |
| 14 | Doses, n, thresholds, effect sizes, durations, power without a verified source are tagged **AI-PROPOSED EXPERIMENTAL PARAMETER**. | `reasoning_guards.py`, `database.py` |
| 15 | The final synthesis sees only verified claims, frozen classifications, validated comparisons and verified numbers; a traceability appendix links every cited statement to the exact source passage. | `deep_reasoner.py`, `enhanced_report.py` |

### Report-reliability layer ("a polished report must never outrun its evidence")

| Problem | What the engine now does | Where |
|---|---|---|
| Wrong buckets | case reports / series and experimentally induced disease models are their own types; farm & veterinary species (buffalo, goat, horse …) are animal evidence; every model gets an explicit translational distance to the question | `study_classifier.py` |
| Combination treatments | "A, B and C were given; recovery observed" is evidence for the regimen only - claims about one component are rejected | `clinical_context.py`, `semantic_audit.py` |
| Recovery ≠ cure | outcome ladder improvement < partial response < recovery < complete response < remission < sustained remission < cure; "cure" questions get an operational definition (sustained, ≥ 12 months follow-up); overstated outcomes are rejected | `clinical_context.py` |
| Observation ≠ causation | inference type (randomised / controlled / association / observed after treatment / case); causal wording without a controlled design is downgraded to "causal claim unsupported" | `clinical_context.py`, `semantic_audit.py` |
| Absence of evidence | "cannot be cured / does not work" is only allowed when a direct study tested it and found no effect; otherwise rewritten to "no evidence demonstrating … was identified" | `reasoning_guards.py` |
| Not found ≠ not searched | per evidence type: deep-read / found but not deep-read (budget) / inaccessible / excluded / not screened / not found in the searched sources | `report_validator.py` |
| Budget bias | "X of Y eligible records were deep-read" at the top of every report; certainty capped | `report_validator.py` |
| Hard rules | human claims without human evidence, endpoint leaps, cure inflation and uncited conclusions are withheld or rewritten (listed in the report) | `reasoning_guards.py` |
| Abstract-only | claims tagged "abstract-only verified" vs "full-text verified" | `semantic_audit.py` |
| Grounding ≠ correctness | 8-part grounding profile (relevance, intervention, population, outcome, direction, causal, quantitative, context) + granular claim statuses | `semantic_audit.py` |
| False QC | statistics / power / dose / n / comparator / follow-up are reported as "not assessable" or "unknown" unless the data exists | `evidence_quality.py` |
| Quality dimensions | grounding, study quality, causal strength, relevance, replication (same group / independent / other species), experimental quality, risk of bias, translational relevance | `evidence_quality.py` |
| Disease definition | what each study actually studied (cause/induction, subtype, site, severity); different causes / induced models are never pooled | `clinical_context.py` |
| Hypotheses & gaps | each lists the exact claim IDs it came from; mechanism observed / inferred / proposed; AI numbers shown apart; regimen-level gap rewriting; PK gaps only for systemic drugs | `reasoning_guards.py`, `research_engine.py` |
| STRING | queried only for genes named in the question or verified claims; unrelated networks go to an appendix; interaction ≠ disease association ≠ mechanism ≠ target | `research_engine.py`, `enhanced_report.py` |
| Certainty | "BioSynthetica certainty (internal framework, GRADE-inspired - not a formal GRADE assessment)", capped by the evidence | `report_validator.py` |
| Pre-publication validation | PRISMA arithmetic, citations, certainty cap, traceability, labelling checked before the report is written | `report_validator.py` |

### Evidence identity & relevance gate ("evidence about a different biological system must never look like evidence for the question")

Every claim carries a full identity - source paper, original experiment (this study vs reported by a review), evidence type, species, disease, disease subtype, cell/tissue, intervention/exposure, comparator, dose, duration, outcome, direction, statistics, causal language and the exact source passage - and a **hard rule**: if its disease, model, exposure or outcome does not match the question, it cannot enter the evidence pool (`evidence_identity.py`).

| Problem (real "causes of leukimia" dossier) | What the engine now does |
|---|---|
| Misspelled question | corrected before searching ("leukimia" -> "leukemia"), shown "as typed" in the dossier and website |
| Off-question papers in the buckets (MDS, arterial plaques, myocardial ischaemia, RASA3 reticulocytes) | excluded as off-question with the reason; never counted in any bucket |
| Benzene-worker study called breast-cancer cell culture; MSC co-culture called in silico | exposed-worker / patient-sample studies = human observational; co-cultures = cell culture |
| "What Are" as the intervention | question words are never a subject; such claims are rejected; the fallback extractor reads the subject from the sentence |
| 1.00 grounding for a quote about another disease | grounding (text match) is shown separately from question match; the relevance gate decides |
| Mechanism presented as cause (ROS -> apoptosis) | evidence role: disease risk / intermediate / mechanistic / context; only disease-risk findings speak to causes |
| Association presented as causation; incomplete chains | association wording enforced; "exposure -> intermediate -> disease" chains need every link |
| Summary overstating causes; characterising studies with no verified finding | hard rules rewrite or withhold; causes questions always get the deterministic bottom line |
| Contradictions citing claims not in the report; cross-context "genuine" contradictions | contradictions use only this run's verified claim IDs (C#); in vitro vs in vivo = context difference |
| PRISMA arithmetic (235 + 30 - 10 != 225) | recomputed before publishing; an inconsistent flow is refused |
| Review-reported claims counted as verified experiments | primary-study claims vs claims reported by reviews kept apart |
| "No human epidemiological studies" while showing workers / cohorts | human clinical vs human observational/epidemiological split; contradictory sentences rewritten |
| Hypotheses labelled empirically observed; leukemia treated as one disease | untraced entities -> "AI-generated mechanistic hypothesis informed by literature"; AML / ALL / CML / CLL subtype coverage tracked |
| Gaps assuming untested links or built on vague observations | gap chain warnings; weakly anchored gaps labelled |
| False blanket QC sentence | replaced by what could actually be checked |

### Code-review fixes (4-point review)

| Bug | Fix |
|---|---|
| "Metformin Reduce Breast" as the drug name (first-3-words slicing) | the subject is the exposure named in the question, with multi-word names kept whole ("vitamin D", "drought stress"); no subject is invented for causes questions |
| "cell viability / apoptosis" lumped together | viability and apoptosis are separate outcome families with their own directions |
| Contradiction inflation (patients vs cells) | model-tier gate: results from different experimental systems are tagged "different systems - not a contradiction" and are not counted as disagreements in any headline |
| MCF-7 study misclassified as animal | the old keyword-count scorer is gone; sentences about previous or future work ("previously shown in mice") no longer decide a study's design |
| STRING presented as a causal highway | "Hypothetical mechanistic pathway" with an explicit caution that protein interactions do not prove treatment-induced causality |
| (found while re-testing) sentence about another disease citing an unrelated study | withheld as a citation mismatch |
| (bug hunt) blank / punctuation-only questions started a research run | refused with a plain message |
| (bug hunt) HTML in a question could add images to the Report tab | images / embeds are never rendered from report text; reports escape the question |
| (bug hunt) interactive helper scripts crashed on `--help` or closed input | they print usage / exit cleanly |
| (bug hunt) duplicate entries in the setup checker skipped a version check | duplicates removed |

### Fixes from a real no-key run ("Does metformin reduce breast cancer proliferation in humans?")

| Problem in the dossier | Fix |
|---|---|
| "Metformin Reduce Breast" everywhere; stored claims from older versions reused | claims are re-extracted when made by an older extractor, in another mode, or for another question |
| Sentences that never mention metformin became metformin claims ("EZH2 overexpression promoted ...", "This is followed by enhanced proliferation") | the subject must be named in the sentence (or the one before it); the audit rejects a quote that attributes the result to another agent |
| "MET" (= metformin) not recognised; section headings used as quotes; p-value sentences skipped | paper-defined abbreviations are resolved; headings are never sentences; "(P < 0.05)" sentences are read |
| Reviews counted as human studies; an 89-patient study and cell studies counted as animal | publisher article-type, full text without Methods/Results and review-only abstracts mark reviews; retrospective cohorts and quote-level animal treatment are recognised |
| n = 40994 / 450000 / 231 (cell counts, cell-line names) | n comes only from study units (patients, mice, replicates) |
| Fake regimens ("0, 5, 10 and 25 mM metformin", "metformin in vitro, and proliferation ...") | a regimen needs two different agents |
| "HMGB1-induced", "Type 2", "89 breast cancer" as disease definitions | inductions need a disease/model; subtypes must qualify the disease; only real qualifiers kept |
| "[ 8 ]" citations, "not all studies have found", self-contradictory sentences, "05), and ..." fragments | treated as prior work, conflicting evidence, inconsistent and broken respectively - never verified |
| Contradictions between different genes ("cyclin D1" vs "MIP1a"); "0 mM", "Figure 3G", "IFN-γ + NKT" as dose / co-treatment | generic expression needs the same gene; zero doses, figure panels and marker notation ignored |
| Bottom line pooled apoptosis UP with proliferation DOWN | only findings on the question's endpoint, per system; uncited AI bottom lines replaced |
| STRING network about the MET receptor and "SET" | abbreviations defined for non-genes, English-word symbols and review-only text are excluded |
| Question echo / withheld marker in the no-key summary; raw JSON experiments; template "Metformin reduce breast regulates ..." | summary counts per endpoint and system; experiments shown as text; templates use the real exposure and endpoint |

### Search & retrieval fixes + auto-healer (from a real "Can leukimia be cured" run rated 0/10 for retrieval)

| Problem | Fix |
|---|---|
| The typed sentence was sent to the databases (PubMed returns nothing for "Can leukemia be cured?") | concept queries built from the question - corrected spelling, disease + spelling variants (leukemia OR leukaemia), what is asked (cure / remission / survival; causes / risk; effect / biomarkers), one query per disease subtype (ALL, AML, CML, APL, CLL) - with or without an AI key (`research_planner.build_keyword_queries`) |
| 0 of 50 records excluded at screening; an Indonesian image-classification paper was deep-read | eligibility rules with a stated reason: English text, the question's disease named, no image/data-classification method papers unless the question is about diagnosis or AI, a treatment-outcome concept for cure/treatment questions and a risk concept for causes questions (`prisma_screening.eligibility`) |
| Only 2 of 10 papers read (13 had no text) | the deep-read keeps going down the eligible list until K papers with text are read |
| "Can" became a drug: "Can regulates biological response ..." | no hypothesis without a verified finding; without an AI key each hypothesis is derived from one verified claim (C#): what was observed, in which system and subtype, and whether it replicates one step closer to the target organism - no invented mechanism; earlier runs' hypotheses are retired |
| Cure studies not understood | treated-patient cohorts recognised; "was treated with X ... were cured" links the outcome to X; "10-year event-free survival" is 10 years of follow-up; the cure bottom line names the subtype |

**Auto-healer (`report_auditor.py`).** After every run the engine checks its own report for the failure patterns seen in real reports (question words as subjects, too few papers read, nothing answering the question, off-question / non-English papers included, implausible n, template hypotheses, broken report text, inconsistent PRISMA). Repairable problems trigger an automatic repair round (broader concept queries, larger screening / deep-read budget, forced re-extraction; at most 2 rounds, the best round is kept). Everything it did is shown in "Automatic quality check & repairs"; anything it could not repair is written to `BUG_REPORT.md` (downloadable from the website). It repairs the RUN - it never edits its own program code, because a program that rewrites itself from its reports cannot check its own edits.

### Fixes from a real AI-key run ("Does metformin reduce breast cancer proliferation in humans?")

| Problem in the report | Fix |
|---|---|
| "insulin and IGF-1 remain key drivers of ... anti-apoptotic signalling" became "Metformin: apoptosis decreased" | "X remain(s) key drivers / mediators / promoters of ..." and "X drives / promotes ..." make X the subject, so the claim fails the intervention check |
| "anti-hyperglycemic actions" became "glycemic increased" | anti-hyperglycemic / antidiabetic / glucose-lowering = glucose DOWN |
| A tamoxifen "P1 trial" sentence was credited to metformin | "As in the P1 trial", "as reported in the X trial", "similar to the X trial" = cited prior work |
| "improved decreased" | a describing word (improved, reduced, anti-...) is never accepted as an endpoint |
| Presurgical trials in women classified as animal; C22 quarantined against its own mislabel | a trial with people counts as human even if mice are mentioned in passing; background sentences ("has antiproliferative effects in mice") never decide the system; a placebo / randomised comparison in women next to the result is human evidence whatever the paper label |
| Null results purged ("no significant effect" failed direction / significance) | "did not significantly change" is a verified no_effect finding; null findings are shown in the bottom line and weighed against positive ones |
| "0.5, 25 mM", "time-, dose-dependent manners", "0-20 mmol/l from 24 to 96 hours" treated as drug combinations | doses, units and time words are never regimen components; two different agents are needed |
| RFS / OS / CBC not matched to the spelled-out endpoints; "longer survival" had no direction | standard clinical abbreviations are expanded; longer / shorter are directions |
| Contradictions and H1 built on a review's sentence | results a review only relays are excluded from contradictions and hypotheses |
| Research gaps repeated three times | gaps are de-duplicated and recomputed (not accumulated) on every run and repair round |
| Ki-67 fell only in women with HOMA > 2.8; the phase III adjuvant trial was null | each finding carries its subgroup ("only in women with HOMA > 2.8", "overall population"); the bottom line says when an effect is confined to a subgroup; an observational benefit contradicted by a null randomised trial becomes a "discrepancy" gap, not "no RCT evidence"; "in humans" questions add a trial query (randomized / phase III / presurgical / window of opportunity) |
| Typos and unusual questions | larger spelling vocabulary (organs, diseases, drugs, endpoints, question verbs - "reduse" -> "reduce", while "induce" stays "induce"), then NCBI's PubMed spell checker (ESpell) for any word still unknown (skipped silently offline); verbs ("lower", "slow") are no longer glued into the searched name; broad fallback queries are always added |
| Fewer papers read for unusual questions | screening scores are relative to the best record (the weakest record of a small pool is no longer forced to 0 and excluded); the question's own drug counts as a treatment concept; when fewer records than the deep-read budget are eligible, screening continues down the ranking and then re-admits records excluded only for a missing keyword; the auto-healer re-runs whenever deep-read budget was left unused while records remained unread |

### Fixes from the full v6 dossier review (2026-10-13, 24 studies, AI synthesis)

| Problem in the report | Fix |
|---|---|
| "significantly prolonged latency in tumor growth" became "tumour growth increased" and created two ghost disagreements | tumour latency / onset / appearance belong to tumour growth with inverse polarity (prolonged latency = growth DOWN); "delayed the appearance and growth of carcinoma" = tumour growth DOWN |
| "less pronounced or not statistically significant" = survival decreased; "altered with increasing concentrations" = increase; "studies assessing cell viability ... demonstrated enhanced efficacy" = viability increased | "not statistically significant" is no effect; dose phrases ("increasing concentrations") are not directions; an endpoint that is only the object of "assessing / measuring" has no direction |
| Combination results (ketogenic diet + metformin, metformin + palbociclib, tamoxifen-induced inhibition enhanced by metformin, co-treatment with H2O2 / flavone) counted as metformin's effect | combinations are detected however they are worded; a claim crediting the agent alone fails attribution, and a claim about the whole regimen gets the role "combination" (context, never an answer) when the question asks about one agent |
| Cell-only papers classified Animal (fetal bovine serum, mouse antibodies, "murine 4T1 cells"); cell quotes in mouse papers called animal; trial tissue called observational | reagents named after animals are ignored; an animal word without an in-vivo cue (injected, bearing, mg/kg, xenograft ...) is the cells' origin; incubation / mM / "cells" with no animal word is a cell finding; tissue from a trial's participants is the trial's evidence (classifier 1.9) |
| "no direct human clinical evidence demonstrates that metformin reduces proliferation" replaced by "the sample does contain human evidence" (listing a review and an animal paper); VALIDATION FAILED on the guard's own marker | absence statements are endpoint-specific (contradicted only by a study of that type that measured that endpoint, in that clause); reviews never count; the validator ignores withheld / rewritten markers; withheld drafts go back to the AI once with the reason (reflection round) |
| Scope notes printed at the start of the NEXT sentence; spurious "[note: clinical response not measured]"; "consistent anti-growth effects" | notes stay inside their sentence; "improvement in survival" is survival; contrast clauses without a citation are not checked against the other clause; "consistent ... effects" is softened |
| Replication tests shown as "Mechanistic hypotheses proposed by the AI"; mediator "oxidative"; null falsified by "no difference"; "expression" pooled across METTL3 / p53 / miRNA | hypothesis kind is read from the hypothesis itself; routes are noun phrases ("oxidative stress", "cyclooxygenase-2"); a null result is falsified by "a significant change"; expression findings carry their gene / protein and are never pooled, compared or replicated across targets; a bare "expression" / "activation" with no target is not a measurable endpoint |
| "MET" (metformin) sent to STRING and modelled in 3D | standard drug abbreviations of the run's agents are never genes |
| "active comparator" for "compared to the control group"; "oral" as an anatomical site; "mild" severity from "mild hypoglycaemia"; "tamoxifen-induced" as a disease induction; "metforminformin"; "Since" / "combined" as agents; 'HMGB1-induced' as causal wording; diabetic subgroup in a cell paper | each fixed at its source (comparator, route, severity, induction, whole-word abbreviation replacement, agent cleaning, hyphenated labels, comorbidity only for people) |
| 14 eligible human studies never deep-read; Ki-67 window-of-opportunity trials not searched | for questions about people the deep-read budget goes to trials, then observational studies, then animal, then cell studies (reviews last); a "biomarker x presurgical / window-of-opportunity / neoadjuvant" query is added |
| Other details | review claims always headlined "A review reports"; Layer 1 lists only clinical outcomes as "other clinical outcomes"; "N of M show a reduction" lists the N; audit flags name the claim and paper (not "paper:1"); molar values in animal papers may be plasma levels; implausible animal n discarded; Graph 1 groups references by system and names targets; traceability shows each finding once, with the findings each sentence is about |

Re-asking the same question reuses its record, but each run's evidence is scoped to that run's included papers.
The website shows all of this in the **Evidence matrix**, **Audit** (contradictions + quarantined claims),
**Hypotheses**, **PRISMA** and **Summary → Traceability** views.

## Works for ALL of biology

Evidence is sorted by organism into **Human · Animal (lab, farm and wild) · In vitro · Plant · Microbial · Other**.
The engine detects what the question is about (human health, plants, microbes, animals or general biology):

* certainty is capped only when evidence from **that** organism group is missing
  (a rice question is not penalised for having no human trials);
* the STRING protein network uses the organism named in the question (rice, maize, Arabidopsis,
  E. coli, B. subtilis, P. aeruginosa, M. tuberculosis, yeast, fly, worm, zebrafish, chicken, ...).

## Speed

* One deep protocol: 150 abstracts screened and 25 papers read in full by default (up to 300 / 40 under Advanced
  options). Several API keys read papers in parallel; no run-time estimates are shown because they depend on the
  providers' rate limits.
* `run.bat` / `run.sh` reinstall packages only when `requirements.txt` changes, so later starts take seconds.
* The server loads the engine and the embedding model in the background as soon as it starts.
* Biggest real-world speed-ups: a **Groq** key (fastest AI) and a free **Semantic Scholar** key.

## New features

| Feature | Module | Cost |
|---|---|---|
| Two more free literature sources: **OpenAlex** (250M works, keyword + OpenAlex semantic search) and **Semantic Scholar** (relevance search, citation counts, TL;DR summaries) | `retrieval/openalex.py`, `retrieval/semantic_scholar.py` | ₹0 (optional free keys) |
| **Citation chasing**: references of, and papers citing, the 3 best records are pulled from the Semantic Scholar citation graph and re-screened | `prisma_screening.py` | ₹0 |
| Open-access PDF links supplied by OpenAlex / Semantic Scholar are tried before Unpaywall (no email needed) | `prisma_screening.py` | ₹0 |
| Local vector search over every screened abstract (ChromaDB on disk + bge-small on CPU; offline hashing fallback) | `vector_store.py` | ₹0 |
| PRISMA funnel: screen up to 50 abstracts across 4 databases, deep-read the top 10 | `prisma_screening.py` | ₹0 |
| Legal open-access full text via Unpaywall (DOI + your email → OA PDF → parsed sections) | `unpaywall.py`, `pdf_ingest.py` | ₹0 |
| Human / Animal / In vitro / Plant / Microbial / Other evidence buckets (deterministic, flags mixed designs) | `evidence_matrix.py` | ₹0 |
| Markdown + CSV Evidence Matrix next to the executive summary | `evidence_matrix.py`, `enhanced_report.py` | ₹0 |
| Physical protein–protein interactions from STRING-db | `string_db.py` | ₹0 |
| Deep reasoning with free open-weight models: Groq `openai/gpt-oss-120b` → `qwen/qwen3.8-27b` → Gemini thinking → NVIDIA NIM; reasoning trace shown in the UI | `ai_router.ask_reasoning`, `deep_reasoner.py` | free tiers |
| Grounding audit: citations to studies not in the matrix are stripped; certainty capped at "low" without direct evidence; claims are scoped to the retrieved sample ("Within the retrieved sample (S1–S10) ...") and over-general statements are flagged | `deep_reasoner.py` | — |
| Endpoint → biomarker query expansion (proliferation → Ki-67, PCNA; apoptosis → caspase-3; drought → stomatal conductance; resistance → MIC ...), with or without an AI key | `research_planner.py` | ₹0 |
| STRING caution: interaction scores are shown as association, never as a causal cascade | `deep_reasoner.py`, `enhanced_report.py` | — |
| Key tester, live progress stream, PRISMA diagram, interaction network, screening log, history, downloads | `server.py`, `static/`, `frontend/` | — |

DeepSeek-R1 is no longer served on Groq, so `gpt-oss-120b` (reasoning effort *high*)
is the default free reasoning model. Override with `GROQ_REASONING_MODELS`.

## Command line (unchanged + new flag)

```bash
python research_engine.py "How does metformin affect breast cancer?"                 # original
python research_engine.py "How does metformin affect breast cancer?" --prisma --email you@uni.edu
python research_engine.py "..." --agent                                             # ReAct agent
```

## v15 (2026-10-23)

Two code reviews. The first warned that the rules might be tuned to metformin and breast cancer; the second asked for
four tools from evidence-based medicine.

### Tested outside oncology - and fixed where it broke

The engine's rules were run on wording from six unrelated fields. Every one of these findings had been rejected or
mislabelled, and all now work:

| Field | What went wrong | Now |
|---|---|---|
| CRISPR in yeast | the species epithet ('cerevisiae') was read as a drug name | a Latin binomial is the organism, both words of it |
| CRISPRi in tuberculosis | 'bacterial survival' was filed as PATIENT survival, and the question was treated as tuberculosis medicine | a microbe's growth / survival is its own endpoint family; a laboratory question about a microbe is microbiology |
| CRISPR-Cas9 in zebrafish | 'Danio rerio' was not recognised as an animal; 'fin outgrowth was absent' had no direction | Latin names of the model animals are known; an absent structure is a decrease |
| drought in maize | 'improve' was not read as a direction | the verbs other fields use are directions too (improve, attenuate, ameliorate, alleviate, mitigate, sensitise, restore) |
| pirfenidone in lung fibrosis | the exposure came out as 'pirfenidone attenuate idiopathic pulmonary' | a verb or a disease modifier is never part of the drug name |
| Chagas disease | the disease became the word 'disease' plus a drug called 'Chagas' | a disease named after a person or place stays one name |

Two deeper fixes behind those: a question whose endpoint is not one of the built-in medical families (fluconazole
sensitivity, fin outgrowth, relative water content) could never be **answered** - every finding was filed as context
only; and in 'Does X cause Y?' the outcome Y was mistaken for a second exposure. Epidemiology's own way of stating a
direction - 'associated with incident cardiomyopathy (hazard ratio 2.4, 95% CI 1.6 to 3.6)' - is now read as a
direction, with a non-significant ratio read as no effect.

### Pooled effect instead of counting studies

The answer used to count votes ('2 studies show a reduction, 4 show no difference'), which gives a 15-patient trial
the same weight as a 3,500-patient one. The engine now pools the effect sizes the papers report, each weighted by its
own precision (inverse-variance, DerSimonian-Laird random-effects), and shows a forest plot with each study's weight
and the I-squared heterogeneity index. Computed locally; no API cost.

Rules it keeps: one endpoint at a time; ratios never pooled with absolute differences; human, animal and cell studies
pooled separately; publications of one trial counted once; two papers reporting an identical estimate and interval
counted once (identical numbers cannot come from two independent samples); a study reporting no interval listed as
not poolable rather than dropped; and nothing pooled from fewer than two independent studies. Validation now **fails**
a report whose pooled estimate includes no difference while its answer states an effect.

### Three methodological watchdogs

- **Misidentified cell lines.** MDA-MB-435 was studied as breast cancer for years before genetic testing showed it is
  the M14 melanoma line; others were overgrown by HeLa. A claim from such a line, used as a model of the disease
  asked about, is now quarantined, and the report says what the line actually is. From the ICLAC register.
- **Assay-interference compounds (PAINS).** Curcumin, resveratrol, quercetin and EGCG give false positives in almost
  every laboratory assay (they fluoresce, aggregate, oxidise). Their laboratory results now carry that caveat; their
  trial results do not, because an assay artefact cannot affect a result measured in people.
- **Outcome switching.** A trial registers the outcome it will measure. The engine now compares a paper's stated
  primary outcome with its ClinicalTrials.gov entry and flags a mismatch. Needs the network; skipped quietly offline.

### v16 - speed (the one architecture change that was worth making)

A real run spent 21 minutes in extraction: **74 AI calls produced 65 retries**. Three threads shared one free key,
each got "429 too many requests", and each then slept a second and retried the next model on the same exhausted key.
The retries were the run time. Two changes fix that, and both are reversible:

**1. Requests are paced instead of colliding.** A token bucket per provider, shared by every thread, sends requests
at the rate that provider accepts (its published free-tier limit, e.g. ~28/min for Groq, ~14/min for Gemini), so
they are not refused in the first place. When a provider does refuse, the wait it ASKS for is obeyed
("try again in 7.5s"), that provider is stood down, and the next ready provider is used immediately - the old code
slept on the exhausted key and tried its other models, which were rate-limited too. Measured on a simulated free
tier: refusals fell by 97%, and all 74 calls completed where the old code silently gave up on 45 of them.

**2. Short papers are read several per AI call.** A free key accepts a fixed number of requests per minute, so the
only way under that floor is to ask fewer times. Abstract-only papers are packed three to a call and the answer split
by paper tag. Measured end to end: a quick scan went from 8 AI calls to 3. Full runs improve less, because papers
with full text are still read one at a time - deliberately, since they are long.

Safety: every batched claim is still verified against ITS OWN paper's text, so a claim attributed to the wrong paper
is dropped by the grounding check exactly as before. A batch that fails returns nothing and every paper in it is read
the ordinary way. `BIOSYN_NO_BATCH=1` and `BIOSYN_NO_PACING=1` switch each change off; `BIOSYN_RATE_<PROVIDER>` sets
a higher rate for a paid key.

**The honest ceiling:** 74 requests on one key that accepts 28 per minute cannot finish in under about 2.6 minutes,
whatever the code does. Below that needs fewer calls (batching) or more keys. **Adding a second free key roughly
halves the AI time** - the report now says so in its timing table.

### v15.1 - speed fix (the registry check could add 15 minutes)

The outcome-switching check shipped unbounded: one lookup per trial identifier, each with the 25-second timeout and
two retries the engine uses for its evidence sources. With an unreachable registry that is about 79 seconds per
trial, so a sample with a dozen registered trials turned a 5-minute run into 20 minutes.

It is now bounded three ways and can never dominate a run: **one** attempt with a 6-second timeout, at most 8 trials
and 20 seconds for the whole check, and it stops after two failures in a row (an unreachable registry is noticed
once, not twelve times). Each trial is looked up once however many papers name it, a paper that never states a
primary outcome causes no lookup at all, and `BIOSYN_NO_REGISTRY=1` switches the check off. When the check stops
early the report says it was **not completed** instead of showing a clean pass.

Worst case, measured: 25 studies against a dead registry now cost 12 seconds and 2 requests, where before they cost
roughly 33 minutes.

### Deliberately not done

Replacing the text rules with a local NLP model (spaCy / DeBERTa), job-queue workers (Celery / RQ), moving from
SQLite to PostgreSQL, and loading the UMLS / MeSH vocabularies were all suggested. Each would change the program's
architecture, which is fixed by request. One research run at a time is intended for a desktop tool.

## v14 (2026-10-22)

Found by running the program end to end and reading both reports (quick and full) line by line.

- **Real results were thrown away.** When the AI put the drug itself ('Metformin use') or the whole sentence in the
  outcome field, a clearly reported result ('Ki-67 decreased by 11% ... p = 0.02', 'reduced recurrence, HR 0.78') was
  rejected. The endpoint is now read from the quote when the quote names exactly one; the claim is tagged so it can
  be checked. In the test run this turned 'not answered' into a correct answer backed by three randomised trials.
- **Every study is counted.** The answer cited one claim when seven studies had the same result; it now cites all of
  them ('C4, C5, C6; 3 studies'), and publications of one trial count once ('7 observational publications (1
  independent cohort)'). Laboratory support is given as a count ('1 of 4 show a reduction').
- **AI sentences the evidence does not support** are withheld or rewritten, clause by clause: 'larger effects are seen
  only in animals' with no animal study read, 'animal xenograft studies [S4]' where S4 is a human trial, 'mouse doses
  exceed human exposure' while the dose check says physiological (rewritten from the dose check), 'null rat study'
  with no rat study read. An uncited explanation is labelled as AI interpretation. A withheld human / animal / cell
  summary is replaced by what the verified findings of that kind show.
- **Pathway steps** keep a citation only when that study's verified findings mention the step's genes.
- **Conflicts**: laboratory tumour growth vs human survival is labelled as different endpoints; an effect vs no
  significant difference is 'different results', not 'opposite results'.
- **Numbers**: 'n = 24 per group' is not the total; an AI sample size the paper never mentions loses to the paper's
  stated total; 'randomised ... for 2 weeks before surgery' is treatment time, not follow-up; identical statistics
  with 20 vs 760 patients are not one trial; the species named in the paper ('obese rats') beats an extracted one.
- Natural order of studies (S9 before S12), singular wording ('1 testable hypothesis'), and a dose gap is shown with
  the report's own dose check.

## v13 (2026-10-22)

- **Body measurements are not doses.** 'BMI >= 27 kg/m2' and 'weighing 70 kg' are never read as a drug dose (the
  quick-scan report showed 'dose 27 kg'); doses per body weight (mg/kg) still count.
- **The intervention is the drug, not a mechanism.** 'through indirect and direct mechanisms of action' is never a
  regimen component, and a regimen is only shown for a study when one of its components is the studied drug (no more
  'chemotherapy + direct mechanisms of action' for a metformin study).
- **Publication years.** Each paper gets the year it was FIRST published (when PubMed, Crossref, Europe PMC and the
  full-text XML disagree, the earliest wins; the PMC release date is ignored), and a year after today is dropped
  instead of cited. Papers dated this year are genuine.

## Bugs fixed in the uploaded code

* `core_utils.py` (imported by `database.py`) was missing — added `now_iso()`.
* `ResearchDB` lacked six methods other modules call: `clear_contradictions`,
  `save_claim_group` (contradiction detection crashed every Deep Research run),
  `add_experiment_design`, `update_hypothesis_status`, `upsert_dataset`,
  `log_provider_event`. Added, using the tables already in the schema.
* Retired model IDs (`gemma2-9b-it`, `gemini-2.0-flash`, `gemini-1.5-flash`) replaced
  with current ones.
* A malformed full-text XML from a source no longer aborts the whole investigation.
* SQLite now runs in WAL mode with a busy timeout and a write lock (safe concurrent writes; the
  history page can read while a job writes).
* Progress printed by the engine's worker threads now reaches the live log in the browser.
* The live-progress stream is non-blocking and stops as soon as the browser tab is closed.
* Contradiction detection no longer groups monotherapy with combination therapy
  ("metformin" vs "metformin plus doxorubicin"), which produced false contradictions.
* Gemini reasoning retries without thought summaries if a model rejects that option.

## Getting your free keys (one AI key required)

| Key | Where to get it | What it unlocks |
|---|---|---|
| Groq | https://console.groq.com/keys | Fast AI claim extraction + deep reasoning (gpt-oss-120b) |
| Gemini | https://aistudio.google.com/apikey | Backup AI + Gemini "thinking" reasoning |
| NVIDIA NIM | https://build.nvidia.com | Second backup AI |
| Email (not a key) | your own address | Unpaywall legal open-access PDFs |
| OpenAlex | https://openalex.org/settings/api | Higher OpenAlex search allowance |
| Semantic Scholar | https://www.semanticscholar.org/product/api | Dedicated S2 quota, citation chasing |

One AI key (Groq is the easiest) is required and enough. Use **Test keys** on the API keys page to check them.
Keys stay in your browser and are sent only with your own research requests.

## What you get from each investigation

* **Executive summary** with a one-line bottom line and a GRADE-style certainty rating
* **Evidence Matrix** (Markdown + CSV): every deep-read study, its model, design, sample size,
  direction of effect and p-value, sorted into Human / Animal / In vitro buckets
* **PRISMA funnel**: how many records were found, de-duplicated, screened, excluded (with reasons)
  and deep-read
* **STRING protein-interaction network** for the genes/proteins mentioned
* **Contradictions** between studies, hypotheses and research gaps
* **Reasoning trace** from the AI model, plus an audit of removed citations
* Downloads: full report (.md), matrix (.csv), dossier (.json); past runs under **Past investigations**
* Upload your own CSV/Excel data to test a hypothesis against the literature
* **Local library** tab: semantic search over every abstract you have ever screened (works offline)
* Works for any area of biology (medicine, plants, microbes, animals, ecology)
* Three modes: **Evidence dossier** (PRISMA, recommended), **Deep pipeline**, **ReAct agent**

## Files in this package

| Part | What it is |
|---|---|
| `server.py` + 45 engine `.py` files (incl. `spelling.py`, `structure.py`, `omics.py`) | The research engine and web server |
| `retrieval/` | PubMed, Europe PMC, Crossref, OpenAlex, Semantic Scholar clients |
| `static/` | The web page (HTML, CSS, JavaScript) |
| `run.bat` / `run.sh` | One-click start for Windows / macOS-Linux |
| `check_setup.py` | Tells you about missing, old or misplaced files |
| `requirements.txt` | Python packages (installed automatically by `run.bat`) |
| `.env.example` | Optional settings for the command-line tools only |
| `frontend/` | TypeScript source of the web page (only needed to rebuild `static/app.js`) |
| `tests/` | Offline tests (no internet or keys needed) |
| `literature_search.py`, `read_results.py`, `data_engine.py`, `research_pipeline.py`, `scientific_reasoner.py`, `synthesis_engine.py`, `rich_evidence_parser.py`, `app.py`, `main.py` | Extra command-line tools (not needed by the website) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `check_setup.py` says NOT READY | Unzip into a NEW empty folder; never mix with old copies |
| "Python not found" | Install Python 3.10+ from python.org and tick "Add Python to PATH" |
| First run is slow | It downloads the 130 MB embedding model once |
| No AI summary, only a tally | No working AI key: add one and press **Test keys** |
| Page won't open | Open http://127.0.0.1:8000 yourself; keep the black window open |
| Port 8000 busy | Close the other program, or run `set PORT=8001` then `python server.py` and open http://127.0.0.1:8001 |

## Tests (no internet or keys needed)

```bash
python tests/test_units.py        # unit checks
python tests/test_regressions.py  # one check per fixed bug
python tests/test_evidence_audit.py  # one or more checks per evidence-audit safeguard (#2-#15)
python tests/test_report_reliability.py  # replays a real leukoderma dossier: a check for each of 43 review points
python tests/test_evidence_identity.py   # replays a real leukemia dossier: one or more checks per review point (68 checks)
python tests/test_code_review.py         # the 4-point code review + the full bug hunt (29 checks)
python tests/test_metformin_dossier.py   # replays a real no-key metformin dossier line by line (49 checks)
python tests/test_retrieval_healer.py    # replays the "Can leukimia be cured" dossier + the auto-healer (25 checks)
python tests/test_ai_key_dossier.py      # replays the AI-key metformin review point by point + real papers (53 checks)
python tests/test_upgrades.py            # key gate, AI task roles, virtual lab, uploads, experiment, censoring, healer, pages (45 checks)
python tests/test_search.py              # spelling, "Did you mean", autocomplete, exact-as-typed, search pages (44 checks)
python tests/test_v4.py                  # sample-report fixes, level-by-level answer, any-biology screening, agent budget,
                                         # 3D structures, omics statistics, website (70 checks)
python tests/test_v5.py                  # line-by-line review of the v4 metformin dossier: anti-proliferative = DOWN,
                                         # Discussion background / speculation, retracted papers, knockdown background,
                                         # combination regimens, endpoint-aware answer, three-part contradiction wording,
                                         # semantic grounding, plausible n, replication without reviews, thread-safe
                                         # database, symptom questions, section-aware models, comorbidities, human cues,
                                         # rank-based screening floor, random-question diagnosis (46 checks)
python tests/test_v6.py                  # the four 2026-10-08 review lists: replication vs mechanistic hypotheses,
                                         # read-outs never agents, per-arrow mechanism levels, three-layer dossier,
                                         # network / structure wording, pocket confidence (21 checks)
python tests/test_v7.py                  # every line of the full v6 metformin dossier + the reviewer's notes: latency,
                                         # combinations, cell vs animal, endpoint-specific absence, scope notes, hypothesis
                                         # kinds, targets, drug abbreviations, deep-read priority, reflection round (76 checks)
python tests/test_v8.py                  # speed: quick scan, saved-result reuse, fewer AI calls per paper, parallel reading,
                                         # one repair round, stage timings, failure-proofing (17 checks)
python tests/test_v9.py                  # the 2026-10-15 review: endpoint contradictions, comparator-relative directions,
                                         # per-check validation, falsification logic, study registry, runtime (24 checks)
python tests/test_v10.py                 # the two 2026-10-16 reviews line by line + runtime: significance, endpoints,
                                         # doses, classification, trials, guards, gaps, validation layers, console (61 checks)
python tests/test_v11.py                 # the 2026-10-19 review: separate subgroups, gaps from uncertainties, trials in
                                         # Layer 1, within-arm changes, proving instead of withholding, retries (24 checks)
python tests/test_v12.py                 # the 25-paper review: one record per study, change magnitude, within-arm changes,
                                         # cross-section validation, summaries from findings, endpoint relabels, doses (22 checks)
python tests/test_v13.py                 # the metformin quick-scan review: BMI is not a dose, mechanisms are not drugs,
                                         # publication year = first publication and never after today (14 checks)
python tests/test_v14.py                 # bugs found by reading the program's own reports: endpoints repaired from the
                                         # quote, every study counted, unsupported AI sentences, conflicts, n, follow-up (27 checks)
python tests/test_v15.py                 # six unrelated fields (yeast, tuberculosis, zebrafish, maize, lung fibrosis,
                                         # Chagas) + meta-analysis, cell-line identity, PAINS, outcome switching (51 checks)
python tests/test_v16.py                 # speed: request pacing, a rate limit switches provider, several papers per
                                         # AI call, the registry check bounded (23 checks)
python tests/offline_harness.py   # full app on :8765 with synthetic fixtures + mock LLM
```

## Rebuilding the front-end

`static/app.js` is compiled from `frontend/src/app.ts`:

```bash
cd frontend && npm install && npm run build
```

## v22 - correctness, one evidence record, hypotheses vs discoveries (additive)

Three layers added without changing any existing module's behaviour (only `research_engine.py` and
`hypothesis_evaluator.py` were touched, additively). Each has an off-switch.

| Layer | File | What it does | Off-switch |
|---|---|---|---|
| A. Gold standard | `gold_eval.py`, `gold/` | Scores a finished run against papers a person verified: study-type accuracy, finding recall/missed, precision, false-positive claims, citation accuracy, per difficult-case tag. Read-only. `python gold_eval.py --dossier research_dossier_<id>.json` | `BIOSYN_GOLD_EVAL=0` |
| B. Canonical record | `evidence_record.py` | One record with stable IDs (STU-/EXP-/OUT-/CIT-, content hashes, not positional S1/E1). Counts derive from it; a consistency check compares it with the matrix totals and the report's "Evidence accounting" line. A mismatch is critical and goes to BUG_REPORT.md. | `BIOSYN_EVIDENCE_RECORD=0` |
| C. Hypothesis vs discovery | `hypothesis_schema.py` | Each hypothesis must have supporting claim IDs, a competing explanation, a testable prediction, a falsification condition and an experiment. Missing parts are labelled "speculative", never invented. Causal / "proves" wording is flagged. A hypothesis is a "discovery" only when a data test supported it AND the dataset declares an interventional design. | `BIOSYN_HYPOTHESIS_SCHEMA=0` |

`v22_layers.py` attaches them at the end of `investigate()` (new dossier keys `evidence_record`, `hypothesis_audit`,
`gold_eval`, `v22`; one extra report section). The hypothesis prompt now also asks for `competing_explanations`, which
are saved. The gold set ships EMPTY of verified papers (only a template and a synthetic scorer fixture): add real,
person-verified papers in `gold/papers/` - see `gold/README.md`.

    python tests/test_v22.py

## v23 - merge of the v17/v18 correctness fixes into the v22 tree

The v22 tree was developed in parallel with v17 and v18 and did not contain their fixes. v23 carries every
one of them across, keeps all of v22's layers, and fixes one regression the parallel work introduced.
No module boundary, call order or data shape changed: every item below is a narrower test inside an
existing function.

| # | Symptom in a real report | Cause | Fix |
|---|---|---|---|
| 1 | A sentence from a paper's **reference list** was reported as that paper's own finding | a quote is scored on wording, with no check of where in the document it sits | `is_bibliography_quote()` in `semantic_audit.py`: a quote taken from a reference list / bibliography section, or carrying reference-list shape (author-year-journal-volume-pages), can never be marked `this_study` |
| 2 | "**merits further investigation**" was recorded as a result | the aim/speculation patterns covered stated aims but not future-work wording | `AIM_OR_SPECULATION` extended with the future-research family (merits/warrants/deserves further investigation, further studies are needed, future work should, remains to be determined, should be investigated, it would be interesting to) |
| 3 | The same answer printed in two different sections | two sections selected the same claim independently | one selection, both sections read it |
| 4 | Run died on a provider **413 (prompt too large)** | the deep-reasoning prompt had one fixed size | `PROMPT_TIERS` in `deep_reasoner.py`: on 413 the prompt is trimmed a tier at a time, then a `COMPACT_PROMPT` fallback; the run continues instead of failing |
| 5 | A mechanism chain showed one label for every step | the label was computed once for the chain | per-step mechanism labels |
| 6 | A **10 mM** cell-culture dose was reported as if relevant to people | the dose check compared the number with nothing | the dose is judged against achievable human exposure, and says so when it is far above it |
| 7 | The status line could read as if the report were publishable | wording | the status line states **NOT READY TO PUBLISH** when any critical check is open |
| 8 | Deep reading could not be pushed past 40 papers | `deep_read_k` ceiling | ceiling raised to **80** (`server.py`) |
| 9 | A study with a **critical** methodological flag was still cited in the bottom line | the conclusion builder saw every verified row | critically flagged studies are excluded from the conclusion, **and the exclusion plus its reason is stated** in the conclusion (`report_validator.py`); their findings stay in the evidence matrix and the flags section |
| 10 | Regression from the parallel work: "**relative water content**" under drought was filed as plant growth / yield | the term was listed in the `plant_growth_yield` pattern | removed from that pattern so it falls through to `stress_tolerance`, which is what it measures |
| 11 | When the only candidate study was excluded by #9, the report said "no verified primary finding" with no reason | the empty-result branch had no view of the exclusions | that branch now names the excluded studies and their flags, and states that this is **evidence that cannot be used**, not absence of evidence |

### What was run before release

| Check | Result |
|---|---|
| Full test suite, 27 files, Python 3.10 / 3.11 / 3.13 | **0 failures** on all three |
| Offline end-to-end run (`quick` and `full`), report validation | **0 failed checks**, every report section present |
| `check_setup.py` from a fresh unzip | pass |

    python tests/test_v17.py
    python tests/test_v18.py

### Not shipped in the zip (all of it is rebuilt automatically)

`cache/`, `vector_store/`, `papers/raw/`, `runs/`, `research_knowledge.db`, `research_dossier_*.json`,
`__pycache__/`. These are run artifacts. Shipping another machine's `cache/ai_answers` in particular would
make your first run replay someone else's AI answers instead of calling your own key. **On the first run the
engine downloads its embedding model (~65 MB) once**; after that it is local and the run is offline-capable.

## v24 - the faults a real vitiligo run published while every check passed

Run of 2026-10-25, question *"can vitiligo be cured and how"*. All 32 pre-publication checks reported
**passed**, and the report still contained the errors below. The last row is the important one: the checker
could not see any of the others.

The worst of them came from a single root cause. **S20 (Tulic 2019)** added CXCL10 to *cultured melanocytes*
and read the result on an IncuCyte, a live-cell imager. The engine filed that measurement as **human
observational**, let the relevance gate admit it as a **direct answer** to the cure question, renamed
"melanocyte death" to **"patient survival"**, listed the *decrease* among human findings that "point to a
benefit", and generated **H3: a randomised controlled human trial of CXCL10, predicted to decrease patient
survival** - an experiment nobody may run.

| # | What the report published | Why it was wrong | Fix |
|---|---|---|---|
| 1 | "CXCL10 using IncuCyte → patient survival ↓", filed as human evidence | a cell measurement wearing the paper's design label | an assay that only exists with cultured cells (`IN_VITRO_ASSAY`) settles the system for that claim; and "survival"/"death" attached to a **cell type** is cell viability, never a patient outcome |
| 2 | "While the →", "pi3k pathway leads →", "di dose-dependently →", "melanocytes →", "PI3K-AKT-mTOR signaling pathway →", and the research question itself | none of these is something a study *did* to anything | `implausible_subject()` rejects sentence fragments, instruments, pathways, bare cell types and the question; `strip_instrument()` turns "CXCL10 using IncuCyte" into "CXCL10" |
| 3 | "ultraviolet B → DNA damage ↓ **decrease**" | UV-B *causes* DNA damage; the quote credited a miR-31-5p mimic | a damaging exposure may not be credited with reducing its own damage when the quote names the rescuing agent |
| 4 | "T-96 → **metastasis invasion** ↓" on a vitiligo paper | the quote counted CD8+ T cells by whole-mount staining - immune infiltration | a cancer endpoint is re-read as immune infiltration when neither the question nor the paper is about cancer |
| 5 | "9 in vitro studies" in the summary, "7" in the table | two different, both-correct counts under one label | the summary now says which studies are filed under another design and what the bucket table counts |
| 6 | "pmel17 vitiligo", "ccl8 vitiligo", "skin vitiligo" as distinct disease definitions → a false "studies do not share one disease definition" | gene symbols glued to the disease name | a gene or protein symbol is never a form of a disease; model identifiers (4T1) and real subtypes survive |
| 7 | "0.2 months of follow-up" for a cell-culture study | a dish has an incubation time, not clinical follow-up | follow-up is reported only for human designs |
| 13 | **all 32 checks "passed"** | the checker had no test for any of the above | four new checks (below) |

### The four new pre-publication checks

| Check | Severity |
|---|---|
| no cell or in-silico measurement is reported as a patient outcome | critical |
| every published intervention is something that was administered or manipulated | critical |
| a cancer endpoint appears only where the question or the paper is about cancer | major |
| **no proposed human experiment predicts harm to the people in it** | critical |

The last one is the one to show: it refuses to publish a hypothesis that would randomise people to something
the report itself predicts will hurt them.

    python tests/test_v24.py

### What was run before release

| Check | Result |
|---|---|
| Full test suite, 28 files, Python 3.10 / 3.11 / 3.13 | **1047 pass, 0 fail** on all three |
| Offline end-to-end run (`quick` and `full`) | 36 checks, **0 failed**, every section present |
| `check_setup.py` from a fresh unzip | pass |

### v24.1 - "it is not running"

`main.py` was a leftover 25-line Gemini connection test. Running it - the obvious thing to do with a file
called *main* - raised `RuntimeError: GEMINI_API_KEY was not found. Check your .env file.` and stopped. The
program was fine; the file anyone would open was not, and it contradicted the rule that keys are typed into
the web page and never kept in a file.

| File | Was | Now |
|---|---|---|
| `main.py` | a Gemini hello-world that crashed without a `.env` | starts the real web app (`python main.py` → http://127.0.0.1:8000) |
| `app.py` | the old Streamlit page; `python app.py` printed a wall of warnings and started nothing | says in three lines what to run instead |

`run.bat`, `run.sh` and `server.py` were always correct and are unchanged.
