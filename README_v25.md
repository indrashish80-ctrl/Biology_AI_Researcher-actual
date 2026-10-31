# BioSynthetica v25 — accuracy you can measure, Tiers 1–4

v25 fixes the errors found in v24.2, adds the six accuracy principles, and builds Tiers 1–4 from the roadmap.
Every number below was **measured** by `python gold_bench.py` (`gold/benchmark_results.md`). None is a target.

## Measured accuracy (2026-10-27)

| what | set | result |
|---|---|---|
| direction of an effect read from a doctor-annotated evidence sentence | **held-out**, 323 sentences, never used for tuning | **75.5%** (v24.2 reader: 58.5%) |
| opposite-direction errors (said "increase" when the paper says "decrease", or the reverse) | held-out | 34 of 300 sentences read (11.3%) |
| sentences the reader could not read | held-out | 23 (v24.2: 81) |
| direction reading | development set, 162 sentences (**tuned on, optimistic**) | 90.7% |
| retrieval recall@5, hybrid + rerank | 162 gold questions over 645 abstracts | 100% (BM25 alone 97.5%, R@1 94.4% vs 88.3%) |
| verdict-rule violations | 509 gold-derived cases | 0 |
| metamorphic fuzzer, negated templates now included | 81,600 sentences, 4 seeds | 0 wrong (v24.2: 6 of 10,000, plus 196 hidden by leaving out the negated templates) |

**Quote the held-out number (75.5%)**, not the development number. Read the honest caveats:
* Every gold paper is a randomised trial (Evidence Inference 2.0), so the gold set covers human RCT abstracts.
  Cell and animal evidence is covered only by the fuzzer and the replayed dossiers in `tests/`.
* The retrieval questions reuse the annotators' wording, so retrieval is an easy lexical test. Real questions
  are harder.
* Part B (extraction recall) in the shipped benchmark ran in **rules mode** (no AI). It measures the no-key
  fallback, not the AI extractor. With a key, run `python gold_bench.py --only B --ai` to measure the real recall and
  the quote guard's drop rate.
* The gold review was an **AI review** (Claude) of doctor annotations. Two or more annotators agreed on every
  kept label, and the 14 excluded prompts have written reasons in `gold/ei_review.json`. A person should countersign
  before high-stakes use. Use the Review page to do it.

## Errors in v24.2 that are fixed

| | error | effect |
|---|---|---|
| E1 | the fuzzer left the negated templates out of its main run | it hid 196 wrong readings ("did not lower" read as a decrease) |
| E2 | "Anti-TNF therapy ... decreased IL-6" read TNF as the endpoint; "after loss of p53" read as a decrease of the endpoint; "reversed the decrease" read as a decrease; "improved blood pressure" read as a rise; HDL read from the LDL clause | wrong direction |
| E3 | Lancet mid-dot decimals: `p=0·19` was read as **p = 0** (significant); a Unicode minus made `−5.6 to 14.4` into `5.6 to 14.4` (significant); "confidence interval (CI)" was not parsed | false significance |
| E4 | the spell-checker "corrected" real words ("exposed" → "exposure", "repigmentation" → "depigmentation", the opposite) | searched the wrong thing |
| E5 | "riociguat compared with placebo" gave the exposure "riociguat compared"; "lifespan" was taken as an exposure | wrong search |
| E6 | one paper from three sources (DOI only / PMID only / neither) became three records | double counting |
| E7 | the regression tracker compared claims by database ID, so every re-run looked 100% changed | useless diffs |
| E8 | `claim_verifier` was called a second reader but re-read the same output deterministically | false assurance (now documented and superseded) |
| E9 | `test_v7` checked a hard-coded version list | the test failed on every new version |
| E10 | `check_setup.py` had duplicate dictionary keys, so 11 older version checks silently never ran | missed stale files |
| E11 | the lab job queue used multiprocessing "spawn", which re-imports the caller's main script in every job | a script without a `__main__` guard ran again inside each experiment (now a plain subprocess) |
| E12 | the archive read `license-type="open-access"` over the CC BY URL, so CC BY texts were marked "do not redistribute" | wrong licence |
| E13 | `INSERT OR IGNORE` + `lastrowid`: an ignored duplicate ingestion job was reported as queued | duplicate jobs counted |
| E14 | reviews citing "Garcia et al. (2019)" were not linked to the trial they include (the "." of "et al." broke the match) | review + trial double counted |

### Found in the code audits after the first release (regression tests: `tests/test_v25.py` R1-R24)

| | error | effect |
|---|---|---|
| R1 | the quote guard dropped a minus sign at the start of a text (`"" in "-_"` is True in Python) | `-0.42%` was read as no number |
| R2 | the quote guard did not read Lancet mid-dot decimals | `0·19` became two numbers, 0 and 19, so a correct finding could be dropped |
| R3 | scientific notation was not read; "3D" was read as "3 days" | `1.2×10^5` was read as 5 |
| R4 | pooling crashed on a ratio interval reaching 0, and used effects whose estimate lay outside their own interval | crash / a misread study pooled |
| R5 | Egger's test on points lying exactly on a line gave t ≈ 10^15, p = 0 | a false "publication bias" signal |
| R6 | the bias analysis crashed when there was no pooled estimate | the layer failed |
| R7 | pooled studies lost their CI level, so a 99% interval was re-read as 95% | wrong standard error in the bias tests |
| R8 | the review queue reported an ignored duplicate as a new item (`lastrowid` after `INSERT OR IGNORE`) | duplicate counts |
| R9 | a decision on a review item that does not exist was "saved" | silent no-op |
| R10 | `POST /api/review/export` was declared after `/api/review/{item_id}`, so "export" was parsed as an id | the Export button always failed (422) |
| R11 | an unknown simulation job id answered 200 | the page polled forever |
| R12 | the experiment validator crashed on a wrong type (a string for `parameters`, `observables`, sweep values ...) | HTTP 500 / a crashed job |
| R13 | a numeric string in `expected.value` passed validation and crashed the run (`13.1 > "5"`); threshold low >= high was accepted | crashed job / endless bisection |
| R14 | the dossier diff looked in a hard-coded `runs/` folder, not the server's job folder | "job not found" |
| R15 | a claim whose statement is exactly the old intervention name was never renamed (the same `""` trap as R1) | stale wording |
| R16 | `benchmark_eval.py` and the gold importer opened files without `encoding="utf-8"` | crash on Windows with non-ASCII text |
| R17 | the job queue claimed a Windows memory cap (Job Object) that was never implemented | now implemented (R20) |
| R18 | the server used `@app.on_event("startup")`, deprecated by FastAPI | replaced by a lifespan handler that also stops the ingestion worker on shut-down |
| R19 | the ingestion worker's stop flag was never cleared | after a stop it could never be started again |
| R20 | Windows had no CPU or memory cap for lab jobs | a Job Object (kernel32 through ctypes, no pywin32) now caps CPU time and committed memory |
| R21 | the tests overwrote the real `RESEARCH_REPORT.md` and wrote `research_dossier_*.json`, `research_knowledge.db`, `runs/uploads/*`, `vector_store/` and fixture papers into the project folder | `run_tests.py` gives every test file its own temporary folder (`BIOSYN_DB_PATH`, `BIOSYN_OUTPUT_DIR`, `BIOSYN_RUNS_DIR`, `BIOSYN_RAW_DIR`, `VECTOR_STORE_PATH`); the test files also stopped sharing one database while running in parallel |
| R22 | the research engine's vector store defaulted to `vector_store/` in the CURRENT folder | started from another folder, the engine silently used a second, empty store |
| R23 | a job killed by a cap on Windows was reported as "failed" (the exit code is positive there) | now "resource_limit" on every system |
| R24 | 13 f-strings without placeholders, 4 unused variables, 75 unused imports | removed (pyflakes is clean apart from intentional imports marked `noqa`) |

## The six accuracy principles, as code

1. **Quote, not paraphrase** — `quote_guard.py`. Every AI finding must carry a verbatim quote. Code locates it
   in the source after harmless normalisation (dashes, spaces, ligatures) or snaps it to the one matching sentence,
   then stores the paper's own characters. Every number in the finding must appear in the quote. Otherwise the
   finding is **dropped**, and every drop is recorded (`extraction_guard` in the dossier).
2. **Work moved into code** — p-values, effect sizes, confidence intervals and sample sizes are parsed from the
   quote by `stats_parsing` / `meta_analysis.extract_effect`. Code wins over the model.
3. **A second, independent reader** — `independent_reader.py`. It uses a different prompt (a reviewer's results
   table) and does not see the first reader's output. On disagreement, the finding is quarantined. Switches:
   `BIOSYN_SECOND_READER=auto|1|0` and `BIOSYN_DISAGREEMENT=quarantine|drop|keep`. Findings only it saw go to the
   review queue.
4. **Structured sources first** — each finding's reading confidence depends on its text basis (JATS XML > PDF >
   abstract), quote mode and second-reader agreement. A finding read from an abstract only is never "high".
5. **Measured against people** — `gold/` (45 papers, 162 findings + 323 held-out) and `gold_bench.py`.
6. **A person in the loop** — the Review page (`/review`) shows the quote, the source and the reason. Approve,
   correct or reject. Approved items can be exported as new gold drafts.

## Tier 1
* **Gold set + metrics:** recall@k, MRR, citation accuracy, unsupported-claim rate, and direction accuracy per
  difficult case (`gold_bench.py`).
* **Hybrid retrieval:** SQLite FTS5/BM25 (`keyword_index.py`) + vectors, merged by reciprocal rank and reranked
  on PICO match (`hybrid_retrieval.py`). MeSH-style synonyms and UK/US spellings are in `mesh_synonyms.py`; add the
  full MeSH with `BIOSYN_MESH_FILE`. Screening blends it in. `BIOSYN_HYBRID=0` restores the old blend.
  `BIOSYN_RERANKER=cross` uses a cross-encoder.
* **Query parser** (`query_parser.py`): population, intervention, comparator, outcome, question type and expected
  direction.
* **Publication status** (`publication_status.py`): retractions, errata and expressions of concern from PubMed
  publication types, correction links, Crossref updates and titles. Dependent claims are flagged, retracted ones
  are quarantined, and `POST /api/status/recheck` re-checks papers.
* **Calibrated verdicts** (`verdict_calibration.py`): strongly supported, supported with limits, mixed,
  preliminary, insufficient. Each has written criteria. One trial published twice counts once.

## Tier 2
* Extraction cache (`extraction_cache.py`), keyed by DOI/PMID + extractor version + text + question.
* Full-text archive with licence tracking (`fulltext_archive.py`): each text is downloaded once.
* Background ingestion (`ingestion_queue.py`): watched topics are re-searched daily, failed jobs retry with
  back-off, and records without identifiers are deduplicated.
* Embedding and result caches (`retrieval_cache.py`), invalidated when the index changes.
* Analysis tiers (`analysis_tiers.py`): quick → research → deep, with automatic escalation (speed "auto") when
  the evidence is thin, conflicting, abstract-only or incomplete.
* Run metrics (`run_metrics.py`): p50/p95 latency, AI calls, cost and cache hit rate, tracked against gold accuracy
  per code version (`GET /api/metrics/trend`).

## Tier 3
* Evidence dependency graph: same trial, shared dataset, overlapping cohort, review includes. It reports the
  **effective number of independent sources**.
* Random-effects meta-analysis of independent RCTs (REML + Hartung-Knapp + prediction interval + leave-one-out),
  with publication bias checks: funnel plot, Egger test, trim-and-fill, and registry vs published.
* Cross-scale chain (molecule → cell → tissue → organism → human). Each link is tested, contradicted or untested.
* Typed knowledge-graph edges in `entity_relations`, each linked to the claim it rests on.
* Versioned scientific memory: corrected papers and new papers are mapped to the dossiers they affect.
* Reproducible dossiers: a manifest records versions, search and screening decisions. Use `python
  regression_tracker.py A.json B.json` or `POST /api/dossier/diff` to compare two runs.
* Risk-of-bias checklists: RoB 2 for RCTs, ROBINS-I for observational studies, SYRCLE for animal studies. Only
  full texts are assessed.
* Numeric audit: original and normalised units side by side, with abstract numbers checked against the tables.

## Tier 4 (virtual lab, optional: `pip install -r requirements-lab.txt`)
* `model_library/`: 7 model cards (PK, SIR, logistic growth, Michaelis-Menten, ...) with assumptions,
  parameter ranges, limits and references.
* `simulator.py`: Tellurium / libRoadRunner. Each result records the model version, solver and tolerances, and can
  be exported as SBML.
* `experiment_spec.py`: the AI may only *propose* a spec, and the validator refuses unknown models or
  parameters, out-of-range values, and runs or sweeps that are too long.
* `sensitivity.py` (Morris, local, threshold search by bisection), `model_validation.py` (fit on training
  points, test on held-out points), and `experiment_log.py`, which keeps every run, including failed and rejected
  ones.
* `job_queue.py`: every experiment runs in its own process with CPU, memory and wall-time caps (POSIX: setrlimit;
  Windows: a Job Object; each result reports `caps_active`).
  Caps are set by `BIOSYN_JOB_CPU_S`, `BIOSYN_JOB_MEM_MB` and `BIOSYN_JOB_WALL_S`; `BIOSYN_LAB_WORKERS` sets the
  number of workers.
* Every report is split into three labelled sections: **A. Published evidence**, **B. Simulation results**
  (in silico), **C. AI hypotheses** (untested).

## New pages and API
* Pages: `/review` (the review queue) and `/simulate` (the virtual lab). The research page now offers Quick,
  Research, Deep and Auto.
* API: `/api/v25/info`, `/api/query/parse`, `/api/search/hybrid`, `/api/review`, `/api/status/recheck`,
  `/api/memory/impact`, `/api/graph/{rq}`, `/api/dossier/diff`, `/api/metrics/trend`, `/api/gold/benchmark`,
  `/api/ingest/*` and `/api/lab/*`. The endpoint list is in `server_v25.py`.

## Tests
    python run_tests.py          # 29 files, 1175 checks, no internet or keys
    python metamorphic.py --seed 1 --per-template 300
    python gold_bench.py         # writes gold/benchmark_results.{json,md}

## Not done (Tier 5)
Postgres, a vLLM local model server and multi-user accounts are **not** in v25. The database is still a single
SQLite file per machine.
