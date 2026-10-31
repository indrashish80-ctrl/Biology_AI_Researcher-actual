"""
check_setup.py  -  run this FIRST:   python check_setup.py

Tells you, in plain words:
  * which files are missing or in the wrong folder
  * which files are OLD versions (mixed in from an earlier download)
  * duplicate "(1)" copies that should be deleted
  * which Python packages still need installing
and finally whether the app is ready to start.
"""

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
problems = []
warnings = []

REQUIRED = """
adversarial_critic.py ai_router.py app.py autonomous_agent.py bio_databases.py claim_extraction.py
contradiction_detection.py core_utils.py data_engine.py database.py dataset_discovery.py deep_reasoner.py
enhanced_report.py evidence_bridge.py evidence_engine.py evidence_matrix.py http_client.py
hypothesis_evaluator.py jats_ingest.py literature_search.py llm_json.py main.py ncbi_throttle.py
paper_extraction.py pathway_synthesizer.py pdf_ingest.py posthoc_analysis.py prisma_screening.py
provider_health.py read_results.py report_generator.py requirements.txt research_engine.py
research_pipeline.py research_planner.py rich_evidence_parser.py scientific_reasoner.py scientific_tools.py
server.py statistical_analysis.py stats_parsing.py string_db.py
study_classifier.py semantic_audit.py evidence_quality.py reasoning_guards.py clinical_context.py report_validator.py evidence_identity.py report_auditor.py auto_healer.py virtual_lab.py spelling.py structure.py omics.py synthesis_engine.py unpaywall.py vector_store.py
meta_analysis.py methodology_flags.py ai_pacing.py
metadata_authority.py section_roles.py v22_layers.py evidence_record.py hypothesis_schema.py gold_eval.py
quote_guard.py independent_reader.py review_store.py mesh_synonyms.py query_parser.py keyword_index.py hybrid_retrieval.py
publication_status.py verdict_calibration.py arm_direction.py gold_bench.py extraction_cache.py fulltext_archive.py
ingestion_queue.py retrieval_cache.py analysis_tiers.py run_metrics.py evidence_graph.py publication_bias.py cross_scale.py
knowledge_graph.py scientific_memory.py dossier_repro.py risk_of_bias.py numeric_audit.py simulator.py experiment_spec.py
experiment_log.py sensitivity.py model_validation.py job_queue.py dossier_sections.py v25_layers.py server_v25.py
completeness_gate.py strict_answer_layer.py regression_tracker.py
library_db.py library_vectors.py library_loader.py library_search.py user_papers.py preread.py passage_select.py bench_library.py sim_verifier.py
study_funding.py quality_profile.py risk_numbers.py causal_checks.py dose_plausibility.py confidence_calibration.py
evidence_reviewer.py research_value.py model_uncertainty.py lab_agent.py v26_layers.py server_v26.py mcid.json
retrieval/clinicaltrials.py static/v26.js gold/calibration.json gold/calibration/ei_calibration.jsonl
model_library/pk_oral_2cmt.json model_library/tumour_logistic_drug.json model_library/gene_negative_autoregulation.json
model_library/toggle_switch.json model_library/cell_state_quiescence.json model_library/metabolic_feedback_pathway.json
model_library/pk_oral_1cmt.json model_library/pkpd_emax.json model_library/tumour_gompertz_drug.json
model_library/viral_tcl.json model_library/sir.json model_library/glucose_minimal.json model_library/enzyme_mm_inhibitor.json
gold/ei_review.json gold/questions.jsonl static/review.html static/simulate.html static/v25.js
retrieval/__init__.py retrieval/crossref.py retrieval/europe_pmc.py retrieval/models.py retrieval/openalex.py
retrieval/pubmed.py retrieval/semantic_scholar.py retrieval/source_manager.py
static/index.html static/research.html static/results.html static/lab.html static/library.html
static/history.html static/keys.html static/styles.css static/app.js static/vendor/marked.min.js static/vendor/purify.min.js static/vendor/3Dmol-min.js
""".split()

# text that only exists in the CURRENT version of each file (old copies lack it)
NEWEST_MARKERS = {
    "ai_router.py": "_groq_skip",
    "retrieval/source_manager.py": "semantic_scholar",
    "retrieval/models.py": "def plausible_year",
    "retrieval/pubmed.py": "ncbi_wait",
    "server.py": "def keep_awake",
    "hypothesis_evaluator.py": "def infer_expected_direction",
    "statistical_analysis.py": "def order_treatment_control",
    "bio_databases.py": "ncbi_wait",
    "evidence_matrix.py": "def _n_compatible",
    "semantic_audit.py": "def is_bibliography_quote",   # v18: a reference entry is not a finding
    "study_classifier.py": "def _species_from_text",
    "deep_reasoner.py": "PROMPT_TIERS",   # v17: an over-long reasoning prompt is trimmed, not abandoned
    "evidence_identity.py": "def online_spelling",
    "claim_extraction.py": "def prefetch_claims",   # v16: several short papers per AI call
    "contradiction_detection.py": "def _result_word",
    "report_auditor.py": "def repair_plan",
    "auto_healer.py": "def refresh_knowledge_base",
    "virtual_lab.py": "def protein_structure",
    "spelling.py": "def question_word",
    "structure.py": "confidence_note",
    "omics.py": "def squeeze_var",
    "prisma_screening.py": "def design_tier",
    "research_planner.py": "fallbacks",
    "static/app.js": "function contradictionLine",
    "database.py": "class _LockedConnection",
    "research_engine.py": "dose_note",
    "report_generator.py": "def _n_independent",
    "autonomous_agent.py": "def objective_met",
    "enhanced_report.py": "def graphs_markdown",
    "reasoning_guards.py": "def species_not_in_sample",
    "pathway_synthesizer.py": "evidence_level",
    "string_db.py": "def find_species",
    "static/index.html": "feature-grid",
    "static/research.html": "optStructures",
    "static/library.html": "libSpell",
    "stats_parsing.py": "def _body_measure",
    "jats_ingest.py": "pmc-release",
    "report_validator.py": "one design per study",
    "clinical_context.py": "def _names_intervention",
    "http_client.py": "def host_paused",
    "provider_health.py": "def record_usage",
    "evidence_quality.py": "def own_exposures",
    "meta_analysis.py": "def forest_plot",
    "methodology_flags.py": "MISIDENTIFIED_LINES",
    "ai_pacing.py": "def wait_turn",
}

# v25 markers, kept as a list: a file can carry markers of several generations and a dict would silently keep one
V25_MARKERS = [
    ("semantic_audit.py", "_NEGATED_CUE"),   # negated verbs are no effect; agent names; reversal of a fall
    ("claim_extraction.py", "def record_guard_stats"),   # verbatim-quote guard + second reader
    ("evidence_identity.py", "def _opposite_prefix"),   # 'repigmentation' is never 'corrected' to 'depigmentation'
    ("stats_parsing.py", "mid-dot decimal"),   # 'p=0\u00b719' was read as p = 0
    ("meta_analysis.py", "def pool_advanced"),   # REML + Hartung-Knapp for several randomised trials
    ("prisma_screening.py", "fulltext_archive"),
    ("research_engine.py", "v25_layers"),
    ("server.py", "server_v25"),
    ("database.py", "extraction_json"),
    ("retrieval/source_manager.py", "Deduplicator"),
    ("regression_tracker.py", "by CONTENT"),
    ("evidence_matrix.py", "risk_of_bias_detail"),
    # v26
    ("research_engine.py", "v26_layers"),                 # quality / causal / plausibility / risk / review layers
    ("server.py", "server_v26"),                          # literature library, your papers, lab agent API
    ("server.py", "def cancel_job"),                      # cancel a running research job
    ("prisma_screening.py", "library_search"),            # local-first search
    ("claim_extraction.py", "passage_select"),            # question-aware passages + pre-read reuse
    ("quote_guard.py", "_SCI"),                           # scientific notation, leading minus, mid-dot
    ("meta_analysis.py", "def subgroup_analysis"),
    ("publication_bias.py", "def registry_gap"),
    ("analysis_tiers.py", "research_value"),
    ("job_queue.py", "_apply_windows_limits"),            # CPU / memory caps on Windows too
    ("core_utils.py", "class JobCancelled"),
    ("static/app.js", "user_collections"),
]

# files with more than one generation of markers (the dict above keeps one per file)
EXTRA_MARKERS = [("semantic_audit.py", "def is_bibliography_quote"), ("claim_extraction.py", "def prefetch_claims"),
                 ("evidence_identity.py", "def online_spelling"), ("stats_parsing.py", "def _body_measure"),
                 ("meta_analysis.py", "def forest_plot"), ("prisma_screening.py", "def design_tier"),
                 ("research_engine.py", "dose_note"), ("server.py", "def keep_awake"),
                 ("database.py", "class _LockedConnection"), ("retrieval/source_manager.py", "semantic_scholar"),
                 ("evidence_matrix.py", "def _n_compatible")]

# files that belong INSIDE retrieval/ but are sometimes dropped in the main folder
RETRIEVAL_ONLY = ["crossref.py", "europe_pmc.py", "models.py", "openalex.py", "pubmed.py",
                  "semantic_scholar.py", "source_manager.py"]

print("=" * 70)
print(f"BioSynthetica setup check   (folder: {ROOT})")
print("=" * 70)

# 1. Python version
if sys.version_info < (3, 10):
    problems.append(f"Python {sys.version.split()[0]} is too old - install Python 3.10 or newer from python.org")

# 2. missing files
for rel in REQUIRED:
    if not (ROOT / rel).is_file():
        hint = ""
        name = Path(rel).name
        found = [p for p in ROOT.rglob(name) if "node_modules" not in p.parts and ".venv" not in p.parts]
        if found:
            hint = f"   <- found at '{found[0].relative_to(ROOT)}', MOVE it to '{rel}'"
        problems.append(f"MISSING  {rel}{hint}")

# 3. old versions
for rel, marker in list(NEWEST_MARKERS.items()) + V25_MARKERS + EXTRA_MARKERS:
    p = ROOT / rel
    if p.is_file():
        try:
            if marker not in p.read_text(encoding="utf-8", errors="ignore"):
                problems.append(f"OLD VERSION  {rel}   <- replace it with the latest file")
        except OSError:
            pass

# 4. retrieval files sitting in the main folder (they shadow the real ones)
for name in RETRIEVAL_ONLY:
    if (ROOT / name).is_file():
        problems.append(f"WRONG FOLDER  {name} is in the main folder - it belongs in retrieval/ (delete the main-folder copy)")

# 5. duplicates like "database (1).py"
for p in ROOT.rglob("*"):
    if p.is_file() and "(" in p.name and ")" in p.name and "node_modules" not in p.parts and ".venv" not in p.parts:
        warnings.append(f"DUPLICATE  {p.relative_to(ROOT)}   <- delete it")

# 5b. scripts that overwrite the program files with OLD code (another tool's "fixer" scripts)
for name in ("make_fixed_zip.py",):
    if (ROOT / name).is_file():
        problems.append(f"OLD-CODE WRITER  {name}   <- delete it: running it replaces current files with older versions "
                        f"(use auto_healer.py / the built-in report_auditor.py, which never edit code)")

# 6. misnamed package file
if (ROOT / "retrieval" / "_init_.py").exists() and not (ROOT / "retrieval" / "__init__.py").exists():
    problems.append("RENAME  retrieval/_init_.py  ->  retrieval/__init__.py  (two underscores each side)")

# 7. packages
for mod, pip_name in [("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("multipart", "python-multipart"),
                      ("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"), ("openai", "openai"),
                      ("google.genai", "google-genai"), ("dotenv", "python-dotenv"), ("pypdf", "pypdf"),
                      ("openpyxl", "openpyxl"), ("chromadb", "chromadb"), ("fastembed", "fastembed")]:
    try:
        importlib.import_module(mod)
    except Exception:
        problems.append(f"PACKAGE NOT INSTALLED  {pip_name}   <- run:  pip install -r requirements.txt")
try:                                              # optional: the Simulator page
    importlib.import_module("tellurium")
except Exception:
    warnings.append("OPTIONAL  tellurium is not installed - the Simulator page needs it:  pip install -r requirements-lab.txt")

# 8. the engine itself imports
if not any(x.startswith(("MISSING", "WRONG", "RENAME", "PACKAGE")) for x in problems):
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    for mod in ("research_engine", "server"):
        try:
            importlib.import_module(mod)
        except Exception as exc:
            problems.append(f"IMPORT ERROR in {mod}: {type(exc).__name__}: {exc}")

print()
if warnings:
    for w in warnings:
        print("  [!]", w)
    print()
if problems:
    for p in problems:
        print("  [X]", p)
    print()
    print(f"NOT READY: {len(problems)} problem(s) above. Fix them, then run  python check_setup.py  again.")
    print("Easiest fix: unzip the latest BioSynthetica zip (BioSynthetica_v26) into a NEW empty folder and use only that.")
    sys.exit(1)
print("  [OK] All files present, all up to date, all packages installed.")
print()
print("READY. Start the app with:   python server.py")
print("Then open in your browser:   http://127.0.0.1:8000")
