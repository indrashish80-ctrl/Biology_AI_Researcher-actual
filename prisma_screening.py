"""
prisma_screening.py
-------------------
PRISMA-style literature funnel on top of the existing SourceManager:

  Identification   every query x every source (Europe PMC, PMC OA, PubMed, Crossref)
  De-duplication   SourceManager._merge_hits (DOI > PMID > PMCID > title)
  Screening        up to N=50 titles/abstracts, ranked by local semantic similarity
                   (vector_store, CPU) blended with SourceManager.score_candidate
  Eligibility      deep-read the top K=10: JATS XML (PMC / Europe PMC) ->
                   Unpaywall OA PDF -> abstract only
  Included         the K papers handed to the existing extraction pipeline

Every screened record is also upserted into the persistent local vector store,
so the library grows with each investigation and is searchable offline.

ACCOUNTING (ledger): every unique record gets an ID (R0001 ...) and its path through
the funnel is tracked - identified -> duplicate? -> screened -> excluded? (reason) ->
full text sought -> full text obtained? -> excluded at full text? (reason) -> deep-read ->
included. All PRISMA counts are DERIVED from the ledger and then checked against each
other; if any identity fails a PrismaConsistencyError is raised and the report shows
"PRISMA CONSISTENCY ERROR" instead of an inconsistent flow diagram.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from retrieval.models import PaperCandidate
from retrieval.source_manager import SourceManager, _tokenize
from vector_store import LocalVectorStore, cosine_scores

REVIEW_TITLE_RE = re.compile(
    r"\b(review|meta-analysis|systematic review|overview|perspectives? on|recent advances|progress in|"
    r"editorial|commentary|erratum|correction|retraction|protocol for)\b", re.I)


class PrismaConsistencyError(Exception):
    """The PRISMA counts do not reconcile - never publish such a flow diagram."""

    def __init__(self, errors: List[str], counts: Dict[str, Any]):
        super().__init__("PRISMA CONSISTENCY ERROR: " + " | ".join(errors))
        self.errors, self.counts = errors, counts


def validate_prisma(counts: Dict[str, Any], ledger: List[Dict[str, Any]], paper_ids: List[int]) -> List[str]:
    """Every identity that must hold in a correct PRISMA flow. Returns the list of violations."""
    c, errs = counts, []

    def eq(name, left, right):
        if left != right:
            errs.append(f"{name}: {left} != {right}")

    eq("identified = sum(by source)", c["identified"], sum(c["by_source"].values()))
    if "identified_databases" in c:
        eq("identified = database records + citation-chasing records", c["identified"],
           c["identified_databases"] + c.get("identified_citation_chasing", 0))
    eq("identified - duplicates = unique records", c["identified"] - c["duplicates_removed"], c["after_dedup"])
    eq("unique records = screened + not screened", c["after_dedup"], c["screened"] + c["not_screened"])
    eq("screened = excluded at screening + eligible", c["screened"], c["excluded_screening"] + c["eligible"])
    eq("sum(screening exclusion reasons) = excluded at screening", sum(c["exclusion_reasons"].values()), c["excluded_screening"])
    eq("eligible = full text assessed + not assessed (budget)", c["eligible"], c["fulltext_assessed"] + c["eligible_not_deep_read"])
    eq("full text assessed = included + excluded at full text", c["fulltext_assessed"], c["included"] + c["excluded_fulltext"])
    eq("sum(full-text exclusion reasons) = excluded at full text", sum(c["fulltext_exclusion_reasons"].values()), c["excluded_fulltext"])
    eq("deep-read = included", c["deep_read"], c["included"])
    eq("sum(text basis) = included", sum(c["full_text_basis"].values()), c["included"])
    eq("included records = included papers", c["included"], len(paper_ids))
    eq("ledger size = unique records", len(ledger), c["after_dedup"])
    ids = [r["record_id"] for r in ledger]
    if len(set(ids)) != len(ids):
        errs.append("record IDs are not unique")
    pids = [r["paper_id"] for r in ledger if r.get("included")]
    if len(set(pids)) != len(pids):
        errs.append("one paper is included more than once")
    return errs


def candidate_text(c: PaperCandidate) -> str:
    return f"{c.title}. {c.abstract or ''}".strip()


# ---------------------------------------------------------------------------------------------- eligibility rules
_EN_FUNC = {"the", "of", "and", "in", "with", "for", "to", "on", "by", "is", "are", "was", "were", "from", "a", "an", "as", "at",
            "or", "its", "their", "this", "that", "after", "versus", "vs", "among", "between"}
_FOREIGN_FUNC = {"pada", "dan", "yang", "untuk", "dengan", "dari", "ini", "dalam", "terhadap", "menggunakan", "penerapan",
                 "der", "die", "das", "und", "bei", "mit", "von", "zur", "des", "les", "et", "une", "avec", "pour", "dans",
                 "los", "las", "el", "del", "en", "con", "para", "uma", "com", "dos", "nas", "nos", "na", "da", "e", "di",
                 "della", "nel", "sulla", "che", "het", "een", "van", "voor", "och", "och", "på", "av"}
_AI_METHOD = re.compile(r"\b(?:convolutional neural networks?|\bCNNs?\b|efficientnet\w*|resnet\w*|vgg-?\d+|yolo\w*|deep learning|"
                        r"machine learning|transfer learning|image classification|image recognition|classification of [\w\s]{0,40}images|"
                        r"computer vision|neural network model|segmentation model|dataset of [\w\s]{0,30}images|smear images)\b", re.I)
_AI_QUESTION = re.compile(r"\b(?:diagnos\w*|classif\w*|detect\w*|machine learning|deep learning|artificial intelligence|\bAI\b|"
                          r"predict\w*|imaging|image\w*|screening test)\b", re.I)
_TREAT_CONCEPT = re.compile(r"\b(?:treat\w*|therap\w*|remission|cur(?:e|ed|es|ative|able)|surviv\w*|relapse\w*|respon\w*|outcome\w*|"
                            r"trial\w*|transplant\w*|chemotherap\w*|immunotherap\w*|regimen\w*|efficacy|drug\w*|inhibitor\w*|"
                            r"dose\w*|administ\w*|car-?t|stem cell|prognos\w*|mortality)\b", re.I)
_FEATURE_CONCEPT = re.compile(r"\b(?:symptom\w*|signs?|present(?:ed|ing|ation)\w*|manifest\w*|clinical (?:features?|picture|course)|"
                              r"lesions?|macules?|patches|depigment\w*|hypopigment\w*|diagnos\w*|dermoscop\w*|wood'?s lamp|"
                              r"differential|case report|case series|phenotype)\b", re.I)
_CAUSE_CONCEPT = re.compile(r"\b(?:risk\w*|exposure\w*|exposed|caus\w*|associat\w*|incidence|cohort|case-control|etiolog\w*|"
                            r"aetiolog\w*|mutation\w*|genetic\w*|carcinogen\w*|leukemogen\w*|susceptib\w*|predispos\w*|"
                            r"radiation|benzene|smok\w*|infection\w*|germline|translocation\w*|origin)\b", re.I)


def is_non_english(title: str, abstract: str = "") -> bool:
    """Title/abstract not in English (the claim audit reads English text only)."""
    t = f"{title or ''} {abstract or ''}"
    if re.search(r"[\u0400-\u04ff\u0590-\u06ff\u0900-\u0dff\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", t):
        return True
    words = re.findall(r"[a-zà-ÿ]+", (title or "").lower())
    if len(words) < 4:
        return False
    en = sum(w in _EN_FUNC for w in words)
    foreign = sum(w in _FOREIGN_FUNC and w not in _EN_FUNC for w in words)
    return foreign >= 2 and foreign > en


def eligibility(cand: PaperCandidate, frame: Dict[str, Any]) -> Optional[str]:
    """Reason to EXCLUDE a record at title/abstract screening, or None when it is eligible.
    Rules: English; the question's disease is named; not an off-question computational method paper;
    for cure / treatment / prognosis questions a treatment-outcome concept, for causes questions a risk concept."""
    from evidence_identity import paper_on_question
    title, abstract = cand.title or "", cand.abstract or ""
    text = f"{title} {abstract}"
    from semantic_audit import is_retracted
    if is_retracted(title, abstract):
        return "retracted / withdrawn publication (retracted results are not evidence)"
    if is_non_english(title, abstract):
        return "not in English (the evidence audit reads English text only)"
    if frame.get("disease"):
        on = paper_on_question(title, abstract, frame)
        if not on.get("on_question"):
            return f"does not concern {frame['disease']} ({on.get('detail')})"
    q = frame.get("question") or ""
    if _AI_METHOD.search(text) and not _AI_QUESTION.search(q):
        return "computational method paper (image / data classification) - reports no biological or clinical outcome for this question"
    intent = frame.get("intent")
    names_exposure = any(re.search(r"\b" + re.escape(e.split()[0][:max(4, len(e.split()[0]) - 1)]), text, re.I)
                         for e in frame.get("exposures") or [] if e.strip())
    medical = frame.get("domain", "human_medicine" if frame.get("disease") else "general_biology") == "human_medicine"
    # a two-part question ('can it be cured AND what are its symptoms'): a record answering the other part is eligible
    answers_features = "clinical_features" in (frame.get("intents") or []) and _FEATURE_CONCEPT.search(text)
    if intent in ("cure", "treatment", "prognosis") and frame.get("disease") and medical and not _TREAT_CONCEPT.search(text) \
            and not (intent == "treatment" and names_exposure) and not answers_features:   # own drug = a treatment concept
        return "does not address treatment or outcomes (cure / remission / survival / response)"
    # the risk-factor vocabulary is clinical; a basic-biology 'what causes X' question (biofilm formation, root bending)
    # is answered by papers that never say 'risk' or 'exposure', so the filter applies to disease questions only
    if intent == "etiology" and frame.get("disease") and medical and not _CAUSE_CONCEPT.search(text):
        return "does not address causes or risk factors"
    return None


class PrismaScreener:
    def __init__(
        self,
        source_manager: SourceManager,
        vector_store: Optional[LocalVectorStore] = None,
        unpaywall: Any = None,
        log: Optional[Callable[[str], None]] = None,
        exclude_reviews: bool = True,
        sources: Optional[List[str]] = None,
        citation_chasing: bool = True,
        abstract_only: bool = False,
        user_collections: Optional[List[str]] = None,
    ):
        self.mgr = source_manager
        self.abstract_only = abstract_only      # quick mode: read abstracts only (no full-text / PDF downloads)
        self.vs = vector_store
        self.unpaywall = unpaywall
        self.log = log or print
        self.exclude_reviews = exclude_reviews
        self.sources = [s for s in (sources or self.mgr.ALLOWED_SOURCES) if s in self.mgr.ALLOWED_SOURCES]
        # v26: library collections searched first ('literature' + the user's own collections when asked for)
        self.collections = ["literature"] + [c for c in (user_collections or []) if str(c).startswith("user:")]
        self.citation_chasing = citation_chasing and "semantic_scholar" in self.sources

    # ------------------------------------------------------------------ identification
    def identify(self, queries: List[str], per_source_limit: int, question: Optional[str] = None) -> Dict[str, Any]:
        registry: Dict[str, PaperCandidate] = {}
        by_source: Dict[str, int] = {s: 0 for s in self.sources}
        raw_total = 0
        # v26: the local library first; the coverage rule decides whether the external sources are needed at all
        local = None
        try:
            import library_search
            local = library_search.identify(question or " ".join(queries), queries, k=max(60, per_source_limit * 6),
                                            collections=self.collections)
        except Exception as exc:
            self.log(f"  [Library] local search skipped: {type(exc).__name__}: {str(exc)[:100]}")
        registered: Dict[str, Dict[str, Any]] = {}
        if local is not None:
            for c in local["candidates"]:
                nct = (c.raw_metadata or {}).get("nct")
                if nct and not c.pmid and not c.doi:
                    registered[nct] = {"nct": nct, "title": c.title, "summary": (c.abstract or "")[:600], "source": "library"}
            local["candidates"] = [c for c in local["candidates"]
                                   if not ((c.raw_metadata or {}).get("nct") and not c.pmid and not c.doi)]
            by_source["local_library"] = len(local["candidates"])
            raw_total += len(local["candidates"])
            self.mgr._merge_hits(registry, local["candidates"])
            d = local["decision"]
            self.log(f"[Library] {d['local_hits']} local records ({d['strong_hits']} strong matches; snapshot "
                     f"{d.get('snapshot')}). External search: {'yes' if d['use_online'] else 'SKIPPED'} - "
                     + "; ".join(d["reasons"]))
        online = [s for s in self.sources if s != "clinicaltrials"] if (local is None or local["decision"]["use_online"]) else []
        # v26: registered studies (ClinicalTrials.gov + registry records in the library) are listed separately, never
        # screened as evidence - they show ongoing work and completed trials without a result
        if "clinicaltrials" in self.sources and (local is None or local["decision"]["use_online"]):
            try:
                for q in queries[:3]:
                    for r in self.mgr.clinicaltrials.search(q, max_results=min(50, per_source_limit * 2)):
                        registered.setdefault(r["nct"], {"nct": r["nct"], "title": r["title"], "summary": (r.get("abstract") or "")[:600],
                                                         "source": "ClinicalTrials.gov", "year": r.get("year"),
                                                         "registry": r.get("registry"), "funding": r.get("funding")})
            except Exception as exc:
                self.log(f"  [ClinicalTrials.gov] skipped: {type(exc).__name__}: {str(exc)[:100]}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = {}
            for q in queries:
                for src in online:
                    futs[ex.submit(self.mgr._query_source, src, q, per_source_limit)] = src
            if "openalex" in online and question:
                # one vector search on OpenAlex's side for the question as asked (1 req/s limit)
                futs[ex.submit(self.mgr._query_source, "openalex_semantic", question,
                               min(50, per_source_limit * 2))] = "openalex"
            for fut in concurrent.futures.as_completed(futs):
                hits = fut.result() or []
                by_source[futs[fut]] += len(hits)
                raw_total += len(hits)
                self.mgr._merge_hits(registry, hits)
        unique = [c for c in registry.values() if c.title]
        return {"records": unique, "raw_total": raw_total, "by_source": by_source,
                "local_library": (local or {}).get("decision"), "online_sources": online,
                "registered_studies": list(registered.values())}

    # ------------------------------------------------------------------ screening
    def screen(self, question: str, records: List[PaperCandidate], screen_n: int,
               min_eligible: int = 0) -> List[Dict[str, Any]]:
        q_tokens = _tokenize(question)
        lexical = [SourceManager.score_candidate(c, q_tokens) for c in records]
        embedder = self.vs.embedder if self.vs else None
        semantic = cosine_scores(embedder, question, [candidate_text(c) for c in records]) if embedder else [0.0] * len(records)

        # put both scores on [0,1] then blend: meaning matters more than keyword overlap
        def norm(xs):
            # relative to the best record (min-max scaling forced the weakest record of a small pool to 0 and excluded
            # a relevant phase III trial as 'off-topic')
            hi = max(xs, default=0)
            return [0.5 if hi <= 0 else max(0.0, x) / hi for x in xs]

        lex_n, sem_n = norm(lexical), norm(semantic)
        # v25: keyword (BM25 + biomedical synonyms) and PICO coverage join the semantic score; BIOSYN_HYBRID=0 restores
        # the v24 blend. A failure here never stops screening.
        hyb = None
        if os.getenv("BIOSYN_HYBRID", "1") != "0" and records:
            try:
                from hybrid_retrieval import rerank_records
                hyb = rerank_records(question, records, semantic)
            except Exception as exc:
                print(f"  [Hybrid notice] keyword reranking skipped ({type(exc).__name__}: {str(exc)[:80]})")
                hyb = None
        rows = []
        for i, (c, lx, sm, lxn, smn) in enumerate(zip(records, lexical, semantic, lex_n, sem_n)):
            if hyb:
                h = hyb[i]
                c.relevance_score = round(0.45 * smn + 0.15 * lxn + 0.15 * h["bm25_norm"] + 0.25 * h["pico"], 4)
            else:
                c.relevance_score = round(0.6 * smn + 0.4 * lxn, 4)
            row = {"cand": c, "lexical": round(lx, 3), "semantic": round(sm, 4), "score": c.relevance_score}
            if hyb:
                row.update({"bm25": hyb[i]["bm25_norm"], "pico": hyb[i]["pico"], "pico_matched": hyb[i]["matched"]})
            rows.append(row)
        rows.sort(key=lambda r: r["score"], reverse=True)

        # records with an abstract outrank title-only records of similar relevance (they can actually be read)
        rows.sort(key=lambda r: r["score"] + (0.05 if r["cand"].abstract else 0.0), reverse=True)
        screened = rows[:screen_n]
        for r in rows[screen_n:]:
            r["decision"], r["reason"] = "not_screened", "beyond screening budget (lower-ranked record)"
        best = screened[0]["score"] if screened else 1.0
        from evidence_identity import question_frame
        frame = question_frame(question)

        # rank floor: the best-ranked records that pass every eligibility rule are kept whatever their score relative to
        # the single best record (one near-perfect hit must not push 40 decent papers below a relative cut-off)
        # The floor is rank-based: the top N records (N = the deep-read budget or a third of the screen, at least 25)
        # are judged only on eligibility + basic keyword relevance (they share a content word with the question),
        # never on their score relative to record #1.
        protect = max(int(min_eligible or 0), 25, screen_n // 3)
        rank_of = {id(r): i for i, r in enumerate(rows)}

        def keyword_relevant(r) -> bool:
            return r["lexical"] > 0 or r["score"] >= 0.12 * best

        def decide(r):
            c: PaperCandidate = r["cand"]
            why = eligibility(c, frame)
            if self.exclude_reviews and REVIEW_TITLE_RE.search(c.title or ""):
                r["decision"], r["reason"] = "excluded", "secondary literature (review / editorial / correction)"
            elif why:
                r["decision"], r["reason"] = "excluded", why
            elif (rank_of[id(r)] >= protect and r["score"] < 0.35 * best) or \
                    (rank_of[id(r)] < protect and not keyword_relevant(r)):
                r["decision"], r["reason"] = "excluded", "off-topic on title/abstract screen"
            elif not c.abstract and r["semantic"] <= 0:
                r["decision"], r["reason"] = "excluded", "no abstract and no title relevance"
            else:
                r["decision"], r["reason"] = "eligible", "relevant on title/abstract screen" + ("" if c.abstract else " (title only)")
        for r in screened:
            decide(r)
        if min_eligible:
            # Unusual questions: the strict screen can leave fewer eligible records than the deep-read budget while
            # many retrieved records were never looked at. Screen further down the ranking first ...
            n_el = sum(1 for r in screened if r["decision"] == "eligible")
            extra = 0
            for r in rows[screen_n:]:
                if n_el >= min_eligible or extra >= max(screen_n, 2 * min_eligible):
                    break
                decide(r)
                r["reason"] += " (screened beyond the budget: too few eligible records)"
                extra += 1
                n_el += r["decision"] == "eligible"
            # ... then re-admit records excluded only for lacking a keyword concept or for a low relative score
            # (they still name the disease, are in English and are primary literature)
            if n_el < min_eligible:
                soft = [r for r in rows if r.get("decision") == "excluded" and r["score"] >= 0.15 * best and r["cand"].abstract
                        and (r["reason"].startswith("does not address") or r["reason"] == "off-topic on title/abstract screen")]
                for r in sorted(soft, key=lambda x: -x["score"])[: min_eligible - n_el]:
                    r["decision"] = "eligible"
                    r["reason"] = "relevant on a relaxed screen (too few records met the strict criteria; " + r["reason"] + ")"
        return rows

    # ------------------------------------------------------------------ full-text acquisition
    def _direct_oa_pdf(self, cand: PaperCandidate, pdf_dir: Path) -> Optional[Path]:
        """Open-access PDF link already supplied by OpenAlex / Semantic Scholar (no email needed)."""
        url = (cand.raw_metadata or {}).get("oa_pdf_url")
        if not url or not str(url).startswith(("http://", "https://")):
            return None
        import re as _re
        pdf_dir.mkdir(parents=True, exist_ok=True)
        out = pdf_dir / (_re.sub(r"[^a-zA-Z0-9_\-]", "_", cand.doi or cand.dedup_key)[:120] + ".pdf")
        if out.exists() and out.stat().st_size > 1000:
            return out
        try:
            data = self.mgr.http.get_bytes(url, max_bytes=40 * 1024 * 1024)
        except Exception:
            return None
        if data[:5] == b"%PDF-":
            out.write_bytes(data)
            return out
        return None

    def _acquire(self, cand: PaperCandidate, pdf_dir: Path) -> Dict[str, Any]:
        """Network-only step (runs in worker threads):
        JATS XML -> OA PDF link from OpenAlex/Semantic Scholar -> Unpaywall PDF."""
        out = {"xml": None, "pdf": None, "unpaywall": None, "pdf_source": None}
        try:
            if cand.pmcid or cand.has_full_text_xml or cand.pmid:
                out["xml"] = self.mgr.download_xml(cand)
        except Exception as exc:
            self.log(f"  [Full text] XML fetch failed for {cand.dedup_key}: {str(exc)[:100]}")
        if not out["xml"]:
            # v25: a PDF archived by an earlier run is reused - every paper is downloaded once
            try:
                import fulltext_archive as fa
                arch = fa.lookup(fa.keys_for(cand.doi, cand.pmid, cand.pmcid, cand.dedup_key), kind="pdf")
                if arch:
                    out["pdf"], out["pdf_source"], out["archived"] = Path(arch["path"]), "oa_pdf", arch
            except Exception:
                pass
        if not out["xml"] and not out["pdf"]:
            out["pdf"] = self._direct_oa_pdf(cand, pdf_dir)
            if out["pdf"]:
                out["pdf_source"] = "oa_pdf"
        if not out["xml"] and not out["pdf"] and cand.doi and self.unpaywall is not None and self.unpaywall.enabled:
            info = self.unpaywall.resolve(cand.doi)
            out["unpaywall"] = info
            if info and info.get("best_pdf_url"):
                out["pdf"] = self.unpaywall.download_pdf(info, pdf_dir)
                if out["pdf"]:
                    out["pdf_source"] = "unpaywall_pdf"
        return out

    @staticmethod
    def design_tier(cand: Any, human_focus: bool) -> int:
        """Deep-read priority by level of evidence for a question about PEOPLE: randomised / clinical trials first,
        then observational human studies, then animal, then organoid / cell studies, reviews last. For other questions
        every design has the same tier (the screening score alone decides)."""
        if not human_focus:
            return 0
        from study_classifier import classify_study
        try:
            et = classify_study({"title": cand.title or "", "abstract": cand.abstract or ""}).get("evidence_type")
        except Exception:
            return 3
        return {"human_clinical": 0, "human_observational": 1, "case_report": 2, "animal": 2,
                "experimental_disease_model": 2, "organoid_ex_vivo": 3, "cell_culture": 3, "review_meta": 5}.get(et, 4)

    def deep_read(self, db: Any, eligible: List[Dict[str, Any]], deep_k: int,
                  research_question_id: int, question: Optional[str] = None) -> Dict[str, Any]:
        pass

        # Records that can actually be read first (abstract, PMC / open full text); records with no retrievable text
        # are replaced by the next eligible ones until K papers are read (or 4K records were tried). For a question about
        # people the budget goes to human trials and human observational studies before animal and cell studies (the
        # budget used to fill up with cell papers while 14 eligible human studies were never read).
        human_focus = False
        if question:
            try:
                from evidence_identity import question_frame
                fr = question_frame(question)
                human_focus = fr.get("organism_restriction") == "human" or \
                    (fr.get("domain") == "human_medicine" and fr.get("organism_restriction") in (None, "", "human"))
            except Exception:
                human_focus = False
        def full_text_known(c):            # a free full text can be checked (methods, statistics, subgroups) - an abstract cannot
            return bool(c.pmcid or getattr(c, "has_full_text_xml", False) or (c.raw_metadata or {}).get("oa_pdf_url")
                        or getattr(c, "is_open_access", False))
        ordered = sorted(eligible, key=lambda r: (0 if (r["cand"].abstract or r["cand"].pmcid
                                                        or (r["cand"].raw_metadata or {}).get("oa_pdf_url")) else 1,
                                                  self.design_tier(r["cand"], human_focus),
                                                  # within a design tier, papers whose full text is free come first
                                                  0 if (self.abstract_only or full_text_known(r["cand"])) else 1))
        pdf_dir = self.mgr.raw_dir / "oa_pdf"
        paper_ids: List[int] = []
        basis: Dict[int, str] = {}
        oa_links: Dict[int, Any] = {}
        first_record_for_pid: Dict[int, str] = {}
        pos, limit = 0, min(len(ordered), max(deep_k * 4, deep_k + 6))
        while pos < limit and len(paper_ids) < deep_k:
            need = deep_k - len(paper_ids)
            if need <= 0:
                break
            batch = ordered[pos: min(limit, pos + need + max(3, need // 2))]
            pos += len(batch)
            if self.abstract_only:
                acquired = [{"xml": None, "pdf": None, "unpaywall": None, "pdf_source": None} for _ in batch]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                    acquired = list(ex.map(lambda r: self._acquire(r["cand"], pdf_dir), batch))
            for r, acq in zip(batch, acquired):
                if len(paper_ids) >= deep_k:
                    r["ft_stage"], r["ft_reason"] = "not_assessed", "eligible; beyond deep-read budget"
                    continue
                self._read_one(db, r, acq, research_question_id, paper_ids, basis, oa_links, first_record_for_pid)
        return {"paper_ids": paper_ids, "basis": basis, "unpaywall": oa_links}

    def _read_one(self, db: Any, r: Dict[str, Any], acq: Dict[str, Any], research_question_id: int, paper_ids: List[int],
                  basis: Dict[int, str], oa_links: Dict[int, Any], first_record_for_pid: Dict[int, str]) -> None:
        from pdf_ingest import ingest_pdf
        if True:
            c: PaperCandidate = r["cand"]
            if not acq["xml"] and not acq["pdf"] and not c.abstract:
                r["ft_stage"], r["ft_reason"] = "excluded", "no retrievable text (no abstract, no open full text)"
                return
            src_q = "; ".join(c.source_queries) or c.source_db
            pid, how = None, "abstract"
            if acq["xml"]:
                c.local_xml_path = str(acq["xml"])
                try:
                    pid, how = self.mgr.ingest_candidate(db, c, research_question_id), "jats_xml"
                except Exception as exc:   # malformed XML from the source must not kill the run
                    self.log(f"  [Full text] XML unreadable ({str(exc)[:80]}); falling back.")
                    pid, how = None, "abstract"
                    try:
                        Path(acq["xml"]).unlink(missing_ok=True)
                    except OSError:
                        pass
            if not pid and acq["pdf"]:
                paper = {"title": c.title, "authors": ", ".join(c.authors), "journal": c.journal,
                         "year": str(c.year) if c.year else None, "doi": c.doi, "pmid": c.pmid, "pmcid": c.pmcid,
                         "abstract": c.abstract, "full_text": None, "is_open_access": True}
                try:
                    pid = ingest_pdf(db, acq["pdf"], paper, research_question_id, source_query=src_q)
                except Exception as exc:
                    self.log(f"  [Unpaywall] PDF parse failed ({str(exc)[:80]}); using abstract.")
                if pid:
                    try:
                        from metadata_authority import record_candidate
                        record_candidate(db, c)
                    except Exception:
                        pass
                    how = acq.get("pdf_source") or "unpaywall_pdf"
                    db.mark_processed("paper", c.dedup_key, research_question_id)
            if not pid:
                c.local_xml_path = None
                pid = self.mgr.ingest_candidate(db, c, research_question_id)
            # JATS/PDF ingest may lack year/journal/authors: backfill from the search metadata
            try:
                db._conn.execute(
                    "UPDATE papers SET year=COALESCE(year, ?), journal=COALESCE(NULLIF(journal,''), ?), "
                    "authors=COALESCE(NULLIF(authors,''), ?), abstract=COALESCE(NULLIF(abstract,''), ?) WHERE id=?",
                    (str(c.year) if c.year else None, c.journal, ", ".join(c.authors) or None, c.abstract, pid))
                db._conn.commit()
            except Exception:
                pass
            if pid in first_record_for_pid:
                # two search records resolved to the same stored paper: count it once
                r["ft_stage"] = "excluded"
                r["ft_reason"] = "duplicate of an already included paper (detected after full-text retrieval)"
                r["duplicate_of"] = first_record_for_pid[pid]
                return
            # v25: archive the full text once, with its licence; the run database records how this paper was read
            try:
                import fulltext_archive as fa
                rec = None
                if how == "jats_xml" and acq.get("xml"):
                    rec = fa.store_candidate_file(c, acq["xml"], "jats_xml", "PMC / Europe PMC")
                elif acq.get("pdf") and how != "abstract":
                    rec = acq.get("archived") or fa.store_candidate_file(c, acq["pdf"], "pdf", acq.get("pdf_source") or "pdf",
                                                                           acq.get("unpaywall"))
                fa.record_basis(db, pid, how, rec)
            except Exception:
                pass
            r["paper_id"], r["text_basis"], r["ft_stage"], r["ft_reason"] = pid, how, "included", "deep-read"
            first_record_for_pid[pid] = r.get("record_id") or str(pid)
            if acq.get("unpaywall"):
                oa_links[pid] = acq["unpaywall"]
            paper_ids.append(pid)
            basis[pid] = how
            self.log(f"  [Deep-read] {how:<14} {c.title[:80]}")

    # ------------------------------------------------------------------ full run
    def run(self, db: Any, queries: List[str], question: str, research_question_id: int,
            screen_n: int = 50, deep_k: int = 10) -> Dict[str, Any]:
        screen_n, deep_k = max(1, int(screen_n)), max(1, int(deep_k))
        per_source = max(5, min(25, math.ceil(screen_n * 1.6 / max(1, len(queries)) / 2)))
        self.log(f"[PRISMA] Identification: {len(queries)} queries x {len(self.sources)} sources "
                 f"({', '.join(self.sources)}; up to {per_source} each)...")
        ident = self.identify(queries, per_source, question=question)
        records = ident["records"]
        self.log(f"[PRISMA] {ident['raw_total']} records identified -> {len(records)} after de-duplication.")

        rows = self.screen(question, records, screen_n, min_eligible=deep_k)

        # Citation chasing (Semantic Scholar citation graph): references of, and papers citing,
        # the 3 best-ranked eligible records join the pool, then everything is re-screened.
        chased_added = 0
        if self.citation_chasing and ident.get("online_sources", self.sources):
            seeds = [r["cand"] for r in rows if r.get("decision") == "eligible"][:3]
            if seeds:
                self.log(f"[PRISMA] Citation chasing on the top {len(seeds)} records (references + citing papers)...")
                registry = {c.dedup_key: c for c in records}
                before = len(registry)
                for s in seeds:
                    try:
                        hits = self.mgr.semantic_scholar.citation_neighbours(s, limit=15)
                    except Exception as exc:
                        self.log(f"  [Citation chasing] skipped: {str(exc)[:100]}")
                        continue
                    ident["raw_total"] += len(hits)
                    ident["by_source"]["citation_chasing"] = ident["by_source"].get("citation_chasing", 0) + len(hits)
                    self.mgr._merge_hits(registry, hits)
                records = [c for c in registry.values() if c.title]
                chased_added = len(registry) - before
                self.log(f"[PRISMA] Citation chasing added {chased_added} new unique records.")
                if chased_added:
                    rows = self.screen(question, records, screen_n, min_eligible=deep_k)
        for i, r in enumerate(rows, 1):           # unique, stable-within-run record IDs
            r["record_id"] = f"R{i:04d}"
        screened = [r for r in rows if r["decision"] != "not_screened"]
        eligible = [r for r in screened if r["decision"] == "eligible"]
        excluded = [r for r in screened if r["decision"] == "excluded"]
        self.log(f"[PRISMA] Screened {len(screened)} abstracts: {len(eligible)} eligible, {len(excluded)} excluded.")

        # Grow the persistent local library with every screened record.
        if self.vs is not None and records:
            try:
                self.vs.upsert({
                    "id": r["cand"].dedup_key, "text": candidate_text(r["cand"]),
                    "metadata": {"title": r["cand"].title, "year": r["cand"].year, "journal": r["cand"].journal,
                                 "doi": r["cand"].doi, "pmid": r["cand"].pmid, "pmcid": r["cand"].pmcid,
                                 "url": r["cand"].url, "source_db": r["cand"].source_db,
                                 "is_open_access": bool(r["cand"].is_open_access),
                                 "question": question[:300], "decision": r["decision"]},
                } for r in rows)
            except Exception as exc:
                self.log(f"  [VectorStore] index update skipped: {str(exc)[:120]}")

        self.log(f"[PRISMA] Deep-reading the top {min(deep_k, len(eligible))} eligible papers (full text where legal & open)...")
        dr = self.deep_read(db, eligible, deep_k, research_question_id, question=question) if eligible else \
            {"paper_ids": [], "basis": {}, "unpaywall": {}}

        # ---- ledger: one entry per unique record; every count below is derived from it
        from study_classifier import classify_study
        ledger = []
        for r in rows:
            c: PaperCandidate = r["cand"]
            ft = r.get("ft_stage") or ("not_assessed" if r["decision"] == "eligible" else None)
            try:
                guess = classify_study({"title": c.title, "abstract": c.abstract})["evidence_type"]
            except Exception:
                guess = "unclear"
            ledger.append({
                "design_guess": guess,
                "record_id": r["record_id"], "dedup_key": c.dedup_key, "title": c.title, "source_db": c.source_db,
                "doi": c.doi, "pmid": c.pmid, "pmcid": c.pmcid,
                "screened": r["decision"] != "not_screened", "screen_decision": r["decision"], "screen_reason": r["reason"],
                "fulltext_stage": ft, "fulltext_reason": r.get("ft_reason"),
                "text_basis": r.get("text_basis") if ft == "included" else None,
                "duplicate_of": r.get("duplicate_of"),
                "included": ft == "included", "paper_id": r.get("paper_id") if ft == "included" else None,
            })

        def tally(key, rows_):
            out: Dict[str, int] = {}
            for x in rows_:
                out[x[key]] = out.get(x[key], 0) + 1
            return out

        L_screened = [x for x in ledger if x["screened"]]
        L_excl = [x for x in L_screened if x["screen_decision"] == "excluded"]
        L_elig = [x for x in L_screened if x["screen_decision"] == "eligible"]
        L_ft_excl = [x for x in L_elig if x["fulltext_stage"] == "excluded"]
        L_incl = [x for x in L_elig if x["fulltext_stage"] == "included"]
        chase_raw = ident["by_source"].get("citation_chasing", 0)
        counts = {
            "identified": ident["raw_total"],
            "identified_databases": ident["raw_total"] - chase_raw,
            "identified_citation_chasing": chase_raw,
            "by_source": ident["by_source"],
            "duplicates_removed": ident["raw_total"] - len(ledger),
            "after_dedup": len(ledger),
            "via_citation_chasing": chased_added,
            "sources_used": (["local_library"] if ident.get("local_library") else []) + list(ident.get("online_sources", self.sources)),
            "local_library": ident.get("local_library"),
            "not_screened": len(ledger) - len(L_screened),
            "screened": len(L_screened),
            "excluded_screening": len(L_excl),
            "exclusion_reasons": tally("screen_reason", L_excl),
            "eligible": len(L_elig),
            "fulltext_assessed": len(L_ft_excl) + len(L_incl),
            "excluded_fulltext": len(L_ft_excl),
            "fulltext_exclusion_reasons": tally("fulltext_reason", L_ft_excl),
            "eligible_not_deep_read": sum(1 for x in L_elig if x["fulltext_stage"] == "not_assessed"),
            "deep_read": len(L_incl),
            "full_text_basis": tally("text_basis", L_incl),
            "included": len(L_incl),
        }
        errors = validate_prisma(counts, ledger, dr["paper_ids"])
        consistency = {"ok": not errors, "errors": errors, "checked_identities": 15}
        if errors:
            self.log("[PRISMA] " + str(PrismaConsistencyError(errors, counts)))
        table = []
        for i, r in enumerate(rows[: max(screen_n, 1)], 1):
            c: PaperCandidate = r["cand"]
            table.append({
                "record_id": r["record_id"],
                "rank": i, "title": c.title, "year": c.year, "journal": c.journal, "doi": c.doi, "pmid": c.pmid,
                "pmcid": c.pmcid, "url": c.url, "source_db": c.source_db, "has_abstract": bool(c.abstract),
                "semantic": r["semantic"], "lexical": r["lexical"], "score": r["score"],
                "decision": ("included" if r.get("ft_stage") == "included" else
                             "excluded (full text)" if r.get("ft_stage") == "excluded" else r["decision"]),
                "reason": r.get("ft_reason") if r.get("ft_stage") == "excluded" else r["reason"],
                "paper_id": r.get("paper_id") if r.get("ft_stage") == "included" else None,
                "text_basis": r.get("text_basis") if r.get("ft_stage") == "included" else None,
                "citations": (c.raw_metadata or {}).get("cited_by_count"),
                "tldr": (c.raw_metadata or {}).get("tldr"),
                "via": (c.raw_metadata or {}).get("via"),
            })
        return {"counts": counts, "consistency": consistency, "ledger": ledger, "screening_table": table,
                "registered_studies": ident.get("registered_studies") or [],
                "paper_ids": dr["paper_ids"],
                "text_basis": dr["basis"], "unpaywall": dr["unpaywall"],
                "vector_store": self.vs.info() if self.vs else None}


def adopt_existing_claims(db: Any, paper_ids: List[int], research_question_id: int) -> int:
    """
    ClaimExtractor skips papers that already have claims (from an earlier question).
    Copy those verified claims into this research question so the matrix,
    contradiction detection and report see them. Returns number copied.
    """
    copied = 0
    cols = [r[1] for r in db._conn.execute("PRAGMA table_info(claims)").fetchall() if r[1] not in ("id",)]
    for pid in paper_ids:
        have = db._conn.execute("SELECT COUNT(*) FROM claims WHERE paper_id=? AND research_question_id=?",
                                (pid, research_question_id)).fetchone()[0]
        if have:
            continue
        rows = db._conn.execute(
            "SELECT * FROM claims WHERE paper_id=? AND (research_question_id IS NULL OR research_question_id<>?) "
            "AND research_question_id = (SELECT research_question_id FROM claims WHERE paper_id=? ORDER BY id DESC LIMIT 1)",
            (pid, research_question_id, pid)).fetchall()
        for row in rows:
            d = dict(row)
            d["research_question_id"] = research_question_id
            db._conn.execute(f"INSERT INTO claims ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                             tuple(d.get(c) for c in cols))
            copied += 1
    db._conn.commit()
    return copied
