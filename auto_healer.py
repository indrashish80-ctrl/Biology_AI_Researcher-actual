"""
auto_healer.py  -  your report healer, upgraded
-----------------------------------------------
Reads a finished BioSynthetica report (RESEARCH_REPORT.md / .txt, dossier.json, or a whole job folder),
finds known failure patterns, and repairs what can be repaired SAFELY.

    python auto_healer.py                          # asks for the report (VS Code friendly, as before)
    python auto_healer.py --report RESEARCH_REPORT.md
    python auto_healer.py --report runs/<job>/     # report + dossier.json in one folder
    python auto_healer.py --report r.md --fix      # also repair the knowledge base (backup made first)
    python auto_healer.py --report r.md --fix --rerun   # ... then re-run the question with a repaired plan

What it checks (each from a real report):
  H1  AI cited findings as (S1.E1.F1) and sentences were withheld          -> handled in code now; flags old copies
  H2  question words used as a drug ("Metformin Reduce Breast", "Can regulates")  -> stale claims in the database
  H3  too many statements withheld / rewritten (over-censoring)            -> listed one by one for review
  H4  duplicated research gaps                                             -> stale gaps in the database
  H5  report produced without an AI key (deterministic mode is no longer allowed)
  H6  implausible sample sizes (cell counts shown as n)
  H7  human trial shown as animal evidence (placebo / women in an 'Animal' row)
  H8  program files older than this healer (mixed versions in the folder)
  Q1-Q8  every check of the built-in run auditor (report_auditor.py) when dossier.json is available

What --fix does (never edits program code - a self-rewriting program cannot check its own edits):
  * copies research_knowledge.db to research_knowledge.backup-<time>.db
  * deletes claims made by an older extractor version, so they are re-extracted on the next run
  * clears stored audit verdicts, so every claim is re-checked with the current rules
  * clears stored research gaps and retires old hypotheses (they are rebuilt from the evidence)
  --purge-db deletes the whole knowledge base instead (after the same backup) - your original option.
  --rerun re-runs the question with the repair plan of report_auditor (larger budgets, extra queries).
Everything is written to HEAL_REPORT.md next to the report.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "research_knowledge.db"


# --------------------------------------------------------------------------- reading the run
def load_run(path: Path) -> Dict[str, Any]:
    """Report text + dossier (if any) from a file or a job folder."""
    report, dossier = "", None
    if path.is_dir():
        for name in ("RESEARCH_REPORT.md", "report.md"):
            if (path / name).is_file():
                report = (path / name).read_text(encoding="utf-8", errors="ignore")
                break
        if (path / "dossier.json").is_file():
            dossier = json.loads((path / "dossier.json").read_text(encoding="utf-8", errors="ignore"))
    elif path.suffix.lower() == ".json":
        dossier = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        sib = path.with_name("RESEARCH_REPORT.md")
        report = sib.read_text(encoding="utf-8", errors="ignore") if sib.is_file() else ""
    else:
        report = path.read_text(encoding="utf-8", errors="ignore")
        sib = path.with_name("dossier.json")
        if sib.is_file():
            dossier = json.loads(sib.read_text(encoding="utf-8", errors="ignore"))
    m = re.search(r"\*\*Research question:\*\*\s*(.+)", report) or re.search(r"Research question:\s*(.+)", report)
    question = (m.group(1).strip() if m else "") or (dossier or {}).get("question") or (dossier or {}).get("research_question") or ""
    return {"report": report, "dossier": dossier, "question": question.replace("&lt;", "<").replace("&gt;", ">")}


# --------------------------------------------------------------------------- checks
class DossierAuditor:
    def __init__(self, report_text: str, dossier: Optional[Dict[str, Any]] = None, question: str = ""):
        self.report, self.dossier, self.question = report_text or "", dossier, question

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        r, out = self.report, []

        def add(code, kind, detail, fix, target="knowledge base"):
            out.append({"code": code, "type": kind, "detail": detail, "fix": fix, "target_file": target})

        withheld = re.findall(r"⛔\s*\[statement withheld[^\]]*\]", r)
        if re.search(r"\(S\d+\.E\d+\.F\d+", r) and withheld:
            add("H1", "AI_CITATION_FORMAT_MISMATCH",
                f"{len(withheld)} sentence(s) withheld while the AI cited findings as (S1.E1.F1)",
                "the current deep_reasoner.py converts (S1.E1.F1) to [S1] before checking; if this report is new, "
                "your deep_reasoner.py is an old copy - replace it from the latest package", "deep_reasoner.py")
        stale = re.findall(r"\b(?:[A-Z][a-z]+ (?:Reduce|Reduces|Cause|Causes|Increase|Affect)s? [A-Z][a-z]+|Can regulates|"
                           r"What Are (?:demonstrated|regulates)|Does [A-Z][a-z]+ regulates)\b", r)
        if stale:
            add("H2", "STALE_DATABASE_CACHE", f"question words used as a drug/subject: {sorted(set(stale))[:3]}",
                "--fix deletes claims made by older extractor versions so they are re-extracted")
        reasons = re.findall(r"Guard - [^:]+: ([^\n.]{5,140})", r)
        if len(withheld) >= 3 or len(reasons) >= 5:
            add("H3", "OVER_CENSORING", f"{len(withheld)} withheld marker(s), {len(reasons)} guard note(s) - review list below",
                "each withheld sentence is listed in HEAL_REPORT.md with its reason; send the list if a correct "
                "sentence was withheld", "reasoning_guards.py")
        gaps = re.findall(r"^\s*\d+\.\s+\*\*(?:\[[A-Z]+\]\s*)?(.+?)\*\*", r, re.M)
        dup = sorted({g for g in gaps if gaps.count(g) > 1})
        if dup:
            add("H4", "DUPLICATE_GAPS", f"{len(dup)} research gap(s) repeated: {dup[0][:90]}",
                "--fix clears stored gaps; the current engine also de-duplicates them")
        if re.search(r"Synthesis engine:\s*deterministic \(no AI key\)", r):
            add("H5", "NO_KEY_RUN", "this report was produced without an AI key",
                "the current version refuses to run without a key - add a Groq / Gemini / NVIDIA key and run again",
                "server.py / research_engine.py")
        big = [int(x) for x in re.findall(r"\|\s*(\d{4,7})\s*\|", r) if int(x) > 20000]
        if big:
            add("H6", "IMPLAUSIBLE_N", f"sample size(s) {big[:4]} look like cell counts",
                "--fix re-audits; the current stats_parsing.py never counts cells as n", "stats_parsing.py")
        if re.search(r"Animal \(in vivo\)[^\n]{0,200}(?:placebo|women|patients)", r) or \
                re.search(r"(?:placebo|women)[^\n]{0,120}Animal \(in vivo\) \(L3\)", r):
            add("H7", "HUMAN_AS_ANIMAL", "a human study (placebo / women) appears as animal evidence",
                "--fix clears stored classifications and verdicts so the current classifier re-labels them",
                "study_classifier.py")
        if self.dossier is not None:
            try:
                import report_auditor as ra
                for i in ra.audit_run(self.dossier, r, self.question or "", {"screen_n": 50, "deep_read_k": 10}):
                    add(i["code"], "RUN_AUDIT", i["detail"], i["repair"], "run")
            except Exception as exc:           # an old dossier format must not stop the healer
                add("Q?", "RUN_AUDIT_SKIPPED", f"dossier could not be audited: {exc}", "re-run the question")
        return out

    def withheld_list(self) -> List[str]:
        """Every withheld / rewritten sentence with the reason the guard gave (for H3 review)."""
        items = re.findall(r"⛔\s*\[(statement withheld[^\]]*)\]", self.report)
        items += [f"guard: {x}" for x in re.findall(r"Guard - ([^\n]{5,200}?)(?= Guard - |\n|$)", self.report)]
        return items


def code_is_current() -> List[str]:
    """Runs check_setup.py and returns its OLD VERSION / MISSING lines (mixed versions in the folder)."""
    try:
        res = subprocess.run([sys.executable, str(ROOT / "check_setup.py")], capture_output=True, text=True, timeout=120,
                             cwd=ROOT)
        return [ln.strip(" [X]") for ln in res.stdout.splitlines() if "OLD VERSION" in ln or "MISSING" in ln
                or "OLD-CODE" in ln]
    except Exception as exc:
        return [f"check_setup.py could not run: {exc}"]


# --------------------------------------------------------------------------- repairs (knowledge base only)
class CodeRepairman:
    """Kept for compatibility with the first version. It repairs DATA, never program code."""

    @staticmethod
    def backup_db(db_path: Path = DB_PATH) -> Optional[Path]:
        if not db_path.exists():
            return None
        dst = db_path.with_name(f"research_knowledge.backup-{datetime.now():%Y%m%d-%H%M%S}.db")
        with sqlite3.connect(db_path) as src, sqlite3.connect(dst) as out:
            src.backup(out)                         # consistent copy even while WAL files exist
        return dst

    @staticmethod
    def refresh_knowledge_base(db_path: Path = DB_PATH) -> Dict[str, int]:
        """Stale claims re-extracted, verdicts re-checked, gaps rebuilt, old hypotheses retired."""
        from claim_extraction import EXTRACTOR_VERSION
        stats = {"stale_claims_deleted": 0, "verdicts_cleared": 0, "gaps_cleared": 0, "hypotheses_retired": 0}
        if not db_path.exists():
            return stats
        con = sqlite3.connect(db_path)
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(claims)")}
            if "flags_json" in cols:
                cur = con.execute("DELETE FROM claims WHERE flags_json IS NULL OR flags_json NOT LIKE ?",
                                  (f'%extractor:{EXTRACTOR_VERSION}:%',))
                stats["stale_claims_deleted"] = cur.rowcount
            if "verification_status" in cols:
                cur = con.execute("UPDATE claims SET verification_status=NULL, semantic_audit_json=NULL")
                stats["verdicts_cleared"] = cur.rowcount
            stats["gaps_cleared"] = con.execute("DELETE FROM research_gaps").rowcount
            hcols = {r[1] for r in con.execute("PRAGMA table_info(hypotheses)")}
            if "status" in hcols:
                stats["hypotheses_retired"] = con.execute(
                    "UPDATE hypotheses SET status='superseded' WHERE status NOT IN ('merged','superseded')").rowcount
            for t in ("contradictions", "claim_groups"):
                try:
                    con.execute(f"DELETE FROM {t}")
                except sqlite3.OperationalError:
                    pass
            try:      # frozen study classifications are re-decided with the current classifier
                con.execute("DELETE FROM study_classifications")
            except sqlite3.OperationalError:
                pass
            con.commit()
        finally:
            con.close()
        return stats

    @staticmethod
    def purge_database_cache(db_path: Path = DB_PATH) -> bool:
        """Your original option: delete the whole knowledge base (a backup is made first)."""
        if not db_path.exists():
            return True
        CodeRepairman.backup_db(db_path)
        try:
            for p in (db_path, db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")):
                if p.exists():
                    p.unlink()
            return True
        except Exception as e:
            print(f"Could not delete DB. Close the web server first! Error: {e}")
            return False


def rerun(question: str, dossier: Optional[Dict[str, Any]], report: str) -> Optional[str]:
    """Re-run the question with the run auditor's repair plan (needs an AI key in .env)."""
    import report_auditor as ra
    from research_engine import NoAIKeyError, ResearchEngine
    budgets = {"screen_n": 50, "deep_read_k": 10, "max_papers": 6}
    issues = ra.audit_run(dossier or {}, report, question, budgets) if dossier else []
    plan = ra.repair_plan(issues, budgets, question, 0) or {"screen_n": 85, "deep_read_k": 15, "extra_queries": []}
    try:
        with ResearchEngine() as eng:
            eng._extra_queries = plan.get("extra_queries") or []
            out = ROOT / "RESEARCH_REPORT.healed.md"
            eng.investigate(question, output_report=str(out), use_prisma=True,
                            screen_n=plan.get("screen_n", 85), deep_read_k=plan.get("deep_read_k", 15))
            return str(out)
    except NoAIKeyError as exc:
        print(f"[X] {exc}")
        return None


# --------------------------------------------------------------------------- main
def main() -> None:
    print("=" * 70)
    print("AUTONOMOUS REPORT AUDITOR & HEALER")
    print("=" * 70)
    ap = argparse.ArgumentParser(description="Check a BioSynthetica report and repair what is safely repairable.")
    ap.add_argument("--report", help="RESEARCH_REPORT.md / .txt, dossier.json, or a job folder")
    ap.add_argument("--fix", action="store_true", help="repair the knowledge base (backup first)")
    ap.add_argument("--purge-db", action="store_true", help="delete the whole knowledge base (backup first)")
    ap.add_argument("--rerun", action="store_true", help="re-run the question with the repair plan (needs an AI key)")
    ap.add_argument("--db", default=str(DB_PATH), help="knowledge base path (default: research_knowledge.db)")
    args = ap.parse_args()

    report_file = args.report
    if not report_file:                       # VS Code friendly fallback, as in your version
        print("\n[?] No report file provided via command line.")
        cands = sorted(list(ROOT.glob("*.md")) + list(ROOT.glob("*.txt")) + list((ROOT / "runs").glob("*/RESEARCH_REPORT.md")))
        for t in cands[:20]:
            print(f"  - {t.relative_to(ROOT)}")
        try:
            report_file = input("\nType the report file (or job folder) and press Enter:\n> ").strip()
        except EOFError:
            report_file = ""
    if not report_file:
        print("No file specified. Exiting.")
        sys.exit(1)
    path = Path(report_file)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else ROOT / path
    if not path.exists():
        print(f"\n[X] '{report_file}' not found.")
        sys.exit(1)

    run = load_run(path)
    auditor = DossierAuditor(run["report"], run["dossier"], run["question"])
    anomalies = auditor.detect_anomalies()
    old_code = code_is_current()
    if old_code:
        anomalies.append({"code": "H8", "type": "MIXED_VERSIONS", "detail": "; ".join(old_code[:4]),
                          "fix": "replace these files from the latest package (do not run other 'fixer' scripts)",
                          "target_file": "program folder"})

    print(f"\nReport: {path.name} | question: {run['question'] or 'unknown'}")
    if not anomalies:
        print("[OK] No known failure pattern found.")
    else:
        print(f"\n[!] {len(anomalies)} problem(s):")
        for a in anomalies:
            print(f"  [{a['code']}] {a['type']}: {a['detail']}\n        fix: {a['fix']}")

    actions: List[str] = []
    db_path = Path(args.db)
    if args.purge_db:
        ok = CodeRepairman.purge_database_cache(db_path)
        actions.append("knowledge base deleted (backup kept)" if ok else "knowledge base could NOT be deleted (close the server)")
    elif args.fix:
        b = CodeRepairman.backup_db(db_path)
        st = CodeRepairman.refresh_knowledge_base(db_path)
        actions.append(f"backup: {b.name if b else 'no database found'}; " + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in st.items()))
    if args.rerun and run["question"]:
        out = rerun(run["question"], run["dossier"], run["report"])
        actions.append(f"re-run written to {out}" if out else "re-run not possible (no AI key)")

    lines = [f"# Heal report - {datetime.now():%Y-%m-%d %H:%M}", "", f"**Report:** {path}", f"**Question:** {run['question']}", ""]
    lines += ["## Problems found", ""] + ([f"- **{a['code']} {a['type']}** - {a['detail']}  \n  _Fix:_ {a['fix']}" for a in anomalies]
                                         or ["- none"])
    wl = auditor.withheld_list()
    if wl:
        lines += ["", "## Withheld / rewritten statements (review for over-censoring)", ""] + [f"- {x}" for x in wl]
    lines += ["", "## Actions taken", ""] + ([f"- {x}" for x in actions] or ["- none (run with --fix to repair the knowledge base)"])
    out_path = (path if path.is_dir() else path.parent) / "HEAL_REPORT.md"
    try:
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWritten: {out_path}")
    except OSError:
        print("\n".join(lines))
    for a in actions:
        print(f"  [done] {a}")
    if anomalies and not (args.fix or args.purge_db):
        print("\nRun again with --fix to repair the knowledge base (a backup is made first).")


if __name__ == "__main__":
    main()
