"""
core_utils.py
-------------
Small shared helpers. database.py imports `now_iso` from here; this file was
missing from the upload, so ResearchDB could not be imported without it.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    """UTC timestamp in ISO-8601 format (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_table(path):
    """Read a user's data table: CSV / TSV / TXT (separator sniffed, UTF-8 or Latin-1), Excel or JSON."""
    import os
    import pandas as pd
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                with open(path, encoding=enc) as fh:
                    head = fh.read(20000)
            except UnicodeDecodeError:
                continue
            # the separator is chosen from the HEADER line (comma, tab, semicolon); sniffing the whole file turned a
            # one-column file into two empty 'Unnamed' columns
            first = head.splitlines()[0] if head.strip() else ""
            seps = {",": first.count(","), "\t": first.count("\t"), ";": first.count(";")}
            sep = max(seps, key=seps.get) if any(seps.values()) else ("\t" if ext == ".tsv" else ",")
            df = pd.read_csv(path, sep=sep, encoding=enc)
            df = df.loc[:, [c for c in df.columns if not (str(c).startswith("Unnamed:") and df[c].isna().all())]]
            return df
        raise ValueError("unreadable text encoding")
    if ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path)
        except ImportError:
            raise ValueError("old .xls files need the 'xlrd' package - save the sheet as .xlsx or .csv instead")
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported file type {ext} (use CSV, TSV, Excel or JSON)")


def ref_key(ref: object) -> tuple:
    """Natural order for study / claim references: S2 before S10 (a plain sort puts S12 before S9)."""
    import re as _re
    t = str(ref)
    m = _re.match(r"([A-Za-z]*)(\d+)(.*)", t)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (t, 0, "")


def output_path(name: "str | os.PathLike[str]") -> Path:
    """Where a default output file goes. A relative name is put in BIOSYN_OUTPUT_DIR when that is set (the test runner
    sets it, so tests never write research_dossier_*.json / RESEARCH_REPORT.md into the project folder); otherwise the
    name is returned unchanged (relative to the current folder, as before)."""
    p = Path(name)
    d = os.getenv("BIOSYN_OUTPUT_DIR")
    if d and not p.is_absolute():
        root = Path(d).resolve()                 # absolute: calling output_path twice never nests the folder
        root.mkdir(parents=True, exist_ok=True)
        return root / p
    return p


class JobCancelled(BaseException):
    """Raised at the next safe point (a stage change or an AI call) after the user cancels a research job.
    A BaseException (like KeyboardInterrupt) on purpose: the pipeline's many `except Exception` guards - which keep a
    failing optional step from stopping a run - must NOT swallow a cancellation."""
