"""
spelling.py
-----------
"Did you mean ...?" for biology questions, tuned for this domain.

A general-purpose speller picks the most common English word ('metfromin' -> 'merbromin', 'reduse' -> 'refuse').
Here every candidate is ranked with a DOMAIN prior:

  vocabulary  the program's own biomedical terms (disease names, endpoints, drug tables, question words) +
              the words of every paper already read on this computer (it learns your field) +
              an English word-frequency list (pyspellchecker, MIT licence) when installed
  candidates  Damerau-Levenshtein distance 1-2 (insert, delete, substitute, swap), a sound-alike key
              ('exercize', 'nuemonia', 'fosfate'), and run-together words ('breastcancer' -> 'breast cancer')
  ranking     cheap keyboard-neighbour and swapped-letter typos, sound-alike bonus, domain words first,
              then frequency; the first letter must match unless the words sound alike
  safety      known words are never changed, gene symbols / abbreviations (TP53, HER2, mTOR) are left alone,
              words under 4 letters are left alone, and a tie between two candidates is left as typed
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
_LOCK = threading.Lock()
_CACHE: Dict[str, object] = {}

_KEYS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_POS = {ch: (r, c) for r, row in enumerate(_KEYS) for c, ch in enumerate(row)}


def _near(a: str, b: str) -> bool:
    """Neighbouring keys on a QWERTY keyboard (a 'fat-finger' typo)."""
    if a not in _POS or b not in _POS:
        return False
    (r1, c1), (r2, c2) = _POS[a], _POS[b]
    return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1


def edit_cost(a: str, b: str, limit: float = 2.5, weighted: bool = True) -> float:
    """Weighted Damerau-Levenshtein: swapped letters and neighbouring-key slips cost 0.5, other edits 1
    (weighted=False: every slip counts 1, i.e. the number of mistakes)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > limit:
        return limit + 1
    prev2: List[float] = []
    prev = [float(j) for j in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [float(i)] + [0.0] * lb
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                sub = 0.0
            else:
                sub = 0.5 if weighted and _near(a[i - 1], b[j - 1]) else 1.0
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + (0.5 if weighted else 1.0))
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[lb]


def phonetic(w: str) -> str:
    """Sound-alike spelling with the vowels kept: 'fosfate' and 'phosphate' both give 'fosfate'."""
    w = re.sub(r"^(?:p(?=[ns])|k(?=n)|g(?=n))", "", w.lower())
    for a, b in (("ph", "f"), ("ck", "k"), ("sch", "sk"), ("ch", "k"), ("qu", "kw"), ("x", "ks"), ("z", "s"),
                 ("y", "i"), ("gh", "g")):
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k")
    return re.sub(r"(.)\1+", r"\1", w)


# Words built like scientific terms. They are often real but rare (ketamine, organoid, proteome, pyroptosis), so they
# are never "corrected" into an everyday English word, and only into a biomedical term that is one slip away.
_SCI = re.compile(
    r"(?:omes?|omics?|osis|oses|itis|ine|ines|ide|ides|ases?|ins?|ols?|mab|nib|tides?|gens?|oids?|phages?|cytes?|"
    r"blasts?|ectomy|ology|lysis|genesis|ptosis|amines?|azoles?|cillin|mycin|cyclines?|statins?|pril|sartan|olol|"
    r"vir|ylation|ations?)$|^(?:micro|poly|epi|trans|anti|neuro|cardio|onco|immuno|bio|chemo|hepat|nephr|gastr|"
    r"derm|myo|osteo|lipo|glyco|phospho|meta|proteo|pharm|nano|cyto|geno|endo|exo)")


def sound_key(w: str) -> str:
    """A small sound-alike key: ph/f, c/k/s, z/s, x/ks, doubled letters and vowels after the first letter."""
    w = re.sub(r"^(?:p(?=[ns])|k(?=n)|g(?=n))", "", w.lower())     # silent first letters: pneumonia, psoriasis, knee
    for a, b in (("ph", "f"), ("ck", "k"), ("sch", "sk"), ("ch", "k"), ("qu", "kw"), ("x", "ks"), ("z", "s"),
                 ("y", "i"), ("gh", "g")):
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k")
    w = re.sub(r"(.)\1+", r"\1", w)
    return w[:1] + re.sub(r"[aeiou]", "", w[1:])


# --------------------------------------------------------------------------- vocabulary
def _rx_words(text: str) -> Set[str]:
    """Whole words inside a regular expression ('acute (?:myeloid|lymphoid)' -> acute, myeloid, lymphoid);
    stems such as 'leuk(?:a)?emi' or 'recover\\w*' are skipped so fragments never count as words."""
    return set(re.findall(r"(?<![A-Za-z\\])([a-z]{4,})(?=$|[\s|)])", text.lower()))


def _program_terms() -> Set[str]:
    """Biomedical words already defined in the program (no separate word list to maintain)."""
    words: Set[str] = set()
    try:
        import evidence_identity as ei
        words |= set(ei._LEXICON) | set(ei._Q_STOP) | set(ei.CLASS_WORDS)
        for name, d in ei.DISEASES.items():
            words |= set(re.findall(r"[a-z]{4,}", name.lower()))
            words |= _rx_words(str(d.get("words", "")))
    except Exception:
        pass
    try:
        from semantic_audit import ENDPOINT_FAMILIES, ENDPOINT_ACRONYMS
        for fam, rx, _ in ENDPOINT_FAMILIES:
            words |= set(re.findall(r"[a-z]{4,}", fam.replace("_", " "))) | _rx_words(rx)
        words |= {w for v in ENDPOINT_ACRONYMS.values() for w in re.findall(r"[a-z]{4,}", v.lower())}
    except Exception:
        pass
    try:
        from clinical_context import _DRUG_ABBR
        words |= {w for v in _DRUG_ABBR.values() for w in re.findall(r"[a-z]{4,}", v)}
    except Exception:
        pass
    try:
        from evidence_quality import EXPOSURE_REFERENCE
        words |= {w for k in EXPOSURE_REFERENCE for w in re.findall(r"[a-z]{4,}", k.lower())}
    except Exception:
        pass
    try:
        from research_planner import INTENT_TERMS
        words |= {w for v in INTENT_TERMS.values() for t in v for w in re.findall(r"[a-z]{4,}", t.lower())}
    except Exception:
        pass
    return {w for w in words if w.isalpha()}


def _library_terms(db_path: Optional[Path] = None, min_count: int = 2) -> Dict[str, int]:
    """Word counts from the titles / abstracts of papers already in the local knowledge base."""
    if db_path is None:
        try:
            from database import DEFAULT_DB_PATH as db_path        # follows BIOSYN_DB_PATH
        except Exception:
            db_path = ROOT / "research_knowledge.db"
    db_path = Path(db_path)
    counts: Dict[str, int] = {}
    if not db_path.exists():
        return counts
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        for title, abstract in con.execute("SELECT title, abstract FROM papers LIMIT 20000"):
            for w in re.findall(r"[a-z]{4,}", f"{title or ''} {abstract or ''}".lower()):
                counts[w] = counts.get(w, 0) + 1
        con.close()
    except Exception:
        return {}
    return {w: c for w, c in counts.items() if c >= min_count}


def _english() -> Dict[str, int]:
    try:
        from spellchecker import SpellChecker
        wf = SpellChecker().word_frequency
        return {w: int(wf[w]) for w in wf.keys() if w.isalpha()}
    except Exception:
        return {}


def _build() -> Dict[str, object]:
    domain: Dict[str, int] = {w: 1000 for w in _program_terms()}
    for w, c in _library_terms().items():
        domain[w] = domain.get(w, 0) + 50 + c
    english = _english()
    by_sound: Dict[str, List[str]] = {}
    by_len: Dict[Tuple[str, int], List[str]] = {}
    for w in set(domain) | {w for w, f in english.items() if f >= 5}:
        by_sound.setdefault(sound_key(w), []).append(w)
        by_len.setdefault((w[0], len(w)), []).append(w)
    return {"domain": domain, "english": english, "by_sound": by_sound, "by_len": by_len, "t": time.time()}


_REFRESHING = threading.Event()


def _refresh_in_background() -> None:
    try:
        fresh = _build()
        with _LOCK:
            _CACHE.clear()
            _CACHE.update(fresh)
    finally:
        _REFRESHING.clear()


def vocabulary(refresh: bool = False) -> Dict[str, object]:
    """{'domain': {word: weight}, 'english': {word: freq}, 'by_sound': {key: [words]}, 'by_len': {(letter, n): [words]}}
    Built once (about 2 s); every 10 minutes it is rebuilt in the background so newly read papers' words are learnt
    without ever making a search wait."""
    with _LOCK:
        if _CACHE and not refresh:
            if time.time() - float(_CACHE.get("t", 0)) > 600 and not _REFRESHING.is_set():
                _REFRESHING.set()
                threading.Thread(target=_refresh_in_background, name="speller-refresh", daemon=True).start()
            return dict(_CACHE)
        fresh = _build()
        _CACHE.clear()
        _CACHE.update(fresh)
        return dict(_CACHE)


def is_known(word: str, v: Optional[Dict[str, object]] = None) -> bool:
    v = v or vocabulary()
    w = word.lower()
    if w in v["domain"] or v["english"].get(w, 0) >= 2:
        return True
    # plurals / possessives / common endings of a known word ('cancers', 'alzheimers', 'treated')
    for suf in ("s", "es", "'s", "ed", "ing", "ly"):
        if w.endswith(suf) and (w[: -len(suf)] in v["domain"] or v["english"].get(w[: -len(suf)], 0) >= 2):
            return True
    return False


def _looks_like_symbol(token: str) -> bool:
    """Gene / protein symbols and abbreviations are never 'corrected': TP53, HER2, mTOR, IL-6, BRCA1, MCF-7."""
    return bool(re.search(r"\d", token)) or (sum(c.isupper() for c in token) >= 2) or (token[:1].islower() and any(c.isupper() for c in token[1:]))


def best_correction(word: str) -> Optional[str]:
    """The single best correction of one word, or None if the word is known, a symbol, or ambiguous."""
    if len(word) < 4 or _looks_like_symbol(word):
        return None
    v = vocabulary()
    low = word.lower()
    if is_known(low, v):
        return None
    if not v["english"] and len(low) < 6:
        return None                          # without the English word list a short everyday word ('rice') looks unknown
    maxd = 1.0 if len(low) <= 5 else 2.0
    cands: Dict[str, float] = {}
    raw: Dict[str, float] = {}                                 # the plain typing distance, for the safety rules
    letters = set(low)
    key = sound_key(low)
    for n in range(len(low) - int(maxd), len(low) + int(maxd) + 1):
        for w in v["by_len"].get((low[0], n), []):              # the first letter must match ...
            if len(letters.symmetric_difference(w)) > 2 * maxd + 1:
                continue                                        # quick reject before the full distance
            d = edit_cost(low, w, maxd)
            if d <= maxd:
                cands[w] = raw[w] = d
    for w in v["by_sound"].get(sound_key(low), []):             # ... unless it sounds alike ('nuemonia', 'fosfate')
        d = edit_cost(low, w, 4.0)
        if phonetic(w) == phonetic(low) or d <= 2.0:
            cands[w] = min(cands.get(w, 9.0), 0.4 + d * 0.2)
            raw[w] = min(raw.get(w, 9.0), d)
    if not cands:
        return None

    sci = bool(_SCI.search(low))
    phon = phonetic(low)

    def confident(w: str) -> bool:
        """Only a correction a careful reader would make: the safety rules in the module docstring."""
        slips = edit_cost(low, w, 3.0, weighted=False)       # number of mistakes, whatever kind
        same_sound = phonetic(w) == phon
        if v["domain"].get(w):
            if sci:                                           # 'colestrol' -> 'cholesterol': same consonants, 2 slips
                return slips <= 1 or same_sound or (slips <= 2 and len(low) >= 8 and sound_key(w) == key)
            return slips <= (2 if len(low) >= 7 else 1) or same_sound
        # an everyday English word: never for a scientific-looking or very short word, one slip at most, common words only
        return (not sci and len(low) >= 5 and (slips <= 1 or same_sound) and v["english"].get(w, 0) >= 100)

    cands = {w: d for w, d in cands.items() if confident(w)}
    if not cands:
        return None

    def score(w: str) -> float:
        dom = v["domain"].get(w, 0)
        freq = v["english"].get(w, 0)
        prior = (3.0 if dom else 0.0) + math.log10(1 + freq + dom) * 0.35
        return cands[w] * 2.0 - prior - (0.3 if sound_key(w) == key else 0.0)

    ranked = sorted(cands, key=score)
    if len(ranked) > 1 and abs(score(ranked[0]) - score(ranked[1])) < 0.15 and \
            bool(v["domain"].get(ranked[0])) == bool(v["domain"].get(ranked[1])):
        return None                                             # two equally good answers: leave it as typed
    best = ranked[0]
    if low.endswith("s") and not best.endswith("s") and edit_cost(low[:-1], best) < edit_cost(low, best):
        best += "s"                                             # keep the plural / possessive ('alzhiemers')
    return best


def split_joined(word: str) -> Optional[str]:
    """'breastcancer' -> 'breast cancer' when both halves are known words."""
    v = vocabulary()
    low = word.lower()
    if len(low) < 8 or is_known(low, v):
        return None
    for i in range(4, len(low) - 3):
        a, b = low[:i], low[i:]
        if (a in v["domain"] or v["english"].get(a, 0) > 50) and (b in v["domain"] or v["english"].get(b, 0) > 50):
            return f"{a} {b}"
    return None


# the word a question starts with; a slip here ('Csn leukoderma be cured', 'Wht causes', 'Dose metformin') must be
# fixed even though the word is short, or the slip is later read as the name of a drug or gene
_Q_WORDS = ("can", "does", "do", "did", "is", "are", "was", "were", "what", "how", "why", "which", "who", "when",
            "where", "will", "would", "could", "should", "may", "might", "has", "have")


def question_word(word: str) -> Optional[str]:
    """The question word a mistyped first word stands for, or None ('Csn' -> 'can', 'Dose' -> 'does')."""
    low = word.lower()
    if low in _Q_WORDS or len(low) < 2 or len(low) > 6 or not low.isalpha() or (len(word) > 1 and word.isupper()):
        return None
    v = vocabulary()
    rare = v["english"].get(low, 0) < 1500 and not v["domain"].get(low)     # 'doe', 'wat', 'hoe', 'wen'
    if low not in ("dose",) and is_known(low) and not rare:
        return None                                           # a common real word ('Cells ...', 'Low ...') stays
    near = [q for q in _Q_WORDS if abs(len(q) - len(low)) <= 1 and edit_cost(low, q, 1.5, weighted=False) <= 1
            and low[0] == q[0]]
    if len(near) > 1:                                         # 'wht': a dropped letter is the likeliest slip -> 'what'
        sub = [q for q in near if _is_subsequence(low, q)]
        near = sub if len(sub) == 1 else near
    return near[0] if len(near) == 1 else None


def _is_subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(ch in it for ch in small)


def correct_question(question: str) -> Dict[str, object]:
    """{'question': corrected, 'original', 'corrections': [(typed, fixed)], 'suggestion': text or None}"""
    q = question or ""
    corrections: List[Tuple[str, str]] = []
    first = re.match(r"\s*([A-Za-z]+)\b", q)
    if first:
        qw = question_word(first.group(1))
        if qw == "does" and first.group(1).lower() == "dose" and \
                re.match(r"\s*(?:of|escalation|response|dependent|finding|reduction|adjustment)\b", q[first.end(1):], re.I):
            qw = None                                         # 'Dose escalation of ...' is a real question
        if qw:
            typed = first.group(1)
            qw = qw.capitalize() if typed[:1].isupper() else qw
            q = q[: first.start(1)] + qw + q[first.end(1):]
            corrections.append((typed, qw))

    def fix(m: re.Match) -> str:
        w = m.group(0)
        if w.lower().endswith("'s"):                         # "Alzhiemer's" -> "Alzheimer's"
            base = w[:-2]
            new = best_correction(base)
            if new and new.lower() != base.lower():
                new = (new.capitalize() if base[:1].isupper() else new) + w[-2:]
                corrections.append((w, new))
                return new
            return w
        new = split_joined(w) or best_correction(w)
        if not new or new.lower() == w.lower():
            return w
        new = new.capitalize() if w[:1].isupper() else new
        corrections.append((w, new))
        return new

    fixed = re.sub(r"(?<![\w-])[A-Za-z]+(?:'s)?(?![\w])|(?<=-)[A-Za-z]+(?![\w])|[A-Za-z]+(?=-)", fix, q)
    fixed = re.sub(r"\s+", " ", fixed).strip()
    return {"question": fixed, "original": question or "", "corrections": corrections,
            "suggestion": fixed if corrections else None}
