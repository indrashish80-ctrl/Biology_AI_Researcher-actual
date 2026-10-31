"""
arm_direction.py  (v25)
=======================
Which ARM does a sentence report the change for? A trial abstract often states the result from the comparator's side:

    "The OSA diagnosis rate was higher in the RA+ arm"            (RA+ is the COMPARATOR)  -> intervention: decrease
    "Grade 3-4 neutropenia was more common with the FOLFOX4 regimen" (FOLFOX4 = comparator) -> intervention: decrease
    "Happiness increased during the nature-based designs compared with self-selected entertainment" (comparator first)
    "26 units in the control group as compared to 4 units in the tranexamic acid group"      -> intervention: decrease

A finding is stored as 'intervention vs comparator', so a change attributed to the comparator is flipped. The arm
names are matched by their DISTINCTIVE words (the words one arm has and the other has not), by abbreviations the
abstract defines ('frovatriptan (F)'), and by generic control words (placebo, control, usual care, sham, vehicle).

    relative_direction(sentence, raw_direction, intervention, comparator) -> {'direction', 'rule', 'arm'}

When it cannot tell which arm a sentence speaks about, the raw direction is returned unchanged (rule 'none').
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

ARM_VERSION = "1.0"

_CONTROL = r"(?:placebo|control|controls|usual[- ]care|standard[- ]care|sham|vehicle|untreated|no[- ]treatment|" \
           r"conventional|comparator|reference)"
_STOP = set("""the a an of and or with in on for to by from at as group groups arm arms patients participants subjects
treatment treated therapy receiving received who were was is are be mg g kg ml dose doses daily day days week weeks
plus alone only combination combined versus vs compared""".split())
_UP = r"(?:higher|greater|larger|more|increased|elevated|longer|better|superior|enhanced|raised)"
_DOWN = r"(?:lower|less|fewer|smaller|reduced|decreased|shorter|worse|inferior|diminished)"


def _tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9+\-]*", (s or "").lower()) if t not in _STOP and len(t) > 1]


def _abbrev(arm: str) -> Optional[str]:
    m = re.search(r"\((?:group\s+)?([A-Za-z0-9+\-/.]{1,12})(?:\s+group)?\)", arm or "", re.I)
    return m.group(1) if m else None


def _group_label(arm: str) -> Optional[str]:
    """'laser iridotomy (LPI) group 1' -> a pattern for 'group 1'; 'Rocuronium (R group)' -> 'R group' / 'group R'."""
    m = re.search(r"\bgroup\s*([A-Za-z0-9+.]{1,6})\b", arm or "", re.I) or re.search(r"\b([A-Za-z0-9+.]{1,6})\s+group\b", arm or "", re.I)
    if not m or m.group(1).lower() in ("the", "a", "an", "of", "control", "placebo"):
        return None
    g = re.escape(m.group(1).lower())
    return rf"group\s*{g}|{g}\s+group"


def context_abbreviations(text: str) -> Dict[str, str]:
    """Abbreviations a paper defines: 'transversus abdominis plane (TAP) block' -> {'TAP': 'transversus abdominis plane'}."""
    out: Dict[str, str] = {}
    for m in re.finditer(r"((?:[A-Za-z][\w\-]*\s+){1,7})\((?:[Gg]roup\s+)?([A-Z][A-Za-z0-9+\-/.]{0,9})(?:\s+group)?\)", text or ""):
        words, ab = m.group(1).split(), m.group(2)
        letters = [c.lower() for c in ab if c.isalpha()]
        if not letters:
            continue
        # the shortest run of words ending before '(' whose initials spell the abbreviation ('fluticasone furoate
        # nasal spray (FFNS)'); a definition that does not spell it is only kept when nothing better exists
        best = None
        for n in range(1, len(words) + 1):
            run = words[-n:]
            initials = "".join(w[0].lower() for w in run if w[0].isalpha())
            if initials == "".join(letters) or (len(run) >= 2 and initials.startswith("".join(letters)[:len(run)]) and
                                               len(run) == len(letters)):
                best = " ".join(run)
                break
        if best:
            out[ab] = best
        elif ab not in out:
            out[ab] = " ".join(words[-max(1, min(len(words), len(letters))):])
    return out


def arm_patterns(intervention: str, comparator: str, context: str = "") -> Tuple[List[str], List[str]]:
    """Regex alternatives naming the intervention arm and the comparator arm (distinctive words only), plus the
    abbreviations the paper defines for either arm."""
    ti, tc = _tokens(intervention), _tokens(comparator)
    si, sc = set(ti), set(tc)
    ip = [re.escape(t) for t in ti if t not in sc and len(t) >= 3]
    cp = [re.escape(t) for t in tc if t not in si and len(t) >= 3]
    for arm, pats in ((intervention, ip), (comparator, cp)):
        a = _abbrev(arm)
        if a:
            pats.insert(0, re.escape(a.lower()))
            pats.insert(0, rf"group\s+{re.escape(a.lower())}|{re.escape(a.lower())}\s+group")
        gl = _group_label(arm)
        if gl:
            pats.insert(0, gl)
    if re.search(_CONTROL, comparator or "", re.I):
        cp.append(_CONTROL)
    for ab, long in context_abbreviations(context).items():
        lt = set(_tokens(long))
        if not lt:
            continue
        oi, oc = len(lt & si) / len(lt), len(lt & sc) / len(lt)
        if oi >= 0.6 and oi > oc:
            ip.insert(0, re.escape(ab.lower()))
        elif oc >= 0.6 and oc > oi:
            cp.insert(0, re.escape(ab.lower()))
    if re.search(r"\+|plus|combination|combined", comparator or "", re.I) and not re.search(r"\+|plus|combination|combined",
                                                                                            intervention or "", re.I):
        cp.extend([r"\w+\+", "combination"])
    if re.search(r"\+|plus|combination|combined", intervention or "", re.I) and not re.search(r"\+|plus|combination|combined",
                                                                                              comparator or "", re.I):
        ip.extend([r"\w+\+", "combination"])
    return ip, cp


def _find(pats: List[str], text: str) -> List[int]:
    out = []
    for p in pats:
        for m in re.finditer(rf"(?<![a-z0-9]){p}(?![a-z0-9])", text, re.I):
            out.append(m.start())
    return sorted(set(out))


_REF = r"(?:than|(?:as |when )?compared (?:with|to)|in comparison (?:with|to)|versus|vs\.?|relative to|superior to|inferior to|" \
       r"measured against|against|over)"
_GENERIC_I = r"(?:study|intervention|treatment|treated|experimental|observation|active|test|trial|verum)\s+(?:group|arm|patients|subjects)"
_SUPER_DOWN = r"(?:lowest|shortest|smallest|(?<!at )least|fewest|minimal)"
_SUPER_UP = r"(?:highest|longest|largest|greatest|(?<!at )most|maximal|best)"


def _fuzzy_hits(words: List[str], text: str) -> List[int]:
    """Positions of a distinctive arm word, tolerating one typing slip in words of 6+ letters ('Tibolne')."""
    import difflib
    out = []
    for m in re.finditer(r"[a-z0-9][a-z0-9+\-]*", text):
        w = m.group(0)
        for t in words:
            if len(t) >= 6 and len(w) >= 5 and t[0] == w[0] and difflib.SequenceMatcher(None, t, w).ratio() >= 0.85:
                out.append(m.start())
    return out


def _arm_positions(pats: List[str], words: List[str], text: str) -> List[int]:
    return sorted(set(_find(pats, text) + _fuzzy_hits(words, text)))


def _as_reference(pos: int, text: str) -> bool:
    """True when the arm mention at `pos` is the REFERENCE of a comparison ('... than the placebo group')."""
    before = text[max(0, pos - 45):pos]
    return re.search(rf"\b{_REF}\s+(?:that\s+(?:in|of|with)\s+|those\s+(?:in|receiving|treated with|given)\s+|"
                     rf"patients\s+(?:in|receiving|treated with)\s+|the\s+outcome\s+in\s+)?(?:the\s+)?(?:[\w\-+]+\s+){{0,3}}$",
                     before) is not None


def focus_clause(sentence: str, endpoint: str) -> str:
    """The clause about the endpoint's own qualifier (time point / subgroup) when the sentence reports several:
    'improved in inoperable patients (...), while in persistent/recurrent patients the change was ... p = 0.17'."""
    qual = re.findall(r"(?:at|after|by|within)\s+\d+\s*(?:-?\s*)?(?:weeks?|months?|days?|years?|h|hours?)|"
                      r"in\s+(?:the\s+)?[\w/\-]+(?:\s+[\w/\-]+)?\s+(?:patients|group|subgroup|women|men|children|cohort)",
                      (endpoint or "").lower())
    if not qual:
        return sentence
    parts = re.split(r";|\bwhile\b|\bwhereas\b|,\s*but\b|\bbut\b", sentence, flags=re.I)
    if len(parts) < 2:
        return sentence
    for q in qual:
        key = r"\s+".join(re.escape(w) for w in q.strip().split())
        hits = [pt for pt in parts if re.search(key, pt, re.I)]
        if len(hits) == 1:
            return hits[0]
    return sentence


def relative_direction(sentence: str, raw: Optional[str], intervention: str, comparator: str,
                       endpoint: str = "", significant: Optional[bool] = None, context: str = "") -> Dict[str, Any]:
    """Direction of the INTERVENTION relative to the COMPARATOR. raw may be None (the endpoint was not located);
    per-arm numbers and superlatives can still decide it when the test was significant."""
    if not sentence or not (intervention or comparator):
        return {"direction": raw, "rule": "none", "arm": None}
    low = sentence.lower()
    # several results in one sentence ('OPC had shorter stay (21.3 vs 32.4 h), IP were more likely to achieve vaginal
    # birth (53% vs 28%)'): read the clause that carries the endpoint
    ep_words = [w for w in _tokens(endpoint) if len(w) >= 4]
    parts = re.split(r";|\),\s+|,\s+(?=(?:and\s+)?[\w\-]+(?:\s+[\w\-]+)?\s+(?:were|was|had|showed|required|received)\b)", low)
    if ep_words and len(parts) > 1:
        scored = sorted(((sum(1 for w in ep_words if w in pt), i) for i, pt in enumerate(parts)), reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            low = parts[scored[0][1]]
    ip, cp = arm_patterns(intervention, comparator, context)
    ci_words = [t for t in _tokens(intervention) if t not in set(_tokens(comparator)) and len(t) >= 3]
    cc_words = [t for t in _tokens(comparator) if t not in set(_tokens(intervention)) and len(t) >= 3]
    if re.search(_CONTROL + r"|standard|conventional|usual", comparator or "", re.I):
        ip = ip + [_GENERIC_I]
    pi, pc = _arm_positions(ip, ci_words, low), _arm_positions(cp, cc_words, low)
    flip = {"increase": "decrease", "decrease": "increase"}
    have = raw in ("increase", "decrease")

    # 1. a comparison: '<A> ... higher ... than / compared with / superior to <B>' - the arm BEFORE the reference
    #    marker is the one the change belongs to; the arm after it is the reference
    ref = re.search(rf"\b{_REF}\b", low)
    if have and ref and (pi or pc):
        # the REFERENCE is only the phrase right after the marker ('Compared with patients in the chemotherapy group,
        # patients in the combination group had longer PFS': the subject comes after the reference phrase)
        tail = low[ref.end():]
        stop = re.search(r",|;|\(|\bthe\s+\w+\s+(?:group|arm)\s+(?:had|showed|was|were)\b|\b(?:had|showed|was|were|resulted|"
                         r"achieved|demonstrated|produced|reduced|increased|improved)\b", tail)
        span = (ref.end(), ref.end() + (stop.start() if stop else min(len(tail), 70)))
        in_ref = lambda x: span[0] <= x < span[1]
        ref_i = [x for x in pi if in_ref(x)]
        ref_c = [x for x in pc if in_ref(x)]
        subj_i = [x for x in pi if not in_ref(x)]
        subj_c = [x for x in pc if not in_ref(x)]
        if ref_c and not ref_i and (subj_i or not subj_c):
            return {"direction": raw, "rule": "comparator_is_reference", "arm": "intervention"}
        if ref_i and not ref_c and (subj_c or not subj_i):
            return {"direction": flip[raw], "rule": "intervention_is_reference", "arm": "comparator"}
        if subj_c and not subj_i:
            return {"direction": flip[raw], "rule": "comparator_is_subject", "arm": "comparator"}
        if subj_i and not subj_c:
            return {"direction": raw, "rule": "intervention_is_subject", "arm": "intervention"}

    # 2. 'X was higher / more common in (with, for) the <arm> group' with no reference marker in between
    m = re.search(rf"\b(?:{_UP}|{_DOWN}|more common|less common|more frequent|less frequent|more likely|less likely)\b"
                  rf"((?:(?!\b{_REF}\b)[^.;]){{0,60}}?)\b(?:in|with|for|among|after|on|under|during)\s+(?:the\s+)?"
                  rf"([^,;.()]{{1,60}})", low)
    if have and m:
        seg = m.group(2)
        c_hit = bool(_arm_positions(cp, cc_words, seg))
        i_hit = bool(_arm_positions(ip, ci_words, seg))
        if c_hit and not i_hit:
            return {"direction": flip[raw], "rule": "change_in_comparator_arm", "arm": "comparator"}
        if i_hit and not c_hit:
            return {"direction": raw, "rule": "change_in_intervention_arm", "arm": "intervention"}

    # 3. superlatives: 'the lowest pain score ... was obtained for oxymetazoline+lidocaine'
    sm = re.search(rf"\b({_SUPER_DOWN}|{_SUPER_UP})\b", low)
    if sm and significant is not False and (pi or pc):
        d = "decrease" if re.fullmatch(_SUPER_DOWN, sm.group(1)) else "increase"
        after = low[sm.end():]
        if _arm_positions(ip, ci_words, after) and not _arm_positions(cp, cc_words, after):
            return {"direction": d, "rule": "superlative_for_intervention", "arm": "intervention"}
        if _arm_positions(cp, cc_words, after) and not _arm_positions(ip, ci_words, after):
            return {"direction": flip[d], "rule": "superlative_for_comparator", "arm": "comparator"}

    # 4. numbers per arm: '36% in RAO vs. 66% in RA+', '26 units ... control ... 4 units ... tranexamic acid'
    if not have and pi and pc and significant is not False and len(pi) == 1 and len(pc) == 1:
        # only when the sentence gave no direction word, and names each arm exactly once (multi-arm and
        # multi-time-point sentences pair the wrong numbers)
        # spreads are not results: '(range, 3-11%)', '(SD 1.2)', '± 14.6', '(IQR 2-5)', '[95% CI ...]'
        low = re.sub(r"[(\[][^()\[\]]{0,40}\b(?:range|iqr|sd|se|sem|ci|interquartile|min|max)\b[^()\[\]]{0,40}[)\]]", lambda m: " " * len(m.group(0)), low)
        low = re.sub(r"(?:±|\+/-)\s*\d+(?:\.\d+)?", lambda m: " " * len(m.group(0)), low)
        low = re.sub(r"[(\[]\s*-?\d+(?:\.\d+)?\s*%?\s*(?:to|-|–|,)\s*-?\d+(?:\.\d+)?\s*%?\s*[)\]]",
                     lambda m: " " * len(m.group(0)), low)            # a bare interval '(32.2% to 46.0%)'
        pi, pc = _arm_positions(ip, ci_words, low), _arm_positions(cp, cc_words, low)
        nums = [n for n in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:%|units?|mg|ml|kg|mmhg|times|months?|days?|hours?|hrs?|"
                                       r"points?)?(?![\w.])", low)
                if not re.match(r"\s*(?:%\s*)?(?:ci|confidence)", low[n.end():n.end() + 15])
                and not re.search(r"(?:p\s*[<=>]\s*|95\s*%?\s*$)", low[max(0, n.start() - 6):n.start()])]

        def near(pos_list, n):
            return min((abs(p - n.start()) for p in pos_list), default=10 ** 6)
        # a dose written after the arm name ('esomeprazole 20 mg, 29.3%') is not a result; percentages win when present
        nums = [n for n in nums if not re.match(r"\s*(?:mg|g|mcg|ug|iu)\b", low[n.end():n.end() + 5])] or nums
        pct = [n for n in nums if low[n.end():n.end() + 2].strip().startswith("%")]
        if len(pct) >= 2:
            nums = pct
        vi = vc = []
        # pairing by ORDER: '<armA> ... 8 ... <armB> ... 16' (arm first) or '56% ... <armA> versus 32% ... <armB>'
        a_i, a_c = pi[0], pc[0]
        first, second = (a_i, a_c) if a_i < a_c else (a_c, a_i)
        between = [n for n in nums if first < n.start() < second]
        after = [n for n in nums if n.start() > second]
        before = [n for n in nums if n.start() < first]
        if between and after and not before:            # arm, number, arm, number
            v_first, v_second = float(between[0].group(1)), float(after[0].group(1))
        elif before and between:                        # number, arm, number, arm
            v_first, v_second = float(before[-1].group(1)), float(between[-1].group(1))
        else:
            v_first = v_second = None
        if v_first is not None:
            vi, vc = ([v_first], [v_second]) if first == a_i else ([v_second], [v_first])
        else:
            vi = [float(n.group(1)) for n in nums if near(pi, n) < near(pc, n) and near(pi, n) < 45]
            vc = [float(n.group(1)) for n in nums if near(pc, n) < near(pi, n) and near(pc, n) < 45]
        if vi and vc and vi[0] != vc[0] and (have or significant):
            d = "increase" if vi[0] > vc[0] else "decrease"
            return {"direction": d, "rule": "per_arm_values", "arm": "both"}

    # 5. only the comparator is named, as the place of the change (not as the reference): '... more common with FOLFOX4'
    if have and pc and not pi:
        if any(not _as_reference(x, low) for x in pc):
            cue = re.search(rf"\b(?:{_UP}|{_DOWN}|increas\w*|decreas\w*|reduc\w*|improv\w*|common|frequent)\b", low)
            if cue and any(abs(p - cue.start()) < 80 and not _as_reference(p, low) for p in pc):
                return {"direction": flip[raw], "rule": "only_comparator_named", "arm": "comparator"}
    return {"direction": raw, "rule": "none", "arm": None}
