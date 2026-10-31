"""
structure.py
------------
3D protein structures: fetch, read, analyse, find pockets, estimate flexibility. Runs on the user's computer.

  fetch_structure      experimental structure from the RCSB PDB, predicted model from the AlphaFold database (by
                       UniProt accession or gene name), or a new prediction from a sequence (ESMFold, ESM Atlas);
                       every download is cached locally
  parse_structure      PDB and mmCIF (atoms, chains, ligands, method, resolution, helix / strand records)
  analyze_structure    chains and sequences, secondary structure, size, model confidence (pLDDT) or experimental
                       quality (resolution, B-factors), buried / exposed residues, bound ligands
  find_pockets         pockets on the surface by a buriedness grid search (PocketPicker idea, Weisel et al. 2007):
                       probe points next to the surface enclosed by protein in most directions, clustered; lining
                       residues and any ligand already sitting there are reported
  ligand_contacts      which residues touch a bound ligand (hydrogen-bond-capable pairs, hydrophobic contacts)
  flexibility          Gaussian network model (Bahar et al. 1997): predicted per-residue flexibility from the
                       contact topology, checked against the experimental B-factors when they exist
  residue_context      the structural setting of one residue (for a mutation: buried or exposed, confidence,
                       secondary structure, in a pocket, near a ligand)

Honesty rules (printed with every result): a pocket is a geometric cavity, not a predicted binding affinity; the
network model is a coarse-grained estimate, not all-atom molecular dynamics; a structural context is not a prediction
of a mutation's effect. Binding free energies (dG), docking poses and molecular-dynamics trajectories are NOT
computed - they would need force-field software (GROMACS, AMBER, AutoDock Vina) that this program does not ship.
"""

from __future__ import annotations

import hashlib
import math
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "structures"

BASIS = "in silico structural analysis - computed from the 3D coordinates; not experimental evidence"
POCKET_NOTE = ("Geometric pockets found by a buriedness grid search. A pocket shows where a small molecule could fit; it "
               "is not a prediction of binding strength (dG) - that needs docking / free-energy software.")
FLEX_NOTE = ("Gaussian network model: a coarse-grained elastic model of the alpha-carbon contact network. It estimates "
             "which regions are rigid or flexible; it is not an all-atom molecular-dynamics simulation.")

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
       "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O", "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H",
       "HIE": "H", "HIP": "H", "CYX": "C", "ASX": "B", "GLX": "Z", "UNK": "X"}
HYDROPHOBIC = set("AVILMFWC")
WATER = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"}
IONS = {"NA", "K", "CL", "MG", "CA", "ZN", "FE", "FE2", "MN", "CU", "CU1", "CO", "NI", "CD", "HG", "IOD", "BR", "SO4",
        "PO4", "NO3", "LI", "CS", "SR", "BA"}
GLYCANS = {"NAG", "NDG", "MAN", "BMA", "FUC", "FUL", "GAL", "GLC", "BGC", "SIA", "GLA", "XYS", "A2G", "NGA"}
# common crystallisation additives: reported, but never called "the ligand"
ADDITIVES = {"GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "MPD", "ACT", "ACE", "FMT", "DMS", "TRS", "MES", "EPE", "BME",
             "IPA", "EOH", "MOH", "CIT", "TAR", "IMD", "NH4", "SCN", "BOG", "LDA", "DTT", "UNX", "UNL"}
VDW = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90, "F": 1.47, "CL": 1.75, "BR": 1.85,
       "I": 1.98, "FE": 1.40, "ZN": 1.39, "MG": 1.73, "CA": 1.97, "NA": 2.27, "K": 2.75, "MN": 1.40, "CU": 1.40}
POLAR = {"N", "O"}


# --------------------------------------------------------------------------- fetching (user-side, cached)
def _http(http: Any = None) -> Any:
    if http is not None:
        return http
    from http_client import HttpClient
    return HttpClient(cache_dir=str(CACHE / "http"), timeout=30, retries=1)


def _cache_file(key: str, ext: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)[:80]}.{ext}"


def fetch_pdb(pdb_id: str, http: Any = None) -> Dict[str, Any]:
    """An experimental structure from the RCSB Protein Data Bank (PDB format; mmCIF when no PDB file exists)."""
    pid = (pdb_id or "").strip().upper()
    if not re.fullmatch(r"[0-9][A-Z0-9]{3}", pid):
        raise ValueError("a PDB ID has 4 characters and starts with a digit, e.g. 1A8O or 6LU7")
    for ext in ("pdb", "cif"):
        f = _cache_file(f"pdb_{pid}", ext)
        if f.exists() and f.stat().st_size > 500:
            return {"text": f.read_text(encoding="utf-8", errors="replace"), "format": ext, "source": "pdb", "id": pid,
                    "url": f"https://www.rcsb.org/structure/{pid}"}
    h = _http(http)
    last = None
    for ext in ("pdb", "cif"):
        try:
            txt = h.get_text(f"https://files.rcsb.org/download/{pid}.{ext}", max_bytes=80 * 1024 * 1024)
            if txt and ("ATOM" in txt or "_atom_site" in txt):
                _cache_file(f"pdb_{pid}", ext).write_text(txt, encoding="utf-8")
                return {"text": txt, "format": ext, "source": "pdb", "id": pid, "url": f"https://www.rcsb.org/structure/{pid}"}
        except Exception as exc:          # large entries exist only as mmCIF
            last = exc
    raise RuntimeError(f"could not download PDB entry {pid}: {last}")


def uniprot_for_gene(gene: str, taxon: int = 9606, http: Any = None) -> Dict[str, Any]:
    """The reviewed UniProt entry for a gene symbol in one organism (default human, taxon 9606)."""
    g = (gene or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9\-_.]{1,20}", g):
        raise ValueError("give a gene symbol such as TP53, EGFR or ACE2")
    data = _http(http).get_json("https://rest.uniprot.org/uniprotkb/search",
                                params={"query": f"gene_exact:{g} AND organism_id:{int(taxon)} AND reviewed:true",
                                        "fields": "accession,protein_name,gene_names,length", "format": "json", "size": 1})
    res = (data or {}).get("results") or []
    if not res:
        raise RuntimeError(f"no reviewed UniProt entry for gene {g} in taxon {taxon}")
    r = res[0]
    name = (((r.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName") or {}).get("value")
    return {"accession": r.get("primaryAccession"), "protein": name, "length": (r.get("sequence") or {}).get("length")}


def fetch_alphafold(accession_or_gene: str, taxon: int = 9606, http: Any = None) -> Dict[str, Any]:
    """A predicted structure from the AlphaFold Protein Structure Database (pLDDT confidence in the B-factor column)."""
    key = (accession_or_gene or "").strip()
    info: Dict[str, Any] = {}
    if not re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}", key.upper()):
        info = uniprot_for_gene(key, taxon, http)
        key = info["accession"]
    acc = key.upper()
    f = _cache_file(f"af_{acc}", "pdb")
    if f.exists() and f.stat().st_size > 500:
        return {"text": f.read_text(encoding="utf-8", errors="replace"), "format": "pdb", "source": "alphafold",
                "id": acc, "url": f"https://alphafold.ebi.ac.uk/entry/{acc}", **info}
    h = _http(http)
    meta = h.get_json(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}")
    entry = meta[0] if isinstance(meta, list) and meta else meta if isinstance(meta, dict) else {}
    url = entry.get("pdbUrl")
    if not url:
        raise RuntimeError(f"AlphaFold has no model for {acc}")
    txt = h.get_text(url, max_bytes=60 * 1024 * 1024)
    f.write_text(txt, encoding="utf-8")
    return {"text": txt, "format": "pdb", "source": "alphafold", "id": acc, "url": f"https://alphafold.ebi.ac.uk/entry/{acc}",
            "protein": info.get("protein") or entry.get("uniprotDescription"), "gene": entry.get("gene"), **info}


def predict_esmfold(sequence: str, http: Any = None, timeout: float = 120.0) -> Dict[str, Any]:
    """A NEW structure prediction from an amino-acid sequence with ESMFold (ESM Metagenomic Atlas service, up to 400
    residues). The service is run by a third party and is sometimes unavailable; the error then says so."""
    seq = re.sub(r"[^A-Za-z]", "", sequence or "").upper()
    if not seq or re.search(r"[^ACDEFGHIKLMNPQRSTVWY]", seq):
        raise ValueError("give a protein sequence in one-letter code (20 standard amino acids)")
    if len(seq) > 400:
        raise ValueError(f"ESMFold online accepts up to 400 residues (this sequence has {len(seq)}); use AlphaFold DB by gene")
    digest = hashlib.sha1(seq.encode()).hexdigest()[:16]
    f = _cache_file(f"esm_{digest}", "pdb")
    if f.exists() and f.stat().st_size > 500:
        return {"text": f.read_text(encoding="utf-8"), "format": "pdb", "source": "esmfold", "id": f"ESMFold:{digest}",
                "sequence": seq}
    from urllib.request import Request, urlopen
    req = Request("https://api.esmatlas.com/foldSequence/v1/pdb/", data=seq.encode(), method="POST",
                  headers={"Content-Type": "text/plain", "User-Agent": "BioSynthetica/1.0"})
    try:
        with urlopen(req, timeout=timeout) as r:
            txt = r.read(20 * 1024 * 1024).decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"the ESMFold service did not answer ({exc}); try AlphaFold DB by gene name instead") from exc
    if "ATOM" not in txt:
        raise RuntimeError("the ESMFold service returned no structure")
    f.write_text(txt, encoding="utf-8")
    return {"text": txt, "format": "pdb", "source": "esmfold", "id": f"ESMFold:{digest}", "sequence": seq}


# --------------------------------------------------------------------------- parsing
def _num(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _element(name: str, given: str, resname: str) -> str:
    e = (given or "").strip().upper()
    if e and e.isalpha():
        return e
    n = re.sub(r"[^A-Za-z]", "", name or "").upper()
    if resname.upper() in IONS and n:
        return n[:2] if n[:2] in VDW else n[:1]
    return n[:1] if n else "C"


def parse_pdb(text: str) -> Dict[str, Any]:
    atoms: List[Dict[str, Any]] = []
    helices: List[Tuple[str, int, int]] = []
    sheets: List[Tuple[str, int, int]] = []
    title, method, resolution = [], None, None
    model_seen = False
    for line in (text or "").splitlines():
        rec = line[:6]
        if rec == "MODEL ":
            if model_seen:
                pass
            model_seen = True
        elif rec == "ENDMDL":
            break                                   # first model only (NMR ensembles)
        elif rec in ("ATOM  ", "HETATM"):
            if len(line) < 54:
                continue
            alt = line[16:17]
            if alt not in (" ", "A", "1"):
                continue                            # first alternative location only
            name = line[12:16].strip()
            resn = line[17:20].strip()
            atoms.append({"het": rec == "HETATM" and resn not in ("MSE",), "name": name, "resn": resn,
                          "chain": line[21:22].strip() or "A", "resi": int(_num(line[22:26], 0)), "icode": line[26:27].strip(),
                          "x": _num(line[30:38]), "y": _num(line[38:46]), "z": _num(line[46:54]),
                          "b": _num(line[60:66], 0.0) if len(line) >= 66 else 0.0,
                          "el": _element(name, line[76:78] if len(line) >= 78 else "", resn)})
        elif rec == "HELIX ":
            try:
                helices.append((line[19:20].strip() or "A", int(line[21:25]), int(line[33:37])))
            except ValueError:
                pass
        elif rec == "SHEET ":
            try:
                sheets.append((line[21:22].strip() or "A", int(line[22:26]), int(line[33:37])))
            except ValueError:
                pass
        elif rec == "TITLE ":
            title.append(line[10:80].strip())
        elif rec == "EXPDTA":
            method = line[10:79].strip()
        elif rec == "REMARK" and line[6:10].strip() == "2" and "RESOLUTION" in line:
            m = re.search(r"(\d+\.\d+)\s*ANGSTROM", line)
            if m:
                resolution = float(m.group(1))
    return {"atoms": atoms, "helices": helices, "sheets": sheets, "title": " ".join(title).strip() or None,
            "method": method, "resolution": resolution}


def _cif_tokens(line: str) -> List[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def parse_cif(text: str) -> Dict[str, Any]:
    """mmCIF: the _atom_site loop plus title, method, resolution and helix / strand ranges."""
    lines = (text or "").splitlines()
    single: Dict[str, str] = {}
    loops: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln == "loop_":
            cols: List[str] = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("_"):
                cols.append(lines[i].strip().split()[0])
                i += 1
            rows: List[List[str]] = []
            buf: List[str] = []
            while i < len(lines):
                s = lines[i]
                st = s.strip()
                if not st or st == "#" or st == "loop_" or (st.startswith("_") and not buf) or st.startswith("data_"):
                    break
                if s.startswith(";"):                      # multi-line text value
                    txt = [s[1:]]
                    i += 1
                    while i < len(lines) and not lines[i].startswith(";"):
                        txt.append(lines[i])
                        i += 1
                    buf.append("\n".join(txt).strip())
                else:
                    buf.extend(_cif_tokens(s))
                while len(buf) >= len(cols) and cols:
                    rows.append(buf[:len(cols)])
                    buf = buf[len(cols):]
                i += 1
            if cols:
                cat = cols[0].split(".")[0]
                loops[cat] = (cols, rows)
            continue
        if ln.startswith("_") and not ln.startswith("_atom_site."):
            parts = _cif_tokens(ln)
            if len(parts) >= 2:
                single[parts[0]] = parts[1]
            elif len(parts) == 1 and i + 1 < len(lines):
                nxt = lines[i + 1]
                if nxt.startswith(";"):
                    txt = [nxt[1:]]
                    i += 2
                    while i < len(lines) and not lines[i].startswith(";"):
                        txt.append(lines[i])
                        i += 1
                    single[parts[0]] = " ".join(t.strip() for t in txt).strip()
                else:
                    t = _cif_tokens(nxt)
                    single[parts[0]] = t[0] if t else ""
                    i += 1
        i += 1
    atoms: List[Dict[str, Any]] = []
    cols, rows = loops.get("_atom_site", ([], []))
    idx = {c.split(".", 1)[1]: k for k, c in enumerate(cols)}

    def g(r: List[str], *names: str, default: str = "") -> str:
        for n in names:
            if n in idx and idx[n] < len(r) and r[idx[n]] not in ("?", "."):
                return r[idx[n]]
        return default

    first_model = None
    for r in rows:
        model = g(r, "pdbx_PDB_model_num", default="1")
        if first_model is None:
            first_model = model
        if model != first_model:
            break
        alt = g(r, "label_alt_id")
        if alt not in ("", "A", "1"):
            continue
        resn = g(r, "auth_comp_id", "label_comp_id")
        name = g(r, "auth_atom_id", "label_atom_id").strip('"')
        atoms.append({"het": g(r, "group_PDB") == "HETATM" and resn != "MSE", "name": name, "resn": resn,
                      "chain": g(r, "auth_asym_id", "label_asym_id", default="A"),
                      "resi": int(_num(g(r, "auth_seq_id", "label_seq_id", default="0"), 0)),
                      "icode": g(r, "pdbx_PDB_ins_code"),
                      "x": _num(g(r, "Cartn_x")), "y": _num(g(r, "Cartn_y")), "z": _num(g(r, "Cartn_z")),
                      "b": _num(g(r, "B_iso_or_equiv"), 0.0), "el": _element(name, g(r, "type_symbol"), resn)})
    helices, sheets = [], []
    hc, hr = loops.get("_struct_conf", ([], []))
    hi = {c.split(".", 1)[1]: k for k, c in enumerate(hc)}
    for r in hr:
        try:
            helices.append((r[hi["beg_auth_asym_id"]], int(r[hi["beg_auth_seq_id"]]), int(r[hi["end_auth_seq_id"]])))
        except (KeyError, ValueError, IndexError):
            pass
    sc, sr = loops.get("_struct_sheet_range", ([], []))
    si = {c.split(".", 1)[1]: k for k, c in enumerate(sc)}
    for r in sr:
        try:
            sheets.append((r[si["beg_auth_asym_id"]], int(r[si["beg_auth_seq_id"]]), int(r[si["end_auth_seq_id"]])))
        except (KeyError, ValueError, IndexError):
            pass
    def first(item: str) -> Optional[str]:
        """A value written either as a single item or as the first row of a loop."""
        if single.get(item) not in (None, "?", "."):
            return single[item]
        cat, _, key = item.partition(".")
        lc, lr = loops.get(cat, ([], []))
        if item in lc and lr and lr[0][lc.index(item)] not in ("?", "."):
            return lr[0][lc.index(item)]
        return None
    for item in ("_exptl.method", "_struct.title"):
        v = first(item)
        if v:
            single[item] = v
    res = first("_refine.ls_d_res_high") or first("_em_3d_reconstruction.resolution") or first("_reflns.d_resolution_high")
    return {"atoms": atoms, "helices": helices, "sheets": sheets, "title": single.get("_struct.title"),
            "method": single.get("_exptl.method"), "resolution": _num(res, 0.0) or None}


def parse_structure(text: str, fmt: Optional[str] = None) -> Dict[str, Any]:
    t = text or ""
    f = (fmt or "").lower()
    if f not in ("pdb", "cif"):
        f = "cif" if re.search(r"^data_|^_atom_site\.", t, re.M) else "pdb"
    s = parse_cif(t) if f == "cif" else parse_pdb(t)
    if not s["atoms"]:
        raise ValueError("no atoms found - is this a PDB or mmCIF file?")
    s["format"] = f
    return s


# --------------------------------------------------------------------------- helpers
def _coords(atoms: List[Dict[str, Any]]) -> np.ndarray:
    return np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float).reshape(-1, 3)


def _protein_atoms(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [a for a in s["atoms"] if not a["het"] and a["resn"] in AA3 and a["el"] != "H"]


def _residues(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered residues of the protein chains with their CA atom."""
    out, seen = [], {}
    for a in _protein_atoms(s):
        key = (a["chain"], a["resi"], a["icode"])
        if key not in seen:
            seen[key] = len(out)
            out.append({"chain": a["chain"], "resi": a["resi"], "icode": a["icode"], "resn": a["resn"],
                        "aa": AA3.get(a["resn"], "X"), "ca": None, "b": None, "atoms": []})
        r = out[seen[key]]
        r["atoms"].append(a)
        if a["name"] == "CA":
            r["ca"] = (a["x"], a["y"], a["z"])
            r["b"] = a["b"]
    return [r for r in out if r["ca"] is not None]


def _ligands(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for a in s["atoms"]:
        if a["het"] and a["resn"] not in WATER:
            groups.setdefault((a["chain"], a["resi"], a["resn"]), []).append(a)
    out = []
    for (ch, ri, rn), ats in groups.items():
        kind = ("ion" if rn in IONS or len(ats) == 1 else "additive" if rn in ADDITIVES
                else "glycan (attached sugar)" if rn in GLYCANS else "ligand")
        out.append({"chain": ch, "resi": ri, "resn": rn, "n_atoms": len(ats), "kind": kind, "atoms": ats})
    return out


def _tree(xyz: np.ndarray):
    from scipy.spatial import cKDTree
    return cKDTree(xyz)


def _ss_from_records(s: Dict[str, Any], residues: List[Dict[str, Any]]) -> Optional[List[str]]:
    if not s["helices"] and not s["sheets"]:
        return None
    ss = []
    for r in residues:
        c, i = r["chain"], r["resi"]
        if any(c == hc and b <= i <= e for hc, b, e in s["helices"]):
            ss.append("H")
        elif any(c == sc and b <= i <= e for sc, b, e in s["sheets"]):
            ss.append("E")
        else:
            ss.append("C")
    return ss


def _ss_from_ca(residues: List[Dict[str, Any]]) -> List[str]:
    """Secondary structure from alpha-carbon geometry alone (P-SEA-like distance criteria, Labesse et al. 1997):
    helix d(i,i+2)=5.5+-0.5, d(i,i+3)=5.3+-0.5, d(i,i+4)=6.4+-0.6 A; strand 6.4+-0.6, 10.4+-0.6, 13.0+-0.6 A."""
    n = len(residues)
    ss = ["C"] * n
    ca = np.array([r["ca"] for r in residues], dtype=float) if n else np.zeros((0, 3))

    def d(i: int, j: int) -> float:
        if j >= n or residues[i]["chain"] != residues[j]["chain"] or residues[j]["resi"] - residues[i]["resi"] != j - i:
            return float("nan")
        return float(np.linalg.norm(ca[i] - ca[j]))
    hel = [False] * n
    strd = [False] * n
    for i in range(n):
        d2, d3, d4 = d(i, i + 2), d(i, i + 3), d(i, i + 4)
        if abs(d2 - 5.5) <= 0.5 and abs(d3 - 5.3) <= 0.5 and abs(d4 - 6.4) <= 0.6:
            for k in range(i, min(n, i + 5)):
                hel[k] = True
        if abs(d2 - 6.4) <= 0.6 and abs(d3 - 10.4) <= 0.6 and abs(d4 - 13.0) <= 0.6:
            for k in range(i, min(n, i + 5)):
                strd[k] = True
    for i in range(n):
        ss[i] = "H" if hel[i] else "E" if strd[i] else "C"
    return ss


# --------------------------------------------------------------------------- analysis
def analyze_structure(struct: Dict[str, Any], source: str = "pdb") -> Dict[str, Any]:
    residues = _residues(struct)
    if not residues:
        raise ValueError("the file has no protein residues with alpha-carbons")
    ca = np.array([r["ca"] for r in residues], dtype=float)
    chains: Dict[str, List[Dict[str, Any]]] = {}
    for r in residues:
        chains.setdefault(r["chain"], []).append(r)
    ss = _ss_from_records(struct, residues)
    ss_basis = "HELIX / SHEET records of the file"
    if ss is None:
        ss, ss_basis = _ss_from_ca(residues), "alpha-carbon geometry (P-SEA-like criteria)"
    centre = ca.mean(axis=0)
    rg = float(np.sqrt(((ca - centre) ** 2).sum(axis=1).mean()))
    span = float(max(np.ptp(ca[:, k]) for k in range(3)))
    # burial: neighbours within 10 A of each alpha-carbon (>= 24 = buried core, <= 14 = surface; Miller-like cut-offs)
    tree = _tree(ca)
    nb = np.array([len(x) - 1 for x in tree.query_ball_point(ca, 10.0)])
    buried = nb >= 24
    exposed = nb <= 14
    predicted = source in ("alphafold", "esmfold")
    bvals = np.array([r["b"] if r["b"] is not None else 0.0 for r in residues], dtype=float)
    conf: Dict[str, Any] = {}
    if predicted:
        pl = bvals * (100.0 if bvals.max() <= 1.0 else 1.0)       # ESMFold reports pLDDT on 0-1
        conf = {"kind": "pLDDT (model confidence, 0-100)", "mean_plddt": round(float(pl.mean()), 1),
                "very_high_ge90": round(float((pl >= 90).mean()), 3), "confident_70_90": round(float(((pl >= 70) & (pl < 90)).mean()), 3),
                "low_50_70": round(float(((pl >= 50) & (pl < 70)).mean()), 3), "very_low_lt50": round(float((pl < 50).mean()), 3),
                "reading": ("high-confidence model" if pl.mean() >= 85 else "mostly confident model" if pl.mean() >= 70
                            else "low-confidence model - interpret positions with care" if pl.mean() >= 50
                            else "very low confidence - likely disordered or poorly predicted")}
    else:
        conf = {"kind": "experimental", "method": struct.get("method"), "resolution_A": struct.get("resolution"),
                "mean_b_factor": round(float(bvals.mean()), 1) if len(bvals) else None,
                "reading": (None if not struct.get("resolution") else
                            "high resolution (atomic detail reliable)" if struct["resolution"] <= 2.0 else
                            "medium resolution" if struct["resolution"] <= 3.0 else
                            "low resolution - side-chain positions uncertain")}
    ligs = _ligands(struct)
    seqs = [{"chain": c, "n_residues": len(rs), "sequence": "".join(r["aa"] for r in rs),
             "first_residue": rs[0]["resi"], "last_residue": rs[-1]["resi"]} for c, rs in chains.items()]
    n = len(residues)
    return {
        "tool": "analyze_structure", "basis": BASIS, "source": source, "title": struct.get("title"),
        "format": struct.get("format"), "n_atoms": len(struct["atoms"]), "n_residues": n, "chains": seqs,
        "secondary_structure": {"helix": round(ss.count("H") / n, 3), "strand": round(ss.count("E") / n, 3),
                                "coil": round(ss.count("C") / n, 3), "basis": ss_basis,
                                "string": "".join(ss)[:5000]},
        "radius_of_gyration_A": round(rg, 2), "max_dimension_A": round(span, 1),
        "compactness": ("compact globular" if rg <= 2.2 * n ** 0.38 + 1.5 else "elongated or multi-domain"),
        "surface": {"buried_fraction": round(float(buried.mean()), 3), "exposed_fraction": round(float(exposed.mean()), 3),
                    "basis": "alpha-carbon neighbours within 10 A (>= 24 buried, <= 14 exposed)"},
        "confidence": conf,
        "ligands": [{k: v for k, v in L.items() if k != "atoms"} for L in ligs if L["kind"] != "additive"],
        "additives": sorted({L["resn"] for L in ligs if L["kind"] == "additive"}),
    }


# --------------------------------------------------------------------------- pockets (buriedness grid search)
def _sphere_dirs(n: int = 30) -> np.ndarray:
    """n nearly uniform directions on a sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def find_pockets(struct: Dict[str, Any], spacing: float = 1.0, ray: float = 10.0, min_buried: int = 17,
                 shell: Tuple[float, float] = (1.4, 4.5), min_volume: float = 10.0, top: int = 8) -> Dict[str, Any]:
    """Buriedness grid search (the PocketPicker idea, Weisel et al. 2007): grid points where a water-sized probe fits
    next to the protein surface (1.4-4.5 A from the van der Waals surface) cast 30 rays of 10 A; a point whose rays hit
    protein in at least `min_buried` of 30 directions is enclosed - a pocket point. Connected points form a pocket.
    Ligands, ions and water are ignored while searching, so a ligand's own site is found (checked against bound ligands
    of deposited structures in tests/test_structures.py)."""
    from scipy import ndimage
    patoms = _protein_atoms(struct)
    if len(patoms) < 30:
        raise ValueError("too few protein atoms for a pocket search")
    xyz = _coords(patoms)
    lo = xyz.min(axis=0) - 5.0
    hi = xyz.max(axis=0) + 5.0
    shape = np.ceil((hi - lo) / spacing).astype(int) + 1
    if int(np.prod(shape)) > 20_000_000:
        spacing = float((np.prod(hi - lo) / 20_000_000) ** (1 / 3)) + 0.05
        shape = np.ceil((hi - lo) / spacing).astype(int) + 1
    gx, gy, gz = (lo[k] + spacing * np.arange(shape[k]) for k in range(3))
    grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    tree = _tree(xyz)
    radii = np.array([VDW.get(a["el"], 1.7) for a in patoms])
    dist, idx = tree.query(grid, k=1, distance_upper_bound=12.0)
    gap = dist - radii[np.minimum(idx, len(radii) - 1)]        # distance to the van der Waals surface
    occ = (gap <= 0).reshape(shape)
    cand = np.where((gap >= shell[0]) & (gap <= shell[1]))[0]
    if not len(cand):
        return {"tool": "find_pockets", "basis": BASIS, "note": POCKET_NOTE, "pockets": [], "grid_spacing_A": spacing}
    ci = np.stack(np.unravel_index(cand, shape), axis=1)
    steps = np.arange(1, int(round(ray / spacing)) + 1)
    bur = np.zeros(len(cand), dtype=np.int16)
    for dv in _sphere_dirs(30):
        hit = np.zeros(len(cand), dtype=bool)
        for k in steps:
            q = np.rint(ci + dv * k).astype(int)
            ok = np.all((q >= 0) & (q < shape), axis=1)
            h = np.zeros(len(cand), dtype=bool)
            h[ok] = occ[q[ok, 0], q[ok, 1], q[ok, 2]]
            hit |= h
        bur += hit
    keep_pts = cand[bur >= min_buried]
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    mask[keep_pts] = True
    bur_full = np.zeros(int(np.prod(shape)), dtype=np.int16)
    bur_full[cand] = bur
    lab, n = ndimage.label(mask.reshape(shape), structure=np.ones((3, 3, 3)))
    if n == 0:
        return {"tool": "find_pockets", "basis": BASIS, "note": POCKET_NOTE, "pockets": [], "grid_spacing_A": spacing}
    vox = spacing ** 3
    flat_lab = lab.reshape(-1)
    sizes = np.bincount(flat_lab)[1:]
    order = [i + 1 for i in np.argsort(-sizes) if sizes[i] * vox >= min_volume][: top * 3]
    # a very large groove (an enzyme cleft, a protein-protein interface) is split into sub-pockets of a few hundred A^3
    from scipy.cluster.vq import kmeans2
    regions: List[Tuple[np.ndarray, Optional[float]]] = []
    for li in order:
        sel = np.where(flat_lab == li)[0]
        vol = len(sel) * vox
        if vol > 2500:
            k = int(min(12, math.ceil(vol / 800)))
            _, lab_k = kmeans2(grid[sel], k, seed=0, minit="++")
            regions += [(sel[lab_k == j], round(vol, 0)) for j in range(k) if (lab_k == j).sum() * vox >= min_volume]
        else:
            regions.append((sel, None))
    ligs = [L for L in _ligands(struct) if L["kind"] in ("ligand", "ion")]
    pockets = []
    for sel, cleft in regions:
        pts = grid[sel]
        vol = len(sel) * vox
        b = float(bur_full[sel].mean())
        lining = set()
        for lst in tree.query_ball_point(pts, 4.5):
            lining.update(lst)
        res = sorted({(patoms[k]["chain"], patoms[k]["resi"], patoms[k]["resn"]) for k in lining}, key=lambda t: (t[0], t[1]))
        multi = len({r[0] for r in res}) > 1
        aa = [AA3.get(r[2], "X") for r in res]
        ptree = _tree(pts)
        inside = [f"{L['resn']} {L['chain']}{L['resi']}" for L in ligs
                  if float(np.mean(ptree.query(_coords(L["atoms"]), k=1)[0] <= (4.5 if L["n_atoms"] == 1 else 3.0))) >= 0.5]
        # predicted models: a pocket in a low-confidence region (pLDDT < 70) may not exist in the real protein
        pl_mean, pl_note = None, None
        if struct.get("source") in ("alphafold", "esmfold") and lining:
            bl = np.array([float(patoms[k].get("b") or 0.0) for k in lining])
            bl = bl * (100.0 if bl.max() <= 1.0 else 1.0)
            pl_mean = round(float(bl.mean()), 1)
            pl_note = ("confident region" if pl_mean >= 70 else
                       "LOW-confidence region (pLDDT < 70) - this pocket may be an artefact of an uncertain model")
        pockets.append({
            "mean_plddt": pl_mean, "confidence_note": pl_note,
            "volume_A3": round(vol, 1), "mean_buriedness": round(b / 30, 2), "score": round(vol * (b / 30) ** 2, 1),
            "centre": [round(float(v), 2) for v in pts.mean(axis=0)], "n_lining_residues": len(res),
            "lining_residues": [f"{r[2]}{r[1]}" + (f":{r[0]}" if multi else "") for r in res][:60],
            "hydrophobic_fraction": round(sum(1 for x in aa if x in HYDROPHOBIC) / max(1, len(aa)), 2),
            "contains_bound": inside,
            "size_class": "large" if vol >= 250 else "medium" if vol >= 80 else "small",
            "part_of_groove_A3": cleft,
            "points": [[round(float(v), 1) for v in p] for p in pts[:: max(1, len(pts) // 150)]],
        })
    # confident pockets first: a low-pLDDT pocket never outranks one the model is sure about
    pockets.sort(key=lambda p: (p.get("mean_plddt") is not None and p["mean_plddt"] < 70, -p["score"]))
    for i, p in enumerate(pockets[:top], 1):
        p["rank"] = i
    return {"tool": "find_pockets", "basis": BASIS, "note": POCKET_NOTE,
            "method": "buriedness grid search (30 rays x 10 A from water-sized probe points; PocketPicker idea)",
            "volume_meaning": "probe-accessible grid volume next to the surface (A^3), comparable between pockets",
            "grid_spacing_A": round(spacing, 2), "pockets": pockets[:top]}


# --------------------------------------------------------------------------- ligand contacts
def ligand_contacts(struct: Dict[str, Any], ligand: Optional[str] = None) -> Dict[str, Any]:
    """Residues within 4 A of each bound ligand; N/O pairs within 3.5 A are hydrogen-bond-capable, C-C pairs within 4 A
    are hydrophobic contacts. Read from the deposited coordinates - an observation, not a prediction."""
    patoms = _protein_atoms(struct)
    tree = _tree(_coords(patoms))
    out = []
    for L in _ligands(struct):
        if L["kind"] == "additive" or (ligand and L["resn"].upper() != ligand.upper()):
            continue
        hb, hp, res = [], [], set()
        for a in L["atoms"]:
            for k in tree.query_ball_point([a["x"], a["y"], a["z"]], 4.0):
                p = patoms[k]
                dd = math.dist((a["x"], a["y"], a["z"]), (p["x"], p["y"], p["z"]))
                tag = f"{p['resn']}{p['resi']}"
                res.add((p["chain"], p["resi"], p["resn"]))
                if a["el"] in POLAR and p["el"] in POLAR and dd <= 3.5:
                    hb.append({"ligand_atom": a["name"], "residue": tag, "atom": p["name"], "distance_A": round(dd, 2)})
                elif a["el"] == "C" and p["el"] == "C" and dd <= 4.0:
                    hp.append(tag)
        out.append({"ligand": f"{L['resn']} {L['chain']}{L['resi']}", "kind": L["kind"], "n_atoms": L["n_atoms"],
                    "contact_residues": [f"{r[2]}{r[1]}" for r in sorted(res, key=lambda t: (t[0], t[1]))],
                    "hydrogen_bond_capable": hb[:40], "hydrophobic_contacts": sorted(set(hp))[:40]})
    return {"tool": "ligand_contacts", "basis": BASIS, "ligands": out,
            "note": "Distances from the deposited coordinates (observed binding mode). No binding energy is computed."}


# --------------------------------------------------------------------------- flexibility (Gaussian network model)
def flexibility(struct: Dict[str, Any], cutoff: float = 7.3, max_residues: int = 3000, source: str = "pdb") -> Dict[str, Any]:
    residues = _residues(struct)
    n = len(residues)
    if n < 10:
        raise ValueError("too few residues for a network model")
    if n > max_residues:
        raise ValueError(f"{n} residues is too large for the network model on this computer (limit {max_residues})")
    ca = np.array([r["ca"] for r in residues], dtype=float)
    tree = _tree(ca)
    K = np.zeros((n, n))
    for i, j in tree.query_pairs(cutoff):
        K[i, j] = K[j, i] = -1.0
    K[np.diag_indices(n)] = -K.sum(axis=1)
    w, v = np.linalg.eigh(K)
    nz = w > 1e-6 * max(1.0, w.max())
    if nz.sum() < 2:
        raise ValueError("the contact network is disconnected")
    inv_diag = ((v[:, nz] ** 2) / w[nz]).sum(axis=1)
    msf = inv_diag / inv_diag.mean()
    b = np.array([r["b"] or 0.0 for r in residues], dtype=float)
    corr = None
    if source not in ("alphafold", "esmfold") and b.std() > 1e-6:     # B-factors of a predicted model hold pLDDT
        corr = round(float(np.corrcoef(msf, b)[0, 1]), 3)
    # hinge-like residues: sign change of the slowest non-trivial mode
    slow = v[:, np.where(nz)[0][0]]
    hinges = [f"{residues[i]['resn']}{residues[i]['resi']}" for i in range(1, n)
              if np.sign(slow[i]) != np.sign(slow[i - 1]) and residues[i]["chain"] == residues[i - 1]["chain"]]
    order = np.argsort(-msf)
    return {"tool": "flexibility", "basis": BASIS, "note": FLEX_NOTE, "n_residues": n, "cutoff_A": cutoff,
            "most_flexible": [f"{residues[i]['resn']}{residues[i]['resi']}:{residues[i]['chain']}" for i in order[:10]],
            "most_rigid": [f"{residues[i]['resn']}{residues[i]['resi']}:{residues[i]['chain']}" for i in order[::-1][:10]],
            "hinge_candidates": hinges[:12],
            "correlation_with_experimental_b_factors": corr,
            "validation": (None if corr is None else
                           f"predicted flexibility vs the crystal's B-factors: r = {corr} "
                           + ("(good agreement)" if corr >= 0.5 else "(moderate)" if corr >= 0.3 else "(weak - treat with care)")),
            "profile": [round(float(x), 3) for x in msf[:4000]]}


# --------------------------------------------------------------------------- residue context (for mutations)
def residue_context(struct: Dict[str, Any], resi: int, chain: Optional[str] = None, source: str = "pdb",
                    pockets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    residues = _residues(struct)
    cands = [r for r in residues if r["resi"] == int(resi) and (chain is None or r["chain"] == chain)]
    if not cands:
        raise ValueError(f"residue {resi}{' in chain ' + chain if chain else ''} is not in the structure")
    r = cands[0]
    ca = np.array([x["ca"] for x in residues], dtype=float)
    k = residues.index(r)
    nb = len(_tree(ca).query_ball_point(ca[k], 10.0)) - 1
    ss = _ss_from_records(struct, residues) or _ss_from_ca(residues)
    near_lig = []
    rx = _coords(r["atoms"])
    for L in _ligands(struct):
        if L["kind"] == "additive":
            continue
        d = float(np.min(np.linalg.norm(rx[:, None, :] - _coords(L["atoms"])[None, :, :], axis=-1)))
        if d <= 5.0:
            near_lig.append(f"{L['resn']} ({d:.1f} A)")
    tag = f"{r['resn']}{r['resi']}"
    in_pocket = [p["rank"] for p in (pockets or {}).get("pockets", []) if any(x.startswith(tag) for x in p["lining_residues"])]
    burial = "buried in the core" if nb >= 24 else "on the surface" if nb <= 14 else "partly buried"
    conf = None
    if source in ("alphafold", "esmfold") and r["b"] is not None:
        conf = round(float(r["b"]) * (100 if r["b"] <= 1 else 1), 1)
    return {"tool": "residue_context", "basis": BASIS, "residue": f"{tag}:{r['chain']}", "amino_acid": r["aa"],
            "secondary_structure": {"H": "helix", "E": "strand", "C": "coil / loop"}[ss[k]],
            "neighbours_within_10A": nb, "burial": burial, "plddt": conf, "near_ligands": near_lig,
            "lines_pockets": in_pocket,
            "reading": (f"{tag} is {burial}" + (f", in a {'helix' if ss[k] == 'H' else 'strand' if ss[k] == 'E' else 'loop'}")
                        + (f", next to {', '.join(near_lig)}" if near_lig else "")
                        + (f", lining pocket {in_pocket}" if in_pocket else "")
                        + ". Buried and ligand-contacting positions are where substitutions are most likely to matter; "
                          "this is structural context, not a prediction of the mutation's effect.")}


# --------------------------------------------------------------------------- one call for the website / agent
def load(source: str, identifier: str = "", text: str = "", fmt: Optional[str] = None, taxon: int = 9606,
         http: Any = None) -> Dict[str, Any]:
    """source: 'pdb' (ID), 'alphafold' (UniProt accession or gene), 'esmfold' (sequence), 'file' (text given)."""
    src = (source or "pdb").lower()
    if src == "file":
        if not text:
            raise ValueError("no structure file given")
        if len(text) > 60 * 1024 * 1024:
            raise ValueError("structure file larger than 60 MB")
        got = {"text": text, "format": fmt, "source": "file", "id": "uploaded file"}
        if re.search(r"ALPHAFOLD|AlphaFold", text[:3000]):
            got["source"] = "alphafold"
    elif src == "pdb":
        got = fetch_pdb(identifier, http)
    elif src == "alphafold":
        got = fetch_alphafold(identifier, taxon, http)
    elif src == "esmfold":
        got = predict_esmfold(identifier, http)
    else:
        raise ValueError("source must be pdb, alphafold, esmfold or file")
    return got


def full_report(source: str, identifier: str = "", text: str = "", fmt: Optional[str] = None, taxon: int = 9606,
                pockets: bool = True, flex: bool = True, residue: Optional[int] = None, chain: Optional[str] = None,
                http: Any = None) -> Dict[str, Any]:
    got = load(source, identifier, text, fmt, taxon, http)
    st = parse_structure(got["text"], got.get("format"))
    src = got["source"] if got["source"] != "file" else "pdb"
    out: Dict[str, Any] = {"tool": "structure", "basis": BASIS, "source": got["source"], "id": got.get("id"),
                           "url": got.get("url"), "protein": got.get("protein"),
                           "analysis": analyze_structure(st, src)}
    st["source"] = src                                  # pockets read pLDDT only from predicted models
    if pockets:
        try:
            out["pockets"] = find_pockets(st)
        except Exception as exc:
            out["pockets"] = {"error": str(exc)}
    out["ligand_contacts"] = ligand_contacts(st)
    if flex:
        try:
            out["flexibility"] = flexibility(st, source=src)
        except Exception as exc:
            out["flexibility"] = {"error": str(exc)}
    if residue is not None:
        out["residue"] = residue_context(st, int(residue), chain, src, out.get("pockets") if isinstance(out.get("pockets"), dict) else None)
    out["limits"] = ("Not computed: binding free energy (dG), docking poses, all-atom molecular dynamics - these need "
                     "force-field software (AutoDock Vina, GROMACS, AMBER) that BioSynthetica does not ship.")
    # the 3D viewer gets the coordinates (one model, first 60k atom lines) - drawn in the browser by 3Dmol.js
    out["model_text"] = _trim_model(got["text"], st["format"])
    out["model_format"] = st["format"]
    return out


def _trim_model(text: str, fmt: str, max_lines: int = 120_000) -> str:
    if fmt == "cif":
        lines = text.splitlines()
        return "\n".join(lines[:max_lines])
    keep = []
    for ln in text.splitlines():
        if ln.startswith(("ATOM", "HETATM", "HELIX", "SHEET", "TER", "CRYST1", "HEADER", "TITLE")):
            keep.append(ln)
        elif ln.startswith("ENDMDL"):
            break
        if len(keep) >= max_lines:
            break
    return "\n".join(keep) + "\nEND\n"


def structures_for_genes(genes: List[str], taxon: int = 9606, limit: int = 3, http: Any = None) -> List[Dict[str, Any]]:
    """AlphaFold models (with analysis and pockets) for the first genes of a run - shown on the Results page as the 3D
    context of the proteins the evidence names. Failures are reported per gene and never stop a run."""
    out = []
    for g in [x for x in dict.fromkeys(genes or []) if x][:limit]:
        try:
            rep = full_report("alphafold", g, taxon=taxon, flex=False, http=http)
            p = rep.get("pockets") or {}
            out.append({"gene": g, "id": rep["id"], "url": rep.get("url"), "protein": rep.get("protein"),
                        "analysis": {k: rep["analysis"][k] for k in ("n_residues", "confidence", "secondary_structure",
                                                                      "radius_of_gyration_A")},
                        "top_pockets": [{k: v for k, v in q.items() if k != "points"} for q in (p.get("pockets") or [])[:3]],
                        "model_text": rep["model_text"], "model_format": rep["model_format"]})
        except Exception as exc:
            out.append({"gene": g, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    return out
