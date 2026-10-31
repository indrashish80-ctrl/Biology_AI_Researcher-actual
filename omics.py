"""
omics.py
--------
Expression-data statistics that run on the user's computer (numpy / scipy only).

  normalise_counts       DESeq2-style median-of-ratios size factors (Anders & Huber 2010) for raw RNA-seq counts
  filter_expressed       keep genes with enough counts in enough samples (edgeR filterByExpr-like rule)
  differential_expression
                         linear model per gene (group + optional batch / covariates) on log2 expression with
                         empirical-Bayes moderated t-statistics (limma, Smyth 2004) and Benjamini-Hochberg FDR;
                         batch is modelled IN the test, so it does not inflate or hide group differences
  remove_batch           batch means removed from log expression for plots (limma removeBatchEffect-like)
  pca                    principal components of the samples (do groups or batches separate?)
  cluster_cells          single-cell workflow: library-size normalisation, log1p, highly variable genes, PCA,
                         Ward clustering with the number of clusters chosen by silhouette, marker genes per cluster

What this does NOT do (stated in every result): raw-read processing (FASTQ quality control, alignment with
STAR / HISAT2, quantification) - start from a count or expression matrix; negative-binomial GLMs (DESeq2 / edgeR) -
log-linear moderated t is used, which is what limma-trend does; graph-based clustering / UMAP (Seurat, Scanpy).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

BASIS = "computed on your data - a statistical analysis, not literature evidence"
SCOPE = ("Starts from a count / expression matrix (no FASTQ alignment). Differential expression uses a linear model "
         "with empirical-Bayes moderated t (limma-trend method) and Benjamini-Hochberg FDR - not DESeq2's negative "
         "binomial model; results are usually close for well-replicated designs.")


# --------------------------------------------------------------------------- basic helpers
def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def normalise_counts(counts: np.ndarray) -> Dict[str, Any]:
    """Median-of-ratios size factors. counts: genes x samples (non-negative)."""
    c = np.asarray(counts, dtype=float)
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError("need a genes x samples matrix with at least 2 samples")
    if (c < 0).any():
        raise ValueError("raw counts cannot be negative - is this already log-transformed?")
    pos = (c > 0).all(axis=1)
    if pos.sum() < 10:
        sf = c.sum(axis=0) / np.exp(np.mean(np.log(c.sum(axis=0) + 1)))     # too few shared genes: total-count scaling
        method = "library size (too few genes expressed in every sample for median-of-ratios)"
    else:
        logc = np.log(c[pos])
        geo = logc.mean(axis=1, keepdims=True)
        sf = np.exp(np.median(logc - geo, axis=0))
        method = "median of ratios (DESeq2)"
    sf = np.where(sf > 0, sf, 1.0)
    return {"size_factors": sf, "normalised": c / sf, "method": method}


def filter_expressed(norm: np.ndarray, min_count: float = 10.0, min_samples: int = 2) -> np.ndarray:
    return (np.asarray(norm) >= min_count).sum(axis=1) >= max(1, int(min_samples))


def _trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x (Smyth 2004, Newton iteration on 1/trigamma)."""
    from scipy.special import polygamma
    if x > 1e7:
        return 1.0 / math.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = float(polygamma(1, y))
        dif = tri * (1 - tri / x) / float(polygamma(2, y))
        y -= dif
        if -dif / y < 1e-8:
            break
    return y


def squeeze_var(s2: np.ndarray, df: float) -> Dict[str, Any]:
    """Empirical-Bayes prior for gene variances (limma fitFDist): returns prior df d0, prior variance s0^2 and the
    moderated (posterior) variances."""
    from scipy.special import digamma, polygamma
    s2 = np.asarray(s2, dtype=float)
    ok = np.isfinite(s2) & (s2 > 0)
    if ok.sum() < 3 or df <= 0:
        return {"d0": 0.0, "s0_2": float(np.median(s2[ok])) if ok.any() else 1.0, "post": s2}
    z = np.log(s2[ok])
    e = z - float(digamma(df / 2)) + math.log(df / 2)
    emean = float(e.mean())
    evar = float(e.var(ddof=1)) - float(polygamma(1, df / 2))
    if evar > 0:
        d0 = 2 * _trigamma_inverse(evar)
        s0_2 = math.exp(emean + float(digamma(d0 / 2)) - math.log(d0 / 2))
    else:
        d0 = float("inf")
        s0_2 = math.exp(emean)
    if math.isinf(d0):
        post = np.full_like(s2, s0_2)
    else:
        post = (d0 * s0_2 + df * np.where(ok, s2, s0_2)) / (d0 + df)
    return {"d0": d0, "s0_2": s0_2, "post": post}


# --------------------------------------------------------------------------- differential expression
def _design(groups: Sequence[str], group_a: str, group_b: str, batch: Optional[Sequence[str]]) -> Dict[str, Any]:
    g = list(groups)
    keep = [i for i, x in enumerate(g) if x in (group_a, group_b)]
    cols = [np.ones(len(keep)), np.array([1.0 if g[i] == group_a else 0.0 for i in keep])]
    names = ["intercept", f"{group_a} vs {group_b}"]
    if batch is not None:
        b = [str(batch[i]) for i in keep]
        levels = sorted(set(b))
        for lv in levels[1:]:
            cols.append(np.array([1.0 if x == lv else 0.0 for x in b]))
            names.append(f"batch {lv}")
    X = np.stack(cols, axis=1)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("group and batch are confounded (each batch holds only one group) - the group effect cannot "
                         "be separated from the batch effect; no test is possible with this design")
    return {"X": X, "keep": keep, "names": names}


def moderated_t(logexpr: np.ndarray, groups: Sequence[str], group_a: str, group_b: str,
                batch: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """limma-style test of group_a vs group_b for every row of logexpr (genes x samples)."""
    from scipy import stats
    Y = np.asarray(logexpr, dtype=float)
    d = _design(groups, group_a, group_b, batch)
    X, keep = d["X"], d["keep"]
    Y = Y[:, keep]
    n, p = X.shape
    if (X[:, 1] == 1).sum() < 2 or (X[:, 1] == 0).sum() < 2:
        raise ValueError("each group needs at least 2 samples")
    df = n - p
    if df < 1:
        raise ValueError("not enough samples for this design (residual degrees of freedom < 1)")
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = Y @ X @ XtX_inv                                # genes x p
    resid = Y - beta @ X.T
    s2 = (resid ** 2).sum(axis=1) / df
    sq = squeeze_var(s2, df)
    unscaled = math.sqrt(XtX_inv[1, 1])
    lfc = beta[:, 1]
    df_total = df + (sq["d0"] if math.isfinite(sq["d0"]) else 1e6)
    t = lfc / (np.sqrt(sq["post"]) * unscaled)
    pv = 2 * stats.t.sf(np.abs(t), df_total)
    t_ord = lfc / (np.sqrt(np.where(s2 > 0, s2, np.nan)) * unscaled)
    p_ord = 2 * stats.t.sf(np.abs(t_ord), df)
    return {"log2fc": lfc, "t": t, "p": pv, "fdr": bh_fdr(np.nan_to_num(pv, nan=1.0)), "ordinary_t": t_ord,
            "ordinary_p": p_ord, "mean_a": Y[:, X[:, 1] == 1].mean(axis=1), "mean_b": Y[:, X[:, 1] == 0].mean(axis=1),
            "prior_df": sq["d0"], "prior_var": sq["s0_2"], "residual_df": df, "design": d["names"],
            "n_a": int((X[:, 1] == 1).sum()), "n_b": int((X[:, 1] == 0).sum())}


def differential_expression(matrix: np.ndarray, genes: Sequence[str], samples: Sequence[str], groups: Sequence[str],
                            group_a: str, group_b: str, batch: Optional[Sequence[str]] = None,
                            counts: Optional[bool] = None, fdr: float = 0.05, min_lfc: float = 0.0,
                            top: int = 50) -> Dict[str, Any]:
    """matrix: genes x samples. counts=None: decided from the data (non-negative integers = raw counts)."""
    M = np.asarray(matrix, dtype=float)
    if M.shape != (len(genes), len(samples)):
        raise ValueError("matrix shape does not match the gene and sample lists")
    if np.isnan(M).any():
        M = np.where(np.isnan(M), np.nanmean(M, axis=1, keepdims=True), M)
    is_counts = bool(counts) if counts is not None else bool((M >= 0).all() and np.allclose(M, np.round(M)) and M.max() > 50)
    notes = []
    if is_counts:
        nm = normalise_counts(M)
        nsmall = min(sum(1 for g in groups if g == group_a), sum(1 for g in groups if g == group_b))
        keep_g = filter_expressed(nm["normalised"], 10.0, nsmall)
        logx = np.log2(nm["normalised"][keep_g] + 1.0)
        notes.append(f"raw counts: {nm['method']} normalisation, log2(count + 1); {int((~keep_g).sum())} low-count genes "
                     f"removed (fewer than 10 normalised counts in {nsmall} samples)")
        gene_list = [g for g, k in zip(genes, keep_g) if k]
    else:
        logx = M if M.max() <= 50 else np.log2(np.clip(M, 0, None) + 1.0)
        notes.append("expression values treated as already normalised" + ("" if M.max() <= 50 else "; log2(x + 1) applied"))
        gene_list = list(genes)
    if len(gene_list) < 3:
        raise ValueError("fewer than 3 genes left to test")
    res = moderated_t(logx, groups, group_a, group_b, batch)
    order = np.argsort(res["p"])
    sig = (res["fdr"] < fdr) & (np.abs(res["log2fc"]) >= min_lfc)
    rows = []
    for i in order[:top]:
        rows.append({"gene": gene_list[i], "log2fc": round(float(res["log2fc"][i]), 3), "mean_a": round(float(res["mean_a"][i]), 3),
                     "mean_b": round(float(res["mean_b"][i]), 3), "moderated_t": round(float(res["t"][i]), 3),
                     "p_value": float(f"{res['p'][i]:.3g}"), "fdr": float(f"{res['fdr'][i]:.3g}"),
                     "direction": "up in " + group_a if res["log2fc"][i] > 0 else "down in " + group_a})
    step = max(1, len(gene_list) // 3000)
    volcano = [[round(float(res["log2fc"][i]), 3), round(float(-math.log10(max(res["p"][i], 1e-300))), 3), gene_list[i]]
               for i in range(0, len(gene_list), step)]
    pc = pca(logx, groups=[groups[i] for i in range(len(groups))], batch=batch)
    return {"tool": "differential_expression", "basis": BASIS, "scope": SCOPE, "notes": notes,
            "comparison": f"{group_a} vs {group_b}", "n_a": res["n_a"], "n_b": res["n_b"], "genes_tested": len(gene_list),
            "design": res["design"], "batch_in_model": batch is not None,
            "empirical_bayes": {"prior_df": None if not math.isfinite(res["prior_df"]) else round(res["prior_df"], 2),
                                "prior_variance": round(res["prior_var"], 5), "residual_df": res["residual_df"]},
            "significant": {"fdr_threshold": fdr, "min_abs_log2fc": min_lfc, "n": int(sig.sum()),
                            "up": int((sig & (res["log2fc"] > 0)).sum()), "down": int((sig & (res["log2fc"] < 0)).sum())},
            "top_genes": rows, "volcano": volcano, "pca": pc}


# --------------------------------------------------------------------------- batch + PCA
def remove_batch(logexpr: np.ndarray, batch: Sequence[str], groups: Optional[Sequence[str]] = None) -> np.ndarray:
    """Remove batch means while protecting the group effect (for plotting only - tests model batch directly)."""
    Y = np.asarray(logexpr, dtype=float)
    b = [str(x) for x in batch]
    levels = sorted(set(b))
    cols = [np.ones(len(b))]
    if groups is not None:
        gl = sorted(set(groups))
        cols += [np.array([1.0 if g == lv else 0.0 for g in groups]) for lv in gl[1:]]
    nb0 = len(cols)
    cols += [np.array([1.0 if x == lv else 0.0 for x in b]) - (sum(1 for x in b if x == lv) / len(b)) for lv in levels[1:]]
    X = np.stack(cols, axis=1)
    beta, *_ = np.linalg.lstsq(X, Y.T, rcond=None)
    return Y - (X[:, nb0:] @ beta[nb0:]).T


def pca(logexpr: np.ndarray, groups: Optional[Sequence[str]] = None, batch: Optional[Sequence[str]] = None,
        n_top: int = 500, n_pc: int = 3) -> Dict[str, Any]:
    Y = np.asarray(logexpr, dtype=float)
    v = Y.var(axis=1)
    idx = np.argsort(-v)[: min(n_top, len(v))]
    Z = Y[idx] - Y[idx].mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (S ** 2) / max(1e-12, (S ** 2).sum())
    k = min(n_pc, len(S))
    coords = (Vt[:k].T * S[:k]).tolist()
    out = {"variance_explained": [round(float(x), 3) for x in var[:k]], "coords": [[round(c, 3) for c in row] for row in coords],
           "genes_used": int(len(idx))}

    def separation(labels: Optional[Sequence[str]]) -> Optional[float]:
        if labels is None or len(set(labels)) < 2:
            return None
        pc1 = np.array([r[0] for r in coords])
        lab = np.array([str(x) for x in labels])
        between = sum((lab == L).sum() * (pc1[lab == L].mean() - pc1.mean()) ** 2 for L in set(lab))
        total = ((pc1 - pc1.mean()) ** 2).sum()
        return round(float(between / total), 3) if total > 0 else None
    out["pc1_explained_by_group"] = separation(groups)
    out["pc1_explained_by_batch"] = separation(batch)
    if out["pc1_explained_by_batch"] and out["pc1_explained_by_group"] is not None and \
            out["pc1_explained_by_batch"] > max(0.5, out["pc1_explained_by_group"]):
        out["warning"] = "the main source of variation (PC1) follows BATCH more than group - include batch in the model"
    return out


# --------------------------------------------------------------------------- single-cell clustering
def cluster_cells(matrix: np.ndarray, genes: Sequence[str], cells_are_rows: bool = True, n_hvg: int = 2000,
                  n_pc: int = 20, k_range: Sequence[int] = range(2, 11), seed: int = 0) -> Dict[str, Any]:
    from scipy.cluster.vq import kmeans2
    X = np.asarray(matrix, dtype=float)
    if not cells_are_rows:
        X = X.T
    if X.shape[1] != len(genes):
        raise ValueError("the number of genes does not match the matrix")
    if X.shape[0] < 20:
        raise ValueError("need at least 20 cells")
    if (X < 0).any():
        raise ValueError("counts cannot be negative")
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    L = np.log1p(X / lib * 1e4)
    mean = L.mean(axis=0)
    var = L.var(axis=0)
    disp = np.where(mean > 0.01, var / np.maximum(mean, 1e-9), 0)
    hv = np.argsort(-disp)[: min(n_hvg, X.shape[1])]
    Z = L[:, hv]
    Z = (Z - Z.mean(axis=0)) / np.maximum(Z.std(axis=0), 1e-9)
    Z = np.clip(Z, -10, 10)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    # keep only components above the random-noise limit (Marchenko-Pastur edge for standardised data): noise PCs would
    # blur real populations when k-means works on them
    eig = (S ** 2) / max(1, Z.shape[0] - 1)
    mp_edge = (1 + math.sqrt(Z.shape[1] / Z.shape[0])) ** 2
    n_signal = int((eig > mp_edge).sum())
    if n_signal == 0:
        return {"tool": "cluster_cells", "basis": BASIS, "n_cells": int(X.shape[0]), "n_genes": int(X.shape[1]),
                "n_clusters": 1, "cluster_sizes": [int(X.shape[0])], "silhouette": None, "markers": [],
                "silhouette_reading": "no population structure above random noise - the cells look like one population",
                "method": "PCA: no component rises above the random-noise limit (Marchenko-Pastur), so no clusters are called",
                "scope": "A simplified single-cell workflow (not Seurat / Scanpy).", "embedding_pc12": [
                    [round(float(U[i, 0] * S[0]), 3), round(float(U[i, 1] * S[1]), 3), 0] for i in range(0, X.shape[0], max(1, X.shape[0] // 3000))]}
    k_pc = int(min(n_pc, max(2, n_signal)))
    P = U[:, :k_pc] * S[:k_pc]
    # Ward hierarchical clustering on the informative components (handles populations of unequal spread better than
    # k-means); large data sets: the tree is built on up to 4000 cells and the rest join the nearest cluster centre
    from scipy.cluster.hierarchy import fcluster, linkage
    rng = np.random.default_rng(seed)
    sub = np.arange(len(P)) if len(P) <= 4000 else np.sort(rng.choice(len(P), 4000, replace=False))
    tree = linkage(P[sub], method="ward")
    best = None
    for k in [k for k in k_range if 2 <= k < len(sub)]:
        lab_s = fcluster(tree, k, criterion="maxclust") - 1
        if len(np.unique(lab_s)) < 2:
            continue
        cent = np.stack([P[sub][lab_s == c].mean(axis=0) for c in range(lab_s.max() + 1)])
        lab = np.argmin(((P[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1), axis=1)
        lab[sub] = lab_s
        sil = _silhouette(P, lab)
        if best is None or sil > best[0] + 1e-9:
            best = (sil, int(lab.max() + 1), lab)
    if best is None:
        cent, lab = kmeans2(P, 2, seed=seed, minit="++")
        best = (_silhouette(P, lab), 2, lab)
    sil, k, lab = best
    markers = []
    for c in range(k):
        inside = lab == c
        if inside.sum() < 2 or (~inside).sum() < 2:
            continue
        diff = L[inside].mean(axis=0) - L[~inside].mean(axis=0)
        top = np.argsort(-diff)[:8]
        markers.append({"cluster": int(c), "n_cells": int(inside.sum()),
                        "markers": [{"gene": genes[g], "log_fold_change": round(float(diff[g]), 3)} for g in top]})
    var = (S ** 2) / max(1e-12, (S ** 2).sum())
    return {"tool": "cluster_cells", "basis": BASIS,
            "method": ("library-size normalisation, log1p, highly variable genes, PCA (components above the noise limit), "
                       "Ward hierarchical clustering (number of clusters chosen by silhouette)"),
            "scope": "A simplified single-cell workflow - not graph-based (Leiden / Louvain) clustering or UMAP as in "
                     "Seurat / Scanpy; use it to see whether clear cell populations exist and what marks them.",
            "n_cells": int(X.shape[0]), "n_genes": int(X.shape[1]), "highly_variable_genes": int(len(hv)),
            "principal_components_used": k_pc,
            "n_clusters": int(k), "silhouette": round(float(sil), 3),
            "silhouette_reading": ("well separated" if sil >= 0.5 else "moderately separated" if sil >= 0.25
                                   else "weak structure - clusters may not be real populations"),
            "cluster_sizes": [int((lab == c).sum()) for c in range(k)], "markers": markers,
            "embedding_pc12": [[round(float(P[i, 0]), 3), round(float(P[i, 1]), 3), int(lab[i])]
                               for i in range(0, X.shape[0], max(1, X.shape[0] // 3000))],
            "variance_explained_pc12": [round(float(x), 3) for x in var[:2]]}


def _silhouette(P: np.ndarray, lab: np.ndarray, max_n: int = 2000) -> float:
    idx = np.arange(len(P))
    if len(P) > max_n:
        idx = np.random.default_rng(0).choice(len(P), max_n, replace=False)
    Q, l2 = P[idx], lab[idx]
    D = np.sqrt(((Q[:, None, :] - Q[None, :, :]) ** 2).sum(axis=-1))
    labs = np.unique(l2)
    if len(labs) < 2:
        return -1.0
    s = []
    for i in range(len(Q)):
        same = l2 == l2[i]
        if same.sum() <= 1:
            s.append(0.0)
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, l2 == L].mean() for L in labs if L != l2[i])
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))


# --------------------------------------------------------------------------- reading a table for the website
def from_table(df: Any) -> Dict[str, Any]:
    """A pandas table with gene IDs in the first column and one column per sample -> matrix pieces."""
    import pandas as pd
    t = df.copy()
    first = t.columns[0]
    if not pd.api.types.is_numeric_dtype(t[first]):
        genes = t[first].astype(str).tolist()
        t = t.drop(columns=[first])
    else:
        genes = [str(i) for i in t.index]
    num = t.apply(pd.to_numeric, errors="coerce")
    bad = [c for c in num.columns if num[c].isna().all()]
    num = num.drop(columns=bad)
    if num.shape[1] < 2:
        raise ValueError("need at least 2 numeric sample columns next to the gene column")
    return {"genes": genes, "samples": [str(c) for c in num.columns], "matrix": num.to_numpy(dtype=float),
            "dropped_columns": [str(c) for c in bad]}
