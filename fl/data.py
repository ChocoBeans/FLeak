"""Load CLL (Broad, Nat Genet 2022) from the cBioPortal bulk download and split into cohort nodes."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cll_broad_2022"
LABEL_COL = "IGHV_MUTATION_STATUS"  # 'mutated' (M-CLL) -> 1, 'unmutated' (U-CLL) -> 0


def load_cll(n_genes=100, min_site_size=40):
    """Return (X, y, cohort, genes) for patients with RNA-seq and an IGHV label.

    X: log2(TPM+1) of the `n_genes` highest-variance genes, float32, shape (n, n_genes).
    y: 1 = IGHV-mutated, 0 = unmutated. cohort: contributing cohort per patient.
    Cohorts smaller than `min_site_size` are pooled into "OTHER".
    """
    pat = pd.read_csv(DATA_DIR / "data_clinical_patient.txt", sep="\t", comment="#")
    smp = pd.read_csv(DATA_DIR / "data_clinical_sample.txt", sep="\t", comment="#")
    df = smp.merge(pat, on="PATIENT_ID")
    df = df[df[LABEL_COL].isin(["mutated", "unmutated"])]

    expr = pd.read_csv(DATA_DIR / "data_mrna_seq_tpm.txt", sep="\t")
    expr = expr.dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
    expr = np.log2(expr.T.astype(float) + 1)  # rows = samples

    df = df[df["SAMPLE_ID"].isin(expr.index)].reset_index(drop=True)
    expr = expr.loc[df["SAMPLE_ID"]]
    genes = expr.var().nlargest(n_genes).index.tolist()
    X = expr[genes].fillna(0).to_numpy(dtype=np.float32)
    y = (df[LABEL_COL] == "mutated").to_numpy(dtype=np.int64)

    cohort = df["COHORT"]
    small = cohort.map(cohort.value_counts()) < min_site_size
    cohort = cohort.where(~small, "OTHER").to_numpy()
    return X, y, cohort, genes


def split_by_site(X, y, cohort):
    """Dict of cohort -> (X_site, y_site): one FL client per contributing cohort."""
    return {s: (X[cohort == s], y[cohort == s]) for s in np.unique(cohort)}


def make_splits(X, y, cohort, seed=0, test_size=0.2, val_size=0.2):
    """Shared held-out test set plus per-node train/val sets.

    The test set is stratified by (cohort, label) and held out before the node split, so
    centralized and federated runs are evaluated on the same patients. The remainder is
    split per node into train/val (stratified by label). Returns
    (test, nodes) with test = (X, y, cohort) and nodes = {site: {"train": (X, y), "val": (X, y)}}.
    """
    strata = np.char.add(cohort.astype(str), y.astype(str))
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_size, stratify=strata, random_state=seed)

    nodes = {}
    for s in np.unique(cohort):
        site = rest[cohort[rest] == s]
        tr, va = train_test_split(site, test_size=val_size, stratify=y[site], random_state=seed)
        nodes[s] = {"train": (X[tr], y[tr]), "val": (X[va], y[va])}
    return (X[test], y[test], cohort[test]), nodes


if __name__ == "__main__":
    X, y, cohort, _ = load_cll()
    print(f"X={X.shape}  positives={y.sum()} ({y.mean():.1%})")
    for s, (xs, ys) in split_by_site(X, y, cohort).items():
        print(f"  {s:6s} n={len(ys):4d} pos={ys.mean():.2f}")
    (Xt, yt, ct), nodes = make_splits(X, y, cohort)
    print(f"test n={len(yt)} pos={yt.mean():.2f}")
    for s, d in nodes.items():
        print(f"  {s:6s} train={len(d['train'][1])} val={len(d['val'][1])}")
