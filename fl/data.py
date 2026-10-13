"""Load CLL (Broad, Nat Genet 2022) from the cBioPortal bulk download and split into cohort nodes.

Selection and preprocessing choices are documented in DATA_NOTES.txt.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cll_broad_2022"
LABEL_COL = "IGHV_MUTATION_STATUS"  # 'mutated' (M-CLL) -> 1, 'unmutated' (U-CLL) -> 0

# Genes excluded before variance-based selection: immunoglobulin loci (they encode the IGHV label
# itself), sex-linked genes and hemoglobin (unrelated to the label / blood contamination), and
# unannotated or non-coding transcripts.
_EXCLUDE = re.compile(
    r"^(IG[HKL][VDJCMGAE]|IGJ|IGLL|VPREB|JCHAIN|MIR\d|SNOR|LINC|AC\d{6}|AL\d{6}|AP\d{6}|"
    r"RP\d+-|CT[ABCD]-|HB[ABDEGMQZ]\d?$|XIST$|TSIX$|RPS4Y|DDX3Y|KDM5D|UTY$|EIF1AY|ZFY$|"
    r"USP9Y|TXLNGY|NLGN4Y|PRKY|TBL1Y)"
)


def load_cll(n_genes=100, min_site_size=40):
    """Return (X, y, cohort, patient, genes) for samples with RNA-seq and an IGHV label.

    X: log2(TPM+1) of the `n_genes` highest-variance genes after excluding immunoglobulin,
    sex-linked and noisy genes; float32, shape (n_samples, n_genes).
    y: 1 = IGHV-mutated, 0 = unmutated. cohort: contributing cohort per sample.
    patient: PATIENT_ID per sample (a few patients have two samples).
    Cohorts smaller than `min_site_size` patients are pooled into "OTHER".
    """
    pat = pd.read_csv(DATA_DIR / "data_clinical_patient.txt", sep="\t", comment="#")
    smp = pd.read_csv(DATA_DIR / "data_clinical_sample.txt", sep="\t", comment="#")
    df = smp.merge(pat, on="PATIENT_ID")

    expr = pd.read_csv(DATA_DIR / "data_mrna_seq_tpm.txt", sep="\t")
    expr = expr.dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
    expr = np.log2(expr.T.astype(float) + 1)  # rows = samples

    df = df[df["SAMPLE_ID"].isin(expr.index)]
    df = df[df[LABEL_COL].isin(["mutated", "unmutated"])].reset_index(drop=True)
    expr = expr.loc[df["SAMPLE_ID"]]

    expr = expr[[g for g in expr.columns if not _EXCLUDE.match(g)]]
    genes = expr.var().nlargest(n_genes).index.tolist()
    X = expr[genes].fillna(0).to_numpy(dtype=np.float32)
    y = (df[LABEL_COL] == "mutated").to_numpy(dtype=np.int64)

    cohort = df["COHORT"]
    sizes = df.drop_duplicates("PATIENT_ID")["COHORT"].value_counts()
    cohort = cohort.where(cohort.map(sizes) >= min_site_size, "OTHER").to_numpy()
    return X, y, cohort, df["PATIENT_ID"].to_numpy(), genes


def split_by_site(X, y, cohort):
    """Dict of cohort -> (X_site, y_site): one FL client per contributing cohort."""
    return {s: (X[cohort == s], y[cohort == s]) for s in np.unique(cohort)}


def make_splits(X, y, cohort, patient, seed=0, test_size=0.2, val_size=0.2):
    """Shared held-out test set plus per-node train/val sets, split at the patient level.

    All samples of a patient go to the same split. The test set is stratified by
    (cohort, label) and held out before the node split, so centralized and federated runs are
    evaluated on the same patients. The remainder is split per node into train/val (stratified
    by label). Returns (test, nodes) with test = (X, y, cohort) and
    nodes = {site: {"train": (X, y), "val": (X, y)}}.
    """
    pids, first = np.unique(patient, return_index=True)
    py, pc = y[first], cohort[first]
    assert all(len(set(y[patient == p])) == 1 for p in pids), "label differs within a patient"

    def rows(chosen):
        return np.flatnonzero(np.isin(patient, pids[chosen]))

    strata = np.char.add(pc.astype(str), py.astype(str))
    rest, test = train_test_split(np.arange(len(pids)), test_size=test_size, stratify=strata,
                                  random_state=seed)
    nodes = {}
    for s in np.unique(pc):
        site = rest[pc[rest] == s]
        tr, va = train_test_split(site, test_size=val_size, stratify=py[site], random_state=seed)
        r_tr, r_va = rows(tr), rows(va)
        nodes[s] = {"train": (X[r_tr], y[r_tr]), "val": (X[r_va], y[r_va])}
    r_te = rows(test)
    return (X[r_te], y[r_te], cohort[r_te]), nodes


def _zscore(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def standardize_pooled(test, nodes):
    """Centralized setting: z-score every split with mean/std of the pooled training data."""
    Xtr = np.vstack([d["train"][0] for d in nodes.values()])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    nodes = {s: {k: (_zscore(Xs, mu, sd), ys) for k, (Xs, ys) in d.items()} for s, d in nodes.items()}
    return (_zscore(test[0], mu, sd), *test[1:]), nodes


def standardize_local(test, nodes):
    """Federated setting: each node z-scores with the mean/std of its own training data.

    Each test patient is scaled with the statistics of their own cohort's node, as if the
    global model were evaluated at that site.
    """
    out, Xte = {}, test[0].copy()
    for s, d in nodes.items():
        Xtr = d["train"][0]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        out[s] = {k: (_zscore(Xs, mu, sd), ys) for k, (Xs, ys) in d.items()}
        m = test[2] == s
        Xte[m] = _zscore(test[0][m], mu, sd)
    return (Xte.astype(np.float32), *test[1:]), out


if __name__ == "__main__":
    X, y, cohort, patient, genes = load_cll()
    print(f"X={X.shape} patients={len(np.unique(patient))} positives={y.sum()} ({y.mean():.1%})")
    print("genes:", ", ".join(genes[:15]), "...")
    (Xt, yt, ct), nodes = make_splits(X, y, cohort, patient)
    print(f"test n={len(yt)} pos={yt.mean():.2f}")
    for s, d in nodes.items():
        print(f"  {s:6s} train={len(d['train'][1])} val={len(d['val'][1])}")
