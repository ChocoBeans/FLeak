"""Load the raw CLL files, join/filter them, select features, and split into cohort nodes.

`python -m fl.preprocess` writes data/processed/samples_joined.csv, per-node
data/processed/nodes/<node>/{train,val}.csv and the shared data/processed/test.csv.
Selection and preprocessing choices are documented in DATA_NOTES.txt.
"""
import gzip
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cll_broad_2022"
PROCESSED_DIR = ROOT / "data" / "processed"
REFERENCE_DIR = ROOT / "data" / "reference"
GENCODE_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/"
               "gencode.v19.annotation.gtf.gz")
LABEL_COL = "IGHV_MUTATION_STATUS"  # 'mutated' (M-CLL) -> 1, 'unmutated' (U-CLL) -> 0
SEED = 4171

# Name patterns excluded before variance-based selection: immunoglobulin loci (they encode the IGHV
# label itself), hemoglobin (blood contamination), and unannotated or non-coding transcripts
# (clone-named, miRNA/snoRNA/lincRNA).
_EXCLUDE = re.compile(
    r"^(IG[HKL][VDJCMGAE]|IGJ|IGLL|VPREB|JCHAIN|MIR\d|SNOR|LINC|AC\d{6}|AL\d{6}|AP\d{6}|"
    r"RP\d+-|CT[ABCD]-|HB[ABDEGMQZ]\d?$)"
)
# Excluded using the GENCODE v19 annotation: Y-chromosome genes and XIST/TSIX (sex-linked), and
# pseudogenes, immunoglobulin loci and small RNAs by gene biotype.
_SEX_NAMES = {"XIST", "TSIX"}
_BIOTYPE_EXCLUDE = re.compile(r"pseudogene|^IG_|^(?:snRNA|snoRNA|miRNA|misc_RNA|rRNA)$")


def load_gene_annotation():
    """GENCODE v19 (hg19) gene table indexed by gene name, with columns `type` and `chr`.

    The annotation matches the gene names in the expression file. Downloaded once to
    data/reference/ and cached as a compact CSV.
    """
    cache = REFERENCE_DIR / "gencode_v19_genes.csv"
    if not cache.exists():
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        gtf = REFERENCE_DIR / "gencode.v19.annotation.gtf.gz"
        if not gtf.exists():
            urllib.request.urlretrieve(GENCODE_URL, gtf)
        rows = []
        with gzip.open(gtf, "rt") as f:
            for line in f:
                if line[0] == "#":
                    continue
                cols = line.split("\t")
                if cols[2] == "gene":
                    name = re.search(r'gene_name "([^"]+)"', cols[8]).group(1)
                    biotype = re.search(r'gene_type "([^"]+)"', cols[8]).group(1)
                    rows.append((name, biotype, cols[0]))
        pd.DataFrame(rows, columns=["gene", "type", "chr"]).drop_duplicates("gene").to_csv(
            cache, index=False)
    return pd.read_csv(cache).set_index("gene")


def _excluded_by_annotation():
    """Gene names to drop per the GENCODE annotation (sex-linked, pseudogenes, Ig, small RNAs)."""
    ann = load_gene_annotation()
    drop = (ann["chr"] == "chrY") | ann["type"].str.contains(_BIOTYPE_EXCLUDE)
    return set(ann.index[drop]) | _SEX_NAMES


def load_patients():
    """data_clinical_patient.txt: one row per patient (label, sex, cohort, ...)."""
    return pd.read_csv(DATA_DIR / "data_clinical_patient.txt", sep="\t", comment="#")


def load_samples():
    """data_clinical_sample.txt: one row per tumour sample, linked to its patient."""
    return pd.read_csv(DATA_DIR / "data_clinical_sample.txt", sep="\t", comment="#")


def load_expression_sample_ids():
    """Sample IDs that have RNA-seq (the column headers of the expression matrix)."""
    header = pd.read_csv(DATA_DIR / "data_mrna_seq_tpm.txt", sep="\t", nrows=0)
    return set(header.columns[1:])


_MONTH_SYMBOL = {"Sep": "SEPT", "Mar": "MARCH", "Dec": "DEC"}


def _fix_date_symbol(symbol):
    """Undo spreadsheet date conversion of gene symbols (e.g. '10-Sep' -> 'SEPT10')."""
    m = re.fullmatch(r"(\d{1,2})-(Sep|Mar|Dec)", symbol)
    if not m:
        return symbol
    n, month = m.groups()
    return "SEP15" if (n, month) == ("15", "Sep") else f"{_MONTH_SYMBOL[month]}{n}"


def load_expression():
    """data_mrna_seq_tpm.txt as a genes x samples TPM frame indexed by Hugo symbol.

    Date-mangled symbols (SEPT10 stored as '10-Sep') are restored; rows with a repeated symbol
    keep only the first.
    """
    expr = pd.read_csv(DATA_DIR / "data_mrna_seq_tpm.txt", sep="\t").dropna(subset=["Hugo_Symbol"])
    expr["Hugo_Symbol"] = expr["Hugo_Symbol"].map(_fix_date_symbol)
    return expr.drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")


def build_joined_table(save=True):
    """Join sample and patient tables, keep samples with RNA-seq and an IGHV label.

    Steps: (1) join the sample table to the patient table on PATIENT_ID, (2) keep samples that are
    columns of the expression matrix, (3) keep samples whose label is mutated/unmutated.
    Saved to data/processed/samples_joined.csv when `save` is set.
    """
    df = load_samples().merge(load_patients(), on="PATIENT_ID")
    df = df[df["SAMPLE_ID"].isin(load_expression_sample_ids())]
    df = df[df[LABEL_COL].isin(["mutated", "unmutated"])].reset_index(drop=True)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DIR / "samples_joined.csv", index=False)
    return df


def _top_variance_expression(samples, n_genes):
    """log2(TPM+1) of the `n_genes` highest-variance non-excluded genes; rows follow `samples`."""
    expr = np.log2(load_expression().T.astype(float) + 1).loc[samples]  # rows = samples
    drop = _excluded_by_annotation()
    expr = expr[[g for g in expr.columns if g not in drop and not _EXCLUDE.match(g)]]
    return expr[expr.var().nlargest(n_genes).index].fillna(0)


def _node_names(df, min_site_size):
    """Cohort per sample; cohorts with fewer than `min_site_size` patients become "OTHER"."""
    sizes = df.drop_duplicates("PATIENT_ID")["COHORT"].value_counts()
    return df["COHORT"].where(df["COHORT"].map(sizes) >= min_site_size, "OTHER").to_numpy()


def load_cll(n_genes=100, min_site_size=40):
    """Return (X, y, cohort, patient, genes) for samples with RNA-seq and an IGHV label.

    X: log2(TPM+1) of the `n_genes` highest-variance genes after excluding sex-linked,
    pseudogene, immunoglobulin and noisy genes; float32, shape (n_samples, n_genes).
    y: 1 = IGHV-mutated, 0 = unmutated. cohort: node per sample.
    patient: PATIENT_ID per sample (a few patients have two samples).
    Cohorts smaller than `min_site_size` patients are pooled into "OTHER".
    """
    df = build_joined_table(save=False)
    feats = _top_variance_expression(df["SAMPLE_ID"], n_genes)
    y = (df[LABEL_COL] == "mutated").to_numpy(dtype=np.int64)
    return (feats.to_numpy(dtype=np.float32), y, _node_names(df, min_site_size),
            df["PATIENT_ID"].to_numpy(), feats.columns.tolist())


def split_by_site(X, y, cohort):
    """Dict of cohort -> (X_site, y_site): one FL client per contributing cohort."""
    return {s: (X[cohort == s], y[cohort == s]) for s in np.unique(cohort)}


def split_indices(y, cohort, patient, seed=SEED, test_size=0.2, val_size=0.2):
    """Row indices of the shared test set and of each node's train/val sets (patient level).

    All samples of a patient go to the same split. The test set is stratified by
    (cohort, label) and held out before the node split, so centralized and federated runs are
    evaluated on the same patients. The remainder is split per node into train/val (stratified
    by label). Returns (test_idx, {node: {"train": idx, "val": idx}}).
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
        nodes[s] = {"train": rows(tr), "val": rows(va)}
    return rows(test), nodes


def make_splits(X, y, cohort, patient, seed=SEED, test_size=0.2, val_size=0.2):
    """Array version of `split_indices`.

    Returns (test, nodes) with test = (X, y, cohort) and
    nodes = {site: {"train": (X, y), "val": (X, y)}}.
    """
    test_idx, node_idx = split_indices(y, cohort, patient, seed, test_size, val_size)
    nodes = {s: {k: (X[i], y[i]) for k, i in d.items()} for s, d in node_idx.items()}
    return (X[test_idx], y[test_idx], cohort[test_idx]), nodes


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


def export_processed(n_genes=100, min_site_size=40, seed=SEED):
    """Write samples_joined.csv, nodes/<node>/{train,val}.csv and the pooled test.csv.

    Each split file has SAMPLE_ID, PATIENT_ID, COHORT (original), NODE, label (1 = IGHV-mutated)
    and the unscaled log2(TPM+1) values of the selected genes. test.csv is the shared held-out
    test set for all nodes (NODE says which cohort each patient belongs to). Standardization is
    not applied here (see standardize_pooled / standardize_local).
    """
    df = build_joined_table(save=True)
    feats = _top_variance_expression(df["SAMPLE_ID"], n_genes)
    y = (df[LABEL_COL] == "mutated").to_numpy(dtype=np.int64)
    node = _node_names(df, min_site_size)
    test_idx, node_idx = split_indices(y, node, df["PATIENT_ID"].to_numpy(), seed)

    table = pd.concat([
        df[["SAMPLE_ID", "PATIENT_ID", "COHORT"]].assign(NODE=node, label=y),
        feats.reset_index(drop=True),
    ], axis=1)
    for s, d in node_idx.items():
        out = PROCESSED_DIR / "nodes" / s
        out.mkdir(parents=True, exist_ok=True)
        for name, idx in d.items():
            table.iloc[idx].to_csv(out / f"{name}.csv", index=False)
    table.iloc[test_idx].to_csv(PROCESSED_DIR / "test.csv", index=False)
    return table, node_idx, test_idx


if __name__ == "__main__":
    table, node_idx, test_idx = export_processed()
    genes = table.columns[5:]
    print(f"joined: {len(table)} samples from {table['PATIENT_ID'].nunique()} patients -> "
          f"{PROCESSED_DIR / 'samples_joined.csv'}")
    print(f"{len(genes)} genes: {', '.join(genes[:12])} ...")
    print(f"{'':10s}{'n':>6s}{'% mutated':>11s}")
    for s, d in node_idx.items():
        for name, idx in d.items():
            print(f"{s + '/' + name:10s}{len(idx):6d}{table['label'].iloc[idx].mean():11.0%}")
    print(f"{'test':10s}{len(test_idx):6d}{table['label'].iloc[test_idx].mean():11.0%}  "
          f"-> {PROCESSED_DIR / 'test.csv'}")
