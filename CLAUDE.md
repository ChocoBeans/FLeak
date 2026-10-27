# FLeak

Mini-project: federated learning (FL) on CLL RNA-seq data (cBioPortal `cll_broad_2022`) to test
(1) whether FL matches centralized performance and (2) how much raw patient data leaks from FL
updates (gradient inversion, membership inference) and what defenses (DP, secure aggregation) cost.

- `README.md` — project plan, task definition, threat model, setup and run instructions.
- `DATA_NOTES.txt` — every data selection/preprocessing decision with its reasoning and the
  limitations. Read it before changing anything in `utils/preprocess.py`; update it when a data
  decision changes.

## Commands

```bash
source venv/bin/activate
pip install -r requirements.txt
python -m utils.preprocess      # run from the repo root; downloads data on first run, writes data/processed/
```

Python 3.11.4+ (developed on 3.12). First run needs internet (study archive ~136 MB, GENCODE v19
annotation ~38 MB into `data/reference/`). `data/` and `results/` are gitignored.

## Layout

- `utils/preprocess.py` — download, join/filter, gene selection, splits, CSV export (`python -m utils.preprocess`).
- `utils/standardization.py` — `standardize_pooled` / `standardize_local`.
- `utils/train.py`, `utils/dataset.py` — shared training/evaluation and CSV reader (stubs, no code yet).
- `models/logistic_regression.py`, `models/mlp.py` — the two models (raw-logit output, train with
  `BCEWithLogitsLoss`); `models.build_model("logreg" | "mlp", n_features)` builds either. Logistic
  regression is the main model; the MLP is the comparison.
- `centralized/run.py` — centralized baseline (stub).
- `federated/{task,client_app,server_app,custom_strategy}.py` + `federated/pyproject.toml` — imported
  Flower quickstart-pytorch example (CIFAR-10 CNN, FedAdagrad, W&B logging), only the import paths and
  the `pyproject.toml` component paths changed (`pytorchexample.` -> `federated.`). It is a template:
  it still uses CIFAR-10. `custom_strategy.py` is kept for reference only (unrelated to the project for
  now; edit it later when deeper into the project). The earlier stub files `client.py`, `strategy.py`,
  `run.py` were deleted as redundant. `flwr[simulation]` and `torch` are in `requirements.txt` and match `federated/pyproject.toml`; keep
  the two in sync. The vision/W&B packages (`flwr-datasets`, `torchvision`, `wandb`) were left out
  on purpose (no CNNs), so `task.py` and `custom_strategy.py` do not import until they are adapted
  (they use `datasets`, `flwr_datasets`, `torchvision`, `wandb`).
- `attacks/`, `notebooks/`, `results/` — empty so far. Results go to `results/centralized/` and `results/federated/`.
- `data/` — raw study, GENCODE reference and `processed/` outputs (all gitignored).

Dependency direction: `utils` <- `models` <- `centralized`, `federated`, `attacks`. `centralized` and
`federated` must not import each other; both use the same `utils.train` and the same hyperparameters.
Run modules from the repo root (`python -m centralized.run`); the Flower app in `federated/` is launched with `flwr run`.

In code: `from utils.preprocess import load_cll, make_splits` and
`from utils.standardization import standardize_pooled, standardize_local`. There is no reader for the
`data/processed/` CSVs yet (`utils/dataset.py` is a stub); `load_cll` + `make_splits` give the same split.

## Status

### Done
- Dataset chosen: CLL (Broad, Nat Genet 2022). Task: IGHV mutated (1) vs unmutated (0) from gene
  expression. Replaced the original TCGA-BRCA plan (no ER/HER2 labels in that study).
- `utils/preprocess.py`: auto-download, join sample + patient tables, filter to samples in the
  expression matrix and with an IGHV label (700 samples, 696 patients), gene filtering with
  GENCODE v19 (drops chrY/XIST, pseudogenes, Ig loci, small RNAs, noisy transcripts), restores
  date-mangled symbols (`10-Sep` -> `SEPT10`), top-100 variance genes, log2(TPM+1).
- Nodes = contributing cohort (ICGC, GCLLSG, DFCI, OTHER); patient-level train/val per node and a
  shared 20% test set, seed 4171. Outputs in `data/processed/` (`nodes/<node>/{train,val}.csv`,
  `test.csv`, `samples_joined.csv`) were verified against the raw files.
- `standardize_pooled` (centralized) and `standardize_local` (federated) implemented in
  `utils/standardization.py`; applied at load time, not stored in the CSVs (which hold unscaled
  log2(TPM+1) values).
- Models written in PyTorch (so Flower, Opacus and the attacks can reuse them): logistic regression
  and a small MLP, one file each in `models/`. An earlier single-file attempt was removed.
- Project scaffold created: `models/`, `centralized/`, `federated/` (renamed from `fl/`), plus stub
  `utils/train.py` and `utils/dataset.py`. README and DATA_NOTES are up to date with the above.

### To do
- Centralized baseline (`centralized/run.py`) on pooled training data, evaluated on the shared test set (pooled and per node).
- FedAvg with Flower (`federated/`) across the 4 nodes (same model, same test set, local standardization).
- Metrics: AUC-ROC primary; also accuracy, F1, precision, recall; pooled and per node; mean ± std
  over seeds.
- Attacks in `attacks/`: gradient inversion (cosine similarity of reconstruction) and membership
  inference (AUC, membership counted per patient).
- Defenses: DP via Opacus (epsilon sweep 1, 3, 8, inf) and secure aggregation (simulated);
  privacy-utility and leakage-reduction curves.
- Open questions: feature set size (try 50 / 100 / 500 genes); whether four nodes (smallest has 34
  train samples) are enough for stable training; Flower setup that exposes per-client gradients;
  week 7 proposal framing.

## Decisions and gotchas

- Keep the model simple (logistic regression / small MLP) so the attacks stay interpretable.
- Immunoglobulin genes are excluded on purpose: they are a near-proxy for the IGHV label and
  clone fingerprints. Sex-linked genes are excluded on purpose. X-linked genes are kept.
- Do not compute scaling statistics from validation or test data. Centralized uses pooled train
  statistics; each FL node uses its own train statistics (a known limitation, listed in DATA_NOTES).
- Splits are by patient, never by sample: 4 patients have two samples (all in OTHER's train set).
  Count membership per patient in the membership-inference attack.
- OTHER (UCSD + MDACC + NHLBI) is tiny: 34 train / 8 val / 10 test samples, so its per-node
  metrics are noisy. GCLLSG (34% mutated) differs most from the pooled 51%.
- Seeds should vary model init and batch order; the data split uses `SEED = 4171` in
  `utils/preprocess.py`. Variation from different splits is not captured unless added.
- Gene variance selection uses all 700 samples (label-free); noted as a limitation.
