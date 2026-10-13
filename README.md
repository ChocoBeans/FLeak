# FLeak

*2026-10-08*

Mini-project proposal: federated learning on Chronic Lymphocytic Leukemia (CLL, Broad, Nat Genet 2022) data via cBioPortal, testing whether FL matches centralized performance and how much raw patient data leaks from FL model updates.

## Motivation

Federated learning (FL) is increasingly proposed as a way to let multiple institutions collaboratively train models on sensitive biomedical data (genomic, clinical, imaging) without centralizing raw patient records. Published results show real gains from this approach: COVID-19 outcome prediction improved 16–38% at 24h/72h, and rare tumour segmentation improved substantially under FL versus single-site training.

However, FL is frequently described as "privacy-preserving" without that claim being rigorously tested. Reviews of FL frameworks in biomedical research note scarce integration of privacy-preserving techniques (differential privacy, secure aggregation) and a predominant reliance on simpler, less protected architectures — meaning raw data may still be at risk of leaking through shared model updates via gradient inversion or membership inference attacks.

This project has two goals:
1. Demonstrate that FL can match centralized model performance on a real cancer classification task across simulated institutional silos.
2. Measure how much raw patient data actually leaks from vanilla FL updates, then apply and evaluate defenses (differential privacy, secure aggregation) to close that gap — reporting the resulting privacy-utility tradeoff.

## Dataset & goal

**Dataset:** cBioPortal — Chronic Lymphocytic Leukemia (Broad, Nature Genetics 2022), study ID `cll_broad_2022`. 1,154 samples in the study, 715 with RNA-seq; 700 samples (from 696 patients) have both RNA-seq and an IGHV label.

**Task:** Binary classification — IGHV-mutated (M-CLL) vs. IGHV-unmutated (U-CLL) CLL, a lab-measured marker that separates two clinically distinct forms of the disease with different prognosis and treatment. Input is gene expression (RNA-seq, TPM). Classes are roughly balanced (51% mutated).

**Features:** The 100 highest-variance genes, log2(TPM+1) transformed, after excluding immunoglobulin genes (they encode the label), sex-linked genes, and noisy/non-coding transcripts. No curated panel is assumed.

**Node split:** Samples are grouped by contributing cohort (`COHORT`: ICGC, GCLLSG, DFCI, NHLBI, UCSD, MDACC), pooling cohorts under 40 patients into "OTHER". This simulates different institutions as different nodes. This gives 4 nodes (n = 289 / 229 / 130 / 52 samples) whose label rates differ (34%–63% mutated) — the non-IID condition FL needs to prove itself against.

**Train/val/test:** Splits are made at the patient level (a patient's samples never straddle splits). A shared test set (20% of patients, stratified by cohort and label) is held out first and used for both centralized and federated models; the rest is split within each node into train/val (80/20).

**Preprocessing:** Features are z-scored using training data only — pooled statistics for the centralized model, each node's own statistics for federated learning. Selection and preprocessing choices are documented in `DATA_NOTES.txt`.

## Why CLL over other cancer types

- Recent (2022), large, and open, with RNA-seq and a clean binary clinical label — no survival analysis or censored data.
- The label is lab-measured, not derived from the expression data, so the task is not circular.
- Multiple contributing cohorts give a natural multi-institution split with heterogeneous label rates.
- Limitation: only four usable nodes after filtering, and no direct FL benchmark from prior work for this dataset.

## Part 1: Centralized vs. Federated

| Item | Choice |
| --- | --- |
| Train | Classifier (logistic regression or small MLP) on the top-variance gene set |
| Label | IGHV mutation status (binary) |
| Nodes | Split by contributing cohort (`COHORT`) |
| Metrics | AUC-ROC (primary), plus accuracy, F1, precision, recall; reported pooled and per node, as mean ± std over seeds |
| Comparison | Centralized model vs. FedAvg (Flower) |

Keep the model simple (logistic regression / small MLP) so the threat-model attacks in Part 2 are easier to implement and interpret.

## Part 2: Threat model

**Attacks:**
- Gradient inversion attack — reconstruct approximate patient feature vectors from a client's shared gradient update.
- Membership inference attack — determine whether a given patient's record was in a specific site's training set.

**Attack success metrics:**
- Reconstruction similarity (e.g., cosine similarity between reconstructed and real feature vectors) for inversion.
- AUC for membership inference.

**Defenses:**
- Differential privacy via Opacus — add calibrated noise to gradients, sweep privacy budgets (ε = 1, 3, 8, ∞).
- Secure aggregation — simulated via Flower's support or a simplified masking scheme, so the server only sees summed updates, never individual client gradients.

**Core result:** Privacy-utility tradeoff curve (AUC-ROC vs. ε) and leakage-reduction curve (attack success vs. ε). Look for a "knee" in the curve — a privacy budget where attack success drops sharply while accuracy cost stays small.

## Next steps / open questions

- [x] Decide cBioPortal access route: bulk download (`https://datahub.assets.cbioportal.org/cll_broad_2022.tar.gz`)
- [ ] Confirm cohort split has enough patients per node for stable training (4 nodes, smallest n=52)
- [ ] Pick final feature set size (top-N variance genes; try N = 50 / 100 / 500)
- [ ] Choose Flower setup for server-side gradient interception needed for the attack
- [ ] Frame for week 7 proposal: two research questions — (1) does FL match centralized performance for cancer classification across institutional silos, and (2) how much raw patient data leaks from FL updates, and what's the cost of closing that leak

## Project layout

```
FLeak/
├── data/        # raw and preprocessed CLL data from cBioPortal (gitignored)
├── fl/          # Flower client/server code
├── attacks/     # gradient inversion and membership inference attack implementations
├── results/     # metrics output (CSV, gitignored)
├── notebooks/   # exploratory analysis
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Research referenced

- [Efficacy of federated learning on genomic data: a study on the UK Biobank and the 1000 Genomes Project](https://pmc.ncbi.nlm.nih.gov/articles/PMC10937521/) — FL applicability for phenotype and ancestry prediction; shows federated models approach centralized performance despite inter-node heterogeneity.
- [Federated learning frameworks: quality and interoperability for biomedical research](https://pmc.ncbi.nlm.nih.gov/articles/PMC12862364/) — review noting scarce integration of privacy-preserving techniques and reliance on simpler architectures in current FL frameworks.
- [Federated Learning for the pathogenicity annotation of genetic variants](https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaf523/64325337/btaf523.pdf) — multi-institutional FL vs. centralized comparison for coding SNVs, non-coding SNVs, and CNVs; found larger centralized-vs-FL divergence for CNVs specifically.
- [Technical and Legal Aspects of Federated Learning in Bioinformatics: Applications, Challenges and Opportunities](https://arxiv.org/pdf/2503.09649) — documents FL's measured gains in COVID-19 outcome prediction and rare tumour segmentation.
- [Eleven quick tips for Biomedical Federated Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC13496424/) — practical guidance on setting up biomedical FL projects, including use of public data for initial test scenarios.
- [Federated Learning for Histopathology Image Classification: A Systematic Review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12785327/) — PRISMA review of 24 FL studies in histopathology (2020–2025), showing steady growth in that subfield.
- [pfl-research: simulation framework for accelerating research in Private Federated Learning](https://arxiv.org/pdf/2404.06430) — reference for FL simulation frameworks and private FL research tooling.
- [Real-World Image Datasets for Federated Learning](https://arxiv.org/pdf/1910.11089) — background on non-IID data partitioning for FL benchmarks.
