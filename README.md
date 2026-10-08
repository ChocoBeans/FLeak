# FLeak

*2026-10-08*

Mini-project proposal: federated learning on TCGA Breast Invasive Carcinoma (BRCA) data via cBioPortal, testing whether FL matches centralized performance and how much raw patient data leaks from FL model updates.

## Motivation

Federated learning (FL) is increasingly proposed as a way to let multiple institutions collaboratively train models on sensitive biomedical data (genomic, clinical, imaging) without centralizing raw patient records. Published results show real gains from this approach: breast density classification accuracy improved 6% (generalizability up 46%), COVID-19 outcome prediction improved 16–38% at 24h/72h, and rare tumour segmentation improved substantially under FL versus single-site training.

However, FL is frequently described as "privacy-preserving" without that claim being rigorously tested. Reviews of FL frameworks in biomedical research note scarce integration of privacy-preserving techniques (differential privacy, secure aggregation) and a predominant reliance on simpler, less protected architectures — meaning raw data may still be at risk of leaking through shared model updates via gradient inversion or membership inference attacks.

This project has two goals:
1. Demonstrate that FL can match centralized model performance on a real cancer classification task across simulated institutional silos.
2. Measure how much raw patient data actually leaks from vanilla FL updates, then apply and evaluate defenses (differential privacy, secure aggregation) to close that gap — reporting the resulting privacy-utility tradeoff.

## Dataset & goal

**Dataset:** cBioPortal — TCGA Breast Invasive Carcinoma (BRCA), ~1,100 patients.

**Task:** Binary classification — HER2-positive vs. HER2-negative (or ER-positive vs. ER-negative) breast cancer, using gene expression (RNA-seq) as input features.

**Features:** A curated panel of ~50–200 known breast cancer genes (e.g., PAM50 panel or a COSMIC cancer gene census subset), not whole-exome data, to keep preprocessing light.

**Node split:** Split patients by tissue source site (TSS) / contributing institution listed in cBioPortal for TCGA-BRCA. This simulates realistic "different hospitals as different nodes," with natural heterogeneity (demographics, sequencing batches) — the non-IID condition FL needs to prove itself against.

## Why BRCA over other cancer types

- Largest TCGA cohort, so a multi-institution node split still leaves enough data per node.
- Well-documented and used in prior FL literature (e.g., breast density classification work reported +6% accuracy, +46% generalizability under FL) — gives a benchmark to compare against and cite.
- Clean binary label available (HER2 or ER/PR receptor status) — a real clinical task without survival analysis or censored data.

## Part 1: Centralized vs. Federated

| Item | Choice |
| --- | --- |
| Train | Classifier (logistic regression or small MLP) on curated breast cancer gene panel |
| Label | HER2 or ER receptor status (binary) |
| Nodes | Split by TCGA-BRCA contributing institution (TSS) |
| Metrics | Recall (primary), plus precision, F1, AUC-ROC |
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

**Core result:** Privacy-utility tradeoff curve (recall vs. ε) and leakage-reduction curve (attack success vs. ε). Look for a "knee" in the curve — a privacy budget where attack success drops sharply while accuracy cost stays small.

## Next steps / open questions

- [ ] Decide cBioPortal access route: REST API vs. bulk download
- [ ] Confirm TSS/institution split has enough patients per node for stable training
- [ ] Pick final gene panel (PAM50 vs. broader cancer gene census subset)
- [ ] Choose Flower setup for server-side gradient interception needed for the attack
- [ ] Frame for week 7 proposal: two research questions — (1) does FL match centralized performance for cancer classification across institutional silos, and (2) how much raw patient data leaks from FL updates, and what's the cost of closing that leak

## Project layout

```
FLeak/
├── data/        # raw and preprocessed TCGA-BRCA data from cBioPortal (gitignored)
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
- [Technical and Legal Aspects of Federated Learning in Bioinformatics: Applications, Challenges and Opportunities](https://arxiv.org/pdf/2503.09649) — documents FL's measured gains in breast density classification, COVID-19 outcome prediction, and rare tumour segmentation.
- [Eleven quick tips for Biomedical Federated Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC13496424/) — practical guidance on setting up biomedical FL projects, including use of public data for initial test scenarios.
- [Federated Learning for Histopathology Image Classification: A Systematic Review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12785327/) — PRISMA review of 24 FL studies in histopathology (2020–2025), showing steady growth in that subfield.
- [pfl-research: simulation framework for accelerating research in Private Federated Learning](https://arxiv.org/pdf/2404.06430) — reference for FL simulation frameworks and private FL research tooling.
- [Real-World Image Datasets for Federated Learning](https://arxiv.org/pdf/1910.11089) — background on non-IID data partitioning for FL benchmarks.
