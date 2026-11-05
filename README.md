# Genetic Algorithm-Based Missing Value Imputation: Conditional Superiority and Deployment-Time Fragility Under Nested Cross-Validation

Official code repository and replication benchmark for the paper:

> **Genetic Algorithm-Based Missing Value Imputation: Conditional Superiority and Deployment-Time Fragility Under Nested Cross-Validation**  
> Moch. Faiz Febriawan, Kartika Chandra Dewi, and Hasanuddin Al-Habib  
> *CAUCHY: Jurnal Matematika Murni dan Aplikasi*, 2026.

---

## 📌 Overview

Missing data poses a pervasive challenge in tabular analytics. While genetic algorithm (GA) imputation optimizing downstream classifier fitness has demonstrated promising results in prior literature, existing evaluations have predominantly operated under idealized **clean benchmark assumptions** where test data remain fully observed.

This repository implements a rigorous, deployment-aware evaluation of evolutionary imputation paradigms (**global GA-based** and **localized GA-KNN**) evaluated under a **5-fold outer by 3-fold inner nested cross-validation protocol** across:
- **6 Missingness Proportions**: $15\%, 30\%, 45\%, 60\%, 75\%, 90\%$ (MCAR)
- **5 Random Seeds**: $42, 101, 202, 303, 404$
- **7 Imputation Paradigms**: Mean, Median, KNN, MICE, MissForest, GA-based, GA-KNN
- **4 Downstream Tuning Regimes**: Default RF, RandomizedSearchCV, HalvingSearchCV, Optuna (TPE)
- **Total Factorial Models**: $5 \times 6 \times 5 \times 7 \times 4 = 4{,}200$ outer models (yielding $8{,}400$ dual-scenario performance evaluations).

### 🔬 Core Empirical Findings
1. **Conditional Superiority Under Clean Validation**: GA-KNN achieves conditional discriminative advantages over standard unsupervised KNN across severe sparsity regimes ($p_{\text{corr}} < 0.001$). Under $75\%$ missingness, GA-KNN attains a modest numerical gain over MissForest ($\Delta = +0.0058$ ROC-AUC); however, under conservative Nadeau–Bengio cross-validation dependence correction, this difference is not statistically significant ($t_{\text{corr}} = 1.46, p_{\text{corr}} = 0.156$).
2. **Acute Deployment Fragility**: When evaluated under realistic test-time conditions where incoming inference queries are corrupted by missingness, GA-based methods suffer the steepest performance collapse ($\Delta\text{ROC-AUC} = -0.1200$), dropping to $0.8365 \pm 0.1122$, where they are significantly outperformed by simple Mean ($0.8644 \pm 0.0935, p_{\text{Holm}} < 0.00001$) and Median ($0.8626 \pm 0.0943, p_{\text{Holm}} < 0.00001$) across $60\%$--$90\%$ missingness due to metric distortion in high-dimensional subspace projection.
3. **Reconstruction–Discrimination Decoupling**: Minimizing reconstruction error (RMSE/PFC) does not monotonically maximize downstream classification performance. MissForest attains the lowest imputation reconstruction error ($1048.5 \pm 391.3$ RMSE) but is outperformed in downstream discriminative ROC-AUC by simple Mean imputation ($1605.8 \pm 35.3$ RMSE) under severe deployment missingness.

---

## 🏗️ Zero-Leakage Architecture (Algorithm 1)

The evaluation enforces a strict three-stage quarantine barrier to eliminate data leakage:

```
[Full Dataset (N=10,127, D=19)]
        │
        ▼ (5-Fold Stratified Outer Partitioning)
 ┌─────────────────────────────────────────────────────────┐
 │ Outer Training Set (80%, N=8,101)                       │
 │  ├── Stage 1: Inner-Fold Preprocessing & GA Evolution   │
 │  │    └── 3-Fold Inner CV Fitness Evaluation (RF proxy) │
 │  ├── Stage 2: Inner-Fold Downstream Classifier Tuning   │
 │  │    └── 3-Fold Inner CV (Default, Random, Halving,    │
 │  │        Optuna TPE)                                   │
 └─────────────────────────────────────────────────────────┘
        │
   [QUARANTINE BARRIER: Holdout Partition Strictly Locked]
        │
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 3: Out-of-Sample Holdout Evaluation (N=2,026)     │
 │  ├── (A) Clean Outer Validation (Benchmark Control)     │
 │  └── (B) Incomplete Deployment (Post-Split MCAR +       │
 │          Inductive k-NN Reference Mapping)              │
 └─────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
nested-cv-ga-imputation/
├── README.md                          # Comprehensive documentation and reproduction guide
├── environment.yml                    # Conda virtual environment definition
├── requirements.txt                   # Pip dependency requirements
├── LICENSE                            # MIT Open Source License
├── data/
│   ├── BankChurners.csv               # Kaggle benchmark dataset (N=10,127, D=19)
│   └── README.md                      # Dataset documentation and preprocessing details
├── notebooks/
│   ├── p1_9.ipynb                     # Main 4,200 nested CV factorial experiment notebook
│   └── p1_9_sensitivity.ipynb         # GA hyperparameter sensitivity analysis notebook
├── scripts/
│   └── p1_9_sensitivity.py            # Standalone sensitivity analysis execution script
└── results/
    ├── csv/
    │   ├── ga_nested_cv_results_p1_9.csv      # Full raw results across all 4,200 model runs
    │   ├── ga_sensitivity_results.csv         # OFAT parameter sensitivity metrics (Table 11)
    │   ├── pairwise_stats_clean_p1_9.csv      # Paired t-test results under Clean scenario (Table 9)
    │   ├── pairwise_stats_incomplete_p1_9.csv  # Paired t-test results under Incomplete scenario (Table 10)
    │   └── pairwise_nested_cv_stats_p1_9.csv  # Multiplicity-corrected inferential statistics
    └── figures/
        ├── fig_model_hierarchy.png            # Experimental workflow diagram (Figure 3)
        ├── ga_nested_cv_plot_p1_9.png         # Downstream performance trajectories
        ├── evaluasi_metrik_clean_val.png      # Clean holdout metric distributions
        └── evaluasi_metrik_incomp_val.png     # Incomplete deployment metric distributions
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

Clone this repository and create the Conda virtual environment:

```bash
git clone https://github.com/faizgeorgie9/nested-cv-ga-imputation.git
cd nested-cv-ga-imputation

# Create and activate environment using Conda
conda env create -f environment.yml
conda activate nested-cv-ga-imputation

# Or install dependencies via pip
pip install -r requirements.txt
```

### 2. Running Experiments

#### Main Factorial Nested Cross-Validation (4,200 Models)
Launch Jupyter Notebook and open `notebooks/p1_9.ipynb`:

```bash
jupyter notebook notebooks/p1_9.ipynb
```
*Note: Due to the comprehensive factorial matrix ($5 \text{ folds} \times 6 \text{ rates} \times 5 \text{ seeds} \times 7 \text{ imputers} \times 4 \text{ tuners} = 4{,}200 \text{ models}$), complete re-execution requires substantial computational runtime. All raw factorial results are pre-computed and stored in `results/csv/ga_nested_cv_results_p1_9.csv` for immediate inspection and analysis.*

#### GA Hyperparameter Sensitivity Analysis
To reproduce the One-Factor-At-A-Time (OFAT) parameter sensitivity analysis (Table 11 in the manuscript):

```bash
python scripts/p1_9_sensitivity.py
# Or run interactively in: notebooks/p1_9_sensitivity.ipynb
```

---

## 📊 Summary of Main Results

### Trajectory across Escalating Missingness (ROC-AUC)

| Imputation Method | Clean (15%) | Clean (60%) | Clean (75%) | Incomp (15%) | Incomp (60%) | Incomp (75%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean** | $0.9799$ | $0.9622$ | $0.9482$ | $0.9663$ | $\mathbf{0.8662}$ | $\mathbf{0.7995}$ |
| **Median** | $0.9798$ | $0.9602$ | $0.9457$ | $0.9660$ | $0.8647$ | $0.7975$ |
| **KNN** | $0.9787$ | $0.9440$ | $0.9172$ | $0.9611$ | $0.8150$ | $0.7323$ |
| **MICE** | $0.9808$ | $0.9535$ | $0.9272$ | $0.9671$ | $0.8403$ | $0.7697$ |
| **MissForest** | $\mathbf{0.9830}$ | $\mathbf{0.9617}$ | $0.9413$ | $\mathbf{0.9687}$ | $0.8469$ | $0.7690$ |
| **GA-based** | $0.9828$ | $0.9606$ | $0.9464$ | $0.9658$ | $0.8337$ | $0.7504$ |
| **GA-KNN** | $0.9827$ | $0.9612$ | $\mathbf{0.9471}$ | $0.9659$ | $0.8340$ | $0.7503$ |

---

## 📖 Citation

If you use this codebase or benchmark results in your research, please cite our paper:

```bibtex
@article{febriawan2026genetic,
  author    = {Febriawan, Moch. Faiz and Dewi, Kartika Chandra and Al-Habib, Hasanuddin},
  title     = {Genetic Algorithm-Based Missing Value Imputation: Conditional Superiority and Deployment-Time Fragility Under Nested Cross-Validation},
  journal   = {CAUCHY: Jurnal Matematika Murni dan Aplikasi},
  volume    = {11},
  number    = {2},
  pages     = {1--20},
  year      = {2026},
  doi       = {10.18860/cauchy.v11i2.xxxxx}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

