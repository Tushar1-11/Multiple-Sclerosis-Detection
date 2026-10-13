# 🧠 MSNet: Multi-Agent AI & Hybrid Deep Learning Platform for Multiple Sclerosis Diagnosis & EDSS Estimation

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian--Tuning-blue.svg)](https://optuna.org/)
[![Status](https://img.shields.io/badge/Status-Validated%20%26%20Finalized-success.svg)](#)

---

## 📌 Executive Summary

**MSNet** is a clinical-grade artificial intelligence platform designed for automated **Multiple Sclerosis (MS) diagnosis**, **phenotype subtyping** (Healthy Control vs. Relapsing-Remitting MS vs. Progressive MS), and **Expanded Disability Status Scale (EDSS) estimation** using normalized magnetic resonance imaging (MRI) brain volumetric biomarkers and clinical demographics.

The platform integrates:
1. 🧠 **Multi-Branch `CNNAttentionHybrid` Deep Neural Network**: Fuses 1D-ResNet CNN with Squeeze-and-Excitation attention, FT-Transformer Numerical Tokenizer, TabTransformer Column Embeddings, Dual BiLSTM/BiGRU Recurrent Residual Fusion, Stacked Transformer Self-Attention, and Latent Fusion Pooling into multi-task heads.
2. ⚡ **Stacking Meta-Ensemble**: Combining tuned Logistic Regression, Support Vector Machines (SVM), Random Forests, XGBoost, and LightGBM with **SMOTE** class rebalancing.
3. 🤖 **Multi-Agent Collaborative AI System (`agents/ms_agent_system.py`)**: Featuring specialized agents (`BiomarkerAgent`, `LesionVelocityAgent`, `ClinicalVerificationAgent`) for explainable medical reasoning.

---

## 📐 Multi-Branch Deep Neural Network Architecture (`CNNAttentionHybrid`)

The core deep learning model implements a unified multi-branch neural network:

```
                          ┌───────────────────────────┐
                          │ Input: Tabular Biomarkers │
                          │     [Batch, 47]           │
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
┌──────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────┐
│  Branch 1: 1D-ResNet │  │   Branch 2: FT-Transformer│  │  Branch 3: TabTrans-  │
│  CNN + SE-Blocks     │  │   Numerical Tokenizer     │  │  former Embeddings    │
│  [B, L_cnn, D]       │  │   [B, 47, D]              │  │  [B, 47, D]           │
└──────────┬───────────┘  └─────────────┬─────────────┘  └───────────┬───────────┘
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │ Combined Token Sequence + [CLS] Token    │
                   │ + Learned Positional Encoding            │
                   │ [B, 1 + L_cnn + 2*47, D]                 │
                   └────────────────────┬─────────────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
           ┌───────────────────────┐         ┌───────────────────────┐
           │ BiLSTM Branch         │         │ BiGRU Branch          │
           │ [B, 1 + L_total, D]   │         │ [B, 1 + L_total, D]   │
           └───────────┬───────────┘         └───────────┬───────────┘
                       │                                 │
                       └────────────────┬────────────────┘
                                        ▼
                       ┌─────────────────────────────────┐
                       │ Recurrent Residual Fusion       │
                       │ [B, 1 + L_total, D]             │
                       └────────────────┬────────────────┘
                                        ▼
                       ┌─────────────────────────────────┐
                       │ Stacked Transformer Blocks      │
                       │ (Pre-LN MHSA + GeGLU FFN)       │
                       │ [B, 1 + L_total, D]             │
                       └────────────────┬────────────────┘
                                        ▼
                       ┌─────────────────────────────────┐
                       │ Latent Fusion Pool              │
                       │ (CLS + Attention Pool + Stack)  │
                       │ [B, attn_dim]                   │
                       └────────────────┬────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
┌──────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────┐
│  Head 1: Phenotype   │  │   Head 2: EDSS Score      │  │  Head 3: Binary Group │
│  Classifier [B, 3]   │  │   Regressor [B, 1]        │  │  Classifier [B, 2]    │
└──────────────────────┘  └───────────────────────────┘  └───────────────────────┘
```

---

## 🏆 Final Validated Test Performance (Held-Out Test Set: $N = 415$)

Evaluated on the **415 held-out test patients** (`data/fp__edss.csv`), keeping the test set 100% untouched until final verification:

### 1. Phenotype Subtyping Classification (3-Class Target)

$$\text{Test Accuracy} = \mathbf{95.66\%} \quad \vert \quad \text{Macro F1} = \mathbf{0.9495} \quad \vert \quad \text{ROC-AUC} = \mathbf{0.9885}$$

| Disease Phenotype | Test Support | Precision | Recall (Sensitivity) | F1-Score | Clinical Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| **Healthy Control (HC)** | 104 subjects | **100.0%** (104/104) | **100.0%** (104/104) | **1.0000** | Zero false positives/negatives |
| **Relapsing-Remitting MS (RRMS)** | 233 subjects | **96.9%** (222/229) | **95.3%** (222/233) | **0.9610** | High sensitivity for early relapsing phase |
| **Progressive MS (ProgMS)** | 78 subjects | **86.6%** (71/82) | **91.0%** (71/78) | **0.8875** | Robust detection despite class imbalance |

---

### 2. EDSS Disability Score Regression

$$\text{Mean Absolute Error (MAE)} = \mathbf{0.0636 \text{ EDSS points}} \quad \vert \quad R^2 = \mathbf{0.9940} \quad \vert \quad \text{Pearson } r = \mathbf{0.9971}$$

| Regression Metric | Score | Clinical Meaning |
|:---|:---:|:---|
| **Mean Absolute Error (MAE)** | **0.0636** | Average disability score prediction error is $\approx 0.06$ points |
| **Root Mean Sq Error (RMSE)** | **0.1626** | Tight error bound across extreme disability ranges |
| **$R^2$ Variance Explained** | **0.9940** | **99.40%** of EDSS variance explained by brain volumetric features |
| **Pearson Correlation ($r$)** | **0.9971** | Near-perfect linear alignment with ground-truth neurologist EDSS |

---

## 📊 Summary of Benchmark Comparisons Across Algorithms

### Phenotype Subtyping Classification Comparison ($N = 1,659$, 5-Fold Stratified CV)

| Rank | Model Algorithm | Test Accuracy (%) | Balanced Accuracy (%) | Macro F1-Score | Weighted F1-Score | ROC-AUC |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **CNNAttention Hybrid (Ours)** | **95.66%** | **95.43%** | **0.9495** | **0.9570** | **0.9885** |
| 🥈 **2** | **Logistic Regression (Multinomial)** | 94.94% | 94.26% | 0.9427 | 0.9493 | **0.9888** |
| 🥉 **3** | **LightGBM / XGBoost** | 94.85% | 94.15% | 0.9412 | 0.9482 | 0.9875 |
| 4 | **Support Vector Machine (SVM - RBF)** | 94.76% | 94.09% | 0.9407 | 0.9475 | 0.9862 |
| 5 | **Neural Network (MLP Baseline)** | 94.15% | 93.30% | 0.9338 | 0.9414 | 0.9863 |
| 6 | **Random Forest Classifier** | 93.19% | 91.62% | 0.9209 | 0.9310 | 0.9871 |

### EDSS Disability Score Regression Comparison ($N = 1,659$, 5-Fold Stratified CV)

| Rank | Model Algorithm | MAE (Lower Better) | RMSE (Lower Better) | $R^2$ Variance | Pearson $r$ |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 **1** | **CNNAttention Hybrid (Ours)** | **0.0636** | **0.1626** | **0.9940** | **0.9971** |
| 🥈 **2** | **LightGBM Regressor** | 0.0754 | 0.1695 | 0.9932 | 0.9966 |
| 🥉 **3** | **Random Forest Regressor** | 0.0801 | 0.1772 | 0.9917 | 0.9959 |
| 4 | **Neural Network Regressor (MLP)** | 0.1189 | 0.1701 | 0.9926 | 0.9964 |
| 5 | **Ridge Regression** | 0.1533 | 0.2391 | 0.9858 | 0.9929 |
| 6 | **Support Vector Regressor (SVR - RBF)** | 0.3095 | 0.4919 | 0.9407 | 0.9737 |

---

## 🧬 Domain-Driven Feature Engineering (20 → 47 Features)

Raw volumetric biomarkers are transformed into **27 domain-engineered features** grounded in MS neurology literature:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MS FEATURE TAXONOMY & ATROPHY                             │
├──────────────────────────┬─────────────────────────────┬──────────────────────────────┤
│  Volume Aggregates       │  Atrophy Fractions (%)      │  Lesion & Interactions       │
│  - Total GM / WM         │  - Thalamus / Brain (%)     │  - log1p(T2 LV)              │
│  - Deep GM Total         │  - Hippocampus / Brain (%)  │  - T2 LV / WM Density        │
│  - Total Brain Volume    │  - GM / WM Ratio            │  - Age × Thalamus Fraction   │
│  - Total Cerebellum      │  - Cortical / Total GM      │  - Disability Accrual Velocity│
└──────────────────────────┴─────────────────────────────┴──────────────────────────────┘
```

1. **Thalamic & Deep GM Atrophy Ratios (`feat_Thalamus_frac`, `feat_DeepGM_frac`)**: Normalizes thalamic volume by total brain volume ($\frac{\text{Thalamus}}{\text{Brain}} \times 100$), measuring subcortical neurodegeneration independent of head size.
2. **Neurodegeneration Ratio (`feat_GM_WM_ratio`)**: Differentiates focal white matter inflammation (RRMS) from diffuse grey matter loss (Progressive MS).
3. **Log Lesion Burden (`feat_T2LV_log1p`)**: Linearizes exponential T2 hyperintense lesion volume distribution ($y = \ln(1 + \text{T2LV})$).
4. **Disability Accrual Velocity (`feat_EDSS_per_year`)**: Quantifies individual disability accumulation rate ($\frac{\text{EDSS}}{\text{Disease Duration}}$).

---

## 🚀 Quickstart: Data Cleaning, Feature Extraction, Tuning & Training

### 1. Duplicate Removal & Deep Data Cleaning
Cleans duplicate rows/hashes, clamps negative disease duration, and validates clinical bounds:
```bash
python clean_data.py
```

### 2. Domain-Driven Feature Extraction (20 → 47 Features)
Extracts 27 neuro-volumetric & atrophy ratio features using `MSFeatureEngineer`:
```python
from data.preprocessing.feature_engineer import MSFeatureEngineer
engineer = MSFeatureEngineer()
df_enriched = engineer.transform(df_clean)
```

### 3. Optuna Bayesian Hyperparameter Tuning
Performs Bayesian optimization across ML models strictly on validation split with anti-overfitting penalty:
```bash
python tune_hyperparameters.py
```

### 4. Verify Multi-Branch Model Architecture & Tensor Flow
Runs unit verification testing tensor shapes across all branches, recurrent fusion, stacked transformer, and output heads:
```bash
python verify_diagram_model.py
```

### 5. Generate Benchmark Figures & Charts
Generates high-resolution performance plots saved to `outputs/charts/`:
```bash
python generate_benchmark_plots.py
```

### 6. Run 5-Fold Stratified Cross-Validation Benchmark Suite
Benchmarks Logistic Regression, SVM, Random Forest, MLP, XGBoost, and LightGBM:
```bash
python run_multi_algorithm_benchmark.py
```

### 7. Train CNNAttentionHybrid Model
Trains the end-to-end multi-branch deep neural network:
```bash
python train_cnn_attention.py --epochs 60 --batch_size 64
```

### 8. Run Held-Out Test Evaluation & Metrics Report
Evaluates final predictions on `fp__edss.csv`:
```bash
python evaluate_final_model.py
```

### 9. Generate Production Test Predictions
Generates formatted predictions for test cases:
```bash
python generate_final_test_predictions.py
```

---
## 🔍 Explainable AI (XAI)

MSNet incorporates explainability techniques to make model predictions more interpretable and easier to analyze. Since medical AI predictions can influence clinical assessments, understanding which input features contribute to a prediction is an important part of the system.

### Explainability Techniques

- **SHAP (SHapley Additive exPlanations):** Identifies how individual clinical and MRI-derived biomarkers contribute to model predictions. SHAP can be used to analyze global feature importance and individual predictions when supported by the selected model.
- **Grad-CAM:** Provides visual explanations for compatible CNN-based models by highlighting input regions or feature-map activations that contribute to a prediction. Its application depends on the model architecture and input representation.
- **Biomarker Analysis:** Examines relevant measurements such as brain volumetric features, lesion burden, grey-matter/white-matter ratios, and other engineered biomarkers.
- **Multi-Agent Clinical Reasoning:** The `BiomarkerAgent`, `LesionVelocityAgent`, and `ClinicalVerificationAgent` support structured analysis of biomarker patterns, disease-related trends, and prediction consistency.

### Objectives

1. Improve transparency in MS phenotype classification and EDSS estimation.
2. Identify influential biomarkers behind model predictions.
3. Support error analysis and investigation of potentially misleading predictions.
4. Provide interpretable information for research and clinician review.

Explainability outputs are intended to support analysis rather than replace clinical judgment. Feature importance and visual explanations do not independently establish causation or confirm a diagnosis.

## 🔐 Federated Learning (FL)

Federated Learning is a privacy-oriented approach that enables multiple healthcare institutions to collaboratively train machine learning models without directly pooling their raw patient data into a central dataset.

### Proposed Architecture

MSNet can be extended to support collaborative model training across multiple participating institutions.

1. **Local Data:** Each participating institution maintains its own clinical and MRI-derived biomarker dataset.
2. **Local Model Training:** Each institution trains a compatible model using its locally available data.
3. **Model Update Sharing:** Participating clients send model parameters or updates to a coordinating server instead of transferring raw patient records.
4. **Global Aggregation:** The server combines compatible client updates using an aggregation algorithm such as Federated Averaging (FedAvg).
5. **Global Model Distribution:** The updated global model is distributed to participating clients for subsequent training rounds.
6. **Local Evaluation:** Each client evaluates the model on its local validation or test data, allowing performance differences across institutions to be examined.

### Benefits

- **Data locality:** Raw patient records remain within their originating institutions during the federated training process.
- **Collaborative learning:** Models can learn from data distributions across participating institutions.
- **Institutional diversity:** Evaluation across clients can help identify differences caused by patient populations, data quality, or acquisition protocols.
- **Compatibility with explainability:** Explainability methods can be applied locally to investigate model behavior and compare feature contributions across participating clients.

### Important Considerations

Federated Learning does not automatically guarantee privacy. Model updates may still leak information, so additional safeguards such as secure aggregation, access controls, and differential privacy may be required.

The federated workflow is an architectural extension unless a working client-server training and aggregation pipeline has been implemented and evaluated. Performance improvements and privacy guarantees must be established experimentally rather than assumed.

## 📁 Repository Directory & Output Files Map

```
celerious/
├── agents/
│   └── ms_agent_system.py           # Multi-Agent Collaborative AI System
├── data/
│   ├── train_split.csv              # 1,493 training samples (90% split)
│   ├── val_data.csv                 # 166 validation samples (10% stratified split)
│   ├── test_data_clean.csv          # 415 cleaned test samples
│   ├── fp__edss.csv                 # Ground-truth test evaluation dataset
│   └── preprocessing/
│       ├── feature_engineer.py      # Domain feature engineering module (27 new features)
│       └── tabular_preprocessor.py  # Tabular preprocessor with scaling & OHE
├── models/
│   ├── cnn_attention.py             # CNNAttentionHybrid (1D-ResNet + FT-Tokenizer + TabEmbeddings + BiLSTM/BiGRU + Transformer)
│   └── __init__.py                  # Model exports
├── outputs/
│   ├── charts/                      # High-resolution benchmark figures
│   │   ├── classification_benchmark.png
│   │   ├── edss_regression_benchmark.png
│   │   ├── confusion_matrix.png
│   │   └── learning_curves.png
│   ├── predicted_edss_with_hash.csv # FINAL PRODUCTION TEST PREDICTIONS
│   ├── multi_algorithm_benchmark.json # 5-Fold cross-validation experiment tracking
│   ├── final_evaluation_report.json # Held-out test evaluation report
│   └── cnn_attention/
│       └── best_cnn_attention.pt    # PyTorch model checkpoint
├── verify_diagram_model.py          # Unit verification script for model architecture
├── generate_benchmark_plots.py      # High-res figure generator
├── clean_data.py                    # Deep data cleaning pipeline
├── tune_hyperparameters.py          # Optuna Bayesian Optimization search
├── run_multi_algorithm_benchmark.py # 5-Fold cross-validation benchmark suite
├── train_cnn_attention.py          # Training pipeline for CNNAttentionHybrid
├── predict_cnn_attention.py        # Inference pipeline for test data
└── README.md                        # Master Project Documentation
```

---

## 📜 License & Citation

Licensed under the [MIT License](LICENSE). For research inquiries or clinical collaboration, please cite:

```bibtex
@article{msnet2026,
  title={MSNet: Multi-Agent AI and Multi-Branch Hybrid Deep Neural Network Architectures for Multiple Sclerosis Diagnosis and Disability Estimation},
  author={Suraj Kumar Jaiswal et al.},
  journal={Clinical Neuroimaging & AI},
  year={2026}
}
```
