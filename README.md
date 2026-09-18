# TabNet-Based Wi-Fi Cyberattack Classification

A deep-learning approach to network intrusion detection on Wi-Fi traffic, evaluating how **feature selection** and **class balancing** techniques affect **TabNet's** performance on a heavily imbalanced, high-dimensional dataset.

## Overview

This project applies **TabNet** — an attention-based deep learning architecture for tabular data — to classify cyberattacks on Wi-Fi networks using the **AWID3** dataset. Rather than using TabNet out of the box, the project systematically evaluates how different **feature selection** and **class balancing** strategies affect classification performance, across nine experimental scenarios.

## Dataset

**AWID3** (Aegean Wi-Fi Intrusion Dataset 3), published by the University of the Aegean (Chatzoglou et al., 2021). This project uses four classes:

| Class | Observations | Share |
|---|---:|---:|
| Normal | 7,779,003 | 97.49% |
| Malware | 131,611 | 1.65% |
| Botnet | 56,891 | 0.71% |
| SSH Brute Force | 11,882 | 0.15% |
| **Total** | **7,979,387** | 100% |

The imbalance ratio between the majority class (Normal) and the smallest minority class (SSH Brute Force) is roughly **655:1** — the core motivation for testing class-balancing techniques.

Not included in this repo due to size (~8M rows). Original source: [AWID3 dataset page](https://icsdweb.aegean.gr/awid/awid3).

## Method

- **Model**: TabNet — sequential attention-based feature selection for tabular deep learning
- **Feature selection**: Mutual Information (MI), Random Forest Feature Importance (RFFI)
- **Class balancing**: SMOTE, SMOTE-Tomek
- **Preprocessing**: two-stage pipeline (per-class cleaning → combine/transform/split/scale) to keep peak RAM manageable on a 16 GB machine; missing-value and constant-feature removal (254 → 21 features); data-leakage detection and removal (dropped `frame.time_epoch`, `frame.time_relative`, `radiotap.timestamp.ts` — features with zero-overlap value ranges across classes)
- **Experimental design**: 9 scenarios crossing {none, MI, RFFI} × {none, SMOTE, SMOTE-Tomek}

## Results

Macro-averaged metrics across all nine scenarios:

| Scenario | Feature Selection | Balancing | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---:|---:|---:|---:|
| 1 | — | — | 0.9931 | 0.9540 | 0.8608 | **0.9034** |
| 2 | MI | — | 0.9818 | 0.8446 | 0.6425 | 0.6927 |
| 3 | RFFI | — | 0.9922 | 0.9005 | 0.8449 | 0.8656 |
| 4 | — | SMOTE | 0.9931 | 0.8675 | 0.8877 | 0.8654 |
| 5 | — | SMOTE-Tomek | 0.9642 | 0.5823 | 0.9608 | 0.6935 |
| 6 | MI | SMOTE | 0.9932 | 0.9322 | 0.8758 | 0.9016 |
| 7 | MI | SMOTE-Tomek | 0.9303 | 0.4518 | 0.9374 | 0.5547 |
| 8 | RFFI | SMOTE | 0.9878 | 0.8750 | 0.7805 | 0.7959 |
| 9 | RFFI | SMOTE-Tomek | 0.9582 | 0.5247 | 0.9537 | 0.6406 |

**Key findings:**
- The unmodified baseline (Scenario 1) and MI + SMOTE (Scenario 6) achieved the highest macro F1-scores (0.9034 and 0.9016), suggesting TabNet's built-in instance-wise feature selection already handles this classification task well without much external help.
- SMOTE-Tomek scenarios (5, 7, 9) pushed minority-class recall above 0.93, but at a steep cost to precision — net F1 ends up lower than baseline.
- RFFI was generally more stable than MI when used without balancing (compare Scenario 2 vs. 3).

## Repo Structure

```
.
├── awid3_preprocessing.py   # Data cleaning, leakage removal, transform, split, scale
├── run_scenario.py          # Feature selection + balancing + TabNet training/evaluation per scenario
├── requirements.txt
├── results/                 # Confusion matrices, feature importance plots, per-scenario metrics
└── README.md
```

## Reproducing Results

1. Download the AWID3 dataset and place it locally (update the path in `awid3_preprocessing.py`).
2. Install dependencies: `pip install -r requirements.txt`
3. Run `awid3_preprocessing.py` (set `TARGET_CLASS` for each of SSH/Botnet/Malware, per the two-stage design) to produce the cleaned, split, scaled dataset.
4. Run `run_scenario.py`, setting `SCENARIO` to 1–9, to train TabNet under each feature selection × balancing combination and produce metrics, confusion matrices, and plots.

## Notes

- Machine used for experiments: 11th Gen Intel Core i5-11400H, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU, Python 3.13.
- Preprocessing runs on CPU (Pandas/NumPy-bound); TabNet training uses GPU via CUDA.
