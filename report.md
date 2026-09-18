# TabNet-Based Wi-Fi Cyberattack Classification: A Comparative Study of Feature Selection and Class Balancing Techniques

## Abstract

Wi-Fi networks are ubiquitous but remain a common target for cyberattacks, and building an effective Intrusion Detection System (IDS) for them is complicated by two recurring problems: very high-dimensional traffic data and severe class imbalance between normal and attack traffic. This project investigates how two feature selection techniques (Mutual Information and Random Forest Feature Importance) and two class-balancing techniques (SMOTE and SMOTE-Tomek) affect the performance of **TabNet**, an attention-based deep learning architecture for tabular data, when classifying Wi-Fi cyberattacks on the AWID3 dataset. Nine experimental scenarios — combining each feature selection method with each balancing method, plus a baseline — were evaluated on identical held-out test data. The unmodified baseline achieved the best overall balance of metrics (99.31% accuracy, 90.34% macro F1-score), while balancing techniques improved minority-class recall at a cost to precision, illustrating a genuine trade-off between detection sensitivity and overall stability that a real IDS deployment would need to weigh deliberately.

---

## 1. Introduction

### 1.1 Background

Wi-Fi has become one of the most common ways people connect to the internet, and its ubiquity in homes, offices, campuses, and public spaces makes it a critical part of everyday digital life. That same ubiquity makes it an attractive target: wireless networks are exposed to attacks such as sniffing, spoofing, deauthentication, injection, and denial-of-service, any of which can compromise user data or destabilize a network. Cyberattacks of this kind are not a hypothetical risk — national cybersecurity monitoring in the source country of this research recorded billions of traffic anomalies within a seven-month period, the majority of them malware-related, alongside several high-profile ransomware incidents affecting public infrastructure and financial services in the same period.

An Intrusion Detection System (IDS) monitors network activity and flags potential attacks, either by matching known attack signatures or by detecting anomalous behavior. Traditional signature-based IDS struggles against attack patterns it hasn't seen before, while anomaly-based IDS tends to produce a high rate of false positives. These limitations have driven growing interest in machine learning (ML) and deep learning (DL) based IDS, which can learn attack patterns directly from historical traffic data and generalize to variations they weren't explicitly trained on.

A common benchmark dataset for this kind of research is **AWID3** (Aegean Wi-Fi Intrusion Dataset 3), published by the University of the Aegean, which captures realistic Wi-Fi traffic across a range of attack types against the IEEE 802.11 protocol. Like most real-world IDS datasets, AWID3 presents two major modeling challenges:

1. **High dimensionality** — hundreds of statistical traffic features, many of which are irrelevant or redundant, which can slow training and hurt accuracy if left unaddressed. Feature selection methods (filter-based, such as Mutual Information; embedded, such as Random Forest Feature Importance) are a standard way to address this.
2. **Severe class imbalance** — normal traffic vastly outnumbers attack traffic, biasing models toward the majority class and producing poor recall on the attacks that actually matter to detect. Balancing techniques such as SMOTE (synthetic oversampling of minority classes) and SMOTE-Tomek (SMOTE combined with removal of overlapping class boundary samples) are commonly used to counteract this.

**TabNet** is a relatively recent deep learning architecture purpose-built for tabular data. It uses a sequential attention mechanism with sparsemax-based feature selection, giving it built-in, learned, per-instance feature selection alongside the representational power of deep learning — while also offering some interpretability into which features drove a given prediction. This makes it a natural candidate for tabular, high-dimensional IDS data like AWID3, and the central question of this project is how TabNet's *own* built-in feature selection interacts with *external* feature selection and balancing techniques applied ahead of training.

### 1.2 Problem Statement

1. How can a TabNet model be built under different combinations of feature selection and balancing techniques for Wi-Fi cyberattack classification?
2. How does performance vary across those combinations?

### 1.3 Objectives

1. Build TabNet models under a systematic set of feature selection × balancing scenario combinations for Wi-Fi cyberattack classification.
2. Measure and analyze how each combination affects classification performance.

### 1.4 Scope

- Dataset: AWID3 (Aegean Wi-Fi Intrusion Dataset 3).
- Offline performance analysis only — no real-time IDS deployment.
- Four classes used: **Normal**, **SSH Brute Force**, **Botnet**, **Malware** (the "Attacks Against the Local Nodes" category), totaling 7,979,387 samples.
- Feature selection methods: Mutual Information (MI), Random Forest Feature Importance (RFFI).
- Balancing methods: SMOTE, SMOTE-Tomek.
- Evaluation metrics: accuracy, precision, recall, F1-score (macro-averaged and per-class).
- Data split: 72% train / 8% validation / 20% test, stratified to preserve class proportions.

---

## 2. Background Concepts

**Intrusion Detection Systems (IDS).** Systems that monitor network traffic for malicious activity, either via known signatures or anomaly detection, dating back to foundational work by Denning (1987).

**AWID3.** A large-scale, realistic Wi-Fi intrusion dataset (Chatzoglou et al., 2021) containing statistical traffic features across normal and attack conditions, widely used as an IDS research benchmark.

**Feature selection.**
- *Mutual Information (MI)* — a filter-based method that scores each feature by its statistical dependency with the target label, independent of any specific model.
- *Random Forest Feature Importance (RFFI)* — an embedded method that scores features by their contribution to a trained Random Forest's splits (Breiman, 2001).

**Class balancing.**
- *SMOTE* (Synthetic Minority Over-sampling Technique) — generates synthetic minority-class samples via interpolation between existing minority samples (Chawla et al., 2002).
- *SMOTE-Tomek* — combines SMOTE with Tomek Link removal, which deletes overlapping sample pairs near the class boundary to sharpen class separation (Batista et al., 2004).

**TabNet.** An attention-based deep learning architecture for tabular data (Arik and Pfister, 2021) that performs sequential, instance-wise feature selection using a sparsemax-gated attentive transformer, combined with feature transformer blocks and batch normalization. Its loss function combines categorical cross-entropy with a sparsity-encouraging entropy regularization term on the learned feature masks.

---

## 3. Methodology

### 3.1 Data Preparation

Of AWID3's 254 raw features, the pipeline reduces the feature set through a five-step cleaning process:

| Step | Features remaining |
|---|---:|
| Initial | 254 |
| After removing irrelevant identifiers (packet number, raw timestamp) | 252 |
| After removing features with >10% missing values | 31 |
| After removing zero-variance (constant) features | 24 |
| After removing data-leakage features | 21 |

**Leakage detection:** three timestamp-derived features (`frame.time_epoch`, `frame.time_relative`, `radiotap.timestamp.ts`) were found to have **completely non-overlapping value ranges across classes** — because each attack type was captured in a distinct time window during data collection, these fields let a model "cheat" by learning the capture schedule rather than genuine traffic patterns. They were removed. A fourth field, `wlan.ra` (receiver MAC address), was treated similarly as a partial identifier leak in the code pipeline.

Remaining missing values were imputed (median for numeric features, mode for categorical), and features were standardized (`StandardScaler`, fit on the training split only). Because the full dataset (~7.98M rows) exceeded comfortable memory limits on a 16 GB machine, preprocessing used a **two-stage strategy**: each attack class was cleaned independently and cached to disk (Parquet) in stage one, then combined, transformed, split, and scaled in stage two — keeping peak memory bounded to one class at a time.

### 3.2 Experimental Design

Nine scenarios were defined by crossing feature selection method (none / MI / RFFI) with balancing method (none / SMOTE / SMOTE-Tomek):

| Scenario | Feature Selection | Balancing |
|---|---|---|
| 1 | — | — |
| 2 | MI | — |
| 3 | RFFI | — |
| 4 | — | SMOTE |
| 5 | — | SMOTE-Tomek |
| 6 | MI | SMOTE |
| 7 | MI | SMOTE-Tomek |
| 8 | RFFI | SMOTE |
| 9 | RFFI | SMOTE-Tomek |

When feature selection was applied, the top 2/3 of features by score were retained. Balancing targets were computed adaptively relative to the smallest minority class (SSH Brute Force), oversampling each minority class toward a fixed multiple of that count; for SMOTE-Tomek specifically, the majority (Normal) class was subsampled beforehand, since Tomek Link detection's O(n²) complexity made it computationally infeasible on the full dataset.

Every scenario used identical TabNet hyperparameters (attention/decision dimension of 8, 3 decision steps, sparsemax masking, Adam optimizer, step-decay learning rate schedule, early stopping on validation accuracy) and was evaluated against the **same held-out test set**, ensuring a fair comparison.

---

## 4. Results and Discussion

### 4.1 Class Distribution

| Class | Observations | Share |
|---|---:|---:|
| Normal | 7,779,003 | 97.49% |
| Malware | 131,611 | 1.65% |
| Botnet | 56,891 | 0.71% |
| SSH Brute Force | 11,882 | 0.15% |
| **Total** | **7,979,387** | 100% |

The majority-to-smallest-minority ratio is roughly **655:1** — an extreme imbalance that motivates testing balancing techniques in the first place.

### 4.2 Scenario Comparison

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

### 4.3 Discussion

**The unmodified baseline was hard to beat.** Scenario 1 (no external feature selection, no balancing) and Scenario 6 (MI + SMOTE) achieved the two highest macro F1-scores (0.9034 and 0.9016 respectively). This suggests TabNet's own built-in, instance-wise attention-based feature selection already does much of the work that external feature selection is meant to provide — external feature selection with MI *alone* (Scenario 2) actually hurt performance noticeably (macro F1 dropped to 0.6927), indicating that MI's global feature ranking can discard signal that TabNet's own instance-wise attention would otherwise use adaptively.

**Balancing trades precision for recall.** The three SMOTE-Tomek scenarios (5, 7, 9) all pushed minority-class recall above 0.93 — meaning the model caught the vast majority of attacks — but at a steep cost to precision (as low as 0.4518 in Scenario 7), meaning many normal-traffic samples were misclassified as attacks. Net F1-score for these scenarios ended up *below* the unmodified baseline, illustrating that aggressive rebalancing is not a free lunch: it shifts the operating point of the classifier rather than uniformly improving it.

**RFFI was more stable than MI.** Comparing Scenario 2 (MI only) against Scenario 3 (RFFI only), RFFI produced substantially better and more balanced results, suggesting that an embedded, model-driven importance measure generalized better here than a purely statistical, model-agnostic one.

**Practical implication.** Which scenario is "best" depends on the deployment goal. If overall classification reliability matters most, the unmodified baseline or MI+SMOTE are preferable. If missing an attack is far costlier than a false alarm (a common security posture), a SMOTE-Tomek scenario's much higher recall may be worth its precision cost — with the understanding that analysts will need to triage more false positives.

---

## 5. Conclusion and Future Work

### 5.1 Conclusion

This project built and evaluated TabNet-based classifiers for Wi-Fi cyberattack detection on AWID3 under nine combinations of feature selection and class balancing, all assessed against an identical test set for fair comparison. The unmodified TabNet baseline delivered the strongest overall performance (99.31% accuracy, 90.34% macro F1-score), showing that TabNet's built-in attention-based feature selection is already quite effective on this dataset without external help. RFFI produced more stable results than MI when used without balancing. Balancing techniques (SMOTE, SMOTE-Tomek) measurably improved recall on minority attack classes, particularly SSH Brute Force and Botnet, but often at the cost of precision — most sharply for SMOTE-Tomek — showing that the choice of feature selection and balancing strategy needs to be matched to whether an IDS deployment prioritizes detection sensitivity or overall classification stability.

### 5.2 Future Work

- Extend evaluation to the full set of attack classes available in AWID3, beyond the four studied here.
- Investigate real-time / streaming IDS deployment rather than offline batch evaluation.
- Repeat experiments on more powerful compute infrastructure to enable larger-scale hyperparameter search and less aggressive subsampling for the SMOTE-Tomek scenarios.

---

## References

- Arik, S. Ö. and Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(8):6679–6687.
- Batista, G. E. A. P. A., Prati, R. C., and Monard, M. C. (2004). A study of the behavior of several methods for balancing machine learning training data. *ACM SIGKDD Explorations Newsletter*, 6(1):20–29.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1):5–32.
- Chatzoglou, E., Kambourakis, G., and Kolias, C. (2021). Empirical Evaluation of Attacks Against IEEE 802.11 Enterprise Networks: The AWID3 Dataset. *IEEE Access*.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16:321–357.
- Denning, D. E. (1987). An Intrusion-Detection Model. *IEEE Transactions on Software Engineering*, SE-13(2):222–232.
