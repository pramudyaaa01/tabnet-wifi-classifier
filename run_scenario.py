# pytorch-tabnet and imbalanced-learn are usually not preinstalled on Kaggle
import subprocess, sys

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

for pkg in ["pytorch-tabnet", "imbalanced-learn"]:
    try:
        pip_install(pkg)
        print(f"  {pkg} installed")
    except Exception as e:
        print(f"  Failed to install {pkg}: {e}")
print("Installation complete.")


import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from pytorch_tabnet.tab_model import TabNetClassifier

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"NumPy        : {np.__version__}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA tersedia: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")


# =============================================================================
# CHANGE THIS VALUE: 1, 2, 3, 4, 5, 6, 7, 8, or 9
# =============================================================================
SCENARIO = 1
# =============================================================================

# Definition of the 9 scenarios: (feature_selection, balancing)
SCENARIO_TABLE = {
    1: (None,   None),
    2: ("MI",   None),
    3: ("RFFI", None),
    4: (None,   "SMOTE"),
    5: (None,   "SMOTE-Tomek"),
    6: ("MI",   "SMOTE"),
    7: ("MI",   "SMOTE-Tomek"),
    8: ("RFFI", "SMOTE"),
    9: ("RFFI", "SMOTE-Tomek"),
}

if SCENARIO not in SCENARIO_TABLE:
    raise ValueError(f"SCENARIO must be 1-9, got: {SCENARIO}")

FS_METHOD, BALANCE_METHOD = SCENARIO_TABLE[SCENARIO]

# --- Feature selection parameters ---
TOP_K_FRACTION = 2/3      # take the top 2/3 of features

# --- Balancing parameters (auto, based on the smallest minority class ratio) ---
# For SMOTE: the Normal class is left untouched. SMOTE raises each minority class
# to MINORITY_MULTIPLIER times the count of the smallest minority class (SSH Brute Force).
# Data-driven strategy: the target automatically adapts to dataset size.
MINORITY_MULTIPLIER = 11    # each minority class is SMOTEd to 11x the smallest class count
SMOTE_K             = 5

# --- Special strategy for SMOTE-Tomek (scenarios 5, 7, 9) ---
# SMOTE-Tomek's Tomek detection has O(n^2) complexity, which becomes
# infeasible on the full AWID3 dataset (~5.6 million Normal samples -> an estimated
# 8-12 hours of computation). For scenarios using SMOTE-Tomek, the
# Normal class is randomly subsampled to NORMAL_SUBSAMPLE_FOR_SMOTE_TOMEK
# before SMOTE-Tomek is applied. SMOTE-only scenarios (4, 6, 8) are NOT
# affected by this subsampling.

NORMAL_SUBSAMPLE_FOR_SMOTE_TOMEK = 500_000

# --- Output ---
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"SCENARIO         : {SCENARIO}")
print(f"Feature Selection: {FS_METHOD}")
print(f"Balancing        : {BALANCE_METHOD}")


def find_npz():
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None
    candidates = list(kaggle_input.rglob("processed_data.npz"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


npz_path = find_npz()
if npz_path is None:
    raise FileNotFoundError(
        "processed_data.npz not found in /kaggle/input/.\n"
        "Attach the output of notebook 01b via + Add Input."
    )

print(f"Loading: {npz_path}")
data = np.load(npz_path, allow_pickle=True)

X_train = data["X_train"]
X_val   = data["X_val"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_val   = data["y_val"]
y_test  = data["y_test"]
feature_names = list(data["feature_names"])
class_names   = list(data["class_names"])

print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"  X_val  : {X_val.shape}     y_val  : {y_val.shape}")
print(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")
print(f"  Features: {len(feature_names)}")
print(f"  Classes : {class_names}")


def compute_mi(X, y, seed=SEED):
    """Mutual Information score per feature."""
    scores = mutual_info_classif(X, y, random_state=seed)
    return scores


def compute_rffi(X, y, seed=SEED):
    """Random Forest Feature Importance (MDI) per feature."""
    rf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=seed,
    )
    rf.fit(X, y)
    return rf.feature_importances_


if FS_METHOD is None:
    print("Scenario without feature selection - all features are used.")
    X_train_fs = X_train
    X_val_fs   = X_val
    X_test_fs  = X_test
    selected_names = feature_names
else:
    D = X_train.shape[1]
    K = max(1, int(np.ceil(TOP_K_FRACTION * D)))
    print(f"Feature selection: {FS_METHOD} - selecting {K} out of {D} features")

    t0 = time.time()
    if FS_METHOD == "MI":
        scores = compute_mi(X_train, y_train)
    elif FS_METHOD == "RFFI":
        scores = compute_rffi(X_train, y_train)
    else:
        raise ValueError(FS_METHOD)
    print(f"  Score computation time: {time.time()-t0:.1f} s")

    # Select the K highest-scoring features
    top_idx = np.argsort(scores)[::-1][:K]
    top_idx = np.sort(top_idx)
    selected_names = [feature_names[i] for i in top_idx]

    X_train_fs = X_train[:, top_idx]
    X_val_fs   = X_val[:, top_idx]
    X_test_fs  = X_test[:, top_idx]

    # Save the score ranking for the report appendix
    rank_df = pd.DataFrame({
        "feature": feature_names,
        "score":   scores,
    }).sort_values("score", ascending=False)
    rank_df.to_csv(OUTPUT_DIR / f"fs_scores_scenario_{SCENARIO}.csv", index=False)

    print(f"  Selected features: {selected_names}")
    print(f"  X_train_fs: {X_train_fs.shape}")


import matplotlib.pyplot as plt
import pandas as pd

if FS_METHOD is None:
    print("Scenario without feature selection - visualization skipped.")
else:
    # rank_df was already computed in the previous cell
    rank_df_sorted = rank_df.sort_values("score", ascending=True)  # ascending for barh

    # Mark selected features (top K)
    selected_set = set(selected_names)
    rank_df_sorted["selected"] = rank_df_sorted["feature"].isin(selected_set)

    # ============ Feature importance score bar chart ============
    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(rank_df_sorted))))
    colors = ["#1A4D7A" if sel else "#C8C8C8" for sel in rank_df_sorted["selected"]]
    ax.barh(rank_df_sorted["feature"], rank_df_sorted["score"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"{FS_METHOD} Score", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    method_full = {"MI": "Mutual Information", "RFFI": "Random Forest Feature Importance"}.get(FS_METHOD, FS_METHOD)
    ax.set_title(
        f"Scenario {SCENARIO} - {method_full} Score\n"
        f"({len(selected_names)} of {len(rank_df_sorted)} features selected, highlighted in blue)",
        fontsize=12, pad=10,
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1A4D7A", edgecolor="black", label=f"Selected (top {len(selected_names)})"),
        Patch(facecolor="#C8C8C8", edgecolor="black", label="Not selected"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fs_scores_scenario_{SCENARIO}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved: fs_scores_scenario_{SCENARIO}.png")

    # ============ Selected features summary table ============
    top_table = (
        rank_df_sorted[rank_df_sorted["selected"]]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    top_table.index = top_table.index + 1
    top_table.index.name = "Rank"
    top_table = top_table[["feature", "score"]].rename(
        columns={"feature": "Feature", "score": f"{FS_METHOD} Score"}
    )

    print(f"\nSelected features table (Scenario {SCENARIO} - {method_full}):")
    print(top_table.to_string())


from collections import Counter


if BALANCE_METHOD is None:
    print("Scenario without balancing - original training distribution preserved.")
    X_train_bal = X_train_fs
    y_train_bal = y_train
else:
    print(f"Balancing: {BALANCE_METHOD}")
    print("Initial training data distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name:<10}: {int((y_train == i).sum()):>10,}")

    # --- SMOTE-Tomek SPECIFIC: subsample Normal first ---
    # Tomek detection has O(n^2) complexity, making it infeasible
    # on the full AWID3 dataset. Normal subsampling is only applied for
    # scenarios involving SMOTE-Tomek (5, 7, 9).
    X_pre_smote = X_train_fs
    y_pre_smote = y_train

    if BALANCE_METHOD == "SMOTE-Tomek" and NORMAL_SUBSAMPLE_FOR_SMOTE_TOMEK is not None:
        n_normal_current = int((y_train == 0).sum())
        target_normal = NORMAL_SUBSAMPLE_FOR_SMOTE_TOMEK
        if n_normal_current > target_normal:
            print(f"\n[SMOTE-Tomek-specific subsampling]")
            print(f"  Original Normal class: {n_normal_current:,} samples")
            print(f"  Subsample target : {target_normal:,} samples")
            rng_ss = np.random.default_rng(SEED)
            normal_pos = np.where(y_train == 0)[0]
            keep_normal = rng_ss.choice(normal_pos, size=target_normal, replace=False)
            other_pos = np.where(y_train != 0)[0]
            keep_idx = np.concatenate([keep_normal, other_pos])
            rng_ss.shuffle(keep_idx)
            X_pre_smote = X_train_fs[keep_idx]
            y_pre_smote = y_train[keep_idx]
            print(f"  Distribution after subsampling:")
            for i, name in enumerate(class_names):
                print(f"    {name:<10}: {int((y_pre_smote == i).sum()):>10,}")
        else:
            print(f"  Normal class ({n_normal_current:,}) already <= subsample target, "
                  f"not subsampled.")

    # Automatically compute the SMOTE target: MINORITY_MULTIPLIER x the count of the
    # smallest minority class. This approach is data-driven and keeps the
    # synthetic:original ratio proportional to the original minority class size.
    minority_class_counts = {
        c: int((y_pre_smote == c).sum()) for c in [1, 2, 3]
    }
    smallest_minority_count = min(minority_class_counts.values())
    minority_target = MINORITY_MULTIPLIER * smallest_minority_count
    print(f"\nSmallest minority class: {smallest_minority_count:,} samples")
    print(f"SMOTE target: {MINORITY_MULTIPLIER}x = {minority_target:,} per minority class")

    # Sampling strategy: raise any class with count < target up to the target.
    # Classes already above the target are left as-is (SMOTE does not
    # undersample classes that are already large enough).
    strategy = {}
    for c in [1, 2, 3]:
        current = minority_class_counts[c]
        if current < minority_target:
            strategy[c] = minority_target
        else:
            print(f"  Note: class {class_names[c]} ({current:,}) is already > target, "
                  f"left as-is")

    if not strategy:
        print("All minority classes already exceed the target. No SMOTE applied.")
        X_train_bal = X_pre_smote
        y_train_bal = y_pre_smote
    else:
        smote = SMOTE(sampling_strategy=strategy, k_neighbors=SMOTE_K, random_state=SEED)

        t0 = time.time()
        if BALANCE_METHOD == "SMOTE":
            X_train_bal, y_train_bal = smote.fit_resample(X_pre_smote, y_pre_smote)
        elif BALANCE_METHOD == "SMOTE-Tomek":
            smt = SMOTETomek(smote=smote, random_state=SEED)
            X_train_bal, y_train_bal = smt.fit_resample(X_pre_smote, y_pre_smote)
        else:
            raise ValueError(BALANCE_METHOD)
        print(f"\nBalancing time: {time.time()-t0:.1f} s")

        X_train_bal = X_train_bal.astype(np.float32)
        y_train_bal = y_train_bal.astype(np.int64)

    print(f"\nDistribution after {BALANCE_METHOD}:")
    for i, name in enumerate(class_names):
        print(f"  {name:<10}: {int((y_train_bal == i).sum()):>10,}")
    print(f"  Total: {len(y_train_bal):,}")
    n_normal_final = int((y_train_bal == 0).sum())
    n_smallest_minority_final = min(int((y_train_bal == c).sum()) for c in [1, 2, 3])
    print(f"  Imbalance ratio (Normal vs smallest minority): "
          f"{n_normal_final / max(n_smallest_minority_final, 1):.1f}x")

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Compute distribution before and after balancing
counts_before = Counter(y_train.tolist())
counts_after  = Counter(y_train_bal.tolist())

before_vals = [counts_before.get(i, 0) for i in range(len(class_names))]
after_vals  = [counts_after.get(i, 0)  for i in range(len(class_names))]

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Consistent colors per class
class_colors = ["#1A4D7A", "#C0392B", "#E67E22", "#27AE60"]

# ============ Left: distribution before balancing (log scale) ============
ax = axes[0]
bars = ax.bar(class_names, before_vals, color=class_colors, edgecolor="black", linewidth=0.7)
ax.set_yscale("log")
ax.set_ylabel("Number of Samples (log scale)", fontsize=11)
ax.set_title("Before Balancing", fontsize=12, pad=8)
ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")
ax.tick_params(axis="x", labelsize=10)
# Annotate values
for bar, val in zip(bars, before_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f"{val:,}", ha="center", va="bottom", fontsize=9)

# ============ Right: distribution after balancing (log scale) ============
ax = axes[1]
bars = ax.bar(class_names, after_vals, color=class_colors, edgecolor="black", linewidth=0.7)
ax.set_yscale("log")
ax.set_ylabel("Number of Samples (log scale)", fontsize=11)
title_right = "Before Balancing (unchanged)" if BALANCE_METHOD is None else f"After {BALANCE_METHOD}"
ax.set_title(title_right, fontsize=12, pad=8)
ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")
ax.tick_params(axis="x", labelsize=10)
for bar, val in zip(bars, after_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f"{val:,}", ha="center", va="bottom", fontsize=9)

# Synchronize y-scale between both subplots
ymax = max(max(before_vals), max(after_vals)) * 2.5
ymin = max(1, min(min(before_vals), min(after_vals)) * 0.5)
for ax in axes:
    ax.set_ylim(ymin, ymax)

# Suptitle
balance_desc = "No Balancing" if BALANCE_METHOD is None else BALANCE_METHOD
plt.suptitle(
    f"Scenario {SCENARIO} - Training Data Class Distribution ({balance_desc})",
    fontsize=13, y=1.02,
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"class_distribution_scenario_{SCENARIO}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Figure saved: class_distribution_scenario_{SCENARIO}.png")

# ============ Summary table ============
print(f"\nTraining data class distribution table (Scenario {SCENARIO}):")
print(f"{'Class':<12}{'Before':>14}{'After':>14}{'Change':>14}")
print("-" * 54)
for i, cls in enumerate(class_names):
    b, a = before_vals[i], after_vals[i]
    delta = a - b
    sign = "+" if delta > 0 else ""
    print(f"{cls:<12}{b:>14,}{a:>14,}{sign+format(delta, ','):>14}")
print("-" * 54)
print(f"{'Total':<12}{sum(before_vals):>14,}{sum(after_vals):>14,}"
      f"{'+' if sum(after_vals)>sum(before_vals) else '':>1}"
      f"{format(sum(after_vals)-sum(before_vals), ','):>13}")


device = "cuda" if torch.cuda.is_available() else "cpu"

tabnet_params = dict(
    n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
    n_independent=2, n_shared=2, momentum=0.02, mask_type="sparsemax",
    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params=dict(step_size=10, gamma=0.9),
    seed=SEED, verbose=10, device_name=device,
)

clf = TabNetClassifier(**tabnet_params)

print(f"Training TabNet on {device} ...")
t0 = time.time()
clf.fit(
    X_train=X_train_bal, y_train=y_train_bal,
    eval_set=[(X_val_fs, y_val)],
    eval_name=["val"],
    eval_metric=["accuracy", "balanced_accuracy"],
    max_epochs=50,
    patience=15,
    batch_size=8192,
    virtual_batch_size=256,
    num_workers=0,
    drop_last=False,
)
train_time = time.time() - t0
print(f"\nTraining complete in {train_time/60:.2f} minutes")


print("Predicting on test data ...")
t0 = time.time()
y_pred = clf.predict(X_test_fs)
pred_time = time.time() - t0

acc        = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec_macro  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro   = f1_score(y_test, y_pred, average="macro", zero_division=0)

prec_per = precision_score(y_test, y_pred, average=None, zero_division=0)
rec_per  = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_per   = f1_score(y_test, y_pred, average=None, zero_division=0)

print(f"\n=== SCENARIO {SCENARIO} RESULTS ===")
print(f"  Accuracy        : {acc:.4f}")
print(f"  Precision (macro): {prec_macro:.4f}")
print(f"  Recall (macro)   : {rec_macro:.4f}")
print(f"  F1-score (macro) : {f1_macro:.4f}")

print(f"\n  Per-class:")
print(f"  {'Class':<12}{'Precision':>12}{'Recall':>12}{'F1':>12}")
for i, name in enumerate(class_names):
    print(f"  {name:<12}{prec_per[i]:>12.4f}{rec_per[i]:>12.4f}{f1_per[i]:>12.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix (rows=actual, columns=predicted):")
print("  " + " " * 12 + "".join(f"{n:>12}" for n in class_names))
for i, row in enumerate(cm):
    print(f"  {class_names[i]:<12}" + "".join(f"{v:>12,}" for v in row))

print(f"\n  Classification report:")
print(classification_report(y_test, y_pred, target_names=class_names,
                            digits=4, zero_division=0))


results = {
    "scenario": SCENARIO,
    "fs_method": FS_METHOD,
    "balance_method": BALANCE_METHOD,
    "accuracy": float(acc),
    "precision_macro": float(prec_macro),
    "recall_macro": float(rec_macro),
    "f1_macro": float(f1_macro),
    "per_class": {
        name: {
            "precision": float(prec_per[i]),
            "recall": float(rec_per[i]),
            "f1": float(f1_per[i]),
        }
        for i, name in enumerate(class_names)
    },
    "confusion_matrix": cm.tolist(),
    "selected_features": list(selected_names),
    "n_features_used": len(selected_names),
    "n_train_samples": int(len(y_train_bal)),
    "training_time_minutes": float(train_time / 60),
    "prediction_time_seconds": float(pred_time),
    "device": device,
}

out_path = OUTPUT_DIR / f"results_scenario_{SCENARIO}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {out_path}")

# Save model
clf.save_model(str(OUTPUT_DIR / f"tabnet_scenario_{SCENARIO}_model"))
print(f"Model saved to {OUTPUT_DIR}/tabnet_scenario_{SCENARIO}_model.zip")

