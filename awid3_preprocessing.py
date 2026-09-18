# =============================================================================
# awid3preprocessing.py
# AWID3 Dataset Preprocessing Pipeline
#
# Combines every preprocessing stage from three notebooks:
#   01a  Data Cleaning Per Class
#   01b  Combine, Transform, Split, and Scale
#   01c  Leakage Feature Removal
# =============================================================================


# =============================================================================
#  NOTEBOOK 01A: DATA CLEANING PER CLASS 
# =============================================================================

#  0. Imports and Machine Resource Check 

import os
import gc
import re
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

print(f"NumPy  : {np.__version__}")
print(f"Pandas : {pd.__version__}")
print()

ram_total_gb = psutil.virtual_memory().total / (1024**3)
ram_avail_gb = psutil.virtual_memory().available / (1024**3)
n_cpu = os.cpu_count()
print(f"CPU cores    : {n_cpu}")
print(f"Total RAM    : {ram_total_gb:.1f} GB")
print(f"Available RAM : {ram_avail_gb:.1f} GB")
print()

if ram_avail_gb < 6:
    print("  Available RAM < 6 GB. Recommended: close other applications first.")
    print("    If the run fails due to OOM, lower SUBSET_FRAC_NORMAL to 0.3 or 0.5.")
elif ram_avail_gb < 10:
    print("  Available RAM < 10 GB. The Botnet folder may be tight - watch closely.")
else:
    print("  Available RAM is sufficient for a full data run.")


def print_ram(label=""):
    """Helper to monitor RAM usage during preprocessing."""
    vm = psutil.virtual_memory()
    used_gb  = (vm.total - vm.available) / (1024**3)
    avail_gb = vm.available / (1024**3)
    pct = vm.percent
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}RAM: {used_gb:.1f} GB used ({pct:.0f}%), {avail_gb:.1f} GB available")


#  1. Configuration - SELECT Target Class 
#
# Change TARGET_CLASS to select which class to process: "SSH", "Botnet", "Malware".
# Run this script 3 times (once per class), then continue to Stage 2.

# =============================================================================
# CHANGE THIS VALUE ON EVERY RUN: "SSH", "Botnet", or "Malware"
# =============================================================================
TARGET_CLASS = "SSH"
# =============================================================================

# Location of the raw AWID3 dataset - CHANGE to match your setup
AWID3_ROOT = r"D:\project\AWID3_CSV"   # example for Windows
# AWID3_ROOT = "/home/username/project/AWID3_CSV"   # example for Linux/Mac

# Subsampling of the Normal class (0.0-1.0). Attack classes are always kept at 100%.
#   8 GB RAM  -> 0.3 (0.2 for Botnet)
#   16 GB RAM -> 1.0 for SSH & Malware; 0.5-0.7 for Botnet
#   32 GB RAM -> 1.0 for all
SUBSET_FRAC_NORMAL = 1.0

# Chunk size for streaming CSV read (rows per chunk)
CHUNK_SIZE = 50_000   # recommended 50k for 16 GB RAM; 25k for 8 GB RAM

# Data cleaning threshold
MISSING_THRESHOLD = 0.10   # drop features with missing > 10%

# Output directory
OUTPUT_DIR = Path("./preprocessed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PARQUET = OUTPUT_DIR / f"cleaned_{TARGET_CLASS}.parquet"
OUTPUT_META_01A = OUTPUT_DIR / f"cleaned_{TARGET_CLASS}_metadata.json"

valid_classes = {"SSH", "Botnet", "Malware"}
if TARGET_CLASS not in valid_classes:
    raise ValueError(
        f"TARGET_CLASS must be one of {valid_classes}, "
        f"but got: '{TARGET_CLASS}'"
    )

print(f"TARGET_CLASS       : {TARGET_CLASS}")
print(f"AWID3_ROOT         : {AWID3_ROOT}")
print(f"SUBSET_FRAC_NORMAL : {SUBSET_FRAC_NORMAL}")
print(f"CHUNK_SIZE         : {CHUNK_SIZE:,}")
print(f"OUTPUT_PARQUET     : {OUTPUT_PARQUET.absolute()}")


#  2. Automatic Detection of Class Folder Path 

AWID3_ROOT_PATH = Path(AWID3_ROOT)

if not AWID3_ROOT_PATH.exists():
    raise FileNotFoundError(
        f'AWID3_ROOT not found: {AWID3_ROOT_PATH}\n'
        f'Make sure the path is correct and the folder has been extracted from the ZIP.'
    )


def canonical_name(folder_name: str):
    """Normalize folder name: '7.SSH' -> 'SSH'."""
    cleaned = re.sub(r'^\d+[.\-_\s]*', '', folder_name).strip()
    for canon in ['SSH', 'Botnet', 'Malware']:
        if cleaned.lower() == canon.lower():
            return canon
    return None


def find_class_folder(target_class: str):
    """Find the folder for target_class in AWID3_ROOT, return Path or None."""
    candidates = []
    for path in AWID3_ROOT_PATH.rglob('*'):
        if not path.is_dir():
            continue
        if canonical_name(path.name) != target_class:
            continue
        csv_files = list(path.glob('*.csv'))
        if csv_files:
            candidates.append((path, len(csv_files)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


class_folder = find_class_folder(TARGET_CLASS)

if class_folder is None:
    raise FileNotFoundError(
        f'Folder for class {TARGET_CLASS} not found in {AWID3_ROOT_PATH}.\n'
        f'Make sure the AWID3 dataset has been extracted into that folder.\n'
        f'Expected structure: {AWID3_ROOT}/<7.>{TARGET_CLASS}/*.csv'
    )

n_csv = len(list(class_folder.glob('*.csv')))
total_size_mb = sum(f.stat().st_size for f in class_folder.glob('*.csv')) / (1024**2)
print(f'Folder {TARGET_CLASS} found at:')
print(f'  {class_folder}')
print(f'Number of .csv files: {n_csv}')
print(f'Total size     : {total_size_mb:,.1f} MB')


#  3. Streaming CSV Load 

def load_class_folder(folder_path: Path,
                      label_name: str,
                      subset_frac_normal: float = 1.0,
                      chunk_size: int = 100_000,
                      seed: int = 42) -> pd.DataFrame:
    """
    Stream-read every .csv file in an AWID3 class folder, filtering and
    subsampling Normal rows on the fly within each chunk.
    """
    files = sorted(folder_path.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No .csv files found in {folder_path}')

    rng = np.random.default_rng(seed)
    kept_chunks = []
    n_normal_seen = 0
    n_normal_kept = 0
    n_attack_kept = 0

    for f_idx, f in enumerate(files, 1):
        print(f'    [{f_idx}/{len(files)}] {f.name}', flush=True)
        reader = pd.read_csv(
            f,
            low_memory=False,
            na_values=['', ' ', '?', '-'],
            chunksize=chunk_size,
        )
        for chunk in reader:
            # Normalize label column
            if 'Label' not in chunk.columns:
                cand = [c for c in chunk.columns
                        if c.lower() in ('label', 'class', 'attack')]
                if not cand:
                    raise KeyError(f'Label column not found in {f.name}')
                chunk = chunk.rename(columns={cand[0]: 'Label'})

            chunk['Label'] = chunk['Label'].astype(str).str.strip()
            label_lower = chunk['Label'].str.lower()
            is_normal = label_lower.eq('normal')
            is_attack = label_lower.eq(label_name.lower())

            attack_rows = chunk.loc[is_attack].copy()
            if len(attack_rows) > 0:
                attack_rows['Label'] = label_name
                n_attack_kept += len(attack_rows)

            normal_rows_all = chunk.loc[is_normal]
            n_normal_seen += len(normal_rows_all)
            if subset_frac_normal >= 1.0:
                normal_rows = normal_rows_all.copy()
            elif len(normal_rows_all) > 0:
                keep_mask = rng.random(len(normal_rows_all)) < subset_frac_normal
                normal_rows = normal_rows_all.loc[keep_mask].copy()
            else:
                normal_rows = normal_rows_all.copy()

            if len(normal_rows) > 0:
                normal_rows['Label'] = 'Normal'
                n_normal_kept += len(normal_rows)

            if len(attack_rows) > 0 or len(normal_rows) > 0:
                kept_chunks.append(pd.concat(
                    [normal_rows, attack_rows], ignore_index=True
                ))

            del chunk, attack_rows, normal_rows, normal_rows_all

    print(f'    Total Normal rows read: {n_normal_seen:,}')
    print(f'    Normal rows kept      : {n_normal_kept:,}')
    print(f'    {label_name} rows kept      : {n_attack_kept:,}')

    if not kept_chunks:
        return pd.DataFrame()
    return pd.concat(kept_chunks, ignore_index=True)


print_ram("before-load")
print(f"\nStarting to load folder {TARGET_CLASS} ...")
t0 = time.time()

df_raw = load_class_folder(
    folder_path=class_folder,
    label_name=TARGET_CLASS,
    subset_frac_normal=SUBSET_FRAC_NORMAL,
    chunk_size=CHUNK_SIZE,
    seed=SEED + hash(TARGET_CLASS) % 1000,
)

print(f"\nLoad time: {time.time()-t0:.1f} s")
print(f"Shape df_raw: {df_raw.shape}")
print(f"\nClass distribution:")
print(df_raw['Label'].value_counts().to_string())
print()
print_ram("after-load")


#  4. Split X and y 

y_raw = df_raw["Label"].copy()
X_raw = df_raw.drop(columns=["Label"])

del df_raw
gc.collect()

print(f"Initial X_raw dimensions: {X_raw.shape}")
print(f"Number of features: {X_raw.shape[1]}")
print_ram("after-split-xy")


#  4b. Mixed-Type Feature Transformation 

BINARY_FLAG_FEATURES = [
    'radiotap.present.tsft',
    'radiotap.present.flags',
    'radiotap.present.rate',
    'radiotap.present.channel',
    'radiotap.present.dbm_antsignal',
    'radiotap.present.antenna',
    'radiotap.present.rxflags',
]

HEX_PATTERN      = re.compile(r'^0x[0-9a-fA-F]+$')
MAC_PATTERN      = re.compile(r'^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$')
NUMERIC_PATTERN  = re.compile(r'^-?\d+(\.\d+)?$')


def smart_parse(val):
    """Parse a single string value to float (cascading attempts)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan

    if NUMERIC_PATTERN.match(s):
        try:
            return float(s)
        except ValueError:
            pass

    if HEX_PATTERN.match(s):
        try:
            return float(int(s, 16))
        except ValueError:
            pass

    if MAC_PATTERN.match(s):
        try:
            clean_hex = s.replace(':', '')
            return float(int(clean_hex, 16))
        except ValueError:
            pass

    parsed_s = s[0] + s[1:].replace('-', ' -')
    parts = parsed_s.split()
    values = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            pass
    if values:
        return float(np.mean(values))

    return np.nan


def smart_parse_first(val):
    """Variant of smart_parse for binary flag columns: takes only the first value."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    parsed_s = s[0] + s[1:].replace('-', ' -')
    parts = parsed_s.split()
    if parts:
        try:
            return float(parts[0])
        except ValueError:
            pass
    return np.nan


def _is_str_col(series):
    return (
        series.dtype == 'object'
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


LEAKY_TO_SKIP = {'frame.number', 'frame.time'}

converted_cols = []
binary_flag_cols_converted = []
kept_categorical = []

for col in list(X_raw.columns):
    if not _is_str_col(X_raw[col]):
        continue
    if col in LEAKY_TO_SKIP:
        continue

    non_null = X_raw[col].dropna()
    if len(non_null) == 0:
        continue
    sample_values = non_null.head(3).tolist()

    if col in BINARY_FLAG_FEATURES:
        X_raw[col] = X_raw[col].apply(smart_parse_first)
        binary_flag_cols_converted.append((col, sample_values))
        continue

    sample_size = min(500, len(non_null))
    sample = non_null.sample(sample_size, random_state=42)
    n_parseable = sum(1 for v in sample if not pd.isna(smart_parse(v)))
    success_rate = n_parseable / sample_size

    if success_rate >= 0.5:
        X_raw[col] = X_raw[col].apply(smart_parse)
        converted_cols.append((col, sample_values, success_rate))
    else:
        kept_categorical.append((col, sample_values))

print(f'Columns converted to numeric: {len(converted_cols)}')
for col, samples, rate in converted_cols:
    print(f'   {col:<40} samples: {samples}  (rate {rate*100:.0f}%)')

print(f'\nBinary flag columns (first value): {len(binary_flag_cols_converted)}')
for col, samples in binary_flag_cols_converted:
    print(f'   {col:<40} samples: {samples}')

print(f'\nCategorical columns kept        : {len(kept_categorical)}')
for col, samples in kept_categorical:
    print(f'   {col:<40} samples: {samples}  (will be label-encoded in 01b)')

dtype_summary = X_raw.dtypes.value_counts()
print(f'\nData type summary after conversion:')
print(dtype_summary.to_string())
print_ram('after-mixed-type-handling')


#  5. Drop Leaky and Irrelevant Features 

LEAKY_FEATURES_01A = [
    "frame.number",
    "frame.time",
]

leaky_present = [f for f in LEAKY_FEATURES_01A if f in X_raw.columns]
leaky_missing  = [f for f in LEAKY_FEATURES_01A if f not in X_raw.columns]

print(f"Leaky features found and dropped: {leaky_present}")
if leaky_missing:
    print(f"Leaky features not present in the data: {leaky_missing}")

X_clean = X_raw.drop(columns=leaky_present)
print(f"\nX dimensions after dropping leaky features: {X_clean.shape}")


#  6. Drop Features with Missing Value > 10% 

missing_frac = X_clean.isna().mean().sort_values(ascending=False)
n_with_missing = (missing_frac > 0).sum()
print(f"Number of features with at least one missing value: {n_with_missing}")

if n_with_missing > 0:
    print("\nTop 10 features by missing rate:")
    for feat, frac in missing_frac.head(10).items():
        marker = " drop" if frac > MISSING_THRESHOLD else ""
        print(f"  {feat:<45} {frac*100:>6.2f}%  {marker}")

to_drop_missing = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()
X_clean = X_clean.drop(columns=to_drop_missing)

print(f"\nRemoving {len(to_drop_missing)} features with missing > {MISSING_THRESHOLD*100:.0f}%")
print(f"X dimensions after dropping missing-heavy features: {X_clean.shape}")


#  7. Drop Constant Features 

nunique = X_clean.nunique(dropna=False)
to_drop_const = nunique[nunique <= 1].index.tolist()

print(f"Removing {len(to_drop_const)} constant features")
if 0 < len(to_drop_const) <= 15:
    print("Features removed:")
    for feat in to_drop_const:
        print(f"  - {feat}")

X_clean = X_clean.drop(columns=to_drop_const)
print(f"\nX dimensions after dropping constant features: {X_clean.shape}")


#  8. Impute Remaining Missing Values 

def is_categorical(series):
    return (
        series.dtype == "object"
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


impute_log = {"numeric_median": 0, "categorical_mode": 0}

for col in X_clean.columns:
    if X_clean[col].isna().any():
        if is_categorical(X_clean[col]):
            mode = X_clean[col].mode()
            fill_value = mode.iloc[0] if len(mode) > 0 else "UNK"
            impute_log["categorical_mode"] += 1
        else:
            fill_value = X_clean[col].median()
            impute_log["numeric_median"] += 1
        X_clean[col] = X_clean[col].fillna(fill_value)

print(f"Imputation complete:")
print(f"  - numeric features (median)    : {impute_log['numeric_median']}")
print(f"  - categorical features (mode)  : {impute_log['categorical_mode']}")

assert X_clean.isna().sum().sum() == 0, "There are still missing values!"
print(f"\nVerification: total missing = {X_clean.isna().sum().sum()} ")
print_ram("after-impute")


#  8b. Downcast Data Types 

mem_before_df = X_clean.memory_usage(deep=True).sum() / (1024**2)

for col in X_clean.columns:
    dtype = X_clean[col].dtype
    if dtype == "float64":
        X_clean[col] = X_clean[col].astype("float32")
    elif dtype == "int64":
        col_min = X_clean[col].min()
        col_max = X_clean[col].max()
        if col_min >= np.iinfo("int32").min and col_max <= np.iinfo("int32").max:
            X_clean[col] = X_clean[col].astype("int32")

gc.collect()

mem_after_df = X_clean.memory_usage(deep=True).sum() / (1024**2)
print(f"DataFrame memory: {mem_before_df:.1f} MB  {mem_after_df:.1f} MB")
print(f"Reduction      : {(1 - mem_after_df/mem_before_df)*100:.1f}%")
print_ram("after-downcast")


#  9. Recombine X and y, then Save to Parquet 

df_cleaned = X_clean.copy()
df_cleaned["Label"] = y_raw.values

print(f"Final shape: {df_cleaned.shape}")
print(f"Class distribution:")
print(df_cleaned["Label"].value_counts().to_string())
print()

t0 = time.time()
df_cleaned.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
print(f"Save time: {time.time()-t0:.1f} s")

file_size_mb = OUTPUT_PARQUET.stat().st_size / (1024**2)
print(f"File saved      : {OUTPUT_PARQUET}")
print(f"File size       : {file_size_mb:.2f} MB")


#  10. Save Cleaning Metadata 

metadata_01a = {
    "target_class": TARGET_CLASS,
    "source_folder": str(class_folder),
    "subset_frac_normal": SUBSET_FRAC_NORMAL,
    "missing_threshold": MISSING_THRESHOLD,
    "seed": SEED,
    "n_rows_final": int(len(df_cleaned)),
    "n_features_final": int(X_clean.shape[1]),
    "features_dropped_leaky":    leaky_present,
    "features_dropped_missing":  to_drop_missing,
    "features_dropped_constant": to_drop_const,
    "features_kept": X_clean.columns.tolist(),
    "class_counts": {
        k: int(v) for k, v in df_cleaned["Label"].value_counts().items()
    },
    "imputed_numeric": impute_log["numeric_median"],
    "imputed_categorical": impute_log["categorical_mode"],
}

with open(OUTPUT_META_01A, "w") as f:
    json.dump(metadata_01a, f, indent=2)

print(f"Metadata saved to {OUTPUT_META_01A}")
print()
print("Metadata summary:")
print(json.dumps(metadata_01a, indent=2))


#  11. Sanity Check - Reload Verification 

df_reloaded = pd.read_parquet(OUTPUT_PARQUET)

assert df_reloaded.shape == df_cleaned.shape, \
    f"Shape mismatch: {df_reloaded.shape} vs {df_cleaned.shape}"
print(f"Shape match: {df_reloaded.shape} ")

original_counts = df_cleaned["Label"].value_counts().to_dict()
reloaded_counts = df_reloaded["Label"].value_counts().to_dict()
assert original_counts == reloaded_counts, "Class counts mismatch!"
print(f"Class counts match ")

for col in df_cleaned.columns[:5]:
    assert df_cleaned[col].iloc[:100].equals(df_reloaded[col].iloc[:100]), \
        f"Value mismatch in column {col}"
print(f"Values identical for the first 100 sample rows ")
print(f"\nReload verification PASSED")

print(f"\n{'='*60}")
print(f"Stage 1 complete for class: {TARGET_CLASS}")
print(f"Repeat with a different TARGET_CLASS for the other classes,")
print(f"then run Stage 2 (the 01b section below).")
print(f"{'='*60}\n")


# =============================================================================
#  NOTEBOOK 01B: COMBINE, TRANSFORM, SPLIT, AND SCALE 
# =============================================================================
#
# Prerequisite: all three files cleaned_SSH.parquet, cleaned_Botnet.parquet, and
# cleaned_Malware.parquet must already exist in OUTPUT_DIR before this section
# is run. In the 3-run pipeline, run this section after finishing
# Stage 1 for all three classes.

#  Stage 2 Configuration 

INPUT_DIR   = Path("./preprocessed")
OUTPUT_NPZ  = OUTPUT_DIR / "processed_data.npz"
OUTPUT_META_01B = OUTPUT_DIR / "preprocessing_metadata.json"

CLASS_MAPPING = {
    "Normal":  0,
    "SSH":     1,
    "Botnet":  2,
    "Malware": 3,
}
CLASS_NAMES = list(CLASS_MAPPING.keys())
N_CLASSES   = len(CLASS_NAMES)

TEST_SIZE = 0.20   # 80% train_val, 20% test
VAL_SIZE  = 0.10   # of train_val, 90% train, 10% val

print(f"INPUT_DIR    : {INPUT_DIR.absolute()}")
print(f"OUTPUT_DIR   : {OUTPUT_DIR.absolute()}")
print(f"CLASS_MAPPING: {CLASS_MAPPING}")


#  2. Find Parquet Files from Notebook 01A 

def find_cleaned_parquets():
    if not INPUT_DIR.exists():
        return {}
    found = {}
    for cls in ["SSH", "Botnet", "Malware"]:
        target = INPUT_DIR / f"cleaned_{cls}.parquet"
        if target.exists():
            found[cls] = target
    return found


parquet_paths = find_cleaned_parquets()
missing_cls = [c for c in ["SSH", "Botnet", "Malware"] if c not in parquet_paths]

if missing_cls:
    raise FileNotFoundError(
        f"The following parquet files were not found in {INPUT_DIR}: {missing_cls}\n"
        f"Found: {list(parquet_paths.keys())}\n\n"
        f"Make sure you have run Stage 1 for all three classes SSH,\n"
        f"Botnet, and Malware. Check that the .parquet files exist in the ./preprocessed/ folder"
    )

print("Parquet files found:")
for cls, path in parquet_paths.items():
    size_mb = path.stat().st_size / (1024**2)
    print(f"  {cls:<10}: {path.name}  ({size_mb:.1f} MB)")


#  3. Load All Three Parquet Files 

dfs = {}
for cls, path in parquet_paths.items():
    t0 = time.time()
    df = pd.read_parquet(path)
    dt = time.time() - t0
    dfs[cls] = df
    print(f"  {cls:<10}: shape={df.shape}, load time={dt:.1f}s")
    print(f"  {'':10}  distribution: {dict(df['Label'].value_counts())}")
    print()

print_ram("after-load-parquets")


#  4. Reconcile Column Schema 

col_sets = {cls: set(df.columns) for cls, df in dfs.items()}
common_cols = set.intersection(*col_sets.values())

print("Column schema per class:")
for cls, cols in col_sets.items():
    extra = cols - common_cols
    print(f"  {cls:<10}: {len(cols)} columns (unique: {len(extra)})")
    if extra and len(extra) <= 10:
        print(f"  {'':10}  unique columns: {sorted(extra)}")

print(f"\nNumber of columns across ALL classes (intersection): {len(common_cols)}")
assert "Label" in common_cols, "Label column missing from some files!"

common_cols_sorted = sorted(common_cols - {"Label"}) + ["Label"]

for cls in list(dfs.keys()):
    dfs[cls] = dfs[cls][common_cols_sorted]

print(f"\nAfter reconciliation, all dataframes share {len(common_cols_sorted)} identical columns.")


#  5. Combine the Three DataFrames 

print_ram("before-concat")
t0 = time.time()
df_combined = pd.concat(list(dfs.values()), ignore_index=True)
del dfs
gc.collect()
print(f"Concat time: {time.time()-t0:.1f}s")
print(f"Combined shape: {df_combined.shape}")

df_combined = df_combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print("Class distribution after combining:")
print(df_combined["Label"].value_counts().to_string())
print()

counts = df_combined["Label"].value_counts()
ratio = counts.max() / counts.min()
print(f"Imbalance ratio (max/min): {ratio:.1f}x")
print_ram("after-concat")


#  6. Visualize Class Distribution 

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts_ordered = [int(df_combined["Label"].value_counts().get(c, 0)) for c in CLASS_NAMES]
colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

axes[0].bar(CLASS_NAMES, counts_ordered, color=colors)
axes[0].set_title("Class Distribution (linear scale)")
axes[0].set_ylabel("Number of Samples")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(CLASS_NAMES, counts_ordered, color=colors)
axes[1].set_yscale("log")
axes[1].set_title("Class Distribution (log scale)")
axes[1].set_ylabel("Number of Samples (log)")
axes[1].grid(True, alpha=0.3, axis="y")

for ax in axes:
    for i, v in enumerate(counts_ordered):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Figure saved to {OUTPUT_DIR / 'class_distribution.png'}")


#  7. Split X and y, Label-Encode Target 

y_raw_combined = df_combined["Label"].copy()
X_raw_combined = df_combined.drop(columns=["Label"])
del df_combined

y_int = y_raw_combined.map(CLASS_MAPPING).astype(np.int64).values
assert not np.any(np.isnan(y_int)), "Some labels could not be mapped!"

print(f"Shape X_raw: {X_raw_combined.shape}")
print(f"Shape y_int: {y_int.shape}")
print("\nClass distribution (integer):")
for name, idx in CLASS_MAPPING.items():
    count = int((y_int == idx).sum())
    pct = count / len(y_int) * 100
    print(f"  {idx} ({name:<10}): {count:>12,}  ({pct:.2f}%)")


#  8. Label-Encode Categorical Features 

cat_cols = [c for c in X_raw_combined.columns if is_categorical(X_raw_combined[c])]
num_cols = [c for c in X_raw_combined.columns if c not in cat_cols]

print(f"Categorical features: {len(cat_cols)}")
print(f"Numeric features    : {len(num_cols)}")
if 0 < len(cat_cols) <= 20:
    print("\nCategorical features:")
    for col in cat_cols:
        n_unique = X_raw_combined[col].nunique()
        print(f"  - {col:<40} ({n_unique} unique values)")

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_raw_combined[col] = le.fit_transform(X_raw_combined[col].astype(str))
    label_encoders[col] = le

print(f"Label encoding complete for {len(cat_cols)} columns")
print(f"All features are now numeric: {X_raw_combined.dtypes.value_counts().to_dict()}")


#  9. Stratified Train / Val / Test Split 

feature_names = X_raw_combined.columns.tolist()
X_values = X_raw_combined.values
del X_raw_combined

X_trval, X_test, y_trval, y_test = train_test_split(
    X_values, y_int,
    test_size=TEST_SIZE,
    stratify=y_int,
    random_state=SEED,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval,
    test_size=VAL_SIZE,
    stratify=y_trval,
    random_state=SEED,
)

del X_values, y_int, X_trval, y_trval

print(f"Shape train : {X_train.shape}   ({len(y_train):,} samples)")
print(f"Shape val   : {X_val.shape}     ({len(y_val):,} samples)")
print(f"Shape test  : {X_test.shape}    ({len(y_test):,} samples)")

print("\nClass proportions per split (%):")
print(f"{'Class':<10}{'Train':>10}{'Val':>10}{'Test':>10}")
for name, idx in CLASS_MAPPING.items():
    p_tr = (y_train == idx).mean() * 100
    p_va = (y_val   == idx).mean() * 100
    p_te = (y_test  == idx).mean() * 100
    print(f"{name:<10}{p_tr:>9.2f}%{p_va:>9.2f}%{p_te:>9.2f}%")


#  10. Standard Scaling 

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)

del X_train, X_val, X_test

train_mean = X_train_scaled.mean(axis=0)
train_std  = X_train_scaled.std(axis=0)
print(f"After scaling (training data):")
print(f"  Feature mean (min, max): ({train_mean.min():.6f}, {train_mean.max():.6f})")
print(f"  Feature std  (min, max): ({train_std.min():.6f}, {train_std.max():.6f})")
print(f"  Expected: mean ~ 0, std ~ 1")


#  11. Save to .npz 

t0 = time.time()
np.savez_compressed(
    OUTPUT_NPZ,
    X_train=X_train_scaled,
    X_val=X_val_scaled,
    X_test=X_test_scaled,
    y_train=y_train.astype(np.int64),
    y_val=y_val.astype(np.int64),
    y_test=y_test.astype(np.int64),
    feature_names=np.array(feature_names, dtype=object),
    class_names=np.array(CLASS_NAMES, dtype=object),
)
dt = time.time() - t0

file_size_mb = OUTPUT_NPZ.stat().st_size / (1024**2)
print(f"Save time: {dt:.1f} s")
print(f"File saved     : {OUTPUT_NPZ}")
print(f"File size      : {file_size_mb:.2f} MB")


#  12. Save Preprocessing Metadata 

metadata_01b = {
    "seed": SEED,
    "class_mapping": CLASS_MAPPING,
    "shapes": {
        "X_train": list(X_train_scaled.shape),
        "X_val":   list(X_val_scaled.shape),
        "X_test":  list(X_test_scaled.shape),
    },
    "n_features_final": len(feature_names),
    "n_categorical_features": len(cat_cols),
    "n_numeric_features":     len(num_cols),
    "class_counts_train": {
        CLASS_NAMES[i]: int((y_train == i).sum()) for i in range(N_CLASSES)
    },
    "class_counts_val": {
        CLASS_NAMES[i]: int((y_val == i).sum()) for i in range(N_CLASSES)
    },
    "class_counts_test": {
        CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(N_CLASSES)
    },
    "scaling": "StandardScaler (z-score, fit only on train)",
    "split_ratios": {"train": 0.72, "val": 0.08, "test": 0.20},
    "parquet_sources": {k: str(v) for k, v in parquet_paths.items()},
}

with open(OUTPUT_META_01B, "w") as f:
    json.dump(metadata_01b, f, indent=2)

print(f"Metadata saved to {OUTPUT_META_01B}")
print("\nRingkasan:")
print(json.dumps(metadata_01b, indent=2))


#  13. Sanity Check - Reload Verification 

for name, arr in [("X_train", X_train_scaled),
                  ("X_val",   X_val_scaled),
                  ("X_test",  X_test_scaled)]:
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"  {name:<10}: NaN={n_nan}, inf={n_inf}, shape={arr.shape}, dtype={arr.dtype}")
    assert n_nan == 0 and n_inf == 0, f"NaN/inf present in {name}!"

data_chk = np.load(OUTPUT_NPZ, allow_pickle=True)
assert np.array_equal(data_chk["X_train"], X_train_scaled), "X_train mismatch!"
assert np.array_equal(data_chk["y_train"], y_train.astype(np.int64)), "y_train mismatch!"
assert np.array_equal(data_chk["X_test"],  X_test_scaled),  "X_test mismatch!"
assert np.array_equal(data_chk["y_test"],  y_test.astype(np.int64)), "y_test mismatch!"

print("\nReload verification PASSED ")
print(f"File ready for the training notebook: {OUTPUT_NPZ}")


# =============================================================================
#  NOTEBOOK 01C: LEAKAGE FEATURE REMOVAL 
# =============================================================================

#  1. Load Output Data from 01b 

DATA_DIR    = Path(r'C:\project\preprocessed')
INPUT_PATH  = DATA_DIR / 'processed_data.npz'
OUTPUT_PATH = DATA_DIR / 'processed_data_clean.npz'

data = np.load(INPUT_PATH, allow_pickle=True)
X_train_c = data['X_train']
X_val_c   = data['X_val']
X_test_c  = data['X_test']
y_train_c = data['y_train']
y_val_c   = data['y_val']
y_test_c  = data['y_test']
feature_names_c = list(data['feature_names'])
class_names_c   = list(data['class_names'])

print(f'Train shape : {X_train_c.shape}')
print(f'Val   shape : {X_val_c.shape}')
print(f'Test  shape : {X_test_c.shape}')
print(f'Classes     : {class_names_c}')
print(f'\n{len(feature_names_c)} input features:')
for i, f in enumerate(feature_names_c):
    print(f'  [{i:2d}] {f}')


#  2. Determine Features to Drop 

LEAKY_FEATURES_01C = [
    'frame.time_epoch',       # Absolute timestamp: SSH/Botnet/Malware do not overlap
    'frame.time_relative',    # Relative timestamp: per-class median differs greatly
    'radiotap.timestamp.ts',  # Hardware timestamp: SSH vs Botnet do not overlap
    'wlan.ra',                # Receiver MAC: partial identifier leakage
]

missing_feats = [f for f in LEAKY_FEATURES_01C if f not in feature_names_c]
if missing_feats:
    raise ValueError(f'Leaky features not found in feature_names: {missing_feats}')

leaky_idx     = [feature_names_c.index(f) for f in LEAKY_FEATURES_01C]
keep_idx      = [i for i in range(len(feature_names_c)) if i not in leaky_idx]
keep_features = [feature_names_c[i] for i in keep_idx]

print(f'TO BE DROPPED  ({len(LEAKY_FEATURES_01C)} features):')
for f in LEAKY_FEATURES_01C:
    print(f'   {f}')
print(f'\nTO BE KEPT ({len(keep_features)} features):')
for f in keep_features:
    print(f'   {f}')


#  3. Empirical Leakage Verification 

Xs, _, ys, _ = train_test_split(
    X_train_c, y_train_c,
    train_size=200_000, stratify=y_train_c, random_state=42,
)

rows = []
for f in LEAKY_FEATURES_01C:
    j = feature_names_c.index(f)
    for c, cname in enumerate(class_names_c):
        mask = (ys == c)
        if mask.sum() == 0:
            continue
        vals = Xs[mask, j]
        rows.append({
            'feature': f, 'class': cname,
            'min': float(vals.min()),
            'median': float(np.median(vals)),
            'max': float(vals.max()),
            'range': float(vals.max() - vals.min()),
            'n': int(mask.sum()),
        })

df_intervals = pd.DataFrame(rows)
print('Per-class interval for leaky features:')
print(df_intervals.to_string(index=False, float_format=lambda x: f'{x:9.4f}'))

print('\n\nClass pairs with zero interval overlap (perfect leakage):')
print('-' * 60)
for f in LEAKY_FEATURES_01C:
    df_f = df_intervals[df_intervals['feature'] == f].set_index('class')
    for c1 in class_names_c:
        for c2 in class_names_c:
            if c1 >= c2 or c1 not in df_f.index or c2 not in df_f.index:
                continue
            a = df_f.loc[c1]
            b = df_f.loc[c2]
            no_overlap = (a['max'] < b['min']) or (b['max'] < a['min'])
            if no_overlap:
                print(f'  {f:30s}  {c1:8s}  {c2:8s}  (gap = '
                      f'{max(a["min"], b["min"]) - min(a["max"], b["max"]):.4f})')

csv_path = DATA_DIR / 'leakage_evidence.csv'
df_intervals.to_csv(csv_path, index=False)
print(f'\nTable saved to: {csv_path}')


#  4. Drop Leaky Features from All Splits 

X_train_clean = X_train_c[:, keep_idx]
X_val_clean   = X_val_c[:,   keep_idx]
X_test_clean  = X_test_c[:,  keep_idx]

print(f'Before: train={X_train_c.shape}, val={X_val_c.shape}, test={X_test_c.shape}')
print(f'After:  train={X_train_clean.shape}, val={X_val_clean.shape}, test={X_test_clean.shape}')
print(f'Reduction: {X_train_c.shape[1]} -> {X_train_clean.shape[1]} features '
      f'({len(LEAKY_FEATURES_01C)} leaky dropped, {len(keep_features)} kept)')

for name, arr in [('X_train_clean', X_train_clean),
                  ('X_val_clean',   X_val_clean),
                  ('X_test_clean',  X_test_clean)]:
    assert arr.dtype in (np.float32, np.float64), f'{name} unexpected dtype: {arr.dtype}'
    assert not np.isnan(arr).any(), f'{name} contains NaN'
    assert not np.isinf(arr).any(), f'{name} contains inf'
print('\nDtype + NaN/inf validation: PASS')


#  5. Save to processed_data_clean.npz 

np.savez_compressed(
    OUTPUT_PATH,
    X_train=X_train_clean,
    X_val=X_val_clean,
    X_test=X_test_clean,
    y_train=y_train_c,
    y_val=y_val_c,
    y_test=y_test_c,
    feature_names=np.array(keep_features),
    class_names=np.array(class_names_c),
)

size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
print(f'Saved to: {OUTPUT_PATH}')
print(f'File size: {size_mb:.1f} MB')


#  6. Sanity Check - Reload and Verify 

chk = np.load(OUTPUT_PATH, allow_pickle=True)

assert chk['X_train'].shape == X_train_clean.shape, 'X_train shape mismatch'
assert chk['X_val'].shape   == X_val_clean.shape,   'X_val shape mismatch'
assert chk['X_test'].shape  == X_test_clean.shape,  'X_test shape mismatch'

assert np.array_equal(chk['X_train'], X_train_clean), 'X_train contents changed during save/load'
assert np.array_equal(chk['y_train'], y_train_c),     'y_train labels changed'

loaded_features = [str(f) for f in chk['feature_names']]
loaded_classes  = [str(c) for c in chk['class_names']]
assert loaded_features == keep_features, 'feature_names mismatch'
assert loaded_classes  == class_names_c, 'class_names mismatch'

for leaky in LEAKY_FEATURES_01C:
    assert leaky not in loaded_features, f'Leaky feature {leaky} still present!'

print('=' * 60)
print('SANITY CHECK: PASS ')
print('=' * 60)
print(f'Train shape : {chk["X_train"].shape}')
print(f'Val   shape : {chk["X_val"].shape}')
print(f'Test  shape : {chk["X_test"].shape}')
print(f'Features ({len(loaded_features)}): {loaded_features}')
print(f'Classes : {loaded_classes}')
print(f'\nFile ready to use: {OUTPUT_PATH}')
print(f'\nNext steps:')
print(f'  1. Update DATA_PATH in the 9 scenario notebooks to processed_data_clean.npz')
print(f'  2. Re-run scenarios on Kaggle')
print(f'  3. Compare new results vs old results (which used processed_data.npz)')
