from pathlib import Path
import pandas as pd
import numpy as np
import wfdb

base_dir = Path(r"physionet.org\files\crisdb\1.0.0")

group_map = {"e": "Encainide", "f": "Flecainide", "m": "Moricizine"}

def load_records(base_dir):
    with open(base_dir / "RECORDS", "r") as f:
        return [line.strip() for line in f if line.strip()]

def parse_record_path(rec):
    parts = rec.split("/")
    group = parts[0]
    name = parts[-1]
    subject_id = name[1:4]
    phase_code = name[-1]
    return {
        "record_path": rec,
        "record_name": name,
        "group_code": group,
        "subject_id": subject_id,
        "phase_code": phase_code,
        "drug": group_map.get(group, None),
    }

def rr_features_from_record(record_path):
    group = record_path.split("/")[0]
    name = record_path.split("/")[-1]
    rec_base = str(base_dir / group / name)

    try:
        ann = wfdb.rdann(rec_base, "atr")
        header = wfdb.rdheader(rec_base)
        sample_rate = header.fs
    except Exception:
        return {
            "n_beats": np.nan,
            "mean_rr_s": np.nan,
            "std_rr_s": np.nan,
            "rmssd_s": np.nan,
            "median_rr_s": np.nan,
            "rr_status": "bad_annotation",
        }

    samples = np.array(ann.sample, dtype=float)
    if len(samples) < 2:
        return {
            "n_beats": len(samples),
            "mean_rr_s": np.nan,
            "std_rr_s": np.nan,
            "rmssd_s": np.nan,
            "median_rr_s": np.nan,
            "rr_status": "too_few_beats",
        }

    rr = np.diff(samples) / sample_rate
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0]

    if len(rr) == 0:
        return {
            "n_beats": len(samples),
            "mean_rr_s": np.nan,
            "std_rr_s": np.nan,
            "rmssd_s": np.nan,
            "median_rr_s": np.nan,
            "rr_status": "no_valid_rr",
        }

    diff_rr = np.diff(rr) if len(rr) > 1 else np.array([np.nan])
    rmssd = np.sqrt(np.nanmean(diff_rr**2)) if len(diff_rr) > 0 else np.nan

    return {
        "n_beats": len(samples),
        "mean_rr_s": float(np.mean(rr)),
        "std_rr_s": float(np.std(rr, ddof=1)) if len(rr) > 1 else np.nan,
        "rmssd_s": float(rmssd),
        "median_rr_s": float(np.median(rr)),
        "rr_status": "ok",
    }

records = load_records(base_dir)
rec_df = pd.DataFrame([parse_record_path(r) for r in records])

pairs = (
    rec_df.pivot_table(
        index=["group_code", "subject_id", "drug"],
        columns="phase_code",
        values="record_name",
        aggfunc="first"
    ).reset_index()
)
pairs.columns.name = None
pairs = pairs.rename(columns={"a": "baseline_record", "b": "therapy_record"})

rows = []
for _, row in pairs.iterrows():
    baseline = row["baseline_record"]
    therapy = row["therapy_record"]

    if pd.notna(baseline):
        bfeat = rr_features_from_record(f"{row['group_code']}/{baseline}")
    else:
        bfeat = {"n_beats": np.nan, "mean_rr_s": np.nan, "std_rr_s": np.nan, "rmssd_s": np.nan, "median_rr_s": np.nan, "rr_status": "missing_record"}

    if pd.notna(therapy):
        tfeat = rr_features_from_record(f"{row['group_code']}/{therapy}")
    else:
        tfeat = {"n_beats": np.nan, "mean_rr_s": np.nan, "std_rr_s": np.nan, "rmssd_s": np.nan, "median_rr_s": np.nan, "rr_status": "missing_record"}

    rows.append({
        "group_code": row["group_code"],
        "subject_id": row["subject_id"],
        "drug": row["drug"],
        "baseline_record": baseline,
        "therapy_record": therapy,
        "baseline_n_beats": bfeat["n_beats"],
        "baseline_mean_rr_s": bfeat["mean_rr_s"],
        "baseline_std_rr_s": bfeat["std_rr_s"],
        "baseline_rmssd_s": bfeat["rmssd_s"],
        "baseline_median_rr_s": bfeat["median_rr_s"],
        "baseline_rr_status": bfeat["rr_status"],
        "therapy_n_beats": tfeat["n_beats"],
        "therapy_mean_rr_s": tfeat["mean_rr_s"],
        "therapy_std_rr_s": tfeat["std_rr_s"],
        "therapy_rmssd_s": tfeat["rmssd_s"],
        "therapy_median_rr_s": tfeat["median_rr_s"],
        "therapy_rr_status": tfeat["rr_status"],
    })

feat_df = pd.DataFrame(rows)

for col in ["mean_rr_s", "std_rr_s", "rmssd_s", "median_rr_s"]:
    feat_df[f"delta_{col}"] = feat_df[f"therapy_{col}"] - feat_df[f"baseline_{col}"]

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)
feat_df.to_csv(out_dir / "cast_rr_features.csv", index=False)

print(feat_df.head())
print("Saved output/cast_rr_features.csv")