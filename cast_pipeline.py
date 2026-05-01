from pathlib import Path
import pandas as pd

base_dir = Path(r"physionet.org\files\crisdb\1.0.0")


records_path = base_dir / "RECORDS"
with open(records_path, "r") as f:
    records = [line.strip() for line in f if line.strip()]

def clean_record(rec):
    name = rec.split("/")[-1]
    group = name[0]
    subject_id = name[1:4]
    phase = name[-1]
    return group, subject_id, phase, name

rows = []
for rec in records:
    group, sid, phase, name = clean_record(rec)
    rows.append([rec, name, group, sid, phase])

df = pd.DataFrame(
    rows,
    columns=["record_path", "record_name", "group_code", "subject_id", "phase_code"]
)

group_map = {"e": "Encainide", "f": "Flecainide", "m": "Moricizine"}
phase_map = {"a": "Baseline", "b": "On-therapy"}

df["drug"] = df["group_code"].map(group_map)
df["phase"] = df["phase_code"].map(phase_map)

pairs = (
    df.pivot_table(
        index=["group_code", "subject_id", "drug"],
        columns="phase_code",
        values="record_name",
        aggfunc="first"
    ).reset_index()
)
pairs.columns.name = None
pairs = pairs.rename(columns={"a": "baseline_record", "b": "therapy_record"})

print(df.head(10))
print(pairs.head(10))

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "records_parsed.csv", index=False)
pairs.to_csv(out_dir / "subject_pairs.csv", index=False)

print("\nPreprocessing complete.")
print(f"Total records: {len(df)}")
print(f"Subject pairs: {len(pairs)}")