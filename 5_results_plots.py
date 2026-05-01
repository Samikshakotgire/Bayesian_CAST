from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

out_dir = Path("output")
summary_path = out_dir / "bayesian_summary.csv"
features_path = out_dir / "cast_rr_features.csv"

summary_df = pd.read_csv(summary_path)
feat_df = pd.read_csv(features_path)

# 1) Group counts
group_counts = feat_df["drug"].value_counts().reset_index()
group_counts.columns = ["drug", "count"]
group_counts.to_csv(out_dir / "group_counts.csv", index=False)

plt.figure(figsize=(7, 4))
plt.bar(group_counts["drug"], group_counts["count"], color=["#4C78A8", "#F58518", "#54A24B"])
plt.title("Subjects per Drug Group")
plt.xlabel("Drug")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(out_dir / "group_counts.png", dpi=200)
plt.close()

# 2) Mean change in RR by drug
rr_change = (
    feat_df.groupby("drug")[["delta_mean_rr_s", "delta_std_rr_s", "delta_rmssd_s", "delta_median_rr_s"]]
    .mean()
    .reset_index()
)
rr_change.to_csv(out_dir / "rr_change_by_drug.csv", index=False)

plt.figure(figsize=(7, 4))
plt.bar(rr_change["drug"], rr_change["delta_mean_rr_s"], color=["#4C78A8", "#F58518", "#54A24B"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Mean Change in RR Interval by Drug")
plt.xlabel("Drug")
plt.ylabel("Δ Mean RR (s)")
plt.tight_layout()
plt.savefig(out_dir / "delta_mean_rr_by_drug.png", dpi=200)
plt.close()

# 3) Baseline vs therapy mean RR
pair_long = pd.concat([
    feat_df[["drug", "baseline_mean_rr_s"]].rename(columns={"baseline_mean_rr_s": "mean_rr_s"}).assign(phase="Baseline"),
    feat_df[["drug", "therapy_mean_rr_s"]].rename(columns={"therapy_mean_rr_s": "mean_rr_s"}).assign(phase="On-therapy")
])

plt.figure(figsize=(8, 4))
for phase, color in [("Baseline", "#4C78A8"), ("On-therapy", "#E45756")]:
    vals = pair_long.loc[pair_long["phase"] == phase, "mean_rr_s"].dropna()
    plt.hist(vals, bins=30, alpha=0.45, label=phase, color=color)
plt.title("Distribution of Mean RR: Baseline vs On-therapy")
plt.xlabel("Mean RR (s)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "mean_rr_histogram.png", dpi=200)
plt.close()

# 4) Posterior forest plot from summary values if trace exists
# If you saved only summary.csv, show a simple coefficient table plot
if "mean" in summary_df.columns and ("beta" in summary_df["Unnamed: 0"].astype(str).values if "Unnamed: 0" in summary_df.columns else True):
    pass

# 5) Plot posterior summary-like bar chart if summary has beta rows
sum_df = summary_df.copy()
if "Unnamed: 0" in sum_df.columns:
    sum_df = sum_df.rename(columns={"Unnamed: 0": "param"})
elif "index" in sum_df.columns:
    sum_df = sum_df.rename(columns={"index": "param"})
else:
    sum_df["param"] = sum_df.index.astype(str)

if "param" in sum_df.columns and "mean" in sum_df.columns:
    beta_rows = sum_df[sum_df["param"].astype(str).str.contains("beta", case=False, na=False)]
    if len(beta_rows) > 0:
        plt.figure(figsize=(7, 4))
        plt.bar(beta_rows["param"].astype(str), beta_rows["mean"], color="#72B7B2")
        plt.axhline(0, color="black", linewidth=1)
        plt.title("Posterior Mean Coefficients")
        plt.xlabel("Parameter")
        plt.ylabel("Posterior Mean")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "posterior_means.png", dpi=200)
        plt.close()

# 6) Save a compact viva table
viva_table = pd.DataFrame({
    "item": [
        "Total subjects",
        "Encainide",
        "Flecainide",
        "Moricizine",
        "Rows with baseline data",
        "Rows with therapy data"
    ],
    "value": [
        feat_df["subject_id"].nunique(),
        (feat_df["drug"] == "Encainide").sum(),
        (feat_df["drug"] == "Flecainide").sum(),
        (feat_df["drug"] == "Moricizine").sum(),
        feat_df["baseline_mean_rr_s"].notna().sum(),
        feat_df["therapy_mean_rr_s"].notna().sum()
    ]
})
viva_table.to_csv(out_dir / "viva_table.csv", index=False)

print("Saved plots and tables in output/")
print(viva_table)