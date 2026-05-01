from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

out_dir = Path("output")
feat_path = out_dir / "cast_rr_features.csv"

df = pd.read_csv(feat_path)

# keep subjects with both baseline and therapy info
model_df = df.copy()

# main response: change in mean RR
model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=["delta_mean_rr_s"])

# encode drug group
drug_map = {"Encainide": 0, "Flecainide": 1, "Moricizine": 2}
model_df["drug_code"] = model_df["drug"].map(drug_map)

# simple Bayesian regression:
# delta_mean_rr_s ~ Normal(alpha + beta_drug, sigma)
y = model_df["delta_mean_rr_s"].values.astype(float)
x = model_df["drug_code"].values.astype(int)

with pm.Model() as model:
    alpha = pm.Normal("alpha", 0, 1)
    beta = pm.Normal("beta", 0, 1, shape=3)
    sigma = pm.HalfNormal("sigma", 1)

    mu = alpha + beta[x]
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(
        2000,
        tune=1000,
        chains=2,
        target_accept=0.9,
        random_seed=42,
        cores=1
    )

summary = az.summary(trace, var_names=["alpha", "beta", "sigma"])
print(summary)

# posterior probabilities
beta_post = trace.posterior["beta"].stack(sample=("chain", "draw")).values
prob_positive = (beta_post > 0).mean(axis=1)

for drug, p in zip(["Encainide", "Flecainide", "Moricizine"], prob_positive):
    print(f"P(effect > 0) for {drug}: {p:.3f}")

# save outputs
summary.to_csv(out_dir / "bayesian_summary.csv")

# trace plot
az.plot_trace(trace, var_names=["alpha", "beta", "sigma"])
plt.tight_layout()
plt.savefig(out_dir / "bayesian_trace.png", dpi=200)
plt.close()

# forest plot
az.plot_forest(trace, var_names=["alpha", "beta"], combined=True)
plt.tight_layout()
plt.savefig(out_dir / "bayesian_forest.png", dpi=200)
plt.close()

print("Saved outputs in output/")