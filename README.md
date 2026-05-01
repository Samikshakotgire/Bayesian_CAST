# CAST Bayesian Analysis Pipeline

Bayesian hierarchical modeling of drug effects on heart rate variability using the CAST (Cardiac Arrhythmia Suppression Trial) database.

## Overview

This project analyzes baseline vs. on-therapy RR-interval (heart rate) features from 809 subject pairs treated with three antiarrhythmic drugs:
- **Encainide** (286 subjects)
- **Flecainide** (229 subjects)
- **Moricizine** (294 subjects)

The pipeline uses Bayesian hierarchical modeling to quantify drug effects on mean RR interval while accounting for inter-subject variability.

## Dataset

**CAST Database (PhysioNet)**
- Total records: 1,543 ECG recordings
- Subject pairs: 809 (baseline + on-therapy per subject)
- Structure: Organized by drug group (e, f, m) with phase codes (a=baseline, b=on-therapy)
- Source: https://physionet.org/content/crisdb/1.0.0/

## Pipeline

### 1. `cast_pipeline.py`
Reads CAST RECORDS file and creates structured baseline/therapy pairs.

**Output:**
- `records_parsed.csv` — All 1,543 records with metadata
- `subject_pairs.csv` — 809 baseline/therapy pairs by drug

```bash
python cast_pipeline.py

2. 3_rr_feature_extraction.py
Extracts RR-interval statistics from annotated beat times for each record.

Features extracted:

n_beats — Number of detected beats
mean_rr_s — Mean RR interval (seconds)
std_rr_s — Standard deviation of RR intervals
rmssd_s — Root mean square of successive RR differences
median_rr_s — Median RR interval
Delta versions — Change from baseline to on-therapy
Output:

cast_rr_features.csv — RR features for all records
.\venv\Scripts\python.exe .\3_rr_feature_extraction.py

3. 4_bayesian_model.py
Fits hierarchical Bayesian model with drug-specific effects on mean RR interval.

Output:

bayesian_summary.csv — Posterior means, credible intervals, Rhat
bayesian_trace.nc — Full posterior samples
python .\4_bayesian_model.py

4. 5_results_plots.py
Generates summary plots and statistics.

Outputs:

group_counts.csv — Subject counts per drug
Visualization PNGs (distributions, posteriors)
python .\5_results_plots.py
Quick Start
Key Results
Model Convergence: Rhat < 1.01 ✓

Drug Effects (P(effect > 0)):

Encainide: 51.1%
Flecainide: 49.1%
Moricizine: 48.8%