"""
Pre-pipeline Fairness Evaluation of Delphi-2M
====================================================================
Measures the fairness and predictive performance of Delphi-2M's raw
predictions across three ethnic groups (in this case: White, Black, Asian) 
before any bias mitigation is applied.


Inputs : predictions.parquet   [patient_id, ethnicity, token, log_hazard, hazard_prob]
         outcomes.csv          [patient_id, token, event]

Outputs: evaluation_baseline/per_group_metrics.csv
         evaluation_baseline/disparity_summary.csv
         evaluation_baseline/plots/auc_by_group.png
         evaluation_baseline/plots/selection_rate_by_group.png
         evaluation_baseline/plots/dp_difference_distribution.png
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_negative_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)

sys.path.insert(0, os.path.dirname(__file__))
import step0_label_map as s0


### CONFIG ###
PREDICTIONS = "Agent/pipeline_output/predictions.parquet"    # Delphi-2M predictions
OUTCOMES    = "Agent/pipeline_output/outcomes.csv"           # Outcome tokens data
LABEL_MAP   = "Agent/Agent_data/conv_token_icd.csv"          # ICD-10 to token conversion
OUTPUT_DIR  = "Agent/pipeline_output/evaluation_baseline"    # Output directory

TOP_Q       = 0.95   # binarisation threshold
MIN_N       = 5      # minimum patients per group per token
GROUPS      = ["White", "Black", "Asian"]    # Ethnicity groups to be evaluated 
REFERENCE   = "White"                        # Baseline ethnicity group




# Compute AUC 
def _auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_prob)
    

#  Compute per-group fairness and performance metrics for each outcome token using fairlearn's MetricFrame.
def compute_per_group_metrics(preds, outcomes):
    merged = preds.merge(
        outcomes[["patient_id", "token", "event"]],
        on=["patient_id", "token"], how="inner"
    )

    outcome_tokens = merged[merged["event"] == 1]["token"].unique()
    merged = merged[merged["token"].isin(outcome_tokens)].copy()

    print(f"  Tokens with outcomes: {len(outcome_tokens)}")
    print(f"  Patients: {merged['patient_id'].nunique()}")
    print(f"  Positive rate: {merged['event'].mean():.5f}")
    print(f"  Group counts:\n"
          f"{merged.groupby('ethnicity')['patient_id'].nunique().to_string()}\n")

    per_group_rows = []
    disparity_rows = []
    n_computed = 0

    for tok, sub in merged.groupby("token"):
        if len(sub) < MIN_N or sub["event"].nunique() < 2:
            continue

        grp_sizes = sub.groupby("ethnicity").size()
        small     = grp_sizes[grp_sizes < 2].index.tolist()
        sub       = sub[~sub["ethnicity"].isin(small)].copy()
        if len(sub) < MIN_N or sub["event"].nunique() < 2:
            continue

        y    = sub["event"].values
        p    = sub["hazard_prob"].values
        yp   = (p >= np.quantile(p, TOP_Q)).astype(int)
        sens = pd.Series(sub["ethnicity"].values, name="ethnicity")

        try:
            mf = MetricFrame(
                metrics={
                    "selection_rate":      selection_rate,
                    "true_positive_rate":  true_positive_rate,
                    "false_negative_rate": false_negative_rate,
                },
                y_true=y, y_pred=yp, sensitive_features=sens)
            by_grp = mf.by_group.reset_index()
            by_grp["token"] = tok
        except Exception as e:
            print(f"  MetricFrame skipped token {tok}: {e}")
            continue

        for grp, gs in sub.groupby("ethnicity"):
            auc = _auc(gs["event"].values, gs["hazard_prob"].values) \
                  if len(gs) >= MIN_N and gs["event"].nunique() == 2 else None
            by_grp.loc[by_grp["ethnicity"] == grp, "auc"] = auc
            by_grp.loc[by_grp["ethnicity"] == grp, "n"]   = len(gs)

        per_group_rows.append(by_grp)

        try:
            dp_diff = demographic_parity_difference(y, yp, sensitive_features=sens)
            eo_diff = equalized_odds_difference(y, yp, sensitive_features=sens)
        except Exception:
            dp_diff, eo_diff = None, None

        disparity_rows.append({
            "token":         tok,
            "n_total":       len(sub),
            "dp_difference": dp_diff,
            "eo_difference": eo_diff,
            "flagged":       abs(dp_diff) > 0.03 if dp_diff is not None else False,
        })
        n_computed += 1

    per_group_df = pd.concat(per_group_rows, ignore_index=True) \
                   if per_group_rows else pd.DataFrame()
    disparity_df = pd.DataFrame(disparity_rows)

    print(f"  Tokens with metrics computed: {n_computed}")
    if not disparity_df.empty:
        print(f"  Tokens flagged (|dp_diff| > 0.03): "
              f"{int(disparity_df['flagged'].sum())}")

    return per_group_df, disparity_df



### Plots ###
def plot_metrics(per_group_df, disparity_df, output_dir):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    colors = {"White": "#4C72B0", "Black": "#DD8452", "Asian": "#55A868"}

    # Plot 1 - Mean AUC by group (for project, White performance as baseline reference)
    if "auc" in per_group_df.columns:
        auc_summary = (per_group_df.groupby("ethnicity")["auc"]
                       .agg(mean="mean", std="std").reset_index())
        white_auc = auc_summary.loc[
            auc_summary["ethnicity"] == "White", "mean"]
        white_auc = float(white_auc.iloc[0]) if not white_auc.empty else None

        fig, ax = plt.subplots(figsize=(6, 4))
        for _, row in auc_summary.iterrows():
            ax.bar(row["ethnicity"], row["mean"],
                   yerr=row["std"], capsize=4,
                   color=colors.get(row["ethnicity"], "grey"), alpha=0.85)
        if white_auc is not None:
            ax.axhline(white_auc, color=colors["White"], ls="--", lw=1.2,
                       label=f"White baseline ({white_auc:.3f})")
        ax.set_ylabel("Mean AUC across outcome tokens")
        ax.set_xlabel("Ethnic group")
        ax.set_title("AUC by ethnic group (raw Delphi-2M predictions)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "auc_by_group.png"), dpi=150)
        plt.close(fig)
        print("  Saved: auc_by_group.png")

    # Plot 2 - Mean selection rate by group (for project, White performance as baseline)
    if "selection_rate" in per_group_df.columns:
        sr_summary = (per_group_df.groupby("ethnicity")["selection_rate"]
                      .agg(mean="mean", std="std").reset_index())
        white_sr = sr_summary.loc[
            sr_summary["ethnicity"] == "White", "mean"]
        white_sr = float(white_sr.iloc[0]) if not white_sr.empty else None

        fig, ax = plt.subplots(figsize=(6, 4))
        for _, row in sr_summary.iterrows():
            ax.bar(row["ethnicity"], row["mean"],
                   yerr=row["std"], capsize=4,
                   color=colors.get(row["ethnicity"], "grey"), alpha=0.85)
        if white_sr is not None:
            ax.axhline(white_sr, color=colors["White"], ls="--", lw=1.2,
                       label=f"White baseline ({white_sr:.3f})")
        ax.set_ylabel("Mean selection rate across outcome tokens")
        ax.set_xlabel("Ethnic group")
        ax.set_title("Selection rate by ethnic group (raw Delphi-2M predictions)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "selection_rate_by_group.png"), dpi=150)
        plt.close(fig)
        print("  Saved: selection_rate_by_group.png")

    # Plot 3 - Distribution of DP difference
    if not disparity_df.empty and "dp_difference" in disparity_df.columns:
        dp_vals = disparity_df["dp_difference"].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dp_vals.abs(), bins=20, color="#4C72B0", alpha=0.75,
                edgecolor="white")
        ax.axvline(0.1, color="red", ls="--", lw=0.9,
                   label="Flag threshold (0.1)")
        ax.set_xlabel("Demographic parity difference (absolute)")
        ax.set_ylabel("Number of disease tokens")
        ax.set_title("Distribution of demographic parity difference\n"
                     "(raw Delphi-2M predictions)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "dp_difference_distribution.png"),
                    dpi=150)
        plt.close(fig)
        print("  Saved: dp_difference_distribution.png")




# Print terminal summary
def print_summary(per_group_df, disparity_df):
    print("\n" + "="*60)
    print("BASELINE FAIRNESS EVALUATION — Delphi-2M")
    print("="*60)

    if not per_group_df.empty:
        print("\nMean metrics by ethnic group (across all outcome tokens):")
        cols = [c for c in ["auc", "selection_rate",
                             "true_positive_rate", "false_negative_rate"]
                if c in per_group_df.columns]
        summary = per_group_df.groupby("ethnicity")[cols].mean()
        print(summary.round(4).to_string())

    if not disparity_df.empty:
        print(f"\nDisparity summary ({len(disparity_df)} tokens evaluated):")
        print(f"  Mean |DP difference|:  "
              f"{disparity_df['dp_difference'].abs().mean():.4f}")
        print(f"  Mean |EO difference|:  "
              f"{disparity_df['eo_difference'].abs().mean():.4f}")
        print(f"  Tokens flagged (|dp| > 0.1): "
              f"{int(disparity_df['flagged'].sum())}")
        print("\nTop 10 most biased tokens by |DP difference|:")
        top10 = (disparity_df.sort_values("dp_difference", key=abs,
                                           ascending=False)
                 .head(10)[["token", "dp_difference", "eo_difference",
                              "n_total", "flagged"]])
        print(top10.to_string(index=False))
    print("="*60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    preds     = pd.read_parquet(PREDICTIONS)
    outcomes  = s0._read_csv(OUTCOMES)
    label_map = s0.load_label_map(LABEL_MAP)

    for df in [preds, outcomes]:
        df["patient_id"] = df["patient_id"].astype(str)
        df["token"]      = df["token"].astype(int)

    print(f"Predictions: {preds.shape} | Outcomes: {outcomes.shape}")
    print(f"Groups in data: {preds['ethnicity'].unique().tolist()}\n")

    print("Computing per-group metrics...")
    per_group_df, disparity_df = compute_per_group_metrics(preds, outcomes)

    if not per_group_df.empty:
        per_group_df = s0.apply_labels(per_group_df, label_map)
    if not disparity_df.empty:
        disparity_df = s0.apply_labels(disparity_df, label_map)

    per_group_df.to_csv(
        os.path.join(OUTPUT_DIR, "per_group_metrics.csv"), index=False)
    disparity_df.to_csv(
        os.path.join(OUTPUT_DIR, "disparity_summary.csv"), index=False)
    print(f"\nSaved outputs to: {OUTPUT_DIR}/")

    print("Generating plots...")
    plot_metrics(per_group_df, disparity_df, OUTPUT_DIR)

    print_summary(per_group_df, disparity_df)


if __name__ == "__main__":
    main()
