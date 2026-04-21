"""
Step 5 — Corrected Output + Fairness Evaluation
===================================================
Assembles the corrected predictions from step 4, writes audit logs,
then re-runs fairlearn MetricFrame to compare bias before and after
the ThresholdOptimizer correction.


Inputs : threshold_decisions.parquet  (step 4)
         predictions.parquet          (step 2, raw)
         outcomes.csv                 (step 2)
Outputs: corrected_predictions.parquet
         patient_audit_log.csv
         disease_correction_summary.csv
         evaluation/auc_comparison.csv
         evaluation/dp_comparison.csv
         evaluation/plots/
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import step0_label_map as s0
import matplotlib.pyplot as plt
from step3_audit import compute_metrics, FLAG_THRESHOLD


TOP_K = 10


def run(thresh_path, raw_pred_path, outcomes_path,
        out_path="corrected_predictions.parquet",
        audit_path="patient_audit_log.csv",
        disease_path="disease_correction_summary.csv",
        eval_dir="evaluation"):

    os.makedirs(f"{eval_dir}/plots", exist_ok=True)

    thresh   = pd.read_parquet(thresh_path)
    raw_pred = pd.read_parquet(raw_pred_path)
    outcomes = s0._read_csv(outcomes_path)

    for df in [thresh, raw_pred, outcomes]:
        df["patient_id"] = df["patient_id"].astype(str)
        df["token"]      = df["token"].astype(int)

    # Guards: protect against parquet files written by older pipeline versions
    if "fair_decision" not in thresh.columns:
        thresh["fair_decision"] = (thresh["hazard_prob"] >= 0.5).astype(int)
    if "threshold_optimized" not in thresh.columns:
        thresh["threshold_optimized"] = False

    n_optimized = int(thresh["threshold_optimized"].sum())
    n_tokens_optimized = int(
        thresh[thresh["threshold_optimized"]]["token"].nunique()
    )
    print(f"Step 5 | {thresh['patient_id'].nunique()} patients | "
          f"{n_optimized} rows threshold-optimized "
          f"across {n_tokens_optimized} token(s)")

    ### Per-patient audit log 
    records = []
    for pid, grp in thresh.groupby("patient_id"):
        top = grp.nlargest(TOP_K, "hazard_prob")
        records.append({
            "patient_id":          pid,
            "ethnicity":           grp["ethnicity"].iloc[0],
            "mean_hazard_top10":   top["hazard_prob"].mean(),
            "n_threshold_changed": int(grp["threshold_optimized"].sum()),
            "pct_high_risk_raw":   (grp["hazard_prob"] >= 0.5).mean(),
            "pct_high_risk_fair":  grp["fair_decision"].mean(),
        })
    audit = pd.DataFrame(records)

    ### Disease correction summary
    disease_summary = (thresh.groupby(["token","ethnicity"])
                       .agg(mean_hazard=("hazard_prob","mean"),
                            pct_high_risk_raw=("hazard_prob",
                                               lambda x: (x>=0.5).mean()),
                            pct_high_risk_fair=("fair_decision","mean"),
                            n_optimized=("threshold_optimized","sum"),
                            n=("patient_id","count"))
                       .reset_index())

    thresh.to_parquet(out_path, index=False)
    audit.to_csv(audit_path, index=False)
    disease_summary.to_csv(disease_path, index=False)

    print(f"Mean % high risk — raw:  {audit['pct_high_risk_raw'].mean():.4f}")
    print(f"Mean % high risk — fair: {audit['pct_high_risk_fair'].mean():.4f}")
    print("By group:")
    print(audit.groupby("ethnicity")[
        ["pct_high_risk_raw","pct_high_risk_fair"]].mean().to_string())

    ### Fairness evaluation: MetricFrame before and after 
    print("\nRunning MetricFrame on raw predictions...")
    _, fs_raw, _ = compute_metrics(raw_pred, outcomes)

    ### Only replace hazard_prob with fair_decision for rows where threshold_optimized is True 
    corr_for_audit = thresh.copy()
    optimized_mask = corr_for_audit["threshold_optimized"].fillna(False).astype(bool)
    corr_for_audit.loc[optimized_mask, "hazard_prob"] = \
        corr_for_audit.loc[optimized_mask, "fair_decision"].astype(float)

    print("Running MetricFrame on corrected predictions...")
    _, fs_corr, _ = compute_metrics(corr_for_audit, outcomes)

    if fs_raw.empty or fs_corr.empty:
        print("[WARNING] Not enough data for before/after comparison.")
        return thresh, audit

    ### AUC comparison
    auc = (fs_raw[["token","overall_auc"]]
           .rename(columns={"overall_auc":"auc_before"})
           .merge(fs_corr[["token","overall_auc"]]
                  .rename(columns={"overall_auc":"auc_after"}),
                  on="token", how="inner"))
    auc["auc_delta"] = auc["auc_after"] - auc["auc_before"]
    auc.to_csv(f"{eval_dir}/auc_comparison.csv", index=False)

    ### Demographic parity comparison
    dp = (fs_raw[["token","dp_difference","flagged"]]
          .rename(columns={"dp_difference":"dp_before","flagged":"flagged_before"})
          .merge(fs_corr[["token","dp_difference","flagged"]]
                 .rename(columns={"dp_difference":"dp_after","flagged":"flagged_after"}),
                 on="token", how="inner"))
    dp["dp_delta"]      = dp["dp_after"] - dp["dp_before"]
    dp["newly_fixed"]   = dp["flagged_before"] & ~dp["flagged_after"]
    dp["newly_flagged"] = ~dp["flagged_before"] & dp["flagged_after"]
    dp.to_csv(f"{eval_dir}/dp_comparison.csv", index=False)

    print(f"\nAUC  — before: {auc['auc_before'].mean():.4f} | "
          f"after: {auc['auc_after'].mean():.4f}")
    print(f"DP   — tokens fixed: {dp['newly_fixed'].sum()} | "
          f"newly flagged: {dp['newly_flagged'].sum()}")

    ### Plots ###
    colors = {"White": "#4C72B0", "Black": "#DD8452", "Asian": "#55A868"}

    # Plot 1: AUC scatter before vs after
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(auc["auc_before"], auc["auc_after"], alpha=0.5, s=15)
    lo = min(auc["auc_before"].min(), auc["auc_after"].min()) - 0.02
    hi = max(auc["auc_before"].max(), auc["auc_after"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="No change")
    ax.set_xlabel("AUC before"); ax.set_ylabel("AUC after")
    ax.set_title("AUC before vs after ThresholdOptimizer")
    ax.legend(); plt.tight_layout()
    fig.savefig(f"{eval_dir}/plots/auc_before_after.png", dpi=150)
    plt.close(fig)

    # Plot 2: Mean AUC per ethnic group before and after
    mf_before, _, _ = compute_metrics(raw_pred, outcomes)
    mf_after,  _, _ = compute_metrics(corr_for_audit, outcomes)
    if not mf_before.empty and not mf_after.empty \
            and "auc" in mf_before.columns and "auc" in mf_after.columns:
        auc_grp_before = (mf_before.groupby("ethnicity")["auc"]
                          .mean().reset_index()
                          .rename(columns={"auc": "auc_before"}))
        auc_grp_after  = (mf_after.groupby("ethnicity")["auc"]
                          .mean().reset_index()
                          .rename(columns={"auc": "auc_after"}))
        auc_grp = auc_grp_before.merge(auc_grp_after, on="ethnicity")

        white_before = float(auc_grp.loc[
            auc_grp["ethnicity"] == "White", "auc_before"].iloc[0]) \
            if "White" in auc_grp["ethnicity"].values else None

        x = range(len(auc_grp)); width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([i - width/2 for i in x], auc_grp["auc_before"],
               width, label="Before", color="#E8593C", alpha=0.8)
        ax.bar([i + width/2 for i in x], auc_grp["auc_after"],
               width, label="After",  color="#1D9E75", alpha=0.8)
        if white_before is not None:
            ax.axhline(white_before, color=colors["White"], ls="--", lw=1.2,
                       label=f"White baseline ({white_before:.3f})")
        ax.set_xticks(list(x))
        ax.set_xticklabels(auc_grp["ethnicity"].tolist())
        ax.set_ylabel("Mean AUC across outcome tokens")
        ax.set_xlabel("Ethnic group")
        ax.set_title("Mean AUC per ethnic group: before vs after")
        ax.legend(); plt.tight_layout()
        fig.savefig(f"{eval_dir}/plots/auc_by_group_before_after.png", dpi=150)
        plt.close(fig)

    # Plot 3: DP difference before vs after
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(dp[["dp_before","dp_after"]].max()) + 0.05, 25)
    ax.hist(dp["dp_before"], bins=bins, alpha=0.6, label="Before",
            color="#E8593C")
    ax.hist(dp["dp_after"],  bins=bins, alpha=0.6, label="After",
            color="#1D9E75")
    ax.axvline(FLAG_THRESHOLD, color="gray", ls="--", lw=0.8,
               label=f"Flag threshold ({FLAG_THRESHOLD})")
    ax.set_xlabel("Demographic parity difference")
    ax.set_ylabel("Number of tokens")
    ax.set_title("Demographic parity: before vs after")
    ax.legend(); plt.tight_layout()
    fig.savefig(f"{eval_dir}/plots/dp_before_after.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {eval_dir}/plots/")
    return thresh, audit


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--thresh",    required=True)
    p.add_argument("--raw-preds", required=True)
    p.add_argument("--outcomes",  required=True)
    p.add_argument("--output",    default="corrected_predictions.parquet")
    p.add_argument("--audit",     default="patient_audit_log.csv")
    p.add_argument("--disease",   default="disease_correction_summary.csv")
    p.add_argument("--eval-dir",  default="evaluation")
    a = p.parse_args()
    run(a.thresh, a.raw_preds, a.outcomes,
        a.output, a.audit, a.disease, a.eval_dir)