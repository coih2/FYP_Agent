"""
Step 3 — Bias Audit (fairlearn MetricFrame)
=====================================================
Computes per-group fairness metrics using fairlearn's MetricFrame.


Inputs : predictions.csv, outcomes.csv
Outputs: audit/metricframe_by_disease.csv
         audit/fairness_summary.csv
         audit/mean_risk_by_group.csv
         audit/small_group_log.csv        (if any groups are undersized)
"""
import warnings
warnings.filterwarnings("ignore")  # suppress UserWarning noise from fairlearn/sklearn
import numpy as np
import pandas as pd
import step0_label_map as s0
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (


    MetricFrame, selection_rate, true_positive_rate, false_negative_rate,
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, equalized_odds_ratio,
)

# Set constants 
MIN_N = 5
TOP_Q = 0.95
FLAG_THRESHOLD = 0.03

REFERENCE_GROUP = "White"
TARGET_GROUPS   = {"Black", "Asian"}


# Help to run MetricFrame 
def _binarise(probs): return (probs >= np.quantile(probs, TOP_Q)).astype(int)

def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else None

# Returns None on failure
def _safe(fn, *args, **kw):
    try: return fn(*args, **kw)
    except Exception: return None


### Compute metrics

def compute_metrics(preds_df, outcomes_df):
    preds_df    = preds_df.copy()
    outcomes_df = outcomes_df.copy()
    preds_df["patient_id"]    = preds_df["patient_id"].astype(str)
    outcomes_df["patient_id"] = outcomes_df["patient_id"].astype(str)
    preds_df["token"]         = preds_df["token"].astype(int)
    outcomes_df["token"]      = outcomes_df["token"].astype(int)

    merged = preds_df.merge(outcomes_df[["patient_id","token","event"]],
                            on=["patient_id","token"], how="inner")

    print(f"  Merged rows: {len(merged)} | "
          f"unique tokens: {merged['token'].nunique()} | "
          f"patients: {merged['patient_id'].nunique()}")
    print(f"  Overall event rate: {merged['event'].mean():.5f} "
          f"({int(merged['event'].sum())} positive labels out of {len(merged)})")

    # Show the tokens that have event=1 after the merge (debug)
    tokens_with_positives = merged[merged["event"]==1]["token"].unique()
    print(f"  Tokens with at least one event=1 after merge: {len(tokens_with_positives)}")
    if len(tokens_with_positives) > 0:
        sample_tok = tokens_with_positives[0]
        sample = merged[merged["token"] == sample_tok]
        print(f"  Sample token {sample_tok}: {len(sample)} rows | "
              f"event=1: {sample['event'].sum()} | "
              f"event=0: {(sample['event']==0).sum()} | "
              f"groups: {sample['ethnicity'].unique().tolist()}")
    else:
        print("No tokens have event=1 after merge — "
              "the merge is dropping all positive labels.")
        print(f"  Checking dtypes: predictions patient_id={preds_df['patient_id'].dtype} "
              f"token={preds_df['token'].dtype}")
        print(f"  outcomes patient_id={outcomes_df['patient_id'].dtype} "
              f"token={outcomes_df['token'].dtype}")
        # Show overlap between patient IDs
        pred_pids = set(preds_df["patient_id"].unique())
        out_pids  = set(outcomes_df[outcomes_df["event"]==1]["patient_id"].unique())
        overlap   = pred_pids & out_pids
        print(f"  Patient ID overlap: {len(overlap)} patients in common "
              f"(preds has {len(pred_pids)}, outcomes has {len(out_pids)})")

    # Keep only tokens that appear at least once as an outcome (event=1).
    outcome_tokens = merged[merged["event"] == 1]["token"].unique()
    merged = merged[merged["token"].isin(outcome_tokens)].copy()
    print(f"  Restricting to {len(outcome_tokens)} tokens that appear as outcomes "
          f"({merged['token'].nunique()} tokens remain after filter)")

    mf_rows, fs_rows, small_log = [], [], []
    n_skip_size = n_skip_noclass = n_skip_postfilter = 0

    for tok, sub in merged.groupby("token"):
        n_total   = len(sub)
        n_pos     = int(sub["event"].sum())
        n_classes = sub["event"].nunique()
        grp_sizes = sub.groupby("ethnicity").size().to_dict()

        # Print details for first 5 tokens to diagnose
        if n_skip_size + n_skip_noclass + n_skip_postfilter + len(fs_rows) < 5:
            print(f"  Token {tok}: n={n_total} pos={n_pos} "
                  f"classes={n_classes} groups={grp_sizes}")

        if len(sub) < MIN_N:
            n_skip_size += 1
            continue
        if sub["event"].nunique() < 2:
            n_skip_noclass += 1
            continue

        grp_sizes = sub.groupby("ethnicity").size()
        small = grp_sizes[grp_sizes < 2].index.tolist()
        for g in small:
            small_log.append({"token": tok, "ethnicity": g,
                               "n": int(grp_sizes[g]), "note": "below_5_for_metricframe"})
        sub = sub[~sub["ethnicity"].isin(small)].copy()
        if len(sub) < MIN_N or sub["event"].nunique() < 2:
            n_skip_postfilter += 1
            continue

        y, s, sens = sub["event"].values, sub["hazard_prob"].values, sub["ethnicity"].values
        yp = _binarise(s)

        try:
            # Pass sensitive_features as a pandas Series with a named index (so MetricFrame uses 'ethnicity')
            sens_series = pd.Series(sens, name="ethnicity")
            mf = MetricFrame(
                metrics={"selection_rate": selection_rate,
                         "true_positive_rate": true_positive_rate,
                         "false_negative_rate": false_negative_rate},
                y_true=y, y_pred=yp, sensitive_features=sens_series)
            by_grp = mf.by_group.reset_index()
            by_grp["token"] = tok
            by_grp["group_role"] = by_grp["ethnicity"].apply(
                lambda g: "reference" if g == REFERENCE_GROUP else "target")
            by_grp["n"] = by_grp["ethnicity"].map(grp_sizes)

            for g, gs in sub.groupby("ethnicity"):
                auc = _auc(gs["event"].values, gs["hazard_prob"].values) if len(gs) >= MIN_N else None
                if len(gs) < MIN_N:
                    small_log.append({"token": tok, "ethnicity": g,
                                      "n": len(gs), "note": f"below_{MIN_N}_for_auc_ece"})
                by_grp.loc[by_grp["ethnicity"]==g, "auc"] = auc
            mf_rows.append(by_grp)
        except Exception as e:
            print(f"  MetricFrame skipped token {tok}: {e}"); continue

        fs_rows.append({
            "token":        tok,
            "n_patients":   len(sub),
            "overall_auc":  _auc(y, s),
            "dp_difference": _safe(demographic_parity_difference, y, yp, sensitive_features=sens_series),
            "dp_ratio":      _safe(demographic_parity_ratio,      y, yp, sensitive_features=sens_series),
            "eo_difference": _safe(equalized_odds_difference,     y, yp, sensitive_features=sens_series),
            "eo_ratio":      _safe(equalized_odds_ratio,          y, yp, sensitive_features=sens_series),
        })

    print(f"  Tokens skipped — too few patients (<{MIN_N}): {n_skip_size}")
    print(f"  Tokens skipped — only one outcome class:      {n_skip_noclass}")
    print(f"  Tokens skipped — after small-group removal:   {n_skip_postfilter}")
    print(f"  Tokens with MetricFrame computed:             {len(fs_rows)}")

    if not mf_rows:
        print(" [WARNING] No tokens passed all filters. Possible causes:")
        print(f"    1. MIN_N={MIN_N} is too high for your dataset size.")
        print(f"       Try lowering it at the top of step3_audit.py.")
        print(f"    2. Most tokens have no event=1 labels (positive outcome rate is very low).")
        print(f"       Check outcomes.csv — if event=1 rows are rare, masking may have")
        print(f"       not captured enough outcome events.")
        print(f"    3. Patient counts per ethnic group per token are all below 5.")
        print(f"       With only 3 groups this is unlikely but worth checking.")

    mf_df = pd.concat(mf_rows, ignore_index=True) if mf_rows else pd.DataFrame()
    fs_df = pd.DataFrame(fs_rows)
    if not fs_df.empty:
        fs_df["flagged"] = fs_df["dp_difference"].abs() > FLAG_THRESHOLD
        fs_df = fs_df.sort_values("dp_difference", key=abs, ascending=False)

    return mf_df, fs_df, pd.DataFrame(small_log)

# Run audit
def run(pred_path, outcomes_path, output_dir="audit"):
    import os; os.makedirs(output_dir, exist_ok=True)
    preds    = pd.read_parquet(pred_path) if str(pred_path).endswith(".parquet") \
               else s0._read_csv(pred_path)
    outcomes = s0._read_csv(outcomes_path)
    print(f"Audit | predictions {preds.shape} | outcomes {outcomes.shape}")

    mf_df, fs_df, small_df = compute_metrics(preds, outcomes)
    # Placeholder files if empty to prevent crash later on
    if mf_df.empty:
        mf_df = pd.DataFrame(columns=["token","ethnicity","selection_rate",
                                       "true_positive_rate","false_negative_rate",
                                       "group_role","n","auc"])
    if fs_df.empty:
        fs_df = pd.DataFrame(columns=["token","n_patients","overall_auc",
                                       "dp_difference","dp_ratio","eo_difference",
                                       "eo_ratio","flagged"])
    mf_df.to_csv(f"{output_dir}/metricframe_by_disease.csv", index=False)
    fs_df.to_csv(f"{output_dir}/fairness_summary.csv", index=False)
    if not small_df.empty:
        small_df.to_csv(f"{output_dir}/small_group_log.csv", index=False)
        print(f"[WARNING] {len(small_df)} (disease, group) pairs below minimum N — "
              f"see small_group_log.csv")

    # Restrict mean_risk to outcome tokens only
    outcome_tokens = outcomes[outcomes["event"] == 1]["token"].unique()
    mean_risk = (preds[preds["token"].isin(outcome_tokens)]
                 .groupby(["token","ethnicity"])["hazard_prob"]
                 .agg(mean="mean", std="std", n="count").reset_index())
    mean_risk.to_csv(f"{output_dir}/mean_risk_by_group.csv", index=False)

    if not fs_df.empty:
        n_flagged = fs_df["flagged"].sum()
        print(f"Diseases flagged (|dp_diff| > {FLAG_THRESHOLD}): {n_flagged}")
        print(fs_df[["token","dp_difference","eo_difference","overall_auc"]]
              .head(10).to_string(index=False))
    else:
        n_flagged = 0

    bias_detected = int(n_flagged) > 0
    print(f"Bias detected: {bias_detected}")

    return {"metricframe": mf_df, "fairness_summary": fs_df,
            "mean_risk": mean_risk, "small_group_log": small_df,
            "bias_detected": bias_detected}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--outcomes",    required=True)
    p.add_argument("--output-dir",  default="audit")
    a = p.parse_args()
    run(a.predictions, a.outcomes, a.output_dir)