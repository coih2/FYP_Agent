"""
Step 4 — fairlearn ThresholdOptimizer (flagged tokens only)
==============================================================
Applies fairlearn's ThresholdOptimizer for disease tokens that were flagged as biased by Step 3.
Unflagged tokens retain their original 0.5 threshold on hazard_prob, preserving their predictive accuracy unchanged.


Inputs : predictions.parquet, outcomes.csv,
         flagged_tokens (set of token IDs from Step 3)
Outputs: threshold_decisions.parquet, threshold_optimizers.pkl
"""

import pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import step0_label_map as s0
from sklearn.base import BaseEstimator, ClassifierMixin
from fairlearn.postprocessing import ThresholdOptimizer


REFERENCE_GROUP = "White"
TARGET_GROUPS   = {"Black", "Asian"}
MIN_THRESH      = 10
CONSTRAINT      = "equalized_odds"
OBJECTIVE       = "balanced_accuracy_score"

# Sklearn wrapper so ThresholdOptimizer can treat Delphi's pre-computed hazard_prob scores as a fitted classifier.
class _DelphiScorer(BaseEstimator, ClassifierMixin):
    def __init__(self, probs):
        self.probs = probs
    def fit(self, X, y):
        self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        p = self.probs[X.flatten().astype(int)]
        return np.column_stack([1 - p, p])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run(pred_path, outcomes_path,
        thresh_path="threshold_decisions.parquet",
        to_pkl="threshold_optimizers.pkl",
        constraint=CONSTRAINT, objective=OBJECTIVE,
        flagged_tokens=None): # None --> all tokens corrected if step 3 is skipped
    preds    = pd.read_parquet(pred_path) if str(pred_path).endswith(".parquet") \
               else s0._read_csv(pred_path)
    outcomes = s0._read_csv(outcomes_path)
    for df in [preds, outcomes]:
        df["patient_id"] = df["patient_id"].astype(str)
        df["token"]      = df["token"].astype(int)

    merged = preds.merge(outcomes[["patient_id","token","event"]],
                         on=["patient_id","token"], how="inner")
    target = merged[merged["ethnicity"].isin(TARGET_GROUPS)].copy()

    if flagged_tokens is not None:
        print(f"Step 4 | ThresholdOptimizer | constraint={constraint} | "
              f"correcting {len(flagged_tokens)} flagged token(s) only")
    else:
        print(f"Step 4 | ThresholdOptimizer | constraint={constraint} | "
              f"correcting all tokens")

    tos, n_fit, n_skip, n_bypass = {}, 0, 0, 0

    for tok, sub in target.groupby("token"):
        # Skip tokens not flagged as biased by Step 3
        if flagged_tokens is not None and tok not in flagged_tokens:
            n_bypass += 1
            continue

        sub = sub.reset_index(drop=True)
        if len(sub) < MIN_THRESH or sub["event"].nunique() < 2:
            n_skip += 1; continue
        if (sub.groupby("ethnicity")["event"].nunique() < 2).any():
            n_skip += 1; continue

        X  = np.arange(len(sub)).reshape(-1, 1)
        to = ThresholdOptimizer(
            estimator      = _DelphiScorer(sub["hazard_prob"].values),
            constraints    = constraint,
            objective      = objective,
            predict_method = "predict_proba",
            prefit         = True)
        try:
            to.fit(X, sub["event"].values,
                   sensitive_features=sub["ethnicity"].values)
            tos[tok] = to; n_fit += 1
        except Exception:
            n_skip += 1

    print(f"  Fitted: {n_fit} | Skipped: {n_skip} | "
          f"Bypassed (not flagged): {n_bypass}")

    records = []
    for tok, sub in merged.groupby("token"):
        sub         = sub.reset_index(drop=True)
        decisions   = (sub["hazard_prob"].values >= 0.5).astype(int)
        target_mask = sub["ethnicity"].isin(TARGET_GROUPS).values

        if tok in tos and target_mask.sum() > 0:
            ti = np.where(target_mask)[0]
            try:
                decisions[target_mask] = tos[tok].predict(
                    ti.reshape(-1, 1),
                    sensitive_features=sub.loc[target_mask,
                                               "ethnicity"].values)
            except Exception:
                pass

        sub["fair_decision"] = decisions

        # threshold_optimized is True only for target-group
        is_flagged_tok = (flagged_tokens is None) or (tok in flagged_tokens)
        sub["threshold_optimized"] = (
            target_mask & (tok in tos) & is_flagged_tok
        )
        records.append(sub)

    result    = pd.concat(records, ignore_index=True)

    # Drop old fair_decision/ threshold_optimized if they already exist in preds
    preds_clean = preds.drop(
        columns=[c for c in ["fair_decision", "threshold_optimized"]
                 if c in preds.columns])

    thresh_df = preds_clean.merge(
        result[["patient_id","token","fair_decision","threshold_optimized"]],
        on=["patient_id","token"], how="left")

    thresh_df.to_parquet(thresh_path, index=False)
    pickle.dump(tos, open(to_pkl, "wb"))
    print(f"Saved: {thresh_path}")
    return thresh_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--outcomes",    required=True)
    p.add_argument("--output",      default="threshold_decisions.parquet")
    p.add_argument("--constraint",  default=CONSTRAINT)
    p.add_argument("--objective",   default=OBJECTIVE)
    a = p.parse_args()
    run(a.predictions, a.outcomes, a.output,
        constraint=a.constraint, objective=a.objective)