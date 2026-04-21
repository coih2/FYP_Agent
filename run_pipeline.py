import os, sys, time, shutil
import pandas as pd


### CONFIG ###
HEALTH_DATA     = "Agent_data/patients_input.csv"       # patient_id, age, token
ETHNICITY_DATA  = "Agent_data/ethnicity_input.csv"      # patient_id, ethnicity
LABEL_MAP       = "Agent_data/conv_token_icd.csv"
CHECKPOINT_DIR  = "/Users/isabellehunt/Delphi/Delphi-2M"      # folder containing Delphi-2M model checkpoints
OUTPUT_DIR      = "pipeline_output"
DELPHI_REPO     = "/Users/isabellehunt/Delphi"                # folder containing model.py

DEVICE          = "cpu"       # "cpu", "cuda", or "mps"
CONSTRAINT      = "equalized_odds"
OBJECTIVE       = "balanced_accuracy_score"
SKIP_STEPS      = ""          # e.g. "1,2" to skip steps 1 and 2


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, DELPHI_REPO)

import step0_label_map      as s0
import step1_record_linkage as s1
import step2_inference      as s2
import step3_audit          as s3
import step4_calibration    as s4
import step5_evaluate       as s5



def p(folder, name): return os.path.join(folder, name)
 
def _label_csv(path, label_map):
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    if "token" not in df.columns: return
    s0.apply_labels(df, label_map).to_csv(path, index=False)
 
def _label_parquet(path, label_map):
    if not os.path.exists(path): return
    df = pd.read_parquet(path)
    if "token" not in df.columns: return
    s0.apply_labels(df, label_map).to_parquet(path, index=False)
 
def _still_flagged(eval_dir):
    path = p(eval_dir, "dp_comparison.csv")
    if not os.path.exists(path): return False
    df = pd.read_csv(path)
    return "flagged_after" in df.columns and bool(df["flagged_after"].sum() > 0)
 
 
def main():
    t0   = time.time()
    out  = OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    skip = set(SKIP_STEPS.split(",")) if SKIP_STEPS else set()
 
    # Fixed paths, defined in step files
    linked    = p(out, "linked_data.csv")
    preds     = p(out, "predictions.parquet")
    outcomes  = p(out, "outcomes.csv")
    mask_rep  = p(out, "masking_report.csv")
    audit_dir = p(out, "audit")
    mf_csv    = p(audit_dir, "metricframe_by_disease.csv")
    eval_dir  = p(out, "evaluation")
    label_map_path = p(out, "label_map.csv")
 
    # Step 1.1 (0) always runs
    label_map = s0.run(LABEL_MAP, label_map_path)
 
    # Steps 1.2 to 3, run once
    if "1" not in skip:
        s1.run(HEALTH_DATA, ETHNICITY_DATA, linked)
 
    if "2" not in skip:
        s2.run(linked, CHECKPOINT_DIR, preds, outcomes, mask_rep,
               DEVICE, "float32", 1337, 32)
 
    if "3" not in skip:
        audit_result  = s3.run(preds, outcomes, audit_dir)
        bias_detected = audit_result["bias_detected"]
        # Extract the set of flagged token IDs to pass to Step 4
        fs = audit_result["fairness_summary"]
        flagged_tokens = set(
            fs.loc[fs["flagged"], "token"].astype(int).tolist()
        ) if not fs.empty else set()
        print(f"  Flagged tokens to correct: {sorted(flagged_tokens)}")
    else:
        # If step 3 is skipped, assume bias is present and correct all tokens
        bias_detected  = True
        flagged_tokens = None
 
    if not bias_detected:
        print("\n✓ No bias detected in step 3 — steps 4 and 5 are skipped.")
        print("  Copying raw predictions to corrected_predictions.parquet.")
        shutil.copy(preds, p(out, "corrected_predictions.parquet"))
    else:
        # Steps 4 to 5, iterate until no bias remains
        input_preds = preds
        iteration   = 0
 
        while True:
            iteration += 1
            label = f"_iter{iteration}" if iteration > 1 else ""
            def ip(name):
                base, ext = os.path.splitext(name)
                return p(out, f"{base}{label}{ext}")
 
            thresh_path    = ip("threshold_decisions.parquet")
            corrected_path = ip("corrected_predictions.parquet")
            audit_log_path = ip("patient_audit_log.csv")
            dis_sum_path   = ip("disease_correction_summary.csv")
            iter_eval      = p(eval_dir, f"iter{iteration}") if iteration > 1 \
                             else eval_dir
 
            print(f"\n{'─'*40}\nCorrection pass {iteration}\n{'─'*40}")
 
            if "4" not in skip:
                s4.run(input_preds, outcomes, thresh_path,
                       constraint=CONSTRAINT, objective=OBJECTIVE,
                       flagged_tokens=flagged_tokens)
 
            if "5" not in skip:
                s5.run(thresh_path, preds, outcomes, corrected_path,
                       audit_log_path, dis_sum_path, iter_eval)
 
            # Stop if no bias remains after this pass
            if not _still_flagged(iter_eval):
                print(f"\n No diseases remain flagged after pass {iteration}."
                      f" Pipeline converged.")
                break
 
            # Stop if no improvement since last pass (avoid infinite loop)
            prev_eval = p(eval_dir, f"iter{iteration-1}") \
                        if iteration > 1 else None
            if prev_eval and os.path.exists(prev_eval):
                prev_flagged = sum(1 for _ in open(
                    p(prev_eval, "dp_comparison.csv"))) - 1
                curr_flagged = sum(1 for _ in open(
                    p(iter_eval, "dp_comparison.csv"))) - 1
                if curr_flagged >= prev_flagged:
                    print(f"\n No improvement after pass {iteration} "
                          f"({curr_flagged} tokens still flagged). "
                          f"Stopping.")
                    break
 
            input_preds = corrected_path
 
        # Copy final corrected output 
        if label:
            shutil.copy(corrected_path,
                        p(out, "corrected_predictions.parquet"))
 
    # Apply ICD labels to all outputs 
    print("\nApplying ICD labels...")
    for f in ["outcomes.csv", "patient_audit_log.csv",
              "disease_correction_summary.csv",
              p("audit", "metricframe_by_disease.csv"),
              p("audit", "fairness_summary.csv"),
              p("audit", "mean_risk_by_group.csv"),
              p("evaluation", "auc_comparison.csv"),
              p("evaluation", "dp_comparison.csv")]:
        _label_csv(p(out, f) if not f.startswith(out) else f, label_map)
    for f in ["predictions.parquet", "threshold_decisions.parquet",
              "corrected_predictions.parquet"]:
        _label_parquet(p(out, f), label_map)
 
    ### Termnal Summary
    print(f"\n{'='*50}")
    print(f"Done in {time.time()-t0:.1f}s  →  {out}/")
    print(f"{'='*50}")
    for fname, desc in [
        ("label_map.csv",                    "Token to ICD mapping"),
        ("linked_data.csv",                  "Health data + ethnicity"),
        ("outcomes.csv",                     "Ground-truth labels"),
        ("predictions.parquet",              "Raw Delphi predictions"),
        ("audit/fairness_summary.csv",       "Baseline bias audit"),
        ("audit/metricframe_by_disease.csv", "Per-(token,group) MetricFrame"),
        ("threshold_decisions.parquet",      "After ThresholdOptimizer"),
        ("corrected_predictions.parquet",    "Final corrected predictions"),
        ("patient_audit_log.csv",            "Per-patient correction log"),
        ("evaluation/auc_comparison.csv",    "AUC before vs after"),
        ("evaluation/dp_comparison.csv",     "Demographic parity before vs after"),
        ("evaluation/plots/",                "Before/after plots"),
    ]:
        mark = "✓" if os.path.exists(p(out, fname)) else "✗"
        print(f"  {mark}  {fname:<45} {desc}")
 
 
if __name__ == "__main__":
    main()