"""
Step 1.2 — Record Linkage
====================================================================
Join the health trajectory dataset with the ethnicity dataset on patient_id.

Inputs : primary CSV      [patient_id, age, token]
         ethnicity CSV    [patient_id, ethnicity]  
Output : linked_data.csv  (primary data with ethnicity column added)
"""
import pandas as pd
import step0_label_map as s0



VALID_GROUPS   = {"Asian", "Black", "White"}
MIN_GROUP_SIZE = 50


def run(primary_path, ethnicity_path, output_path="linked_data.csv"):
    primary   = s0._read_csv(primary_path)
    ethnicity = s0._read_csv(ethnicity_path).drop_duplicates("patient_id")
    merged    = primary.merge(ethnicity[["patient_id", "ethnicity"]],
                              on="patient_id", how="left")

    # Warn about any values that are not Asian / Black / White
    unexpected = set(merged["ethnicity"].dropna().unique()) - VALID_GROUPS
    if unexpected:
        print(f"[WARNING] Unexpected ethnicity values (kept as-is): {unexpected}")

    linked    = merged[merged["ethnicity"].notna()].copy()
    unmatched = merged[merged["ethnicity"].isna()]

    n_total = primary["patient_id"].nunique()
    print(f"Linked {linked['patient_id'].nunique()}/{n_total} patients")
    if not unmatched.empty:
        print(f"Unmatched (no ethnicity record): {unmatched['patient_id'].nunique()}")

    counts = linked.groupby("ethnicity")["patient_id"].nunique()
    for grp, n in counts.items():
        flag = " *** SMALL — may affect metric reliability ***" if n < MIN_GROUP_SIZE else ""
        print(f"  {grp:<10} {n}{flag}")

    linked.to_csv(output_path, index=False)
    return linked


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Step 1: Link health data with ethnicity")
    p.add_argument("--health-data",    required=True,
                   help="Path to primary CSV: patient_id, age, token")
    p.add_argument("--ethnicity-data", required=True,
                   help="Path to ethnicity CSV: patient_id, ethnicity")
    p.add_argument("--output",         default="linked_data.csv",
                   help="Where to save the linked output")
    a = p.parse_args()
    run(a.health_data, a.ethnicity_data, a.output)