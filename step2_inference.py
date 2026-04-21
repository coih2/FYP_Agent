"""
Step 2 — Delphi-2M Inference with Outcome Masking
=====================================================
Loads Delphi, masks each patient's final-age tokens as outcomes,
runs inference on the remaining history, and saves predictions.


Inputs : linked_data.csv
Outputs: predictions.parquet  [patient_id, ethnicity, token, log_hazard, hazard_prob]
         outcomes.csv         [patient_id, token, event]
         masking_report.csv   [patient_id, final_age, n_outcomes, excluded, reason]


The Delphi-2M model inference code is used as instructed by the model developers.
(Shmatko, A., Jung, A.W., Gaurav, K. et al. 
Learning the natural history of human disease with generative transformers. 
Nature 647, 248–256 (2025). 
https://doi.org/10.1038/s41586-025-09529-3)
"""
import os
import torch
import numpy as np
import pandas as pd
import step0_label_map as s0
from contextlib import nullcontext



DTYPE_MAP = {"float32": torch.float32, "float64": torch.float64,
             "bfloat16": torch.bfloat16, "float16": torch.float16}


def run(linked_path, checkpoint_dir,
        pred_path="predictions.parquet",
        outcomes_path="outcomes.csv",
        report_path="masking_report.csv",
        device="cpu", dtype="float32", seed=1337, batch_size=32):

    # Load model
    try:
        from model import Delphi, DelphiConfig
    except ImportError:
        raise ImportError("Add Delphi repo to PYTHONPATH: "
                          "export PYTHONPATH=/path/to/Delphi:$PYTHONPATH")
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    ckpt  = torch.load(os.path.join(checkpoint_dir, "ckpt.pt"),
                       map_location=device)
    conf  = DelphiConfig(**ckpt["model_args"])
    model = Delphi(conf)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    if device != "cpu":
        model.to(DTYPE_MAP[dtype])
    vocab_size = conf.vocab_size
    block_size = conf.block_size
    print(f"Loaded Delphi | vocab={vocab_size} block={block_size} device={device}")

    # Load and mask data
    df = s0._read_csv(linked_path)
    df["patient_id"] = df["patient_id"].astype(str)

    history_rows, outcome_rows, report_rows = [], [], []
    for pid, grp in df.groupby("patient_id"):
        grp       = grp.sort_values("age")
        final_age = grp["age"].max()
        history   = grp[grp["age"] < final_age]
        outcomes  = grp[grp["age"] == final_age]
        in_vocab  = outcomes["token"].between(0, vocab_size - 1).sum()

        if len(history) == 0 or in_vocab == 0:
            reason = ("no_history" if len(history) == 0
                      else "outcomes_not_in_vocab")
            report_rows.append({"patient_id": pid, "final_age": final_age,
                                 "n_outcomes": len(outcomes),
                                 "excluded": True, "reason": reason})
            continue

        history_rows.append(history)
        for _, r in outcomes.iterrows():
            if 0 <= r["token"] < vocab_size:
                outcome_rows.append({"patient_id": pid,
                                     "token": int(r["token"]), "event": 1})
        report_rows.append({"patient_id": pid, "final_age": final_age,
                             "n_outcomes": len(outcomes),
                             "excluded": False, "reason": ""})

    history_df  = pd.concat(history_rows, ignore_index=True)
    outcomes_e1 = pd.DataFrame(outcome_rows)
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"Masking: {len(history_rows)} included | "
          f"{len(report_rows)-len(history_rows)} excluded | "
          f"{len(outcomes_e1)} event=1 labels")

    # Build sequences 
    valid_pids = set(outcomes_e1["patient_id"])
    sequences  = {}
    for pid, grp in history_df.groupby("patient_id"):
        if pid not in valid_pids:
            continue
        grp = grp.sort_values("age")
        sequences[pid] = {
            "age_toks": grp["age"].astype(int).tolist(),
            "dis_toks": grp["token"].astype(int).tolist(),
            "eth":      grp["ethnicity"].iloc[0],
        }
    print(f"Sequences built for {len(sequences)} patients")

    # Run model and produce predictions
    ctx     = (torch.autocast("cuda", torch.float16)
               if "cuda" in device else nullcontext())
    pids    = list(sequences.keys())
    records = []

    with torch.no_grad():
        for i in range(0, len(pids), batch_size):
            batch    = pids[i:i + batch_size]
            age_seqs = [sequences[p]["age_toks"][-block_size:] for p in batch]
            dis_seqs = [sequences[p]["dis_toks"][-block_size:] for p in batch]
            max_len  = max(len(s) for s in dis_seqs)

            age_pad = torch.tensor(
                [[0]*(max_len-len(s))+s for s in age_seqs],
                dtype=torch.long, device=device)
            dis_pad = torch.tensor(
                [[0]*(max_len-len(s))+s for s in dis_seqs],
                dtype=torch.long, device=device)

            with ctx:
                logits, _, _ = model(dis_pad, age_pad)

            for j, pid in enumerate(batch):
                last     = len(dis_seqs[j]) - 1
                lh       = logits[j, last, :vocab_size].float().cpu().numpy()
                probs    = torch.softmax(torch.tensor(lh), 0).numpy()
                eth      = sequences[pid]["eth"]
                for tok, (l, pr) in enumerate(zip(lh, probs)):
                    records.append({"patient_id": pid, "ethnicity": eth,
                                    "token": tok, "log_hazard": float(l),
                                    "hazard_prob": float(pr)})

            if i % (batch_size * 10) == 0:
                print(f"  {min(i+batch_size, len(pids))}/{len(pids)} patients")

    preds = pd.DataFrame(records)
    preds["patient_id"] = preds["patient_id"].astype(str)
    preds["token"]      = preds["token"].astype(int)

    # Build full outcomes (event=0 and event=1) 
    event1_set    = set(zip(outcomes_e1["patient_id"], outcomes_e1["token"]))
    full_outcomes = preds[["patient_id","token"]].copy()
    full_outcomes["event"] = full_outcomes.apply(
        lambda r: 1 if (r.patient_id, r.token) in event1_set else 0, axis=1)

    # Diagnostic
    print(f"\n  outcomes_e1 shape: {outcomes_e1.shape}")
    print(f"  Sample outcomes_e1 tokens: {outcomes_e1['token'].head().tolist()}")
    print(f"  Sample preds tokens:       {preds['token'].head().tolist()}")
    print(f"  event=1 labels created: {full_outcomes['event'].sum()}")
    print(f"  event=0 labels created: {(full_outcomes['event']==0).sum()}")

    preds.to_parquet(pred_path, index=False)
    full_outcomes.to_csv(outcomes_path, index=False)
    print(f"Saved predictions {preds.shape} | "
          f"positive rate {full_outcomes['event'].mean():.5f}")
    return preds, full_outcomes


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--linked-data",    required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--predictions",    default="predictions.parquet")
    p.add_argument("--outcomes",       default="outcomes.csv")
    p.add_argument("--report",         default="masking_report.csv")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--dtype",          default="float32")
    p.add_argument("--seed",           type=int, default=1337)
    p.add_argument("--batch-size",     type=int, default=32)
    a = p.parse_args()
    run(a.linked_data, a.checkpoint_dir, a.predictions, a.outcomes,
        a.report, a.device, a.dtype, a.seed, a.batch_size)