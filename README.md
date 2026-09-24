# Bias Detection and Mitigation in Predictive Modelling for Healthcare

An agentic pipeline for detecting and mitigating demographic bias in **Delphi-2M**, a GPT-2-based multi-disease prediction model trained on UK Biobank data. The pipeline audits model fairness across ethnic groups and applies targeted post-processing corrections where disparities are found.

## Setup

This project builds on the original Delphi-2M model. Before running this pipeline, follow the [Delphi-2M setup and training instructions](https://github.com/gerstung-lab/delphi) in the Gerstung Lab repository to train the model on the UK Biobank simulated data (`train.bin`). The trained model checkpoint is required before proceeding.

**Train and test dataset** are also required, both following the format described in the original Delphi-2M paper — each record should include: `subject_id`, `age`, and `token` (ICD-coded disease outcome). For the bias audit, a seperate **ethnicity dataset** mapping `subject_id` to ethnic group is required. 

## Pipeline

```
Step 1  Data Preparation     — Preprocess Synthea™ synthetic patient records
Step 2  Delphi-2M Inference  — Generate disease trajectory predictions 
Step 3  Bias Audit           — Compute demographic parity & equalised odds per token
Step 4  Threshold Correction — Apply ThresholdOptimizer (fairlearn) to flagged tokens only
Step 5  Re-evaluation        — Re-audit; repeat while flagged tokens remain
```

## Output

The pipeline produces per-token fairness summaries, before/ after demographic parity comparisons, and AUC plots stratified by ethnicity. Both are saved to the output/ directory.

## Stack

Python · PyTorch · fairlearn · pandas · Synthea™

## Running

```bash
python evaluate_delphi.py   # baseline evaluation
python run.py               # full pipeline
```
