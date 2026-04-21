"""
Step 1.1
========================================
Loads token to ICD mapping CSV and makes it available to all downstream steps. 


Input  : mapping CSV   [token, icd_code]
Output : label_map.csv (saved to the output directory)
"""

import pandas as pd

# Read CSV with automatic encoding detection.
def _read_csv(path, **kwargs):
    kwargs.pop("encoding", None)
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252",
                "iso-8859-1", "mac_roman", "cp1250", "utf-16"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise e  
    raise ValueError(
        f"Could not read {path} with any known encoding.\n"
        f"To detect the correct encoding, run:\n"
        f"  pip install chardet\n"
        f"  python -c \"import chardet; "
        f"print(chardet.detect(open('{path}','rb').read()))\"")



def load_label_map(label_map_path):
    df = _read_csv(label_map_path)

    if "token" not in df.columns or "icd_code" not in df.columns:
        raise ValueError(
            f"Label map must have columns 'token' and 'icd_code'. "
            f"Found: {list(df.columns)}"
        )

    df["token"] = df["token"].astype(int)
    print(f"Label map loaded: {len(df)} tokens | "
          f"example: token={df['token'].iloc[0]} → {df['icd_code'].iloc[0]}")
    return df[["token", "icd_code"]]

### Join icd_code onto any DataFrame that has a 'token' column.
# If icd_code already exists (e.g. from a previous labelling pass), it is dropped first to avoid duplicate column errors.
def apply_labels(df, label_map):
    if "token" not in df.columns:
        raise ValueError("Cannot apply labels: DataFrame has no 'token' column.")
    if "icd_code" in df.columns:
        df = df.drop(columns=["icd_code"])
    return df.merge(label_map[["token", "icd_code"]], on="token", how="left")


def run(label_map_path, output_path="label_map.csv"):
    label_map = load_label_map(label_map_path)
    label_map.to_csv(output_path, index=False)
    print(f"Label map saved to: {output_path}")
    return label_map


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Step 0: Load token-to-ICD label map")
    p.add_argument("--label-map", required=True,
                   help="Path to your token→ICD mapping CSV")
    p.add_argument("--output",    default="label_map.csv")
    a = p.parse_args()
    run(a.label_map, a.output)