"""Split the synthetic manifest into train/val/test."""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main():
    manifest_path = PROJECT_ROOT / "data" / "raw" / "synthetic" / "manifest.csv"
    splits_dir = PROJECT_ROOT / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} rows")
    print(df["label"].value_counts())

    unique_docs = np.asarray(df["doc_id"].unique())
    train_docs, temp_docs = train_test_split(unique_docs, test_size=0.30, random_state=42)
    val_docs, test_docs = train_test_split(temp_docs, test_size=0.50, random_state=42)

    def assign(doc):
        if doc in set(train_docs): return "train"
        if doc in set(val_docs): return "val"
        return "test"

    df["split"] = df["doc_id"].apply(assign)

    # Save split manifests
    for split in ["train", "val", "test"]:
        sub = df[df.split == split].reset_index(drop=True)
        sub.to_csv(splits_dir / f"{split}.csv", index=False)
        print(f"\n{split}: {len(sub)}")
        print(sub["label"].value_counts())

    print(f"\nSplits saved to {splits_dir}")


if __name__ == "__main__":
    main()