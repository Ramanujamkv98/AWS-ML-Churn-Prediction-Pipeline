import argparse
import os
import pandas as pd
import numpy as np
import s3fs


def read_s3_prefix(prefix: str) -> pd.DataFrame:
    """
    Reads all CSV, Parquet, or extensionless files under an S3 prefix
    and concatenates them.

    prefix example:
      s3://my-bucket/curated/user_features_v1/
    """
    fs = s3fs.S3FileSystem()

    # List all objects under prefix (recursive)
    files = fs.glob(prefix.rstrip("/") + "/**")

    # Remove folder markers / delimiter objects
    data_files = [
        f for f in files
        if not f.endswith("/") and not f.endswith(";")
    ]

    # Detect by extension (if present)
    csv_files = [f"s3://{f}" for f in data_files if f.lower().endswith(".csv")]
    pq_files = [f"s3://{f}" for f in data_files if f.lower().endswith(".parquet")]

    # 1) Parquet files
    if pq_files:
        dfs = [pd.read_parquet(f) for f in pq_files]
        return pd.concat(dfs, ignore_index=True)

    # 2) CSV files
    if csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        return pd.concat(dfs, ignore_index=True)

    # 3) Extensionless files: try Parquet first, then CSV
    noext_files = [
        f"s3://{f}" for f in data_files
        if "." not in os.path.basename(f)
    ]
    if noext_files:
        dfs = []
        for f in noext_files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                dfs.append(pd.read_csv(f))
        return pd.concat(dfs, ignore_index=True)

    raise ValueError(f"No readable data files found under prefix: {prefix}")


def load_input(input_path: str) -> pd.DataFrame:
    """
    Loads input from:
      - S3 prefix ending with '/' (reads all files)
      - Single file path (tries Parquet first, then CSV)
    """
    # Prefix case
    if input_path.startswith("s3://") and input_path.endswith("/"):
        return read_s3_prefix(input_path)

    # Single file case: try PARQUET first (handles extensionless parquet)
    parquet_err = None
    try:
        return pd.read_parquet(input_path)
    except Exception as e:
        parquet_err = e

    # Fallback: try CSV
    csv_err = None
    try:
        return pd.read_csv(input_path)
    except Exception as e:
        csv_err = e

    raise ValueError(
        f"Failed to read input as Parquet or CSV: {input_path}\n"
        f"Parquet error: {parquet_err}\n"
        f"CSV error: {csv_err}"
    )


def main(args):
    input_path = args.input_path
    output_path = args.output_path

    # ----------------------------
    # 1) Load data
    # ----------------------------
    df = load_input(input_path)

    # ----------------------------
    # 2) Basic validation
    # ----------------------------
    required_cols = [
        "user_id",
        "obs_end_date",
        "churned_14d",
        "tokens_per_session_7d",
        "primary_model_7d",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure date type
    df["obs_end_date"] = pd.to_datetime(df["obs_end_date"], errors="coerce")
    if df["obs_end_date"].isna().any():
        bad_rows = int(df["obs_end_date"].isna().sum())
        raise ValueError(f"obs_end_date has {bad_rows} unparsable values.")

    # Ensure label is numeric 0/1
    df["churned_14d"] = pd.to_numeric(df["churned_14d"], errors="coerce").fillna(0).astype(int)

    # ----------------------------
    # 3) Drop leakage columns
    # ----------------------------
    leakage_cols = ["unsubscribe_ts"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # ----------------------------
    # 4) Feature engineering
    # ----------------------------
    df["tokens_per_session_7d"] = pd.to_numeric(df["tokens_per_session_7d"], errors="coerce").fillna(0)
    df["activated_800"] = (df["tokens_per_session_7d"] >= 800).astype(int)

    df["obs_month"] = df["obs_end_date"].dt.month
    df["obs_dow"] = df["obs_end_date"].dt.dayofweek

    # ----------------------------
    # 5) Handle missing values
    # ----------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    if df["primary_model_7d"].isna().any():
        df["primary_model_7d"] = df["primary_model_7d"].fillna("unknown")
    df["primary_model_7d"] = df["primary_model_7d"].astype(str)

    # ----------------------------
    # 6) One-hot encode categorical variables
    # ----------------------------
    df = pd.get_dummies(df, columns=["primary_model_7d"], drop_first=True)

    # ----------------------------
    # 7) Time-based split by obs_end_date
    # ----------------------------
    df = df.sort_values("obs_end_date").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # ----------------------------
    # 8) Save outputs (local path)
    # ----------------------------
    os.makedirs(output_path, exist_ok=True)

    train_df.to_parquet(os.path.join(output_path, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_path, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_path, "test.parquet"), index=False)

    print("✅ Processing complete")
    print(f"Input rows: {n}")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="S3 prefix ending with / or a single CSV/Parquet file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Local output directory (e.g., /tmp/churn_processed)",
    )
    args = parser.parse_args()

    main(args)
