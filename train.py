import argparse, os, joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


def main(args):
    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path   = os.path.join(args.data_dir, "val.parquet")

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    label_col = "churned_14d"
    drop_cols = ["user_id", "obs_end_date"]  # keep label out, drop IDs/dates
    feature_cols = [c for c in train_df.columns if c not in drop_cols + [label_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[label_col].astype(int)

    X_val = val_df[feature_cols]
    y_val = val_df[label_col].astype(int)

    os.makedirs(args.model_dir, exist_ok=True)

    # -----------------------
    # 1) XGBoost
    # -----------------------
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train, y_train)

    # -----------------------
    # 2) Logistic Regression (with scaling via Pipeline)
    # -----------------------
    logreg = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False is safe for sparse; ok here too
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    logreg.fit(X_train, y_train)

    # Save
    joblib.dump(xgb, os.path.join(args.model_dir, "churn_xgb.joblib"))
    joblib.dump(logreg, os.path.join(args.model_dir, "churn_logreg_pipeline.joblib"))
    joblib.dump(feature_cols, os.path.join(args.model_dir, "feature_columns.joblib"))

    print("✅ Models saved to:", args.model_dir)
    print("Saved files:")
    print("- churn_xgb.joblib")
    print("- churn_logreg_pipeline.joblib")
    print("- feature_columns.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder with train/val/test parquet")
    parser.add_argument("--model_dir", type=str, required=True, help="Where to save model artifacts")
    args = parser.parse_args()
    main(args)
