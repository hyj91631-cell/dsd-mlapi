import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--class-weight", type=str, default="balanced_subsample")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--label-column", type=str, required=True)
    parser.add_argument("--categorical-features", type=str, required=True)
    parser.add_argument("--numerical-features", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args = parser.parse_args()

    categorical_features = [f.strip() for f in args.categorical_features.split(",")]
    numerical_features = [f.strip() for f in args.numerical_features.split(",")]

    train_files = [os.path.join(args.train, f) for f in os.listdir(args.train) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

    y_train = df[args.label_column].values
    X_train = df.drop(columns=[args.label_column])

    print(f"Training on {len(df)} samples, {X_train.shape[1]} features.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("binary_classifier", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight=args.class_weight,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )),
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(args.model_dir, "model.joblib"))

    feature_columns = numerical_features + categorical_features
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)

    print(f"Model saved to {args.model_dir}")


if __name__ == "__main__":
    main()
