import io
import json
import os

import joblib
import pandas as pd

FEATURE_COLUMNS = None


def model_fn(model_dir):
    """Load the sklearn Pipeline and feature column metadata."""
    global FEATURE_COLUMNS
    with open(os.path.join(model_dir, "feature_columns.json")) as f:
        FEATURE_COLUMNS = json.load(f)
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type):
    """Parse CSV payload into a DataFrame with named columns."""
    if request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        df.columns = FEATURE_COLUMNS
        return df
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Return positive-class probability for each row."""
    return model.predict_proba(input_data)[:, 1]


def output_fn(prediction, accept):
    """Serialize predictions as CSV."""
    return ",".join(str(p) for p in prediction), "text/csv"
