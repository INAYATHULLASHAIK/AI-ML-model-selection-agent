import pandas as pd
import numpy as np

def detect_problem_type(target):
    # Numeric targets with many distinct values are treated as regression.
    # Small-cardinality numeric targets are treated as classification.
    if pd.api.types.is_numeric_dtype(target):
        unique = target.nunique(dropna=True)
        if unique <= 10:
            return "Classification"
        return "Regression"
    return "Classification"

def analyze_dataset(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicates": int(df.duplicated().sum()),
        "preview": df.head(10).replace({np.nan: None}).to_dict(orient="records")
    }
