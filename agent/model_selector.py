import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)

RANDOM_STATE = 42

def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers)

def classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB()
    }

def regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE),
        "KNN Regressor": KNeighborsRegressor(),
        "Support Vector Regressor": SVR(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Ridge Regression": Ridge()
    }

def run_model_selection(df, target_column):
    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    df = df.dropna(subset=[target_column]).copy()
    if len(df) < 10:
        raise ValueError("Not enough usable rows after removing missing target values.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = "Classification" if (
        not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10
    ) else "Regression"

    if X.shape[1] == 0:
        raise ValueError("The dataset needs at least one feature column.")

    if problem_type == "Classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two classes.")

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y.astype(str))

        counts = pd.Series(y_encoded).value_counts()
        stratify = y_encoded if counts.min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=RANDOM_STATE,
            stratify=stratify
        )

        preprocessor = make_preprocessor(X)
        results = []
        best = None

        for name, model in classification_models().items():
            try:
                # GaussianNB cannot consume sparse matrices.
                if name == "Naive Bayes":
                    from sklearn.preprocessing import FunctionTransformer
                    from scipy import sparse
                    pre = make_preprocessor(X)
                    pipe = Pipeline([
                        ("preprocessor", pre),
                        ("to_dense", FunctionTransformer(
                            lambda x: x.toarray() if sparse.issparse(x) else x
                        )),
                        ("model", model)
                    ])
                else:
                    pipe = Pipeline([
                        ("preprocessor", preprocessor),
                        ("model", model)
                    ])

                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)

                row = {
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, pred), 4),
                    "Precision": round(precision_score(y_test, pred, average="weighted", zero_division=0), 4),
                    "Recall": round(recall_score(y_test, pred, average="weighted", zero_division=0), 4),
                    "F1 Score": round(f1_score(y_test, pred, average="weighted", zero_division=0), 4)
                }
                results.append(row)

                score = row["F1 Score"]
                if best is None or score > best["score"]:
                    best = {"name": name, "score": score, "model": pipe, "pred": pred}

            except Exception as exc:
                results.append({"Model": name, "Error": str(exc)})

        if not best:
            raise ValueError("None of the classification models could be trained.")

        labels = encoder.classes_.tolist()
        cm = confusion_matrix(y_test, best["pred"], labels=np.arange(len(labels))).tolist()

        return {
            "problem_type": problem_type,
            "target_column": target_column,
            "best_model": best["name"],
            "best_score": best["score"],
            "reason": "It achieved the highest weighted F1 score among the successfully evaluated models.",
            "results": results,
            "chart_labels": [r["Model"] for r in results if "F1 Score" in r],
            "chart_primary": [r["F1 Score"] for r in results if "F1 Score" in r],
            "chart_secondary": [r["Accuracy"] for r in results if "Accuracy" in r],
            "confusion_matrix": cm,
            "confusion_labels": labels
        }

    # Regression
    if y.nunique() < 2:
        raise ValueError("Regression requires variation in the target values.")

    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = make_preprocessor(X)
    results = []
    best = None

    for name, model in regression_models().items():
        try:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            row = {
                "Model": name,
                "MAE": round(mean_absolute_error(y_test, pred), 4),
                "MSE": round(mean_squared_error(y_test, pred), 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2_score(y_test, pred), 4)
            }
            results.append(row)

            score = row["R2"]
            if best is None or score > best["score"]:
                best = {"name": name, "score": score, "pred": pred}

        except Exception as exc:
            results.append({"Model": name, "Error": str(exc)})

    if not best:
        raise ValueError("None of the regression models could be trained.")

    actual = y_test.tolist()
    predicted = np.asarray(best["pred"]).tolist()

    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "best_model": best["name"],
        "best_score": best["score"],
        "reason": "It achieved the highest R² score among the successfully evaluated models.",
        "results": results,
        "chart_labels": [r["Model"] for r in results if "R2" in r],
        "chart_primary": [r["R2"] for r in results if "R2" in r],
        "chart_secondary": [r["RMSE"] for r in results if "RMSE" in r],
        "actual": actual,
        "predicted": predicted
    }
