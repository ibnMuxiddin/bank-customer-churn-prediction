# working with data
import pandas as pd
import numpy as np

# src files
from data.load import load_data
from data.config import drop_cols
from features.add_attr import FeatureGenerator

# train test split
from sklearn.model_selection import train_test_split, GridSearchCV

# pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# transformers
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

# model
from sklearn.svm import SVC

def main():

    # loading data
    df = load_data("Churn_Modelling.xls")

    # drop columns
    df = drop_cols(df)

    # seprite target columns
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)

    # Log Transform
    log_transformer = FunctionTransformer(np.log1p)

    # log pipline
    log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", log_transformer),
        ("scaler", StandardScaler())
    ])

    # num pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Transfom
    preprocesser = ColumnTransformer(
        transformers=[
            ("log_col", log_pipeline, ["Age"]),
            ("num", num_pipeline, ["Age", "CreditScore", "Balance", "EstimatedSalary"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include=["object", "category"])),
        ],
        remainder="passthrough"
    )

    # full pipeline
    pipeline = ImbPipeline(
        steps = [
            ("add_attr", FeatureGenerator()),
            ("preprocess", preprocesser),
            ("smote", SMOTE(random_state=42)),
            ("model", SVC())
        ]
    )

    # Logistic Regression param grid
    param_grid = [
        {
            "model__penalty": ["l2"],
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs"],
            "model__max_iter": [3000]
        },
        {
            "model__penalty": ["l1", "l2"],
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["saga"],
            "model__max_iter": [3000]
        }
        ]
    
    param_grid_svm = [
    {
        "model__kernel": ["linear"],
        "model__C": [0.01, 0.1, 1, 10]
    },
    {
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", "auto", 0.01, 0.1]
    },
    {
        "model__kernel": ["poly"],
        "model__C": [0.1, 1],
        "model__degree": [2, 3],
        "model__gamma": ["scale", "auto"]
    }
    ]

    # GridSearchCV
    grid_searchcv = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid_svm,
        scoring="recall",
        cv=5,
        n_jobs=-1
    )

    # train model
    grid_searchcv.fit(X_train, y_train)

    print(f"Eng yaxshi parametrlar: {grid_searchcv.best_params_}")
    print(f"Eng yaxshi Recall: {grid_searchcv.best_score_}")

if __name__ == "__main__":
    main()