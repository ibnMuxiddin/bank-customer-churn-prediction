# Working with data
import pandas as pd
import numpy as np

# src files
from data.load import load_data
from data.config import drop_cols
from features.add_attr import FeatureGenerator

# split train and test
from sklearn.model_selection import train_test_split

# Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# encode
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer    
from sklearn.impute import SimpleImputer

# models 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score

# time
import time

def main():

    # load data
    df = load_data("Churn_Modelling.xls")
    df = drop_cols(df)

    # Spliting target column
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)
    
    # Log Transformation
    log_transformer = FunctionTransformer(np.log1p)

    # Transform
    col_trans = ColumnTransformer(
        transformers=[
            ("log", log_transformer, ["Age"]),
            ("scale", StandardScaler(), ["Age", "CreditScore", "Balance", "EstimatedSalary"]),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False), make_column_selector(dtype_include=["object", "category"])),
            ("impute", SimpleImputer(strategy='median'), make_column_selector(dtype_exclude=["object", "category"]))
        ],
        remainder="passthrough"
    )

    # List of models
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(),
        "SVM": SVC(probability=True)

    }

    # train models
    results = []

    for name, model in models.items():

        train_model = ImbPipeline(
            steps = [
                ("add_attr", FeatureGenerator()),
                ("preprocessor", col_trans),
                ("smote", SMOTE(random_state=42)),
                ("model", model)
            ]
        )

        #train model
        start_tr = time.time()
        train_model.fit(X_train, y_train)
        end_tr = time.time()

        # predict
        start_pr = time.time()
        y_pre = train_model.predict(X_test)
        end_pr = time.time()

        y_probs = train_model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pre
        train_time = end_tr-start_tr
        predict_time = end_pr - start_pr 

        # Metrics
        results.append({
            "name": name,
            "accuracy": accuracy_score(y_test, y_pre),
            "recall": recall_score(y_test, y_pre),
            "f1-score": f1_score(y_test, y_pre),
            "roc_auc": roc_auc_score(y_test, y_probs), # Probability ishlatildi
            "train_time": train_time,
            "predict_time": predict_time
        })

    # result to df

    result_df = pd.DataFrame(results)
    print(result_df)
        

if __name__ == "__main__":
    main()