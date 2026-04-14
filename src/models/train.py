# Working with data
import pandas as pd
import numpy as np

# src files
from data.load import load_data
from data.config import preprocess

# split train and test
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# encode
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, accuracy_score, recall_score

# time
import time

def main():

    # load data
    df = load_data("Churn_Modelling.xls")

    # drop columns
    cols = ["CustomerId", "Surname"]
    df = preprocess(df, cols=cols)

    # Spliting target column
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23, stratify=y)

    # Spliting num and cat cols
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(include="object").columns

    # pipelin for numeric cols
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # pipeline for cat cols
    cat_pipeline = Pipeline([
        ("encode", OneHotEncoder())
    ])

    # Transform
    col_trans = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    # List of models
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "K-nn": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression()
    }
    print(models.items())

    # train models
    results = []

    for name, model in models.items():

        train_model = Pipeline([
            ("transfrom", col_trans),
            ("model", model)
        ])

        #train model
        start_tr = time.time()
        train_model.fit(X_train, y_train)
        end_tr = time.time()

        # predict
        start_pr = time.time()
        y_pre = train_model.predict(X_test)
        end_pr = time.time()

        accuracy = accuracy_score(y_test, y_pre)
        f1 = f1_score(y_test, y_pre)
        recll = recall_score(y_test, y_pre)
        train_time = start_tr - end_tr
        predict_time = start_pr - end_pr 

        # test
        results.append({
            "name": name,
            "f1-score": f1,
            "accuracy": accuracy,
            "recall": recll,
            "train_time": train_time,
            "predict_time": predict_time
        })

    # result to df

    result_df = pd.DataFrame(results)
    print(result_df)
        

if __name__ == "__main__":
    main()