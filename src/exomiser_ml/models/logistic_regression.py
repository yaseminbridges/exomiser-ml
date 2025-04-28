from pathlib import Path
from typing import List

import polars as pl
from pheval.utils.file_utils import all_files
from sklearn.linear_model import LogisticRegression


def train_model(training_data: Path, features: List[str]) -> LogisticRegression:
    training_data_df = pl.read_csv(training_data, separator="\t", infer_schema_length=10000)
    X = training_data_df.select(features)
    y = training_data_df.select(["CAUSATIVE_VARIANT"])
    model = LogisticRegression(random_state=42)
    model.fit(X.to_pandas(), y.to_pandas().squeeze())
    return model


def test_model(test_dir: Path, model: LogisticRegression, features: List[str]) -> None:
    for test_file in all_files(test_dir):
        df = pl.read_csv(test_file, separator="\t", infer_schema_length=100000)
        extracted_features = df.select(features)
        new_scores = pl.DataFrame({"NEW_SCORE": model.predict_proba(extracted_features)[:, 1]})
        df_with_new_scores = df.hstack(new_scores)
        df_with_new_scores.write_csv(test_file, separator="\t")


def logistic_regression(training_data: Path, test_dir: Path, features: List[str]) -> None:
    model = train_model(training_data, features)
    test_model(test_dir, model, features)