from pathlib import Path
import polars as pl
from pheval.utils.file_utils import all_files
from sklearn.linear_model import LogisticRegression


def train_model(training_data: Path) -> LogisticRegression:
    training_data_df = pl.read_csv(training_data, separator="\t", infer_schema_length=10000)
    X = training_data_df.drop(["CAUSATIVE_VARIANT"])
    y = training_data_df.select(["CAUSATIVE_VARIANT"])
    model = LogisticRegression(random_state=42)
    model.fit(X.to_pandas(), y.to_pandas().squeeze())
    return model


def test_model(test_dir: Path, model: LogisticRegression) -> None:
    for test_file in all_files(test_dir):
        df = pl.read_csv(test_file, separator="\t", infer_schema_length=100000)
        new_scores = pl.DataFrame({"NEW_SCORE_PPP": model.predict_proba(df)[:, 1]})
        df_with_new_scores = df.hstack(new_scores)
        df_with_new_scores.write_csv(test_file, separator="\t")


def logistic_regression(training_data: Path, test_dir: Path) -> None:
    model = train_model(training_data)
    test_model(test_dir, model)
