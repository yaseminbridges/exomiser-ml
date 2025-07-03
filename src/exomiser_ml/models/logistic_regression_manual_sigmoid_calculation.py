from pathlib import Path
from typing import List
import polars as pl
import numpy as np
from pheval.utils.file_utils import all_files

from exomiser_ml.post_process.post_process import post_process_test_dir
from exomiser_ml.utils.write_metadata import RunMetadata, write_metadata_yaml


def manual_predict_proba(X: np.array, coefficients: np.array, intercept: float):
    """
    X: 2D array of shape (n_samples, n_features)
    coefficients: 1D array of shape (n_features,)
    intercept: scalar
    Returns: array of predicted probabilities
    """
    z = np.dot(X, coefficients) + intercept
    return 1 / (1 + np.exp(-z))  # Apply sigmoid


def manual_predict_proba_on_test_dir(test_dir: Path, features: List[str], coefficients: List[float], intercept: float,
                              output_dir: Path) -> None:
    coefficients = np.array(coefficients)
    for test_file in all_files(test_dir):
        df = pl.read_csv(test_file, separator="\t", infer_schema_length=0)
        extracted_features = df.select(features).to_numpy()
        probabilities = manual_predict_proba(extracted_features, coefficients, intercept)
        new_scores = pl.DataFrame({"NEW_SCORE": probabilities})
        if "NEW_SCORE" in df.columns:
            print(f"Warning: 'NEW_SCORE' already exists in {test_file}. Replacing it.")
            df = df.drop("NEW_SCORE")
        df_with_new_scores = df.hstack(new_scores)
        df_with_new_scores.write_csv(output_dir.joinpath(test_file.name), separator="\t")


def run_manual_logistic_regression_model(test_dir: Path, features: List[str], coefficients: List[float],
                                         intercept: float, output_dir: Path, phenopacket_dir: Path):
    raw_results_dir = output_dir.joinpath("raw_results")
    manual_predict_proba_on_test_dir(test_dir, features, coefficients, intercept, raw_results_dir)
    post_process_test_dir(test_dir=raw_results_dir, phenopacket_dir=phenopacket_dir, output_dir=output_dir)
    metadata = RunMetadata(
        test_size=None,
        output_dir=str(output_dir),
        model_type="LOGISTIC REGRESSION",
        features_used=list(features),
        training_data="N/A",
        test_dir=str(test_dir)
    )
    write_metadata_yaml(metadata, output_dir)
