import shutil
from pathlib import Path
from typing import List, Type

import joblib
import polars as pl
from sklearn.base import ClassifierMixin
from pheval.utils.file_utils import all_files
from xgboost import XGBClassifier

from exomiser_ml.data.create_features.add_features import add_features
from exomiser_ml.data.split_data.split_train_and_test import split_train_and_test
from exomiser_ml.post_process.post_process import post_process_test_dir
from exomiser_ml.utils.write_metadata import RunMetadata, write_metadata_yaml
from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class ModelType(Enum):
    LOGISTIC_REGRESSION = LogisticRegression
    RANDOM_FOREST = RandomForestClassifier
    XGBOOST_CLASSIFIER = XGBClassifier

def save_model(model: Type[ClassifierMixin], model_path: Path):
    if isinstance(model, XGBClassifier):
        model.save_model(str(model_path.with_suffix(".json")))
    else:
        joblib.dump(model, model_path.with_suffix(".pkl"))

def train_model(
        training_data: Path,
        features: List[str],
        model_cls: Type[ClassifierMixin],
        model_path: Path) -> ClassifierMixin:
    training_data_df = pl.read_csv(training_data, separator="\t", infer_schema_length=0)
    X = training_data_df.select(features)
    y = training_data_df.select(["CAUSATIVE_VARIANT"])
    model = model_cls(random_state=42)
    model.fit(X.to_pandas(), y.to_pandas().squeeze())
    save_model(model, model_path)
    return model


def test_model(test_dir: Path, model: ClassifierMixin, features: List[str], output_dir: Path) -> None:
    for test_file in all_files(test_dir):
        df = pl.read_csv(test_file, separator="\t", infer_schema_length=0)
        extracted_features = df.select(features)
        new_scores = pl.DataFrame({"NEW_SCORE": model.predict_proba(extracted_features.to_pandas())[:, 1]})
        if "NEW_SCORE" in df.columns:
            print(f"Warning: 'NEW_SCORE' already exists in {test_file}. Replacing it.")
            df = df.drop("NEW_SCORE")
        df_with_new_scores = df.hstack(new_scores)
        df_with_new_scores.write_csv(output_dir.joinpath(test_file.name), separator="\t")


def run_model(training_data: Path, test_dir: Path, features: List[str], output_dir: Path, phenopacket_dir: Path,
              model: str):
    raw_results_dir = output_dir.joinpath("raw_results")
    pheval_results_dir = output_dir.joinpath("pheval_variant_results")
    model_dir = output_dir.joinpath("model")
    model_cls = ModelType[model.upper()].value
    for dir_path in [raw_results_dir, pheval_results_dir, model_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    model_instance = train_model(training_data, list(features), model_cls, output_dir.joinpath(f"model/{model}"))
    test_model(test_dir, model_instance, list(features), raw_results_dir)
    post_process_test_dir(test_dir=raw_results_dir, phenopacket_dir=phenopacket_dir, output_dir=output_dir)
    metadata = RunMetadata(
        test_size=None,
        output_dir=str(output_dir),
        model_type=model,
        features_used=list(features),
        training_data=str(training_data),
        test_dir=str(test_dir)
    )
    write_metadata_yaml(metadata, output_dir)


def run_pipeline(
        phenopacket_dir: Path,
        result_dir: Path,
        output_dir: Path,
        features: List[str],
        test_size: float,
        model: str,
        filter_clinvar: bool,
):
    model_cls = ModelType[model.upper()].value
    added_features_dir = output_dir.joinpath("added_features")
    train_dir = output_dir.joinpath("results_split/train")
    test_dir = output_dir.joinpath("results_split/test")
    raw_results_dir = output_dir.joinpath("raw_results")
    pheval_results_dir = output_dir.joinpath("pheval_variant_results")
    model_dir = output_dir.joinpath("model")

    for dir_path in [added_features_dir, train_dir, test_dir, raw_results_dir, pheval_results_dir, model_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    add_features(phenopacket_dir=phenopacket_dir, result_dir=result_dir, output_dir=added_features_dir, filter_clinvar=filter_clinvar)
    split_train_and_test(input_dir=added_features_dir, output_dir=output_dir.joinpath("results_split"),
                         test_size=test_size)
    trained_model = train_model(train_dir.joinpath("train.tsv"), features, model_cls, output_dir.joinpath(f"model/{model}"))
    test_model(test_dir, trained_model, features, raw_results_dir)
    post_process_test_dir(test_dir=raw_results_dir, phenopacket_dir=phenopacket_dir, output_dir=output_dir)
    shutil.rmtree(added_features_dir)
    metadata = RunMetadata(
        test_size=test_size,
        output_dir=str(output_dir),
        model_type=model,
        features_used=features,
        training_data=str(train_dir / "train.tsv"),
        test_dir=str(test_dir)
    )
    write_metadata_yaml(metadata, output_dir)
