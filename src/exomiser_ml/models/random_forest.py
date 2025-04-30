import shutil
from pathlib import Path
from typing import List

import click
import polars as pl
from pheval.utils.file_utils import all_files
from sklearn.ensemble import RandomForestClassifier

from exomiser_ml.data.create_features.add_features import add_features
from exomiser_ml.data.split_data.split_train_and_test import split_train_and_test
from exomiser_ml.post_process.post_process import post_process_test_dir
from exomiser_ml.utils.write_metadata import RunMetadata, write_metadata_yaml


def train_model(training_data: Path, features: List[str]) -> RandomForestClassifier:
    training_data_df = pl.read_csv(training_data, separator="\t", infer_schema_length=0)
    X = training_data_df.select(features)
    y = training_data_df.select(["CAUSATIVE_VARIANT"])
    model = RandomForestClassifier(random_state=42)
    model.fit(X.to_pandas(), y.to_pandas().squeeze())
    return model


def test_model(test_dir: Path, model: RandomForestClassifier, features: List[str], output_dir: Path) -> None:
    for test_file in all_files(test_dir):
        df = pl.read_csv(test_file, separator="\t", infer_schema_length=0)
        extracted_features = df.select(features)
        new_scores = pl.DataFrame({"NEW_SCORE": model.predict_proba(extracted_features.to_pandas())[:, 1]})
        if "NEW_SCORE" in df.columns:
            print(f"Warning: 'NEW_SCORE' already exists in {test_file}. Replacing it.")
            df = df.drop("NEW_SCORE")
        df_with_new_scores = df.hstack(new_scores)
        df_with_new_scores.write_csv(output_dir.joinpath(test_file.name), separator="\t")


def random_forest(training_data: Path, test_dir: Path, features: List[str], output_dir: Path) -> None:
    model = train_model(training_data, features)
    test_model(test_dir, model, features, output_dir)


def run_random_forest_pipeline(phenopacket_dir: Path, result_dir: Path, output_dir: Path, features: List[str],
                                     test_size: float):
    output_dir.joinpath("added_features").mkdir(parents=True, exist_ok=True)
    add_features(phenopacket_dir=phenopacket_dir, result_dir=result_dir,
                 output_dir=output_dir.joinpath("added_features"))
    output_dir.joinpath("results_split/train").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("results_split/test").mkdir(parents=True, exist_ok=True)
    split_train_and_test(input_dir=output_dir.joinpath("added_features"),
                         output_dir=output_dir.joinpath("results_split"), test_size=test_size)
    random_forest(training_data=output_dir.joinpath("results_split/train/train.tsv"),
                        test_dir=output_dir.joinpath("results_split/test"), features=features, output_dir=output_dir.joinpath("raw_results"))
    output_dir.joinpath("pheval_variant_results").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("raw_results").mkdir(parents=True, exist_ok=True)
    post_process_test_dir(test_dir=output_dir.joinpath("raw_results"), phenopacket_dir=phenopacket_dir,
                          output_dir=output_dir)
    shutil.rmtree(output_dir.joinpath("added_features"))
    metadata = RunMetadata(
        test_size=test_size,
        output_dir=str(output_dir),
        model_type="RandomForestClassifier",
        features_used=features,
        training_data=str(output_dir.joinpath("results_split/train/train.tsv")),
        test_dir=str(output_dir.joinpath("results_split/test"))
    )
    write_metadata_yaml(metadata, output_dir)


@click.command("run-rf")
@click.option('--training-data', "-t", type=Path, required=True, help="Path to the training data tsv.")
@click.option('--test-dir', "-e", type=Path, required=True,
              help="Path to the test data directory")
@click.option('--features', "-f", multiple=True, required=True, help='List of features to extract.')
@click.option('--output-dir', "-o", type=Path, required=True, help="Path to the output directory.")
@click.option('--phenopacket-dir', "-p", type=Path, required=True, help="Path to the Phenopacket data directory.")
def run_random_forest(training_data: Path, test_dir: Path, features: List[str], output_dir: Path,
                            phenopacket_dir: Path) -> None:
    output_dir.joinpath("pheval_variant_results").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("raw_results").mkdir(parents=True, exist_ok=True)
    random_forest(training_data, test_dir, list(features), output_dir.joinpath("raw_results"))
    post_process_test_dir(test_dir=output_dir.joinpath("raw_results"), phenopacket_dir=phenopacket_dir,
                          output_dir=output_dir)
    metadata = RunMetadata(
        test_size=None,
        output_dir=output_dir,
        model_type="RandomForestClassifier",
        features_used=list(features),
        training_data=training_data,
        test_dir=test_dir
    )
    write_metadata_yaml(metadata, output_dir)


@click.command("run-rf-pipeline")
@click.option('--phenopacket-dir', "-p", type=Path, required=True, help="Path to the Phenopacket data directory.")
@click.option('--result-dir', "-r", type=Path, required=True,
              help="Path to the results directory containing Exomiser variants .tsv files.")
@click.option('--output-dir', "-o", type=Path, required=True, help="Path to the output directory.")
@click.option('--features', "-f", multiple=True, required=True, help='List of features to extract.')
@click.option('--test-size', "-t", type=float, default=0.2, help='Proportion of data to use for testing.')
def run_rf_pipeline(phenopacket_dir: Path, result_dir: Path, output_dir: Path, features: List[str], test_size: float):
    run_random_forest_pipeline(
        phenopacket_dir=phenopacket_dir,
        result_dir=result_dir,
        output_dir=output_dir,
        features=list(features),
        test_size=test_size
    )
