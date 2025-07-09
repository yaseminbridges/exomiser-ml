from pathlib import Path
from typing import List

import click

from exomiser_ml.data.create_features.add_features import add_features
from exomiser_ml.data.split_data.split_train_and_test import split_train_and_test
from exomiser_ml.models.classifier_models import run_model, run_pipeline
from exomiser_ml.models.logistic_regression_manual_sigmoid_calculation import run_manual_logistic_regression_model
from exomiser_ml.post_process.post_process import post_process_test_dir

training_data_option = click.option(
    '--training-data', '-t', type=Path, required=True,
    help="Path to the training data tsv."
)

test_dir_option = click.option(
    '--test-dir', '-e', type=Path, required=True,
    help="Path to the test data directory."
)

features_option = click.option(
    '--features', '-f', multiple=True, required=True,
    help="List of features to extract."
)

output_dir_option = click.option(
    '--output-dir', '-o', type=Path, required=True,
    help="Path to the output directory."
)

phenopacket_dir_option = click.option(
    '--phenopacket-dir', '-p', type=Path, required=True,
    help="Path to the Phenopacket data directory."
)

result_dir_option = click.option(
    '--result-dir', '-r', type=Path, required=True,
    help="Path to the result directory."
)

input_dir_option = click.option(
    '--input-dir', '-i', type=Path, required=True,
    help="Path to the input data directory."
)

test_size_option = click.option(
    '--test-size', '-t', type=float, default=0.2,
    help='Proportion of data to use for testing.'
)

coefficients_option = click.option(
    '--coefficients', '-c', multiple=True, required=True,
    help="Coefficients of the logistic regression model."
)

model_option = click.option(
    '--model', '-m',
    type=click.Choice(["LOGISTIC_REGRESSION", "RANDOM_FOREST", "XGBOOST_CLASSIFIER"]),
    required=True,
    help="Model to run."
)

score_option = click.option(
    '--score', '-s', type=str, default="NEW_SCORE",
    help='Score to rank.'
)

intercept_option = click.option(
    '--intercept', '-y', type=float,
    help='Intercept of the Logistic Regression model.'
)

filter_clinvar_option = click.option(
    "--filter-clinvar/--no-filter-clinvar",
    default=True,
    help="Enable or disable ClinVar evidence filtering (default: enabled)"
)


@click.command("run-model")
@training_data_option
@test_dir_option
@features_option
@output_dir_option
@phenopacket_dir_option
@model_option
def run_model_command(training_data: Path, test_dir: Path, features: List[str], output_dir: Path, phenopacket_dir: Path,
                      model: str):
    run_model(training_data=training_data, test_dir=test_dir, features=features, output_dir=output_dir,
              phenopacket_dir=phenopacket_dir, model=model)


@click.command("run-pipeline")
@phenopacket_dir_option
@result_dir_option
@output_dir_option
@features_option
@test_size_option
@model_option
@filter_clinvar_option
def run_model_pipeline(
        phenopacket_dir: Path,
        result_dir: Path,
        output_dir: Path,
        features: List[str],
        test_size: float,
        model: str,
        filter_clinvar: bool,
):
    run_pipeline(phenopacket_dir=phenopacket_dir,
                 result_dir=result_dir,
                 output_dir=output_dir,
                 features=features,
                 test_size=test_size,
                 model=model,
                 filter_clinvar=filter_clinvar)


@click.command("add-features")
@phenopacket_dir_option
@result_dir_option
@output_dir_option
@filter_clinvar_option
def add_features_command(phenopacket_dir: Path, result_dir: Path, output_dir: Path, filter_clinvar: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_features(phenopacket_dir, result_dir, output_dir, filter_clinvar)


@click.command("split-data")
@input_dir_option
@test_size_option
@output_dir_option
def split_data_command(input_dir: Path, test_size: float, output_dir: Path) -> None:
    split_train_and_test(input_dir=input_dir, test_size=test_size, output_dir=output_dir)


@click.command("post-process")
@test_dir_option
@phenopacket_dir_option
@output_dir_option
@score_option
def post_process_test_dir_command(test_dir: Path, phenopacket_dir: Path, output_dir: Path,
                                  score: str = "NEW_SCORE") -> None:
    output_dir.joinpath("pheval_variant_results").mkdir(parents=True, exist_ok=True)
    post_process_test_dir(test_dir, phenopacket_dir, output_dir, score)


@click.command("manual-predict")
@test_dir_option
@features_option
@coefficients_option
@intercept_option
@output_dir_option
@phenopacket_dir_option
def run_manual_logistic_regression_model_command(test_dir: Path, features: List[str], coefficients: List[float],
                                                 intercept: float, output_dir: Path, phenopacket_dir: Path):
    output_dir.joinpath("pheval_variant_results").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("raw_results").mkdir(parents=True, exist_ok=True)
    run_manual_logistic_regression_model(test_dir, features, coefficients, intercept, output_dir, phenopacket_dir)
