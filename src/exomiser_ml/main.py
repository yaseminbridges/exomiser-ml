from pathlib import Path
from typing import List

import click

from exomiser_ml.data.create_features.add_features import add_features
from exomiser_ml.data.extract_features.extract_features import extract_features
from exomiser_ml.data.split_data.split_train_and_test import split_train_and_test
from exomiser_ml.models.logistic_regression import logistic_regression
from exomiser_ml.post_process.post_process import post_process_test_dir


def main():
    print("Hello from exomiser-ml!")


def run_logistic_regression_pipeline(phenopacket_dir: Path, result_dir: Path, output_dir: Path, features: List[str],
                                     test_size: float):
    add_features(phenopacket_dir=phenopacket_dir, result_dir=result_dir,
                 output_dir=output_dir.joinpath("results_added_features"))
    extract_features(result_dir=output_dir.joinpath("results_added_features"), features=features,
                     output_dir=output_dir.joinpath("results_extracted_features"))
    split_train_and_test(input_dir=output_dir.joinpath("results_extracted_features"),
                         output_dir=output_dir.joinpath("results_split"), test_size=test_size)
    logistic_regression(training_data=output_dir.joinpath("results_split/train/train.tsv"),
                        test_dir=output_dir.joinpath("results_split/test"))
    post_process_test_dir(test_dir=output_dir.joinpath("results_split/test"), phenopacket_dir=phenopacket_dir,
                          output_dir=output_dir.joinpath("pheval_variant_results"))


@click.command("run-lr")
@click.option('--phenopacket-dir', "-p", type=Path)
@click.option('--result-dir', "-r", type=Path)
@click.option('--output-dir', "-o", type=Path)
@click.option('--features', "-f",multiple=True, required=True, help='List of features to extract.')
@click.option('--test-size', "-t", type=float, default=0.2, help='Proportion of data to use for testing.')
def run_lr_pipeline(phenopacket_dir: Path, result_dir: Path, output_dir: Path, features: List[str], test_size: float):
    run_logistic_regression_pipeline(
        phenopacket_dir=phenopacket_dir,
        result_dir=result_dir,
        output_dir=output_dir,
        features=list(features),
        test_size=test_size
    )

if __name__ == "__main__":
    main()
