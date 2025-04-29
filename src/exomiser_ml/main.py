import click

from exomiser_ml.data.create_features.add_features import add_features_command
from exomiser_ml.data.split_data.split_train_and_test import split_data
from exomiser_ml.models.logistic_regression import run_logistic_regression, run_lr_pipeline


@click.group()
def main():
    """main command group"""


@main.group()
def ml():
    """machine learning utilities."""


ml.add_command(run_lr_pipeline)
ml.add_command(add_features_command)
ml.add_command(split_data)
ml.add_command(run_logistic_regression)

if __name__ == "__main__":
    main()
