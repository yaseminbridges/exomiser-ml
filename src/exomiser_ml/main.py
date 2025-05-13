import click

from exomiser_ml.cli import run_model_command, run_model_pipeline, add_features_command, split_data_command, \
    post_process_test_dir_command


@click.group()
def main():
    """main command group"""


@main.group()
def ml():
    """machine learning utilities."""


ml.add_command(add_features_command)
ml.add_command(split_data_command)
ml.add_command(post_process_test_dir_command)
ml.add_command(run_model_command)
ml.add_command(run_model_pipeline)

if __name__ == "__main__":
    main()
