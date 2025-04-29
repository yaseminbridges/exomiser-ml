from pathlib import Path
import random

import click
from pheval.utils.file_utils import all_files
import polars as pl

def split_train_and_test(input_dir: Path, test_size: float, output_dir: Path) -> None:
    output_dir.joinpath("train").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("test").mkdir(parents=True, exist_ok=True)
    result_files = all_files(input_dir)
    random.seed(42)
    random.shuffle(result_files)
    split_idx = int(len(result_files) * (1 - test_size))
    training_data = pl.concat([pl.read_csv(file, separator="\t", infer_schema_length=0) for file in result_files[:split_idx]])
    training_data.write_csv(output_dir.joinpath("train/train.tsv"), separator="\t")
    [f.rename(output_dir.joinpath(f"test/{f.name}")) for f in result_files[split_idx:]]

@click.command("split-data")
@click.option('--input-dir', "-i", type=Path, required=True, help="Path to the input data directory.")
@click.option('--test-size', "-t", type=float, default=0.2, help='Proportion of data to use for testing.')
@click.option('--output-dir', "-o", type=Path, required=True, help="Path to the output directory.")
def split_data(input_dir: Path, test_size: float, output_dir: Path) -> None:
    split_train_and_test(input_dir=input_dir, test_size=test_size, output_dir=output_dir)