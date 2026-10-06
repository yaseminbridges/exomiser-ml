from pathlib import Path
import random

from pheval.utils.file_utils import all_files
import polars as pl

from exomiser_ml.utils.io import read_result, write_result

def create_training_data(input_dir: Path, output_dir: Path):
    training_data = pl.concat([read_result(file) for file in all_files(input_dir)], how="diagonal_relaxed")
    write_result(training_data, output_dir.joinpath("train.parquet"))

def split_train_and_test(input_dir: Path, test_size: float, output_dir: Path) -> None:
    output_dir.joinpath("train").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("test").mkdir(parents=True, exist_ok=True)
    result_files = all_files(input_dir)
    random.seed(42)
    random.shuffle(result_files)
    split_idx = int(len(result_files) * (1 - test_size))
    training_data = pl.concat([read_result(file) for file in result_files[:split_idx]], how="diagonal_relaxed")
    write_result(training_data, output_dir.joinpath("train/train.parquet"))
    [f.rename(output_dir.joinpath(f"test/{f.name}")) for f in result_files[split_idx:]]

