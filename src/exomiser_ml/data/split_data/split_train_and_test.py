from pathlib import Path
import random

from pheval.utils.file_utils import all_files
import polars as pl

def split_train_and_test(input_dir: Path, test_size: float, output_dir: Path) -> None:
    output_dir.joinpath("train").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("test").mkdir(parents=True, exist_ok=True)
    result_files = all_files(input_dir)
    random.seed(42)
    random.shuffle(result_files)
    split_idx = int(len(result_files) * (1 - test_size))
    training_data = pl.concat([pl.read_csv(file, separator="\t", infer_schema_length=None) for file in result_files[:split_idx]])
    training_data.write_csv(output_dir.joinpath("train/train.tsv"), separator="\t")
    [f.rename(output_dir.joinpath(f"test/{f.name}")) for f in result_files[split_idx:]]

