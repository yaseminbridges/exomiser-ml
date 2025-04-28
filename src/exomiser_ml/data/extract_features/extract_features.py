from pathlib import Path
from typing import List

from pheval.utils.file_utils import all_files
import polars as pl


def extract_features(result_dir: Path, features: List[str], output_dir: Path) -> None:
    for result_path in all_files(result_dir):
        result = pl.read_csv(result_path, separator="\t", infer_schema_length=100000)
        selected_features = result.select(features)
        selected_features.write_csv(output_dir.joinpath(result_path.name))
