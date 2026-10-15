from pathlib import Path

import polars as pl

EXOMISER_PARQUET_FILE_SUFFIX = "-exomiser.parquet"
EXOMISER_TSV_FILE_SUFFIX = "-exomiser.variants.tsv"


def read_result(path: Path) -> pl.DataFrame:
    """Read an Exomiser variant result, as parquet or TSV depending on the file suffix."""
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="\t", infer_schema_length=None)


def write_result(df: pl.DataFrame, path: Path) -> None:
    """Write a result as parquet, replacing any existing suffix with .parquet."""
    df.write_parquet(path.with_suffix(".parquet"))


def find_exomiser_result(result_dir: Path, sample_id: str) -> Path:
    """Locate the Exomiser variant result for a sample, preferring parquet over TSV."""
    for suffix in (EXOMISER_PARQUET_FILE_SUFFIX, EXOMISER_TSV_FILE_SUFFIX):
        result_path = result_dir.joinpath(sample_id + suffix)
        if result_path.exists():
            return result_path
    raise FileNotFoundError(f"No Exomiser variant result found for {sample_id} in {result_dir}")
