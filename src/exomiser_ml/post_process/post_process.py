from pathlib import Path

from pheval.post_processing.post_processing import generate_variant_result, SortOrder
from pheval.utils.file_utils import all_files
import polars as pl


def post_process_test_dir(test_dir: Path, phenopacket_dir: Path, output_dir: Path, score: str = "NEW_SCORE") -> None:
    for output_file in all_files(test_dir):
        output = pl.read_csv(output_file, separator="\t", infer_schema_length=10000)
        variant_results = output.select([
            pl.col("CONTIG").alias("chrom").cast(pl.String),
            pl.col("START").alias("start").cast(pl.Int64),
            pl.col("END").alias("end").cast(pl.Int64),
            pl.col("REF").alias("ref").cast(pl.String),
            pl.col("ALT").alias("alt").cast(pl.String),
            pl.col(score).alias("score").cast(pl.Float64),
        ])
        generate_variant_result(results=variant_results,
                                phenopacket_dir=phenopacket_dir,
                                sort_order=SortOrder.DESCENDING,
                                output_dir=output_dir,
                                result_path=Path(str(output_file.name).replace("-exomiser.variants", "")))


