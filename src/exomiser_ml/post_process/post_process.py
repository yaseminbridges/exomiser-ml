from pathlib import Path

import click
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
                                result_path=Path(str(output_file.name).replace("-exomiser.variants.tsv", "")))


@click.command("post-process")
@click.option('--test-dir', "-t", type=Path, required=True,
              help="Path to the test data directory.")
@click.option('--phenopacket-dir', "-p", type=Path, required=True, help="Path to the Phenopacket data directory.")

@click.option('--output-dir', "-o", type=Path, required=True, help="Path to the output directory.")
@click.option('--score', "-s", type=str, default="NEW_SCORE", help='Score to rank.')

def post_process_test_dir_command(test_dir: Path, phenopacket_dir: Path, output_dir: Path, score: str = "NEW_SCORE") -> None:
    output_dir.joinpath("pheval_variant_results").mkdir(parents=True, exist_ok=True)
    post_process_test_dir(test_dir, phenopacket_dir, output_dir, score)