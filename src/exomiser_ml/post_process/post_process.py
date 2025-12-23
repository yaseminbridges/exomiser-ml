from pathlib import Path
import uuid
from pheval.post_processing.post_processing import generate_variant_result, SortOrder
from pheval.utils.file_utils import all_files
import polars as pl
from enum import Enum

class ModeOfInheritance(Enum):
    AUTOSOMAL_DOMINANT = 1
    AD = 1
    AUTOSOMAL_RECESSIVE = 2
    AR = 2
    X_DOMINANT = 1
    XD = 1
    X_RECESSIVE = 2
    XR = 2
    MITOCHONDRIAL = 3
    MT = 3

def extract_variant_result(exomiser_result: pl.DataFrame, score_name) -> pl.DataFrame:
    contributing_variant_only = exomiser_result.filter(
        pl.col("CONTRIBUTING_VARIANT").cast(pl.Int64) == 1  # noqa
    )
    return (
        contributing_variant_only.select(
            [
                pl.col("GENE_SYMBOL"),
                pl.col("CONTIG").alias("chrom").cast(pl.String),
                pl.col("START").cast(pl.Int64).alias("start"),
                pl.col("END").cast(pl.Int64).alias("end"),
                pl.col("REF").alias("ref"),
                pl.col("ALT").alias("alt"),
                pl.col(score_name).alias("score"),
                pl.col("MOI")
                .map_elements(lambda moi: ModeOfInheritance[moi].value, return_dtype=pl.Int8)
                .alias("moi_enum"),
            ]
        )
        .with_columns(
            [
                (pl.col("moi_enum") == 2).alias("is_recessive"),
                pl.when(pl.col("moi_enum") == 2)
                .then(
                    pl.format(
                        "recessive|{}|{}|{}",
                        pl.col("GENE_SYMBOL"),
                        pl.col("score"),
                        pl.col("moi_enum"),
                    )
                )
                .otherwise(
                    pl.format(
                        "dominant|{}|{}|{}|{}|{}|{}",
                        pl.col("chrom"),
                        pl.col("start"),
                        pl.col("end"),
                        pl.col("ref"),
                        pl.col("alt"),
                        pl.col("score"),
                    )
                )
                .alias("group_key"),
            ]
        )
        .with_columns(
            [
                pl.col("group_key")
                .rank("dense")
                .cast(pl.UInt32)
                .map_elements(
                    lambda i: str(uuid.uuid5(uuid.NAMESPACE_DNS, str(i))), return_dtype=pl.String
                )
                .alias("grouping_id")
            ]
        )
    )

def post_process_test_dir(test_dir: Path, phenopacket_dir: Path, output_dir: Path, score: str = "NEW_SCORE") -> None:
    for output_file in all_files(test_dir):
        output = pl.read_csv(output_file, separator="\t", infer_schema_length=10000)
        variant_results = extract_variant_result(output, score)
        # variant_results = output.select([
        #     pl.col("CONTIG").alias("chrom").cast(pl.String),
        #     pl.col("START").alias("start").cast(pl.Int64),
        #     pl.col("END").alias("end").cast(pl.Int64),
        #     pl.col("REF").alias("ref").cast(pl.String),
        #     pl.col("ALT").alias("alt").cast(pl.String),
        #     pl.col(score).alias("score").cast(pl.Float64),
        # ])
        generate_variant_result(results=variant_results,
                                phenopacket_dir=phenopacket_dir,
                                sort_order=SortOrder.DESCENDING,
                                output_dir=output_dir,
                                result_path=Path(str(output_file.name).replace("-exomiser.variants", "")))


