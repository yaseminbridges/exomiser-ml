from pathlib import Path

import polars as pl
from pheval.utils.file_utils import all_files

from exomiser_ml.data.create_features.calculate_acmg_ppp import ACMGPPPCalculator
from exomiser_ml.data.create_features.get_causative_variant import extract_causative_variants

EXOMISER_TSV_FILE_SUFFIX = "-exomiser.variants.tsv"


def get_result(phenopacket_path: Path, result_dir: Path) -> pl.DataFrame:
    result_path = result_dir.joinpath(phenopacket_path.stem + EXOMISER_TSV_FILE_SUFFIX)
    return pl.read_csv(result_path, separator="\t", infer_schema_length=None)


def label_variant(phenopacket_path: Path, result: pl.DataFrame) -> pl.DataFrame:
    causative_variants = extract_causative_variants(phenopacket_path)
    return result.with_columns([
        pl.col("ID").map_elements(lambda x: any(variant in x for variant in causative_variants),
                                  return_dtype=pl.Boolean)
        .alias("CAUSATIVE_VARIANT")
    ])


def add_features(phenopacket_dir: Path, result_dir: Path, output_dir: Path, filter_clinvar: bool) -> None:
    acmg_calculater = ACMGPPPCalculator()
    for phenopacket_path in all_files(phenopacket_dir):
        result = get_result(phenopacket_path, result_dir)
        result = result.with_columns(pl.col("EXOMISER_ACMG_EVIDENCE").fill_null(""))
        labelled_variant = label_variant(phenopacket_path, result)
        acmg_ppp = labelled_variant.with_columns([
            pl.col("EXOMISER_ACMG_EVIDENCE").map_elements(
                lambda x: acmg_calculater.compute_posterior(x, filter_clinvar),
                return_dtype=pl.Float64
            ).alias("ACMG_PPP")
        ])
        acmg_ppp.write_csv(output_dir.joinpath(phenopacket_path.stem + EXOMISER_TSV_FILE_SUFFIX), separator="\t")


