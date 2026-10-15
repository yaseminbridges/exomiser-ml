from pathlib import Path

import polars as pl
from pheval.utils.file_utils import all_files

from exomiser_ml.data.create_features.calculate_acmg_ppp import ACMGPPPCalculator
from exomiser_ml.data.create_features.get_causative_variant import extract_causative_variants
from exomiser_ml.utils.io import find_exomiser_result, read_result, write_result, EXOMISER_PARQUET_FILE_SUFFIX

def get_result(phenopacket_path: Path, result_dir: Path) -> pl.DataFrame:
    return read_result(find_exomiser_result(result_dir, phenopacket_path.stem))


def label_variant(phenopacket_path: Path, result: pl.DataFrame) -> pl.DataFrame:
    causative_variants = extract_causative_variants(phenopacket_path)
    return result.with_columns([
        pl.col("variant").map_elements(lambda x: any(variant in x for variant in causative_variants),
                                  return_dtype=pl.Boolean)
        .alias("CAUSATIVE_VARIANT")
    ])


def add_features(phenopacket_dir: Path, result_dir: Path, output_dir: Path, filter_clinvar: bool, filter_bs4: bool,
                 filter_pp4: bool) -> None:
    acmg_calculater = ACMGPPPCalculator()
    for phenopacket_path in all_files(phenopacket_dir):
        result = get_result(phenopacket_path, result_dir)

        labelled_variant = label_variant(phenopacket_path, result)
        acmg_ppp = labelled_variant.with_columns([
            pl.col("acmgEvidence").map_elements(
                lambda x: acmg_calculater.compute_posterior(x, filter_clinvar, filter_bs4, filter_pp4),
                return_dtype=pl.Float64
            ).alias("ACMG_PPP")
        ])
        # Compound het / homozygous recessive variants share the mean ACMG_PPP of the contributing variants in the
        # gene, only averaging over contributing rows of the same MOI.
        recessive_contributing = pl.col("isContributingVariant") & pl.col("moi").is_in(["AR", "XR"])
        acmg_ppp = acmg_ppp.with_columns(
            pl.when(recessive_contributing)
            .then(
                pl.when(recessive_contributing).then(pl.col("ACMG_PPP"))
                .mean()
                .over(["geneSymbol", "moi"])
            )
            .otherwise(pl.col("ACMG_PPP"))
            .alias("ACMG_PPP_AVE")
        )

        write_result(acmg_ppp, output_dir.joinpath(phenopacket_path.stem + EXOMISER_PARQUET_FILE_SUFFIX))
