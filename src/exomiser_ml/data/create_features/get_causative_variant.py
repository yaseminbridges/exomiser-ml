from pathlib import Path
from typing import List
from pheval.utils.phenopacket_utils import phenopacket_reader, PhenopacketUtil


def extract_causative_variants(phenopacket_path: Path) -> List[str]:
    phenopacket = phenopacket_reader(phenopacket_path)
    causative_variants = PhenopacketUtil(phenopacket).diagnosed_variants()
    return [f"{v.chrom}-{v.pos}-{v.ref}-{v.alt}_" for v in causative_variants]


