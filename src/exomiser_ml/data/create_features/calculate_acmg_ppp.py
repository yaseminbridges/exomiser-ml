import re
from typing import List, Union

# These constants are derived in "Modeling the ACMG/AMP Variant Classification Guidelines as a Bayesian
#  Classification Framework" Tavtigian et al. 2018, DOI:10.1038/gim.2017.210
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6336098/bin/NIHMS915467-supplement-Supplemental_Table_S1.xlsx
# Very_Strong == (2 * Strong) == (2 * Moderate) == (2 * Supporting)
# therefore points Supporting = 1, Moderate = 2, Strong = 4, Very_strong = 8 can be assigned and these fit to a
# Bayesian classification framework where (using the combining rules from Riggs et al. 2016) the posterior
# probabilities are Path >= 0.99, LikelyPath 0.90 - 0.98, LikelyBenign 0.1 - 0.01, Benign < 0.01

class AcmgClassification:
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    UNCERTAIN_SIGNIFICANCE = "Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"


class ACMGPPPCalculator:
    PRIOR_PROB = 0.1
    ODDS_PATH_VERY_STRONG = 350.0
    EXPONENTIAL_PROGRESSION = 2.0
    SUPPORTING_EVIDENCE_EXPONENT = pow(EXPONENTIAL_PROGRESSION, -3)  # 0.125
    ODDS_PATH_SUPPORTING = pow(ODDS_PATH_VERY_STRONG, SUPPORTING_EVIDENCE_EXPONENT)  # 2.08

    # Mapping of ACMG evidence types to their respective weights
    EVIDENCE_WEIGHTS = {
        'A': 8,  # Stand alone
        'VS': 8,  # Very Strong
        'S': 4,  # Strong
        'M': 2,  # Moderate
        'P': 1,  # Supporting
        # modifier categories
        'STANDALONE': 8,  # Stand alone
        'VERYSTRONG': 8,  # Very Strong
        'STRONG': 4,  # Strong
        'MODERATE': 2,  # Moderate
        'SUPPORTING': 1  # Supporting
    }

    @staticmethod
    def normalize_input(acmg_evidence) -> Union[List[str], bool]:
        # Remove brackets and extra spaces, then split by space or comma
        try:
            acmg_evidence = re.sub(r'[\[\]]', '', acmg_evidence).strip()
            acmg_evidence = re.split(r'[ ,]+', acmg_evidence)
            return acmg_evidence
        except TypeError:
            return False

    def parse_evidence(self, evidence: str, filter_clinvar: bool) -> int:
        evidence = evidence.upper()
        if filter_clinvar:
            if "PP5" in evidence or "BP6" in evidence:
                return 0
        match = re.match(r'([BP])([A-Z]+)(\d)(_([A-Z]+))?', evidence)

        if match:
            pathogenicity, strength, category, _, modifier = match.groups()
            key = strength
            if modifier:
                key = modifier
            weight = self.EVIDENCE_WEIGHTS.get(key, 0)
            return weight if pathogenicity == 'P' else -weight
        return 0

    def calc_post_prob_path(self, points: int) -> float:
        # Equation 2 from Tavtigian et al., 2020 (DOI: 10.1002/humu.24088) which is a re-written from of equation 5 from
        # Tavtigian et al., 2018 (DOI: 10.1038/gim.2017.210)
        oddsPath = pow(self.ODDS_PATH_SUPPORTING, points)
        # posteriorProbability = (OddsPathogenicity*Prior P)/((OddsPathogenicity−1)*Prior_P+1)
        return (oddsPath * self.PRIOR_PROB) / ((oddsPath - 1) * self.PRIOR_PROB + 1)

    @staticmethod
    def classification(points) -> str:
        if points >= 10:
            return AcmgClassification.PATHOGENIC
        if 6 <= points <= 9:
            return AcmgClassification.LIKELY_PATHOGENIC
        if 0 <= points <= 5:
            return AcmgClassification.UNCERTAIN_SIGNIFICANCE
        if -6 <= points <= -1:
            return AcmgClassification.LIKELY_BENIGN
        return AcmgClassification.BENIGN


    def compute_posterior(self, evidence_string: str, filter_clinvar: bool) -> float:
        normalized = self.normalize_input(evidence_string)
        if not normalized:
            return 0  # Or 0.0 or np.nan depending on your context
        points = sum(self.parse_evidence(e, filter_clinvar) for e in normalized)
        return self.calc_post_prob_path(points)
