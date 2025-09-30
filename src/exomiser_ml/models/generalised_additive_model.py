import json
from pathlib import Path
from typing import List
import polars as pl
import joblib
from interpret.glassbox import ExplainableBoostingClassifier

from exomiser_ml.post_process.post_process import post_process_test_dir
from exomiser_ml.utils.write_metadata import RunMetadata, write_metadata_yaml
from pheval.utils.file_utils import all_files
import math
from bisect import bisect_right

# ---------- TRAIN ----------
def train_gam(training_data: Path, features: List[str], output_dir: Path):
    df = pl.read_csv(training_data, separator="\t", infer_schema_length=0)
    df = df.with_columns(
        (df["CAUSATIVE_VARIANT"].str.to_lowercase() == "true")
        .cast(pl.Int8)
        .alias("CAUSATIVE_VARIANT")
    )
    X = df.select(features).to_numpy()
    y = df.select(["CAUSATIVE_VARIANT"]).cast(pl.Int8).to_numpy().ravel()
    ebm = ExplainableBoostingClassifier(
        interactions=0,
        learning_rate=0.01,
        max_bins=256,
        max_leaves=3,
        outer_bags=8,
        inner_bags=0,
        random_state=42,
        feature_names=features,
    )
    ebm.fit(X, y)
    joblib.dump(ebm, output_dir.joinpath("model/EBM.pkl"))
    return ebm

def train_and_test_gam(
    training_data: Path,
    test_dir: Path,
    features: List[str],
    output_dir: Path,
    phenopacket_dir: Path
):
    raw_results_dir = output_dir.joinpath("raw_results")
    output_dir.joinpath("model").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("raw_results").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("pheval_variant_results").mkdir(parents=True, exist_ok=True)
    features = list(features)

    # Train model and save
    ebm = train_gam(training_data=training_data, features=features, output_dir=output_dir)
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    for test_file in all_files(test_dir):
        df = pl.read_csv(test_file, separator="\t", infer_schema_length=0)
        X_test = df.select(features).to_numpy()
        probs = ebm.predict_proba(X_test)[:, 1]  # probability of class 1
        new_scores = pl.DataFrame({"NEW_SCORE": probs})
        if "NEW_SCORE" in df.columns:
            print(f"Warning: 'NEW_SCORE' already exists in {test_file}. Replacing it.")
            df = df.drop("NEW_SCORE")
        df_with_new_scores = df.hstack(new_scores)
        df_with_new_scores.write_csv(raw_results_dir.joinpath(test_file.name), separator="\t")

    # Post-process
    post_process_test_dir(
        test_dir=raw_results_dir,
        phenopacket_dir=phenopacket_dir,
        output_dir=output_dir
    )

    # Metadata
    metadata = RunMetadata(
        test_size=None,
        output_dir=str(output_dir),
        model_type="EBM (fitted predict_proba)",
        features_used=list(features),
        training_data=str(training_data),
        test_dir=str(test_dir),
    )
    write_metadata_yaml(metadata, output_dir)

# ---------- MANUAL CALCULATION ----------


def _inv_link(z, link):
    if link in ("logit", "custom_binary"):  # binary default is logit
        return 1.0 / (1.0 + math.exp(-z))
    elif link == "probit":
        # simple probit via erf; for exactness you'd want scipy.norm.cdf
        return 0.5 * (1.0 + math.erf(z / (2**0.5)))
    elif link in ("cloglog", "loglog", "cauchit"):
        # add if you used these; most classifiers use logit
        raise NotImplementedError(f"Link {link} not implemented here.")
    else:
        raise NotImplementedError(f"Link {link} not supported.")

def _bin_index_continuous(value, cuts):
    """
    EBM continuous bins (main effects): missing bin is index 0.
    Non-missing values are placed into 1..N according to cuts.
    We mimic numpy.searchsorted on the cutpoints.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    # cuts are ascending thresholds; bins: (-inf, cuts[0]], (cuts[0], cuts[1]], ..., (cuts[-1], inf)
    # EBM’s exact edge-inclusion may vary slightly; bisect_right works well in practice.
    return 1 + bisect_right(cuts, value)

def _bin_index_categorical(value, mapping):
    """
    mapping is a dict: category -> bin_index (>=1 typically).
    Missing/unknown -> 0 (missing bin).
    """
    if value is None:
        return 0
    return mapping.get(value, 0)

def _resolve_feature_bins_for_term(bins_per_feature, feat_idx, interaction_order):
    """
    bins_[feat_idx] is a list of resolutions:
      index 0 for mains; index 1 for pairs; last for higher-order.
    """
    resolutions = bins_per_feature[feat_idx]
    # choose resolution index = interaction_order - 1 if available, else use last
    idx = interaction_order - 1
    if idx < 0: idx = 0
    if idx >= len(resolutions): idx = len(resolutions) - 1
    return resolutions[idx]

def ebm_predict_proba_row(row, payload):
    """
    row: dict feature_name -> value
    payload: dict with keys:
      intercept, link, feature_names, feature_types, term_features, bins, term_scores
    returns: probability of class 1
    """
    feature_names = payload["feature_names"]
    feature_types = payload["feature_types"]
    bins_pf = payload["bins"]
    term_feats = payload["term_features"]
    term_scores = payload["term_scores"]
    z = payload["intercept"]

    # Precompute per-feature bin index at all needed resolutions for main terms
    # (for interactions we re-bin at the interaction resolution)
    for t_idx, feats in enumerate(term_feats):
        order = len(feats)  # 1 for main, 2 for pair, etc.
        if order == 1:
            f = feats[0]
            ftype = feature_types[f]
            f_name = feature_names[f]
            val = row.get(f_name, None)
            bins_def = _resolve_feature_bins_for_term(bins_pf, f, interaction_order=1)

            if ftype == "continuous":
                # bins_def is list of cutpoints (floats)
                bi = _bin_index_continuous(val, bins_def)
                z += term_scores[t_idx][bi]
            else:
                # nominal/ordinal: bins_def is dict category->bin_index
                bi = _bin_index_categorical(val, bins_def)
                z += term_scores[t_idx][bi]

        else:
            # interaction: compute per-dimension bin index using the appropriate resolution
            # then index the tensor term_scores accordingly
            idxs = []
            for d, f in enumerate(feats):
                ftype = feature_types[f]
                f_name = feature_names[f]
                val = row.get(f_name, None)
                bins_def = _resolve_feature_bins_for_term(bins_pf, f, interaction_order=order)
                if ftype == "continuous":
                    bi = _bin_index_continuous(val, bins_def)
                else:
                    bi = _bin_index_categorical(val, bins_def)
                idxs.append(bi)

            # Drill into the N-dimensional array
            ts = term_scores[t_idx]
            for bi in idxs:
                ts = ts[bi]
            z += ts

    return _inv_link(z, payload["link"])

def manual_predict_ebm(payload_path: Path, test_dir: Path, output_dir: Path, phenopacket_dir: Path):
    with open(payload_path) as f:
        payload = json.load(f)
    raw_results_dir = output_dir.joinpath("raw_results")
    output_dir.joinpath("model").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("raw_results").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("pheval_variant_results").mkdir(parents=True, exist_ok=True)
    feature_cols = payload["feature_names"]
    for test_file in all_files(test_dir):
        df = pl.read_csv(test_file, separator="\t", infer_schema_length=0)
        if "NEW_SCORE" in df.columns:
            print(f"Warning: 'NEW_SCORE' already exists in {test_file}. Replacing it.")
            df = df.drop("NEW_SCORE")
        df = df.with_columns(
            pl.struct(feature_cols).map_elements(
                lambda row: ebm_predict_proba_row(row, payload),  # your scorer
                return_dtype=pl.Float64
            ).alias("NEW_SC0RE")
        )
        df.write_csv(output_dir.joinpath(test_file.name), separator="\t")
    post_process_test_dir(
        test_dir=raw_results_dir,
        phenopacket_dir=phenopacket_dir,
        output_dir=output_dir
    )
    metadata = RunMetadata(
        test_size=None,
        output_dir=str(output_dir),
        model_type="EBM (fitted predict_proba)",
        features_used=list(payload["feature_names"]),
        training_data="",
        test_dir=str(test_dir),
    )
    write_metadata_yaml(metadata, output_dir)
