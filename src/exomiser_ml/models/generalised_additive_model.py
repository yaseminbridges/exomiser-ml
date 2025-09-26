import json
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import joblib
from interpret.glassbox import ExplainableBoostingClassifier

from exomiser_ml.post_process.post_process import post_process_test_dir
from exomiser_ml.utils.write_metadata import RunMetadata, write_metadata_yaml
from pheval.utils.file_utils import all_files


# ---------- TRAIN ----------
def train_gam(training_data: Path, features: List[str], output_dir: Path):
    df = pl.read_csv(training_data, separator="\t", infer_schema_length=0)
    features_copy = features.copy()
    medians = {}
    for feature in features:
        flag = f"is_missing__{feature}"
        df = df.with_columns(df[feature].is_null().cast(pl.Int8).alias(flag))
        med = df[feature].median()
        if med is None or np.isnan(med):
            med = 0.0
        df = df.with_columns(df[feature].fill_null(med))
        medians[feature] = float(med)
        features_copy.append(flag)
    X = df.select(features_copy).to_numpy()
    y = df.select(["CAUSATIVE_VARIANT"]).cast(pl.Int8).to_numpy().ravel()
    ebm = ExplainableBoostingClassifier(
        interactions=0,
        learning_rate=0.01,
        max_bins=256,
        max_leaves=3,
        outer_bags=8,
        inner_bags=0,
        random_state=42,
    )
    ebm.fit(X, y)
    # Save model for reference
    joblib.dump(ebm, output_dir.joinpath("model/EBM.pkl"))
    export = {
        "intercept": float(ebm.intercept_[0]),
        "medians": medians,
        "features": {}
    }
    term_for_feat = {}
    for t_idx, feats_idx in enumerate(ebm.term_features_):
        if len(feats_idx) == 1:
            term_for_feat[features_copy[feats_idx[0]]] = t_idx
    for f in features_copy:
        t = term_for_feat[f]
        cutpoints = list(map(float, ebm.bins_[t][0])) if len(ebm.bins_[t]) else []
        bin_edges = [-1e308] + cutpoints + [1e308]
        bin_scores = list(map(float, ebm.scores_[t]))
        export["features"][f] = {
            "bin_edges": bin_edges,
            "bin_scores": bin_scores
        }
    with open(output_dir.joinpath("model/EBM_export.json"), "w") as f:
        json.dump(export, f, indent=2)
    return medians


# ---------- FORMAT TEST DATA ----------
def format_data_with_missing(df: pl.DataFrame, features: List[str], medians: dict):
    feats = features.copy()
    for f in features:
        flag = f"is_missing__" + f
        df = df.with_columns(df[f].is_null().cast(pl.Int8).alias(flag))
        df = df.with_columns(df[f].fill_null(medians[f]))
        feats.append(flag)
    return df, feats


# ---------- MANUAL PREDICT ----------
def manual_predict_df(df: pl.DataFrame, export: dict) -> pl.Series:
    """Compute NEW_SCORE for all rows using EBM export dict."""
    z = np.full(len(df), export["intercept"], dtype=float)

    # Handle missing + flags
    for f, med in export["medians"].items():
        if f not in df.columns:
            df = df.with_columns(pl.lit(None).alias(f))
        flag = "is_missing__" + f
        df = df.with_columns(df[f].is_null().cast(pl.Int8).alias(flag))
        df = df.with_columns(df[f].fill_null(med))

    # Add contributions
    for f, spec in export["features"].items():
        values = df[f].to_numpy()
        edges = np.array(spec["bin_edges"], dtype=float)
        scores = np.array(spec["bin_scores"], dtype=float)
        idx = np.searchsorted(edges, values, side="right") - 1
        idx = np.clip(idx, 0, len(scores) - 1)
        z += scores[idx]

    probs = 1 / (1 + np.exp(-z))
    return pl.Series("NEW_SCORE", probs)


def manual_predict_gam(gam_export: Path, test_dir: Path, output_dir: Path):
    """Apply manual EBM prediction using exported numbers."""
    with open(gam_export) as f:
        export = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    for test_file in all_files(test_dir):
        df = pl.read_csv(test_file, separator="\t", infer_schema_length=0)
        new_scores = manual_predict_df(df, export)
        if "NEW_SCORE" in df.columns:
            print(f"Warning: 'NEW_SCORE' already exists in {test_file}. Replacing it.")
            df = df.drop("NEW_SCORE")
        df_with_new_scores = df.hstack([new_scores])
        df_with_new_scores.write_csv(output_dir.joinpath(test_file.name), separator="\t")


# ---------- HIGH-LEVEL WRAPPER ----------
def train_and_test_gam(training_data: Path, test_dir: Path, features: List[str], output_dir: Path, phenopacket_dir: Path):
    raw_results_dir = output_dir.joinpath("raw_results")

    medians = train_gam(training_data=training_data, features=features, output_dir=output_dir)

    # Run manual prediction (using exported JSON instead of .predict_proba)
    manual_predict_gam(
        gam_export=output_dir.joinpath("model/EBM_export.json"),
        test_dir=test_dir,
        output_dir=raw_results_dir
    )

    # Post-process
    post_process_test_dir(test_dir=raw_results_dir, phenopacket_dir=phenopacket_dir, output_dir=output_dir)

    # Metadata
    metadata = RunMetadata(
        test_size=None,
        output_dir=str(output_dir),
        model_type="EBM (manual prediction)",
        features_used=list(features),
        training_data=str(training_data),
        test_dir=str(test_dir)
    )
    write_metadata_yaml(metadata, output_dir)

def run_manual_ebm_model(
    test_dir: Path,
    export_json: Path,
    output_dir: Path,
    phenopacket_dir: Path,
):
    raw_results_dir = output_dir.joinpath("raw_results")

    # Run manual predictions on all files
    manual_predict_gam(
        gam_export=export_json,
        test_dir=test_dir,
        output_dir=raw_results_dir,
    )

    # Post-process to standardised results
    post_process_test_dir(
        test_dir=raw_results_dir,
        phenopacket_dir=phenopacket_dir,
        output_dir=output_dir,
    )

    # Write run metadata
    metadata = RunMetadata(
        test_size=None,
        output_dir=str(output_dir),
        model_type="EBM (manual prediction)",
        features_used=[],
        training_data="N/A",
        test_dir=str(test_dir),
    )
    write_metadata_yaml(metadata, output_dir)