# Exomiser ML

Exomiser-ML is a Python-based machine learning pipeline designed to enhance variant prioritisation by integrating various Exomiser scores. It supports classification using Logistic Regression, Random Forest, and XGBoost, and includes utilities for feature extraction, data splitting, model training, and post-processing.

🚀 Features 
* Extracts features from Exomiser variant TSV result files and Phenopackets.
* Supports multiple classifiers: Logistic Regression, Random Forest, and XGBoost.
* Provides CLI commands for training, and full pipeline execution.
* Generates metadata and post-processed results compatible with PhEval for benchmarking.

📦 Installation

Ensure you have Python 3.12 or higher installed.

```bash
pip install exomiser-ml
```

🧪 Usage

The package provides several CLI commands:

1. run-model

Trains and evaluates a model using provided training data and test directory.

```bash
run-model \
  --training-data path/to/train.tsv \
  --test-dir path/to/test_dir \
  --features FEATURE1 FEATURE2 ... \
  --output-dir path/to/output \
  --phenopacket-dir path/to/phenopackets \
  --model MODEL_TYPE
```

Parameters:
* --training-data: Path to the training data TSV file.
* --test-dir: Directory containing test data files.
* --features: List of features to extract.
* --output-dir: Directory to save outputs.
* --phenopacket-dir: Directory containing Phenopacket JSON files.
* --model: Model type to use. Choices: LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST_CLASSIFIER

2. run-pipeline

Executes the full pipeline: feature extraction, data splitting, training, evaluation, and post-processing.

```bash
run-pipeline \
  --phenopacket-dir path/to/phenopackets \
  --result-dir path/to/exomiser_results \
  --output-dir path/to/output \
  --features FEATURE1 FEATURE2 ... \
  --test-size 0.2 \
  --model MODEL_TYPE
```

Parameters:
* --phenopacket-dir: Directory containing Phenopacket JSON files.
* --result-dir: Directory containing Exomiser result TSV files.
* --output-dir: Directory to save outputs.
* --features: List of features to extract.
* --test-size: Proportion of data to use for testing (e.g., 0.2 for 20%).
* --model: Model type to use. Choices: LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST_CLASSIFIER.

3. add-features

Adds features to Exomiser results.

```bash
add-features \
  --phenopacket-dir path/to/phenopackets \
  --result-dir path/to/exomiser_results \
  --output-dir path/to/output
```

4. split-data

Splits data (Exomiser TSV results) into training and testing sets.

```bash
split-data \
  --input-dir path/to/input_data \
  --test-size 0.2 \
  --output-dir path/to/output
```

5. post-process

Post-processes test results for downstream benchmarking with PhEval.

```bash
post-process \
  --test-dir path/to/test_results \
  --phenopacket-dir path/to/phenopackets \
  --output-dir path/to/output \
  --score NEW_SCORE
```