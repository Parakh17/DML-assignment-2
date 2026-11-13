"""
Standalone Pipeline Orchestrator CLI.
Allows running the complete end-to-end recommendation pipeline with a single command.
"""

import time
from src.ingestion.ingest import download_movielens
from src.validation.validate import validate_datasets
from src.preparation.prepare import prepare_data
from src.transformation.transform import transform_data
from src.features.feature_store import build_features
from src.training.train import train_models
from src.evaluation.evaluate import run_evaluation


def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("🚀 RUNNING END-TO-END RECOMMENDATION DATA MANAGEMENT PIPELINE")
    print("=" * 70)

    # 1. Ingestion
    print("\n[Stage 1/7] Ingesting Data...")
    ingest_res = download_movielens()

    # 2. Validation
    print("\n[Stage 2/7] Validating Data Quality & Referential Integrity...")
    validate_res = validate_datasets()

    # 3. Preparation & Time Split
    print("\n[Stage 3/7] Preparing Data & Applying Chronological Split...")
    prepare_res = prepare_data()

    # 4. Transformation
    print("\n[Stage 4/7] Transforming & Computing Interaction Matrix...")
    transform_res = transform_data()

    # 5. Features
    print("\n[Stage 5/7] Building Feature Store & Catalog...")
    feature_res = build_features()

    # 6. Training & MLflow Tracking
    print("\n[Stage 6/7] Training Baseline & Personalized SVD Models...")
    train_res = train_models()

    # 7. Evaluation & Metrics
    print("\n[Stage 7/7] Evaluating Ranking & Rating Metrics...")
    eval_res = run_evaluation()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

