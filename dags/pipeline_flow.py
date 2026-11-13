"""
Pipeline Orchestration Workflow using Prefect.
Orchestrates the complete data management and model lifecycle:
Ingestion -> Validation -> Preparation -> Transformation -> Features -> Training -> Evaluation
"""

import os
import sys
from pathlib import Path

# Ensure Prefect stores its state inside the workspace project directory
os.environ["PREFECT_HOME"] = str(Path(__file__).resolve().parent.parent / ".prefect")

from prefect import flow, task

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.ingest import download_movielens
from src.validation.validate import validate_datasets
from src.preparation.prepare import prepare_data
from src.transformation.transform import transform_data
from src.features.feature_store import build_features
from src.training.train import train_models
from src.evaluation.evaluate import run_evaluation


@task(name="Data Ingestion", retries=2, retry_delay_seconds=5)
def task_ingestion():
    print(">>> Starting Ingestion Task...")
    return download_movielens()


@task(name="Data Quality Validation")
def task_validation(ingest_res):
    print(">>> Starting Validation Task...")
    return validate_datasets()


@task(name="Data Preparation & Split")
def task_preparation(val_res):
    print(">>> Starting Preparation Task (Chronological Split)...")
    return prepare_data()


@task(name="Data Transformation")
def task_transformation(prep_res):
    print(">>> Starting Transformation Task (Interaction Matrix & Summaries)...")
    return transform_data()


@task(name="Feature Store Generation")
def task_features(prep_res):
    print(">>> Starting Feature Store Generation...")
    return build_features()


@task(name="Model Training & MLflow Logging")
def task_training(prep_res, feat_res):
    print(">>> Starting Model Training Task...")
    return train_models()


@task(name="Model Evaluation & Reporting")
def task_evaluation(train_res):
    print(">>> Starting Model Evaluation Task...")
    return run_evaluation()


@flow(name="End-to-End Recommendation Pipeline", log_prints=True)
def recommendation_pipeline_flow():
    """
    Master Prefect workflow executing the end-to-end recommendation pipeline.
    """
    print("\n" + "=" * 60)
    print("  LAUNCHING END-TO-END RECOMMENDATION DATA MANAGEMENT PIPELINE")
    print("=" * 60 + "\n")

    ingest_out = task_ingestion()
    val_out = task_validation(ingest_out)
    prep_out = task_preparation(val_out)
    trans_out = task_transformation(prep_out)
    feat_out = task_features(prep_out)
    train_out = task_training(prep_out, feat_out)
    eval_out = task_evaluation(train_out)

    print("\n" + "=" * 60)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")
    return eval_out


if __name__ == "__main__":
    recommendation_pipeline_flow()
