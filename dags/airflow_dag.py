"""
Apache Airflow DAG specification for the Recommendation System Pipeline.
Handles task dependencies across the complete data management and model lifecycle.
"""

from datetime import datetime, timedelta
try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="recommendation_pipeline_dag",
        default_args=default_args,
        description="End-to-End Recommendation Data Management Pipeline",
        schedule_interval="@daily",
        catchup=False,
        tags=["recommendations", "data-management", "mlops"],
    ) as dag:

        ingest = BashOperator(
            task_id="ingest_data",
            bash_command="python -m src.ingestion.ingest",
        )

        validate = BashOperator(
            task_id="validate_data",
            bash_command="python -m src.validation.validate",
        )

        prepare = BashOperator(
            task_id="prepare_data",
            bash_command="python -m src.preparation.prepare",
        )

        transform = BashOperator(
            task_id="transform_data",
            bash_command="python -m src.transformation.transform",
        )

        features = BashOperator(
            task_id="build_features",
            bash_command="python -m src.features.feature_store",
        )

        train = BashOperator(
            task_id="train_models",
            bash_command="python -m src.training.train",
        )

        evaluate = BashOperator(
            task_id="evaluate_models",
            bash_command="python -m src.evaluation.evaluate",
        )

        # Task dependency graph
        ingest >> validate >> prepare
        prepare >> transform
        prepare >> features
        [transform, features] >> train >> evaluate

