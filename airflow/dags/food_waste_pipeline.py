
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="food_waste_prediction_pipeline",
    description="Smart Food Waste Prediction and Management Pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command="python /opt/airflow/src/data_ingestion.py",
    )

    cleaning = BashOperator(
        task_id="data_cleaning",
        bash_command="python /opt/airflow/src/data_cleaning.py",
    )

    analysis = BashOperator(
        task_id="data_analysis",
        bash_command="python /opt/airflow/src/data_analysis.py",
    )

    prediction = BashOperator(
        task_id="demand_prediction",
        bash_command="python /opt/airflow/src/demand_prediction.py",
    )

    evaluation = BashOperator(
        task_id="model_evaluation",
        bash_command="python /opt/airflow/src/model_evaluation.py",
    )

    recommendation = BashOperator(
        task_id="prep_recommendation",
        bash_command="python /opt/airflow/src/prep_recommendation.py",
    )

    ingestion >> cleaning >> analysis >> prediction >> evaluation >> recommendation
