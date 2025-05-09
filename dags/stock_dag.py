# dags/data_pipeline_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='data_collection_preprocessing_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    description='Pipeline to fetch and preprocess stock data',
) as dag:

    fetch_data = BashOperator(
        task_id='fetch_data',
        bash_command='python scripts/fetch_stock.py'
    )

    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='python scripts/clean_stock.py'
    )

    fetch_data >> preprocess_data
