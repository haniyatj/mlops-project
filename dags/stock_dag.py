from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scripts_path = os.path.join(project_root, "scripts")

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
        bash_command=f'python {scripts_path}/fetch_stock.py',
        cwd=project_root,
        pool='stock_data_pool'
    )

    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command=f'python {scripts_path}/clean_stock.py',
        cwd=project_root,
        pool='stock_data_pool'
    )

    fetch_data >> preprocess_data
