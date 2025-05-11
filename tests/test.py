# test_data_collection_dag.py

import pytest
from airflow.models import DagBag
from airflow.operators.bash import BashOperator

# Fixture to load the DAGs from the current directory
@pytest.fixture()
def dagbag():
    # Adjust 'dag_folder' if your DAGs are located elsewhere
    dagbag = DagBag(dag_folder=".", include_examples=False)
    assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"
    return dagbag

def test_dag_loaded(dagbag):
    dag_id = "data_collection_preprocessing_pipeline"
    dag = dagbag.get_dag(dag_id)
    assert dag is not None, f"DAG {dag_id} not found"
    assert dag.dag_id == dag_id

def test_tasks_present(dagbag):
    dag = dagbag.get_dag("data_collection_preprocessing_pipeline")
    task_ids = list(dag.task_dict.keys())
    expected_tasks = {"fetch_data", "preprocess_data"}

    assert expected_tasks.issubset(set(task_ids)), \
        f"Missing tasks: {expected_tasks - set(task_ids)}"
    assert len(task_ids) == 2, f"Expected 2 tasks, found {len(task_ids)}"

def test_task_dependencies(dagbag):
    dag = dagbag.get_dag("data_collection_preprocessing_pipeline")
    fetch_data = dag.get_task("fetch_data")
    preprocess_data = dag.get_task("preprocess_data")

    downstream_tasks = fetch_data.get_direct_relatives(upstream=False)

    assert preprocess_data in downstream_tasks, \
        "Expected 'preprocess_data' to be downstream of 'fetch_data'"

def test_task_types(dagbag):
    dag = dagbag.get_dag("data_collection_preprocessing_pipeline")
    assert isinstance(dag.get_task("fetch_data"), BashOperator), \
        "'fetch_data' should be a BashOperator"
    assert isinstance(dag.get_task("preprocess_data"), BashOperator), \
        "'preprocess_data' should be a BashOperator"
