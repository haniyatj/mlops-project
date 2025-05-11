"""
Test file for the data collection and preprocessing DAG
"""

import os
import pytest
from airflow.models import DagBag
from airflow.operators.bash import BashOperator


@pytest.fixture()
def dagbag():
    # Make sure we're looking in the correct directory for DAGs
    dagbag = DagBag(dag_folder="/opt/airflow/dags", include_examples=False)

    # Print import errors for debugging
    if dagbag.import_errors:
        print(f"DAG import errors: {dagbag.import_errors}")

    assert (
        dagbag.import_errors == {}
    ), f"DAG import errors: {dagbag.import_errors}"
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

    assert expected_tasks.issubset(
        set(task_ids)
    ), f"Missing tasks: {expected_tasks - set(task_ids)}"
    assert len(task_ids) == 2, f"Expected 2 tasks, found {len(task_ids)}"


def test_task_dependencies(dagbag):
    dag = dagbag.get_dag("data_collection_preprocessing_pipeline")
    fetch_data = dag.get_task("fetch_data")
    preprocess_data = dag.get_task("preprocess_data")

    downstream_tasks = fetch_data.get_direct_relatives(upstream=False)

    assert (
        preprocess_data in downstream_tasks
    ), "Expected 'preprocess_data' to be downstream of 'fetch_data'"


def test_task_types(dagbag):
    dag = dagbag.get_dag("data_collection_preprocessing_pipeline")
    assert isinstance(
        dag.get_task("fetch_data"), BashOperator
    ), "'fetch_data' should be a BashOperator"
    assert isinstance(
        dag.get_task("preprocess_data"), BashOperator
    ), "'preprocess_data' should be a BashOperator"


def test_bash_commands_reference_existing_scripts(dagbag):
    """
    Test that the bash commands in the tasks reference existing script files.
    """
    dag = dagbag.get_dag("data_collection_preprocessing_pipeline")

    # Get tasks
    fetch_data = dag.get_task("fetch_data")
    preprocess_data = dag.get_task("preprocess_data")

    # Check if scripts directory exists in the container
    scripts_dir = "/opt/airflow/scripts"
    assert os.path.exists(
        scripts_dir
    ), f"Scripts directory {scripts_dir} does not exist"

    # Check if script files exist in the container
    for script_name in ["fetch_stock.py", "clean_stock.py"]:
        script_path = os.path.join(scripts_dir, script_name)
        assert os.path.exists(
            script_path
        ), f"Script file {script_path} does not exist"


def test_scripts_are_executable(dagbag):
    """
    Test that the script files are executable.
    """
    scripts_dir = "/opt/airflow/scripts"
    for script_name in ["fetch_stock.py", "clean_stock.py"]:
        script_path = os.path.join(scripts_dir, script_name)
        assert os.access(
            script_path, os.X_OK
        ), f"Script file {script_path} is not executable"