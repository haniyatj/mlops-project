"""
Test file for stock data scripts
"""

import os
import subprocess
import pytest


def test_fetch_stock_script_exists():
    """Test that fetch_stock.py exists"""
    script_path = "/opt/airflow/scripts/fetch_stock.py"
    assert os.path.exists(script_path), f"Script {script_path} does not exist"


def test_clean_stock_script_exists():
    """Test that clean_stock.py exists"""
    script_path = "/opt/airflow/scripts/clean_stock.py"
    assert os.path.exists(script_path), f"Script {script_path} does not exist"


@pytest.mark.integration
def test_fetch_stock_script_runs():
    """Test that fetch_stock.py runs without errors"""
    data_dir = "/opt/airflow/data"
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Run the script
    result = subprocess.run(
        ["/opt/airflow/scripts/fetch_stock.py"],
        cwd="/opt/airflow",
        capture_output=True,
        text=True,
    )

    # Check that the script ran successfully
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Check that a file was created
    files = os.listdir(data_dir)
    assert any(
        f.startswith("stock_data_") for f in files
    ), "No stock data file was created"


@pytest.mark.integration
def test_clean_stock_script_runs():
    """Test that clean_stock.py runs without errors"""
    # First run fetch_stock.py to create input data
    subprocess.run(["/opt/airflow/scripts/fetch_stock.py"], cwd="/opt/airflow")

    # Then run clean_stock.py
    result = subprocess.run(
        ["/opt/airflow/scripts/clean_stock.py"],
        cwd="/opt/airflow",
        capture_output=True,
        text=True,
    )

    # Check that the script ran successfully
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Check that a processed file was created
    processed_dir = "/opt/airflow/processed"
    files = os.listdir(processed_dir)
    assert any(
        f.startswith("processed_stock_") for f in files
    ), "No processed stock data file was created"
