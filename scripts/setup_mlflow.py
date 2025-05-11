import os
import subprocess
import argparse


def setup_mlflow(
        backend_store_uri="sqlite:///mlflow.db",
        artifact_store="./mlruns"):
    """
    Set up and start an MLflow tracking server.

    Parameters:
        backend_store_uri: URI for the database where MLflow will store run parameters, metrics, etc.
        artifact_store: Directory where MLflow will store artifacts (models, plots, etc.)
    """
    # Ensure artifact store directory exists
    if not os.path.exists(artifact_store):
        os.makedirs(artifact_store)

    # Command to start MLflow server
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", artifact_store,
        "--host", "0.0.0.0",
        "--port", "5000"
    ]

    print(f"Starting MLflow server with command: {' '.join(cmd)}")
    print("MLflow UI will be available at http://localhost:5000")
    print(f"Backend store: {backend_store_uri}")
    print(f"Artifact store: {artifact_store}")
    print("\nPress Ctrl+C to stop the server.")

    # Start MLflow server
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up and start an MLflow tracking server")
    parser.add_argument(
        "--backend",
        default="sqlite:///mlflow.db",
        help="Backend store URI (default: sqlite:///mlflow.db)"
    )
    parser.add_argument(
        "--artifacts",
        default="./mlruns",
        help="Artifact store path (default: ./mlruns)"
    )

    args = parser.parse_args()
    setup_mlflow(args.backend, args.artifacts)
