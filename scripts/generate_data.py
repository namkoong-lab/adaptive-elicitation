import argparse
import os
import subprocess
import sys

DATASETS_DIR = "src/dataset"


parser = argparse.ArgumentParser(description="Run a dataset generator.")
parser.add_argument("--dataset", required=True, help="Options: 'twentyq', 'opinion_qa', 'eedi'")
args = parser.parse_args()

dataset = args.dataset
dataset_path = os.path.join(DATASETS_DIR, f"{dataset}.py")

if os.path.isfile(dataset_path):
    subprocess.run(["python", dataset_path])
else:
    print(f"Dataset script not found: {dataset_path}")
    sys.exit(1)