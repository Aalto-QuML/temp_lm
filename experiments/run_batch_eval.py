#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation for all model subfolders in a directory."
    )
    parser.add_argument(
        "models_dir", type=Path, help="Directory containing model subfolders"
    )
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        print(f"Error: {args.models_dir} is not a valid directory.")
        sys.exit(1)

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Path to the evaluation script
    # Based on eval_battery.slurm, the script is src/plotting/evaluation_battery.py
    eval_script = Path("src/plotting/evaluation_battery.py")

    for model_dir in args.models_dir.iterdir():
        if model_dir.is_dir():
            print(f"Running evaluation for: {model_dir.name}")

            # Construct command
            # We assume the evaluation script accepts a path to the model directory via --model_dir
            # If the script strictly requires --model_id, this argument might need to be adjusted.
            cmd = [
                sys.executable,
                str(eval_script),
                "--model_dir",
                str(model_dir.resolve()),
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to evaluate {model_dir.name}: {e}")


if __name__ == "__main__":
    main()
