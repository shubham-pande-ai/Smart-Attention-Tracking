import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "mrl": {
        "slug": "akashshingha850/mrl-eye-dataset",
        "target": Path("data/raw/mrl-eye-dataset"),
        "purpose": "open/closed eye training",
    },
    "gazetrack": {
        "slug": "tooyoungalex/gazetrack",
        "target": Path("data/raw/gazetrack"),
        "purpose": "real gaze-direction training",
    },
    "mpii-gaze": {
        "slug": "tntg/mpii-gaze",
        "target": Path("data/raw/mpii-gaze"),
        "purpose": "real laptop webcam gaze-direction training",
    },
}


def has_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_credentials = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    return kaggle_json.exists() or env_credentials


def download_dataset(name: str) -> None:
    dataset = DATASETS[name]
    target = dataset["target"]
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset["slug"],
        "-p",
        str(target),
        "--unzip",
    ]
    print(f"Downloading {name}: {dataset['purpose']}")
    print(f"Kaggle slug: {dataset['slug']}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download recommended Kaggle datasets for attention tracking.")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(DATASETS),
        default=["mrl", "gazetrack"],
        help="Datasets to download. Defaults to mrl and gazetrack.",
    )
    args = parser.parse_args()

    if not has_kaggle_credentials():
        raise SystemExit(
            "Kaggle credentials were not found. Add ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY, then run this command again."
        )

    for dataset_name in args.datasets:
        download_dataset(dataset_name)


if __name__ == "__main__":
    main()
