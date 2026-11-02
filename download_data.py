"""Download and optionally extract the official SpaceNet 2 building dataset.

The official SpaceNet page documents the AOI 3 Paris training archive as 5.3 GB.
This script uses the AWS CLI so downloads can be resumed/retried by AWS CLI.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

from src.paths import RAW_DIR

DATASETS = {
    "Paris": {
        "key": "SN2_buildings_train_AOI_3_Paris.tar.gz",
        "url": "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz",
        "approx_gb": 5.3,
    },
    "Khartoum": {
        "key": "SN2_buildings_train_AOI_5_Khartoum.tar.gz",
        "url": "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_5_Khartoum.tar.gz",
        "approx_gb": 4.7,
    },
    "Vegas": {
        "key": "SN2_buildings_train_AOI_2_Vegas.tar.gz",
        "url": "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz",
        "approx_gb": 23.0,
    },
    "Shanghai": {
        "key": "SN2_buildings_train_AOI_4_Shanghai.tar.gz",
        "url": "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_4_Shanghai.tar.gz",
        "approx_gb": 23.4,
    },
}


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a tar archive while rejecting path traversal entries."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive path detected: {member.name}")
        tar.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", choices=DATASETS, default="Paris")
    parser.add_argument("--extract", action="store_true", help="Extract after download")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the compressed archive after extraction")
    args = parser.parse_args()

    dataset = DATASETS[args.aoi]
    archive = RAW_DIR / dataset["key"]
    print(f"Selected AOI: {args.aoi}")
    print(f"Official archive size: approximately {dataset['approx_gb']} GB (compressed)")
    print(f"Destination: {archive}")

    if not archive.exists():
        if not shutil.which("aws"):
            raise SystemExit(
                "AWS CLI was not found. Install it from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html, "
                "then configure your AWS account/credentials as required by the SpaceNet download instructions. "
                "Alternatively, download the archive manually and place it in data/raw/."
            )
        command = ["aws", "s3", "cp", dataset["url"], str(archive)]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
    else:
        print("Archive already exists; skipping download.")

    if args.extract:
        print(f"Extracting to {RAW_DIR}. This may require several additional GB of free disk space.")
        safe_extract_tar(archive, RAW_DIR)
        print("Extraction complete.")
        if not args.keep_archive:
            archive.unlink(missing_ok=True)
            print("Removed compressed archive to save disk space.")

    print("Done. Next: python src/prepare_dataset.py --max-scenes 150")


if __name__ == "__main__":
    main()
