"""Download the original UCI Banknote Authentication data file."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/"
    "data_banknote_authentication.txt"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "data_banknote_authentication.txt"


def validate_raw_dataset(path: Path) -> None:
    """Check that a raw Banknote file is non-empty and has five CSV columns."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Dataset file is missing or empty: {path}")

    with path.open("r", encoding="utf-8") as dataset_file:
        rows = [line.strip() for line in dataset_file if line.strip()]

    if not rows:
        raise ValueError(f"Dataset file contains no data rows: {path}")
    if any(len(row.split(",")) != 5 for row in rows):
        raise ValueError("Dataset must contain exactly five comma-separated columns per row.")


def download_dataset(destination: Path = DEFAULT_RAW_PATH, *, url: str = DATA_URL) -> Path:
    """Download and validate the raw UCI file, unless a valid copy already exists.

    Raises:
        RuntimeError: If the remote file cannot be downloaded. The exception includes
            manual-download instructions and never substitutes fabricated data.
    """
    destination = Path(destination)
    if destination.exists():
        validate_raw_dataset(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=30) as response:  # nosec B310: fixed HTTPS UCI URL
            content = response.read()
    except (URLError, OSError) as error:
        raise RuntimeError(
            "Could not download the UCI Banknote Authentication dataset. "
            f"Download {url} manually and save it as {destination}."
        ) from error

    if not content.strip():
        raise RuntimeError("The downloaded dataset was empty; no file was saved.")

    destination.write_bytes(content)
    try:
        validate_raw_dataset(destination)
    except ValueError:
        destination.unlink(missing_ok=True)
        raise
    return destination


def main() -> None:
    """Download the dataset from the command line."""
    parser = argparse.ArgumentParser(description="Download the UCI Banknote dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW_PATH)
    args = parser.parse_args()
    print(f"Dataset ready: {download_dataset(args.output)}")


if __name__ == "__main__":
    main()
